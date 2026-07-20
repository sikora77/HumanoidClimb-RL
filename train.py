import argparse
import json
import os
import time
from typing import Optional

import gymnasium as gym
import humanoid_climb.stances as stances
import numpy as np
import pybullet as p
import stable_baselines3 as sb
import torch
from gymnasium.wrappers import FlattenObservation
from humanoid_climb.climbing_config import ClimbingConfig
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecVideoRecorder,
)

import wandb
from wandb.integration.sb3 import WandbCallback

# Set up CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create directories to hold models, logs, and local videos
model_dir = "models"
log_dir = "logs"
video_dir = "videos"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)


class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        self.logger.record("climb/rollout_count", self.rollout_count)


def make_env(
    env_id: str,
    rank: int,
    seed: int = 0,
    max_steps: int = 1000,
    stance: stances.Stance = stances.STANCE_NONE,
    discrete_grasp: bool = True,
    grasp_reward: bool = True,
    grasp_persist_steps: int = 10,
    render_mode: Optional[str] = None,
) -> gym.Env:
    def _init():
        config = ClimbingConfig("./config.json")
        env = gym.make(
            env_id,
            config=config,
            render_mode=render_mode,
            max_ep_steps=max_steps,
            discrete_grasp=discrete_grasp,
            grasp_reward=grasp_reward,
            grasp_persist_steps=grasp_persist_steps,
        )
        m_env = Monitor(env)
        m_env.reset(seed=seed + rank)
        return m_env

    set_random_seed(seed)
    return _init


def train(env_name, sb3_algo, workers, path_to_model=None):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_name": env_name,
    }
    run = wandb.init(
        project="HumanoidClimb-v3",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    max_ep_steps = 600
    stances.set_root_path("./humanoid_climb")
    stance = stances.STANCE_1

    # 1. Primary training workers (Fast & Headless)
    vec_env = SubprocVecEnv(
        [
            make_env(
                env_name,
                i,
                max_steps=max_ep_steps,
                stance=stance,
                render_mode=None,
            )
            for i in range(workers)
        ],
        start_method="spawn",
    )

    save_path = f"{model_dir}/{run.id}"

    # 2. Isolated visual evaluation worker
    eval_env_raw = SubprocVecEnv(
        [
            make_env(
                env_name,
                workers + 1,
                max_steps=max_ep_steps,
                stance=stance,
                render_mode="rgb_array",
            )
        ],
        start_method="spawn",
    )

    # 3. Video recorder wrapper tracking the evaluation environment (Saves directly to local ./videos)
    eval_env = VecVideoRecorder(
        eval_env_raw,
        video_folder=video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=max_ep_steps,
        name_prefix=f"eval-run-{run.id}",
    )

    # Calculate callback execution frequency to run every 50,000 global environment steps
    video_freq = max(1, 50000 // workers)

    # 4. Bind the recorder environment to the evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/models/",
        log_path=f"{save_path}/logs/",
        eval_freq=video_freq,
        deterministic=True,
        render=False,
    )
    cust_callback = CustomCallback()

    if sb3_algo == "PPO":
        if path_to_model is None:
            model = sb.PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                device=DEVICE,
                tensorboard_log=log_dir,
                batch_size=64,
            )
        else:
            model = sb.PPO.load(path_to_model, env=vec_env)
    elif sb3_algo == "SAC":
        if path_to_model is None:
            model = sb.SAC(
                "MlpPolicy",
                vec_env,
                verbose=1,
                device=DEVICE,
                tensorboard_log=log_dir,
            )
        else:
            model = sb.SAC.load(path_to_model, env=vec_env)
    else:
        print("Algorithm not found")
        return

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=[
            WandbCallback(
                gradient_save_freq=5000,
                model_save_freq=5000,
                model_save_path=save_path,
                verbose=2,
            ),
            eval_callback,
            cust_callback,
        ],
    )
    run.finish()


def test(env, sb3_algo, path_to_model):
    if sb3_algo == "SAC":
        model = sb.SAC.load(path_to_model, env=env)
    elif sb3_algo == "TD3":
        model = sb.TD3.load(path_to_model, env=env)
    elif sb3_algo == "A2C":
        model = sb.A2C.load(path_to_model, env=env)
    elif sb3_algo == "DQN":
        model = sb.DQN.load(path_to_model, env=env)
    elif sb3_algo == "PPO":
        model = sb.PPO.load(path_to_model, env=env)
    else:
        print("Algorithm not found")
        return

    vec_env = model.get_env()
    obs = vec_env.reset()
    score = 0
    step = 0

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        score += reward
        step += 1

        if done:
            print(f"Episode Over, Score: {score}, Steps {step}")
            score = 0
            step = 0

        # Reset on backspace
        keys = p.getKeyboardEvents()
        if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
            score = 0
            step = 0
            env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument(
        "gymenv", help="Gymnasium environment i.e. Humanoid-v4"
    )
    parser.add_argument(
        "sb3_algo", help="StableBaseline3 RL algorithm i.e. SAC, TD3"
    )
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-f", "--file", required=False, default=None)
    parser.add_argument("-s", "--test", metavar="path_to_model")
    args = parser.parse_args()

    if args.train:
        if args.file is None:
            print(f"<< Training from scratch! >>")
            train(args.gymenv, args.sb3_algo, args.workers)
        elif os.path.isfile(args.file):
            print(f"<< Continuing {args.file} >>")
            train(args.gymenv, args.sb3_algo, args.workers, args.file)

    if args.test:
        if os.path.isfile(args.test):
            stances.set_root_path("./humanoid_climb")
            stance = stances.STANCE_14_1
            max_steps = 600

            env = gym.make(
                args.gymenv,
                render_mode="human",
                max_ep_steps=max_steps,
                **stance.get_args(),
            )
            test(env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f"{args.test} not found.")
