import argparse
import os

import gymnasium as gym
import pybullet as p
import stable_baselines3 as sb
import torch
import wandb
from gymnasium.wrappers import RecordVideo
from humanoid_climb import stances
from humanoid_climb.climbing_config import ClimbingConfig
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
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
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.rollout_count = 0
        self.save_path = save_path
        self._saved_initial = False

    def _on_step(self) -> bool:
        if not self._saved_initial:
            print(
                f"--- [CustomCallback] Saving initial model at step {self.num_timesteps} to {self.save_path}/rl_model_initial.zip ---"
            )
            self.model.save(f"{self.save_path}/rl_model_initial.zip")
            self._saved_initial = True
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        self.logger.record("climb/rollout_count", self.rollout_count)


class VideoRecorderCallback(BaseCallback):
    """
    Callback that uses Gymnasium's RecordVideo wrapper combined with
    a front-facing PyBullet camera render patch to capture evaluation runs.
    """

    def __init__(
        self,
        eval_freq: int,
        video_folder: str,
        max_ep_steps: int = 600,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.video_folder = video_folder
        self.max_ep_steps = max_ep_steps
        self.last_eval_step = -1
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        # Trigger on step 0 (first iteration) and every eval_freq steps
        if self.num_timesteps == 0 or (
            self.num_timesteps - self.last_eval_step >= self.eval_freq
        ):
            self.last_eval_step = self.num_timesteps
            self._record_video()
        return True

    def _record_video(self):
        print(
            f"\n--- [VideoRecorder] Recording evaluation video at step {self.num_timesteps} ---"
        )
        config = ClimbingConfig("./config.json")

        base_env = gym.make(
            "HumanoidClimb-v0",
            render_mode="rgb_array",
            max_ep_steps=self.max_ep_steps,
            config=config,
        )

        eval_env = RecordVideo(
            base_env,
            video_folder=self.video_folder,
            episode_trigger=lambda e: True,
            name_prefix=f"eval-step-{self.num_timesteps}",
        )

        obs, _info = eval_env.reset()

        route_info = getattr(base_env.unwrapped, "current_route_info", {})
        r_name = route_info.get("name", "Unknown Route")
        r_diff = route_info.get("difficulty", "N/A")
        r_frames = route_info.get("frames", "N/A")
        print(
            f"--- [VideoRecorder] Route: '{r_name}' | Difficulty: {r_diff} | Frames: {r_frames} ---"
        )

        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = eval_env.step(action)
            total_reward += reward
            steps += 1

        eval_env.close()
        print(
            f"--- [VideoRecorder] Video saved successfully! Reward: {total_reward:.2f}, Steps: {steps} ---\n"
        )


def make_env(
    env_id: str,
    rank: int,
    seed: int = 0,
    max_steps: int = 1000,
    stance: stances.Stance = stances.STANCE_NONE,
    discrete_grasp: bool = True,
    grasp_reward: bool = True,
    grasp_persist_steps: int = 10,
    render_mode: str | None = None,
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


def train(
    env_name,
    sb3_algo,
    workers,
    path_to_model=None,
    use_cpu=False,
    net_arch=None,
    activation="tanh",
):
    if net_arch is None:
        net_arch = [64, 64]
    device_obj = torch.device("cpu") if use_cpu else DEVICE
    print(f"Executing training on device: {device_obj}")

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 50000000,
        "env_name": env_name,
        "net_arch": net_arch,
        "activation": activation,
    }

    run = wandb.init(
        project="HumanoidClimb-v3",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        mode="offline",
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

    # 2. Initialize callbacks including our clean RecordVideo callback
    cust_callback = CustomCallback(save_path=save_path)
    video_callback = VideoRecorderCallback(
        eval_freq=50000, video_folder=video_dir, max_ep_steps=max_ep_steps
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // workers),
        save_path=save_path,
        name_prefix="rl_model",
    )

    activation_map = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
    }
    act_fn = activation_map.get(activation.lower(), torch.nn.Tanh)

    policy_kwargs = {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": act_fn,
    }

    if sb3_algo == "PPO":
        if path_to_model is None:
            model = sb.PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=1e-3,
                verbose=1,
                device=device_obj,
                tensorboard_log=log_dir,
                batch_size=64,
                policy_kwargs=policy_kwargs,
            )
        else:
            model = sb.PPO.load(path_to_model, env=vec_env, device=device_obj)
    elif sb3_algo == "SAC":
        if path_to_model is None:
            model = sb.SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,  # <-- Change the learning rate here!
                verbose=1,
                device=device_obj,
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs,
            )
        else:
            model = sb.SAC.load(path_to_model, env=vec_env, device=device_obj)
    else:
        print("Algorithm not found")
        return

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=[
            WandbCallback(
                gradient_save_freq=50000,
                verbose=2,
            ),
            checkpoint_callback,
            video_callback,
            cust_callback,
        ],
    )
    run.finish()


def test(env, sb3_algo, path_to_model, use_cpu=False):
    device_obj = torch.device("cpu") if use_cpu else DEVICE
    if sb3_algo == "SAC":
        model = sb.SAC.load(path_to_model, env=env, device=device_obj)
    elif sb3_algo == "TD3":
        model = sb.TD3.load(path_to_model, env=env, device=device_obj)
    elif sb3_algo == "A2C":
        model = sb.A2C.load(path_to_model, env=env, device=device_obj)
    elif sb3_algo == "DQN":
        model = sb.DQN.load(path_to_model, env=env, device=device_obj)
    elif sb3_algo == "PPO":
        model = sb.PPO.load(path_to_model, env=env, device=device_obj)
    else:
        print("Algorithm not found")
        return

    vec_env = model.get_env()
    obs = vec_env.reset()
    score = 0
    step = 0

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _info = vec_env.step(action)
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
        "gymenv", help="Gymnasium environment i.e. HumanoidClimb-v0"
    )
    parser.add_argument(
        "sb3_algo", help="StableBaseline3 RL algorithm i.e. PPO, SAC"
    )
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-f", "--file", required=False, default=None)
    parser.add_argument("-s", "--test", metavar="path_to_model")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device execution for PPO/SAC",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Custom layer dimensions for policy & value networks, e.g. --net-arch 256 256",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "elu"],
        help="Activation function for neural network (tanh, relu, elu)",
    )
    args = parser.parse_args()

    if args.train:
        if args.file is None:
            print("<< Training from scratch! >>")
            train(
                args.gymenv,
                args.sb3_algo,
                args.workers,
                use_cpu=args.cpu,
                net_arch=args.net_arch,
                activation=args.activation,
            )
        elif os.path.isfile(args.file):
            print(f"<< Continuing {args.file} >>")
            train(
                args.gymenv,
                args.sb3_algo,
                args.workers,
                path_to_model=args.file,
                use_cpu=args.cpu,
                net_arch=args.net_arch,
                activation=args.activation,
            )

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
            test(
                env, args.sb3_algo, path_to_model=args.test, use_cpu=args.cpu
            )
        else:
            print(f"{args.test} not found.")
