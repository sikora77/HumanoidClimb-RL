import os
import argparse
import gymnasium as gym
import pybullet as p
import humanoid_climb.stances as stances
from stable_baselines3 import PPO
from humanoid_climb.climbing_config import ClimbingConfig

# Initialize stances and tracking pathways
stances.set_root_path("./")
STANCES = [
    stances.STANCE_1,
    stances.STANCE_2,
    stances.STANCE_3,
    stances.STANCE_4,
]

config = ClimbingConfig("./config.json")

# Create the core env natively (maintains the full 1026 observation space)
env = gym.make(
    "HumanoidClimb-v0", render_mode="human", max_ep_steps=50000, config=config
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = [
    "/models/1_10_9_n_n.zip",
    "/models/2_10_9_2_n.zip",
    "/models/3_10_9_2_1.zip",
    "/models/4_18_18_n_n.zip",
]

# Load models safely into the 1026-dimension environment
MODELS = [
    PPO.load(ROOT + MODEL_PATH[i], env=env) for i in range(len(MODEL_PATH))
]
CUR_MODEL = 0
REWARDS = [0 for _ in range(len(MODELS))]
STEPS = [0 for _ in range(len(MODELS))]

# Extract targets and overrides for standard execution tracking
MOTION = [s.stance for s in STANCES]
EXCLUDE = [s.exclude_targets for s in STANCES]
O_ACTION = [s.action_override for s in STANCES]

# Single clean initialization tracking both components
obs, info = env.reset()

done = False
truncated = False
score = 0
step = 0
pause = False
STANCE_TOLERANCE = 700

climb_attempts = 0
successful_attempts = 0

print(
    "--- Visualization Loop Active (Press Space to Pause, Backspace to Reset) ---"
)

while True:
    if not pause:
        # Predict utilizing the current active sub-policy stage
        action, _state = MODELS[CUR_MODEL].predict(obs, deterministic=True)

        for i in range(4):
            if O_ACTION[CUR_MODEL][i] != -1:
                action[17 + i] = O_ACTION[CUR_MODEL][i]

        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

        REWARDS[CUR_MODEL] += reward
        STEPS[CUR_MODEL] += 1

    if STEPS[CUR_MODEL] > STANCE_TOLERANCE:
        truncated = True

    keys = p.getKeyboardEvents()

    # Reset on backspace
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Manual Reset -> Score: {score}, Steps: {step}")
        CUR_MODEL = 0
        REWARDS = [0 for _ in range(len(MODELS))]
        STEPS = [0 for _ in range(len(MODELS))]
        done = False
        truncated = False
        pause = False
        score = 0
        step = 0
        obs, info = env.reset()

    # Pause on space
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        pause = not pause
        print("Paused" if pause else "Unpaused")

    # If the sub-policy checks out successfully, hand over to the next stage
    if info.get("is_success", False):
        print(
            f"Finished stance {CUR_MODEL + 1} with {REWARDS[CUR_MODEL]} ({REWARDS[CUR_MODEL] - 1000}) reward in {STEPS[CUR_MODEL]} steps"
        )
        CUR_MODEL += 1
        if CUR_MODEL > len(MODELS) - 1:
            CUR_MODEL = 0

    # Handle standard environmental fall or time out
    if done or truncated:
        climb_attempts += 1
        if info.get("is_success", False):
            successful_attempts += 1

        print(
            f"ENV TERMINATED | SUCCESS RATE: {(successful_attempts / climb_attempts) * 100:.2f} %\n"
        )

        # Reset staging track sequences
        CUR_MODEL = 0
        REWARDS = [0 for _ in range(len(MODELS))]
        STEPS = [0 for _ in range(len(MODELS))]
        score = 0
        step = 0
        obs, info = env.reset()

env.close()
