import os

import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO

from humanoid_climb.climbing_config import ClimbingConfig

config = ClimbingConfig("./config.json")

# Create the core env natively
env = gym.make(
    "HumanoidClimb-v0", render_mode="human", max_ep_steps=50000, config=config
)

ROOT = os.path.dirname(os.path.abspath(__file__))
# Example generalized model path (replace with actual once trained)
MODEL_PATH = "/models/generalized_model.zip"

model = None
full_model_path = ROOT + MODEL_PATH
if os.path.exists(full_model_path):
    print(f"Loading model from {full_model_path}")
    model = PPO.load(full_model_path)
else:
    print(f"Model not found at {full_model_path}. Running with random actions for verification.")

# Single clean initialization tracking both components
obs, info = env.reset()

done = False
truncated = False
score = 0
step = 0
pause = False

climb_attempts = 0
successful_attempts = 0

print(
    "--- Visualization Loop Active (Press Space to Pause, Backspace to Reset) ---"
)

while True:
    if not pause:
        if model:
            # Predict utilizing the generalized model
            action, _state = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        score += reward
        step += 1

    keys = p.getKeyboardEvents()

    # Reset on backspace
    if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
        print(f"Manual Reset -> Score: {score}, Steps: {step}")
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
        print(f"Route Topped Out! Reward: {score} in {step} steps")
        done = True

    # Handle standard environmental fall or time out
    if done or truncated:
        climb_attempts += 1
        if info.get("is_success", False):
            successful_attempts += 1

        print(
            f"ENV TERMINATED | SUCCESS RATE: {(successful_attempts / climb_attempts) * 100:.2f} %\n"
        )

        score = 0
        step = 0
        obs, info = env.reset()

env.close()
