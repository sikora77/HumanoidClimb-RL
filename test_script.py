
import gymnasium as gym

env = gym.make("HumanoidClimb-v0", render_mode="rgb_array")
obs, info = env.reset()
print("Initial stance:", env.unwrapped.current_stance)
print("Robot Z pos:", env.unwrapped.climber.robot_body.current_position()[2])
print("Left hand pos:", env.unwrapped.climber.effectors[0].current_position())
print("Target hold pos (left hand):", env.unwrapped.targets[env.unwrapped.desired_stance[0]].body.initialPosition if env.unwrapped.desired_stance[0] != -1 else "N/A")
