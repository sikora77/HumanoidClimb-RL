import os
import sys
import numpy as np
import gymnasium as gym

from humanoid_climb.climbing_config import ClimbingConfig


def test_environment():
    print("=== Running Environment Tests ===")
    
    # 1. Initialize environment
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = ClimbingConfig(config_path)
    
    env = gym.make(
        "HumanoidClimb-v0",
        config=config,
        render_mode="rgb_array",
        max_ep_steps=600
    )
    
    # 2. Test reset and observation dimensions
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Returned observation shape: {obs.shape}")
    
    assert obs.shape == env.observation_space.shape, (
        f"Mismatch between obs shape {obs.shape} and observation_space {env.observation_space.shape}"
    )
    print("[PASSED] Observation space dimension check.")
    
    # 3. Test hanging state & start holds attachment on spawn
    unwrapped_env = env.unwrapped
    attached = unwrapped_env.climber.effector_attached_to
    print(f"End effector attachments on spawn: {attached}")
    
    # Left hand (0) and Right hand (1) must be attached to start holds
    assert attached[0] != -1, "Left hand is NOT attached to a start hold!"
    assert attached[1] != -1, "Right hand is NOT attached to a start hold!"
    print(f"[PASSED] Left hand attached to: {attached[0]}")
    print(f"[PASSED] Right hand attached to: {attached[1]}")
    
    # Feet (2 and 3) must NOT be attached on spawn
    assert attached[2] == -1, "Left foot should not be attached on spawn!"
    assert attached[3] == -1, "Right foot should not be attached on spawn!"
    print("[PASSED] Feet are unattached on spawn so agent hangs naturally.")
    
    # Check that agent is not lying on the floor
    on_floor = unwrapped_env.is_on_floor()
    com_height = unwrapped_env.get_com_height()
    print(f"Is on floor: {on_floor}")
    print(f"Center of mass height: {com_height:.2f}m")
    
    assert not on_floor, "Climber is lying on the floor instead of hanging!"
    assert com_height > 0.2, f"Center of mass height {com_height:.2f}m is too low!"
    print("[PASSED] Climber hanging posture check (not on floor, COM elevated).")
    
    # Check that hands are at the expected hold X-coordinate (0.37)
    left_hand_pos = unwrapped_env.climber.parts["left_hand"].current_position()
    right_hand_pos = unwrapped_env.climber.parts["right_hand"].current_position()
    assert abs(left_hand_pos[0] - 0.37) < 0.1, f"Left hand X position is {left_hand_pos[0]}, expected ~0.37"
    assert abs(right_hand_pos[0] - 0.37) < 0.1, f"Right hand X position is {right_hand_pos[0]}, expected ~0.37"
    print("[PASSED] Hands are correctly positioned at the hold coordinates.")
    
    # Check that torso is NOT inside the wall (wall front is at X=0.43, so X must be < 0.43)
    torso_pos = unwrapped_env.climber.robot_body.current_position()
    assert torso_pos[0] < 0.43, f"Torso is inside the wall! Torso X: {torso_pos[0]} >= Wall Front X: 0.43"
    print(f"[PASSED] Torso is not inside the wall (Torso X: {torso_pos[0]:.2f}).")
    
    # 4. Check route curriculum information
    route_info = getattr(unwrapped_env, "current_route_info", {})
    print(f"[Curriculum Info] Route Name: '{route_info.get('name')}'")
    print(f"[Curriculum Info] Difficulty: {route_info.get('difficulty')}")
    print(f"[Curriculum Info] Frames: {route_info.get('frames')[:60]}...")
    assert 'name' in route_info and 'frames' in route_info, "Route curriculum info missing!"
    print("[PASSED] Curriculum route parsing check.")
    
    # 5. Test Start Hold Lock Window (Agent must hold start holds for initial 30 steps)
    print("\nVerifying Start Hold Lock Window over initial 30 steps...")
    for step in range(1, 31):
        # Apply random action (which might include release signals)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        curr_attached = unwrapped_env.climber.effector_attached_to
        lh_attached = curr_attached[0] != -1
        rh_attached = curr_attached[1] != -1
        
        assert lh_attached and rh_attached, (
            f"Step {step}: Agent released start holds prematurely! Attachments: {curr_attached}"
        )
        assert not unwrapped_env.is_on_floor(), f"Step {step}: Agent fell to floor!"
        
        # Verify hold coordinates and wall clearance dynamically during active stepping
        left_hand_pos = unwrapped_env.climber.parts["left_hand"].current_position()
        right_hand_pos = unwrapped_env.climber.parts["right_hand"].current_position()
        assert abs(left_hand_pos[0] - 0.37) < 0.1, f"Step {step}: Left hand drifted to {left_hand_pos[0]}"
        assert abs(right_hand_pos[0] - 0.37) < 0.1, f"Step {step}: Right hand drifted to {right_hand_pos[0]}"
        
        torso_pos = unwrapped_env.climber.robot_body.current_position()
        assert torso_pos[0] < 0.43, f"Step {step}: Torso clipped into wall! Torso X: {torso_pos[0]}"

    print("[PASSED] Start Hold Lock Window check: Agent reliably held start holds for 30 steps!")

    # 6. Test stepping simulation for remaining steps up to 100
    print("\nContinuing simulation steps up to step 100...")
    total_reward = 0
    for step in range(31, 101):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
            
    print(f"[PASSED] Stepping check completed up to step 100. Total reward: {total_reward:.2f}")
    env.close()
    print("\n=== ALL ENVIRONMENT TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_environment()
