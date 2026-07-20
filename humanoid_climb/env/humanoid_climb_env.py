import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import json
import os

from typing import Optional
from pybullet_utils.bullet_client import BulletClient
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.assets.asset import Asset

FINISH_ROLE = 14
FOOT_ROLE = 15
START_ROLE = 12
MIDDLE_ROLE = 13


class HumanoidClimbEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config,
        render_mode: Optional[str] = None,
        max_ep_steps: Optional[int] = 602,
        state_file: Optional[str] = None,
        discrete_grasp: bool = False,
        n_torque_bins: int = 21,
        grasp_reward: bool = False,
        grasp_persist_steps: int = 0,
        kilter_config_path: str = "kilter_config.json",
    ):

        self.config = config
        self.render_mode = render_mode
        self.max_ep_steps = max_ep_steps
        self.steps = 0

        # --- DYNAMIC KILTER BOARD INTEGRATION ---
        self.kilter_config_path = kilter_config_path
        if os.path.exists(self.kilter_config_path):
            with open(self.kilter_config_path, "r") as f:
                self.kilter_cfg = json.load(f)
        else:
            # Safe fallback defaults if config file isn't present yet
            self.kilter_cfg = {
                "kilter_board": {
                    "rows": 12,
                    "cols": 12,
                    "states": {
                        "0": "unlit",
                        "1": "start",
                        "2": "hand_foot",
                        "3": "foot_only",
                        "4": "finish",
                    },
                },
                "training_settings": {
                    "include_wall_state": True,
                    "one_hot_encoded": true,
                },
            }

        board_cfg = self.kilter_cfg["kilter_board"]
        train_cfg = self.kilter_cfg["training_settings"]

        self.kilter_rows = board_cfg["rows"]
        self.kilter_cols = board_cfg["cols"]
        self.num_states = len(board_cfg["states"])
        self.include_wall = train_cfg["include_wall_state"]
        self.one_hot = train_cfg["one_hot_encoded"]

        if self.include_wall:
            if self.one_hot:
                self.wall_state_dim = (
                    self.kilter_rows * self.kilter_cols * self.num_states
                )
            else:
                self.wall_state_dim = self.kilter_rows * self.kilter_cols
        else:
            self.wall_state_dim = 0
        # ----------------------------------------

        self.motion_path = [
            self.config.stance_path[stance]["desired_holds"]
            for stance in self.config.stance_path
        ]
        self.motion_exclude_targets = [
            self.config.stance_path[stance]["ignore_holds"]
            for stance in self.config.stance_path
        ]
        self.action_override = [
            self.config.stance_path[stance]["force_attach"]
            for stance in self.config.stance_path
        ]

        self.init_from_state = False if state_file is None else True
        self.state_file = state_file

        if self.render_mode == "human":
            self._p = BulletClient(p.GUI)
        else:
            self._p = BulletClient(p.DIRECT)

        self.discrete_grasp = discrete_grasp
        self.n_torque_bins = n_torque_bins

        self.grasp_reward = grasp_reward
        self.grasp_attach_bonus = 5.0
        self.grasp_wrong_attach_penalty = -1.0
        self.grasp_waste_penalty = -0.05
        self.grasp_premature_release_penalty = -20.0
        self._prev_attached = [-1, -1, -1, -1]

        self.grasp_persist_steps = grasp_persist_steps
        self._grasp_lock_remaining = [0, 0, 0, 0]
        self._last_grasp_binary = [0, 0, 0, 0]

        if self.discrete_grasp:
            self.action_space = gym.spaces.MultiDiscrete(
                [n_torque_bins] * 17 + [2] * 4
            )
        else:
            self.action_space = gym.spaces.Box(-1, 1, (21,), np.float32)

        # Dynamically calculated space dimension based on configuration parameters
        total_obs_dim = 306 + self.wall_state_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        self.current_stance = []
        self.desired_stance = []
        self.desired_stance_index = 0
        self.best_dist_to_stance = []

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=-90,
            cameraPitch=0,
            cameraTargetPosition=[0, 0, 3],
        )
        self._p.setGravity(0, 0, -9.8)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.config.timestep_interval,
            numSubSteps=self.config.timestep_per_action,
        )

        self.floor = Asset(self._p, self.config.plane)
        self.wall = Asset(self._p, self.config.surface)
        self.climber = Humanoid(self._p, self.config.climber)
        self.prevheight = self.get_com_height()

        self.debug_stance_text = self._p.addUserDebugText(
            text=f"",
            textPosition=[0, 0, 0],
            textSize=1,
            lifeTime=0.1,
            textColorRGB=[1.0, 0.0, 1.0],
        )

        self.targets = dict()
        for key in self.config.holds:
            self.targets[key] = Asset(self._p, self.config.holds[key])
            self._p.addUserDebugText(
                text=key,
                textPosition=self.targets[key].body.initialPosition,
                textSize=0.7,
                lifeTime=0.0,
                textColorRGB=[0.0, 0.0, 1.0],
            )

        self.climber.targets = self.targets

    def get_com_height(self):
        parts = self.climber.parts
        total_mass = sum(
            self._p.getDynamicsInfo(self.climber.robot, part.bodyPartIndex)[0]
            for part in parts.values()
        )
        weighted_height = sum(
            self._p.getDynamicsInfo(self.climber.robot, part.bodyPartIndex)[0]
            * part.get_position()[2]
            for part in parts.values()
        )
        return weighted_height / total_mass

    def _decode_action(self, action):
        if self.discrete_grasp:
            action = np.asarray(action)
            torque_indices = action[:17].astype(np.float32)
            torques = (torque_indices / (self.n_torque_bins - 1)) * 2.0 - 1.0
            grasps = action[17:21].astype(np.float32) * 2.0 - 1.0
            decoded = np.concatenate([torques, grasps]).astype(np.float32)
        else:
            decoded = np.asarray(action, dtype=np.float32)

        if self.grasp_persist_steps > 0:
            for i in range(4):
                intent_binary = 1 if decoded[17 + i] > 0 else 0
                if self._grasp_lock_remaining[i] > 0:
                    intent_binary = self._last_grasp_binary[i]
                    self._grasp_lock_remaining[i] -= 1
                elif intent_binary != self._last_grasp_binary[i]:
                    self._last_grasp_binary[i] = intent_binary
                    self._grasp_lock_remaining[i] = (
                        self.grasp_persist_steps - 1
                    )
                decoded[17 + i] = 1.0 if intent_binary == 1 else -1.0

        return decoded

    def _grasp_event_reward(self, prev_attached, decoded_grasps):
        if not self.grasp_reward:
            return 0.0
        overrides = self.action_override[self.desired_stance_index]
        new_attached = self.climber.effector_attached_to
        v_z = self.climber.speed()[2]
        n_attached_after = sum(1 for a in new_attached if a != -1)
        bonus = 0.0
        for i in range(4):
            if overrides[i] is not None:
                continue
            was = prev_attached[i]
            now = new_attached[i]
            grasp_on = decoded_grasps[i] > 0
            new_attach = was == -1 and now != -1
            new_release = was != -1 and now == -1
            if new_attach:
                if (
                    self.desired_stance[i] != -1
                    and now == self.desired_stance[i]
                ):
                    bonus += self.grasp_attach_bonus
                else:
                    bonus += self.grasp_wrong_attach_penalty
            elif grasp_on and now == -1:
                bonus += self.grasp_waste_penalty
            if new_release and v_z < 0 and n_attached_after < 2:
                bonus += self.grasp_premature_release_penalty
        return bonus

    def step(self, action):
        self._p.stepSimulation()
        self.steps += 1

        prev_attached = list(self.climber.effector_attached_to)
        decoded_action = self._decode_action(action)
        self.climber.apply_action(
            decoded_action, self.action_override[self.desired_stance_index]
        )
        self.update_stance()

        ob = self._get_obs()
        info = self._get_info()

        reward = self.calculate_reward_negative_distance()
        reward += self._grasp_event_reward(
            prev_attached, decoded_action[17:21]
        )

        # 1. Safely fetch the mapping from the config namespace
        grid_mapping = getattr(self.config, "hold_grid_mapping", {})

        # 2. Extract keys of all holds designated as a finish hold
        finish_keys = [
            k for k, v in grid_mapping.items() if v["type"] == FINISH_ROLE
        ]

        # 3. Extract the integer IDs from those keys (e.g., 'hold_12' -> 12)
        finish_indices = [
            int(k.split("_")[1]) for k in finish_keys if "_" in k
        ]

        # 4. Extract what the left hand (index 0) and right hand (index 1) are holding
        left_hand_hold = self.current_stance[0]
        right_hand_hold = self.current_stance[1]

        # 5. Check if both hands are securely registered on any valid finish hold
        lh_on_finish = (left_hand_hold in finish_keys) or (
            left_hand_hold in finish_indices
        )
        rh_on_finish = (right_hand_hold in finish_keys) or (
            right_hand_hold in finish_indices
        )

        if lh_on_finish and rh_on_finish:
            # Assign the giant reward payload
            giant_finish_reward = 2500.0
            reward += giant_finish_reward

            # Force terminate the episode so the agent doesn't farm the reward
            terminated = True
            info["is_success"] = True
            print(
                f"--- ROUTE TOPPED OUT! Giant reward of +{giant_finish_reward} applied. ---"
            )

        reached = self.check_reached_stance()

        terminated = self.terminate_check()
        truncated = self.truncate_check()

        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.climber.reset()
        self.climber.exclude_targets = self.motion_exclude_targets[0]
        self.steps = 0
        self._grasp_lock_remaining = [0, 0, 0, 0]
        self._last_grasp_binary = [0, 0, 0, 0]
        self.current_stance = [-1, -1, -1, -1]
        self.desired_stance_index = 0
        self.motion_path = [
            self.config.stance_path[stance]["desired_holds"]
            for stance in self.config.stance_path
        ]
        self.desired_stance = self.motion_path[0]
        self.best_dist_to_stance = self.get_distance_from_desired_stance()
        self.prevheight = self.get_com_height()
        self.previous_height = self.prevheight

        ob = self._get_obs()
        info = self._get_info()

        for key in self.targets:
            colour = (
                [0.0, 0.7, 0.1, 0.75]
                if key in self.desired_stance
                else [1.0, 0, 0, 0.75]
            )
            self._p.changeVisualShape(
                objectUniqueId=self.targets[key].id,
                linkIndex=-1,
                rgbaColor=colour,
            )

        return np.array(ob, dtype=np.float32), info

    def calculate_reward_negative_distance(self):
        current_dist_away = self.get_distance_from_desired_stance()

        is_closer = (
            1
            if np.sum(current_dist_away) < np.sum(self.best_dist_to_stance)
            else 0
        )
        if is_closer:
            self.best_dist_to_stance = current_dist_away.copy()

        reward = np.clip(-1 * np.sum(current_dist_away), -2, float("inf"))
        if self.is_on_floor():
            reward += (self.max_ep_steps - self.steps) * -2

        return reward

    def calculate_improved_reward(self):
        current_dist_away = self.get_distance_from_desired_stance()
        reward = np.clip(-1 * np.sum(current_dist_away), -2, float("inf"))

        torso_velocity = self.climber.speed()[2]
        reward += max(0, torso_velocity) * 2

        torso_orientation = self.climber.get_orientation()
        slouch_angle = torso_orientation[1]
        target_slouch = -np.pi / 6

        reward += (
            max(0, abs(target_slouch) - abs(slouch_angle - target_slouch))
            * 0.5
        )

        if not self.is_on_floor():
            reward += 0.1

        if self.is_on_floor():
            reward -= 5

        return reward

    def calculate_reward_eq1(self):
        kappa = 0.6
        sigma = 0.5

        sum_values = [0, 0, 0, 0]
        current_dist_away = self.get_distance_from_desired_stance()
        for i, effector in enumerate(self.climber.effectors):
            distance = current_dist_away[i]
            reached = (
                1 if self.current_stance[i] == self.desired_stance[i] else 0
            )
            sum_values[i] = kappa * np.exp(-1 * sigma * distance) + reached

        is_closer = True
        difference_closer = 0

        if np.sum(current_dist_away) > np.sum(self.best_dist_to_stance):
            is_closer = False
            difference_closer = np.sum(self.best_dist_to_stance) - np.sum(
                current_dist_away
            )

        if is_closer:
            for i, best_dist_away in enumerate(self.best_dist_to_stance):
                if current_dist_away[i] < best_dist_away:
                    self.best_dist_to_stance[i] = current_dist_away[i]

        reward = is_closer * np.sum(sum_values) + 0.8 * difference_closer
        reward += 3000 if self.current_stance == self.desired_stance else 0
        if self.is_on_floor():
            reward = -3000

        self.visualise_reward(reward, -2, 2)
        return reward

    def check_reached_stance(self):
        reached = False
        if self.current_stance == self.desired_stance:
            reached = True

            self.desired_stance_index += 1
            if self.desired_stance_index > len(self.motion_path) - 1:
                return reached

            new_stance = self.motion_path[self.desired_stance_index]
            self.climber.exclude_targets = self.motion_exclude_targets[
                self.desired_stance_index
            ]

            for key in self.desired_stance:
                if key == -1:
                    continue
                self._p.changeVisualShape(
                    objectUniqueId=self.targets[key].id,
                    linkIndex=-1,
                    rgbaColor=[1.0, 0.0, 0.0, 0.75],
                )
            self.desired_stance = new_stance

            for key in self.desired_stance:
                if key == -1:
                    continue
                self._p.changeVisualShape(
                    objectUniqueId=self.targets[key].id,
                    linkIndex=-1,
                    rgbaColor=[0.0, 0.7, 0.1, 0.75],
                )

            self.best_dist_to_stance = self.get_distance_from_desired_stance()

        return reached

    def update_stance(self):
        self.current_stance = self.climber.effector_attached_to

        if self.render_mode == "human":
            torso_pos = self.climber.robot_body.current_position()
            torso_pos[1] += 0.15
            torso_pos[2] += 0.35
            self.debug_stance_text = self._p.addUserDebugText(
                text=f"{self.current_stance}",
                textPosition=torso_pos,
                textSize=1,
                lifeTime=0.1,
                textColorRGB=[1.0, 0.0, 1.0],
                replaceItemUniqueId=self.debug_stance_text,
            )

    def get_distance_from_desired_stance(self):
        effector_count = len(self.climber.effectors)
        dist_away = [float("inf") for _ in range(effector_count)]
        effector_positions = [
            effector.get_position() for effector in self.climber.effectors
        ]

        for eff_index in range(effector_count):
            if self.desired_stance[eff_index] == -1:
                dist_away[eff_index] = 0
                continue

            desired_hold_pos = self.targets[
                self.desired_stance[eff_index]
            ].body.get_position()
            current_eff_pos = effector_positions[eff_index]
            distance = np.abs(
                np.linalg.norm(
                    np.array(desired_hold_pos) - np.array(current_eff_pos)
                )
            )
            dist_away[eff_index] = distance
        return dist_away

    def terminate_check(self):
        terminated = False
        if self.desired_stance_index > len(self.motion_path) - 1:
            terminated = True
        if self.is_on_floor():
            terminated = True
        return terminated

    def truncate_check(self):
        return True if self.steps >= self.max_ep_steps else False

    def _get_obs(self):
        obs = []

        # --- BASELINE 306-D OBSERVATION EXTRACTION ---
        states = self._p.getLinkStates(
            self.climber.robot,
            linkIndices=[
                joint.jointIndex for joint in self.climber.ordered_joints
            ],
            computeLinkVelocity=1,
        )

        for state in states:
            (
                worldPos,
                worldOri,
                localInertialPos,
                _,
                _,
                _,
                linearVel,
                angVel,
            ) = state
            obs.extend(
                worldPos + worldOri + localInertialPos + linearVel + angVel
            )

        eff_positions = [
            eff.current_position() for eff in self.climber.effectors
        ]
        for i, c_stance in enumerate(self.desired_stance):
            if c_stance == -1:
                obs.extend([-1, -1, -1, 0])
                continue

            eff_target = self.targets[c_stance]
            dist = np.linalg.norm(
                np.array(eff_target.body.initialPosition)
                - np.array(eff_positions[i])
            )
            obs.extend(eff_target.body.initialPosition)
            obs.append(dist)

        obs.extend(
            -1 if k == -1 else self.targets[k].id for k in self.current_stance
        )
        obs.extend(
            -1 if k == -1 else self.targets[k].id for k in self.desired_stance
        )
        obs.extend(
            [
                1 if self.current_stance[i] == self.desired_stance[i] else 0
                for i in range(len(self.current_stance))
            ]
        )
        obs.extend(self.best_dist_to_stance)
        obs.append(1 if self.is_touching_body(self.floor.id) else 0)
        obs.append(1 if self.is_touching_body(self.wall.id) else 0)

        baseline_vector = np.array(obs, dtype=np.float32)

        # --- DYNAMIC KILTER WALL STATE INJECTION ---
        if not self.include_wall:
            return baseline_vector

        if self.one_hot:
            grid_state = np.zeros(
                (self.kilter_rows, self.kilter_cols, self.num_states),
                dtype=np.float32,
            )
            grid_state[:, :, 0] = (
                1.0  # Set all cells to index 0 ("unlit") by default
            )

            # Map structural targets inside config to their spatial grid coordinates
            for key, hold_asset in self.targets.items():
                if (
                    hasattr(self.config, "hold_grid_mapping")
                    and key in self.config.hold_grid_mapping
                ):
                    r = self.config.hold_grid_mapping[key]["row"]
                    c = self.config.hold_grid_mapping[key]["col"]

                    # Determine Lit state color rules
                    state_idx = 0
                    if key in self.desired_stance:
                        state_idx = 2  # Hand/Foot intermediate target marker

                    grid_state[r, c, 0] = 0.0
                    grid_state[r, c, state_idx] = 1.0

            flat_wall = grid_state.flatten()
        else:
            grid_state = np.zeros(
                (self.kilter_rows, self.kilter_cols), dtype=np.float32
            )
            for key, hold_asset in self.targets.items():
                if (
                    hasattr(self.config, "hold_grid_mapping")
                    and key in self.config.hold_grid_mapping
                ):
                    r = self.config.hold_grid_mapping[key]["row"]
                    c = self.config.hold_grid_mapping[key]["col"]
                    grid_state[r, c] = (
                        2.0 if key in self.desired_stance else 0.0
                    )

            flat_wall = grid_state.flatten()

        return np.concatenate([baseline_vector, flat_wall]).astype(np.float32)

    def _get_info(self):
        info = dict()
        success = (
            True if self.current_stance == self.desired_stance else False
        )
        info["is_success"] = success
        return info

    def is_on_floor(self):
        touching_floor = False
        floor_contact = self._p.getContactPoints(
            bodyA=self.climber.robot, bodyB=self.floor.id
        )
        for i in range(len(floor_contact)):
            contact_body = floor_contact[i][3]
            exclude_list = [
                self.climber.parts["left_foot"].bodyPartIndex,
                self.climber.parts["right_foot"].bodyPartIndex,
            ]
            if contact_body not in exclude_list:
                touching_floor = True
                break
        return touching_floor

    def is_touching_body(self, bodyB):
        contact_points = self._p.getContactPoints(
            bodyA=self.climber.robot, bodyB=bodyB
        )
        return len(contact_points) > 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visualise_reward(self, reward, min, max):
        if self.render_mode != "human":
            return
        value = np.clip(reward, min, max)
        normalized_value = (value - min) / (max - min) * (1 - 0) + 0
        colour = (
            [0.0, normalized_value / 1.0, 0.0, 1.0]
            if reward > 0.0
            else [normalized_value / 1.0, 0.0, 0.0, 1.0]
        )
        self._p.changeVisualShape(
            objectUniqueId=self.climber.robot, linkIndex=-1, rgbaColor=colour
        )
