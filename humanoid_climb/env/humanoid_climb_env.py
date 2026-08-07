import json
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from humanoid_climb.assets.asset import Asset
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.curriculum import Curriculum

FINISH_ROLE = 14
FOOT_ROLE = 15
START_ROLE = 12
MIDDLE_ROLE = 13


class HumanoidClimbEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config,
        render_mode: str | None = None,
        max_ep_steps: int | None = 602,
        state_file: str | None = None,
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
        self.total_env_steps = 0
        csv_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "routes",
            "climbs_intermediate.csv",
        )
        self.curriculum = Curriculum(csv_path)

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
                    "one_hot_encoded": True,
                },
            }

        board_cfg = self.kilter_cfg["kilter_board"]
        train_cfg = self.kilter_cfg["training_settings"]

        self.kilter_rows = board_cfg["rows"]
        self.kilter_cols = board_cfg["cols"]
        self.num_states = len(board_cfg["states"])
        self.include_wall = train_cfg["include_wall_state"]
        self.one_hot = train_cfg["one_hot_encoded"]
        self.num_closest_holds = 15

        # ----------------------------------------

        self.init_from_state = not state_file is None
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
        self.grasp_waste_penalty = 0.0
        self.grasp_premature_release_penalty = -20.0
        self._prev_attached = [-1, -1, -1, -1]

        self.grasp_persist_steps = grasp_persist_steps
        self._grasp_lock_remaining = [0, 0, 0, 0]
        self._last_grasp_binary = [0, 0, 0, 0]
        self._release_intent_buffer = [0, 0, 0, 0]

        if self.discrete_grasp:
            self.action_space = gym.spaces.MultiDiscrete(
                [n_torque_bins] * 17 + [2] * 4
            )
        else:
            self.action_space = gym.spaces.Box(-1, 1, (21,), np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        self.current_stance = []

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._p.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=-90,
            cameraPitch=0,
            cameraTargetPosition=[0, 0, 3],
        )
        self._p.setGravity(0, 0, -15.0)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.config.timestep_interval,
            numSubSteps=self.config.timestep_per_action,
        )

        self.floor = Asset(self._p, self.config.plane)
        self.wall = Asset(self._p, self.config.surface)
        self.climber = Humanoid(self._p, self.config.climber)
        self.prevheight = self.get_com_height()

        self.debug_stance_text = self._p.addUserDebugText(
            text="",
            textPosition=[0, 0, 0],
            textSize=1,
            lifeTime=0.1,
            textColorRGB=[1.0, 0.0, 1.0],
        )

        self.targets = {}
        self._build_route()
        self.climber.targets = self.targets

        # Dynamically calculated observation space dimension matching actual _get_obs() output
        total_obs_dim = len(self._get_obs())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

    def _build_route(self):
        if hasattr(self, "targets") and self.targets:
            for key in list(self.targets.keys()):
                self._p.removeBody(self.targets[key].id)
            self.targets.clear()
        else:
            self.targets = {}

        route = self.curriculum.get_route(self.total_env_steps)
        self.current_route_info = route
        self.config.hold_grid_mapping = {}

        for hx, hy, role in zip(
            route["holes_x"], route["holes_y"], route["role_ids"]
        ):
            key = f"hold_{hx}_{hy}"
            # Unmirrored Y coordinates: hx=8 -> +Y (left), hx=136 -> -Y (right)
            y_phys = (72 - hx) * (2.0 / 144.0)
            # Shift board height up so lowest holds are at 1.6m to prevent straight legs hitting floor
            z_phys = hy * (2.7 / 156.0) + 1.6

            hold_cfg = {
                "asset": "asset_hold",
                "position": [0.37, y_phys, z_phys],
                "orientation": [0, 0, 0, 1],
                "asset_data": self.config.assets.get("asset_hold", {}),
            }

            self.targets[key] = Asset(self._p, hold_cfg)

            self.config.hold_grid_mapping[key] = {
                "row": min(11, int(hy // 13)),
                "col": min(11, int((144 - hx) // 12)),
                "type": role,
            }

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

    def _spawn_on_start_holds(self):
        grid_mapping = getattr(self.config, "hold_grid_mapping", {})
        start_keys = [
            k
            for k, v in grid_mapping.items()
            if v.get("type") == START_ROLE or v.get("type") == 1
        ]
        if not start_keys:
            sorted_holds = sorted(
                self.targets.keys(),
                key=lambda k: self.targets[k].body.initialPosition[2],
            )
            start_keys = sorted_holds[:2]

        target_positions = [
            self.targets[k].body.initialPosition for k in start_keys
        ]
        if len(target_positions) > 0:
            avg_pos = np.mean(target_positions, axis=0)
            # Position torso in front of wall with clearance and below start hold center
            # Agent faces +X by default (arms extend to +0.39). To reach holds at 0.37, base should be at -0.02
            new_base_pos = [-0.02, avg_pos[1], max(0.5, avg_pos[2] - 0.55)]
            self._p.resetBasePositionAndOrientation(
                self.climber.robot, new_base_pos, [0, 0, 0, 1]
            )
            self._p.resetBaseVelocity(
                self.climber.robot, [0, 0, 0], [0, 0, 0]
            )

        # Reset all joint angles to zero so legs hang straight down below pelvis
        for joint in self.climber.joints.values():
            joint.reset_position(0, 0)

        # Sort start holds by Y coordinate (+Y is left side, -Y is right side)
        sorted_start_keys = sorted(
            start_keys, key=lambda k: self.targets[k].body.initialPosition[1]
        )

        # Right hand (index 1) -> lowest Y (+Y left, -Y right -> index 0)
        rh_key = sorted_start_keys[0]
        rh_pos = self.targets[rh_key].body.initialPosition
        self.climber.effectors[1]
        self._p.resetBasePositionAndOrientation(
            self.climber.robot, new_base_pos, [0, 0, 0, 1]
        )
        self.climber.force_attach(
            eff_index=1, target_key=rh_key, force=5000, attach_pos=rh_pos
        )

        # Left hand (index 0) -> highest Y (index -1)
        lh_key = sorted_start_keys[-1]
        lh_pos = self.targets[lh_key].body.initialPosition
        self.climber.force_attach(
            eff_index=0, target_key=lh_key, force=5000, attach_pos=lh_pos
        )

        self.current_stance = list(self.climber.effector_attached_to)

    def _decode_action(self, action):
        if self.discrete_grasp:
            action = np.asarray(action)
            torque_indices = action[:17].astype(np.float32)
            torques = (torque_indices / (self.n_torque_bins - 1)) * 2.0 - 1.0
            grasps = action[17:21].astype(np.float32) * 2.0 - 1.0
            decoded = np.concatenate([torques, grasps]).astype(np.float32)
        else:
            decoded = np.asarray(action, dtype=np.float32)

        for i in range(4):
            intent_binary = 1 if decoded[17 + i] > 0 else 0
            
            # Sticky Grasp Filter: require 3 consecutive releases to detach
            if intent_binary == 0 and self._last_grasp_binary[i] == 1:
                self._release_intent_buffer[i] += 1
                if self._release_intent_buffer[i] < 3:
                    intent_binary = 1
            else:
                self._release_intent_buffer[i] = 0

            if self._grasp_lock_remaining[i] > 0:
                intent_binary = self._last_grasp_binary[i]
                self._grasp_lock_remaining[i] -= 1
            elif (
                self.grasp_persist_steps > 0
                and intent_binary != self._last_grasp_binary[i]
            ):
                self._last_grasp_binary[i] = intent_binary
                self._grasp_lock_remaining[i] = self.grasp_persist_steps - 1
            else:
                self._last_grasp_binary[i] = intent_binary
                
            decoded[17 + i] = 1.0 if intent_binary == 1 else -1.0

        return decoded

    def _grasp_event_reward(self, prev_attached, decoded_grasps):
        if not self.grasp_reward:
            return 0.0
        new_attached = self.climber.effector_attached_to
        v_z = self.climber.speed()[2]
        n_attached_after = sum(1 for a in new_attached if a != -1)
        bonus = 0.0
        for i in range(4):
            was = prev_attached[i]
            now = new_attached[i]
            grasp_on = decoded_grasps[i] > 0
            new_attach = was == -1 and now != -1
            new_release = was != -1 and now == -1
            if new_attach:
                if now not in self.touched_holds:
                    bonus += 15.0 if i >= 2 else 10.0  # +15 for feet, +10 for hands
                    self.touched_holds.add(now)
                else:
                    bonus += self.grasp_attach_bonus  # Small bonus for re-attaching
            elif grasp_on and now == -1:
                bonus += self.grasp_waste_penalty
            if new_release and v_z < 0 and n_attached_after < 2:
                bonus += self.grasp_premature_release_penalty
        return bonus

    def step(self, action):
        self._p.stepSimulation()
        self.steps += 1
        self.total_env_steps += 1

        prev_attached = list(self.climber.effector_attached_to)
        decoded_action = self._decode_action(action)
        self.climber.apply_action(decoded_action)
        self.update_stance()

        ob = self._get_obs()
        info = self._get_info()

        reward = self.calculate_reward_negative_distance()
        reward += self._grasp_event_reward(
            prev_attached, decoded_action[17:21]
        )

        grid_mapping = getattr(self.config, "hold_grid_mapping", {})
        finish_keys = [
            k
            for k, v in grid_mapping.items()
            if v.get("type") == FINISH_ROLE or v.get("type") == 4
        ]
        finish_indices = [
            int(k.split("_")[1]) for k in finish_keys if "_" in k
        ]

        left_hand_hold = self.current_stance[0]
        right_hand_hold = self.current_stance[1]

        lh_on_finish = (left_hand_hold in finish_keys) or (
            left_hand_hold in finish_indices
        )
        rh_on_finish = (right_hand_hold in finish_keys) or (
            right_hand_hold in finish_indices
        )

        if lh_on_finish and rh_on_finish and len(finish_keys) > 0:
            giant_finish_reward = 2500.0
            reward += giant_finish_reward
            terminated = True
            info["is_success"] = True
            print(
                f"--- ROUTE TOPPED OUT! Giant reward of +{giant_finish_reward} applied. ---"
            )
        else:
            terminated = self.terminate_check()

        truncated = self.truncate_check()
        return ob, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.climber.reset()
        self.steps = 0
        # Lock hands for initial 30 steps so climber has a stable start window
        self._grasp_lock_remaining = [30, 30, 0, 0]
        self._last_grasp_binary = [1, 1, -1, -1]
        self._release_intent_buffer = [0, 0, 0, 0]
        self.current_stance = [-1, -1, -1, -1]

        self._build_route()
        self.climber.targets = self.targets
        
        # Populate valid targets for each limb (prevent hands on footholds)
        grid_mapping = getattr(self.config, "hold_grid_mapping", {})
        self.climber.valid_targets = [set(), set(), set(), set()]
        for key in self.targets:
            hold_type = grid_mapping.get(key, {}).get("type", 0)
            is_foot = (hold_type == FOOT_ROLE or hold_type == 15)
            
            if not is_foot:
                self.climber.valid_targets[0].add(key)
                self.climber.valid_targets[1].add(key)
                
            self.climber.valid_targets[2].add(key)
            self.climber.valid_targets[3].add(key)

        self._spawn_on_start_holds()

        # 100-step calming phase: damp base velocity and apply zero action to reach static equilibrium
        # Make sure grasp commands keep hands attached (17, 18) and feet detached (19, 20) during reset
        calm_action = np.zeros(self.action_space.shape, dtype=np.float32)
        calm_action[17:19] = 1.0
        calm_action[19:21] = -1.0

        for _ in range(100):
            self.climber.apply_action(calm_action)
            self._p.resetBaseVelocity(
                self.climber.robot, [0, 0, 0], [0, 0, 0]
            )
            self._p.stepSimulation()

        self.update_stance()

        self.prevheight = self.get_com_height()
        self.max_height = self.prevheight
        self.previous_height = self.prevheight
        self.initial_height = self.prevheight
        
        self.touched_holds = set([h for h in self.current_stance if h != -1])

        ob = self._get_obs()
        info = self._get_info()

        return np.array(ob, dtype=np.float32), info

    def calculate_reward_negative_distance(self):
        com_height = self.get_com_height()
        
        height_reward = 0.0
        if com_height > self.max_height:
            height_reward = (com_height - self.max_height) * 50.0  # Buffed one-time reward
            self.max_height = com_height

        reward = height_reward
        
        # Continuous upward incentive
        passive_height_reward = max(0, com_height - self.initial_height) * 0.1
        reward += passive_height_reward
        
        # Energy Penalty for arms
        arm_actions = self.climber.current_body_actions[11:17]
        energy_penalty = np.sum(np.abs(arm_actions)) * 0.01  # small penalty
        reward -= energy_penalty
        
        # Proximity Reward for unattached limbs
        proximity_reward = 0.0
        effector_positions = [eff.current_position() for eff in self.climber.effectors]
        
        for i in range(4):
            if self.current_stance[i] != -1:  # If limb is attached
                if i >= 2:  # Feet
                    reward += 0.25  # Massive continuous reward for standing on a hold
            else:  # If limb is unattached
                min_dist = float('inf')
                limb_pos = np.array(effector_positions[i])
                
                # Find closest hold not currently being grabbed
                for key, target in self.targets.items():
                    if key not in self.current_stance:
                        target_pos = np.array(target.body.get_position())
                        
                        # Foot Proximity Filter: only target holds below the COM
                        if i >= 2 and target_pos[2] > com_height:
                            continue
                            
                        dist = np.linalg.norm(target_pos - limb_pos)
                        if dist < min_dist:
                            min_dist = dist
                            
                if min_dist != float('inf'):
                    proximity_reward += max(0, 1.5 - min_dist) * 0.1  # Gravity well
                    
        reward += proximity_reward
        
        if not self.is_on_floor():
            reward += 0.05
        else:
            reward -= 10

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

        grid_mapping = getattr(self.config, "hold_grid_mapping", {})
        attached_holds = {
            h for h in self.current_stance if h != -1 and h in self.targets
        }

        for key, asset in self.targets.items():
            if key in attached_holds:
                # Currently held hold -> Bright Lime Green
                color = [0.0, 1.0, 0.0, 1.0]
            else:
                hold_type = grid_mapping.get(key, {}).get("type", 0)
                if hold_type == START_ROLE or hold_type == 1:
                    color = [0.0, 0.7, 1.0, 0.85]  # Cyan for Start holds
                elif hold_type == FINISH_ROLE or hold_type == 4:
                    color = [1.0, 0.2, 0.2, 0.85]  # Red for Finish holds
                elif hold_type == FOOT_ROLE or hold_type == 15:
                    color = [
                        1.0,
                        0.8,
                        0.0,
                        0.85,
                    ]  # Orange/Yellow for Foot holds
                else:
                    color = [0.4, 0.4, 0.4, 0.75]  # Grey for Middle holds

            self._p.changeVisualShape(
                objectUniqueId=asset.id, linkIndex=-1, rgbaColor=color
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
        return bool(self.is_on_floor())

    def truncate_check(self):
        return self.steps >= self.max_ep_steps

    def _get_obs(self):
        obs = []

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
        for pos in eff_positions:
            obs.extend(pos)

        obs.append(1 if self.is_touching_body(self.floor.id) else 0)
        obs.append(1 if self.is_touching_body(self.wall.id) else 0)

        baseline_vector = np.array(obs, dtype=np.float32)

        if not self.include_wall:
            return baseline_vector

        # Egocentric Hold Observation
        torso_pos = np.array(self.climber.robot_body.current_position())
        
        holds_info = []
        for key in self.targets:
            hold_asset = self.targets[key]
            hold_pos = np.array(hold_asset.body.get_position())
            dist = np.linalg.norm(hold_pos - torso_pos)
            
            hold_type = 0
            if hasattr(self.config, "hold_grid_mapping") and key in self.config.hold_grid_mapping:
                hold_type = self.config.hold_grid_mapping[key].get("type", 0)
                
            dx, dy, dz = hold_pos - torso_pos
            holds_info.append((dist, dx, dy, dz, hold_type))
            
        # Sort by distance and pick the closest N holds
        holds_info.sort(key=lambda x: x[0])
        closest_holds = holds_info[:self.num_closest_holds]
        
        # Pad with zeros if there are fewer holds than self.num_closest_holds
        while len(closest_holds) < self.num_closest_holds:
            closest_holds.append((0.0, 0.0, 0.0, 0.0, 0))
            
        # Append relative coordinates and hold type to the observation
        for hold in closest_holds:
            obs.extend([hold[1], hold[2], hold[3], float(hold[4])])
            
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        info = {}
        info["is_success"] = False
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

    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view_matrix = self._p.computeViewMatrix(
                cameraEyePosition=[-3.5, 0, 1.5],
                cameraTargetPosition=[0, 0, 1.5],
                cameraUpVector=[0, 0, 1],
            )
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(width) / height,
                nearVal=0.1,
                farVal=100.0,
            )
            _, _, rgba, _, _ = self._p.getCameraImage(
                width,
                height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
            )
            return np.reshape(rgba, (height, width, 4))[:, :, :3].astype(
                np.uint8
            )

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
