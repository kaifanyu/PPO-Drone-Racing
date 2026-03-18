# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """
    Racing strategy implementation.

    Reward structure follows Pasumarti (2025), Eq. 4-5:
        r_t = r_prog + r_pass - r_cmd - r_crash

    Observation space (17-dim) follows the paper's Stage 1 design:
        - Relative position to next gate in gate frame  (3)
        - Quaternion orientation in world frame         (4)
        - Linear velocity in body frame                 (3)
        - Angular velocity in body frame                (3)
        - Previous control action                       (4)

    Reset strategy randomizes the starting gate and adds small
    position / yaw noise to improve policy robustness.
    """

    def __init__(self, env: QuadcopterEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            log_keys = ["progress_goal", "gate_pass", "command", "crash"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in log_keys
            }

        # ----------------------------------------------------------------
        # Initialize fixed physics parameters (no domain randomization)
        # ----------------------------------------------------------------
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value

    # ------------------------------------------------------------------
    # REWARDS
    # ------------------------------------------------------------------
    def get_rewards(self) -> torch.Tensor:

        # ----------------------------------------------------------------
        # Gate-crossing detection (needed in both train and play)
        # ----------------------------------------------------------------
        curr_x_gate = self.env._pose_drone_wrt_gate[:, 0]
        prev_x_gate = self.env._prev_x_drone_wrt_gate
        lateral_dist = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:3], dim=1)

        crossed_plane = (prev_x_gate > 0) & (curr_x_gate <= 0)
        within_gate   = lateral_dist < 0.6
        gate_passed   = crossed_plane & within_gate

        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (
                self.env._idx_wp[ids_gate_passed] + 1
            ) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            new_wp = self.env._idx_wp[ids_gate_passed]
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[new_wp, :2]
            self.env._desired_pos_w[ids_gate_passed, 2]  = self.env._waypoints[new_wp, 2]
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[new_wp, :3],
                self.env._waypoints_quat[new_wp, :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3]
            )

        self.env._prev_x_drone_wrt_gate[:] = curr_x_gate

        # Crash detection (needed in both modes to update _crashed counter)
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # Update distance tracking (needed for progress reward next step)
        curr_dist = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        delta_dist = self.env._last_distance_to_goal - curr_dist
        self.env._last_distance_to_goal[:] = curr_dist

        if self.cfg.is_train:
            # All reward scale lookups are safely inside this block
            progress_reward  = delta_dist * self.env.rew['progress_goal_reward_scale']

            gate_pass_bonus = torch.zeros(self.num_envs, device=self.device)
            if len(ids_gate_passed) > 0:
                centering = torch.clamp(1.0 - lateral_dist[ids_gate_passed], 0.0, 1.0)
                gate_pass_bonus[ids_gate_passed] = centering * self.env.rew['gate_pass_reward_scale']

            actions      = self.env._actions
            prev_actions = self.env._previous_actions
            cmd_penalty = (
                self.env.rew['command_penalty_scale']        * torch.linalg.norm(actions, dim=1)
                + self.env.rew['command_smooth_penalty_scale'] * torch.sum((actions - prev_actions) ** 2, dim=1)
            )
            crash_reward = crashed.float() * self.env.rew['crash_penalty']

            rewards_dict = {
                "progress_goal": progress_reward,
                "gate_pass":     gate_pass_bonus,
                "command":       -cmd_penalty,
                "crash":         -crash_reward,
            }
            reward = torch.sum(torch.stack(list(rewards_dict.values())), dim=0)
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward
            )
            for key, value in rewards_dict.items():
                self._episode_sums[key] += value
        else:
            # Play mode — return zero reward, state tracking still updated above
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward
    # ------------------------------------------------------------------
    # OBSERVATIONS
    # ------------------------------------------------------------------

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        17-dimensional observation vector matching Pasumarti (2025) Stage 1:

            [0:3]   Relative position to next gate in gate frame     (3)
            [3:7]   Quaternion [w, x, y, z] in world frame           (4)
            [7:10]  Linear velocity in body frame                     (3)
            [10:13] Angular velocity (body rates) in body frame       (3)
            [13:17] Previous control action                           (4)

        Using the gate-relative position as the primary navigation signal
        is crucial — it makes the observation invariant to the drone's
        absolute world position and naturally encodes where the drone needs
        to go relative to its current target gate.
        """

        # Position of drone relative to current target gate, expressed in
        # the gate's local frame (x = forward through gate, y/z = lateral)
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate          # (N, 3)

        # Quaternion gives full orientation — important for the controller
        # to understand tilt, which directly affects thrust direction
        drone_quat_w = self.env._robot.data.root_quat_w               # (N, 4)

        # Body-frame velocities let the policy understand momentum and
        # make predictive corrections rather than purely reactive ones
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b     # (N, 3)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b         # (N, 3)

        # Previous action closes the loop — the policy can reason about
        # what commands it already issued and avoid abrupt changes
        prev_actions = self.env._previous_actions                     # (N, 4)

        obs = torch.cat(
            [
                drone_pos_gate_frame,   # 3
                drone_quat_w,           # 4
                drone_lin_vel_b,        # 3
                drone_ang_vel_b,        # 3
                prev_actions,           # 4
            ],
            dim=-1,
        )  # Total: 17 dims

        observations = {"policy": obs}
        return observations

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """
        Reset strategy with randomized starting gate and position noise.

        Key design choices:
        - Drones reset at a randomly sampled gate along the full track.
          This exposes the policy to all parts of the course from early
          in training rather than only learning the first gate.
        - Small uniform noise on position (±0.3 m) and yaw (±0.2 rad)
          improves robustness and helps generalise to the sim2real gap.
        - Starting 1.5–2.5 m behind the gate (randomized) so the drone
          always has a clear approach trajectory.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # ----------------------------------------------------------------
        # Logging (training only)
        # ----------------------------------------------------------------
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = {}
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = {}
            self.env.extras["log"].update(extras)
            self.env.extras["log"]["Episode_Termination/died"] = (
                torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            )
            self.env.extras["log"]["Episode_Termination/time_out"] = (
                torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            )

        # ----------------------------------------------------------------
        # Robot / joint reset
        # ----------------------------------------------------------------
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]
            self.env._all_target_models_paths = [
                [f"{env_path}/{name}" for name in model_prim_names_in_env]
                for env_path in self.env.scene.env_prim_paths
            ]
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)

        # Stagger episode lengths at bulk reset to decorrelate environments
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Reset action / controller buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joint state to defaults
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # ----------------------------------------------------------------
        # Determine starting gate and initial pose
        # ----------------------------------------------------------------
        if self.cfg.is_train:
            # Randomly assign each resetting env a starting gate index.
            # This is the key curriculum trick: the policy sees all gates
            # from the start of training rather than needing to learn
            # gates 1–N only after mastering gate 0.
            n_waypoints = self.env._waypoints.shape[0]
            waypoint_indices = torch.randint(
                0, n_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            # Randomized approach distance: 1.5–2.5 m behind gate
            x_local = -(1.5 + torch.rand(n_reset, device=self.device))   # [-2.5, -1.5]
            # Small lateral and vertical noise for robustness
            y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

        else:
            # Play mode: fixed approach to the initial waypoint
            waypoint_indices = self.env._initial_wp * torch.ones(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5).expand(n_reset)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0,  1.0).expand(n_reset)
            z_local = torch.zeros(n_reset, device=self.device)

        # Retrieve waypoint world positions and gate yaw
        x0_wp   = self.env._waypoints[waypoint_indices, 0]
        y0_wp   = self.env._waypoints[waypoint_indices, 1]
        z_wp    = self.env._waypoints[waypoint_indices, 2]
        theta   = self.env._waypoints[waypoint_indices, -1]   # gate yaw in world frame

        # Rotate local offset into world frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_wp  + z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Point the drone toward the target gate plus small yaw noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise   = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # ----------------------------------------------------------------
        # Write state to simulation
        # ----------------------------------------------------------------
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # ----------------------------------------------------------------
        # Reset bookkeeping variables
        # ----------------------------------------------------------------
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2]  = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0
        self.env._yaw_n_laps[env_ids]     = 0
        self.env._crashed[env_ids]        = 0

        # Compute initial gate-relative pose and reset previous x tracker
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )
        # Drone starts behind the gate (positive x in gate frame)
        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0