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
            log_keys = ["progress_goal", "gate_pass", "command", "crash", "alive"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in log_keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:

        # ----------------------------------------------------------------
        # Progress toward CURRENT target gate (computed before any gate
        # update so delta_dist always references the same gate on both ends)
        # ----------------------------------------------------------------
        curr_dist = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        delta_dist = self.env._last_distance_to_goal - curr_dist
        self.env._last_distance_to_goal[:] = curr_dist

        # ----------------------------------------------------------------
        # Gate-crossing detection: current target gate only
        # _pose_drone_wrt_gate is in the current target gate's local frame,
        # so this can only trigger for _idx_wp's gate, not any other gate.
        # ----------------------------------------------------------------
        curr_x_gate = self.env._pose_drone_wrt_gate[:, 0]
        prev_x_gate = self.env._prev_x_drone_wrt_gate
        lateral_dist = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:3], dim=1)

        # Drone must cross the gate plane in the correct direction (positive → non-positive)
        crossed_plane = (prev_x_gate > 0) & (curr_x_gate <= 0)
        # Drone must be within the gate opening (gate_side = 1.0 m, half-extent = 0.5 m)
        within_gate   = lateral_dist < 0.5
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
            # Reset distance tracking to the new target gate so next step's
            # delta_dist computes progress toward the new gate correctly.
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                self.env._desired_pos_w[ids_gate_passed] - self.env._robot.data.root_link_pos_w[ids_gate_passed], dim=1
            )
            # Keep _prev_x consistent with the updated (new) gate frame.
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = self.env._pose_drone_wrt_gate[ids_gate_passed, 0]

        self.env._prev_x_drone_wrt_gate[:] = curr_x_gate
        # Restore the new-gate x for passing envs (overwritten by the line above).
        if len(ids_gate_passed) > 0:
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = self.env._pose_drone_wrt_gate[ids_gate_passed, 0]

        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        if self.cfg.is_train:
            progress_reward = delta_dist * self.env.rew['progress_goal_reward_scale']

            gate_pass_bonus = torch.zeros(self.num_envs, device=self.device)
            if len(ids_gate_passed) > 0:
                centering = torch.clamp(1.0 - lateral_dist[ids_gate_passed], 0.0, 1.0)
                gate_pass_bonus[ids_gate_passed] = centering * self.env.rew['gate_pass_reward_scale']

            actions      = self.env._actions
            prev_actions = self.env._previous_actions
            cmd_penalty = (
                self.env.rew['command_penalty_scale'] * torch.linalg.norm(actions, dim=1)
                + self.env.rew['command_smooth_penalty_scale'] * torch.sum((actions - prev_actions) ** 2, dim=1)
            )
            crash_reward = crashed.float() * self.env.rew['crash_penalty']

            alive_bonus = (1.0 - crashed.float()) * self.env.rew['alive_bonus_scale']

            rewards_dict = {
                "progress_goal": progress_reward,
                "gate_pass":     gate_pass_bonus,
                "command":       -cmd_penalty,
                "crash":         -crash_reward,
                "alive":         alive_bonus,
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
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        20-dimensional observation vector:

            [0:3]   Relative position to current gate in gate frame   (3)
            [3:7]   Quaternion [w, x, y, z] in world frame            (4)
            [7:10]  Linear velocity in body frame                      (3)
            [10:13] Angular velocity (body rates) in body frame        (3)
            [13:17] Previous control action                            (4)
            [17:20] Next gate position in current gate frame           (3)
        """
        n_waypoints = self.env._waypoints.shape[0]

        # Gate-relative position: x=forward through gate, y/z=lateral offset
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate          # (N, 3)
        # Full orientation — needed to understand thrust direction
        drone_quat_w        = self.env._robot.data.root_quat_w        # (N, 4)
        # Body-frame velocities allow the policy to anticipate momentum
        drone_lin_vel_b     = self.env._robot.data.root_com_lin_vel_b # (N, 3)
        drone_ang_vel_b     = self.env._robot.data.root_ang_vel_b     # (N, 3)
        # Previous action lets the policy reason about what it already issued
        prev_actions        = self.env._previous_actions              # (N, 4)

        # Next gate position in current gate frame.
        # Lets the policy pre-plan the turn toward gate N+1 before passing gate N.
        next_wp             = (self.env._idx_wp + 1) % n_waypoints
        curr_gate_pos       = self.env._waypoints[self.env._idx_wp, :3]   # (N, 3)
        curr_gate_quat      = self.env._waypoints_quat[self.env._idx_wp]  # (N, 4)
        next_gate_pos_w     = self.env._waypoints[next_wp, :3]            # (N, 3)
        next_gate_in_curr_frame, _ = subtract_frame_transforms(
            curr_gate_pos, curr_gate_quat, next_gate_pos_w
        )  # (N, 3)

        obs = torch.cat(
            [
                drone_pos_gate_frame,       # 3
                drone_quat_w,               # 4
                drone_lin_vel_b,            # 3
                drone_ang_vel_b,            # 3
                prev_actions,               # 4
                next_gate_in_curr_frame,    # 3
            ],
            dim=-1,
        )  # Total: 20 dims

        observations = {"policy": obs}
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

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
            n_waypoints = self.env._waypoints.shape[0]
            waypoint_indices = torch.randint(
                0, n_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            x0_wp = self.env._waypoints[waypoint_indices, 0]
            y0_wp = self.env._waypoints[waypoint_indices, 1]
            z_wp  = self.env._waypoints[waypoint_indices, 2]
            theta = self.env._waypoints[waypoint_indices, -1]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # --- Gate-approach spawn: 1.5–2.5 m behind the target gate ---
            x_local   = -(1.5 + torch.rand(n_reset, device=self.device))
            y_local   = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            z_local   = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
            x_rot     = cos_theta * x_local - sin_theta * y_local
            y_rot     = sin_theta * x_local + cos_theta * y_local
            approach_x = x0_wp - x_rot
            approach_y = y0_wp - y_rot
            approach_z = z_wp  + z_local

            # --- Mid-track spawn: random point between previous gate and current gate ---
            # Teaches the policy to navigate between gates, not just approach a single gate.
            prev_wp = (waypoint_indices - 1) % n_waypoints
            prev_x  = self.env._waypoints[prev_wp, 0]
            prev_y  = self.env._waypoints[prev_wp, 1]
            prev_z  = self.env._waypoints[prev_wp, 2]
            alpha   = torch.rand(n_reset, device=self.device)  # 0=near prev gate, 1=near current gate
            mid_x   = prev_x + alpha * (x0_wp - prev_x) + torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            mid_y   = prev_y + alpha * (y0_wp - prev_y) + torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            mid_z   = prev_z + alpha * (z_wp  - prev_z) + torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)

            # 50% gate-approach, 50% mid-track
            use_midtrack = torch.rand(n_reset, device=self.device) < 0.5
            initial_x = torch.where(use_midtrack, mid_x, approach_x)
            initial_y = torch.where(use_midtrack, mid_y, approach_y)
            initial_z = torch.where(use_midtrack, mid_z, approach_z)

        else:
            # Play mode: fixed approach to the initial waypoint
            waypoint_indices = self.env._initial_wp * torch.ones(
                n_reset, device=self.device, dtype=self.env._idx_wp.dtype
            )
            x0_wp = self.env._waypoints[waypoint_indices, 0]
            y0_wp = self.env._waypoints[waypoint_indices, 1]
            z_wp  = self.env._waypoints[waypoint_indices, 2]
            theta = self.env._waypoints[waypoint_indices, -1]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            x_local   = torch.empty(1, device=self.device).uniform_(-3.0, -0.5).expand(n_reset)
            y_local   = torch.empty(1, device=self.device).uniform_(-1.0,  1.0).expand(n_reset)
            x_rot     = cos_theta * x_local - sin_theta * y_local
            y_rot     = sin_theta * x_local + cos_theta * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = z_wp

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

        # Use the known reset position directly rather than reading root_link_pos_w,
        # which may be stale (physics hasn't stepped yet after write_root_link_pose_to_sim).
        reset_pos = torch.stack([initial_x, initial_y, initial_z], dim=1)
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - reset_pos, dim=1
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