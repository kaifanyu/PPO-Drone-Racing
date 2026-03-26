# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets.

Merged Strategy — combines the best of both original and final versions:

Dynamics:
    - Domain randomization of TWR, aero drag, and PID gains (from _final_)
    - Re-randomized on every reset for continuous sim-to-real coverage

Rewards (7 components):
    - progress_goal       : dense distance-reduction to current gate (_final_ split logic)
    - target_reached_bonus: +1.0 gate pass (centering-weighted, from _original_) / -1.0 miss
    - command             : action magnitude + smoothness penalty (from _original_)
    - alive               : small per-step survival bonus (from _original_)
    - lap_incomplete      : per-step penalty until 3 laps complete (from _final_)
    - death_cost          : terminal override on crash/OOB (both)

Observations (26D):
    - drone_pos_w (3)           : absolute world position
    - drone_lin_vel_b (3)       : body-frame linear velocity
    - drone_ang_vel_b (3)       : body-frame angular velocity
    - drone_quat_w (4)          : world-frame orientation quaternion
    - drone_pos_gate_frame (3)  : position relative to current gate in gate frame
    - gate_pos_b (3)            : current gate position in drone body frame
    - prev_actions (4)          : previous control command (memory signal)
    - next_gate_in_curr_frame(3): next gate position expressed in current gate frame (lookahead)

Reset / Curriculum:
    - Replay buffer per gate: stores last 100 crossing states (pos, vel, quat) (_final_)
    - 50% replay-buffer spawn (with Gaussian perturbation), 50% geometric spawn
    - Geometric fallback uses _original_'s richer scheme:
        50% gate-approach (1.5–2.5 m behind gate) and 50% mid-track interpolation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    wrap_to_pi,
    matrix_from_quat,
)

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Merged quadcopter racing strategy — best of original and final versions.

    See module docstring for a full description of design choices.
    """
    DEFAULT_REWARDS = {
        "progress_goal_reward_scale":            5.0,
        "target_reached_bonus_reward_scale":     10.0,
        "command_penalty_scale":                 0.0005,
        "command_smooth_penalty_scale":          0.0002,
        "alive_bonus_scale":                     0.01,
        "lap_incomplete_penalty_reward_scale":   -0.01,
        "death_cost":                            -1.0,
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, env: "QuadcopterEnv"):
        self.env = env
        if not hasattr(env, "rew") or env.rew is None:
            env.rew = dict(self.DEFAULT_REWARDS)
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # ---- Episode-sum logging ----
        if self.cfg.is_train and hasattr(env, "rew"):
            log_keys = [
                "progress_goal",
                "target_reached_bonus",
                "command",
                "alive",
                "lap_incomplete_penalty",
            ]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in log_keys
            }

        # ---- Domain randomization ranges (multipliers relative to base values) ----
        self._twr_range          = (0.95, 1.05)
        self._k_aero_xy_range    = (0.5,  2.0)
        self._k_aero_z_range     = (0.5,  2.0)
        self._kp_omega_rp_range  = (0.85, 1.15)
        self._ki_omega_rp_range  = (0.85, 1.15)
        self._kd_omega_rp_range  = (0.70, 1.30)
        self._kp_omega_y_range   = (0.85, 1.15)
        self._ki_omega_y_range   = (0.85, 1.15)
        self._kd_omega_y_range   = (0.70, 1.30)

        # ---- Spawn geometry parameters (with cfg fallback defaults) ----
        self.spawn_lateral_expand = getattr(self.cfg, "spawn_lateral_expand", 3.0)
        self.spawn_before_min     = getattr(self.cfg, "spawn_before_min",     0.5)
        self.spawn_before_max     = getattr(self.cfg, "spawn_before_max",     4.0)
        self.spawn_z_min          = getattr(self.cfg, "spawn_z_min",          0.05)
        self.spawn_z_max          = getattr(self.cfg, "spawn_z_max",          3.0)

        # ---- Apply initial domain randomization ----
        self._randomize_dynamics(torch.arange(self.num_envs, device=self.device))

        # ---- Tracking tensors ----
        self._prev_pose_drone_wrt_gate = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_distance_to_gate    = torch.zeros(self.num_envs,    device=self.device)

        # ---- Per-gate replay buffer for curriculum reset ----
        self._gate_buffer_size    = 100
        self._num_gates           = len(env._waypoints) if hasattr(env, "_waypoints") else 6
        # [num_gates, buffer_size, 10]  — stores [pos(3), vel(3), quat(4)]
        self._gate_state_buffers  = torch.zeros(
            self._num_gates, self._gate_buffer_size, 10, device=self.device
        )
        self._gate_buffer_counts  = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)
        self._gate_buffer_indices = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _randomize_dynamics(self, env_ids: torch.Tensor) -> None:
        """Randomize physics parameters for the given environments.

        Called both at __init__ and at the start of every reset, providing
        continuous domain randomization throughout training.
        """
        n = len(env_ids)

        # Thrust-to-weight ratio
        twr_mult = torch.empty(n, device=self.device).uniform_(*self._twr_range)
        self.env._thrust_to_weight[env_ids] = self.env._twr_value * twr_mult

        # Aerodynamic drag
        k_xy_mult = torch.empty(n, device=self.device).uniform_(*self._k_aero_xy_range)
        k_z_mult  = torch.empty(n, device=self.device).uniform_(*self._k_aero_z_range)
        self.env._K_aero[env_ids, :2] = self.env._k_aero_xy_value * k_xy_mult.unsqueeze(1)
        self.env._K_aero[env_ids,  2] = self.env._k_aero_z_value  * k_z_mult

        # PID gains — roll/pitch
        kp_rp = torch.empty(n, device=self.device).uniform_(*self._kp_omega_rp_range)
        ki_rp = torch.empty(n, device=self.device).uniform_(*self._ki_omega_rp_range)
        kd_rp = torch.empty(n, device=self.device).uniform_(*self._kd_omega_rp_range)
        self.env._kp_omega[env_ids, :2] = self.env._kp_omega_rp_value * kp_rp.unsqueeze(1)
        self.env._ki_omega[env_ids, :2] = self.env._ki_omega_rp_value * ki_rp.unsqueeze(1)
        self.env._kd_omega[env_ids, :2] = self.env._kd_omega_rp_value * kd_rp.unsqueeze(1)

        # PID gains — yaw
        kp_y = torch.empty(n, device=self.device).uniform_(*self._kp_omega_y_range)
        ki_y = torch.empty(n, device=self.device).uniform_(*self._ki_omega_y_range)
        kd_y = torch.empty(n, device=self.device).uniform_(*self._kd_omega_y_range)
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value * kp_y
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value * ki_y
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value * kd_y

        # Motor time constants — not randomized
        self.env._tau_m[env_ids] = self.env._tau_m_value

    # ------------------------------------------------------------------
    # Gate crossing helpers
    # ------------------------------------------------------------------

    def _detect_gate_crossing(
        self,
        current_pose_gate: torch.Tensor,
        prev_pose_gate: torch.Tensor,
        expanded_half: float,
        gate_half: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect gate pass and gate miss events.

        A crossing is detected when the drone moves from positive-x to non-positive-x
        in the current gate's local frame (gate normal = +X).

        Returns:
            gate_passed: [num_envs] — crossed plane AND within gate bounds
            gate_missed: [num_envs] — crossed plane AND within expanded bounds but OUTSIDE gate
        """
        prev_x    = prev_pose_gate[:, 0]
        current_x = current_pose_gate[:, 0]
        crossed_gate_plane = (prev_x > 0.0) & (current_x <= 0.0)

        within_gate = (
            (torch.abs(current_pose_gate[:, 1]) <= gate_half) &
            (torch.abs(current_pose_gate[:, 2]) <= gate_half)
        )
        within_expanded = (
            (torch.abs(current_pose_gate[:, 1]) <= expanded_half) &
            (torch.abs(current_pose_gate[:, 2]) <= expanded_half)
        )

        gate_passed = crossed_gate_plane & within_gate
        gate_missed = crossed_gate_plane & within_expanded & (~within_gate)
        return gate_passed, gate_missed

    def _calculate_progress_reward(
        self,
        crossed_gate_plane: torch.Tensor,
        gate_centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense progress reward: reduction in distance to current gate.

        For environments that just crossed the gate plane, progress is measured
        as the distance from the previous position to the plane (not past it),
        avoiding artificially large rewards from overshooting.

        Returns:
            progress:         [num_envs]
            current_distance: [num_envs]  (used to update tracking)
        """
        current_pos  = self.env._robot.data.root_link_pos_w
        prev_distance = self._prev_distance_to_gate.clone()

        ids_crossed     = torch.where(crossed_gate_plane)[0]
        ids_not_crossed = torch.where(~crossed_gate_plane)[0]
        current_distance = torch.zeros(self.num_envs, device=self.device)

        if ids_crossed.numel() > 0:
            # Progress = how far the drone was from the plane at the previous step
            gate_plane_dist = self._prev_pose_drone_wrt_gate[ids_crossed, 0]
            current_distance[ids_crossed] = 0.0
            prev_distance[ids_crossed]    = gate_plane_dist

        if ids_not_crossed.numel() > 0:
            current_distance[ids_not_crossed] = torch.norm(
                current_pos[ids_not_crossed] - gate_centers[ids_not_crossed], dim=1
            )

        progress = prev_distance - current_distance
        return progress, current_distance

    def _calculate_next_gate_progress(
        self,
        crossed_gate_plane: torch.Tensor,
        current_gate_centers: torch.Tensor,
        next_gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extra progress toward gate N+1, computed immediately after crossing gate N.

        Uses the current gate center as reference so the agent is rewarded for
        already flying in the right direction when it passes.
        """
        next_progress = torch.zeros(self.num_envs, device=self.device)
        ids_crossed   = torch.where(crossed_gate_plane)[0]
        if ids_crossed.numel() == 0:
            return next_progress

        drone_pos        = self.env._robot.data.root_link_pos_w[ids_crossed]
        prev_gate_centers = current_gate_centers[ids_crossed]
        next_gate_pos    = self.env._waypoints[next_gate_indices[ids_crossed], :3]
        next_gate_quat   = self.env._waypoints_quat[next_gate_indices[ids_crossed], :]

        prev_gate_in_next, _ = subtract_frame_transforms(next_gate_pos, next_gate_quat, prev_gate_centers)
        drone_in_next, _     = subtract_frame_transforms(next_gate_pos, next_gate_quat, drone_pos)

        next_progress[ids_crossed] = (
            torch.norm(prev_gate_in_next, dim=1) - torch.norm(drone_in_next, dim=1)
        )
        return next_progress

    def _handle_gate_crossing(self, gate_passed: torch.Tensor) -> None:
        """Advance waypoint index on successful crossing; store state to replay buffer.

        Only processes the CURRENT target gate, enforcing sequential gate order.
        """
        ids = torch.where(gate_passed)[0]
        if ids.numel() == 0:
            return

        current_gate_idx = self.env._idx_wp[ids]
        drone_pos  = self.env._robot.data.root_link_pos_w[ids]
        drone_vel  = self.env._robot.data.root_com_lin_vel_w[ids]
        drone_quat = self.env._robot.data.root_quat_w[ids]

        # Store crossing state in per-gate circular replay buffer
        for i, env_idx in enumerate(ids):
            gate_id   = current_gate_idx[i].item()
            write_idx = self._gate_buffer_indices[gate_id]
            self._gate_state_buffers[gate_id, write_idx, :3] = drone_pos[i]
            self._gate_state_buffers[gate_id, write_idx, 3:6] = drone_vel[i]
            self._gate_state_buffers[gate_id, write_idx, 6:]  = drone_quat[i]
            self._gate_buffer_indices[gate_id] = (write_idx + 1) % self._gate_buffer_size
            self._gate_buffer_counts[gate_id]  = min(
                self._gate_buffer_counts[gate_id] + 1, self._gate_buffer_size
            )

        # Advance waypoint pointer
        self.env._idx_wp[ids] = (self.env._idx_wp[ids] + 1) % self.env._waypoints.shape[0]
        self.env._n_gates_passed[ids] += 1

        # Update tracking for the NEW target gate
        new_gate_pos  = self.env._waypoints[self.env._idx_wp[ids], :3]
        new_gate_quat = self.env._waypoints_quat[self.env._idx_wp[ids], :]
        self._prev_distance_to_gate[ids] = torch.norm(drone_pos - new_gate_pos, dim=1)
        new_pose, _ = subtract_frame_transforms(new_gate_pos, new_gate_quat, drone_pos)
        self._prev_pose_drone_wrt_gate[ids] = new_pose
        self.env._prev_x_drone_wrt_gate[ids] = new_pose[:, 0]

    def _update_tracking_variables(
        self,
        gate_passed: torch.Tensor,
        current_distance: torch.Tensor,
    ) -> None:
        """Carry forward tracking state for all envs that did NOT successfully pass."""
        ids_not_passed = torch.where(~gate_passed)[0]
        if ids_not_passed.numel() > 0:
            self._prev_pose_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed].clone()
            )
            self.env._prev_x_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed, 0]
            )
            self._prev_distance_to_gate[ids_not_passed] = current_distance[ids_not_passed].clone()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def get_rewards(self) -> torch.Tensor:
        """Compute per-timestep rewards.

        Components
        ----------
        progress_goal         dense    distance reduction to current gate
        target_reached_bonus  sparse   +scale on pass (centering-weighted), -scale on miss
        command               dense    action magnitude + smoothness penalty
        alive                 dense    small survival bonus
        lap_incomplete        dense    per-step penalty until 3 full laps
        death_cost            sparse   terminal override on crash/OOB
        """
        # ---- Gate geometry ----
        pose_gate   = self.env._pose_drone_wrt_gate
        gate_half   = (
            getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5
            if hasattr(self.env, "_gate_model_cfg_data") else 0.5
        )
        expand        = getattr(self.cfg, "gate_detection_expand", 1.4)
        expanded_half = gate_half * expand

        gate_centers      = self.env._waypoints[self.env._idx_wp, :3]
        self.env._desired_pos_w[:] = gate_centers
        num_gates         = self.env._waypoints.shape[0]
        next_gate_indices = (self.env._idx_wp + 1) % num_gates

        # ---- Crash detection ----
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask    = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # ---- Gate crossing detection ----
        gate_passed, gate_missed = self._detect_gate_crossing(
            current_pose_gate=pose_gate,
            prev_pose_gate=self._prev_pose_drone_wrt_gate,
            expanded_half=expanded_half,
            gate_half=gate_half,
        )
        crossed_gate_plane = gate_passed | gate_missed

        # ---- Progress reward (dense) ----
        progress_current, current_distance = self._calculate_progress_reward(
            crossed_gate_plane, gate_centers
        )
        progress_next = self._calculate_next_gate_progress(
            gate_passed, gate_centers, next_gate_indices
        )
        progress = progress_current + progress_next

        # ---- Gate reward (sparse) ----
        # Pass: centering-weighted bonus from _original_ (smoother gradient signal)
        # Miss: flat penalty from _final_
        ids_passed = torch.where(gate_passed)[0]
        ids_missed = torch.where(gate_missed)[0]
        lateral_dist = torch.linalg.norm(pose_gate[:, 1:3], dim=1)

        gate_reward = torch.zeros(self.num_envs, device=self.device)
        if ids_passed.numel() > 0:
            centering = torch.clamp(1.0 - lateral_dist[ids_passed], 0.0, 1.0)
            gate_reward[ids_passed] = centering           # 0–1, scaled below
        if ids_missed.numel() > 0:
            gate_reward[ids_missed] = -1.0

        # Advance waypoint only on a successful pass
        self._handle_gate_crossing(gate_passed)

        # ---- Command penalty (from _original_) ----
        actions      = self.env._actions
        prev_actions = self.env._previous_actions
        cmd_penalty  = (
            self.env.rew["command_penalty_scale"]        * torch.linalg.norm(actions, dim=1)
            + self.env.rew["command_smooth_penalty_scale"] * torch.sum((actions - prev_actions) ** 2, dim=1)
        )

        # ---- Alive bonus (from _original_) ----
        alive_bonus = (1.0 - crashed.float()) * self.env.rew["alive_bonus_scale"]

        # ---- Lap incomplete penalty (from _final_) ----
        laps_completed   = self.env._n_gates_passed // num_gates
        not_done_three   = (laps_completed < 3).float()
        lap_penalty      = self.env.rew["lap_incomplete_penalty_reward_scale"] * not_done_three

        # ---- Update tracking for next timestep ----
        self._update_tracking_variables(gate_passed, current_distance)

        # ---- Combine and scale ----
        if self.cfg.is_train:
            rewards_dict = {
                "progress_goal":        progress    * self.env.rew["progress_goal_reward_scale"],
                "target_reached_bonus": gate_reward * self.env.rew["target_reached_bonus_reward_scale"],
                "command":              -cmd_penalty,
                "alive":                alive_bonus,
                "lap_incomplete_penalty": lap_penalty,
            }
            reward = torch.sum(torch.stack(list(rewards_dict.values())), dim=0)

            # Log episode sums before death override
            for key, value in rewards_dict.items():
                self._episode_sums[key] += value

            # Death cost overrides everything on termination
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew["death_cost"],
                reward,
            )
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    # ------------------------------------------------------------------
    # Observations  (26D)
    # ------------------------------------------------------------------

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Build the 26-D observation vector.

        Index  Signal                          Dim  Source
        -----  ------------------------------  ---  ------
        0:3    drone_pos_w                      3   world position
        3:6    drone_lin_vel_b                  3   body-frame linear velocity
        6:9    drone_ang_vel_b                  3   body-frame angular velocity
        9:13   drone_quat_w                     4   world-frame orientation
        13:16  drone_pos_gate_frame             3   position in current gate frame
        16:19  gate_pos_b                       3   current gate in drone body frame
        19:23  prev_actions                     4   previous control command
        23:26  next_gate_in_curr_frame          3   next gate expressed in current gate frame
        -----  ------------------------------  ---
               Total                           26
        """
        n_waypoints = self.env._waypoints.shape[0]

        # Drone state
        drone_pos_w     = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b
        drone_quat_w    = self.env._robot.data.root_quat_w

        # Gate-relative position (in gate frame)
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Current gate position in drone body frame
        current_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]
        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, current_gate_pos_w)

        # Previous actions (memory signal)
        prev_actions = self.env._previous_actions

        # Next gate position in current gate frame (lookahead)
        next_wp             = (self.env._idx_wp + 1) % n_waypoints
        curr_gate_quat      = self.env._waypoints_quat[self.env._idx_wp]
        next_gate_pos_w     = self.env._waypoints[next_wp, :3]
        next_gate_in_curr_frame, _ = subtract_frame_transforms(
            current_gate_pos_w, curr_gate_quat, next_gate_pos_w
        )

        obs = torch.cat(
            [
                drone_pos_w,              # 3
                drone_lin_vel_b,          # 3
                drone_ang_vel_b,          # 3
                drone_quat_w,             # 4
                drone_pos_gate_frame,     # 3
                gate_pos_b,               # 3
                prev_actions,             # 4
                next_gate_in_curr_frame,  # 3
            ],
            dim=-1,
        )  # Total: 26

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids: Optional[torch.Tensor]) -> None:
        """Reset specific environments.

        Spawn strategy (training only):
            50% — sample from per-gate replay buffer (Gaussian-perturbed crossing state)
            50% — geometric spawn using _original_'s 50/50 gate-approach / mid-track scheme
        """
        if env_ids is None:
            env_ids_tensor = self.env._robot._ALL_INDICES
        else:
            env_ids_tensor = env_ids
            if env_ids_tensor.shape[0] == self.num_envs:
                env_ids_tensor = self.env._robot._ALL_INDICES

        # Re-randomize dynamics for these environments
        self._randomize_dynamics(env_ids_tensor)

        # ---- Episode logging ----
        if self.cfg.is_train and hasattr(self, "_episode_sums"):
            extras = {}
            for key in self._episode_sums:
                avg = torch.mean(self._episode_sums[key][env_ids_tensor])
                extras[f"Episode_Reward/{key}"] = avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids_tensor] = 0.0
            self.env.extras["log"] = {}
            self.env.extras["log"].update(extras)
            self.env.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(
                self.env.reset_terminated[env_ids_tensor]
            ).item()
            self.env.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
                self.env.reset_time_outs[env_ids_tensor]
            ).item()

        self.env._robot.reset(env_ids_tensor)

        # ---- Initialize gate model paths once ----
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            names = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]
            self.env._all_target_models_paths = [
                [f"{p}/{n}" for n in names] for p in self.env.scene.env_prim_paths
            ]
            self.env._models_paths_initialized = True

        n_reset = env_ids_tensor.shape[0]

        # Stagger episode lengths at bulk reset to decorrelate environments
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # ---- Reset buffers ----
        self.env._actions[env_ids_tensor]                = 0.0
        self.env._previous_actions[env_ids_tensor]       = 0.0
        self.env._previous_yaw[env_ids_tensor]           = 0.0
        self.env._motor_speeds[env_ids_tensor]           = 0.0
        self.env._previous_omega_meas[env_ids_tensor]    = 0.0
        self.env._previous_omega_err[env_ids_tensor]     = 0.0
        self.env._omega_err_integral[env_ids_tensor]     = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids_tensor]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids_tensor]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        default_root_state = self.env._robot.data.default_root_state[env_ids_tensor]

        # ================================================================
        # Determine spawn position (training only)
        # ================================================================
        if self.cfg.is_train:
            n_waypoints = self.env._waypoints.shape[0]

            # Random target gate for each resetting env
            waypoint_indices = torch.randint(
                0, n_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )

            pos  = torch.zeros(n_reset, 3, device=self.device)
            vel  = torch.zeros(n_reset, 3, device=self.device)
            quat = torch.zeros(n_reset, 4, device=self.device)

            # Perturbation bounds for buffer-based spawning
            pos_noise_std = 0.3   # metres
            vel_noise_std = 0.3   # m/s
            ori_noise_std = 0.15  # radians (yaw only)

            gate_half_spawn = (
                getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5
                if hasattr(self.env, "_gate_model_cfg_data") else 0.5
            )
            lat_half = gate_half_spawn * self.spawn_lateral_expand

            for i in range(n_reset):
                gate_id      = waypoint_indices[i].item()
                buffer_count = self._gate_buffer_counts[gate_id].item()

                # 50% chance to use replay buffer (if it has data)
                use_buffer = (buffer_count > 0) and (torch.rand(1).item() > 0.5)

                if use_buffer:
                    # ---- Replay-buffer spawn ----
                    sample_idx    = torch.randint(0, buffer_count, (1,), device=self.device).item()
                    sampled_state = self._gate_state_buffers[gate_id, sample_idx]

                    base_pos  = sampled_state[:3]
                    base_vel  = sampled_state[3:6]
                    base_quat = sampled_state[6:]

                    pos[i] = base_pos + torch.randn(3, device=self.device) * pos_noise_std
                    vel[i] = base_vel + torch.randn(3, device=self.device) * vel_noise_std

                    # Perturb yaw only
                    base_yaw = torch.atan2(
                        2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
                        1.0 - 2.0 * (base_quat[2] ** 2 + base_quat[3] ** 2),
                    )
                    perturbed_yaw = base_yaw + torch.randn(1, device=self.device).item() * ori_noise_std
                    quat[i] = quat_from_euler_xyz(
                        torch.zeros(1, device=self.device),
                        torch.zeros(1, device=self.device),
                        torch.tensor([perturbed_yaw], device=self.device),
                    )[0]

                else:
                    # ---- Geometric spawn (_original_'s gate-approach / mid-track scheme) ----
                    x0_wp = self.env._waypoints[gate_id, 0]
                    y0_wp = self.env._waypoints[gate_id, 1]
                    z_wp  = self.env._waypoints[gate_id, 2]
                    theta = self.env._waypoints[gate_id, -1]
                    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

                    # Gate-approach: 1.5–2.5 m behind the target gate
                    x_local = -(1.5 + torch.rand(1, device=self.device))
                    y_local = torch.empty(1, device=self.device).uniform_(-0.3, 0.3)
                    z_local = torch.empty(1, device=self.device).uniform_(-0.2, 0.2)
                    ax = x0_wp - (cos_t * x_local - sin_t * y_local)
                    ay = y0_wp - (sin_t * x_local + cos_t * y_local)
                    az = z_wp  + z_local

                    # Mid-track: random interpolation between previous and current gate
                    prev_gate_id = (gate_id - 1) % n_waypoints
                    px = self.env._waypoints[prev_gate_id, 0]
                    py = self.env._waypoints[prev_gate_id, 1]
                    pz = self.env._waypoints[prev_gate_id, 2]
                    alpha = torch.rand(1, device=self.device)
                    mx = px + alpha * (x0_wp - px) + torch.empty(1, device=self.device).uniform_(-0.3, 0.3)
                    my = py + alpha * (y0_wp - py) + torch.empty(1, device=self.device).uniform_(-0.3, 0.3)
                    mz = pz + alpha * (z_wp  - pz) + torch.empty(1, device=self.device).uniform_(-0.2, 0.2)

                    # 50/50 gate-approach vs mid-track
                    use_mid = torch.rand(1).item() < 0.5
                    px_f = mx if use_mid else ax
                    py_f = my if use_mid else ay
                    pz_f = mz if use_mid else az

                    pos[i] = torch.stack([px_f.squeeze(), py_f.squeeze(), pz_f.squeeze()])
                    vel[i] = 0.0

                    # Point drone toward gate + small yaw noise
                    yaw = torch.atan2(y0_wp - py_f, x0_wp - px_f)
                    yaw_noise = torch.empty(1, device=self.device).uniform_(-0.2, 0.2)
                    quat[i] = quat_from_euler_xyz(
                        torch.zeros(1, device=self.device),
                        torch.zeros(1, device=self.device),
                        (yaw + yaw_noise).reshape(1),
                    )[0]

            default_root_state[:, 0:3]  = pos
            default_root_state[:, 3:7]  = quat
            default_root_state[:, 7:10] = vel
            default_root_state[:, 10:]  = 0.0

        else:
            # ---- Play mode: fixed approach to the initial waypoint ----
            n_reset_play = env_ids_tensor.shape[0]
            waypoint_indices = self.env._initial_wp * torch.ones(
                n_reset_play, device=self.device, dtype=self.env._idx_wp.dtype
            )

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0,  1.0)
            x_rot   = cos_t * x_local - sin_t * y_local
            y_rot   = sin_t * x_local + cos_t * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)
            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            default_root_state[:, 3:7] = quat

        # ---- Write to simulation ----
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids_tensor)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)

        # ---- Update bookkeeping ----
        self.env._idx_wp[env_ids_tensor] = waypoint_indices
        self.env._desired_pos_w[env_ids_tensor, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids_tensor,  2] = self.env._waypoints[waypoint_indices,  2].clone()

        self.env._last_distance_to_goal[env_ids_tensor] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids_tensor, :2]
            - self.env._robot.data.root_link_pos_w[env_ids_tensor, :2],
            dim=1,
        )
        self.env._n_gates_passed[env_ids_tensor] = 0
        self.env._yaw_n_laps[env_ids_tensor]     = 0
        self.env._crashed[env_ids_tensor]         = 0

        # Gate-relative pose and tracking init
        self.env._pose_drone_wrt_gate[env_ids_tensor], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids_tensor], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids_tensor], :],
            self.env._robot.data.root_link_state_w[env_ids_tensor, :3],
        )
        self._prev_pose_drone_wrt_gate[env_ids_tensor] = (
            self.env._pose_drone_wrt_gate[env_ids_tensor].clone()
        )
        self.env._prev_x_drone_wrt_gate[env_ids_tensor] = (
            self.env._pose_drone_wrt_gate[env_ids_tensor, 0]
        )

        drone_pos_reset = self.env._robot.data.root_link_pos_w[env_ids_tensor]
        gate_pos_reset  = self.env._waypoints[self.env._idx_wp[env_ids_tensor], :3]
        self._prev_distance_to_gate[env_ids_tensor] = torch.norm(
            drone_pos_reset - gate_pos_reset, dim=1
        )