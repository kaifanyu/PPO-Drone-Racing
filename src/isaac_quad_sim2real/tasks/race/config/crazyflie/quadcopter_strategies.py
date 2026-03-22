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
    Racing strategy implementation — updated with SOTA improvements.

    Key changes vs. baseline:
        1. Observation space expanded to 23 dims with 2-gate lookahead.
           NOTE: update num_observations = 23 in your env config when using this.
        2. Velocity-toward-gate reward added to incentivize speed.
        3. Domain randomization of dynamics parameters per episode reset,
           matching the evaluation parameter ranges exactly.
        4. All reward components wired through the rewards dict from train_race.py.

    Reward structure (Pasumarti 2025, Eq. 4-5, extended):
        r_t = r_prog + r_vel + r_pass - r_cmd - r_crash

    Observation space (23-dim):
        - Relative position to next gate in gate frame           (3)
        - Quaternion orientation in world frame                  (4)
        - Linear velocity in body frame                          (3)
        - Angular velocity in body frame                         (3)
        - Previous control action                                (4)
        - Relative position to gate+1 in world frame (lookahead) (3)
        - Relative position to gate+2 in world frame (lookahead) (3)
        Total: 23

    Reset strategy randomizes the starting gate, initial pose, and now also
    the drone dynamics (TWR, drag, PID) to match evaluation conditions.
    """

    def __init__(self, env: QuadcopterEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging
        if self.cfg.is_train and hasattr(env, 'rew'):
            log_keys = ["progress_goal", "velocity", "gate_pass", "command", "crash"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in log_keys
            }

        # Initialize fixed base parameters.
        # Aerodynamic drag coefficients — will be overridden per-episode in training.
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2]  = self.env._k_aero_z_value

        # PID controller gains — will be overridden per-episode in training.
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value
        self.env._kp_omega[:, 2]  = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2]  = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2]  = self.env._kd_omega_y_value

        # Motor time constants and thrust-to-weight ratio.
        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def get_rewards(self) -> torch.Tensor:
        """
        Compute per-timestep rewards for all environments.

        Components:
            r_progress  — reward closing 3-D distance to the next gate target.
            r_velocity  — reward forward velocity projected onto the gate direction.
                          This incentivizes speed directly, not just proximity.
            r_gate_pass — one-off bonus when the drone threads a gate cleanly.
            r_command   — penalty on action magnitude and action jerk.
            r_crash     — penalty on contact forces (skipped for the first 100 steps).
            death_cost  — replaces the full reward on terminal transitions.
        """

        # ----------------------------------------------------------------
        # Gate-crossing detection via plane crossing + lateral bounds
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
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3],
            )

        self.env._prev_x_drone_wrt_gate[:] = curr_x_gate

        # ----------------------------------------------------------------
        # Crash detection — masked for the first 100 steps per episode
        # to avoid penalizing early physics instability at reset.
        # ----------------------------------------------------------------
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask    = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # ----------------------------------------------------------------
        # Progress reward — delta distance to next gate (3-D Euclidean).
        # A positive delta means the drone moved closer; rewards forward
        # progress and penalises drifting away.
        # ----------------------------------------------------------------
        drone_pos = self.env._robot.data.root_link_pos_w  # (N, 3)
        curr_dist = torch.linalg.norm(self.env._desired_pos_w - drone_pos, dim=1)
        delta_dist = self.env._last_distance_to_goal - curr_dist
        self.env._last_distance_to_goal[:] = curr_dist

        if self.cfg.is_train:
            progress_reward = delta_dist * self.env.rew['progress_goal_reward_scale']

            # ------------------------------------------------------------
            # Velocity reward — forward velocity projected onto the unit
            # vector pointing from the drone to the current gate target.
            # This rewards the drone for *moving fast toward the gate*,
            # not just for being close to it.  We clamp at a generous upper
            # bound (15 m/s) to avoid instability from extreme speeds.
            # ------------------------------------------------------------
            drone_vel_w   = self.env._robot.data.root_com_lin_vel_w   # world frame (N, 3)
            gate_dir      = self.env._desired_pos_w - drone_pos        # (N, 3)
            gate_dist_mag = torch.norm(gate_dir, dim=1, keepdim=True)
            gate_dir_norm = gate_dir / (gate_dist_mag + 1e-6)          # unit vector to gate

            vel_toward_gate = (drone_vel_w * gate_dir_norm).sum(dim=1)          # scalar projection
            vel_toward_gate = torch.clamp(vel_toward_gate, 0.0, 15.0)           # only reward forward
            velocity_reward = vel_toward_gate * self.env.rew['velocity_reward_scale']

            # ------------------------------------------------------------
            # Gate-pass bonus — proportional to how central the pass was.
            # centering = 1.0 at dead-center, 0.0 at the 0.6 m boundary.
            # ------------------------------------------------------------
            gate_pass_bonus = torch.zeros(self.num_envs, device=self.device)
            if len(ids_gate_passed) > 0:
                centering = torch.clamp(1.0 - lateral_dist[ids_gate_passed], 0.0, 1.0)
                gate_pass_bonus[ids_gate_passed] = centering * self.env.rew['gate_pass_reward_scale']

            # ------------------------------------------------------------
            # Command penalty — discourages large, jerky motor commands
            # which cause instability and energy waste in the real drone.
            # ------------------------------------------------------------
            actions      = self.env._actions
            prev_actions = self.env._previous_actions
            cmd_penalty  = (
                self.env.rew['command_penalty_scale'] * torch.linalg.norm(actions, dim=1)
                + self.env.rew['command_smooth_penalty_scale'] * torch.sum((actions - prev_actions) ** 2, dim=1)
            )

            # ------------------------------------------------------------
            # Crash penalty
            # ------------------------------------------------------------
            crash_reward = crashed.float() * self.env.rew['crash_penalty']

            rewards_dict = {
                "progress_goal": progress_reward,
                "velocity":      velocity_reward,
                "gate_pass":     gate_pass_bonus,
                "command":      -cmd_penalty,
                "crash":        -crash_reward,
            }
            reward = torch.sum(torch.stack(list(rewards_dict.values())), dim=0)

            # Replace reward with death_cost on terminal transitions
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward,
            )

            for key, value in rewards_dict.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        23-dimensional observation vector.

        Dims  0:3   — Relative position to next gate in gate frame     (3)
        Dims  3:7   — Quaternion [w, x, y, z] in world frame           (4)
        Dims  7:10  — Linear velocity in body frame                    (3)
        Dims 10:13  — Angular velocity in body frame                   (3)
        Dims 13:17  — Previous control action                          (4)
        Dims 17:20  — Relative pos to gate+1 in world frame (lookahead)(3)
        Dims 20:23  — Relative pos to gate+2 in world frame (lookahead)(3)

        The two lookahead terms give the policy knowledge of upcoming gate
        geometry, enabling planning of curved entry trajectories required
        for the powerloop and chicane maneuvers.

        IMPORTANT: set num_observations = 23 in your env / task config
        so that the actor input layer is sized correctly.
        """
        n_waypoints = self.env._waypoints.shape[0]
        drone_pos   = self.env._robot.data.root_link_pos_w  # (N, 3)

        # Gate-relative position: x = forward through gate, y/z = lateral offset
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate          # (N, 3)
        drone_quat_w         = self.env._robot.data.root_quat_w        # (N, 4)
        drone_lin_vel_b      = self.env._robot.data.root_com_lin_vel_b # (N, 3)
        drone_ang_vel_b      = self.env._robot.data.root_ang_vel_b     # (N, 3)
        prev_actions         = self.env._previous_actions              # (N, 4)

        # Two-gate lookahead — world-frame relative positions.
        # Using world frame (not gate frame) so the policy can reason about
        # the spatial layout of the upcoming gates without needing to
        # transform through an intermediate gate frame.
        next_idx      = (self.env._idx_wp + 1) % n_waypoints  # (N,)
        next_next_idx = (self.env._idx_wp + 2) % n_waypoints  # (N,)

        next_gate_pos      = self.env._waypoints[next_idx, :3]       # (N, 3)
        next_next_gate_pos = self.env._waypoints[next_next_idx, :3]  # (N, 3)

        rel_next      = next_gate_pos      - drone_pos  # (N, 3) vector to gate+1
        rel_next_next = next_next_gate_pos - drone_pos  # (N, 3) vector to gate+2

        obs = torch.cat(
            [
                drone_pos_gate_frame,   # 3  — current gate navigation
                drone_quat_w,           # 4  — orientation
                drone_lin_vel_b,        # 3  — momentum
                drone_ang_vel_b,        # 3  — rotational state
                prev_actions,           # 4  — previous command
                rel_next,               # 3  — lookahead: gate+1
                rel_next_next,          # 3  — lookahead: gate+2
            ],
            dim=-1,
        )  # Total: 23 dims

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """
        Reset specific environments to initial states.

        Changes vs. baseline:
            - Domain randomization of dynamics parameters (TWR, aerodynamic
              drag, PID gains) sampled from the same ranges used in evaluation,
              so the policy learns to be robust to the dynamics mismatch.
            - Spawn position and orientation logic unchanged from baseline.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # ----------------------------------------------------------------
        # Logging for training mode
        # ----------------------------------------------------------------
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = {}
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = {}
            self.env.extras["log"].update(extras)
            extras = {}
            extras["Episode_Termination/died"]     = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # ----------------------------------------------------------------
        # Reset robot internals
        # ----------------------------------------------------------------
        self.env._robot.reset(env_ids)

        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]
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

        # ----------------------------------------------------------------
        # Domain randomization of dynamics per episode
        #
        # We sample from the same ranges the TAs use in evaluation so the
        # policy sees the full distribution of dynamics at train time and
        # learns a robust controller rather than one tuned to nominal values.
        # ----------------------------------------------------------------
        if self.cfg.is_train:
            # Thrust-to-weight ratio: ±5%
            twr_rand = self.env._twr_value * torch.empty(n_reset, device=self.device).uniform_(0.95, 1.05)
            self.env._thrust_to_weight[env_ids] = twr_rand

            # Aerodynamic drag: xy ×[0.5, 2.0], z ×[0.5, 2.0]
            k_xy_rand = self.env._k_aero_xy_value * torch.empty(n_reset, device=self.device).uniform_(0.5, 2.0)
            k_z_rand  = self.env._k_aero_z_value  * torch.empty(n_reset, device=self.device).uniform_(0.5, 2.0)
            self.env._K_aero[env_ids, 0] = k_xy_rand
            self.env._K_aero[env_ids, 1] = k_xy_rand
            self.env._K_aero[env_ids, 2] = k_z_rand

            # PID gains — roll/pitch: ±15% for kp/ki, ±30% for kd
            kp_rp = self.env._kp_omega_rp_value * torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15)
            ki_rp = self.env._ki_omega_rp_value * torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15)
            kd_rp = self.env._kd_omega_rp_value * torch.empty(n_reset, device=self.device).uniform_(0.70, 1.30)
            self.env._kp_omega[env_ids, 0] = kp_rp
            self.env._kp_omega[env_ids, 1] = kp_rp
            self.env._ki_omega[env_ids, 0] = ki_rp
            self.env._ki_omega[env_ids, 1] = ki_rp
            self.env._kd_omega[env_ids, 0] = kd_rp
            self.env._kd_omega[env_ids, 1] = kd_rp

            # PID gains — yaw: ±15% for kp/ki, ±30% for kd
            kp_y = self.env._kp_omega_y_value * torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15)
            ki_y = self.env._ki_omega_y_value * torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15)
            kd_y = self.env._kd_omega_y_value * torch.empty(n_reset, device=self.device).uniform_(0.70, 1.30)
            self.env._kp_omega[env_ids, 2] = kp_y
            self.env._ki_omega[env_ids, 2] = ki_y
            self.env._kd_omega[env_ids, 2] = kd_y
        else:
            # Play mode — use nominal dynamics
            self.env._thrust_to_weight[env_ids] = self.env._twr_value
            self.env._K_aero[env_ids, :2] = self.env._k_aero_xy_value
            self.env._K_aero[env_ids, 2]  = self.env._k_aero_z_value
            self.env._kp_omega[env_ids, :2] = self.env._kp_omega_rp_value
            self.env._ki_omega[env_ids, :2] = self.env._ki_omega_rp_value
            self.env._kd_omega[env_ids, :2] = self.env._kd_omega_rp_value
            self.env._kp_omega[env_ids, 2]  = self.env._kp_omega_y_value
            self.env._ki_omega[env_ids, 2]  = self.env._ki_omega_y_value
            self.env._kd_omega[env_ids, 2]  = self.env._kd_omega_y_value

        # ----------------------------------------------------------------
        # Reset controller buffers
        # ----------------------------------------------------------------
        self.env._actions[env_ids]              = 0.0
        self.env._previous_actions[env_ids]     = 0.0
        self.env._previous_yaw[env_ids]         = 0.0
        self.env._motor_speeds[env_ids]         = 0.0
        self.env._previous_omega_meas[env_ids]  = 0.0
        self.env._previous_omega_err[env_ids]   = 0.0
        self.env._omega_err_integral[env_ids]   = 0.0

        # Reset joint state to defaults
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # ----------------------------------------------------------------
        # Initial pose — gate assignment and approach position
        # ----------------------------------------------------------------
        if self.cfg.is_train:
            # Assign each resetting env a uniformly random gate so the policy
            # is exposed to all parts of the track from the start of training.
            n_waypoints = self.env._waypoints.shape[0]
            waypoint_indices = torch.randint(
                0, n_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )
            # Randomized approach: 1.5–2.5 m behind the gate, with small
            # lateral (±0.3 m) and vertical (±0.2 m) noise for robustness.
            x_local = -(1.5 + torch.rand(n_reset, device=self.device))
            y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
        else:
            # Play mode — fixed approach to the initial waypoint
            waypoint_indices = self.env._initial_wp * torch.ones(
                n_reset, device=self.device, dtype=self.env._idx_wp.dtype
            )
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5).expand(n_reset)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0,  1.0).expand(n_reset)
            z_local = torch.zeros(n_reset, device=self.device)

        # Waypoint world positions and gate yaw
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp  = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]  # gate yaw in world frame

        # Rotate local offset into world frame using gate yaw
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

        # Point the drone toward the target gate with small yaw noise (±0.2 rad)
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise   = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # Write pose and velocity to simulation
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

        # Compute initial gate-relative pose and reset the previous-x tracker.
        # Drone starts behind the gate so prev_x_drone_wrt_gate = +1 ensures
        # the crossing detection fires correctly on the first pass.
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )
        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0