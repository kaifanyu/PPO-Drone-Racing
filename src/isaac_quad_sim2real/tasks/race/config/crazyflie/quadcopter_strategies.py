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
    """Default strategy for quadcopter racing rewards, observations, and resets.
    
    This strategy implements a dense-reward racing policy that encourages the drone to:
    1. Make continuous progress toward the current gate (dense distance-based reward)
    2. Successfully pass through gates in the correct direction (sparse bonus)
    3. Avoid crashes and obstacles (dense penalty)
    4. Complete 3 full laps of the race track (lap completion incentive)
    
    Key Features:
    -------------
    - Dense progress reward based on Euclidean distance reduction to gate centers
    - Gate crossing detection using velocity projection in gate frame
    - Domain randomization of dynamics parameters for sim-to-real transfer
    - Comprehensive observation space including drone state and gate information
    """

    def __init__(self, env: QuadcopterEnv):
        """Initialize the racing strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [
                key.split("_reward_scale")[0]
                for key, value in env.rew.items()
                if key != "death_cost" and value is not None
            ]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Define domain randomization ranges (multipliers relative to base values)
        # These ranges are used to simulate sim2real gap during evaluation
        self._twr_range = (0.95, 1.05)
        self._k_aero_xy_range = (0.5, 2.0)
        self._k_aero_z_range = (0.5, 2.0)
        self._kp_omega_rp_range = (0.85, 1.15)
        self._ki_omega_rp_range = (0.85, 1.15)
        self._kd_omega_rp_range = (0.7, 1.3)
        self._kp_omega_y_range = (0.85, 1.15)
        self._ki_omega_y_range = (0.85, 1.15)
        self._kd_omega_y_range = (0.7, 1.3)
        
        # Spawn randomization parameters for curriculum reset (with fallback defaults)
        # These are used when replay buffer is empty for a given gate
        self.spawn_circle_scale = getattr(self.cfg, 'spawn_circle_scale', 3.0)
        self.spawn_lateral_expand = getattr(self.cfg, 'spawn_lateral_expand', 3.0)
        self.spawn_before_min = getattr(self.cfg, 'spawn_before_min', 0.5)
        self.spawn_before_max = getattr(self.cfg, 'spawn_before_max', 4.0)
        self.spawn_after_min = getattr(self.cfg, 'spawn_after_min', 0.5)
        self.spawn_after_max = getattr(self.cfg, 'spawn_after_max', 4.0)
        self.spawn_z_min = getattr(self.cfg, 'spawn_z_min', 0.05)
        self.spawn_z_max = getattr(self.cfg, 'spawn_z_max', 3.0)

        # Initialize parameters with domain randomization for all environments
        # This helps the policy generalize to dynamics variations at evaluation time
        self._randomize_dynamics(torch.arange(self.num_envs, device=self.device))

        # Track previous position in gate frame for velocity calculation
        self._prev_pose_drone_wrt_gate = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Track previous distance to target for progress reward calculation
        self._prev_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        
        # Replay buffer for curriculum reset: store historical states at each gate
        # Each gate has a buffer of size [buffer_size, state_dim=10] storing [pos(3), vel(3), quat(4)]
        self._gate_buffer_size = 100  # Keep last 100 crossings per gate
        self._num_gates = len(env._waypoints) if hasattr(env, '_waypoints') else 6
        # Initialize buffers: [num_gates, buffer_size, state_dim]
        self._gate_state_buffers = torch.zeros(
            self._num_gates, self._gate_buffer_size, 10, device=self.device
        )
        # Track how many states are stored in each gate's buffer (for sampling)
        self._gate_buffer_counts = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)
        # Track next write index for each gate's buffer (circular)
        self._gate_buffer_indices = torch.zeros(self._num_gates, dtype=torch.long, device=self.device)

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        """Randomize dynamics parameters for specified environments.
        
        This method implements domain randomization to help the policy generalize
        to dynamics variations that will be present during evaluation.
        
        Args:
            env_ids: Tensor of environment indices to randomize.
        """
        n = len(env_ids)
        
        # Thrust-to-weight ratio (TWR): 0.95x to 1.05x
        twr_mult = torch.empty(n, device=self.device).uniform_(*self._twr_range)
        self.env._thrust_to_weight[env_ids] = self.env._twr_value * twr_mult
        
        # Aerodynamic drag coefficients: 0.5x to 2.0x
        k_aero_xy_mult = torch.empty(n, device=self.device).uniform_(*self._k_aero_xy_range)
        k_aero_z_mult = torch.empty(n, device=self.device).uniform_(*self._k_aero_z_range)
        self.env._K_aero[env_ids, :2] = self.env._k_aero_xy_value * k_aero_xy_mult.unsqueeze(1)
        self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value * k_aero_z_mult
        
        # PID gains for roll and pitch
        kp_rp_mult = torch.empty(n, device=self.device).uniform_(*self._kp_omega_rp_range)
        ki_rp_mult = torch.empty(n, device=self.device).uniform_(*self._ki_omega_rp_range)
        kd_rp_mult = torch.empty(n, device=self.device).uniform_(*self._kd_omega_rp_range)
        self.env._kp_omega[env_ids, :2] = self.env._kp_omega_rp_value * kp_rp_mult.unsqueeze(1)
        self.env._ki_omega[env_ids, :2] = self.env._ki_omega_rp_value * ki_rp_mult.unsqueeze(1)
        self.env._kd_omega[env_ids, :2] = self.env._kd_omega_rp_value * kd_rp_mult.unsqueeze(1)
        
        # PID gains for yaw
        kp_y_mult = torch.empty(n, device=self.device).uniform_(*self._kp_omega_y_range)
        ki_y_mult = torch.empty(n, device=self.device).uniform_(*self._ki_omega_y_range)
        kd_y_mult = torch.empty(n, device=self.device).uniform_(*self._kd_omega_y_range)
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value * kp_y_mult
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value * ki_y_mult
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value * kd_y_mult
        
        # Motor time constants (keep constant, not in randomization spec)
        self.env._tau_m[env_ids] = self.env._tau_m_value

    def _detect_gate_crossing(
        self,
        current_pose_gate: torch.Tensor,
        prev_pose_gate: torch.Tensor,
        expanded_half: float,
        gate_half: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect if the drone has crossed through a gate (passed/missed).
        
        This method measures the velocity vector (position change from previous to current)
        in the gate frame, where the +X axis aligns with the gate normal. A negative X velocity
        indicates movement through the gate (inward direction).
        
        Args:
            current_pose_gate: Current drone position in gate frame [num_envs, 3]
            prev_pose_gate: Previous drone position in gate frame [num_envs, 3]
            expanded_half: Half-size of expanded gate opening (for detection bounds)
            gate_half: Half-size of actual gate opening (for pass/miss judgment)
            
        Returns:
            gate_passed: Boolean tensor indicating successful gate passage [num_envs]
        """
        # Gate crossing detection: similar to ManagerBased command logic
        # 1) Must cross the gate plane in the forward direction (x from + to <= 0)
        prev_x = prev_pose_gate[:, 0]
        current_x = current_pose_gate[:, 0]
        crossed_gate_plane = (prev_x > 0.0) & (current_x <= 0.0)
        
        # 2) Check lateral alignment at current step against gate opening
        within_gate_bounds = (
            (torch.abs(current_pose_gate[:, 1]) <= gate_half) &
            (torch.abs(current_pose_gate[:, 2]) <= gate_half)
        )
        
        # 3) Allow an expanded detection window for misses, if provided
        within_expanded_at_cross = (
            (torch.abs(current_pose_gate[:, 1]) <= expanded_half) &
            (torch.abs(current_pose_gate[:, 2]) <= expanded_half)
        )
        
        gate_passed = crossed_gate_plane & within_gate_bounds
        gate_missed = crossed_gate_plane & within_expanded_at_cross & (~within_gate_bounds)
        
        # NOTE: gate_passed/gate_missed are only for the CURRENT target gate.
        # If drone skips a gate (e.g., goes from gate 0 to gate 2), it will NOT trigger
        # detection for gate 2 because pose_gate is always relative to current target (_idx_wp).
        
        return gate_passed, gate_missed

    def _calculate_progress_reward(
        self,
        crossed_gate_plane: torch.Tensor,
        gate_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate dense progress reward based on distance reduction to current gate.
        
        Progress is computed as the reduction in distance from previous to current timestep.
        For environments that just crossed a gate, progress is the distance from the previous
        position to the gate plane.
        
        Args:
            crossed_gate_plane: Boolean mask indicating which envs crossed the gate [num_envs]
            gate_centers: Position of current gate centers in world frame [num_envs, 3]
            
        Returns:
            progress: Dense reward signal, positive when moving closer to gate [num_envs]
        """
        current_pos = self.env._robot.data.root_link_pos_w
        prev_distance = self._prev_distance_to_gate.clone()
        
        ids_crossed = torch.where(crossed_gate_plane)[0]
        ids_not_crossed = torch.where(~crossed_gate_plane)[0]
        
        current_distance = torch.zeros(self.num_envs, device=self.device)
        
        # For envs that crossed: progress = distance from prev_pos to gate plane
        if ids_crossed.numel() > 0:
            prev_pose_gate_crossed = self._prev_pose_drone_wrt_gate[ids_crossed]
            gate_plane_distance = prev_pose_gate_crossed[:, 0]  # x in gate frame
            current_distance[ids_crossed] = 0.0  # At gate plane
            prev_distance[ids_crossed] = gate_plane_distance
        
        # For envs that didn't cross: use euclidean distance to current gate center
        if ids_not_crossed.numel() > 0:
            current_gate_pos = gate_centers[ids_not_crossed]
            current_distance[ids_not_crossed] = torch.norm(
                current_pos[ids_not_crossed] - current_gate_pos, dim=1
            )
        
        # Progress = reduction in distance (positive when closer, negative when farther)
        progress = prev_distance - current_distance
        return progress, current_distance

    def _calculate_next_gate_progress(
        self,
        crossed_gate_plane: torch.Tensor,
        current_gate_centers: torch.Tensor,
        next_gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute additional progress immediately after crossing the current gate.
        
        This term measures how much closer the drone is to the next gate, assuming the
        starting reference for the next segment is exactly the current gate center.
        Both the reference point (current gate center) and the drone pose are
        expressed in the next gate's coordinate frame, so the progress mirrors the
        existing "prev_distance - current_distance" logic.
        """
        next_progress = torch.zeros(self.num_envs, device=self.device)
        ids_crossed = torch.where(crossed_gate_plane)[0]
        if ids_crossed.numel() == 0:
            return next_progress

        drone_pos = self.env._robot.data.root_link_pos_w[ids_crossed]
        prev_gate_centers = current_gate_centers[ids_crossed]
        next_gate_pos = self.env._waypoints[next_gate_indices[ids_crossed], :3]
        next_gate_quat = self.env._waypoints_quat[next_gate_indices[ids_crossed], :]

        prev_gate_in_next_frame, _ = subtract_frame_transforms(
            next_gate_pos, next_gate_quat, prev_gate_centers
        )
        drone_pose_next_gate, _ = subtract_frame_transforms(
            next_gate_pos, next_gate_quat, drone_pos
        )

        prev_distance = torch.norm(prev_gate_in_next_frame, dim=1)
        current_distance = torch.norm(drone_pose_next_gate, dim=1)
        next_progress[ids_crossed] = prev_distance - current_distance
        return next_progress

    def _handle_gate_crossing(self, crossed_gate_plane: torch.Tensor) -> None:
        """Handle gate crossing events: advance waypoint index and update tracking variables.
        
        Only processes crossings of the CURRENT target gate (self.env._idx_wp) in the CORRECT direction.
        This ensures sequential gate passing: must pass gate 0, then gate 1, then gate 2, etc.
        
        Also stores the crossing state into the replay buffer for curriculum learning.
        
        Args:
            crossed_gate_plane: Boolean mask indicating which envs crossed the CURRENT target gate [num_envs]
        """
        ids_to_next_gate = torch.where(crossed_gate_plane)[0]
        
        if ids_to_next_gate.numel() > 0:
            # Verify that we're crossing the CURRENT target gate (not a skipped gate)
            # This is already ensured by _detect_gate_crossing which only checks self.env._idx_wp
            # Store current state to replay buffer before advancing
            current_gate_idx = self.env._idx_wp[ids_to_next_gate]
            drone_pos = self.env._robot.data.root_link_pos_w[ids_to_next_gate]
            drone_vel = self.env._robot.data.root_com_lin_vel_w[ids_to_next_gate]
            drone_quat = self.env._robot.data.root_quat_w[ids_to_next_gate]
            
            # For each unique gate crossed, store states to its buffer
            for i, env_idx in enumerate(ids_to_next_gate):
                gate_id = current_gate_idx[i].item()
                write_idx = self._gate_buffer_indices[gate_id]
                
                # Store state: [pos(3), vel(3), quat(4)]
                self._gate_state_buffers[gate_id, write_idx, :3] = drone_pos[i]
                self._gate_state_buffers[gate_id, write_idx, 3:6] = drone_vel[i]
                self._gate_state_buffers[gate_id, write_idx, 6:] = drone_quat[i]
                
                # Update circular buffer index
                self._gate_buffer_indices[gate_id] = (write_idx + 1) % self._gate_buffer_size
                # Update count (saturate at buffer size)
                self._gate_buffer_counts[gate_id] = min(
                    self._gate_buffer_counts[gate_id] + 1, self._gate_buffer_size
                )
            
            # Advance to next gate
            self.env._idx_wp[ids_to_next_gate] = (
                self.env._idx_wp[ids_to_next_gate] + 1
            ) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_to_next_gate] += 1
            
            # Update tracking for new gate
            new_gate_pos = self.env._waypoints[self.env._idx_wp[ids_to_next_gate], :3]
            new_gate_quat = self.env._waypoints_quat[self.env._idx_wp[ids_to_next_gate], :]
            
            self._prev_distance_to_gate[ids_to_next_gate] = torch.norm(
                drone_pos - new_gate_pos, dim=1
            )
            
            new_pose_drone_wrt_gate, _ = subtract_frame_transforms(
                new_gate_pos, new_gate_quat, drone_pos
            )
            self._prev_pose_drone_wrt_gate[ids_to_next_gate] = new_pose_drone_wrt_gate
            self.env._prev_x_drone_wrt_gate[ids_to_next_gate] = new_pose_drone_wrt_gate[:, 0]


    def _update_tracking_variables(
        self,
        gate_passed: torch.Tensor,
        current_distance: torch.Tensor,
    ) -> None:
        """Update previous pose and distance for next timestep.
        
        Note: Environments that successfully passed gates have already been updated in _handle_gate_crossing()
        to track the new gate. This function updates all other environments (including those that missed gates).
        
        Args:
            gate_passed: Boolean mask of envs that successfully passed gate [num_envs]
            current_distance: Current distance to gate for all envs [num_envs]
        """
        # Update all environments that did NOT successfully pass the gate
        # This includes both environments that missed gates and those that didn't cross
        ids_not_passed = torch.where(~gate_passed)[0]
        if ids_not_passed.numel() > 0:
            self._prev_pose_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed].clone()
            )
            self.env._prev_x_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed, 0]
            )
            self._prev_distance_to_gate[ids_not_passed] = current_distance[ids_not_passed].clone()

    def get_rewards(self) -> torch.Tensor:
        """Compute per-timestep rewards for all environments.
        
        Reward Structure:
        -----------------
        1. progress_goal (dense): Euclidean distance reduction to current gate center.
           - Range: typically ±0.2 m per step, scaled by progress_goal_reward_scale
           - Positive when moving closer, negative when drifting away
           
        2. target_reached_bonus (sparse): Gate pass/miss reward.
           - +1.0 when successfully passing through a gate (within bounds)
           - -1.0 when missing a gate (crossed plane but outside bounds)
           - Both scaled by target_reached_bonus_reward_scale (default: 400.0)
           
        3. lap_incomplete_penalty (dense negative): Per-step penalty until 3 laps completed.
           - Encourages completing the full race
           - Scaled by lap_incomplete_penalty_reward_scale
           
        4. death_cost (sparse negative): Large penalty applied on episode termination.
           - Only applied when reset_terminated is True (crash, altitude violation, etc.)
           - Overrides all other rewards in the termination timestep
           
        Returns:
            reward: Total reward for current timestep [num_envs]
        """
        # ========== Gate Setup ==========
        pose_gate = self.env._pose_drone_wrt_gate  # [x,y,z] in gate frame
        gate_half = getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5 \
            if hasattr(self.env, "_gate_model_cfg_data") else 0.5
        expand = getattr(self.cfg, "gate_detection_expand", 1.4)
        expanded_half = gate_half * expand
        
        gate_centers = self.env._waypoints[self.env._idx_wp, :3]
        self.env._desired_pos_w[:] = gate_centers  # Update desired position marker
        num_gates = self.env._waypoints.shape[0]
        next_gate_indices = (self.env._idx_wp + 1) % num_gates
        
        # ========== Crash Detection ==========
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # ========== Gate Crossing Detection ==========
        # IMPORTANT: pose_gate is relative to CURRENT target gate (self.env._idx_wp)
        # This ensures we only detect crossings of the current target gate, preventing
        # bonus rewards for skipping gates (e.g., going from gate 0 directly to gate 2).
        # The detection only triggers when:
        # 1. Current target is gate N
        # 2. Drone crosses gate N in correct direction
        # 3. Only then does _idx_wp advance to gate N+1
        gate_passed, gate_missed = self._detect_gate_crossing(
            current_pose_gate=pose_gate,  # Relative to current target gate (_idx_wp)
            prev_pose_gate=self._prev_pose_drone_wrt_gate,
            expanded_half=expanded_half,
            gate_half=gate_half,
        )
        # crossed_gate_plane includes ALL envs that crossed the plane (passed OR missed)
        crossed_gate_plane = gate_passed | gate_missed
        
        # ========== Reward Components ==========
        # 1. Progress reward (dense)
        progress_current, current_distance = self._calculate_progress_reward(
            crossed_gate_plane, gate_centers
        )
        # Only calculate next gate progress for successfully passed gates
        progress_next_gate = self._calculate_next_gate_progress(
            gate_passed, gate_centers, next_gate_indices
        )

        # Combine progress from current gate and next gate
        progress = progress_current + progress_next_gate
        
        # 2. Gate crossing reward (sparse): +1.0 for pass, -1.0 for miss
        # Only advance waypoint index on successful pass (not on miss)
        self._handle_gate_crossing(gate_passed)
        
        # Combine pass (+1.0) and miss (-1.0) into single reward signal
        # Both will be scaled by target_reached_bonus_reward_scale
        gate_reward = gate_passed.float() * 1.0 + gate_missed.float() * (-1.0)
        
        # 5. Update tracking for next timestep
        self._update_tracking_variables(gate_passed, current_distance)
        
        # ========== Compute Final Reward ==========
        if self.cfg.is_train:
            # Scale each reward component
            progress_reward = progress * self.env.rew['progress_goal_reward_scale']
            # Use target_reached_bonus_reward_scale for both pass (+1.0) and miss (-1.0)
            gate_reward_scaled = gate_reward * self.env.rew['target_reached_bonus_reward_scale']
            # Lap incomplete penalty: per-step penalty until 3 laps done
            num_gates = self.env._waypoints.shape[0]
            laps_completed = self.env._n_gates_passed // num_gates
            not_done_three = (laps_completed < 3).float()
            lap_penalty = (
                self.env.rew['lap_incomplete_penalty_reward_scale'] * not_done_three
            )
            
            # Aggregate all components
            rewards = {
                "progress_goal": progress_reward,
                "target_reached_bonus": gate_reward_scaled,
                "lap_incomplete_penalty": lap_penalty,
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Log episodic sums BEFORE applying death cost override
            # This way we track what the agent "earned" in each component
            for key, value in rewards.items():
                self._episode_sums[key] += value
            
            # Apply death cost on termination
            # When died, the actual reward received is death_cost (overrides everything)
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward
            )
        else:
            # Eval mode: no rewards
            reward = torch.zeros(self.num_envs, device=self.device)
        
        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Construct observation vector for the policy network.
        Observation Space (19D total):
        ------------------------------
        - drone_pos_w (3): Drone position in world frame [x, y, z]
        - drone_lin_vel_b (3): Linear velocity in body frame
        - drone_ang_vel_b (3): Angular velocity in body frame
        - drone_quat_w (4): Orientation quaternion [w, x, y, z]
        - drone_pos_gate_frame (3): Drone position relative to current gate in gate frame
        - gate_pos_b (3): Current gate position in drone body frame
        
        Returns:
            observations: Dict with "policy" key containing concatenated observation tensor [num_envs, 19]
        """
        # ========== Drone States ==========
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        # ========== Gate Information ==========
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]

        # Drone position relative to gate (in gate frame)
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Gate position relative to drone (in body frame)
        gate_pos_b, _ = subtract_frame_transforms(
            drone_pos_w, drone_quat_w, current_gate_pos_w
        )

        # ========== Concatenate Observation Vector ==========
        obs = torch.cat(
            [
                drone_pos_w,            # (3) World position
                drone_lin_vel_b,        # (3) Body-frame linear velocity
                drone_ang_vel_b,        # (3) Body-frame angular velocity
                drone_quat_w,           # (4) Orientation quaternion
                drone_pos_gate_frame,   # (3) Position in gate frame
                gate_pos_b,             # (3) Gate in body frame
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None:
            env_ids_tensor = self.env._robot._ALL_INDICES
        else:
            env_ids_tensor = env_ids
            if env_ids_tensor.shape[0] == self.num_envs:
                env_ids_tensor = self.env._robot._ALL_INDICES

        # Randomize dynamics parameters for the resetting environments
        # This provides continuous domain randomization throughout training
        self._randomize_dynamics(env_ids_tensor)

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids_tensor] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids_tensor]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids_tensor]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids_tensor)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = env_ids_tensor.shape[0]
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids_tensor] = 0.0
        self.env._previous_actions[env_ids_tensor] = 0.0
        self.env._previous_yaw[env_ids_tensor] = 0.0
        self.env._motor_speeds[env_ids_tensor] = 0.0
        self.env._previous_omega_meas[env_ids_tensor] = 0.0
        self.env._previous_omega_err[env_ids_tensor] = 0.0
        self.env._omega_err_integral[env_ids_tensor] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids_tensor]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids_tensor]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        default_root_state = self.env._robot.data.default_root_state[env_ids_tensor]

        # TODO ----- START ----- Curriculum reset: sample from replay buffer with bounded perturbation
        # For each env, randomly select a gate and sample a historical state from that gate's buffer
        
        n_waypoints = self.env._waypoints.shape[0]
        
        # Randomly select gates for each resetting env
        start_indices = torch.randint(
            low=0, high=n_waypoints, size=(n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
        )
        
        # Initialize outputs
        pos = torch.zeros(n_reset, 3, device=self.device)
        vel = torch.zeros(n_reset, 3, device=self.device)
        quat = torch.zeros(n_reset, 4, device=self.device)
        waypoint_indices = start_indices.clone()
        
        # Perturbation bounds for curriculum
        pos_noise_std = 0.3  # meters
        vel_noise_std = 0.3  # m/s
        ori_noise_std = 0.15  # radians (yaw only)
        
        # For each env, sample from its selected gate's buffer (if available)
        # 50% probability to use geometric sampling even if buffer has data (to avoid local minima)
        use_geometric_prob = 0.5
        for i in range(n_reset):
            gate_id = start_indices[i].item()
            buffer_count = self._gate_buffer_counts[gate_id].item()
            
            # Decide whether to use buffer or geometric sampling
            use_buffer = (buffer_count > 0) and (torch.rand(1, device=self.device).item() > use_geometric_prob)
            
            if use_buffer:
                # Sample a random historical state from this gate's buffer
                sample_idx = torch.randint(0, buffer_count, (1,), device=self.device).item()
                sampled_state = self._gate_state_buffers[gate_id, sample_idx]
                
                # Extract state components
                base_pos = sampled_state[:3]
                base_vel = sampled_state[3:6]
                base_quat = sampled_state[6:]
                
                # Add bounded Gaussian noise
                pos[i] = base_pos + torch.randn(3, device=self.device) * pos_noise_std
                vel[i] = base_vel + torch.randn(3, device=self.device) * vel_noise_std
                
                # Perturb orientation (yaw only)
                base_yaw = torch.atan2(
                    2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
                    1.0 - 2.0 * (base_quat[2]**2 + base_quat[3]**2)
                )
                perturbed_yaw = base_yaw + torch.randn(1, device=self.device).item() * ori_noise_std
                quat[i] = quat_from_euler_xyz(
                    torch.zeros(1, device=self.device),
                    torch.zeros(1, device=self.device),
                    torch.tensor([perturbed_yaw], device=self.device)
                )[0]
            else:
                # Use geometric sampling (either buffer is empty or 10% random chance)
                gate_center = self.env._waypoints[gate_id, :3]
                gate_normal = self.env._normal_vectors[gate_id]
                gate_quat_mat = matrix_from_quat(self.env._waypoints_quat[gate_id].unsqueeze(0))[0]
                gate_y = gate_quat_mat[:, 1]
                gate_z = gate_quat_mat[:, 2]
                
                gate_half = getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5
                lat_half = gate_half * self.spawn_lateral_expand
                
                # Sample in front of gate
                d = torch.empty(1, device=self.device).uniform_(
                    self.spawn_before_min, self.spawn_before_max
                )
                off_y = torch.empty(1, device=self.device).uniform_(-lat_half, lat_half)
                off_z = torch.empty(1, device=self.device).uniform_(-lat_half, lat_half)
                
                pos[i] = gate_center + d * gate_normal + off_y * gate_y + off_z * gate_z
                pos[i, 2] = torch.empty(1, device=self.device).uniform_(
                    self.spawn_z_min, self.spawn_z_max
                ).item()
                
                vel[i] = 0.0  # Start from rest
                yaw = torch.atan2(gate_normal[1], gate_normal[0])
                quat[i] = quat_from_euler_xyz(
                    torch.zeros(1, device=self.device),
                    torch.zeros(1, device=self.device),
                    torch.tensor([yaw], device=self.device)
                )[0]

        # Write pose and velocity to default root state
        default_root_state[:, 0:3] = pos
        default_root_state[:, 3:7] = quat
        default_root_state[:, 7:10] = vel  # Linear velocity
        default_root_state[:, 10:13] = 0.0  # Angular velocity (start with zero)
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids_tensor] = waypoint_indices

        self.env._desired_pos_w[env_ids_tensor, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids_tensor, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids_tensor] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids_tensor, :2] - self.env._robot.data.root_link_pos_w[env_ids_tensor, :2], dim=1
        )
        self.env._n_gates_passed[env_ids_tensor] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids_tensor)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)

        # Reset variables
        self.env._yaw_n_laps[env_ids_tensor] = 0

        self.env._pose_drone_wrt_gate[env_ids_tensor], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids_tensor], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids_tensor], :],
            self.env._robot.data.root_link_state_w[env_ids_tensor, :3]
        )

        # Initialize previous pose and distance for velocity/progress calculation
        self._prev_pose_drone_wrt_gate[env_ids_tensor] = self.env._pose_drone_wrt_gate[env_ids_tensor].clone()
        # Only update the resetting environments, not all environments
        self.env._prev_x_drone_wrt_gate[env_ids_tensor] = self.env._pose_drone_wrt_gate[env_ids_tensor, 0]
        # Initialize previous distance to gate center
        drone_pos = self.env._robot.data.root_link_pos_w[env_ids_tensor]
        gate_pos = self.env._waypoints[self.env._idx_wp[env_ids_tensor], :3]
        self._prev_distance_to_gate[env_ids_tensor] = torch.norm(drone_pos - gate_pos, dim=1)

        self.env._crashed[env_ids_tensor] = 0
        
