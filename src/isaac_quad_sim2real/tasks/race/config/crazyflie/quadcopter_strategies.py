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


class ResetBuffer:
    """
    Circular replay buffer that stores drone states the moment of gate crossing. Once drone learns to pass
    gate 3, we can restart from here to get more training for harder segments
    """

    # 17 dimensional state:    
    STATE_DIM = 17  # pos(3) + lin_vel(3) + quat(4) + ang_vel(3) + motor_speeds(4)
    
    def __init__(
        self,
        num_gates: int,
        buffer_size: int,
        device: str,
        noise_std: Optional[Dict[str, float]] = None
    ):
        self.num_gates = num_gates  # Number of gates in the track
        self.buffer_size = buffer_size # Maximum number of states to store per gate
        self.device = device
        
        # Default noise configuration
        default_noise = {
            'pos': 0.1,                 # Position noise (m)
            'lin_vel': 0.05,            # Linear velocity noise 
            'quat': 0.15,               # Orientation noise as yaw (rad)
            'ang_vel': 0.1,             # Angular velocity noise
            'motor_speeds': 100.0       # Motor speed noise
        }
        self.noise_std = {**default_noise, **(noise_std or {})}
        
        # State buffer: [num_gates, buffer_size, state_dim]
        self._buffer = torch.zeros(
            num_gates, buffer_size, self.STATE_DIM, device=device
        )
        # Track counts and indices for each gate
        self._counts = torch.zeros(num_gates, dtype=torch.long, device=device)
        self._indices = torch.zeros(num_gates, dtype=torch.long, device=device)
    
    
    def store(
        self,
        gate_id: int,
        pos: torch.Tensor,
        lin_vel: torch.Tensor,
        quat: torch.Tensor,
        ang_vel: torch.Tensor,
        motor_speeds: torch.Tensor
    ):
        """ Store a state in the buffer for a specific gate."""
        write_idx = self._indices[gate_id].item()
        
        self._buffer[gate_id, write_idx, 0:3] = pos
        self._buffer[gate_id, write_idx, 3:6] = lin_vel
        self._buffer[gate_id, write_idx, 6:10] = quat
        self._buffer[gate_id, write_idx, 10:13] = ang_vel
        self._buffer[gate_id, write_idx, 13:17] = motor_speeds
        
        # Update circular buffer index
        self._indices[gate_id] = (write_idx + 1) % self.buffer_size
        # Update count (saturate at buffer size)
        self._counts[gate_id] = min(self._counts[gate_id] + 1, self.buffer_size)
    
    def get_count(self, gate_id: int) -> int:
        """Get the number of stored states for a gate."""
        return self._counts[gate_id].item()
    
    # Sample a random state from the buffer with optional noise.
    def sample(
        self,
        gate_id: int,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        buffer_count = self._counts[gate_id].item()
        if buffer_count == 0:
            raise ValueError(f"No states stored for gate {gate_id}")
        
        # Random sample index
        sample_idx = torch.randint(0, buffer_count, (1,), device=self.device).item()
        state = self._buffer[gate_id, sample_idx]
        
        # Extract components
        pos = state[0:3].clone()
        lin_vel = state[3:6].clone()
        quat = state[6:10].clone()
        ang_vel = state[10:13].clone()
        motor_speeds = state[13:17].clone()
        
        if add_noise:
            # Add position noise
            pos += torch.randn(3, device=self.device) * self.noise_std['pos']
            
            # Add linear velocity noise
            lin_vel += torch.randn(3, device=self.device) * self.noise_std['lin_vel']
            
            # Add angular velocity noise
            ang_vel += torch.randn(3, device=self.device) * self.noise_std['ang_vel']
            
            # Add motor speed noise
            motor_speeds += torch.randn(4, device=self.device) * self.noise_std['motor_speeds']
            
            # Perturb orientation (yaw only) - extract yaw, add noise, reconstruct quat
            base_yaw = torch.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2]**2 + quat[3]**2)
            )
            perturbed_yaw = base_yaw + torch.randn(1, device=self.device).item() * self.noise_std['quat']
            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                torch.tensor([perturbed_yaw], device=self.device)
            )[0]
        
        return pos, lin_vel, quat, ang_vel, motor_speeds
    
    def has_samples(self, gate_id: int) -> bool:
        """Check if a gate has any stored samples."""
        return self._counts[gate_id].item() > 0


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

            extra_keys = [
                "backward_crossing",
                "speed_at_crossing",
                "speed_threshold",
                "line_deviation",
                "three_lap_time_bonus",
            ]

            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys + extra_keys
            }

        # Ranges for evaluation with given min / max to mimic the sim2real gap
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
        
        # Track previous position wrt previous gate for backward crossing calculation
        self._prev_pose_drone_wrt_prev_gate = torch.zeros(self.num_envs, 3, device=self.device)


        # Initialize reset buffer for curriculum learning
        # Stores 17-dim states: [pos(3), lin_vel(3), quat(4), ang_vel(3), motor_speeds(4)]
        num_gates = len(env._waypoints) if hasattr(env, '_waypoints') else 6
        noise_std = {
            'pos': 0.3,           # Position noise (m)
            'lin_vel': 0.3,       # Linear velocity noise (m/s)
            'quat': 0.15,         # Orientation noise as yaw (rad)
            'ang_vel': 0.1,       # Angular velocity noise (rad/s)
            'motor_speeds': 50.0  # Motor speed noise (rad/s)
        }
        self._reset_buffer = ResetBuffer(
            num_gates=num_gates,
            buffer_size=100,
            device=self.device,
            noise_std=noise_std
        )

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
        
        # Motor time constants: use fixed base value (no randomization)
        self.env._tau_m[env_ids] = self.env._tau_m_value

    def _detect_gate_crossing(
        self,
        current_gate_position: torch.Tensor,
        prev_gate_position: torch.Tensor,
        expanded_half: float,
        gate_half: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gate local frame: 
        X: normal axis (approach / exit direction)
        Y: horizontal axis (left / right)
        Z: vertical axis (up / down)
        
        To detect a gate crossing, we need it to 
        1. Go through the correct X direction
        2. Make drone pass thorugh the center, ideally middle of Y and middle of Z
        """

        # drone's position expressed in gate's local frame at 2 timesteps
        prev_drone_position = prev_gate_position[:, 0]
        current_drone_position_X = current_gate_position[:, 0]
        current_drone_position_Y = current_gate_position[:, 1]
        current_drone_position_Z = current_gate_position[:, 2]

        # cross the gate going forward (+x -> x <= 0)
        crossed_gate_correct_direction = (prev_drone_position > 0.0) & (current_drone_position_X <= 0.0)
        
        # check if drone fit through gate
        crossed_inside_gate = (
            (torch.abs(current_drone_position_Y) <= gate_half) &
            (torch.abs(current_drone_position_Z) <= gate_half)
        )
        
        # 3) Allow an expanded detection window for misses, if provided
        crossed_inside_expanded_gate = (
            (torch.abs(current_drone_position_Y) <= expanded_half) &
            (torch.abs(current_drone_position_Z) <= expanded_half)
        )

        # 3 possible ways for gate crossing: passed correctly, missed gate, passed backwards
        gate_passed = crossed_gate_correct_direction & crossed_inside_gate
        gate_missed = crossed_gate_correct_direction & crossed_inside_expanded_gate & (~crossed_inside_gate)
        crossed_gate_backward = ((prev_drone_position < 0.0) & (current_drone_position_X >= 0.0) & crossed_inside_expanded_gate)

        return gate_passed, gate_missed, crossed_gate_backward
    
    def _calculate_progress_reward(
        self,
        crossed_gate_correct_direction: torch.Tensor,
        gate_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Dense reward signal at every timestep based on how much closer drone got to current gate compared to last"""
        
        # stored distrance from last timestep
        prev_distance = self._prev_distance_to_gate.clone()

        current_pos = self.env._robot.data.root_link_pos_w

        successful_crossed = torch.where(crossed_gate_correct_direction)[0]     # id's for success / fail gate crossings
        not_successful_crossed = torch.where(~crossed_gate_correct_direction)[0]
        
        current_distance = torch.zeros(self.num_envs, device=self.device)
        
        # for successful cross: progress = distance from prev_pos to gate plane
        if successful_crossed.numel() > 0:
            # X coords of pose that succeeded
            prev_distance[successful_crossed] = self._prev_pose_drone_wrt_gate[successful_crossed][:, 0]
            current_distance[successful_crossed] = 0.0

        # for failed cross: progress = euclidean distance to current gate center
        if not_successful_crossed.numel() > 0:
            current_gate_pos = gate_centers[not_successful_crossed]
            current_distance[not_successful_crossed] = torch.norm(current_pos[not_successful_crossed] - current_gate_pos, dim=1)
        
        # positive progress is made when curr < prev
        progress = prev_distance - current_distance
        return progress, current_distance


    def _calculate_next_gate_progress(
        self,
        crossed_gate_correct_direction: torch.Tensor,
        current_gate_centers: torch.Tensor,
        next_gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reward flying towards next gate while crossing current gate. Fires on timestep of crossing
        """

        next_progress = torch.zeros(self.num_envs, device=self.device)
        # if we didn't succeed, there's nothing to calc for
        successful_crossed = torch.where(crossed_gate_correct_direction)[0]
        if successful_crossed.numel() == 0:
            return next_progress

        # get current state
        drone_pos = self.env._robot.data.root_link_pos_w[successful_crossed]
        prev_gate_centers = current_gate_centers[successful_crossed]

        # get next gate
        next_gate_pos = self.env._waypoints[next_gate_indices[successful_crossed], :3]
        next_gate_quat = self.env._waypoints_quat[next_gate_indices[successful_crossed], :]

        # subtract to find difference between
        prev_gate_in_next_frame, _ = subtract_frame_transforms(next_gate_pos, next_gate_quat, prev_gate_centers)
        drone_pose_next_gate, _ = subtract_frame_transforms(next_gate_pos, next_gate_quat, drone_pos)

        prev_distance = torch.norm(prev_gate_in_next_frame, dim=1)
        current_distance = torch.norm(drone_pose_next_gate, dim=1)

        # calc distance between to reward crossing gate -> next gate 
        next_progress[successful_crossed] = prev_distance - current_distance
        return next_progress

    def _handle_gate_crossing(self, crossed_gate_correct_direction: torch.Tensor) -> None:
        """
        Stores state right before gate crossing to replay buffer
        Increment # of gates crossed
        Reinit for next gate
        """
        ids_to_next_gate = torch.where(crossed_gate_correct_direction)[0]
        
        # if success
        if ids_to_next_gate.numel() > 0:
            current_gate_idx = self.env._idx_wp[ids_to_next_gate]
            drone_pos = self.env._robot.data.root_link_pos_w[ids_to_next_gate]
            drone_vel = self.env._robot.data.root_com_lin_vel_w[ids_to_next_gate]
            drone_quat = self.env._robot.data.root_quat_w[ids_to_next_gate]
            drone_ang_vel = self.env._robot.data.root_com_ang_vel_w[ids_to_next_gate]
            drone_motor_speeds = self.env._motor_speeds[ids_to_next_gate]
            
            # Store states to reset buffer per gate crossing
            for i, env_idx in enumerate(ids_to_next_gate):
                gate_id = current_gate_idx[i].item()
                self._reset_buffer.store(
                    gate_id=gate_id,
                    pos=drone_pos[i],
                    lin_vel=drone_vel[i],
                    quat=drone_quat[i],
                    ang_vel=drone_ang_vel[i],
                    motor_speeds=drone_motor_speeds[i]
                )
            
            # Advance waypoint index, tick target gate forward by 1
            self.env._idx_wp[ids_to_next_gate] = (self.env._idx_wp[ids_to_next_gate] + 1) % self.env._waypoints.shape[0]    # wraps end back to 0
            self.env._n_gates_passed[ids_to_next_gate] += 1 
            
            # re-init all tracking vars for new gate
            new_gate_pos = self.env._waypoints[self.env._idx_wp[ids_to_next_gate], :3]
            new_gate_quat = self.env._waypoints_quat[self.env._idx_wp[ids_to_next_gate], :]

            self._prev_distance_to_gate[ids_to_next_gate] = torch.norm(drone_pos - new_gate_pos, dim=1)
            new_pose_drone_wrt_gate, _ = subtract_frame_transforms(new_gate_pos, new_gate_quat, drone_pos)

            self._prev_pose_drone_wrt_gate[ids_to_next_gate] = new_pose_drone_wrt_gate
            self.env._prev_x_drone_wrt_gate[ids_to_next_gate] = new_pose_drone_wrt_gate[:, 0]


    def _update_tracking_variables(
        self,
        gate_passed: torch.Tensor,
        current_distance: torch.Tensor,
    ) -> None:
        """Update previous pose and distance for next timestep."""
        # Update envs that did not successfully pass the gate
        ids_not_passed = torch.where(~gate_passed)[0]
        if ids_not_passed.numel() > 0:
            self._prev_pose_drone_wrt_gate[ids_not_passed] = (self.env._pose_drone_wrt_gate[ids_not_passed].clone())
            self.env._prev_x_drone_wrt_gate[ids_not_passed] = (self.env._pose_drone_wrt_gate[ids_not_passed, 0])
            self._prev_distance_to_gate[ids_not_passed] = current_distance[ids_not_passed].clone()

    def _reward_blend(self, full_val: float, start_iter: int, ramp_iters: int = 150) -> float:
        """Linearly ramp a reward from 0 to full_val over ramp_iters after start_iter."""
        t = (self.env.iteration - start_iter) / ramp_iters
        return full_val * max(0.0, min(1.0, t))
    
    def get_rewards(self) -> torch.Tensor:

        # Gate setup
        pose_gate = self.env._pose_drone_wrt_gate
        gate_half = getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5 \
            if hasattr(self.env, "_gate_model_cfg_data") else 0.5
        expand = getattr(self.cfg, "gate_detection_expand", 1.4)
        expanded_half = gate_half * expand

        gate_centers = self.env._waypoints[self.env._idx_wp, :3]
        self.env._desired_pos_w[:] = gate_centers
        num_gates = self.env._waypoints.shape[0]
        next_gate_indices = (self.env._idx_wp + 1) % num_gates

        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # Gate crossing detection
        gate_passed, gate_missed, gate_crossed_backward = self._detect_gate_crossing(
            current_gate_position=pose_gate,
            prev_gate_position=self._prev_pose_drone_wrt_gate,
            expanded_half=expanded_half,
            gate_half=gate_half,
        )
        crossed_gate_correct_direction = gate_passed | gate_missed

        # Progress rewards
        progress_current, current_distance = self._calculate_progress_reward(
            crossed_gate_correct_direction, gate_centers
        )
        progress_next_gate = self._calculate_next_gate_progress(
            gate_passed, gate_centers, next_gate_indices
        )
        progress = progress_current + progress_next_gate

        # Advance waypoint for passed gates
        self._handle_gate_crossing(gate_passed)

        # Gate reward: +1 clean pass, -1 miss
        gate_reward = gate_passed.float() * 1.0 + gate_missed.float() * (-1.0)

        # Update tracking for next timestep
        self._update_tracking_variables(gate_passed, current_distance)

        # Backward crossing detection
        prev_gate_indices = (self.env._idx_wp - 1) % num_gates
        prev_gate_pos  = self.env._waypoints[prev_gate_indices, :3]
        prev_gate_quat = self.env._waypoints_quat[prev_gate_indices, :]
        current_pose_prev_gate, _ = subtract_frame_transforms(
            prev_gate_pos, prev_gate_quat,
            self.env._robot.data.root_link_pos_w
        )
        prev_gate_crossed_backward = (
            (self._prev_pose_drone_wrt_prev_gate[:, 0] < 0.0) &
            (current_pose_prev_gate[:, 0] >= 0.0) &
            (torch.abs(current_pose_prev_gate[:, 1]) <= expanded_half) &
            (torch.abs(current_pose_prev_gate[:, 2]) <= expanded_half)
        )
        self._prev_pose_drone_wrt_prev_gate = current_pose_prev_gate.clone()
        if self.cfg.is_train:
            itr = self.env.iteration

            # ── Curriculum scale helpers ─────────────────────────────────────────
            def ramp(start_itr, end_itr, v_min=0.0, v_max=1.0):
                """Linearly ramp from v_min to v_max between start_itr and end_itr."""
                t = (itr - start_itr) / max(end_itr - start_itr, 1)
                return v_min + (v_max - v_min) * max(0.0, min(1.0, t))

            # Each reward component gets a scalar multiplier that starts at 0
            # and ramps to 1.0 over its assigned window.
            # Phase 1 (0–400):   only progress + velocity — drone learns to fly & approach
            # Phase 2 (400–600): gate crossing bonus fades in
            # Phase 3 (500–700): death cost fades in (softly)
            # Phase 4 (600–800): speed, line deviation, backward penalty fade in
            # Phase 5 (800+):    lap penalty and time bonus active

            scale_progress    = 1.0
            scale_velocity    = 1.0
            scale_gate        = ramp(0,   100)    # was 400→550, active immediately
            scale_death       = ramp(50,  200)    # was 450→650
            scale_speed_cross = ramp(100, 250)    # was 500→650
            scale_speed_thr   = ramp(150, 300)    # was 550→700
            scale_line        = ramp(200, 350)    # was 600→750
            scale_backward    = ramp(200, 350)    # was 600→750
            scale_lap_penalty = ramp(300, 450)    # was 700→850
            scale_time_bonus  = ramp(400, 550)    # was 800→950

            # ── Core rewards ─────────────────────────────────────────────────────
            progress_reward    = progress * self.env.rew['progress_goal_reward_scale'] * scale_progress
            gate_reward_scaled = gate_reward * self.env.rew['target_reached_bonus_reward_scale'] * scale_gate

            laps_completed = self.env._n_gates_passed // num_gates
            lap_penalty = (
                self.env.rew['lap_incomplete_penalty_reward_scale']
                * (laps_completed < 3).float()
                * scale_lap_penalty
            )

            wrong_direction  = (gate_crossed_backward | prev_gate_crossed_backward).float()
            backward_penalty = wrong_direction * self.env.rew.get('wrong_direction_penalty', -200.0) * scale_backward

            # ── Velocity toward gate ─────────────────────────────────────────────
            vel_w             = self.env._robot.data.root_com_lin_vel_w
            drone_pos         = self.env._robot.data.root_link_pos_w
            next_gate_centers = self.env._waypoints[next_gate_indices, :3]

            gate_dir       = gate_centers - drone_pos
            next_gate_dir  = next_gate_centers - drone_pos
            gate_dist      = torch.norm(gate_dir,     dim=1, keepdim=True).clamp(min=0.1)
            next_gate_dist = torch.norm(next_gate_dir, dim=1, keepdim=True).clamp(min=0.1)
            gate_dir_unit      = gate_dir      / gate_dist
            next_gate_dir_unit = next_gate_dir / next_gate_dist

            transition_radius  = 2.0
            next_gate_weight   = (1.0 - (gate_dist.squeeze(1) / transition_radius).clamp(0.0, 1.0)) * 0.3
            curr_gate_weight   = 1.0 - next_gate_weight
            blended_dir = (curr_gate_weight.unsqueeze(1) * gate_dir_unit +
                        next_gate_weight.unsqueeze(1) * next_gate_dir_unit)
            blended_dir = blended_dir / torch.norm(blended_dir, dim=1, keepdim=True).clamp(min=0.1)

            vel_toward_gate = (vel_w * blended_dir).sum(dim=1).clamp(min=0.0)
            velocity_reward = vel_toward_gate * self.env.rew['velocity_toward_gate_reward_scale'] * scale_velocity

            # ── Speed at crossing ────────────────────────────────────────────────
            crossing_speed          = torch.norm(vel_w, dim=1)
            speed_at_crossing_bonus = (
                gate_passed.float() * crossing_speed
                * self.env.rew['speed_at_crossing_reward_scale']
                * scale_speed_cross
            )

            # ── Curriculum speed threshold ───────────────────────────────────────
            base_threshold      = 3.0
            threshold_increment = 0.5
            stage               = min(itr // 200, 6)
            current_threshold   = base_threshold + stage * threshold_increment
            speed               = torch.norm(vel_w, dim=1)
            speed_above         = (speed - current_threshold).clamp(min=0.0)
            threshold_reward    = speed_above * self.env.rew.get('speed_threshold_scale', 15.0) * scale_speed_thr

            # ── Line deviation penalty ───────────────────────────────────────────
            gate_to_gate      = next_gate_centers - gate_centers
            gate_to_gate_dist = torch.norm(gate_to_gate, dim=1, keepdim=True).clamp(min=0.1)
            gate_to_gate_unit = gate_to_gate / gate_to_gate_dist
            drone_offset      = drone_pos - gate_centers
            along             = (drone_offset * gate_to_gate_unit).sum(dim=1, keepdim=True)
            lateral           = drone_offset - along * gate_to_gate_unit
            lateral_dist      = torch.norm(lateral, dim=1)
            between_gates     = (along.squeeze(1) > 0.0) & \
                                (along.squeeze(1) < gate_to_gate_dist.squeeze(1))
            line_penalty = (
                -lateral_dist * between_gates.float()
                * self.env.rew.get('line_deviation_penalty_scale', 1.0)
                * scale_line
            )

            # ── Three lap time bonus ─────────────────────────────────────────────
            prev_laps            = (self.env._n_gates_passed - gate_passed.int()) // num_gates
            just_finished_3_laps = (laps_completed >= 3) & (prev_laps < 3)
            steps_taken          = self.env.episode_length_buf.float()
            step_budget          = 1800.0
            time_bonus = (
                self.env.rew.get('three_lap_time_bonus', 3000.0)
                * (0.1 + 0.9 * (1.0 - (steps_taken / step_budget).clamp(0.0, 1.0)))
                * just_finished_3_laps.float()
                * scale_time_bonus
            )

            # ── Aggregate ────────────────────────────────────────────────────────
            rewards = {
                "progress_goal":          progress_reward,
                "target_reached_bonus":   gate_reward_scaled,
                "lap_incomplete_penalty": lap_penalty,
                "backward_crossing":      backward_penalty,
                "velocity_toward_gate":   velocity_reward,
                "speed_at_crossing":      speed_at_crossing_bonus,
                "speed_threshold":        threshold_reward,
                "line_deviation":         line_penalty,
                "three_lap_time_bonus":   time_bonus,
            }

            reward = torch.sum(torch.stack(list(rewards.values())), dim=0) - time_bonus
            reward = reward.clamp(-300.0, 300.0)   # ADD THIS

            
            for key, value in rewards.items():
                self._episode_sums[key] += value

            # Death cost also ramps — never abrupt
            death_cost = self.env.rew['death_cost'] * scale_death
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * death_cost,
                reward
            )
            reward = reward + time_bonus

        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            
        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including waypoint positions and drone state."""
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = torch.bmm(self.env._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self.env._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3)
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3)
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self.env._robot.data.root_com_lin_vel_b,			# 3 dim (linear vel in body frame)
                attitude_mat.view(attitude_mat.shape[0], -1),			# 9 dim (drone rotation matrix)
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),	# 12 dim (corners of current gate)
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),	# 12 dim (corners of next gate)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # Update yaw tracking
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps

        self.env._previous_actions = self.env._actions.clone()

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None:
            env_ids_tensor = self.env._robot._ALL_INDICES
        else:
            env_ids_tensor = env_ids
            if env_ids_tensor.shape[0] == self.num_envs:
                env_ids_tensor = self.env._robot._ALL_INDICES

        self._prev_pose_drone_wrt_prev_gate[env_ids_tensor] = torch.zeros(
            len(env_ids_tensor), 3, device=self.device
        )
        # Randomize dynamics parameters for the resetting environments
        self._randomize_dynamics(env_ids_tensor)

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            ep_len = max(self.env.episode_length_buf[env_ids_tensor].float().mean().item(), 1)
            gates  = self.env._n_gates_passed[env_ids_tensor].float().mean().item()

            # DEBUG — read BEFORE zeroing
            component_strs = []
            total = 0.0
            for k, v in self._episode_sums.items():
                mean_val = v[env_ids_tensor].mean().item()
                total += mean_val
                component_strs.append(f"{k}={mean_val:.0f}")

            # ── Add curriculum scales to wandb log ───────────────────────────
            itr = self.env.iteration
            def ramp(start_itr, end_itr, v_min=0.0, v_max=1.0):
                t = (itr - start_itr) / max(end_itr - start_itr, 1)
                return v_min + (v_max - v_min) * max(0.0, min(1.0, t))

            self.env.extras["log"].update({
                # Curriculum scale visibility
                "Curriculum/scale_gate":        ramp(400, 550),
                "Curriculum/scale_death":       ramp(450, 650),
                "Curriculum/scale_speed_cross": ramp(500, 650),
                "Curriculum/scale_speed_thr":   ramp(550, 700),
                "Curriculum/scale_line":        ramp(600, 750),
                "Curriculum/scale_backward":    ramp(600, 750),
                "Curriculum/scale_lap_penalty": ramp(700, 850),
                "Curriculum/scale_time_bonus":  ramp(800, 950),

                # Episode stats
                "Episode_Stats/gates_per_episode": self.env._n_gates_passed[env_ids_tensor].float().mean().item(),
                "Episode_Stats/ep_len":            self.env.episode_length_buf[env_ids_tensor].float().mean().item(),
                "Episode_Stats/reward_per_step":   (
                    sum(v[env_ids_tensor].mean().item() for v in self._episode_sums.values())
                    / max(self.env.episode_length_buf[env_ids_tensor].float().mean().item(), 1)
                ),
            })

            print(f"[EPISODE_DEBUG] ep_len={ep_len:.0f} gates={gates:.1f} "
                  f"total={total:.0f} per_step={total/ep_len:.2f}")
            print(f"[EPISODE_DEBUG] breakdown: {' | '.join(component_strs)}")

            # Tensorboard logging + zero-out (must happen after debug print)
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids_tensor] = 0.0   # zero AFTER reading
            
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

        # For each env, randomly select a gate and sample a historical state from that gate's buffer
        n_waypoints = self.env._waypoints.shape[0]
        
        # Randomly select gates for each resetting env
        start_indices = torch.randint(low=0, high=n_waypoints, size=(n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
        
        # Initialize outputs
        pos = torch.zeros(n_reset, 3, device=self.device)
        vel = torch.zeros(n_reset, 3, device=self.device)
        quat = torch.zeros(n_reset, 4, device=self.device)
        ang_vel = torch.zeros(n_reset, 3, device=self.device)
        motor_speeds = torch.zeros(n_reset, 4, device=self.device)
        # Force close spawn to gate 0 for first 400 iterations
        if self.env.iteration < 400:
            start_indices[:] = self.env._initial_wp
            spawn_before_min_eff     = 0.3
            spawn_before_max_eff     = 1.5
            spawn_lateral_expand_eff = 0.3
        else:
            spawn_before_min_eff     = self.spawn_before_min
            spawn_before_max_eff     = self.spawn_before_max
            spawn_lateral_expand_eff = self.spawn_lateral_expand

        waypoint_indices = start_indices.clone()   # <-- MOVED to after the override
            
        # After iteration 1000, 10% chance to spawn on ground in front of initial waypoint for recovery
        if self.env.iteration > 1000:
            percent_ground = 0.1  # 10%
            ground_mask = torch.rand(n_reset, device=self.device) < percent_ground
            ground_local_ids = torch.nonzero(ground_mask, as_tuple=False).squeeze(-1)
            
            if ground_local_ids.numel() > 0:
                n_ground = ground_local_ids.shape[0]
                
                # Sample local coordinates for ground spawn
                x_local = torch.empty(n_ground, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(n_ground, device=self.device).uniform_(-1.0, 1.0)
                
                # Get initial waypoint info
                x0_wp = self.env._waypoints[self.env._initial_wp, 0]
                y0_wp = self.env._waypoints[self.env._initial_wp, 1]
                theta = self.env._waypoints[self.env._initial_wp, -1]
                
                # Rotate local pos to global frame
                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local
                x0 = x0_wp - x_rot
                y0 = y0_wp - y_rot
                z0 = 0.05  # Ground level
                
                pos[ground_local_ids, 0] = x0
                pos[ground_local_ids, 1] = y0
                pos[ground_local_ids, 2] = z0
                
                vel[ground_local_ids] = 0.0  # Start from rest on ground
                ang_vel[ground_local_ids] = 0.0  # No angular velocity
                motor_speeds[ground_local_ids] = 0.0  # Motors off
                
                # Point drone towards target with yaw perturbation (-0.15 to 0.15 rad)
                yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)
                yaw_noise = torch.empty(n_ground, device=self.device).uniform_(-0.15, 0.15)
                yaw0 = yaw0 + yaw_noise
                
                quat[ground_local_ids] = quat_from_euler_xyz(
                    torch.zeros(n_ground, device=self.device),
                    torch.zeros(n_ground, device=self.device),
                    yaw0
                )
                
                # Set waypoint indices to initial waypoint for ground spawn envs
                waypoint_indices[ground_local_ids] = self.env._initial_wp
        else:
            ground_mask = torch.zeros(n_reset, dtype=torch.bool, device=self.device)
        
        # For each env (excluding ground spawn), sample from its selected gate's buffer (if available)
        # Reset proportions: 10% ground, 50% buffer, 40% geometric
        # For non-ground envs (90%): buffer prob = 50/90 ≈ 0.556, geometric prob = 40/90 ≈ 0.444
        use_buffer_prob = 6.0 / 9.0  # 50/(50+40) ≈ 0.556
        for i in range(n_reset):
            # Skip if this env is using ground spawn
            if ground_mask[i]:
                continue
            
            gate_id = start_indices[i].item()
            has_samples = self._reset_buffer.has_samples(gate_id)
            
            # Decide whether to use buffer or geometric sampling
            use_buffer = has_samples and (torch.rand(1, device=self.device).item() < use_buffer_prob)
            
            if use_buffer:
                # Sample from reset buffer with noise
                sampled_pos, sampled_vel, sampled_quat, sampled_ang_vel, sampled_motor_speeds = \
                    self._reset_buffer.sample(gate_id, add_noise=True)
                
                pos[i] = sampled_pos
                vel[i] = sampled_vel
                quat[i] = sampled_quat
                ang_vel[i] = sampled_ang_vel
                motor_speeds[i] = sampled_motor_speeds
            else:
                # Use geometric sampling (either buffer is empty or random chance)
                gate_center = self.env._waypoints[gate_id, :3]
                gate_normal = self.env._normal_vectors[gate_id]
                gate_quat_mat = matrix_from_quat(self.env._waypoints_quat[gate_id].unsqueeze(0))[0]
                gate_y = gate_quat_mat[:, 1]
                gate_z = gate_quat_mat[:, 2]

                gate_half = getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5
                lat_half = gate_half * spawn_lateral_expand_eff    # <-- FIXED

                d = torch.empty(1, device=self.device).uniform_(
                    spawn_before_min_eff, spawn_before_max_eff     # <-- FIXED
                )
                off_y = torch.empty(1, device=self.device).uniform_(-lat_half, lat_half)
                off_z = torch.empty(1, device=self.device).uniform_(-lat_half, lat_half)
                
                pos[i] = gate_center + d * gate_normal + off_y * gate_y + off_z * gate_z
                pos[i, 2] = torch.empty(1, device=self.device).uniform_(
                    self.spawn_z_min, self.spawn_z_max
                ).item()
                
                vel[i] = 0.0  # Start from rest
                ang_vel[i] = 0.0  # No angular velocity
                motor_speeds[i] = 0.0  # Motors off
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
        default_root_state[:, 10:13] = ang_vel  # Angular velocity from buffer or zero
        
        # Write motor speeds to env (will be applied after reset)
        self.env._motor_speeds[env_ids_tensor] = motor_speeds

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
        