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

            extra_keys = ["backward_crossing",  "approach_angle"]  # adds extra backwards_crossing key 

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


        # ------------------------------------------------------------------ #
        # INSERT VIRTUAL WAYPOINT at index 3 (between old gate 3 and gate 4) #
        # ------------------------------------------------------------------ #
        self._virtual_gate_index = 3
        self._virtual_gate_half  = 0.6   # small gate — just big enough to pass through
        
        # Position: north of and above the two touching gates
        # Gate 4 (old idx 3) is at [0.625, 0.0, 0.75] — place virtual gate
        # 1.8m north and 1.8m above so the center is reachable mid-arc
        virtual_wp = torch.tensor(
            [[0.625, 1.8, 2.2,  0.0, 0.0, 1.5708]],  # yaw=π/2 → normal = +Y (north-facing)
            device=self.device
        )

        # Quaternion for yaw=π/2: [cos(π/4), 0, 0, sin(π/4)]
        virtual_quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.tensor([np.pi / 2], device=self.device)
        )  # shape [1, 4]

        # Normal vector for yaw=π/2 is +Y
        virtual_normal = torch.tensor([[0.0, 1.0, 0.0]], device=self.device)

        # Patch all waypoint arrays by inserting at index 3
        env._waypoints = torch.cat([
            env._waypoints[:3],
            virtual_wp,
            env._waypoints[3:]
        ], dim=0)

        env._waypoints_quat = torch.cat([
            env._waypoints_quat[:3],
            virtual_quat,
            env._waypoints_quat[3:]
        ], dim=0)

        env._normal_vectors = torch.cat([
            env._normal_vectors[:3],
            virtual_normal,
            env._normal_vectors[3:]
        ], dim=0)

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
        # Seed the VIRTUAL gate (index 3), not the loop gate
        VIRTUAL_GATE_ID = 3

        # These states represent the drone mid-arc: 
        # elevated, north of gate stack, heading south
        up_pos = torch.tensor(
            [0.625, 2.8, 2.4],           # north of virtual gate, approaching from behind
            dtype=torch.float32, device=self.device
        )
        up_vel = torch.tensor(
            [0.0, -3.5, -0.3],           # heading south toward virtual gate
            dtype=torch.float32, device=self.device
        )
        self._seed_loop_buffer(VIRTUAL_GATE_ID, up_pos, up_vel)

        via_pos = torch.tensor(
            [0.625, 3.5, 2.8],           # further north, higher up — earlier in arc
            dtype=torch.float32, device=self.device
        )
        approach_vel = torch.tensor(
            [0.0, -2.0, -1.0],           # heading south and descending
            dtype=torch.float32, device=self.device
        )
        self._seed_loop_buffer(VIRTUAL_GATE_ID, via_pos, approach_vel)

        LOOP_GATE_ID = 4  # gate 4 counting from 1, at [0.625, 0.0, 0.75]

        # Phase 1: drone has swung north, elevated, beginning descent toward gate 4
        up_pos = torch.tensor(
            [0.4, 2.5, 2.6],    # north of gate 4 (world Y > 0 = gate_frame_X > 0 ✓)
            dtype=torch.float32, device=self.device
        )
        up_vel = torch.tensor(
            [0.1, -3.0, -0.5],  # heading south toward gate 4, slight descent
            dtype=torch.float32, device=self.device
        )
        self._seed_loop_buffer(LOOP_GATE_ID, up_pos, up_vel)

        # Phase 2: crested the arc, coming down from directly above
        via_pos = torch.tensor(
            [0.6, 1.4, 3.0],    # still north of gate 4, higher — on the way down
            dtype=torch.float32, device=self.device
        )
        approach_vel = torch.tensor(
            [0.0, -4.0, -0.8],  # heading directly south and descending
            dtype=torch.float32, device=self.device
        )
        self._seed_loop_buffer(LOOP_GATE_ID, via_pos, approach_vel)

        # Fix the debug check too
        gate_4_pos  = self.env._waypoints[LOOP_GATE_ID, :3]   # reads index 4 = real loop gate
        gate_4_quat = self.env._waypoints_quat[LOOP_GATE_ID, :]
        for label, p in [("Up-phase ", up_pos), ("Via-point", via_pos)]:
            p_in_frame, _ = subtract_frame_transforms(
                gate_4_pos.unsqueeze(0), gate_4_quat.unsqueeze(0), p.unsqueeze(0)
            )
            print(f"[DEBUG] {label} gate_frame_X = {p_in_frame[0,0]:.3f}  (must be > 0)")
        print(f"[INFO]  Gate {LOOP_GATE_ID} buffer seeded with {self._reset_buffer.get_count(LOOP_GATE_ID)} states")


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

    def _seed_loop_buffer(self, loop_gate_id: int, via_pos: torch.Tensor, approach_vel: torch.Tensor):
        """
        Manually inject states into the reset buffer for a gate that requires a
        non-obvious trajectory (e.g. looping above gate 4).  Call once from __init__.

        Args:
            loop_gate_id:  the gate index whose buffer we seed (e.g. 3 = gate before the loop)
            via_pos:       world-frame position drone should be at when it enters the loop (tensor [3])
            approach_vel:  velocity vector pointing along the correct approach arc (tensor [3])
        """
        yaw = torch.atan2(approach_vel[1], approach_vel[0])
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            yaw.unsqueeze(0)
        )[0]

        # Store several slightly noisy copies so the buffer isn't a single point
        for _ in range(20):
            noise_pos = via_pos + torch.randn(3, device=self.device) * 0.3
            noise_vel = approach_vel + torch.randn(3, device=self.device) * 0.5
            noise_vel[2] = approach_vel[2] + torch.randn(1, device=self.device).item() * 0.3

            self._reset_buffer.store(
                gate_id=loop_gate_id,
                pos=noise_pos,
                lin_vel=noise_vel,
                quat=quat,
                ang_vel=torch.zeros(3, device=self.device),
                motor_speeds=torch.ones(4, device=self.device)
            )

    def _detect_gate_crossing(self, current_gate_position, prev_gate_position,
                            expanded_half, gate_half):
        # expanded_half and gate_half are now [num_envs] tensors, not scalars
        prev_x   = prev_gate_position[:, 0]
        curr_x   = current_gate_position[:, 0]
        curr_y   = current_gate_position[:, 1]
        curr_z   = current_gate_position[:, 2]

        crossed_forward  = (prev_x > 0.0) & (curr_x <= 0.0)
        inside_gate      = (curr_y.abs() <= gate_half)     & (curr_z.abs() <= gate_half)
        inside_expanded  = (curr_y.abs() <= expanded_half) & (curr_z.abs() <= expanded_half)

        gate_passed           = crossed_forward & inside_gate
        gate_missed           = crossed_forward & inside_expanded & (~inside_gate)
        crossed_gate_backward = (prev_x < 0.0) & (curr_x >= 0.0) & inside_expanded

        return gate_passed, gate_missed, crossed_gate_backward


    def _calculate_approach_angle_reward(
        self,
        gate_centers: torch.Tensor,
    ) -> torch.Tensor:
        drone_pos = self.env._robot.data.root_link_pos_w
        vel_w = self.env._robot.data.root_com_lin_vel_w

        dist_to_gate = torch.norm(drone_pos - gate_centers, dim=1)
        gate_normals = self.env._normal_vectors[self.env._idx_wp]

        vel_mag = torch.norm(vel_w, dim=1, keepdim=True).clamp(min=0.1)
        vel_unit = vel_w / vel_mag

        alignment = (vel_unit * gate_normals).sum(dim=1)

        # --- CHANGED: 2.0 m radius (was 3.0), no proximity weight ---
        approach_radius = 2.0
        in_range = (dist_to_gate < approach_radius).float()
        approach_reward = alignment.clamp(min=0.0) * in_range

        return approach_reward

    def _calculate_progress_reward(
        self,
        crossed_gate_correct_direction: torch.Tensor,
        gate_centers: torch.Tensor,
    ) -> torch.Tensor:
        LOOP_GATE_IDX = 4
        via_point = torch.tensor([0.5, 2.2, 2.8], device=self.device)

        drone_pos = self.env._robot.data.root_link_pos_w
        gate_frame_x = self.env._pose_drone_wrt_gate[:, 0]

        targeting_loop = (self.env._idx_wp == LOOP_GATE_IDX)
        needs_arc = targeting_loop & (gate_frame_x < 3.0)

        # Effective gate center for distance computation
        effective_centers = gate_centers.clone()
        effective_centers[needs_arc] = via_point.unsqueeze(0).expand(needs_arc.sum(), -1)

        curr_3d_dist = torch.norm(drone_pos - effective_centers, dim=1)
        euclidean_progress = self._prev_distance_to_gate - curr_3d_dist

        prev_x = self._prev_pose_drone_wrt_gate[:, 0].clone()
        current_x = self.env._pose_drone_wrt_gate[:, 0]


        successful_crossed = torch.where(crossed_gate_correct_direction)[0]
        not_successful_crossed = torch.where(~crossed_gate_correct_direction)[0]

        current_distance = torch.zeros(self.num_envs, device=self.device)

        if successful_crossed.numel() > 0:
            prev_x[successful_crossed] = self._prev_pose_drone_wrt_gate[successful_crossed, 0]
            current_distance[successful_crossed] = 0.0

        if not_successful_crossed.numel() > 0:
            current_distance[not_successful_crossed] = current_x[not_successful_crossed].clamp(min=0.0)

        # Original X-axis progress (precise, but penalises looping above/around)
        prev_x_clamped = prev_x.clamp(min=0.0)
        x_progress = prev_x_clamped - current_distance

        # SWITCH: only use X-axis progress when drone is close AND in front of gate.
        # When far away (> 3 m), fall back to 3D euclidean so upward/sideways arcs
        # are NOT penalised — this is what allows the loop above gate 4 to be discovered.
        use_x_progress = (curr_3d_dist < 3.0) & (gate_frame_x > 0.0) & (~needs_arc)
        progress = torch.where(use_x_progress, x_progress, euclidean_progress)

        return progress, current_distance, effective_centers  # <-- add this

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

    def _update_tracking_variables(self, gate_passed, current_distance, gate_centers, effective_centers):
        ids_not_passed = torch.where(~gate_passed)[0]
        if ids_not_passed.numel() > 0:
            self._prev_pose_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed].clone()
            )
            self.env._prev_x_drone_wrt_gate[ids_not_passed] = (
                self.env._pose_drone_wrt_gate[ids_not_passed, 0]
            )
            drone_pos = self.env._robot.data.root_link_pos_w[ids_not_passed]
            # Use effective_centers so prev/curr distances are always measured to the same target
            self._prev_distance_to_gate[ids_not_passed] = torch.norm(
                drone_pos - effective_centers[ids_not_passed], dim=1
            )

    def get_rewards(self) -> torch.Tensor:
        """
        Dense reward for progress goal: distance reduction to current gate center (going towards the next gate)
        Sparse reward for target reached: 1 for successfully pass through gate correctly, -1 for missing / wrong way
        Dense negative lap incomplete: per-step penalty until 3 laps to force full race completion
        Death cost: large penalty for crash / reset. Overrides other rewards (don't want this)
        Roll/Pitch/Yaw penalty: penalize aggressive RPY rates for more stability
        Wrong direction penalty: avoid going back through same gate for gate 4
        """

        # Gate setup

        # Per-gate half size: use small value for virtual gate
        gate_half_default = getattr(self.env._gate_model_cfg_data, "gate_side", 1.0) * 0.5 \
            if hasattr(self.env, "_gate_model_cfg_data") else 0.5

        is_virtual = (self.env._idx_wp == self._virtual_gate_index)
        gate_half_per_env = torch.where(
            is_virtual,
            torch.full((self.num_envs,), self._virtual_gate_half, device=self.device),
            torch.full((self.num_envs,), gate_half_default, device=self.device)
        )

        expand = getattr(self.cfg, "gate_detection_expand", 1.4)
        expanded_half_per_env = gate_half_per_env * expand

        # Pass scalars into _detect_gate_crossing per-env by calling it per-group,
        # OR simplify: use the default for detection and just log separately.
        # Easiest approach — use the mean (virtual gate is rare, safe approximation):
        gate_half     = gate_half_per_env     # keep as tensor
        expanded_half = expanded_half_per_env
        pose_gate = self.env._pose_drone_wrt_gate  # [x,y,z] in gate frame


        gate_centers = self.env._waypoints[self.env._idx_wp, :3]
        self.env._desired_pos_w[:] = gate_centers   # desired pos = gate centers
        num_gates = self.env._waypoints.shape[0]
        next_gate_indices = (self.env._idx_wp + 1) % num_gates
        
        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # Gate Cross detection
        gate_passed, gate_missed, gate_crossed_backward = self._detect_gate_crossing(
            current_gate_position=pose_gate,
            prev_gate_position=self._prev_pose_drone_wrt_gate,
            expanded_half=expanded_half,
            gate_half=gate_half,
        )
        # includes ALL envs that crossed the plane (passed OR missed)
        crossed_gate_correct_direction = gate_passed | gate_missed
        
        # 1. Dense progress reward
        progress_current, current_distance, effective_centers = self._calculate_progress_reward(crossed_gate_correct_direction, gate_centers)
        # only calc reward for flying towards next gate for success gates
        progress_next_gate = self._calculate_next_gate_progress(gate_passed, gate_centers, next_gate_indices)

        # Combine progress from current gate and next gate
        progress = progress_current + progress_next_gate
        
        # 2. Sparse reward for target reached: 1 / -1
        # advance waypoint index for passed gates only
        self._handle_gate_crossing(gate_passed)
        
        # combined rewards for pass (+1.0) and miss (-1.0)
        gate_reward = gate_passed.float() * 1.0 + gate_missed.float() * (-1.0)
        
        #  update tracking for next timestep
        self._update_tracking_variables(gate_passed, current_distance, gate_centers, effective_centers)

        # Compute pose wrt prev gate
        prev_gate_indices = (self.env._idx_wp - 1) % num_gates            # ← define first

        prev_gate_pos  = self.env._waypoints[prev_gate_indices, :3]
        prev_gate_quat = self.env._waypoints_quat[prev_gate_indices, :]
        current_pose_prev_gate, _ = subtract_frame_transforms(
            prev_gate_pos, prev_gate_quat, self.env._robot.data.root_link_pos_w
        )

        is_prev_virtual = (prev_gate_indices == self._virtual_gate_index)
        prev_gate_half_per_env = torch.where(
            is_prev_virtual,
            torch.full((self.num_envs,), self._virtual_gate_half, device=self.device),
            torch.full((self.num_envs,), gate_half_default, device=self.device)
        )
        prev_expanded_half = prev_gate_half_per_env * expand

        prev_gate_crossed_backward = (
            (self._prev_pose_drone_wrt_prev_gate[:, 0] < 0.0) &
            (current_pose_prev_gate[:, 0] >= 0.0) &
            (torch.abs(current_pose_prev_gate[:, 1]) <= prev_expanded_half) &
            (torch.abs(current_pose_prev_gate[:, 2]) <= prev_expanded_half)
        )
        self._prev_pose_drone_wrt_prev_gate = current_pose_prev_gate.clone()
        # store for next timestep
        self._prev_pose_drone_wrt_prev_gate = current_pose_prev_gate.clone()
        
        if self.cfg.is_train:
            # Scale each reward 
            progress_reward = progress * self.env.rew['progress_goal_reward_scale']
            gate_reward_scaled = gate_reward * self.env.rew['target_reached_bonus_reward_scale']
            
            # Lap incomplete penalty per-step until 3
            laps_completed = self.env._n_gates_passed // num_gates
            lap_penalty = (self.env.rew['lap_incomplete_penalty_reward_scale'] * (laps_completed < 3).float())
            
            # penalize aggressive roll and pitch rates for more stability
            roll_pitch_rates = torch.sum(torch.square(self.env._actions[:, 1:3]), dim=1)
            roll_pitch_rates_penalty = roll_pitch_rates * self.env.rew['roll_pitch_rates_reward_scale'] * self.env.step_dt
            
            # penalize aggressive yaw rates
            yaw_rate = torch.square(self.env._actions[:, 3])
            yaw_rate_penalty = yaw_rate * self.env.rew['yaw_rate_reward_scale'] * self.env.step_dt

            # wrong direction penalty
            wrong_direction = (gate_crossed_backward | prev_gate_crossed_backward).float()
            backward_penalty = wrong_direction * self.env.rew.get('wrong_direction_penalty', -200.0)

            # ---- ADD THESE LINES ----
            # Reward velocity component pointing directly at current gate center.
            # Clamped to 0 so flying away from a gate is not double-penalized here.
            vel_w = self.env._robot.data.root_com_lin_vel_w
            # Replace your current velocity reward block with this
            next_gate_centers = self.env._waypoints[next_gate_indices, :3]

            LOOP_GATE_IDX = 4
            # World-frame via-point: high and NW of gate 4, on the correct approach side
            # gate_4_center + ~3.5m * gate_4_normal + altitude
            via_point = torch.tensor([0.5, 2.2, 2.8], device=self.device)

            # Determine if drone needs to arc (targeting gate 4 AND on wrong/exit side)
            targeting_loop = (self.env._idx_wp == LOOP_GATE_IDX)
            gate_frame_x = self.env._pose_drone_wrt_gate[:, 0]
            on_wrong_side = gate_frame_x < 3.0
            needs_arc = targeting_loop & on_wrong_side

            # Effective target: via-point for arcing envs, gate center otherwise
            effective_target = gate_centers.clone()
            effective_target[needs_arc] = via_point.unsqueeze(0).expand(needs_arc.sum(), -1)

            # Recompute velocity reward using effective target instead of raw gate center
            gate_dir = effective_target - self.env._robot.data.root_link_pos_w
            next_gate_dir = next_gate_centers - self.env._robot.data.root_link_pos_w

            gate_dist = torch.norm(gate_dir, dim=1, keepdim=True).clamp(min=0.1)
            next_gate_dist = torch.norm(next_gate_dir, dim=1, keepdim=True).clamp(min=0.1)

            gate_dir_unit = gate_dir / gate_dist
            next_gate_dir_unit = next_gate_dir / next_gate_dist

            # For arc envs: 100% toward via-point; for others: 80/20 blend as before
            blend_w = torch.where(needs_arc.unsqueeze(1), 
                                torch.ones_like(gate_dir_unit),
                                0.8 * gate_dir_unit + 0.2 * next_gate_dir_unit)
            blended_dir = blend_w / torch.norm(blend_w, dim=1, keepdim=True).clamp(min=0.1)

            dist_to_gate_scalar = gate_dist.squeeze(1)
            dist_weight = (1.0 - (dist_to_gate_scalar - 3.0) / 4.0).clamp(min=0.2, max=1.0)


            # AFTER — gate 4 requires an arc, straight-line velocity reward fights it
            # When gate 4 is the target AND drone is far from it, suppress velocity reward entirely
            gate_4_idx = 4
            is_gate_4 = (self.env._idx_wp == gate_4_idx)
            far_from_gate = (dist_to_gate_scalar > 2.0)

            # For gate 4 when far away: drop velocity reward to 0 so the arc isn't punished
            # For all other gates and close range: use normal velocity reward
            velocity_scale = torch.where(
                is_gate_4 & far_from_gate,
                torch.zeros_like(dist_weight),   # fully suppress for gate 4 arc phase
                dist_weight                       # normal distance-weighted scale otherwise
            )
            vel_toward_gate = (vel_w * blended_dir).sum(dim=1).clamp(min=0.0)
            velocity_reward = vel_toward_gate * self.env.rew['velocity_toward_gate_reward_scale'] * velocity_scale


            approach_angle_reward = self._calculate_approach_angle_reward(gate_centers)
            approach_angle_scaled = approach_angle_reward * self.env.rew.get('approach_angle_reward_scale', 3.0)

            # Aggregate all components
            rewards = {
                "progress_goal": progress_reward,
                "target_reached_bonus": gate_reward_scaled,
                "lap_incomplete_penalty": lap_penalty,
                "roll_pitch_rates": roll_pitch_rates_penalty,
                "yaw_rate": yaw_rate_penalty,
                "backward_crossing": backward_penalty,
                "velocity_toward_gate": velocity_reward,   # <-- ADD THIS
                "approach_angle": approach_angle_scaled,   # ADD THIS
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            
            # Track what the agent earned in each component
            for key, value in rewards.items():
                self._episode_sums[key] += value
            
            # overrid all rewards when death
            reward = torch.where(self.env.reset_terminated, torch.ones_like(reward) * self.env.rew['death_cost'],reward)
        else:
            # eval mode no rewards
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

            if not self.env._models_paths_initialized:
                num_models_per_env = self.env._waypoints.size(0)
                model_prim_names_in_env = [
                    f"{self.env.target_models_prim_base_name}_{i}"
                    for i in range(num_models_per_env)
                ]
                self.env._all_target_models_paths = []
                for env_path in self.env.scene.env_prim_paths:
                    paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                    self.env._all_target_models_paths.append(paths_for_this_env)

                self.env._models_paths_initialized = True

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
        waypoint_indices = start_indices.clone()
        
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
        