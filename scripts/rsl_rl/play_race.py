# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--num_laps", type=int, default=3, help="Number of laps to track before exiting.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Gate & Lap Tracking
# ---------------------------------------------------------------------------

@dataclass
class GatePassEvent:
    """Stores data for a single gate-pass event."""
    gate_index: int
    gate_name: str
    lap_number: int
    sim_time_s: float          # simulation time in seconds
    wall_time_s: float         # real wall-clock time in seconds
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    speed_mps: float
    direction: str             # "forward" or "backward"


@dataclass
class LapSummary:
    """Summary for a completed lap."""
    lap_number: int
    start_time_s: float
    end_time_s: float
    gate_events: List[GatePassEvent] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def avg_speed_mps(self) -> float:
        if not self.gate_events:
            return 0.0
        return float(np.mean([e.speed_mps for e in self.gate_events]))

    @property
    def max_speed_mps(self) -> float:
        if not self.gate_events:
            return 0.0
        return float(np.max([e.speed_mps for e in self.gate_events]))


class LapTracker:
    """
    Tracks gate crossings and lap times for a drone racing task.

    Gate crossing is detected when the drone passes within `gate_radius`
    metres of a gate centre and then exits again (rising-edge trigger).
    Direction is determined by the sign of the drone's velocity projected
    onto the gate's forward normal vector.

    Parameters
    ----------
    gate_positions : list of (x, y, z)
        World-space centre of each gate, **in lap order**.
    gate_normals : list of (nx, ny, nz)
        Unit vector pointing in the "forward" direction through each gate.
        If None, defaults to (1, 0, 0) for every gate.
    gate_names : list of str, optional
        Human-readable label for each gate.
    gate_radius : float
        Proximity threshold (metres) used to detect a crossing.
    sim_dt : float
        Simulation time-step in seconds (used to convert step count → time).
    num_laps : int
        Stop tracking after this many completed laps.
    env_index : int
        Which parallel environment to track (default 0).
    """

    def __init__(
        self,
        gate_positions: List[Tuple[float, float, float]],
        gate_normals: Optional[List[Tuple[float, float, float]]] = None,
        gate_names: Optional[List[str]] = None,
        gate_radius: float = 1.0,
        sim_dt: float = 0.02,
        num_laps: int = 3,
        env_index: int = 0,
    ):
        self.gate_positions = [np.array(p, dtype=float) for p in gate_positions]
        self.num_gates = len(gate_positions)

        if gate_normals is None:
            self.gate_normals = [np.array([1.0, 0.0, 0.0]) for _ in gate_positions]
        else:
            self.gate_normals = [np.array(n, dtype=float) for n in gate_normals]

        self.gate_names = gate_names or [f"Gate {i+1}" for i in range(self.num_gates)]
        self.gate_radius = gate_radius
        self.sim_dt = sim_dt
        self.num_laps = num_laps
        self.env_index = env_index

        # State
        self._inside_gate: List[bool] = [False] * self.num_gates  # hysteresis flags
        self._next_expected_gate: int = 0                          # enforce lap order
        self._current_lap: int = 1
        self._step_count: int = 0
        self._wall_start: float = time.perf_counter()

        # Results
        self.gate_events: List[GatePassEvent] = []
        self.lap_summaries: List[LapSummary] = []
        self._current_lap_start_s: float = 0.0
        self.completed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        pos: torch.Tensor,   # shape (num_envs, 3)
        vel: torch.Tensor,   # shape (num_envs, 3)
    ) -> None:
        """Call once per simulation step with position and velocity tensors."""
        if self.completed:
            return

        self._step_count += 1
        sim_time = self._step_count * self.sim_dt
        wall_time = time.perf_counter() - self._wall_start

        drone_pos = pos[self.env_index].cpu().numpy()
        drone_vel = vel[self.env_index].cpu().numpy()
        speed = float(np.linalg.norm(drone_vel))

        for gate_idx in range(self.num_gates):
            gate_pos = self.gate_positions[gate_idx]
            dist = float(np.linalg.norm(drone_pos - gate_pos))
            in_range = dist < self.gate_radius

            # Rising-edge: drone just entered gate radius
            if in_range and not self._inside_gate[gate_idx]:
                self._inside_gate[gate_idx] = True

                # Only register if this is the next gate in sequence
                if gate_idx == self._next_expected_gate:
                    direction = self._compute_direction(gate_idx, drone_vel)

                    event = GatePassEvent(
                        gate_index=gate_idx,
                        gate_name=self.gate_names[gate_idx],
                        lap_number=self._current_lap,
                        sim_time_s=sim_time,
                        wall_time_s=wall_time,
                        position=tuple(drone_pos.tolist()),
                        velocity=tuple(drone_vel.tolist()),
                        speed_mps=speed,
                        direction=direction,
                    )
                    self.gate_events.append(event)

                    # Advance to the next expected gate
                    self._next_expected_gate = (gate_idx + 1) % self.num_gates

                    # If we just passed the LAST gate, a lap is complete
                    if gate_idx == self.num_gates - 1:
                        self._finish_lap(sim_time)

            # Falling-edge: drone has left gate radius
            elif not in_range and self._inside_gate[gate_idx]:
                self._inside_gate[gate_idx] = False

    @property
    def laps_completed(self) -> int:
        return len(self.lap_summaries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_direction(self, gate_idx: int, vel: np.ndarray) -> str:
        normal = self.gate_normals[gate_idx]
        projection = float(np.dot(vel, normal))
        return "forward" if projection >= 0 else "backward"

    def _finish_lap(self, sim_time: float) -> None:
        events_this_lap = [e for e in self.gate_events if e.lap_number == self._current_lap]
        summary = LapSummary(
            lap_number=self._current_lap,
            start_time_s=self._current_lap_start_s,
            end_time_s=sim_time,
            gate_events=events_this_lap,
        )
        self.lap_summaries.append(summary)
        self._current_lap_start_s = sim_time

        if self._current_lap >= self.num_laps:
            self.completed = True
            self._print_full_summary()
        else:
            self._current_lap += 1

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def _print_full_summary(self) -> None:
        sep = "=" * 72
        thin = "-" * 72

        print(f"\n{sep}")
        print(f"  RACE SUMMARY  —  {self.num_laps} Laps Completed")
        print(sep)

        total_duration = self.lap_summaries[-1].end_time_s - self.lap_summaries[0].start_time_s
        print(f"  Total race time : {total_duration:.3f} s")
        print(f"  Total gate passes: {len(self.gate_events)}")
        print(thin)

        for lap in self.lap_summaries:
            print(f"\n  LAP {lap.lap_number}  |  Duration: {lap.duration_s:.3f} s"
                  f"  |  Avg speed: {lap.avg_speed_mps:.2f} m/s"
                  f"  |  Max speed: {lap.max_speed_mps:.2f} m/s")
            print(f"  {'Gate':<14} {'Sim Time (s)':>12} {'Wall Time (s)':>13} "
                  f"{'Speed (m/s)':>11} {'Direction':>10} {'Position (x,y,z)':>24}  "
                  f"{'Velocity (vx,vy,vz)'}")
            print(f"  {thin}")
            for ev in lap.gate_events:
                pos_str = f"({ev.position[0]:+.2f}, {ev.position[1]:+.2f}, {ev.position[2]:+.2f})"
                vel_str = f"({ev.velocity[0]:+.2f}, {ev.velocity[1]:+.2f}, {ev.velocity[2]:+.2f})"
                print(
                    f"  {ev.gate_name:<14} {ev.sim_time_s:>12.3f} {ev.wall_time_s:>13.3f} "
                    f"{ev.speed_mps:>11.3f} {ev.direction:>10}  {pos_str:>24}  {vel_str}"
                )

        print(f"\n{sep}")
        print("  LAP TIME BREAKDOWN")
        print(thin)
        best_lap = min(self.lap_summaries, key=lambda l: l.duration_s)
        for lap in self.lap_summaries:
            marker = " ← best" if lap == best_lap else ""
            print(f"  Lap {lap.lap_number}: {lap.duration_s:.3f} s{marker}")
        print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    if args_cli.follow_robot == -1:
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.eye = (10.7, 0.4, 7.2)
        env_cfg.viewer.lookat = (-2.7, 0.5, -0.3)
    elif args_cli.follow_robot >= 0:
        env_cfg.viewer.eye = (-0.8, 0.8, 0.8)
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = args_cli.follow_robot
        env_cfg.viewer.asset_name = "robot"

    env_cfg.is_train = False
    env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        if args_cli.video_length < args_cli.num_laps * 500:  # rough sanity check
            print(
                f"[WARNING] --video_length={args_cli.video_length} may be too short to capture "
                f"{args_cli.num_laps} full laps. The sim loop exits when the video length is "
                f"reached, so the lap summary will only print if all laps complete first."
            )
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # ------------------------------------------------------------------
    # ✏️  CONFIGURE YOUR GATES HERE
    #     Replace these with the actual world-space gate positions and
    #     forward normals from your course definition.  The order must
    #     match the intended lap direction.
    # ------------------------------------------------------------------
    GATE_POSITIONS = [
        (5.0,  0.0, 1.5),   # Gate 1
        (5.0,  5.0, 1.5),   # Gate 2
        (0.0,  5.0, 1.5),   # Gate 3
        (0.0,  0.0, 1.5),   # Gate 4  ← drone passes this to finish a lap
    ]
    GATE_NORMALS = [
        (0.0,  1.0, 0.0),   # pointing +Y  (drone flies in +Y to go forward)
        (-1.0, 0.0, 0.0),   # pointing -X
        (0.0, -1.0, 0.0),   # pointing -Y
        (1.0,  0.0, 0.0),   # pointing +X
    ]
    GATE_NAMES = ["Start/Finish", "Gate 2", "Gate 3", "Gate 4"]

    # Grab the simulation dt from the environment if available, else default 0.02 s
    sim_dt = getattr(env_cfg, "sim", None)
    sim_dt = getattr(sim_dt, "dt", 0.02) if sim_dt is not None else 0.02

    tracker = LapTracker(
        gate_positions=GATE_POSITIONS,
        gate_normals=GATE_NORMALS,
        gate_names=GATE_NAMES,
        gate_radius=1.2,       # metres — tune to your gate size
        sim_dt=sim_dt,
        num_laps=args_cli.num_laps,
        env_index=0,           # track env 0; change if needed
    )

    print(f"[INFO] LapTracker initialised — tracking {args_cli.num_laps} laps "
          f"across {len(GATE_POSITIONS)} gates (dt={sim_dt:.4f} s).")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

        # ---- Gate / lap tracking ----
        # Try to pull position + velocity from the unwrapped env.
        # Adjust attribute names to match your specific task if needed.
        base_env = env.unwrapped
        if hasattr(base_env, "robot"):
            pos = base_env.robot.data.root_pos_w          # (num_envs, 3)
            vel = base_env.robot.data.root_lin_vel_w       # (num_envs, 3)
            tracker.step(pos, vel)
        else:
            # Fallback: if your env exposes state differently, hook in here.
            pass

        # Exit once all laps are complete (even without video)
        if tracker.completed:
            print("[INFO] All laps complete — exiting simulation loop.")
            break

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                if not tracker.completed:
                    print(
                        f"[WARNING] Video length reached after {timestep} steps but only "
                        f"{tracker.laps_completed}/{args_cli.num_laps} laps were completed. "
                        f"Increase --video_length to capture all laps."
                    )
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()