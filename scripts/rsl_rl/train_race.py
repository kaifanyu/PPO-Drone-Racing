# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to path: {project_root}")

rsl_rl_path = os.path.abspath(os.path.join(project_root, "src/third_parties/rsl_rl_local"))
if os.path.exists(rsl_rl_path):
    sys.path.insert(0, rsl_rl_path)
    print(f"[INFO] Using rsl_rl from: {rsl_rl_path}")
else:
    print(f"[WARNING] rsl_rl not found at: {rsl_rl_path}")

import argparse
from isaaclab.app import AppLauncher
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# reward scaling overrides (optional)
parser.add_argument("--progress_scale", type=float, default=None, help="Override progress_goal_reward_scale.")
parser.add_argument("--target_bonus_scale", type=float, default=None, help="Override target_reached_bonus_reward_scale.")
parser.add_argument("--crash_scale", type=float, default=None, help="Override crash_reward_scale.")
parser.add_argument("--death_cost", type=float, default=None, help="Override death_cost.")
parser.add_argument("--lap_penalty_scale", type=float, default=None, help="Per-step penalty until 3 laps complete.")
parser.add_argument("--roll_pitch_rates_scale", type=float, default=None, help="Penalty for aggressive roll/pitch rates. Recommended: -0.5 to -2.0")
parser.add_argument("--yaw_rate_scale", type=float, default=None, help="Penalty for aggressive yaw rate. Recommended: -0.25 to -1.0")
parser.add_argument("--thrust_to_weight", type=float, default=None, help="Override thrust-to-weight ratio. Default: 3.15")
parser.add_argument("--name", type=str, default=None, help="Override run folder name under logs/rsl_rl/<experiment>.")
parser.add_argument("--reset_time", type=int, default=None, help="Max episode length (in policy steps) for reset. None = no timeout (use full episode_length_s).")

parser.add_argument("--wrong_direction_penalty", type=float, default=None)
parser.add_argument("--out_of_order_gate_penalty", type=float, default=None)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Enable Weights & Biases logging by default and set a default project name
    if getattr(agent_cfg, "logger", None) is None:
        agent_cfg.logger = "wandb"
    if agent_cfg.logger == "wandb" and not getattr(agent_cfg, "wandb_project", None):
        agent_cfg.wandb_project = "isaac-quad-race"

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    if args_cli.name:
        # Use explicit name when provided
        log_dir = os.path.join(log_root_path, args_cli.name)
    else:
        # Default: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # ==========================================================================
    # REWARD SCALES
    # Modify these to shape training behavior.
    # CLI flags (--progress_scale, etc.) override these values when provided.
    # ==========================================================================
    progress_goal_reward_scale          = 20.0 # 20, 5
    target_reached_bonus_reward_scale   = 500.0
    death_cost                          = -200.0
    lap_incomplete_penalty_reward_scale = -0.1
    roll_pitch_rates_reward_scale       = -0.5   # penalty for aggressive roll/pitch rates
    yaw_rate_reward_scale               = -1.0   # penalty for aggressive yaw rate -1, -2
    velocity_toward_gate_reward_scale   = 2.0    # reward for velocity toward gate

    # CLI overrides
    if args_cli.progress_scale is not None:
        progress_goal_reward_scale = float(args_cli.progress_scale)
    if args_cli.target_bonus_scale is not None:
        target_reached_bonus_reward_scale = float(args_cli.target_bonus_scale)
    if args_cli.death_cost is not None:
        death_cost = float(args_cli.death_cost)
    if args_cli.lap_penalty_scale is not None:
        lap_incomplete_penalty_reward_scale = float(args_cli.lap_penalty_scale)
    if args_cli.roll_pitch_rates_scale is not None:
        roll_pitch_rates_reward_scale = float(args_cli.roll_pitch_rates_scale)
    if args_cli.yaw_rate_scale is not None:
        yaw_rate_reward_scale = float(args_cli.yaw_rate_scale)

    rewards = {
        'progress_goal_reward_scale':           progress_goal_reward_scale,
        'target_reached_bonus_reward_scale':    target_reached_bonus_reward_scale,
        'death_cost':                           death_cost,
        'lap_incomplete_penalty_reward_scale':  lap_incomplete_penalty_reward_scale,
        'roll_pitch_rates_reward_scale':        roll_pitch_rates_reward_scale,
        'yaw_rate_reward_scale':                yaw_rate_reward_scale,
        'velocity_toward_gate_reward_scale':    velocity_toward_gate_reward_scale,
        'height_align_reward_scale': -2.0,   # negative since it's a penalty
    }

    # ==========================================================================
    # REWARD SHAPING
    # Fine-grained control over hardcoded reward values in get_rewards().
    # ==========================================================================
    reward_shaping = {
        'gate_pass_bonus':      1.0,    # sparse bonus magnitude for a clean gate pass
        'gate_miss_penalty':   -1.0,    # sparse penalty magnitude for clipping the gate
        'vel_blend_ratio':      0.9,    # fraction of velocity reward aimed at current gate
                                        # (1 - vel_blend_ratio) aimed at next gate 0.8, 0.5, 0.95
        'lap_count_threshold':  3,      # number of laps before lap-incomplete penalty stops
        'gate_detection_expand': 1.4,   # expanded gate half-width multiplier for miss detection
        'crash_mask_steps':     100,    # steps before crash detection activates each episode
    }

    # ==========================================================================
    # DOMAIN RANDOMIZATION RANGES
    # Each tuple is (min_multiplier, max_multiplier) applied to the base value.
    # Wider ranges = harder training, better sim-to-real transfer.
    # ==========================================================================
    
    # Default Death Rate: 0.8333

    # dr_ranges = {
    #     'twr_range':            (0.95, 1.05),   # thrust-to-weight ratio
    #     'k_aero_xy_range':      (0.5,  2.0),    # XY aerodynamic drag coefficient
    #     'k_aero_z_range':       (0.5,  2.0),    # Z aerodynamic drag coefficient
    #     'kp_omega_rp_range':    (0.85, 1.15),   # roll/pitch rate Kp gain
    #     'ki_omega_rp_range':    (0.85, 1.15),   # roll/pitch rate Ki gain
    #     'kd_omega_rp_range':    (0.7,  1.3),    # roll/pitch rate Kd gain
    #     'kp_omega_y_range':     (0.85, 1.15),   # yaw rate Kp gain
    #     'ki_omega_y_range':     (0.85, 1.15),   # yaw rate Ki gain
    #     'kd_omega_y_range':     (0.7,  1.3),    # yaw rate Kd gain
    # }

    # Domain Randomization 0, Death Rate: 0.292

    dr_ranges = {
        'twr_range':            (0.90, 1.10),   # thrust-to-weight ratio         ±5% → ±10%
        'k_aero_xy_range':      (0.4,  2.3),    # XY aerodynamic drag coefficient 0.5–2.0 → 0.4–2.3
        'k_aero_z_range':       (0.4,  2.3),    # Z aerodynamic drag coefficient  0.5–2.0 → 0.4–2.3
        'kp_omega_rp_range':    (0.80, 1.20),   # roll/pitch rate Kp gain         ±15% → ±20%
        'ki_omega_rp_range':    (0.80, 1.20),   # roll/pitch rate Ki gain         ±15% → ±20%
        'kd_omega_rp_range':    (0.65, 1.35),   # roll/pitch rate Kd gain         ±30% → ±35%
        'kp_omega_y_range':     (0.80, 1.20),   # yaw rate Kp gain                ±15% → ±20%
        'ki_omega_y_range':     (0.80, 1.20),   # yaw rate Ki gain                ±15% → ±20%
        'kd_omega_y_range':     (0.65, 1.35),   # yaw rate Kd gain                ±30% → ±35%
    }

    # Domain Randomization 1, Death Rate: 0.91677

    # dr_ranges = {
    #     'twr_range':            (0.80, 1.20),   # thrust-to-weight ratio         ±10% → ±20%
    #     'k_aero_xy_range':      (0.25, 2.75),   # XY aerodynamic drag coefficient 0.4–2.3 → 0.25–2.75
    #     'k_aero_z_range':       (0.25, 2.75),   # Z aerodynamic drag coefficient  0.4–2.3 → 0.25–2.75
    #     'kp_omega_rp_range':    (0.70, 1.30),   # roll/pitch rate Kp gain         ±20% → ±30%
    #     'ki_omega_rp_range':    (0.70, 1.30),   # roll/pitch rate Ki gain         ±20% → ±30%
    #     'kd_omega_rp_range':    (0.50, 1.50),   # roll/pitch rate Kd gain         ±35% → ±50%
    #     'kp_omega_y_range':     (0.70, 1.30),   # yaw rate Kp gain                ±20% → ±30%
    #     'ki_omega_y_range':     (0.70, 1.30),   # yaw rate Ki gain                ±20% → ±30%
    #     'kd_omega_y_range':     (0.50, 1.50),   # yaw rate Kd gain                ±35% → ±50%
    # }

    # ==========================================================================
    # SPAWN RANDOMIZATION
    # Controls where drones are spawned relative to gates during curriculum resets.
    # ==========================================================================
    spawn_cfg = {
        'spawn_circle_scale':   3.0,    # radius of random spawn circle around gate
        'spawn_lateral_expand': 3.0,    # lateral spread of spawn positions
        'spawn_before_min':     0.5,    # min distance spawned before (approaching) the gate
        'spawn_before_max':     4.0,    # max distance spawned before the gate
        'spawn_after_min':      0.5,    # min distance spawned after (past) the gate
        'spawn_after_max':      4.0,    # max distance spawned after the gate
        'spawn_z_min':          0.05,   # min vertical spawn offset
        'spawn_z_max':          3.0,    # max vertical spawn offset
    }

    # OBSERVATION NOISE
    # Gaussian noise (std) injected per observation component during training.
    # Simulates sensor noise to improve sim-to-real transfer.
    # Set any value to 0.0 to disable noise for that component.
    # ==========================================================================
    # No Noise, Death Rate: 0.292
    obs_noise = {
        'lin_vel_std':      0.00,   # linear velocity in body frame (m/s)
        'attitude_std':     0.00,   # rotation matrix elements (unitless)
        'gate_curr_std':    0.00,   # current gate corner positions in body frame (m)
        'gate_next_std':    0.00,   # next gate corner positions in body frame (m)
    }
    # Default Noise, Death Rate: 0.875
    # obs_noise = {
    #     'lin_vel_std':      0.05,   # linear velocity in body frame (m/s)
    #     'attitude_std':     0.01,   # rotation matrix elements (unitless)
    #     'gate_curr_std':    0.05,   # current gate corner positions in body frame (m)
    #     'gate_next_std':    0.05,   # next gate corner positions in body frame (m)
    # }

    env_cfg.is_train = True
    env_cfg.rewards = rewards
    env_cfg.reset_time = args_cli.reset_time

    # Pass reward shaping, DR ranges, spawn config, and obs noise to env
    env_cfg.reward_shaping = reward_shaping
    env_cfg.dr_ranges = dr_ranges
    env_cfg.obs_noise = obs_noise
    env_cfg.gate_detection_expand = reward_shaping['gate_detection_expand']
    for k, v in spawn_cfg.items():
        setattr(env_cfg, k, v)

    # Override thrust_to_weight if specified
    if args_cli.thrust_to_weight is not None:
        env_cfg.thrust_to_weight = args_cli.thrust_to_weight
        print(f"[INFO] Using thrust_to_weight = {args_cli.thrust_to_weight}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, rewards=rewards)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore
    # close sim app
    simulation_app.close()