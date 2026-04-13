# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument("--time_laps", action="store_true", default=False, help="Print simulated lap times. Does not stop after 3 laps.")
parser.add_argument("--end_after_laps", action="store_true", default=False, help="Stop simulation after 3 laps.")

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
import src.isaac_quad_sim2real.tasks   # noqa: F401

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
        env_cfg.viewer.resolution = (1280, 720)
        # env_cfg.viewer.resolution = (1920, 1080)
        # View Default
        # env_cfg.viewer.eye = (10.7, 0.4, 7.2)
        # env_cfg.viewer.lookat = (-2.7, 0.5, -0.3)
        # View for Testing Powerloop
        # env_cfg.viewer.eye    = (8.0, -9.5, 3.2)   # +2.3 back, +3.1 left
        # env_cfg.viewer.lookat = (-2.7, 4.0, -1)  # aim shifted left (+3.5 Y)
        # View for Testing Circle
        env_cfg.viewer.eye = (12.0, 0.4, 7.2) # +1.3, 
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
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs = env.get_observations()
    # Extract tensor from TensorDict for policy
    if hasattr(obs, "get"):  # Check if it's a TensorDict
        obs = obs["policy"]  # Extract the policy observation
    timestep = 0
    # Define your control time step (check your task config, often 0.01 or 0.02)
    # This is (simulation dt * decimation). We will assume 0.02 for this example.
    control_dt = 0.02 
    
    total_steps = 0
    lap_count = 0
    lap_start_step = 0
    prev_gates_passed = 0
    base_env = env.unwrapped
    num_gates = base_env._waypoints.shape[0]
    NUM_LAPS = 3

    if args_cli.time_laps or args_cli.end_after_laps:
        print(f"[INFO] Lap timing enabled. {'Will stop after ' + str(NUM_LAPS) + ' laps.' if args_cli.end_after_laps else 'Running indefinitely.'}")
    else:
        print("[INFO] Running indefinitely (no lap timing or stopping).")

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

            if hasattr(obs, "get"):
                obs = obs["policy"]

            total_steps += 1

            # Lap timing: track gates passed for env 0
            if args_cli.time_laps or args_cli.end_after_laps:
                curr_gates = base_env._n_gates_passed[0].item()
                if curr_gates > prev_gates_passed:
                    # One or more gates passed this step
                    for _ in range(curr_gates - prev_gates_passed):
                        if curr_gates % num_gates == 0:
                            lap_count += 1
                            lap_steps = total_steps - lap_start_step
                            lap_time = lap_steps * control_dt
                            total_time = total_steps * control_dt
                            print(f"[LAP {lap_count}] Lap: {lap_time:.2f}s | Total: {total_time:.2f}s")
                            lap_start_step = total_steps

                            if args_cli.end_after_laps and lap_count >= NUM_LAPS:
                                print(f"[INFO] {NUM_LAPS} laps complete. Stopping.")
                                break
                    prev_gates_passed = curr_gates

            # On episode reset (done), update gate baseline
            if dones[0].item():
                if args_cli.end_after_laps and lap_count >= NUM_LAPS:
                    break
                # Reset per-episode gate counter baseline
                prev_gates_passed = 0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
