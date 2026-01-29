# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a DreamerV3 agent with SheepRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a DreamerV3 agent with SheepRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
parser.add_argument("--total_steps", type=int, default=2000000, help="Total training steps.")
parser.add_argument("--learning_starts", type=int, default=1024, help="Steps before learning starts.")
parser.add_argument("--checkpoint_every", type=int, default=100000, help="Save checkpoint every N steps.")
parser.add_argument("--log_every", type=int, default=512, help="Log metrics every N steps.")
parser.add_argument("--replay_ratio", type=float, default=0.02, help="Replay ratio for training.")
parser.add_argument("--render_mode", type=str, default="human", help="Render mode (human, rgb_array, None).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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
import hydra
import os

import isaaclab_tasks  # noqa: F401

# Clear any existing Hydra instance before initializing
hydra.core.global_hydra.GlobalHydra.instance().clear()

# IMPORTANT: Import the algorithm module BEFORE calling run()
# This registers the algorithm with SheepRL's internal module registry.
from sheeprl.algos.dreamer_v3 import dreamer_v3  # noqa: F401 - import needed for registration
from sheeprl.cli import run
from hydra import compose, initialize


def main():
    """Train with SheepRL DreamerV3 agent."""
    
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Seed: {args_cli.seed}")
    print(f"[INFO] Total steps: {args_cli.total_steps}")
    
    # Build overrides list from CLI arguments
    overrides = [
        "exp=dreamer_v3",
        "algo=dreamer_v3_S",
        "env=isaac",
        f"env.id={args_cli.task}",
        "fabric.accelerator=gpu",
        "fabric.precision=bf16-true",
        f"algo.total_steps={args_cli.total_steps}",
        "algo.cnn_keys.encoder=[]",
        'algo.mlp_keys.encoder=["vector"]',
        "algo.cnn_keys.decoder=[]",
        'algo.mlp_keys.decoder=["vector"]',
        f"env.num_envs={args_cli.num_envs}",
        f"env.capture_video={str(args_cli.video)}",
        f"env.wrapper.render_mode={args_cli.render_mode}",
        "algo.run_test=False",
        f"algo.learning_starts={args_cli.learning_starts}",
        "num_threads=16",
        f"checkpoint.every={args_cli.checkpoint_every}",
        f"metric.log_every={args_cli.log_every}",
        f"algo.replay_ratio={args_cli.replay_ratio}",
        f"seed={args_cli.seed}",
    ]
    
    # Add checkpoint resume if provided
    if args_cli.checkpoint is not None:
        overrides.append(f"checkpoint.resume_from={args_cli.checkpoint}")
    
    print(f"[INFO] SheepRL overrides: {overrides}")
    
    # Initialize Hydra and compose config
    with initialize(version_base=None, config_path="../../../MySheepRL/sheeprl/configs"):
        cfg = compose(config_name="config", overrides=overrides)
        run(cfg)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
