# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained DreamerV3 agent from SheepRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained DreamerV3 agent from SheepRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate (defaults to config).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (optional, overrides config).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt file).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (defaults to config).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--greedy", action="store_true", default=False, help="Use greedy actions (no sampling).")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to run (None = infinite).")
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
import numpy as np
import time
import torch

from lightning import Fabric
from omegaconf import OmegaConf

import isaaclab_tasks  # noqa: F401

# Import SheepRL components
from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.envs.wrappers import VectorizedSingleEnvWrapper


def main():
    """Play with a trained DreamerV3 agent."""
    
    # Validate checkpoint exists
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load the checkpoint config
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    
    # Look for config in the checkpoint directory
    checkpoint_dir = checkpoint_path.parent.parent  # Go up from checkpoint/ to run dir
    config_path = checkpoint_dir / "config.yaml"
    
    if not config_path.exists():
        # Try alternative locations
        alt_config_path = checkpoint_dir / ".hydra" / "config.yaml"
        if alt_config_path.exists():
            config_path = alt_config_path
        else:
            raise FileNotFoundError(
                f"Config file not found. Tried:\n"
                f"  - {checkpoint_dir / 'config.yaml'}\n"
                f"  - {alt_config_path}"
            )
    
    print(f"[INFO] Loading SheepRL config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Use CLI args if provided, otherwise from saved config
    task = args_cli.task if args_cli.task is not None else cfg.env.id
    seed = args_cli.seed if args_cli.seed is not None else cfg.seed
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else cfg.env.num_envs
    
    print(f"[INFO] Task: {task}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Num envs: {num_envs}")
    
    # Update SheepRL config
    cfg.env.num_envs = num_envs
    cfg.env.id = task
    cfg.seed = seed
    
    # Clear any existing Hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Setup Fabric for model loading
    fabric = Fabric(
        accelerator=cfg.fabric.accelerator if hasattr(cfg.fabric, 'accelerator') else "auto",
        devices=1,
        precision=cfg.fabric.precision if hasattr(cfg.fabric, 'precision') else "32-true",
    )
    fabric.launch()
    
    # Load checkpoint state
    state = fabric.load(str(checkpoint_path))
    
    # Create Isaac Lab environment
    print(f"[INFO] Creating environment: {task} with {num_envs} envs")
    
    is_isaac_env = task.startswith("Isaac")
    
    if is_isaac_env:
        # Update wrapper config with current task and num_envs
        cfg.env.wrapper.id = task
        cfg.env.wrapper.num_envs = num_envs
        
        # Isaac Lab environments handle multiple envs internally
        single_env = hydra.utils.instantiate(cfg.env.wrapper, _convert_="all")
        envs = VectorizedSingleEnvWrapper(single_env, num_envs=num_envs)
    else:
        raise NotImplementedError("Non-Isaac environments not yet supported in play.py")
    
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    
    print(f"[INFO] Action space: {action_space}")
    print(f"[INFO] Observation space: {observation_space}")
    print(f"[INFO] Actions dim: {actions_dim}")
    print(f"[INFO] Continuous: {is_continuous}")
    
    # Build the agent with loaded weights
    print("[INFO] Building agent from checkpoint...")
    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"],
        state["actor"],
        state.get("critic"),
        state.get("target_critic"),
    )
    
    # Set player to use the correct number of envs
    player.num_envs = num_envs
    player.init_states()
    
    # Get dt for real-time playback
    dt = getattr(envs.env, 'step_dt', 1.0 / 60.0) if hasattr(envs, 'env') else 1.0 / 60.0
    
    # Reset environment
    print("[INFO] Starting playback...")
    obs, _ = envs.reset(seed=seed)
    
    timestep = 0
    cumulative_rewards = np.zeros(num_envs)
    episode_count = 0
    
    # Playback loop
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # Prepare observations for the model
            torch_obs = prepare_obs(
                fabric, obs, 
                cnn_keys=cfg.algo.cnn_keys.encoder, 
                num_envs=num_envs
            )
            
            # Get actions from the player
            mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
            if len(mask) == 0:
                mask = None
            
            real_actions = player.get_actions(torch_obs, args_cli.greedy, mask)
            
            # Convert actions to numpy
            if is_continuous:
                real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
            else:
                real_actions = torch.stack(
                    [real_act.argmax(dim=-1) for real_act in real_actions], dim=-1
                ).cpu().numpy()
            
            # Step environment
            obs, rewards, terminated, truncated, infos = envs.step(
                real_actions.reshape(envs.action_space.shape)
            )
            
            # Track rewards
            cumulative_rewards += rewards.flatten()
            
            # Check for episode ends
            dones = np.logical_or(terminated, truncated)
            if np.any(dones):
                done_indices = np.where(dones.flatten())[0]
                for idx in done_indices:
                    episode_count += 1
                    print(f"[INFO] Episode {episode_count} (env {idx}): Reward = {cumulative_rewards[idx]:.2f}")
                    cumulative_rewards[idx] = 0.0
                
                # Re-initialize states for done envs
                player.init_states(done_indices.tolist())
        
            timestep += 1
        
        # Check max steps
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            print(f"[INFO] Reached max steps ({args_cli.max_steps})")
            break
        
        # Video recording exit condition
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Recorded {args_cli.video_length} frames")
            break

        # Real-time delay
        elapsed = time.time() - start_time
        sleep_time = dt - elapsed
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    print(f"\n[INFO] Playback finished after {timestep} steps, {episode_count} episodes")
    envs.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
