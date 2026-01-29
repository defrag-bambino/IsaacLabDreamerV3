# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained DreamerV3 agent with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

# Add paths BEFORE parsing args
dreamerv3_path = pathlib.Path(__file__).parent.parent.parent.parent / "dreamerv3"
sys.path.insert(0, str(dreamerv3_path))
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir))

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate DreamerV3 agent with Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (defaults to task from checkpoint).")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment.")

# Checkpoint
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., ~/logdir/dreamerv3/20251216T163553).")

# Evaluation
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, other_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


def main():
    """Evaluate DreamerV3 agent."""
    import numpy as np
    import ruamel.yaml as yaml
    import os
    import pathlib
    
    # Import after Isaac Sim is initialized
    import elements
    import embodied
    from dreamerv3.agent import Agent
    from isaac_env import IsaacBatchedEnv
    
    # Expand checkpoint path
    checkpoint_path = pathlib.Path(os.path.expanduser(args_cli.checkpoint))
    
    # Handle both formats: user can pass either run dir or ckpt dir
    # e.g., ~/logdir/dreamerv3/20251216T163553 OR ~/logdir/dreamerv3/20251216T163553/ckpt
    if checkpoint_path.name == 'ckpt':
        checkpoint_dir = checkpoint_path.parent
        ckpt_dir = checkpoint_path
    else:
        checkpoint_dir = checkpoint_path
        ckpt_dir = checkpoint_path / 'ckpt'
    
    print(f"\n{'='*60}")
    print("DreamerV3 Evaluation with Isaac Lab")
    print(f"{'='*60}")
    print(f"Run directory: {checkpoint_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")
    
    # Load config from checkpoint
    config_path = checkpoint_dir / 'config.yaml'
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found at {config_path}")
        return
    
    print(f"[INFO] Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.YAML(typ='safe').load(f)
    config = elements.Config(config_dict)
    
    # Get task from config or CLI
    task = args_cli.task
    if task is None:
        # Extract task from config (remove 'isaac_' prefix)
        task = config.task
        if task.startswith('isaac_'):
            task = task[6:]
        print(f"[INFO] Using task from checkpoint: {task}")
    else:
        print(f"[INFO] Using task from CLI: {task}")
    
    num_envs = args_cli.num_envs
    
    print(f"Task: {task}")
    print(f"Num envs: {num_envs}")
    print(f"Episodes: {args_cli.episodes}")
    print(f"{'='*60}\n")
    
    # Create environment
    print("[INFO] Creating environment...")
    env = IsaacBatchedEnv(
        task=task,
        num_envs=num_envs,
        device='cuda:0',
        seed=args_cli.seed,
        obs_key='vector',
        act_key='action',
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Setup video recording if requested
    if args_cli.video:
        import cv2
        from collections import defaultdict

        video_folder = os.path.join(str(checkpoint_dir), "videos", "eval")
        os.makedirs(video_folder, exist_ok=True)

        # Video recording setup
        video_writers = {}
        episode_frames = defaultdict(list)

        print(f"[INFO] Recording videos to: {video_folder}")
        print(f"[INFO] Video length: {args_cli.video_length} steps")
        print(f"[INFO] Video interval: {args_cli.video_interval} steps")

        def save_video(episode_idx, frames):
            """Save frames as MP4 video."""
            if not frames:
                return

            video_path = os.path.join(video_folder, f"episode_{episode_idx:03d}.mp4")
            height, width, _ = frames[0].shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")

        video_enabled = True
    else:
        video_enabled = False
    
    # Apply wrappers
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    
    print(f"[INFO] Observation space: {env.obs_space}")
    print(f"[INFO] Action space: {env.act_space}")
    
    # Create agent
    print("[INFO] Creating agent...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(checkpoint_dir),
        seed=args_cli.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    ))
    
    # Load checkpoint
    print(f"[INFO] Looking for checkpoint at: {ckpt_dir}")
    
    # List checkpoint directory contents for debugging
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.iterdir()) if ckpt_dir.is_dir() else []
        print(f"[INFO] Checkpoint directory exists, {len(ckpt_files)} files found")
        if ckpt_files:
            print(f"[INFO] Latest files: {[f.name for f in sorted(ckpt_files)[-3:]]}")
    else:
        print(f"[ERROR] Checkpoint directory not found at {ckpt_dir}")
        print(f"[INFO] Contents of {checkpoint_dir}:")
        if checkpoint_dir.exists():
            for item in checkpoint_dir.iterdir():
                print(f"  - {item.name}")
        else:
            print(f"[ERROR] Run directory {checkpoint_dir} also doesn't exist!")
        return
    
    # Use elements.Checkpoint class (like train.py does)
    print(f"[INFO] Loading agent weights...")
    cp = elements.Checkpoint(ckpt_dir)
    cp.agent = agent
    cp.load()
    print(f"[INFO] Checkpoint loaded successfully")
    
    # Initialize policy
    print(f"[INFO] Initializing policy for {num_envs} environments...")
    carry = agent.init_policy(num_envs)
    
    # Get initial observations
    obs = env.reset()
    obs_shapes = {k: v.shape for k, v in obs.items() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']}
    print(f"[INFO] Initial obs shapes: {obs_shapes}")  # Should be [num_envs, obs_dim] for each component
    
    # Run evaluation
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)
    completed_episodes = []
    completed_lengths = []

    print(f"\n[INFO] Running evaluation for {args_cli.episodes} episodes...")

    step = 0
    episode_step = 0  # Step within current episode for video recording
    current_episode_idx = 0

    while len(completed_episodes) < args_cli.episodes:
        # Get action from policy (batched)
        carry, acts, _ = agent.policy(carry, obs, mode='eval')
        acts['reset'] = obs['is_last'].copy()
        
        # Step environment
        obs = env.step(acts)

        # Capture video frames if enabled
        if video_enabled and episode_step < args_cli.video_length:
            frame = env.render()
            if frame is not None:
                # For batched env, we only record from the first environment
                if isinstance(frame, list):
                    frame = frame[0]  # Take first environment's frame
                episode_frames[current_episode_idx].append(frame)

        # Track rewards
        episode_rewards += obs['reward']
        episode_lengths += 1
        step += num_envs
        episode_step += 1
        
        # Check for completed episodes
        for i in range(num_envs):
            if obs['is_last'][i]:
                completed_episodes.append(float(episode_rewards[i]))
                completed_lengths.append(int(episode_lengths[i]))
                print(f"Episode {len(completed_episodes)}: Reward = {episode_rewards[i]:.2f}, Length = {episode_lengths[i]}")
                
                # Save video for completed episode if recording enabled
                if video_enabled and current_episode_idx in episode_frames:
                    save_video(current_episode_idx, episode_frames[current_episode_idx])
                    del episode_frames[current_episode_idx]

                # Reset tracking for this env
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

                current_episode_idx += 1
                episode_step = 0  # Reset episode step for next episode

                if len(completed_episodes) >= args_cli.episodes:
                    break
    
    # Save any remaining videos
    if video_enabled:
        for ep_idx, frames in episode_frames.items():
            if frames:
                save_video(ep_idx, frames)

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes: {len(completed_episodes)}")
    print(f"Mean Reward: {np.mean(completed_episodes):.2f} +/- {np.std(completed_episodes):.2f}")
    print(f"Mean Length: {np.mean(completed_lengths):.1f} +/- {np.std(completed_lengths):.1f}")
    print(f"Min Reward: {np.min(completed_episodes):.2f}")
    print(f"Max Reward: {np.max(completed_episodes):.2f}")
    if video_enabled:
        print(f"Videos saved to: {video_folder}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Evaluation failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")
    finally:
        # close sim app
        simulation_app.close()

