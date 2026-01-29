# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained DreamerV3 agent with imagination-based error analysis."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a DreamerV3 agent with imagination-based error analysis.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1 for analysis).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (optional, overrides config).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt file).")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of evaluation steps.")
parser.add_argument("--imagination_steps", type=int, default=15, help="Number of imagination steps for prediction.")
parser.add_argument("--stabilization_steps", type=int, default=500, help="Steps before starting error collection.")
parser.add_argument("--error_window_size", type=int, default=50, help="Window size for averaging errors.")
parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save results.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Disable plotting (for headless).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import hydra
import numpy as np
import os
import torch

from lightning import Fabric
from omegaconf import OmegaConf

import isaaclab_tasks  # noqa: F401

# Import SheepRL components
from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.data.buffers import SequentialReplayBuffer
from sheeprl.envs.wrappers import VectorizedSingleEnvWrapper
from sheeprl.utils.distribution import TwoHotEncodingDistribution


def main():
    """Evaluate DreamerV3 agent with imagination-based error analysis."""
    
    # Validate checkpoint exists
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the checkpoint config
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    
    checkpoint_dir = checkpoint_path.parent.parent
    config_path = checkpoint_dir / "config.yaml"
    
    if not config_path.exists():
        alt_config_path = checkpoint_dir / ".hydra" / "config.yaml"
        if alt_config_path.exists():
            config_path = alt_config_path
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
    
    print(f"[INFO] Loading SheepRL config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Use CLI args if provided, otherwise from saved config
    task = args_cli.task if args_cli.task is not None else cfg.env.id
    seed = args_cli.seed
    num_envs = args_cli.num_envs
    eval_steps = args_cli.eval_steps
    imagination_steps = args_cli.imagination_steps
    stabilization_steps = args_cli.stabilization_steps
    error_window_size = args_cli.error_window_size
    
    print(f"[INFO] Task: {task}")
    print(f"[INFO] Seed: {seed}")
    print(f"[INFO] Num envs: {num_envs}")
    print(f"[INFO] Eval steps: {eval_steps}")
    print(f"[INFO] Imagination steps: {imagination_steps}")
    
    # Update SheepRL config
    cfg.env.num_envs = num_envs
    cfg.env.id = task
    cfg.seed = seed
    
    # Clear any existing Hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Setup Fabric
    fabric = Fabric(
        accelerator=cfg.fabric.accelerator if hasattr(cfg.fabric, 'accelerator') else "auto",
        devices=1,
        precision=cfg.fabric.precision if hasattr(cfg.fabric, 'precision') else "32-true",
    )
    fabric.launch()
    
    # Load checkpoint state
    state = fabric.load(str(checkpoint_path))
    
    # Create Isaac Lab environment
    print(f"[INFO] Creating environment: {task}")
    
    is_isaac_env = task.startswith("Isaac")
    
    if is_isaac_env:
        cfg.env.wrapper.id = task
        cfg.env.wrapper.num_envs = num_envs
        single_env = hydra.utils.instantiate(cfg.env.wrapper, _convert_="all")
        envs = VectorizedSingleEnvWrapper(single_env, num_envs=num_envs)
    else:
        raise NotImplementedError("Non-Isaac environments not yet supported")
    
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    
    # Get observation keys
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
    
    print(f"[INFO] Action space: {action_space}")
    print(f"[INFO] Observation space: {observation_space}")
    print(f"[INFO] Obs keys: {obs_keys}")
    
    # Build the agent
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
    
    player.num_envs = num_envs
    player.init_states()
    
    # Create replay buffers
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    rb_initial = SequentialReplayBuffer(eval_steps, num_envs)
    
    # Initialize step data
    step_data = {}
    obs, _ = envs.reset(seed=seed)
    
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["dones"] = np.zeros((1, num_envs, 1))
    step_data["terminated"] = np.zeros((1, num_envs, 1))
    step_data["truncated"] = np.zeros((1, num_envs, 1))
    step_data["rewards"] = np.zeros((1, num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["dones"])
    step_data["stochastic_state"] = player.stochastic_state.detach().cpu().numpy()
    step_data["recurrent_state"] = player.recurrent_state.detach().cpu().numpy()
    
    # Tracking variables
    imags = []
    imag_errors = []
    avg_recent_errors_timeline = []
    avg_recent_rewards_timeline = []
    recent_errors = []
    recent_rewards = []
    
    # Get observation dimension for error normalization
    obs_dim = observation_space["vector"].shape[0] if "vector" in observation_space.spaces else 1
    
    # Rewards prediction tracking
    rewards_list = np.empty((eval_steps, imagination_steps))
    true_rewards_list = []
    
    print(f"[INFO] Starting evaluation for {eval_steps} steps...")
    
    # Main evaluation loop
    for i in range(eval_steps):
        if i == eval_steps // 2:
            print("[INFO] Modifying robot at midpoint...")
            
            # Access the underlying Isaac Lab environment directly
            # envs -> VectorizedSingleEnvWrapper -> IsaacLabWrapper -> Isaac Lab env
            isaac_env = envs.env.env.unwrapped
            
            # Access the robot articulation
            robot = isaac_env.scene["robot"]
            
            # Print joint names to find the right indices
            print("Joint names:", robot.data.joint_names)
            print("Number of joints:", robot.num_joints)
            # Joint names: ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
            #               'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
            #               'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
            # Joint indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            print("Joint position limits:", robot.data.joint_pos_limits)
            # Joint position limits: tensor([[[-1.0472,  1.0472], 0, FL hip
                                            # [-1.0472,  1.0472], 1, FR hip
                                            # [-1.0472,  1.0472], 2, RL hip
                                            # [-1.0472,  1.0472], 3, RR hip
                                            # [-1.5708,  3.4907], 4, FL thigh
                                            # [-1.5708,  3.4907], 5, FR thigh
                                            # [-0.5236,  4.5379], 6, RL thigh
                                            # [-0.5236,  4.5379], 7, RR thigh
                                            # [-2.7227, -0.8378], 8, FL calf
                                            # [-2.7227, -0.8378], 9, FR calf
                                            # [-2.7227, -0.8378], 10, RL calf
                                            # [-2.7227, -0.8378], 11, RR calf]]], device='cuda:0')

            # Lock a joint (e.g., joint 3)
            robot.write_joint_position_limit_to_sim(
                torch.tensor([[[1.0, 1.01]]], device=robot.device), 
                joint_ids=[7]  # RR thigh
            )
            print("  - Locked joint 7 (RR thigh)")

        with torch.no_grad():
            # Prepare observations
            preprocessed_obs = {}
            for k, v in obs.items():
                preprocessed_obs[k] = torch.as_tensor(v[np.newaxis], dtype=torch.float32, device=fabric.device)
                if k in cfg.algo.cnn_keys.encoder:
                    preprocessed_obs[k] = preprocessed_obs[k] / 255.0
            
            mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
            if len(mask) == 0:
                mask = None
            
            # Get actions from player
            real_actions = actions = player.get_actions(preprocessed_obs, mask=mask)
            actions = torch.cat(actions, -1).cpu().numpy()
            if is_continuous:
                real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
            else:
                real_actions = torch.cat([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
        
        # Store step data
        step_data["actions"] = actions.reshape((1, num_envs, -1))
        rb_initial.add(step_data, validate_args=cfg.buffer.validate_args)
        
        # Step environment
        next_obs, rewards, terminated, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
        
        rewards = np.array(rewards).reshape((1, num_envs, -1))
        terminated = np.array(terminated).reshape((1, num_envs, -1))
        truncated = np.array(truncated).reshape((1, num_envs, -1))
        dones = np.logical_or(terminated, truncated).astype(np.uint8).reshape((1, num_envs, -1))
        
        true_rewards_list.append(rewards[0, 0, 0])
        
        step_data["is_first"] = np.zeros_like(step_data["terminated"])
        
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v
        
        for k in obs_keys:
            step_data[k] = next_obs[k][np.newaxis]
        
        obs = next_obs
        
        step_data["dones"] = dones.reshape((1, num_envs, -1))
        step_data["terminated"] = terminated.reshape((1, num_envs, -1))
        step_data["truncated"] = truncated.reshape((1, num_envs, -1))
        step_data["rewards"] = clip_rewards_fn(rewards)
        step_data["stochastic_state"] = player.stochastic_state.detach().cpu().numpy()
        step_data["recurrent_state"] = player.recurrent_state.detach().cpu().numpy()
        
        # Handle done environments
        dones_idxes = dones.nonzero()[0].tolist()
        if len(dones_idxes) > 0:
            reset_data = {}
            for k in obs_keys:
                reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
            reset_data["dones"] = np.ones((1, len(dones_idxes), 1))
            reset_data["terminated"] = np.ones((1, len(dones_idxes), 1))
            reset_data["truncated"] = np.zeros((1, len(dones_idxes), 1))
            reset_data["actions"] = np.zeros((1, len(dones_idxes), np.sum(actions_dim)))
            reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
            reset_data["is_first"] = np.zeros_like(reset_data["dones"])
            rb_initial.add(reset_data, validate_args=cfg.buffer.validate_args)
            
            step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
            step_data["dones"][:, dones_idxes] = np.zeros_like(step_data["dones"][:, dones_idxes])
            step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
            step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
            step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
            player.init_states(dones_idxes)
        
        # Imagination and error calculation
        if i > imagination_steps:
            stochastic_state = player.stochastic_state.clone()
            recurrent_state = player.recurrent_state.clone()
            
            imagined_latent_states = torch.cat((stochastic_state, recurrent_state), -1)
            rb_imagination = SequentialReplayBuffer(imagination_steps, num_envs)
            
            with torch.no_grad():
                for j in range(imagination_steps):
                    # Get imagined actions
                    actions_imag = actor(imagined_latent_states.detach())[0][0]
                    
                    # Imagination step
                    stochastic_state, recurrent_state = world_model.rssm.imagination(
                        stochastic_state, recurrent_state, actions_imag
                    )
                    
                    imagined_latent_states = torch.cat((stochastic_state.view(1, 1, -1), recurrent_state), -1)
                    stochastic_state = stochastic_state.view(1, 1, -1)
                    
                    # Reconstruct observations and rewards
                    rec_obs = world_model.observation_model(imagined_latent_states)
                    predicted_rewards = TwoHotEncodingDistribution(
                        world_model.reward_model(imagined_latent_states), dims=1
                    ).mean
                    rewards_list[i, j] = predicted_rewards[:, :, 0][0].detach().cpu().numpy()[0]
                    
                    # Store imagined observations
                    step_data_imag = {}
                    step_data_imag["vector"] = rec_obs["vector"].unsqueeze(0).detach().cpu().numpy()
                    step_data_imag["actions"] = actions_imag.unsqueeze(0).detach().cpu().numpy()
                    rb_imagination.add(step_data_imag)
            
            imags.append(rb_imagination["vector"][:, 0, 0])
            
            # Calculate prediction error
            actual_obs = rb_initial["vector"][i - imagination_steps + 1:i + 1, 0]
            imagined_obs = rb_imagination["vector"][:, 0, 0]
            
            if len(actual_obs) == imagination_steps:
                current_error = np.mean(np.abs(actual_obs - imagined_obs))
                
                # Collect errors after stabilization
                if i >= stabilization_steps:
                    recent_errors.append(current_error)
                    if len(recent_errors) > error_window_size:
                        recent_errors = recent_errors[-error_window_size:]
                
                # Calculate average recent error
                avg_recent_error = np.mean(recent_errors) if recent_errors else current_error
                avg_recent_errors_timeline.append(avg_recent_error)
                
                # Track rewards
                current_reward = np.mean(step_data["rewards"])
                if i >= stabilization_steps:
                    recent_rewards.append(current_reward)
                    if len(recent_rewards) > error_window_size:
                        recent_rewards = recent_rewards[-error_window_size:]
                
                avg_recent_reward = np.mean(recent_rewards) if recent_rewards else current_reward
                avg_recent_rewards_timeline.append(avg_recent_reward)
        
        # Progress logging
        if i % 500 == 0:
            avg_err = avg_recent_errors_timeline[-1] if avg_recent_errors_timeline else 0
            avg_rew = avg_recent_rewards_timeline[-1] if avg_recent_rewards_timeline else 0
            print(f"Step {i}/{eval_steps}, Avg Error: {avg_err:.6f}, Avg Reward: {avg_rew:.6f}")
    
    print("[INFO] Evaluation complete. Processing results...")
    
    # Process results
    individual_errors = []
    individual_measurements = []
    individual_imaginations = []
    
    for idx in range(len(imags)):
        start_idx = imagination_steps + idx
        end_idx = start_idx + imagination_steps
        actual = rb_initial["vector"][start_idx:end_idx, 0]
        
        if len(actual) == imagination_steps:
            error = np.abs(actual - imags[idx])
            imag_errors.append(np.mean(error))
            individual_errors.append(error)
            individual_imaginations.append(imags[idx])
            individual_measurements.append(actual)
    
    # Process rewards
    true_rewards = np.array(true_rewards_list)
    rewards_list = rewards_list[:len(true_rewards)]
    avg_predicted_rewards = np.mean(rewards_list, axis=1)
    
    # Save results
    save_data = {
        "imag_errors": np.array(imag_errors),
        "avg_recent_errors_timeline": np.array(avg_recent_errors_timeline),
        "avg_recent_rewards_timeline": np.array(avg_recent_rewards_timeline),
        "true_rewards": true_rewards,
        "predicted_rewards": avg_predicted_rewards,
        "reward_prediction_error": np.abs(true_rewards - avg_predicted_rewards),
        "individual_errors": np.array(individual_errors) if individual_errors else np.array([]),
        "individual_measurements": np.array(individual_measurements) if individual_measurements else np.array([]),
        "individual_imaginations": np.array(individual_imaginations) if individual_imaginations else np.array([]),
        "config": {
            "task": task,
            "seed": seed,
            "eval_steps": eval_steps,
            "imagination_steps": imagination_steps,
            "checkpoint": str(checkpoint_path),
        }
    }
    
    results_path = output_dir / "evaluation_results.npy"
    np.save(results_path, save_data)
    print(f"[INFO] Results saved to: {results_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Task: {task}")
    print(f"Total steps: {eval_steps}")
    print(f"Imagination horizon: {imagination_steps}")
    print(f"\nPrediction Error Statistics:")
    if imag_errors:
        print(f"  Mean: {np.mean(imag_errors):.6f}")
        print(f"  Std:  {np.std(imag_errors):.6f}")
        print(f"  Min:  {np.min(imag_errors):.6f}")
        print(f"  Max:  {np.max(imag_errors):.6f}")
    print(f"\nReward Statistics:")
    print(f"  Mean True Reward:      {np.mean(true_rewards):.6f}")
    print(f"  Mean Predicted Reward: {np.mean(avg_predicted_rewards):.6f}")
    print(f"  Reward Prediction MAE: {np.mean(np.abs(true_rewards - avg_predicted_rewards)):.6f}")
    print("="*60)
    
    # Generate plots if not disabled
    if not args_cli.no_plots:
        try:
            import matplotlib.pyplot as plt
            
            print("[INFO] Generating plots...")
            
            # Plot 1: Imagination errors over time
            if imag_errors:
                plt.figure(figsize=(12, 4))
                plt.plot(imag_errors, color='red', alpha=0.7)
                plt.xlabel('Step')
                plt.ylabel('Mean Prediction Error')
                plt.title('World Model Prediction Error Over Time')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "prediction_errors.png", dpi=150)
                plt.close()
            
            # Plot 2: Average error timeline
            if avg_recent_errors_timeline:
                plt.figure(figsize=(12, 4))
                plt.plot(avg_recent_errors_timeline, color='blue', linewidth=2)
                plt.xlabel('Step')
                plt.ylabel('Average Recent Error')
                plt.title('Rolling Average Prediction Error')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "avg_error_timeline.png", dpi=150)
                plt.close()
            
            # Plot 3: Rewards comparison
            plt.figure(figsize=(12, 4))
            plt.plot(true_rewards, label='True Rewards', color='blue', alpha=0.7)
            plt.plot(avg_predicted_rewards, label='Predicted Rewards', color='green', alpha=0.7, linestyle='--')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('True vs Predicted Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "rewards_comparison.png", dpi=150)
            plt.close()
            
            # Plot 4: Reward prediction error
            plt.figure(figsize=(12, 4))
            plt.plot(np.abs(true_rewards - avg_predicted_rewards), color='orange', alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('Absolute Error')
            plt.title('Reward Prediction Error')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "reward_prediction_error.png", dpi=150)
            plt.close()
            
            # Plot 5: Combined error and reward timeline
            if avg_recent_errors_timeline and avg_recent_rewards_timeline:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
                
                ax1.plot(avg_recent_errors_timeline, color='red', linewidth=2)
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Average Recent Error')
                ax1.set_title('Prediction Error Timeline')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(avg_recent_rewards_timeline, color='blue', linewidth=2)
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Average Recent Reward')
                ax2.set_title('Reward Timeline')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "combined_timeline.png", dpi=150)
                plt.close()
            
            # Plot 6: Individual state dimensions (if available)
            if individual_measurements and individual_imaginations:
                data_measure = np.array(individual_measurements)
                data_imag = np.array(individual_imaginations)
                
                if len(data_measure.shape) == 3 and data_measure.shape[2] > 0:
                    num_dims = min(data_measure.shape[2], 12)  # Limit to 12 dimensions
                    
                    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
                    axes = axes.flatten()
                    
                    # Average over imagination steps
                    data_measure_avg = np.mean(data_measure, axis=1)
                    data_imag_avg = np.mean(data_imag, axis=1)
                    
                    for dim in range(num_dims):
                        axes[dim].plot(data_measure_avg[:, dim], label='True', color='blue', alpha=0.7)
                        axes[dim].plot(data_imag_avg[:, dim], label='Imagined', color='green', linestyle='--', alpha=0.7)
                        axes[dim].set_xlabel('Step')
                        axes[dim].set_ylabel(f'Dim {dim}')
                        axes[dim].legend(fontsize=8)
                        axes[dim].grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for dim in range(num_dims, len(axes)):
                        axes[dim].axis('off')
                    
                    plt.suptitle('Individual State Dimensions: True vs Imagined', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(output_dir / "state_dimensions.png", dpi=150)
                    plt.close()
            
            print(f"[INFO] Plots saved to: {output_dir}")
            
        except ImportError:
            print("[WARNING] matplotlib not available, skipping plots")
        except Exception as e:
            print(f"[WARNING] Error generating plots: {e}")
    
    # Cleanup
    envs.close()
    print("[INFO] Evaluation finished!")


if __name__ == "__main__":
    main()
    simulation_app.close()

