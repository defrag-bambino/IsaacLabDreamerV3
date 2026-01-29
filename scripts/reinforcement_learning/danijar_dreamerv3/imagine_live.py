# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained DreamerV3 agent with TRUE imagination-based error analysis.

This script runs the agent in the environment while simultaneously performing
world model imagination (rollouts) to predict future states. It compares the
imagined states with actual states to compute prediction errors, which can
reveal how well the world model has learned the environment dynamics.

The imagination uses the actual RSSM world model to:
1. Encode current observation into latent state
2. Imagine forward N steps using the learned dynamics
3. Decode imagined latent states to predicted observations
4. Predict rewards from imagined states
5. Compare with actual observations/rewards

At the midpoint of evaluation, the robot is modified (a joint is locked) to
observe how the world model prediction error changes when encountering
out-of-distribution dynamics.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
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
parser = argparse.ArgumentParser(description="Evaluate DreamerV3 agent with TRUE imagination-based error analysis.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1 for analysis).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (optional, overrides config).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of evaluation steps.")
parser.add_argument("--imagination_horizon", type=int, default=15, help="Number of steps to imagine forward.")
parser.add_argument("--imagination_interval", type=int, default=50, help="Run imagination every N steps.")
parser.add_argument("--stabilization_steps", type=int, default=500, help="Steps before starting error collection.")
parser.add_argument("--error_window_size", type=int, default=50, help="Window size for averaging errors.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (default: checkpoint_dir/imagination_analysis).")
parser.add_argument("--no_plots", action="store_true", default=False, help="Disable plotting (for headless).")
parser.add_argument("--modify_robot", action="store_true", default=True, help="Modify robot at midpoint to test OOD detection.")
parser.add_argument("--modify_joint", type=int, default=8, help="Joint index to lock when --modify_robot is enabled.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, other_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


def main():
    """Evaluate DreamerV3 agent with TRUE imagination-based error analysis."""
    import os
    import numpy as np
    import ruamel.yaml as yaml
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial as bind
    
    # Import after Isaac Sim is initialized
    import elements
    import embodied
    import embodied.jax.nets as nn
    from dreamerv3.agent import Agent
    from isaac_env import IsaacBatchedEnv
    
    print("\n" + "="*70)
    print("DreamerV3 TRUE Imagination Analysis")
    print("="*70)
    
    # Expand checkpoint path
    checkpoint_path = pathlib.Path(os.path.expanduser(args_cli.checkpoint))
    
    # Handle both formats: user can pass either run dir or ckpt dir
    if checkpoint_path.name == 'ckpt':
        checkpoint_dir = checkpoint_path.parent
        ckpt_dir = checkpoint_path
    else:
        checkpoint_dir = checkpoint_path
        ckpt_dir = checkpoint_path / 'ckpt'
    
    # Setup output directory
    if args_cli.output_dir:
        output_dir = pathlib.Path(args_cli.output_dir)
    else:
        output_dir = checkpoint_dir / "imagination_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    
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
        task = config.task
        if task.startswith('isaac_'):
            task = task[6:]
        print(f"[INFO] Using task from checkpoint: {task}")
    else:
        print(f"[INFO] Using task from CLI: {task}")
    
    num_envs = args_cli.num_envs
    eval_steps = args_cli.eval_steps
    imagination_horizon = args_cli.imagination_horizon
    imagination_interval = args_cli.imagination_interval
    stabilization_steps = args_cli.stabilization_steps
    error_window_size = args_cli.error_window_size
    
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Num envs: {num_envs}")
    print(f"  Eval steps: {eval_steps}")
    print(f"  Imagination horizon: {imagination_horizon}")
    print(f"  Imagination interval: {imagination_interval}")
    print(f"  Stabilization steps: {stabilization_steps}")
    print(f"  Modify robot: {args_cli.modify_robot}")
    print("="*70 + "\n")
    
    # Create environment
    print("[INFO] Creating environment...")
    env = IsaacBatchedEnv(
        task=task,
        num_envs=num_envs,
        device='cuda:0',
        seed=args_cli.seed,
        obs_key='vector',
        act_key='action',
    )
    
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
    print(f"[INFO] Loading checkpoint from {ckpt_dir}...")
    if not ckpt_dir.exists():
        print(f"[ERROR] Checkpoint directory not found at {ckpt_dir}")
        return
    
    cp = elements.Checkpoint(ckpt_dir)
    cp.agent = agent
    cp.load()
    print(f"[INFO] Checkpoint loaded successfully")
    
    # Initialize policy
    print(f"[INFO] Initializing policy for {num_envs} environments...")
    carry = agent.init_policy(num_envs)
    
    # Get initial observations
    obs = env.reset()
    obs_dim = obs['vector'].shape[-1]
    print(f"[INFO] Initial obs shape: {obs['vector'].shape}")
    print(f"[INFO] Observation dimension: {obs_dim}")
    
    # =========================================================================
    # Create imagination function using the agent's world model
    # =========================================================================
    print("[INFO] Setting up imagination functions...")
    
    # Debug: Check what attributes the agent has
    print(f"[DEBUG] Agent type: {type(agent)}")
    print(f"[DEBUG] Agent attributes: {[a for a in dir(agent) if not a.startswith('_')]}")
    
    # Try to access world model components
    # The embodied.jax.Agent wraps the actual model, so we need agent.model
    inner_agent = None
    if hasattr(agent, 'model'):
        inner_agent = agent.model
        print(f"[DEBUG] Inner model type: {type(inner_agent)}")
        print(f"[DEBUG] Inner model attributes: {[a for a in dir(inner_agent) if not a.startswith('_')][:20]}...")
    
    # Try different access paths
    if inner_agent is not None and hasattr(inner_agent, 'dyn'):
        dyn_module = inner_agent.dyn
        enc_module = inner_agent.enc
        dec_module = inner_agent.dec
        rew_module = inner_agent.rew
        feat2tensor_fn = inner_agent.feat2tensor
        print(f"[INFO] Accessed world model via agent.model.dyn/enc/dec/rew")
    elif inner_agent is not None and hasattr(inner_agent, 'modules') and len(inner_agent.modules) >= 4:
        dyn_module = inner_agent.modules[0]
        enc_module = inner_agent.modules[1]
        dec_module = inner_agent.modules[2]
        rew_module = inner_agent.modules[3]
        feat2tensor_fn = inner_agent.feat2tensor
        print(f"[INFO] Accessed world model via agent.model.modules")
    elif hasattr(agent, 'dyn'):
        dyn_module = agent.dyn
        enc_module = agent.enc
        dec_module = agent.dec
        rew_module = agent.rew
        feat2tensor_fn = agent.feat2tensor
        print(f"[INFO] Accessed world model via agent.dyn/enc/dec/rew")
    else:
        print(f"[WARNING] Cannot access world model components - will use fallback")
        dyn_module = None
        dec_module = None
        rew_module = None
        feat2tensor_fn = None
    
    # Helper functions
    f32 = jnp.float32
    
    # Get agent params for world model calls
    agent_params = agent.params
    print(f"[DEBUG] Agent params available: {agent_params is not None}")
    
    # =========================================================================
    # TRUE World Model Imagination Function
    # =========================================================================
    def imagine_with_world_model(dyn_carry, dec_carry, actions_sequence, horizon):
        """
        Perform TRUE imagination rollout using the world model.
        
        This runs the RSSM dynamics forward without seeing actual observations,
        then decodes the imagined latent states to predicted observations.
        
        Args:
            dyn_carry: Current dynamics state {'deter': [B, deter_dim], 'stoch': [B, stoch, classes]}
            dec_carry: Current decoder state (can be empty {})
            actions_sequence: Actions to imagine with {'action': [B, H, act_dim]}
            horizon: Number of steps to imagine
            
        Returns:
            imagined_obs: Dict of predicted observations
            imagined_rewards: Predicted rewards [B, H]
        """
        if dyn_module is None:
            raise RuntimeError("World model not accessible")
        
        # Define the imagination function that will be wrapped
        def _do_imagination():
            H = horizon
            B = dyn_carry['deter'].shape[0]
            
            # Run imagination through the RSSM
            # This predicts future latent states without seeing actual observations
            _, imgfeat, imgact = dyn_module.imagine(
                dyn_carry, actions_sequence, H, training=False)
            
            # Decode imagined latent features to observations
            reset = jnp.zeros((B, H), dtype=bool)
            _, _, recons = dec_module(dec_carry, imgfeat, reset, training=False)
            
            # Predict rewards from imagined features
            feat_tensor = feat2tensor_fn(imgfeat)
            rew_dist = rew_module(feat_tensor, bdims=2)
            imagined_rewards = rew_dist.pred()
            
            # Extract observation predictions
            imagined_obs = {key: dist.pred() for key, dist in recons.items()}
            
            return imagined_obs, imagined_rewards
        
        # Wrap with nj.pure to handle ninjax state
        pure_fn = nj.pure(_do_imagination)
        
        # Create a random seed for ninjax
        rng_seed = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        # Run with agent params, allowing JAX transfers
        # Must pass seed= for nj.seed() to work inside the function
        # Use create=True to allow accessing params that may have different structure
        with jax.transfer_guard('allow'):
            _, (imagined_obs, imagined_rewards) = pure_fn(
                agent_params, seed=rng_seed, create=True, modify=True, ignore=True)
        
        return imagined_obs, imagined_rewards
    
    # Test if world model imagination works
    imagination_available = False
    if dyn_module is not None:
        print("[INFO] Testing world model imagination...")
        # Wrap entire test in transfer_guard to allow all JAX transfers
        with jax.transfer_guard('allow'):
            try:
                # Create test inputs using module's initial state
                test_dyn_carry = dyn_module.initial(1)
                test_dec_carry = dec_module.initial(1) if hasattr(dec_module, 'initial') else {}
                test_actions = {'action': jnp.zeros((1, 5, act_space['action'].shape[0]))}

                print(f"[DEBUG] Test dyn_carry shapes: deter={test_dyn_carry['deter'].shape}, stoch={test_dyn_carry['stoch'].shape}")
                print(f"[DEBUG] Test actions shape: {test_actions['action'].shape}")

                test_obs, test_rew = imagine_with_world_model(
                    test_dyn_carry, test_dec_carry, test_actions, 5)

                imagination_available = True
                print("[INFO] ✓ World model imagination is WORKING!")
                print(f"[DEBUG] Test output shapes - obs keys: {list(test_obs.keys())}, rew: {test_rew.shape}")
            except Exception as e:
                print(f"[WARNING] World model imagination test failed: {e}")
                import traceback
                traceback.print_exc()
                print("[INFO] Will use trajectory analysis as fallback")
                imagination_available = False
    
    if not imagination_available:
        print("[INFO] Using trajectory analysis (observation dynamics) as fallback")
    
    # Storage for analysis
    action_history = []
    obs_history = []
    reward_history = []
    latent_history = []  # Store latent states for world model imagination
    
    # Storage for analysis
    imagination_results = []  # List of (step, imagined_obs, actual_obs, error) tuples
    prediction_errors_obs = []  # Observation prediction errors
    prediction_errors_rew = []  # Reward prediction errors
    avg_errors_timeline = []
    avg_rewards_timeline = []
    recent_errors = []
    recent_rewards = []
    
    modification_step = eval_steps // 2 if args_cli.modify_robot else None
    
    # Helper to safely convert JAX arrays to numpy (handles bfloat16 and GPU transfers)
    def safe_to_numpy(x):
        """Convert JAX array to numpy, handling bfloat16 and device transfers."""
        if isinstance(x, (np.ndarray, float, int)):
            return np.array(x)
        x_host = jax.device_get(x)  # Transfer to host
        if hasattr(x_host, 'dtype') and str(x_host.dtype) == 'bfloat16':
            return np.array(x_host, dtype=np.float32)
        return np.array(x_host)
    
    print(f"\n[INFO] Starting evaluation for {eval_steps} steps...")
    print(f"[INFO] Imagination will run every {imagination_interval} steps")
    print(f"[INFO] Each imagination looks {imagination_horizon} steps ahead")
    
    for step_idx in range(eval_steps):
        # Robot modification at midpoint
        if args_cli.modify_robot and step_idx == modification_step:
            print(f"\n[INFO] Step {step_idx}: Modifying robot...")
            try:
                # Access the underlying Isaac Lab environment
                isaac_env = env._env.unwrapped
                robot = isaac_env.scene["robot"]
                
                print(f"  Joint names: {robot.data.joint_names}")
                print(f"  Locking joint {args_cli.modify_joint}...")
                
                import torch
                robot.write_joint_position_limit_to_sim(
                    torch.tensor([[[.0, .01]]], device=robot.device),
                    joint_ids=[args_cli.modify_joint]
                )
                print(f"  - Joint {args_cli.modify_joint} locked")
            except Exception as e:
                print(f"  [WARNING] Could not modify robot: {e}")
        
        # Store current observation
        current_obs_vec = obs['vector'].copy()
        obs_history.append(current_obs_vec[0])  # Store first env's obs
        
        # Store current latent state for world model imagination
        if imagination_available:
            enc_carry, dyn_carry, dec_carry, prevact = carry
            with jax.transfer_guard('allow'):
                # Debug: check dyn_carry structure on first step
                if step_idx == 0:
                    print(f"[DEBUG] dyn_carry type: {type(dyn_carry)}")
                    print(f"[DEBUG] dyn_carry keys: {dyn_carry.keys() if hasattr(dyn_carry, 'keys') else 'N/A'}")
                    if isinstance(dyn_carry, dict):
                        for k, v in dyn_carry.items():
                            print(f"[DEBUG] dyn_carry['{k}'] type: {type(v)}")
                            if isinstance(v, list) and len(v) > 0:
                                print(f"[DEBUG] dyn_carry['{k}'][0] type: {type(v[0])}, shape: {v[0].shape if hasattr(v[0], 'shape') else 'N/A'}")
                
                # Convert sharded list format to arrays
                # The agent returns lists containing single arrays due to sharding
                def unpack_sharded(x):
                    if isinstance(x, list):
                        # Sharded agent returns [array] - extract the single element
                        if len(x) == 1:
                            return x[0]
                        else:
                            return jnp.concatenate(x, axis=0)
                    return x
                
                stored_dyn = {k: unpack_sharded(v) for k, v in dyn_carry.items()}
                stored_dec = {k: unpack_sharded(v) for k, v in dec_carry.items()} if dec_carry else {}
                
                latent_history.append({
                    'dyn_carry': stored_dyn,
                    'dec_carry': stored_dec,
                })
        
        # Get action from policy
        carry, acts, policy_outs = agent.policy(carry, obs, mode='eval')
        acts['reset'] = obs['is_last'].copy()
        
        # Store action
        action_vec = acts.get('action', np.zeros((num_envs, act_space['action'].shape[0])))
        action_history.append(action_vec[0])
        
        # Step environment
        obs = env.step(acts)
        
        # Track rewards
        reward = float(obs['reward'][0])
        reward_history.append(reward)
        
        # =====================================================================
        # Perform TRUE imagination at intervals
        # =====================================================================
        if step_idx >= stabilization_steps and step_idx % imagination_interval == 0:
            # We have enough history to do imagination
            start_step = step_idx - imagination_horizon
            if start_step >= 0 and start_step + imagination_horizon <= len(obs_history):
                try:
                    # Get actual observations and rewards over the horizon
                    actual_obs = np.array(obs_history[start_step + 1:start_step + 1 + imagination_horizon])
                    actual_rewards = np.array(reward_history[start_step:start_step + imagination_horizon])
                    
                    if imagination_available and len(latent_history) > start_step:
                        # =====================================================
                        # TRUE WORLD MODEL IMAGINATION
                        # =====================================================
                        with jax.transfer_guard('allow'):
                            # Get latent state from start of horizon
                            past_latent = latent_history[start_step]
                            past_dyn_carry = past_latent['dyn_carry']
                            past_dec_carry = past_latent['dec_carry']

                            # Add batch dimension to stored states (they were saved without batch dim)
                            past_dyn_carry = {k: v[None, ...] for k, v in past_dyn_carry.items()}
                            past_dec_carry = {k: v[None, ...] for k, v in past_dec_carry.items()}

                            # Debug: Check shapes before imagination
                            if step_idx == 500:  # Only on first imagination run
                                print(f"[DEBUG] past_dyn_carry['deter'] shape: {past_dyn_carry['deter'].shape}")
                                print(f"[DEBUG] past_dyn_carry['stoch'] shape: {past_dyn_carry['stoch'].shape}")
                                print(f"[DEBUG] Imagination horizon: {imagination_horizon}")
                                print(f"[DEBUG] Action space: {act_space['action'].shape}")

                            # Get actions that were taken
                            actions_np = np.array(action_history[start_step:start_step + imagination_horizon])
                            actions_jax = {'action': jax.device_put(actions_np[None, ...])}

                            # Run world model imagination
                            imagined_obs_dict, imagined_rewards = imagine_with_world_model(
                                past_dyn_carry, past_dec_carry,
                                actions_jax, imagination_horizon
                            )
                            
                            # Extract imagined observations
                            if 'vector' in imagined_obs_dict:
                                imag_obs = safe_to_numpy(imagined_obs_dict['vector'][0])
                            else:
                                imag_obs = np.concatenate(
                                    [safe_to_numpy(v[0]) for v in imagined_obs_dict.values()], axis=-1)
                            
                            imag_rew = safe_to_numpy(imagined_rewards[0])
                            
                            # Compute TRUE prediction errors
                            obs_error = np.mean(np.abs(imag_obs - actual_obs))
                            rew_error = np.mean(np.abs(imag_rew - actual_rewards))
                            
                            imagination_results.append({
                                'step': step_idx,
                                'start_step': start_step,
                                'obs_error': obs_error,
                                'rew_error': rew_error,
                                'imagined_obs': imag_obs,
                                'actual_obs': actual_obs,
                                'imagined_rewards': imag_rew,
                                'actual_rewards': actual_rewards,
                                'method': 'world_model',
                            })
                    else:
                        # =====================================================
                        # FALLBACK: Trajectory Analysis
                        # =====================================================
                        obs_diffs = np.diff(np.array(obs_history[start_step:start_step + imagination_horizon]), axis=0)
                        obs_error = np.mean(np.abs(obs_diffs))
                        rew_error = np.std(actual_rewards)
                        
                        imagination_results.append({
                            'step': step_idx,
                            'start_step': start_step,
                            'obs_error': obs_error,
                            'rew_error': rew_error,
                            'actual_obs': actual_obs,
                            'actual_rewards': actual_rewards,
                            'method': 'trajectory_analysis',
                        })
                    
                    prediction_errors_obs.append(obs_error)
                    prediction_errors_rew.append(rew_error)
                    
                except Exception as e:
                    if step_idx < stabilization_steps + 100:
                        print(f"  [WARNING] Imagination failed at step {step_idx}: {e}")
                    # Use simple fallback
                    try:
                        obs_diffs = np.diff(np.array(obs_history[start_step:start_step + imagination_horizon]), axis=0)
                        obs_error = np.mean(np.abs(obs_diffs))
                        prediction_errors_obs.append(obs_error)
                        prediction_errors_rew.append(0.0)
                    except:
                        pass
        
        # Track rolling averages
        if len(prediction_errors_obs) > 0:
            recent_errors.append(prediction_errors_obs[-1])
            if len(recent_errors) > error_window_size:
                recent_errors = recent_errors[-error_window_size:]
            
            recent_rewards.append(reward)
            if len(recent_rewards) > error_window_size:
                recent_rewards = recent_rewards[-error_window_size:]
            
            avg_error = np.mean(recent_errors)
            avg_reward = np.mean(recent_rewards)
            
            avg_errors_timeline.append(avg_error)
            avg_rewards_timeline.append(avg_reward)
        
        # Progress logging
        if step_idx % 500 == 0:
            avg_err = avg_errors_timeline[-1] if avg_errors_timeline else 0
            avg_rew = avg_rewards_timeline[-1] if avg_rewards_timeline else 0
            n_imag = len(prediction_errors_obs)
            print(f"Step {step_idx}/{eval_steps}, Avg Obs Error: {avg_err:.6f}, "
                  f"Avg Reward: {avg_rew:.4f}, Imagination runs: {n_imag}")
    
    print("\n[INFO] Evaluation complete. Processing results...")
    
    # Convert to arrays
    obs_history = np.array(obs_history)
    reward_history = np.array(reward_history)
    prediction_errors_obs = np.array(prediction_errors_obs)
    prediction_errors_rew = np.array(prediction_errors_rew)
    avg_errors_timeline = np.array(avg_errors_timeline)
    avg_rewards_timeline = np.array(avg_rewards_timeline)
    
    # Save results
    save_data = {
        "prediction_errors_obs": prediction_errors_obs,
        "prediction_errors_rew": prediction_errors_rew,
        "avg_errors_timeline": avg_errors_timeline,
        "avg_rewards_timeline": avg_rewards_timeline,
        "reward_history": reward_history,
        "obs_history": obs_history,
        "config": {
            "task": task,
            "seed": args_cli.seed,
            "eval_steps": eval_steps,
            "imagination_horizon": imagination_horizon,
            "imagination_interval": imagination_interval,
            "stabilization_steps": stabilization_steps,
            "checkpoint": str(checkpoint_dir),
            "modify_robot": args_cli.modify_robot,
            "modification_step": modification_step,
        }
    }
    
    # Save detailed imagination results separately
    if imagination_results:
        detailed_results = {
            'steps': np.array([r['step'] for r in imagination_results]),
            'obs_errors': np.array([r['obs_error'] for r in imagination_results]),
            'rew_errors': np.array([r['rew_error'] for r in imagination_results]),
        }
        np.savez(output_dir / "imagination_detailed.npz", **detailed_results)
    
    results_path = output_dir / "imagination_results.npz"
    np.savez(results_path, **save_data)
    print(f"[INFO] Results saved to: {results_path}")
    
    # Print summary statistics
    # Determine which method was used
    methods_used = set(r.get('method', 'unknown') for r in imagination_results)
    method_str = 'World Model Imagination' if 'world_model' in methods_used else 'Trajectory Analysis'
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Task: {task}")
    print(f"Total steps: {eval_steps}")
    print(f"Imagination horizon: {imagination_horizon} steps")
    print(f"Number of imagination runs: {len(prediction_errors_obs)}")
    print(f"Method: {method_str}")
    
    if len(prediction_errors_obs) > 0:
        print(f"\nObservation Prediction Error Statistics:")
        print(f"  Mean: {np.mean(prediction_errors_obs):.6f}")
        print(f"  Std:  {np.std(prediction_errors_obs):.6f}")
        print(f"  Min:  {np.min(prediction_errors_obs):.6f}")
        print(f"  Max:  {np.max(prediction_errors_obs):.6f}")
        
        if len(prediction_errors_rew) > 0 and np.any(prediction_errors_rew):
            print(f"\nReward Prediction Error Statistics:")
            print(f"  Mean: {np.mean(prediction_errors_rew):.6f}")
            print(f"  Std:  {np.std(prediction_errors_rew):.6f}")
        
        if args_cli.modify_robot and modification_step:
            # Find imagination runs before/after modification
            imag_steps = np.array([r['step'] for r in imagination_results]) if imagination_results else np.array([])
            
            if len(imag_steps) > 0:
                before_mask = imag_steps < modification_step
                after_mask = imag_steps >= modification_step
                
                errors_before = prediction_errors_obs[before_mask]
                errors_after = prediction_errors_obs[after_mask]
                
                if len(errors_before) > 0 and len(errors_after) > 0:
                    print(f"\nBefore Robot Modification ({np.sum(before_mask)} runs):")
                    print(f"  Mean Prediction Error: {np.mean(errors_before):.6f}")
                    print(f"  Std Prediction Error:  {np.std(errors_before):.6f}")
                    print(f"\nAfter Robot Modification ({np.sum(after_mask)} runs):")
                    print(f"  Mean Prediction Error: {np.mean(errors_after):.6f}")
                    print(f"  Std Prediction Error:  {np.std(errors_after):.6f}")
                    
                    change = ((np.mean(errors_after) / np.mean(errors_before)) - 1) * 100
                    print(f"\n  Change: {change:+.1f}%")
                    if change > 20:
                        print("  [!] Significant INCREASE in prediction error after modification!")
                        print("      The world model detected out-of-distribution dynamics.")
    
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(reward_history):.6f}")
    print(f"  Std:  {np.std(reward_history):.6f}")
    print("="*70)
    
    # Generate plots if not disabled
    if not args_cli.no_plots:
        try:
            import matplotlib.pyplot as plt
            
            print("\n[INFO] Generating plots...")
            
            # Plot 1: Observation Prediction errors over time
            if len(prediction_errors_obs) > 0:
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                
                # Subplot 1: Observation errors
                ax1 = axes[0]
                imag_steps = np.array([r['step'] for r in imagination_results]) if imagination_results else np.arange(len(prediction_errors_obs))
                ax1.plot(imag_steps, prediction_errors_obs, 'o-', color='red', 
                         alpha=0.7, markersize=3, linewidth=1, label='Obs Prediction Error')
                
                if args_cli.modify_robot and modification_step:
                    ax1.axvline(x=modification_step, color='blue', linestyle='--', 
                                linewidth=2, label='Robot Modified')
                
                ax1.set_xlabel('Env Step')
                ax1.set_ylabel('Observation Prediction Error')
                ax1.set_title(f'World Model Prediction Error ({method_str})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Subplot 2: Reward prediction errors
                ax2 = axes[1]
                if len(prediction_errors_rew) > 0:
                    ax2.plot(imag_steps[:len(prediction_errors_rew)], prediction_errors_rew, 
                             'o-', color='green', alpha=0.7, markersize=3, linewidth=1, 
                             label='Reward Prediction Error')
                    
                    if args_cli.modify_robot and modification_step:
                        ax2.axvline(x=modification_step, color='blue', linestyle='--', 
                                    linewidth=2, label='Robot Modified')
                    
                ax2.set_xlabel('Env Step')
                ax2.set_ylabel('Reward Prediction Error')
                ax2.set_title('World Model Reward Prediction Error')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "prediction_errors.png", dpi=150)
                plt.close()
                print(f"  Saved: prediction_errors.png")
            
            # Plot 2: Rewards over time
            plt.figure(figsize=(14, 5))
            plt.plot(reward_history, color='blue', alpha=0.5, linewidth=0.5, label='Raw')
            if len(avg_rewards_timeline) > 0:
                # Calculate offset for alignment
                offset = eval_steps - len(avg_rewards_timeline)
                x_values = np.arange(offset, eval_steps)
                # Ensure lengths match
                min_len = min(len(x_values), len(avg_rewards_timeline))
                plt.plot(x_values[:min_len], avg_rewards_timeline[:min_len],
                         color='darkblue', linewidth=2, label='Rolling Avg')
            
            if args_cli.modify_robot and modification_step:
                plt.axvline(x=modification_step, color='red', linestyle='--',
                            linewidth=2, label='Robot Modified')
            
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('Rewards Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "rewards.png", dpi=150)
            plt.close()
            print(f"  Saved: rewards.png")
            
            # Plot 3: Detailed imagination comparison (last few runs)
            if len(imagination_results) >= 3:
                n_show = min(4, len(imagination_results))
                fig, axes = plt.subplots(n_show, 2, figsize=(14, 3*n_show))
                
                # Show results from different phases
                indices = [0, len(imagination_results)//3, 
                          2*len(imagination_results)//3, len(imagination_results)-1][:n_show]
                
                for i, idx in enumerate(indices):
                    result = imagination_results[idx]
                    
                    # Observation comparison (first few dims)
                    ax1 = axes[i, 0] if n_show > 1 else axes[0]
                    n_dims = min(3, result['imagined_obs'].shape[1])
                    for d in range(n_dims):
                        ax1.plot(result['imagined_obs'][:, d], '--', 
                                 label=f'Imag dim {d}', alpha=0.8)
                        ax1.plot(result['actual_obs'][:, d], '-', 
                                 label=f'Actual dim {d}', alpha=0.8)
                    ax1.set_xlabel('Imagination Step')
                    ax1.set_ylabel('Observation Value')
                    ax1.set_title(f'Step {result["step"]}: Obs Error = {result["obs_error"]:.4f}')
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    
                    # Reward comparison
                    ax2 = axes[i, 1] if n_show > 1 else axes[1]
                    ax2.plot(result['imagined_rewards'], '--', color='green', 
                             label='Imagined', linewidth=2)
                    ax2.plot(result['actual_rewards'], '-', color='blue', 
                             label='Actual', linewidth=2)
                    ax2.set_xlabel('Imagination Step')
                    ax2.set_ylabel('Reward')
                    ax2.set_title(f'Step {result["step"]}: Rew Error = {result["rew_error"]:.4f}')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "imagination_comparison.png", dpi=150)
                plt.close()
                print(f"  Saved: imagination_comparison.png")
            
            # Plot 4: Combined error and reward timeline with modification marker
            if len(avg_errors_timeline) > 0 and len(avg_rewards_timeline) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                
                ax1.plot(avg_errors_timeline, color='red', linewidth=2)
                ax1.set_ylabel('Prediction Error (Rolling Avg)')
                ax1.set_title('World Model Prediction Error Timeline')
                ax1.grid(True, alpha=0.3)
                
                if args_cli.modify_robot and modification_step:
                    # Calculate where modification appears in the rolling avg timeline
                    mod_idx_in_timeline = modification_step - (eval_steps - len(avg_errors_timeline))
                    if 0 < mod_idx_in_timeline < len(avg_errors_timeline):
                        ax1.axvline(x=mod_idx_in_timeline, color='blue', linestyle='--', 
                                    linewidth=2, label='Robot Modified')
                        ax1.legend()
                        ax2.axvline(x=mod_idx_in_timeline, color='blue', linestyle='--', 
                                    linewidth=2, label='Robot Modified')
                
                ax2.plot(avg_rewards_timeline, color='blue', linewidth=2)
                ax2.set_xlabel('Timeline Index')
                ax2.set_ylabel('Reward (Rolling Avg)')
                ax2.set_title('Reward Timeline')
                ax2.grid(True, alpha=0.3)
                if args_cli.modify_robot:
                    ax2.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / "combined_timeline.png", dpi=150)
                plt.close()
                print(f"  Saved: combined_timeline.png")
            
            # Plot 5: Observation state dimensions over time
            if obs_history.shape[0] > 0:
                num_dims = min(obs_history.shape[1], 12)
                fig, axes = plt.subplots(3, 4, figsize=(16, 10))
                axes = axes.flatten()
                
                for dim in range(num_dims):
                    axes[dim].plot(obs_history[:, dim], color='blue', alpha=0.7, linewidth=0.5)
                    axes[dim].set_xlabel('Step')
                    axes[dim].set_ylabel(f'Dim {dim}')
                    axes[dim].grid(True, alpha=0.3)
                    
                    if args_cli.modify_robot and modification_step:
                        axes[dim].axvline(x=modification_step, color='red', 
                                          linestyle='--', alpha=0.5)
                
                for dim in range(num_dims, len(axes)):
                    axes[dim].axis('off')
                
                plt.suptitle('Observation State Dimensions Over Time', fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / "state_dimensions.png", dpi=150)
                plt.close()
                print(f"  Saved: state_dimensions.png")
            
            print(f"\n[INFO] All plots saved to: {output_dir}")
            
        except ImportError:
            print("[WARNING] matplotlib not available, skipping plots")
        except Exception as e:
            print(f"[WARNING] Error generating plots: {e}")
            traceback.print_exc()
    
    # Cleanup
    env.close()
    print("\n[INFO] Imagination analysis finished!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Imagination analysis failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")
    finally:
        simulation_app.close()
