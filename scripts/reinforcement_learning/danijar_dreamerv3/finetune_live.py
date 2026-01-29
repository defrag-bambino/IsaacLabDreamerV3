# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to fine-tune a DreamerV3 agent after detecting out-of-distribution dynamics.

This script:
1. Loads a trained DreamerV3 checkpoint
2. Runs the agent in the environment
3. Uses world model imagination to detect prediction errors
4. At 1/4 of total steps, modifies the robot (e.g., locks a joint)
5. When prediction error increases significantly, triggers data collection
6. After collecting enough data, fine-tunes the world model
7. Monitors if fine-tuning improves prediction accuracy

This enables the agent to adapt to changes in robot dynamics (e.g., damage, wear).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import sys
import traceback
import warnings

from isaaclab.app import AppLauncher

# Filter out PyTorch RNN memory warning
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory", category=UserWarning)

# Add paths BEFORE parsing args
dreamerv3_path = pathlib.Path(__file__).parent.parent.parent.parent / "dreamerv3"
sys.path.insert(0, str(dreamerv3_path))
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir))

# add argparse arguments
parser = argparse.ArgumentParser(description="Fine-tune DreamerV3 agent after detecting OOD dynamics.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (optional, overrides config).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")

# Evaluation parameters
parser.add_argument("--eval_steps", type=int, default=20000, help="Total number of evaluation steps.")
parser.add_argument("--imagination_horizon", type=int, default=15, help="Number of steps to imagine forward.")
parser.add_argument("--imagination_interval", type=int, default=50, help="Run imagination every N steps.")
parser.add_argument("--stabilization_steps", type=int, default=500, help="Steps before starting error collection.")
parser.add_argument("--error_window_size", type=int, default=50, help="Window size for averaging errors.")

# Fine-tuning parameters
parser.add_argument("--error_threshold", type=float, default=0.1, help="Error increase ratio to trigger fine-tuning.")
parser.add_argument("--reward_threshold", type=float, default=0.1, help="Reward decrease ratio to trigger fine-tuning.")
parser.add_argument("--finetune_steps_wm", type=int, default=2000, help="Number of fine-tuning steps for world model (Phase 1).")
parser.add_argument("--finetune_steps_policy", type=int, default=8000, help="Number of fine-tuning steps for policy (Phase 2).")
parser.add_argument("--finetune_batch_size", type=int, default=16, help="Batch size for fine-tuning.")
parser.add_argument("--finetune_sequence_length", type=int, default=64, help="Sequence length for fine-tuning.")
parser.add_argument("--data_collection_steps", type=int, default=12000, help="Steps to collect data before fine-tuning.")
parser.add_argument("--target_error_reduction", type=float, default=0.5, help="Target error reduction factor.")

# Robot modification
parser.add_argument("--modify_robot", action="store_true", default=True, help="Modify robot at 1/4 of total steps.")
parser.add_argument("--modify_joint", type=int, default=7, help="Joint index to lock when --modify_robot is enabled.")

# Output
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
parser.add_argument("--no_plots", action="store_true", default=False, help="Disable plotting.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, other_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


def main():
    """Fine-tune DreamerV3 agent after detecting OOD dynamics."""
    import os
    import copy
    import numpy as np
    import ruamel.yaml as yaml
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from functools import partial as bind
    
    # Import after Isaac Sim is initialized
    import elements
    import embodied
    from dreamerv3.agent import Agent
    from isaac_env import IsaacBatchedEnv
    
    print("\n" + "="*70)
    print("DreamerV3 Fine-Tuning with OOD Detection")
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
        output_dir = checkpoint_dir / "finetune_analysis"
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
    
    # Parameters
    num_envs = args_cli.num_envs
    eval_steps = args_cli.eval_steps
    imagination_horizon = args_cli.imagination_horizon
    imagination_interval = args_cli.imagination_interval
    stabilization_steps = args_cli.stabilization_steps
    error_window_size = args_cli.error_window_size
    
    # Fine-tuning parameters - use config values to match compiled model
    error_threshold = args_cli.error_threshold
    reward_threshold = args_cli.reward_threshold
    finetune_steps_wm = args_cli.finetune_steps_wm
    finetune_steps_policy = args_cli.finetune_steps_policy
    # Use batch_size and batch_length from config to match compiled JAX model
    finetune_batch_size = config.batch_size if hasattr(config, 'batch_size') else args_cli.finetune_batch_size
    finetune_sequence_length = config.batch_length if hasattr(config, 'batch_length') else args_cli.finetune_sequence_length
    data_collection_steps = args_cli.data_collection_steps
    target_error_reduction = args_cli.target_error_reduction
    
    print(f"[INFO] Using batch_size={finetune_batch_size}, batch_length={finetune_sequence_length} from config")
    
    modification_step = eval_steps // 4 if args_cli.modify_robot else None
    
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Eval steps: {eval_steps}")
    print(f"  Imagination horizon: {imagination_horizon}")
    print(f"  Stabilization steps: {stabilization_steps}")
    print(f"  Error threshold: {error_threshold}")
    print(f"  Reward threshold: {reward_threshold}")
    print(f"  Fine-tune steps (WM): {finetune_steps_wm}")
    print(f"  Fine-tune steps (Policy): {finetune_steps_policy}")
    print(f"  Data collection steps: {data_collection_steps}")
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
    # Override episode length to 500 seconds (25000 steps at 0.02s control dt)
    env._env.unwrapped.cfg.episode_length_s = 50.0
    print(f"[INFO] Episode length set to {env._env.unwrapped.cfg.episode_length_s}s")
    
    # Apply wrappers
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    
    print(f"[INFO] Observation space: {env.obs_space}")
    print(f"[INFO] Action space: {env.act_space}")
    
    # Setup custom velocity cycling: forward, backward, right, left
    print("\n[INFO] Setting up velocity command cycling...")
    velocity_sequence = [
        (1.0, 0.0, 0.0),    # Forward
        #(-1.0, 0.0, 0.0),   # Backward
        #(0.0, -1.0, 0.0),   # Right (negative y in robot frame)
        #(0.0, 1.0, 0.0),    # Left (positive y in robot frame)
    ]
    env.set_custom_velocities(velocity_sequence)
    print("[INFO] Velocity sequence configured:")
    for i, (vx, vy, vz) in enumerate(velocity_sequence):
        direction = ["Forward", "Backward", "Right", "Left"][i]
        print(f"  {i+1}. {direction}: vx={vx:+.1f}, vy={vy:+.1f}, vz={vz:+.1f}")
    
    # Create agent
    print("[INFO] Creating agent...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(output_dir),  # Use output dir for fine-tuning logs
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
    
    # Create replay buffer for fine-tuning data collection
    # Need to include replay_context in the length (model expects batch_length + replay_context)
    replay_context = config.replay_context if hasattr(config, 'replay_context') else 1
    total_sequence_length = finetune_sequence_length + replay_context
    print(f"[INFO] Creating replay buffer for fine-tuning...")
    print(f"[INFO] Sequence length: {finetune_sequence_length} + {replay_context} context = {total_sequence_length}")
    finetune_replay = embodied.Replay(
        length=total_sequence_length,
        capacity=data_collection_steps * 2,
        directory=str(output_dir / 'finetune_replay'),
        online=True,
    )
    
    # Initialize policy
    print(f"[INFO] Initializing policy for {num_envs} environments...")
    carry = agent.init_policy(num_envs)
    
    # Initialize training (needed for fine-tuning)
    print("[INFO] Initializing training state...")
    train_state = agent.init_train(finetune_batch_size)
    
    # Get initial observations
    obs = env.reset()
    obs_dim = obs['vector'].shape[-1]
    print(f"[INFO] Initial obs shape: {obs['vector'].shape}")
    print(f"[INFO] Observation dimension: {obs_dim}")
    
    # Display current velocity commands
    current_vels = env.get_current_velocity_commands()
    if current_vels is not None:
        print(f"[INFO] Current velocity commands for {num_envs} env(s):")
        for i in range(num_envs):
            print(f"  Env {i}: vx={current_vels[i,0]:+.2f}, vy={current_vels[i,1]:+.2f}, vz={current_vels[i,2]:+.2f}")
    
    # =========================================================================
    # Setup world model imagination (same as imagine_live.py)
    # =========================================================================
    print("[INFO] Setting up imagination functions...")
    
    inner_agent = agent.model if hasattr(agent, 'model') else None
    
    if inner_agent is not None and hasattr(inner_agent, 'dyn'):
        dyn_module = inner_agent.dyn
        enc_module = inner_agent.enc
        dec_module = inner_agent.dec
        rew_module = inner_agent.rew
        feat2tensor_fn = inner_agent.feat2tensor
        print(f"[INFO] Accessed world model via agent.model.dyn/enc/dec/rew")
    else:
        print(f"[WARNING] Cannot access world model components")
        dyn_module = None
        dec_module = None
        rew_module = None
        feat2tensor_fn = None
    
    # Note: Don't capture agent.params here - access it dynamically in the function
    # so it uses updated params after fine-tuning
    
    def imagine_with_world_model(dyn_carry, dec_carry, actions_sequence, horizon):
        """Perform TRUE imagination rollout using the world model."""
        if dyn_module is None:
            raise RuntimeError("World model not accessible")
        
        def _do_imagination():
            H = horizon
            B = dyn_carry['deter'].shape[0]
            _, imgfeat, imgact = dyn_module.imagine(
                dyn_carry, actions_sequence, H, training=False)
            reset = jnp.zeros((B, H), dtype=bool)
            _, _, recons = dec_module(dec_carry, imgfeat, reset, training=False)
            feat_tensor = feat2tensor_fn(imgfeat)
            rew_dist = rew_module(feat_tensor, bdims=2)
            imagined_rewards = rew_dist.pred()
            imagined_obs = {key: dist.pred() for key, dist in recons.items()}
            return imagined_obs, imagined_rewards
        
        pure_fn = nj.pure(_do_imagination)
        rng_seed = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        # Use agent.params directly (not a captured reference) so we get updated params after fine-tuning
        with jax.transfer_guard('allow'):
            _, (imagined_obs, imagined_rewards) = pure_fn(
                agent.params, seed=rng_seed, create=True, modify=True, ignore=True)
        
        return imagined_obs, imagined_rewards
    
    # Test imagination
    imagination_available = False
    if dyn_module is not None:
        print("[INFO] Testing world model imagination...")
        with jax.transfer_guard('allow'):
            try:
                test_dyn_carry = dyn_module.initial(1)
                test_dec_carry = dec_module.initial(1) if hasattr(dec_module, 'initial') else {}
                test_actions = {'action': jnp.zeros((1, 5, act_space['action'].shape[0]))}
                test_obs, test_rew = imagine_with_world_model(
                    test_dyn_carry, test_dec_carry, test_actions, 5)
                imagination_available = True
                print("[INFO] ✓ World model imagination is WORKING!")
            except Exception as e:
                print(f"[WARNING] World model imagination test failed: {e}")
                traceback.print_exc()
    
    if not imagination_available:
        print("[ERROR] World model imagination required for fine-tuning detection")
        return
    
    # Helper to safely convert JAX arrays to numpy
    def safe_to_numpy(x):
        if isinstance(x, (np.ndarray, float, int)):
            return np.array(x)
        x_host = jax.device_get(x)
        if hasattr(x_host, 'dtype') and str(x_host.dtype) == 'bfloat16':
            return np.array(x_host, dtype=np.float32)
        return np.array(x_host)
    
    # =========================================================================
    # Fine-tuning function
    # =========================================================================
    def make_finetune_stream(replay, batch_size):
        """Create a data stream from the replay buffer."""
        while True:
            yield replay.sample(batch_size)
    
    def perform_finetuning(agent, train_state, finetune_replay, steps, batch_size,
                          world_model_only=False, policy_only=False):
        """Perform fine-tuning on collected data with optional phase control.
        
        Args:
            agent: The DreamerV3 agent
            train_state: Current training state
            finetune_replay: Replay buffer with collected data
            steps: Number of fine-tuning steps
            batch_size: Batch size for training
            world_model_only: If True, only train world model (freeze policy)
            policy_only: If True, only train policy (freeze world model)
        """
        phase_name = "WORLD MODEL" if world_model_only else "POLICY" if policy_only else "JOINT"
        print(f"\n[FINETUNE] Starting {phase_name} fine-tuning for {steps} steps...")
        print(f"[FINETUNE] Replay buffer size: {len(finetune_replay)}")
        
        finetune_losses = []
        
        # Check if we have enough data
        if len(finetune_replay) < batch_size:
            print(f"[FINETUNE] Not enough data in replay buffer ({len(finetune_replay)} < {batch_size})")
            return finetune_losses, train_state
        
        # Save original loss scales from config
        original_loss_scales = agent.config.loss_scales.copy()
        
        # Get current scales dict (includes expanded rec for each decoder)
        if hasattr(agent, 'scales'):
            original_scales = dict(agent.scales)
        else:
            # Build scales manually if not available
            original_scales = original_loss_scales.copy()
        
        try:
            # Modify loss scales based on training phase
            if world_model_only:
                # Train ONLY world model (zero out policy losses)
                if hasattr(agent, 'scales'):
                    agent.scales['policy'] = 0.0
                    agent.scales['value'] = 0.0
                    agent.scales['repval'] = 0.0
                # Update config using update() method for immutable config
                agent.config = agent.config.update({
                    'loss_scales.policy': 0.0,
                    'loss_scales.value': 0.0,
                    'loss_scales.repval': 0.0,
                })
                print(f"[FINETUNE] Policy gradients DISABLED (world model adaptation only)")
                print(f"[FINETUNE] Active losses: rec, rew, con, dyn, rep")
            elif policy_only:
                # Train ONLY policy (zero out world model losses)
                if hasattr(agent, 'scales'):
                    agent.scales['rec'] = 0.0
                    agent.scales['rew'] = 0.0
                    agent.scales['con'] = 0.0
                    agent.scales['dyn'] = 0.0
                    agent.scales['rep'] = 0.0
                # Update config using update() method for immutable config
                agent.config = agent.config.update({
                    'loss_scales.rec': 0.0,
                    'loss_scales.rew': 0.0,
                    'loss_scales.con': 0.0,
                    'loss_scales.dyn': 0.0,
                    'loss_scales.rep': 0.0,
                })
                # Keep value losses for policy training
                print(f"[FINETUNE] World model gradients DISABLED (policy adaptation only)")
                print(f"[FINETUNE] Active losses: policy, value, repval")
        
            # Create data stream wrapped by agent.stream() which adds 'seed' key
            raw_stream = make_finetune_stream(finetune_replay, batch_size)
            data_stream = iter(agent.stream(raw_stream))
        
            for step in range(steps):
                try:
                    # Get batch from stream (includes seed added by agent.stream())
                    batch = next(data_stream)
                    if batch is None:
                        print(f"[FINETUNE] No data available in replay buffer")
                        break
                    
                    # Run training step
                    train_state, outs, mets = agent.train(train_state, batch)
                    
                    # Debug: Check what type mets is
                    if step == 0:
                        print(f"[DEBUG] mets type: {type(mets)}")
                        print(f"[DEBUG] mets keys: {list(mets.keys()) if hasattr(mets, 'keys') else 'NO KEYS METHOD'}")
                        print(f"[DEBUG] mets length: {len(mets) if hasattr(mets, '__len__') else 'NO LEN'}")
                    
                    # Collect ALL metrics (including losses and rewards)
                    # CRITICAL: mets dict is reused by embodied, must extract values immediately!
                    step_metrics = {}
                    if len(mets) > 0:  # Only process if mets has data
                        for k, v in list(mets.items()):  # Use list() to avoid iterator issues
                            try:
                                # Use .item() to extract Python scalar from JAX/numpy array
                                if hasattr(v, 'item'):
                                    step_metrics[k] = float(v.item())
                                else:
                                    step_metrics[k] = float(v)
                            except Exception as e:
                                print(f"[WARNING] Failed to convert metric {k}: {e}")
                    
                    # Debug: Print all available metrics on first few steps
                    if step <= 1:
                        print(f"[DEBUG] Step {step} - Collected {len(step_metrics)} metrics: {list(step_metrics.keys())[:10]}")
                    
                    # Store ALL metrics (not just losses) for plotting
                    # Must create NEW dict each time to avoid reference issues
                    if len(step_metrics) > 0:
                        finetune_losses.append(dict(step_metrics))
                    else:
                        # Even if empty, append something to maintain step count
                        finetune_losses.append({})
                    
                    if step % 20 == 0:
                        # Print loss summary (filter to loss/ metrics for display)
                        display_losses = {k: v for k, v in step_metrics.items() if k.startswith('loss/')}
                        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(display_losses.items())[:3]])
                        print(f"[FINETUNE] Step {step}/{steps} - {loss_str}")
                        
                        # Print reward prediction metrics if available
                        reward_metrics = {k: v for k, v in step_metrics.items() 
                                        if 'reward' in k.lower() or 'rew' in k.lower()}
                        if reward_metrics:
                            print(f"           Reward metrics:")
                            for k, v in list(reward_metrics.items())[:5]:  # Show up to 5 reward metrics
                                print(f"             {k}: {v:.6f}")
                        
                        # Print actual vs predicted rewards from batch if available
                        if 'reward' in batch:
                            actual_rewards_np = safe_to_numpy(batch['reward'])
                            print(f"           Batch reward stats:")
                            print(f"             Actual rewards: mean={np.mean(actual_rewards_np):.4f}, std={np.std(actual_rewards_np):.4f}")
                            print(f"             Reward shape: {actual_rewards_np.shape}")
                        
                except Exception as e:
                    print(f"[FINETUNE] Error at step {step}: {e}")
                    traceback.print_exc()
                    break
        
        finally:
            # ALWAYS restore original scales
            if hasattr(agent, 'scales'):
                agent.scales.update(original_scales)
            # Restore config using update() method
            loss_scales_restore = {f'loss_scales.{k}': v for k, v in original_loss_scales.items()}
            agent.config = agent.config.update(loss_scales_restore)
            print(f"[FINETUNE] Restored original loss scales")
        
        print(f"[FINETUNE] Completed {len(finetune_losses)} fine-tuning steps")
        if len(finetune_losses) > 0:
            print(f"[FINETUNE] Sample metrics from last step: {list(finetune_losses[-1].keys())[:5]}")
            
            # Show policy loss progression
            policy_losses = [e.get('loss/policy', 0) for e in finetune_losses if len(e) > 0]
            if len(policy_losses) > 10:
                print(f"[FINETUNE] Policy loss: initial={policy_losses[1]:.4f}, final={policy_losses[-1]:.4f}, " 
                      f"change={policy_losses[-1]-policy_losses[1]:.4f}")
            
            # Show world model loss progression  
            dyn_losses = [e.get('loss/dyn', 0) for e in finetune_losses if len(e) > 0]
            if len(dyn_losses) > 10:
                print(f"[FINETUNE] Dynamics loss: initial={dyn_losses[1]:.4f}, final={dyn_losses[-1]:.4f}, "
                      f"reduction={((dyn_losses[1]-dyn_losses[-1])/dyn_losses[1]*100):.1f}%")
            
            print(f"[DEBUG] Last entry before return: {finetune_losses[-1]}")
            # Verify the data is actually there
            non_empty = sum(1 for entry in finetune_losses if len(entry) > 0)
            print(f"[DEBUG] Non-empty entries: {non_empty}/{len(finetune_losses)}")
        else:
            print(f"[FINETUNE] WARNING: No losses collected!")
        
        # Create a completely new list with new dicts to avoid any reference issues
        losses_copy = []
        for entry in finetune_losses:
            losses_copy.append({k: v for k, v in entry.items()})
        
        print(f"[DEBUG] Returning {len(losses_copy)} entries, first has {len(losses_copy[0]) if losses_copy else 0} keys")
        return losses_copy, train_state
    
    # =========================================================================
    # Storage
    # =========================================================================
    action_history = []
    obs_history = []
    reward_history = []
    latent_history = []
    is_first_history = []  # Track episode boundaries
    
    prediction_errors_obs = []
    prediction_errors_rew = []
    avg_errors_timeline = []
    avg_rewards_timeline = []
    recent_errors = []
    recent_rewards = []
    
    # Fine-tuning state
    baseline_error = None
    baseline_reward = None
    is_collecting_data = False
    is_finetuning = False
    finetuned_once = False
    finetune_trigger_step = -1
    finetune_completed_step = -1  # Track when fine-tuning completed
    data_collection_start_step = -1
    
    # Event tracking
    finetune_events = []
    data_collection_events = []
    finetune_losses_all = []
    skipped_imaginations = 0  # Track skipped imagination due to resets
    
    print(f"\n[INFO] Starting evaluation for {eval_steps} steps...")
    print(f"[INFO] Imagination will run every {imagination_interval} steps")
    print(f"[INFO] Robot modification at step {modification_step}")
    
    # =========================================================================
    # Main evaluation loop
    # =========================================================================
    for step_idx in range(eval_steps):
        # Robot modification at 1/4 of total steps
        if args_cli.modify_robot and step_idx == modification_step:
            print(f"\n{'='*60}")
            print(f"[INFO] Step {step_idx}: MODIFYING ROBOT")
            print(f"{'='*60}")
            try:
                import torch
                isaac_env = env._env.unwrapped
                robot = isaac_env.scene["robot"]
                print(f"  Joint names: {robot.data.joint_names}")
                
                # Damage entire right hind leg (RH): HAA, HFE, KFE
                # Joint pattern: [LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE]
                # Right Hind = indices 3, 7, 11 (RH_HAA, RH_HFE, RH_KFE)
                # Left Hind = indices 0, 4, 8 (LF_HAA, LF_HFE, LF_KFE)
                damaged_joints = [3, 7, 11, 0, 4, 8]  # Right + Left hind leg
                
                print(f"  Severely weakening RIGHT + LEFT HIND LEG joints: {[robot.data.joint_names[i] for i in damaged_joints]}")
                print(f"  Reducing to 10% strength (catastrophic damage)...")
                
                for joint_id in damaged_joints:
                    # Get current limits
                    current_effort_limit = robot.data.joint_effort_limits[0, joint_id].item()
                    current_vel_limit = robot.data.joint_vel_limits[0, joint_id].item()
                    
                    # Reduce to 10% (catastrophic failure)
                    weak_effort_limit = current_effort_limit * 1.0
                    weak_vel_limit = current_vel_limit * 0.2
                    
                    robot.write_joint_effort_limit_to_sim(
                        torch.tensor([[weak_effort_limit]], device=robot.device),
                        joint_ids=[joint_id]
                    )
                    robot.write_joint_velocity_limit_to_sim(
                        torch.tensor([[weak_vel_limit]], device=robot.device),
                        joint_ids=[joint_id]
                    )
                    
                    print(f"    {robot.data.joint_names[joint_id]}: {current_effort_limit:.1f} → {weak_effort_limit:.1f} N⋅m (10%)")
                
                print(f"  ✓ RIGHT + LEFT HIND LEG severely damaged (10% torque, 30% velocity)")
            except Exception as e:
                print(f"  [WARNING] Could not modify robot: {e}")
                import traceback
                traceback.print_exc()
        
        # Store current observation
        current_obs_vec = obs['vector'].copy()
        obs_history.append(current_obs_vec[0])
        
        # Store episode boundary flag
        is_first_history.append(bool(obs['is_first'][0]))
        
        # Store latent state for imagination
        enc_carry, dyn_carry, dec_carry, prevact = carry
        with jax.transfer_guard('allow'):
            def unpack_sharded(x):
                if isinstance(x, list):
                    if len(x) == 1:
                        return x[0]
                    else:
                        return jnp.concatenate(x, axis=0)
                return x
            
            # Convert to numpy arrays for storage (avoids JAX buffer lifecycle issues after fine-tuning)
            stored_dyn = {k: np.array(jax.device_get(unpack_sharded(v))) for k, v in dyn_carry.items()}
            stored_dec = {k: np.array(jax.device_get(unpack_sharded(v))) for k, v in dec_carry.items()} if dec_carry else {}
            
            latent_history.append({
                'dyn_carry': stored_dyn,
                'dec_carry': stored_dec,
            })
        
        # Get action from policy
        carry, acts, policy_outs = agent.policy(carry, obs, mode='eval')
        acts['reset'] = obs['is_last'].copy()
        
        action_vec = acts.get('action', np.zeros((num_envs, act_space['action'].shape[0])))
        action_history.append(action_vec[0])
        
        # Step environment
        obs = env.step(acts)
        
        # Track reward
        reward = float(obs['reward'][0])
        reward_history.append(reward)
        
        # Track velocity command tracking (if available in obs)
        if hasattr(env, 'get_current_velocity_commands'):
            vel_cmd = env.get_current_velocity_commands()
            # This could be used for diagnostics later
        
        # Add to fine-tuning replay buffer (if collecting data or after modification)
        if is_collecting_data or step_idx >= modification_step:
            # Build transition for replay buffer (single-env format, not batched)
            # Include obs, acts, and policy_outs (like train.py does)
            transition = {
                'vector': current_obs_vec[0],  # Single env observation
                'action': action_vec[0],  # Single env action
                'reward': np.array(reward, dtype=np.float32),
                'is_first': np.array(step_idx == 0 or bool(obs['is_first'][0]), dtype=bool),
                'is_last': np.array(bool(obs['is_last'][0]), dtype=bool),
                'is_terminal': np.array(bool(obs.get('is_terminal', obs['is_last'])[0]), dtype=bool),
                'consec': np.array(0, dtype=np.int32),  # Consecutive chunk index (added by Batcher normally)
            }
            # Add policy outputs (dyn/deter, dyn/stoch, etc.)
            for k, v in policy_outs.items():
                if not k.startswith('log/'):
                    transition[k] = v[0] if hasattr(v, '__getitem__') and len(v.shape) > 0 else v
            finetune_replay.add(transition, worker=0)
        
        # =====================================================================
        # Imagination and error calculation
        # =====================================================================
        if step_idx >= stabilization_steps and step_idx % imagination_interval == 0:
            start_step = step_idx - imagination_horizon
            if start_step >= 0 and start_step + imagination_horizon <= len(obs_history):
                # Check if there's an episode boundary within the imagination window
                # Skip imagination if any step in [start_step+1, start_step+imagination_horizon] is a reset
                has_reset = any(is_first_history[start_step + 1 + i] for i in range(imagination_horizon)
                               if start_step + 1 + i < len(is_first_history))
                
                if not has_reset:  # Only imagine if no episode boundaries
                    try:
                        actual_obs = np.array(obs_history[start_step + 1:start_step + 1 + imagination_horizon])
                        actual_rewards = np.array(reward_history[start_step:start_step + imagination_horizon])
                        
                        if len(latent_history) > start_step:
                            with jax.transfer_guard('allow'):
                                past_latent = latent_history[start_step]
                                # Convert numpy arrays back to JAX arrays for imagination
                                past_dyn_carry = {k: jax.device_put(v[None, ...]) for k, v in past_latent['dyn_carry'].items()}
                                past_dec_carry = {k: jax.device_put(v[None, ...]) for k, v in past_latent['dec_carry'].items()}
                                
                                actions_np = np.array(action_history[start_step:start_step + imagination_horizon])
                                actions_jax = {'action': jax.device_put(actions_np[None, ...])}
                                
                                imagined_obs_dict, imagined_rewards = imagine_with_world_model(
                                    past_dyn_carry, past_dec_carry,
                                    actions_jax, imagination_horizon
                                )
                                
                                if 'vector' in imagined_obs_dict:
                                    imag_obs = safe_to_numpy(imagined_obs_dict['vector'][0])
                                else:
                                    imag_obs = np.concatenate(
                                        [safe_to_numpy(v[0]) for v in imagined_obs_dict.values()], axis=-1)
                                
                                imag_rew = safe_to_numpy(imagined_rewards[0])
                                
                                obs_error = np.mean(np.abs(imag_obs - actual_obs))
                                rew_error = np.mean(np.abs(imag_rew - actual_rewards))
                                
                                prediction_errors_obs.append(obs_error)
                                prediction_errors_rew.append(rew_error)
                                
                                # Print reward predictions periodically
                                if len(prediction_errors_obs) % 20 == 0:
                                    print(f"\n  [IMAGINATION] Step {step_idx} (run #{len(prediction_errors_obs)}):")
                                    print(f"    Obs error:  {obs_error:.6f}")
                                    print(f"    Rew error:  {rew_error:.6f}")
                                    print(f"    Predicted rewards: {imag_rew[:5]}")  # First 5 timesteps
                                    print(f"    Actual rewards:    {actual_rewards[:5]}")
                            
                    except Exception as e:
                        if step_idx < stabilization_steps + 200:
                            print(f"  [WARNING] Imagination failed at step {step_idx}: {e}")
                else:
                    # Skip this imagination window due to episode boundary
                    skipped_imaginations += 1
                    if skipped_imaginations <= 3:  # Only print first few times
                        print(f"  [INFO] Skipped imagination at step {step_idx} (episode boundary detected)")
        
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
            
            # Establish baseline before modification
            if step_idx < modification_step and baseline_error is None and len(recent_errors) >= error_window_size:
                baseline_error = avg_error
                baseline_reward = avg_reward
                print(f"\n[BASELINE] Established at step {step_idx}:")
                print(f"  Baseline error: {baseline_error:.6f}")
                print(f"  Baseline reward: {baseline_reward:.6f}")
            
            # =====================================================================
            # Fine-tuning trigger logic
            # =====================================================================
            if step_idx > modification_step and baseline_error is not None and not finetuned_once:
                error_ratio = avg_error / baseline_error if baseline_error > 0 else float('inf')
                reward_ratio = avg_reward / baseline_reward if baseline_reward != 0 else 0
                
                error_trigger = error_ratio > (1 + error_threshold)
                reward_trigger = reward_ratio < (1 - reward_threshold)
                
                # Start data collection
                if not is_collecting_data and not is_finetuning and (error_trigger or reward_trigger):
                    trigger_reasons = []
                    if error_trigger:
                        trigger_reasons.append(f"error↑ {error_ratio:.2f}x")
                    if reward_trigger:
                        trigger_reasons.append(f"reward↓ {reward_ratio:.2f}x")
                    
                    print(f"\n{'='*60}")
                    print(f"[TRIGGER] Fine-tuning triggered at step {step_idx}!")
                    print(f"  Reason: {', '.join(trigger_reasons)}")
                    print(f"  Baseline error: {baseline_error:.6f}, Current: {avg_error:.6f}")
                    print(f"  Baseline reward: {baseline_reward:.6f}, Current: {avg_reward:.6f}")
                    print(f"  Starting data collection for {data_collection_steps} steps...")
                    print(f"{'='*60}")
                    
                    is_collecting_data = True
                    finetune_trigger_step = step_idx
                    data_collection_start_step = step_idx
                    data_collection_events.append(step_idx)
                
                # Check if data collection is complete
                if is_collecting_data and step_idx - data_collection_start_step >= data_collection_steps:
                    print(f"\n{'='*60}")
                    print(f"[FINETUNE] Data collection complete. Starting two-phase fine-tuning...")
                    print(f"  Collected {data_collection_steps} steps of data")
                    print(f"{'='*60}")
                    
                    is_collecting_data = False
                    is_finetuning = True
                    finetune_events.append(step_idx)
                    
                    # =====================================================================
                    # PHASE 1: Train World Model Only
                    # =====================================================================
                    print("\n" + "="*70)
                    print("PHASE 1: WORLD MODEL ADAPTATION")
                    print("="*70)
                    print(f"Goal: Adapt world model to new robot dynamics ({finetune_steps_wm} steps)")
                    print("="*70)
                    
                    losses_wm, train_state = perform_finetuning(
                        agent, train_state, finetune_replay, 
                        steps=finetune_steps_wm, 
                        batch_size=finetune_batch_size,
                        world_model_only=True  # Only train world model, freeze policy
                    )
                    print(f"[PHASE 1] Received {len(losses_wm)} loss entries from world model training")
                    if len(losses_wm) > 1:
                        print(f"[PHASE 1] Entry 1 sample values: loss/dyn={losses_wm[1].get('loss/dyn', 'N/A')}, loss/rew={losses_wm[1].get('loss/rew', 'N/A')}")
                    finetune_losses_all.extend(losses_wm)
                    
                    # =====================================================================
                    # PHASE 2: Train Policy Only
                    # =====================================================================
                    print("\n" + "="*70)
                    print("PHASE 2: POLICY ADAPTATION")
                    print("="*70)
                    print(f"Goal: Learn new behaviors using updated world model ({finetune_steps_policy} steps)")
                    print("="*70)
                    
                    losses_pol, train_state = perform_finetuning(
                        agent, train_state, finetune_replay,
                        steps=finetune_steps_policy,
                        batch_size=finetune_batch_size,
                        policy_only=True  # Only train policy, freeze world model
                    )
                    print(f"[PHASE 2] Received {len(losses_pol)} loss entries from policy training")
                    if len(losses_pol) > 1:
                        print(f"[PHASE 2] Entry 1 sample values: loss/policy={losses_pol[1].get('loss/policy', 'N/A')}, loss/value={losses_pol[1].get('loss/value', 'N/A')}")
                    finetune_losses_all.extend(losses_pol)
                    
                    print(f"\n[FINETUNE] Two-phase training complete!")
                    print(f"[DEBUG] Total losses in finetune_losses_all: {len(finetune_losses_all)}")
                    
                    # Mark as finetuned
                    finetuned_once = True
                    is_finetuning = False
                    
                    # IMPORTANT: Reinitialize policy carry after fine-tuning
                    # The old carry contains references to invalidated JAX buffers
                    print(f"[FINETUNE] Reinitializing policy with updated parameters...")
                    carry = agent.init_policy(num_envs)
                    
                    # Mark when fine-tuning completed
                    finetune_completed_step = step_idx
                    
                    # Don't clear latent_history - old states are still valid features
                    # Just note that states before this point are from the old model
                    print(f"[FINETUNE] Continuing with existing latent history ({len(latent_history)} steps)")
                    print(f"[FINETUNE] Note: Latent states before step {finetune_completed_step} are from pre-fine-tuned model")
                    
                    # Reset error tracking to monitor improvement
                    recent_errors = []
                    recent_rewards = []
                    
                    print(f"\n[FINETUNE] Fine-tuning complete. Monitoring improvement...")
        
        # Progress logging
        if step_idx % 500 == 0:
            avg_err = avg_errors_timeline[-1] if avg_errors_timeline else 0
            avg_rew = avg_rewards_timeline[-1] if avg_rewards_timeline else 0
            n_imag = len(prediction_errors_obs)
            
            status = "STABILIZING" if step_idx < stabilization_steps else \
                     "COLLECTING" if is_collecting_data else \
                     "FINETUNING" if is_finetuning else \
                     "FINETUNED" if finetuned_once else "NORMAL"
            
            print(f"Step {step_idx}/{eval_steps}, Status: {status}, "
                  f"Avg Error: {avg_err:.6f}, Avg Reward: {avg_rew:.4f}, "
                  f"Imagination runs: {n_imag}")
    
    print("\n[INFO] Evaluation complete. Processing results...")
    
    # =========================================================================
    # Save results
    # =========================================================================
    obs_history = np.array(obs_history)
    reward_history = np.array(reward_history)
    prediction_errors_obs = np.array(prediction_errors_obs)
    prediction_errors_rew = np.array(prediction_errors_rew)
    avg_errors_timeline = np.array(avg_errors_timeline)
    avg_rewards_timeline = np.array(avg_rewards_timeline)
    
    save_data = {
        "prediction_errors_obs": prediction_errors_obs,
        "prediction_errors_rew": prediction_errors_rew,
        "avg_errors_timeline": avg_errors_timeline,
        "avg_rewards_timeline": avg_rewards_timeline,
        "reward_history": reward_history,
        "obs_history": obs_history,
        "baseline_error": baseline_error,
        "baseline_reward": baseline_reward,
        "finetune_events": finetune_events,
        "data_collection_events": data_collection_events,
        "finetune_trigger_step": finetune_trigger_step,
        "finetune_completed_step": finetune_completed_step,
        "finetuned": finetuned_once,
        "config": {
            "task": task,
            "seed": args_cli.seed,
            "eval_steps": eval_steps,
            "imagination_horizon": imagination_horizon,
            "modification_step": modification_step,
            "error_threshold": error_threshold,
            "reward_threshold": reward_threshold,
            "finetune_steps_wm": finetune_steps_wm,
            "finetune_steps_policy": finetune_steps_policy,
            "data_collection_steps": data_collection_steps,
        }
    }
    
    results_path = output_dir / "finetune_results.npz"
    np.savez(results_path, **save_data)
    print(f"[INFO] Results saved to: {results_path}")
    
    # Save fine-tuned checkpoint if fine-tuning occurred
    if finetuned_once:
        finetune_ckpt_dir = output_dir / 'ckpt'
        finetune_ckpt_dir.mkdir(exist_ok=True)
        cp_save = elements.Checkpoint(finetune_ckpt_dir)
        cp_save.agent = agent
        cp_save.save()
        print(f"[INFO] Fine-tuned checkpoint saved to: {finetune_ckpt_dir}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "="*70)
    print("FINE-TUNING SUMMARY")
    print("="*70)
    print(f"Task: {task}")
    print(f"Total steps: {eval_steps}")
    print(f"Robot modified at step: {modification_step}")
    print(f"Imagination runs: {len(prediction_errors_obs)}")
    print(f"Skipped imaginations (episode boundaries): {skipped_imaginations}")
    print(f"\nBaseline (before modification):")
    print(f"  Error: {baseline_error:.6f}" if baseline_error else "  Error: Not established")
    print(f"  Reward: {baseline_reward:.6f}" if baseline_reward else "  Reward: Not established")
    print(f"\nTwo-Phase Fine-tuning:")
    print(f"  Triggered: {'Yes' if finetune_trigger_step > 0 else 'No'}")
    if finetune_trigger_step > 0:
        print(f"  Trigger step: {finetune_trigger_step}")
        print(f"  Data collection steps: {data_collection_steps}")
        print(f"  Phase 1 (World Model): {finetune_steps_wm} steps")
        print(f"  Phase 2 (Policy): {finetune_steps_policy} steps")
        print(f"  Total fine-tuning steps: {len(finetune_losses_all) if finetune_losses_all else 0}")
    print(f"  Completed: {'Yes' if finetuned_once else 'No'}")
    
    if len(prediction_errors_obs) > 0:
        # Split errors by modification
        imag_steps = np.linspace(stabilization_steps, eval_steps, len(prediction_errors_obs))
        before_mask = imag_steps < modification_step
        after_mask = imag_steps >= modification_step
        
        if np.any(before_mask) and np.any(after_mask):
            print(f"\nPrediction Error Statistics:")
            print(f"  Before modification: {np.mean(prediction_errors_obs[before_mask]):.6f}")
            print(f"  After modification: {np.mean(prediction_errors_obs[after_mask]):.6f}")
            
            if finetuned_once:
                # Find errors after fine-tuning
                after_finetune_mask = imag_steps >= finetune_trigger_step + data_collection_steps
                if np.any(after_finetune_mask):
                    print(f"  After fine-tuning: {np.mean(prediction_errors_obs[after_finetune_mask]):.6f}")
    
    print("="*70)
    
    # =========================================================================
    # Generate plots
    # =========================================================================
    if not args_cli.no_plots:
        try:
            import matplotlib.pyplot as plt
            
            print("\n[INFO] Generating plots...")
            
            # Plot 1: Error timeline with events
            if len(avg_errors_timeline) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                
                x_values = np.linspace(stabilization_steps, eval_steps, len(avg_errors_timeline))
                
                ax1.plot(x_values, avg_errors_timeline, 'r-', linewidth=2, label='Avg Prediction Error')
                if baseline_error is not None:
                    ax1.axhline(y=baseline_error, color='blue', linestyle='--', 
                                alpha=0.7, label=f'Baseline ({baseline_error:.4f})')
                    ax1.axhline(y=baseline_error * (1 + error_threshold), color='orange', 
                                linestyle=':', alpha=0.7, label='Threshold')
                
                if modification_step:
                    ax1.axvline(x=modification_step, color='purple', linestyle='--', 
                                linewidth=2, label='Robot Modified')
                
                for event in data_collection_events:
                    ax1.axvline(x=event, color='orange', linestyle='-', alpha=0.8, linewidth=2)
                    ax1.axvspan(event, event + data_collection_steps, alpha=0.2, color='orange')
                
                for event in finetune_events:
                    ax1.axvline(x=event, color='green', linestyle='-', alpha=0.8, linewidth=2)
                
                if data_collection_events:
                    ax1.axvline(x=data_collection_events[0], color='orange', linestyle='-', 
                                alpha=0.8, linewidth=2, label='Data Collection')
                if finetune_events:
                    ax1.axvline(x=finetune_events[0], color='green', linestyle='-', 
                                alpha=0.8, linewidth=2, label='Fine-tuning')
                
                ax1.set_ylabel('Prediction Error')
                ax1.set_title('Prediction Error Timeline with Fine-tuning Events')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Reward plot
                ax2.plot(x_values[:len(avg_rewards_timeline)], avg_rewards_timeline, 
                         'b-', linewidth=2, label='Avg Reward')
                if baseline_reward is not None:
                    ax2.axhline(y=baseline_reward, color='green', linestyle='--', 
                                alpha=0.7, label=f'Baseline ({baseline_reward:.4f})')
                
                if modification_step:
                    ax2.axvline(x=modification_step, color='purple', linestyle='--', linewidth=2)
                
                for event in finetune_events:
                    ax2.axvline(x=event, color='green', linestyle='-', alpha=0.8, linewidth=2)
                
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Reward')
                ax2.set_title('Reward Timeline')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "finetune_timeline.png", dpi=150)
                plt.close()
                print(f"  Saved: finetune_timeline.png")
            
            # Plot 2: Fine-tuning losses
            if finetune_losses_all:
                # Filter to only loss metrics for the loss plot
                # Skip empty entries (step 0 may have no data) and find first non-empty entry
                all_keys = []
                for entry in finetune_losses_all:
                    if len(entry) > 0:
                        all_keys = list(entry.keys())
                        break
                loss_keys = [k for k in all_keys if k.startswith('loss/')]
                if loss_keys:
                    n_losses = len(loss_keys)
                    fig, axes = plt.subplots((n_losses + 1) // 2, 2, figsize=(14, 3.5 * ((n_losses + 1) // 2)))
                    if n_losses == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    # Phase 1 is WM steps, Phase 2 is policy steps
                    phase1_end = finetune_steps_wm  # World model training ends
                    
                    for i, key in enumerate(loss_keys):
                        values = [l.get(key, 0) for l in finetune_losses_all]
                        axes[i].plot(values, linewidth=2, alpha=0.8, label='Loss')
                        
                        # Add phase separator
                        if len(values) > phase1_end:
                            axes[i].axvline(x=phase1_end, color='red', linestyle='--', 
                                          linewidth=2, alpha=0.6, label='Phase 1→2')
                            # Add phase labels
                            y_mid = (axes[i].get_ylim()[0] + axes[i].get_ylim()[1]) / 2
                            axes[i].text(phase1_end/2, axes[i].get_ylim()[1]*0.95, 
                                       'WM Only', ha='center', fontsize=9, 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                            axes[i].text(phase1_end + (len(values)-phase1_end)/2, axes[i].get_ylim()[1]*0.95,
                                       'Policy Only', ha='center', fontsize=9,
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                        
                        # Add moving average
                        if len(values) > 10:
                            window = min(50, len(values) // 10)
                            ma = np.convolve(values, np.ones(window)/window, mode='valid')
                            axes[i].plot(range(window-1, len(values)), ma, 
                                       linewidth=2.5, color='orange', alpha=0.7, label='Moving Avg')
                        
                        axes[i].set_title(key.replace('_', ' ').replace('train/', '').title(), fontsize=11)
                        axes[i].set_xlabel('Fine-tuning Step (2-Phase)')
                        axes[i].set_ylabel('Loss')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].legend(fontsize=8)
                    
                    for i in range(len(loss_keys), len(axes)):
                        axes[i].axis('off')
                    
                    plt.suptitle('Fine-tuning Losses Over Training Steps', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(output_dir / "finetune_losses.png", dpi=150)
                    plt.close()
                    print(f"  Saved: finetune_losses.png")
                
                # Plot 3: Focused reward metrics plot (both loss and actual rewards)
                # Find first non-empty entry for keys
                all_keys_for_reward = []
                for entry in finetune_losses_all:
                    if len(entry) > 0:
                        all_keys_for_reward = list(entry.keys())
                        break
                reward_loss_keys = [k for k in all_keys_for_reward if 'loss' in k and 'rew' in k.lower()]
                reward_keys = [k for k in all_keys_for_reward if k == 'rew' or (k.startswith('rew') and 'loss' not in k)]
                
                if reward_loss_keys or reward_keys:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot reward losses
                    if reward_loss_keys:
                        for key in reward_loss_keys:
                            values = [l.get(key, 0) for l in finetune_losses_all]
                            ax1.plot(values, linewidth=2, alpha=0.8, label=key)
                        ax1.set_xlabel('Fine-tuning Step', fontsize=12)
                        ax1.set_ylabel('Reward Loss', fontsize=12)
                        ax1.set_title('Reward Prediction Loss During Fine-tuning', fontsize=14, fontweight='bold')
                        ax1.legend(fontsize=10)
                        ax1.grid(True, alpha=0.3)
                    else:
                        ax1.text(0.5, 0.5, 'No reward loss data', 
                                transform=ax1.transAxes, ha='center', va='center')
                        ax1.set_title('Reward Prediction Loss')
                    
                    # Plot actual reward predictions
                    if reward_keys:
                        for key in reward_keys:
                            values = [l.get(key, 0) for l in finetune_losses_all]
                            ax2.plot(values, linewidth=2, alpha=0.8, label=key)
                        ax2.set_xlabel('Fine-tuning Step', fontsize=12)
                        ax2.set_ylabel('Predicted Reward', fontsize=12)
                        ax2.set_title('Reward Predictions During Fine-tuning', fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=10)
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'No reward prediction data',
                                transform=ax2.transAxes, ha='center', va='center')
                        ax2.set_title('Reward Predictions')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "finetune_reward_metrics.png", dpi=150)
                    plt.close()
                    print(f"  Saved: finetune_reward_metrics.png")
            
            # Plot 4: 3-Phase Comparison (Baseline, Post-Damage, Post-Finetuning)
            if finetuned_once and len(avg_errors_timeline) > 0 and modification_step:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                x_values = np.linspace(stabilization_steps, eval_steps, len(avg_errors_timeline))
                
                # Split into 3 phases
                if finetune_completed_step > 0:
                    # Find indices for each phase
                    modification_idx = int((modification_step - stabilization_steps) / 
                                          (eval_steps - stabilization_steps) * len(avg_errors_timeline))
                    finetune_idx = int((finetune_completed_step - stabilization_steps) / 
                                      (eval_steps - stabilization_steps) * len(avg_errors_timeline))
                    
                    # Phase 1: Before modification (baseline)
                    phase1_errors = avg_errors_timeline[:modification_idx]
                    # Phase 2: After modification, before fine-tuning
                    phase2_errors = avg_errors_timeline[modification_idx:finetune_idx]
                    # Phase 3: After fine-tuning
                    phase3_errors = avg_errors_timeline[finetune_idx:]
                    
                    # Prediction error comparison (3 phases)
                    if len(phase1_errors) > 0 and len(phase2_errors) > 0 and len(phase3_errors) > 0:
                        ax1.hist([phase1_errors, phase2_errors, phase3_errors], bins=20, 
                                label=['Baseline (Healthy)', 'Post-Damage (Pre-FT)', 'Post-Fine-tuning'],
                                alpha=0.7, color=['blue', 'red', 'green'])
                        ax1.axvline(np.mean(phase1_errors), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Baseline: {np.mean(phase1_errors):.4f}')
                        ax1.axvline(np.mean(phase2_errors), color='red', linestyle='--', 
                                   linewidth=2, label=f'Post-Damage: {np.mean(phase2_errors):.4f}')
                        ax1.axvline(np.mean(phase3_errors), color='green', linestyle='--', 
                                   linewidth=2, label=f'Post-FT: {np.mean(phase3_errors):.4f}')
                        ax1.set_xlabel('Prediction Error')
                        ax1.set_ylabel('Frequency')
                        ax1.set_title('Prediction Error Distribution\nAcross 3 Phases')
                        ax1.legend(fontsize=9)
                        ax1.grid(True, alpha=0.3)
                    
                    # Reward comparison (3 phases)
                    phase1_rew = avg_rewards_timeline[:min(modification_idx, len(avg_rewards_timeline))]
                    phase2_rew = avg_rewards_timeline[min(modification_idx, len(avg_rewards_timeline)):
                                                     min(finetune_idx, len(avg_rewards_timeline))]
                    phase3_rew = avg_rewards_timeline[min(finetune_idx, len(avg_rewards_timeline)):]
                    
                    if len(phase1_rew) > 0 and len(phase2_rew) > 0 and len(phase3_rew) > 0:
                        ax2.hist([phase1_rew, phase2_rew, phase3_rew], bins=20,
                                label=['Baseline (Healthy)', 'Post-Damage (Pre-FT)', 'Post-Fine-tuning'],
                                alpha=0.7, color=['blue', 'red', 'green'])
                        ax2.axvline(np.mean(phase1_rew), color='blue', linestyle='--', 
                                   linewidth=2, label=f'Baseline: {np.mean(phase1_rew):.4f}')
                        ax2.axvline(np.mean(phase2_rew), color='red', linestyle='--', 
                                   linewidth=2, label=f'Post-Damage: {np.mean(phase2_rew):.4f}')
                        ax2.axvline(np.mean(phase3_rew), color='green', linestyle='--', 
                                   linewidth=2, label=f'Post-FT: {np.mean(phase3_rew):.4f}')
                        ax2.set_xlabel('Average Reward')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title('Reward Distribution\nAcross 3 Phases')
                        ax2.legend(fontsize=9)
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "finetune_comparison.png", dpi=150)
                    plt.close()
                    print(f"  Saved: finetune_comparison.png")
            
            # Plot 5: Summary statistics
            if finetuned_once and finetune_losses_all and len(finetune_losses_all) > 0:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                
                # Initialize improvement metrics
                improvement = 0.0
                rew_improvement = 0.0
                
                # Debug: print available keys
                print(f"[DEBUG] finetune_losses_all length: {len(finetune_losses_all)}")
                if len(finetune_losses_all) > 1:
                    print(f"[DEBUG] Entry 0 (step 0): {finetune_losses_all[0]}")
                    print(f"[DEBUG] Entry 1 (step 1) type: {type(finetune_losses_all[1])}")
                    print(f"[DEBUG] Entry 1 keys: {list(finetune_losses_all[1].keys())[:10]}")
                elif len(finetune_losses_all) > 0:
                    print(f"[DEBUG] Only one entry available: {finetune_losses_all[0]}")
                else:
                    print(f"[DEBUG] finetune_losses_all is empty!")
                
                # Loss convergence - search for dynamics loss which is usually the main one
                # Use first non-empty entry to find keys
                total_loss_key = None
                first_entry = next((e for e in finetune_losses_all if len(e) > 0), {})
                for possible_key in ['loss/dyn', 'train/loss/dyn', 'loss/model', 'train/loss/model', 'model_loss', 'total_loss']:
                    if possible_key in first_entry:
                        total_loss_key = possible_key
                        break
                
                if total_loss_key:
                    total_losses = [l.get(total_loss_key, 0) for l in finetune_losses_all]
                    if any(total_losses):  # Check if we have non-zero values
                        ax1.plot(total_losses, linewidth=2, color='blue', alpha=0.8)
                        ax1.set_xlabel('Training Step', fontsize=11)
                        ax1.set_ylabel('Model Loss', fontsize=11)
                        ax1.set_title(f'Model Loss Convergence\n({total_loss_key})', fontsize=12)
                        ax1.grid(True, alpha=0.3)
                        
                        # Add annotation for final loss
                        final_loss = total_losses[-1] if total_losses else 0
                        initial_loss = total_losses[0] if total_losses else 0
                        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                        ax1.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
                                transform=ax1.transAxes, ha='center', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax1.text(0.5, 0.5, 'No model loss data found', 
                            transform=ax1.transAxes, ha='center', va='center')
                    ax1.set_title('Model Loss Convergence')
                
                # Reward loss convergence - try multiple possible key names
                # Use first non-empty entry to find keys
                reward_loss_key = None
                first_entry = next((e for e in finetune_losses_all if len(e) > 0), {})
                for possible_key in ['loss/rew', 'train/loss/rew', 'loss/reward', 'train/loss/reward', 'reward_loss', 'rew_loss']:
                    if possible_key in first_entry:
                        reward_loss_key = possible_key
                        break
                
                if reward_loss_key:
                    reward_losses = [l.get(reward_loss_key, 0) for l in finetune_losses_all]
                    if any(reward_losses):  # Check if we have non-zero values
                        ax2.plot(reward_losses, linewidth=2, color='green', alpha=0.8)
                        ax2.set_xlabel('Training Step', fontsize=11)
                        ax2.set_ylabel('Reward Loss', fontsize=11)
                        ax2.set_title(f'Reward Prediction Loss\n({reward_loss_key})', fontsize=12)
                        ax2.grid(True, alpha=0.3)
                        
                        # Add annotation
                        final_rew_loss = reward_losses[-1] if reward_losses else 0
                        initial_rew_loss = reward_losses[0] if reward_losses else 0
                        rew_improvement = ((initial_rew_loss - final_rew_loss) / initial_rew_loss * 100) if initial_rew_loss > 0 else 0
                        ax2.text(0.5, 0.95, f'Improvement: {rew_improvement:.1f}%',
                                transform=ax2.transAxes, ha='center', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax2.text(0.5, 0.5, 'No reward loss data found', 
                            transform=ax2.transAxes, ha='center', va='center')
                    ax2.set_title('Reward Prediction Loss')
                
                # Loss components breakdown (final values) - only show loss/ metrics
                # Use first non-empty entry to find keys
                first_entry = next((e for e in finetune_losses_all if len(e) > 0), {})
                all_keys = list(first_entry.keys())
                # Filter to only loss/ metrics
                loss_keys_only = [k for k in all_keys if k.startswith('loss/')]
                loss_items = [(k, finetune_losses_all[-1].get(k, 0)) for k in loss_keys_only]
                # Keep only non-zero losses
                loss_items = [(k, v) for k, v in loss_items if abs(v) > 1e-10]
                
                if loss_items:
                    loss_names = [k.replace('train/loss/', '').replace('train/', '').replace('_', ' ').title() 
                                 for k, _ in loss_items]
                    final_values = [v for _, v in loss_items]
                    
                    # Sort by value for better visualization
                    sorted_items = sorted(zip(loss_names, final_values), key=lambda x: abs(x[1]), reverse=True)
                    loss_names, final_values = zip(*sorted_items) if sorted_items else ([], [])
                    
                    ax3.barh(range(len(loss_names)), final_values, color='skyblue', alpha=0.8)
                    ax3.set_yticks(range(len(loss_names)))
                    ax3.set_yticklabels(loss_names, fontsize=8)
                    ax3.set_xlabel('Final Loss Value', fontsize=11)
                    ax3.set_title('Loss Components (Final Values)', fontsize=12)
                    ax3.grid(True, alpha=0.3, axis='x')
                else:
                    ax3.text(0.5, 0.5, 'No loss data available', 
                            transform=ax3.transAxes, ha='center', va='center')
                    ax3.set_title('Loss Components (Final Values)')
                
                # Training summary text
                ax4.axis('off')
                summary_text = f"""
Two-Phase Fine-tuning Summary
{'='*40}

Configuration:
  • Phase 1 (WM): {finetune_steps_wm} steps
  • Phase 2 (Policy): {finetune_steps_policy} steps
  • Total steps: {len(finetune_losses_all)}
  • Batch size: {finetune_batch_size}
  • Sequence length: {finetune_sequence_length}
  
Baseline (before modification):
  • Prediction error: {baseline_error:.6f}
  • Average reward: {baseline_reward:.6f}

Results:
  • Model loss improvement: {improvement:.1f}%
  • Reward loss improvement: {rew_improvement:.1f}%
  • Data collected: {data_collection_steps} steps
  
Status: {'✓ Completed' if finetuned_once else '✗ Not completed'}
                """
                ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
                
                plt.suptitle('Fine-tuning Summary Statistics', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / "finetune_summary.png", dpi=150)
                plt.close()
                print(f"  Saved: finetune_summary.png")
            
            print(f"\n[INFO] All plots saved to: {output_dir}")
            
        except ImportError:
            print("[WARNING] matplotlib not available, skipping plots")
        except Exception as e:
            print(f"[WARNING] Error generating plots: {e}")
            traceback.print_exc()
    
    # Cleanup
    env.close()
    print("\n[INFO] Fine-tuning analysis finished!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Fine-tuning analysis failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")
    finally:
        simulation_app.close()

