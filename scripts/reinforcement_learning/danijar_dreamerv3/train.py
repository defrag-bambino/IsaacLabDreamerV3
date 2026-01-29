# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train DreamerV3 (Danijar's implementation) with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import sys
import traceback
from functools import partial as bind

# Add dreamerv3 to path BEFORE any imports from it
dreamerv3_path = pathlib.Path(__file__).parent.parent.parent.parent / "dreamerv3"
sys.path.insert(0, str(dreamerv3_path))

# Add current script directory to path for isaac_env import
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train DreamerV3 agent with Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment.")

# Training arguments
parser.add_argument("--logdir", type=str, default="~/logdir/dreamerv3", help="Directory for logs and checkpoints.")
parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps.")
parser.add_argument("--train_ratio", type=float, default=32.0, help="Training updates per environment step.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
parser.add_argument("--batch_length", type=int, default=64, help="Sequence length for training.")

# DreamerV3 model size
parser.add_argument("--model_size", type=str, default="size12m", 
                    choices=["size1m", "size12m", "size25m", "size50m", "size100m", "size200m", "size400m"],
                    help="Model size preset.")

# Logging
parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging.")
parser.add_argument("--no_tensorboard", action="store_true", default=False, help="Disable TensorBoard logging.")
parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")

# Checkpoint
parser.add_argument("--from_checkpoint", type=str, default=None, 
                    help="Path to checkpoint dir to load weights from (e.g., ~/logdir/dreamerv3/20251216T153443/ckpt).")

# Misc
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--configs", type=str, nargs="+", default=["defaults"], help="Config presets to use.")
parser.add_argument("--jax_platform", type=str, default="cuda", choices=["cuda", "cpu"], help="JAX platform.")

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

import numpy as np
import ruamel.yaml as yaml

# Global variable to store Isaac-specific config (set in main())
isaac_env_config = {}


def _get_logdir(args):
    """Get logdir path - always creates a new timestamped subdirectory."""
    import elements
    return os.path.expanduser(args.logdir) + f'/{elements.timestamp()}'


# Global variable to store the single Isaac Lab environment instance
# (Isaac Lab environments should only be created once since they're vectorized internally)
_isaac_env_instance = None
_isaac_obs_space = None
_isaac_act_space = None


def main():
    """Train DreamerV3 agent with Isaac Lab environment."""
    
    print(f"\n{'='*60}")
    print("DreamerV3 Training with Isaac Lab")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Steps: {args_cli.steps}")
    print(f"Model size: {args_cli.model_size}")
    if args_cli.from_checkpoint:
        print(f"Loading weights from: {args_cli.from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Import dreamerv3 modules AFTER Isaac Lab is initialized
    print("[INFO] Importing DreamerV3 modules...")
    try:
        import elements
        print("[INFO] Imported elements")
    except Exception as e:
        print(f"[ERROR] Failed to import elements: {e}")
        traceback.print_exc()
        raise
    
    try:
        import embodied
        print("[INFO] Imported embodied")
    except Exception as e:
        print(f"[ERROR] Failed to import embodied: {e}")
        traceback.print_exc()
        raise
    
    try:
        from dreamerv3.agent import Agent
        print("[INFO] Imported DreamerV3 Agent")
    except Exception as e:
        print(f"[ERROR] Failed to import DreamerV3 Agent: {e}")
        traceback.print_exc()
        raise
    
    # Print DreamerV3 banner
    print("[INFO] Printing DreamerV3 banner...")
    for line in Agent.banner:
        print(line)
    print("[INFO] Banner printed")
    
    # Load base configs from dreamerv3
    print("[INFO] Loading configs from dreamerv3...")
    config_path = dreamerv3_path / "dreamerv3" / "configs.yaml"
    print(f"[INFO] Config path: {config_path}")
    print(f"[INFO] Config path exists: {config_path.exists()}")
    
    try:
        config_text = elements.Path(str(config_path)).read()
        print(f"[INFO] Config text loaded, length: {len(config_text)}")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        traceback.print_exc()
        raise
    
    configs = yaml.YAML(typ='safe').load(config_text)
    print("[INFO] Configs loaded successfully")
    
    # Make modules available globally for the factory functions
    globals()['elements'] = elements
    globals()['embodied'] = embodied
    globals()['Agent'] = Agent
    
    # Build config from presets
    config = elements.Config(configs['defaults'])
    
    # Apply config presets
    for name in args_cli.configs:
        if name in configs:
            config = config.update(configs[name])
    
    # Apply model size preset
    if args_cli.model_size in configs:
        config = config.update(configs[args_cli.model_size])
    
    # Apply Isaac-specific settings
    isaac_config = {
        'task': f'isaac_{args_cli.task}',
        'seed': args_cli.seed,
        'logdir': _get_logdir(args_cli),
        'batch_size': args_cli.batch_size,
        'batch_length': args_cli.batch_length,
    }
    config = config.update(isaac_config)
    
    # Update run settings
    # run.envs=1: Isaac Lab handles vectorization internally
    # run.debug=True: Disable multiprocessing (USD can't be pickled)
    run_config = {
        'run.steps': args_cli.steps,
        'run.train_ratio': args_cli.train_ratio,
        'run.envs': 1,
        'run.eval_envs': 0,
        'run.debug': True,
    }
    config = config.update(run_config)
    
    # Update JAX platform
    config = config.update({'jax.platform': args_cli.jax_platform})

    # Replay settings
    config = config.update({'replay.size': 1000000})
    
    # Configure logger outputs
    logger_outputs = ['jsonl', 'scope']
    if args_cli.tensorboard and not args_cli.no_tensorboard:
        logger_outputs.append('tensorboard')
    if args_cli.wandb:
        logger_outputs.append('wandb')
    config = config.update({'logger.outputs': logger_outputs})
    
    # Store Isaac-specific settings for environment creation
    global isaac_env_config
    isaac_env_config = {
        'num_envs': args_cli.num_envs,
        'device': 'cuda:0',
    }
    
    # Apply any remaining command line arguments
    if other_args:
        config = elements.Flags(config).parse(other_args)
    
    # Setup logging directory
    logdir = elements.Path(config.logdir)
    print(f'[INFO] Logdir: {logdir}')
    logdir.mkdir()
    config.save(logdir / 'config.yaml')
    
    # Initialize elements timer
    elements.timer.global_timer.enabled = config.logger.timer
    
    # Build training arguments
    print("[INFO] Building training arguments...")
    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )
    print("[INFO] Training arguments built successfully")
    
    # Run training with Isaac-optimized loop
    print("[INFO] Starting Isaac-optimized training loop...")
    train_isaac(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_batched_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args,
        num_envs=args_cli.num_envs,
    )


def train_isaac(make_agent, make_replay, make_env, make_stream, make_logger, args, num_envs):
    """
    Custom training loop optimized for Isaac Lab's internal vectorization.
    
    Unlike the standard embodied.run.train which uses a Driver with separate
    environment processes, this loop directly handles Isaac Lab's batched
    observations and actions, allowing each parallel environment to have
    different behavior.
    
    Key features:
    - Each of the num_envs parallel robots gets its own action from the policy
    - All robots can behave differently based on their individual observations
    - Episode tracking is done per-robot
    - The replay buffer receives transitions from all robots
    """
    import collections
    from functools import partial as bind
    import numpy as np
    import elements
    import embodied
    
    print(f"[INFO] Isaac training loop with {num_envs} parallel envs")
    print(f"[INFO] Each robot will receive its own action from the policy")
    
    # Create components
    # NOTE: Create env FIRST so spaces are cached for the agent
    env = make_env()  # Returns IsaacBatchedEnv - also caches obs/act spaces
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()
    
    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()
    
    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_report = embodied.LocalClock(args.report_every)
    should_save = embodied.LocalClock(args.save_every)
    
    # Initialize policy carry for all parallel envs
    # This creates separate state for each of the num_envs robots
    print(f"[INFO] Initializing policy carry for {num_envs} parallel robots...")
    carry = agent.init_policy(num_envs)
    
    # Get initial observations from all envs
    obs = env.reset()
    obs_shapes = {k: v.shape for k, v in obs.items() if k not in ['reward', 'is_first', 'is_last', 'is_terminal']}
    print(f"[INFO] Initial obs shapes: {obs_shapes}")  # Should be [num_envs, obs_dim] for each component
    
    # Initialize actions for all envs
    act_space = env.act_space
    acts = {
        k: np.zeros((num_envs,) + v.shape, v.dtype)
        for k, v in act_space.items()
    }
    acts['reset'] = np.ones(num_envs, dtype=bool)
    
    # Prepare stream for training
    stream_train = iter(agent.stream(make_stream(replay, 'train')))
    stream_report = iter(agent.stream(make_stream(replay, 'report')))
    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)
    
    # Checkpoint setup (for saving during this run)
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    
    # Load weights from external checkpoint if specified
    if args_cli.from_checkpoint:
        print(f'[INFO] Loading weights from: {args_cli.from_checkpoint}')
        from_ckpt_path = pathlib.Path(args_cli.from_checkpoint).expanduser()
        if from_ckpt_path.exists():
            # Load agent weights
            elements.checkpoint.load(str(from_ckpt_path), dict(
                agent=bind(agent.load, regex='.*'),
            ))
            print(f'[INFO] Loaded agent weights successfully')
            
            # # Load replay buffer from checkpoint's replay directory
            # checkpoint_logdir = from_ckpt_path.parent.parent  # ckpt is inside logdir
            # replay_dir = checkpoint_logdir / 'replay'
            # if replay_dir.exists():
            #     print(f'[INFO] Loading replay buffer from: {replay_dir}')
            #     replay.load(directory=str(replay_dir))
            #     print(f'[INFO] Loaded replay buffer with {len(replay)} transitions')
            # else:
            #     print(f'[WARNING] No replay directory found at {replay_dir}')
        else:
            print(f'[WARNING] Checkpoint not found at {from_ckpt_path}, starting fresh')
    
    # Save initial checkpoint
    cp.save()
    print(f'[INFO] Initial checkpoint saved')
    
    # Episode statistics tracking
    episode_counts = [0] * num_envs
    total_episodes = 0
    
    print(f'[INFO] Start training loop (target: {args.steps} steps)')

    isaac_env = env._env.unwrapped
    robot = isaac_env.scene["robot"]
    print(f"  Joint names: {robot.data.joint_names}")


    
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

    print("\n[INFO] Setting up velocity command cycling...")
    velocity_sequence = [
        (1.0, 0.0, 0.0),    # Forward
        #(-1.0, 0.0, 0.0),   # Backward
        #(0.0, -1.0, 0.0),   # Right (negative y in robot frame)
        #(0.0, 1.0, 0.0),    # Left (positive y in robot frame)
    ]
    env.set_custom_velocities(velocity_sequence)
    print("[INFO] Velocity sequence configured:")



    
    while int(step) < args.steps:
        # === Collect experience from all parallel envs ===
        with elements.timer.section('policy'):
            # Run policy on batched observations
            # obs has shape [num_envs, obs_dim] for each key
            # Policy processes all robots at once but generates DIFFERENT actions for each
            carry, acts_new, policy_outs = agent.policy(carry, obs, mode='train')
            
            # acts_new has shape [num_envs, act_dim] - different action per robot!
            acts = {**acts_new, 'reset': obs['is_last'].copy()}
        
        # Step all environments with their individual actions
        with elements.timer.section('env_step'):
            obs_new = env.step(acts)
        
        # Add transitions to replay buffer (one per parallel env)
        with elements.timer.section('collect'):
            for i in range(num_envs):
                # Extract single-env transition
                tran = {k: v[i] for k, v in obs.items()}
                tran.update({k: v[i] for k, v in acts.items() if k != 'reset'})
                tran.update({k: v[i] for k, v in policy_outs.items() if not k.startswith('log/')})
                
                # Add to replay - each robot has its own stream
                replay.add(tran, worker=i)
                
                # Track episode statistics per robot
                episode = episodes[i]
                if tran['is_first']:
                    episode.reset()
                episode.add('score', tran['reward'], agg='sum')
                episode.add('length', 1, agg='sum')
                episode.add('rewards', tran['reward'], agg='stack')
                
                if tran['is_last']:
                    result = episode.result()
                    score = result.pop('score')
                    length = result.pop('length')
                    logger.add({
                        'score': score,
                        'length': length,
                    }, prefix='episode')
                    
                    # Compute reward rate for non-trivial episodes
                    rew = result.pop('rewards')
                    if len(rew) > 1:
                        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
                    epstats.add(result)
                    
                    episode_counts[i] += 1
                    total_episodes += 1
        
        # Update step counter (count all parallel env steps)
        step.increment(num_envs)
        policy_fps.step(num_envs)
        
        # Move to new observations
        obs = obs_new
        
        # === Training ===
        if len(replay) >= args.batch_size * args.batch_length:
            for _ in range(should_train(step)):
                with elements.timer.section('train'):
                    batch = next(stream_train)
                    carry_train[0], outs, mets = agent.train(carry_train[0], batch)
                    train_fps.step(batch_steps)
                    if 'replay' in outs:
                        replay.update(outs['replay'])
                    train_agg.add(mets, prefix='train')
        
        # === Reporting ===
        if should_report(step) and len(replay):
            agg = elements.Agg()
            report_batches = getattr(args, 'report_batches', 1)
            for _ in range(args.consec_report * report_batches):
                carry_report, mets = agent.report(carry_report, next(stream_report))
                agg.add(mets)
            logger.add(agg.result(), prefix='report')
        
        # === Logging ===
        if should_log(step):
            logger.add(train_agg.result())
            logger.add(epstats.result(), prefix='epstats')
            logger.add(replay.stats(), prefix='replay')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add({'timer': elements.timer.stats()['summary']})
            logger.add({'total_episodes': total_episodes})
            logger.write()
        
        # === Saving ===
        if should_save(step):
            cp.save()
    
    print(f'[INFO] Training complete! Total episodes: {total_episodes}')
    print(f'[INFO] Episodes per robot: {episode_counts}')
    logger.close()
    env.close()


def make_batched_env(config):
    """Create a batched Isaac Lab environment."""
    from isaac_env import IsaacBatchedEnv
    
    global isaac_env_config
    
    # Parse task name (remove 'isaac_' prefix if present)
    task = config.task
    if task.startswith('isaac_'):
        task = task[6:]
    
    num_envs = isaac_env_config.get('num_envs', 16)
    
    print(f"[INFO] Creating IsaacBatchedEnv: {task} with {num_envs} parallel envs")
    
    env = IsaacBatchedEnv(
        task=task,
        num_envs=num_envs,
        device=isaac_env_config.get('device', 'cuda:0'),
        seed=config.seed,
        obs_key='vector',
        act_key='action',
    )
    
    # Cache spaces for make_agent
    global _isaac_obs_space, _isaac_act_space
    notlog = lambda k: not k.startswith('log/')
    _isaac_obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    _isaac_act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    
    return env


def make_agent(config):
    """Create the DreamerV3 agent."""
    import elements
    import embodied
    from dreamerv3.agent import Agent
    
    global _isaac_obs_space, _isaac_act_space
    
    # Use cached spaces if available (to avoid creating another env just for spaces)
    if _isaac_obs_space is None or _isaac_act_space is None:
        env = make_env(config, 0)
        notlog = lambda k: not k.startswith('log/')
        _isaac_obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
        _isaac_act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
        # Don't close the env - we'll reuse it
    
    obs_space = _isaac_obs_space
    act_space = _isaac_act_space
    
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def make_logger(config):
    """Create the logger."""
    import elements
    
    step = elements.Counter()
    logdir = config.logdir
    multiplier = 1  # No action repeat for Isaac envs
    
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
    
    for output in config.logger.outputs:
        if output == 'jsonl':
            outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
            outputs.append(elements.logger.JSONLOutput(
                logdir, 'scores.jsonl', 'episode/score'))
        elif output == 'tensorboard':
            outputs.append(elements.logger.TensorBoardOutput(
                logdir, config.logger.fps))
        elif output == 'wandb':
            name = '/'.join(logdir.split('/')[-4:])
            outputs.append(elements.logger.WandBOutput(name))
        elif output == 'scope':
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
            
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode='train'):
    """Create the replay buffer."""
    import elements
    import embodied
    
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f'{config.replica:05}'
    kwargs = dict(
        length=length, capacity=int(capacity), online=config.replay.online,
        chunksize=config.replay.chunksize, directory=directory)

    if config.replay.fracs.uniform < 1 and mode == 'train':
        assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
            'Gradient scaling for low-precision training can produce invalid loss '
            'outputs that are incompatible with prioritized replay.')
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
        selectors = embodied.selectors
        kwargs['selector'] = selectors.Mixture(dict(
            uniform=selectors.Uniform(),
            priority=selectors.Prioritized(**config.replay.prio),
            recency=selectors.Recency(recency),
        ), config.replay.fracs)

    return embodied.Replay(**kwargs)


def make_env(config, index=0, **overrides):
    """
    Create or return the single Isaac Lab environment instance.
    
    Isaac Lab environments are vectorized internally, so we only create ONE
    environment instance that handles all parallel envs. This function returns
    the same instance on subsequent calls.
    """
    from isaac_env import IsaacEnv
    
    global _isaac_env_instance, isaac_env_config
    
    # Return cached instance if it exists
    if _isaac_env_instance is not None:
        return _isaac_env_instance
    
    # Parse task name (remove 'isaac_' prefix if present)
    task = config.task
    if task.startswith('isaac_'):
        task = task[6:]
    
    # Get Isaac-specific config from global variable
    isaac_cfg = isaac_env_config.copy() if isaac_env_config else {}
    isaac_cfg.update(overrides)
    
    # Use the num_envs from isaac_env_config (the actual number of parallel envs)
    num_envs = isaac_cfg.get('num_envs', 1)
    
    print(f"[INFO] Creating Isaac Lab environment: {task} with {num_envs} parallel envs")
    
    env = IsaacEnv(
        task=task,
        num_envs=num_envs,
        device=isaac_cfg.get('device', 'cuda:0'),
        seed=hash((config.seed, index)) % (2 ** 32 - 1),
        obs_key='vector',
        act_key='action',
    )
    
    wrapped_env = wrap_env(env, config)
    
    # Cache the instance
    _isaac_env_instance = wrapped_env
    
    return wrapped_env


def wrap_env(env, config):
    """Apply standard wrappers to the environment."""
    import embodied
    
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_stream(config, replay, mode):
    """Create the data stream from replay buffer."""
    from embodied.core import streams
    fn = bind(replay.sample, config.batch_size, mode)
    stream = streams.Stateless(fn)
    stream = streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Training failed with exception:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise
    finally:
        # close sim app
        simulation_app.close()

