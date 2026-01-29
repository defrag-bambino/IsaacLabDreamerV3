# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab environment wrapper for Danijar's DreamerV3 (embodied framework)."""

import functools

import elements
import embodied
import numpy as np
import torch


class IsaacBatchedEnv:
    """
    Batched environment wrapper for Isaac Lab that properly exposes the internal
    vectorization to DreamerV3.

    Unlike the standard embodied.Env interface which expects single observations,
    this wrapper returns BATCHED observations and accepts BATCHED actions,
    allowing each parallel environment to have different behavior.

    This is designed to work with a custom training loop (train_isaac) that
    handles the batched data directly.
    """

    def __init__(
        self,
        task: str,
        num_envs: int = 16,
        device: str = "cuda:0",
        seed: int = 0,
        obs_key: str = "vector",
        act_key: str = "action",
        render_mode: str = None,
        **kwargs
    ):
        """
        Initialize the batched Isaac Lab environment wrapper.

        Args:
            task: Isaac Lab task name (e.g., "Isaac-Velocity-Rough-Unitree-Go2-v0")
            num_envs: Number of parallel environments in Isaac Lab
            device: Device to run simulation on
            seed: Random seed
            obs_key: Key name for observations in the output dict
            act_key: Key name for actions in the input dict
            render_mode: Render mode for visualization
            **kwargs: Additional arguments passed to environment config
        """
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 - registers tasks
        from isaaclab_tasks.utils import parse_env_cfg

        self._task = task
        self._num_envs = num_envs
        self._device = device
        self._obs_key = obs_key
        self._act_key = act_key

        # Parse and create environment config
        env_cfg = parse_env_cfg(
            task,
            device=device,
            num_envs=num_envs,
            use_fabric=not kwargs.get("disable_fabric", False)
        )

        # Create the Isaac Lab environment
        self._env = gym.make(task, render_mode=render_mode, cfg=env_cfg)

        # Set seed
        if seed is not None:
            self._env.reset(seed=seed)

        # Get single-env spaces for space definitions
        if hasattr(self._env.unwrapped, 'single_observation_space'):
            self._single_obs_space = self._env.unwrapped.single_observation_space
        else:
            self._single_obs_space = self._env.observation_space


        if hasattr(self._env.unwrapped, 'single_action_space'):
            self._single_act_space = self._env.unwrapped.single_action_space
        else:
            self._single_act_space = self._env.action_space

        # Episode tracking for each parallel environment
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Track which envs are at the start of an episode
        self._is_first = np.ones(num_envs, dtype=bool)

        # Initialize with a reset
        self._last_obs = None
        self._needs_reset = True
        
        # Custom velocity tracking
        self._custom_velocities = None
        self._custom_velocity_index = 0

    @property
    def num_envs(self):
        """Return the number of parallel environments."""
        return self._num_envs

    @property
    def env(self):
        """Return the underlying Isaac Lab environment."""
        return self._env

    @functools.cached_property
    def obs_space(self):
        """Return the observation space in embodied format (single-env space)."""
        spaces = {}

        if hasattr(self._single_obs_space, 'spaces'):
            if "policy" in self._single_obs_space.spaces:
                policy_space = self._single_obs_space.spaces["policy"]
                if hasattr(policy_space, 'spaces'):
                    # Flatten the nested policy dict into individual spaces
                    for k, v in policy_space.spaces.items():
                        spaces[k] = self._convert_space(v)
                else:
                    spaces[self._obs_key] = self._convert_space(policy_space)
            else:
                for k, v in self._single_obs_space.spaces.items():
                    spaces[k] = self._convert_space(v)
        else:
            spaces[self._obs_key] = self._convert_space(self._single_obs_space)
        
        # Add required embodied keys
        spaces['reward'] = elements.Space(np.float32)
        spaces['is_first'] = elements.Space(bool)
        spaces['is_last'] = elements.Space(bool)
        spaces['is_terminal'] = elements.Space(bool)
        
        return spaces

    @functools.cached_property
    def act_space(self):
        """Return the action space in embodied format (single-env space)."""
        spaces = {}
        spaces[self._act_key] = self._convert_space(self._single_act_space)
        spaces['reset'] = elements.Space(bool)
        return spaces

    def reset(self):
        """
        Reset all environments.
        
        Returns:
            Batched observation dict with shape [num_envs, ...] for each key
        """
        obs, info = self._env.reset()
        self._episode_rewards = np.zeros(self._num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self._num_envs, dtype=np.int32)
        self._is_first = np.ones(self._num_envs, dtype=bool)
        self._needs_reset = False
        self._last_obs = obs
        
        return self._make_batched_obs(obs, 
                                       np.zeros(self._num_envs, dtype=np.float32),
                                       is_first=np.ones(self._num_envs, dtype=bool),
                                       is_last=np.zeros(self._num_envs, dtype=bool),
                                       is_terminal=np.zeros(self._num_envs, dtype=bool))

    def step(self, actions):
        """
        Step all environments with batched actions.
        
        Args:
            actions: Dict with batched actions, shape [num_envs, act_dim]
                     Keys: 'action' (required), 'reset' (optional)
        
        Returns:
            Batched observation dict with shape [num_envs, ...] for each key
        """
        if self._needs_reset:
            return self.reset()
        
        # Get batched actions
        act = actions[self._act_key]
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act, device=self._env.unwrapped.device, dtype=torch.float32)
        
        # Ensure correct shape [num_envs, act_dim]
        if act.ndim == 1:
            act = act.unsqueeze(0).expand(self._num_envs, -1)
        
        # Step the environment with batched actions
        obs, reward, terminated, truncated, info = self._env.step(act)
        
        # Convert to numpy
        if isinstance(reward, torch.Tensor):
            reward_np = reward.detach().cpu().numpy().flatten()
        else:
            reward_np = np.asarray(reward).flatten()
        
        if isinstance(terminated, torch.Tensor):
            terminated_np = terminated.detach().cpu().numpy().flatten().astype(bool)
        else:
            terminated_np = np.asarray(terminated).flatten().astype(bool)
            
        if isinstance(truncated, torch.Tensor):
            truncated_np = truncated.detach().cpu().numpy().flatten().astype(bool)
        else:
            truncated_np = np.asarray(truncated).flatten().astype(bool)
        
        # Update episode tracking
        self._episode_rewards += reward_np
        self._episode_lengths += 1
        
        # Determine is_last (done) for each env
        is_last = terminated_np | truncated_np
        
        # Build observation
        # Note: is_first should be True for envs that were auto-reset
        # Isaac Lab auto-resets, so after is_last, the obs is already from a new episode
        result = self._make_batched_obs(
            obs, reward_np,
            is_first=self._is_first.copy(),
            is_last=is_last,
            is_terminal=terminated_np
        )
        
        # Update is_first for next step (will be True for envs that just finished)
        # Since Isaac Lab auto-resets, envs that were done will have fresh obs
        self._is_first = is_last.copy()
        
        # Reset episode tracking for finished envs
        self._episode_rewards = np.where(is_last, 0.0, self._episode_rewards)
        self._episode_lengths = np.where(is_last, 0, self._episode_lengths)
        
        self._last_obs = obs
        
        return result

    def _make_batched_obs(self, obs, reward, is_first, is_last, is_terminal):
        """
        Create batched observation dict.

        Returns dict with shape [num_envs, ...] for each key.
        """
        result = {}

        # Handle Isaac Lab observation format
        if isinstance(obs, dict):
            if "policy" in obs:
                policy_obs = obs["policy"]
                if isinstance(policy_obs, dict):
                    # Flattened observation format - extract each component
                    for key, obs_tensor in policy_obs.items():
                        if isinstance(obs_tensor, torch.Tensor):
                            obs_np = obs_tensor.detach().cpu().numpy()
                        else:
                            obs_np = np.asarray(obs_tensor)

                        # Should be [num_envs, obs_dim] for this component
                        assert obs_np.shape[0] == self._num_envs, f"Expected {self._num_envs} envs for {key}, got {obs_np.shape[0]}"
                        result[key] = obs_np.astype(np.float32)
                else:
                    # Single tensor format (fallback)
                    if isinstance(policy_obs, torch.Tensor):
                        obs_np = policy_obs.detach().cpu().numpy()
                    else:
                        obs_np = np.asarray(policy_obs)

                    assert obs_np.shape[0] == self._num_envs, f"Expected {self._num_envs} envs, got {obs_np.shape[0]}"
                    result[self._obs_key] = obs_np.astype(np.float32)
            else:
                # Handle other observation formats if needed
                for key, obs_tensor in obs.items():
                    if key == "policy":
                        continue  # Already handled above
                    if isinstance(obs_tensor, torch.Tensor):
                        obs_np = obs_tensor.detach().cpu().numpy()
                    else:
                        obs_np = np.asarray(obs_tensor)

                    if obs_np.ndim > 0 and obs_np.shape[0] == self._num_envs:
                        result[key] = obs_np.astype(np.float32)
        else:
            # Single tensor format
            if isinstance(obs, torch.Tensor):
                obs_np = obs.detach().cpu().numpy()
            else:
                obs_np = np.asarray(obs)

            assert obs_np.shape[0] == self._num_envs, f"Expected {self._num_envs} envs, got {obs_np.shape[0]}"
            result[self._obs_key] = obs_np.astype(np.float32)

        result['reward'] = reward.astype(np.float32)
        result['is_first'] = is_first
        result['is_last'] = is_last
        result['is_terminal'] = is_terminal
        
        return result

    def _convert_space(self, space):
        """Convert gym space to elements.Space."""
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        elif hasattr(space, 'shape'):
            return elements.Space(np.float32, space.shape, space.low, space.high)
        else:
            raise ValueError(f"Unknown space type: {type(space)}")

    def render(self):
        """Render the environment."""
        return self._env.render()
    
    def set_custom_velocities(self, velocities: list[tuple[float, float, float]]):
        """
        Set custom velocity commands for reproducible testing.
        
        Args:
            velocities: List of (vx, vy, vz) tuples where:
                - vx: forward/backward velocity (m/s)
                - vy: left/right velocity (m/s)
                - vz: rotation velocity (rad/s)
        
        Example:
            >>> env.set_custom_velocities([
            ...     (1.0, 0.0, 0.0),   # forward
            ...     (0.0, 1.0, 0.0),   # left strafe
            ...     (-1.0, 0.0, 0.0),  # backward
            ...     (0.0, 0.0, 1.0),   # rotate right
            ... ])
        """
        import torch
        
        # Access the command manager
        if hasattr(self._env.unwrapped, 'command_manager'):
            cmd_mgr = self._env.unwrapped.command_manager
            if 'base_velocity' in cmd_mgr._terms:
                cmd_term = cmd_mgr._terms['base_velocity']
                
                # Store the custom velocities and setup cycling
                self._custom_velocities = torch.tensor(
                    velocities, 
                    dtype=torch.float32, 
                    device=cmd_term.vel_command_b.device
                )
                self._custom_velocity_index = 0
                
                # Immediately set velocities for all environments
                num_custom = len(velocities)
                print(f"[INFO] Setting initial velocities for {self._num_envs} environment(s):")
                for env_id in range(self._num_envs):
                    vel_idx = env_id % num_custom
                    cmd_term.vel_command_b[env_id] = self._custom_velocities[vel_idx]
                    # Disable standing/heading to ensure velocity is used
                    cmd_term.is_standing_env[env_id] = False
                    if hasattr(cmd_term, 'is_heading_env'):
                        cmd_term.is_heading_env[env_id] = False
                    vx, vy, vz = velocities[vel_idx]
                    print(f"  Env {env_id}: vx={vx:+.2f}, vy={vy:+.2f}, vz={vz:+.2f}")
                
                # Override the _resample_command method to use our velocities
                # Store reference to self for closure
                env_wrapper = self
                custom_vels = self._custom_velocities
                
                def custom_resample(env_ids):
                    # Cycle through custom velocities
                    num_vels = len(custom_vels)
                    import torch
                    env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
                    
                    for i, env_id in enumerate(env_ids_list):
                        vel_idx = (env_wrapper._custom_velocity_index + i) % num_vels
                        cmd_term.vel_command_b[env_id] = custom_vels[vel_idx]
                    
                    env_wrapper._custom_velocity_index = (env_wrapper._custom_velocity_index + len(env_ids_list)) % num_vels
                    
                    # Disable standing/heading
                    cmd_term.is_standing_env[env_ids] = False
                    if hasattr(cmd_term, 'is_heading_env'):
                        cmd_term.is_heading_env[env_ids] = False
                
                # Replace the method
                cmd_term._resample_command = custom_resample
                
                print(f"[INFO] Custom velocities set: {len(velocities)} commands will cycle")
            else:
                print("[Warning] No 'base_velocity' command term found")
        else:
            print("[Warning] Environment doesn't have a command_manager")
    
    def set_velocity_for_env(self, env_id: int, vx: float, vy: float, vz: float):
        """
        Set velocity command for a specific environment.
        
        Args:
            env_id: Environment index (0 to num_envs-1)
            vx: forward/backward velocity (m/s)
            vy: left/right velocity (m/s)
            vz: rotation velocity (rad/s)
        """
        if hasattr(self._env.unwrapped, 'command_manager'):
            cmd_mgr = self._env.unwrapped.command_manager
            if 'base_velocity' in cmd_mgr._terms:
                cmd_term = cmd_mgr._terms['base_velocity']
                cmd_term.vel_command_b[env_id, 0] = vx
                cmd_term.vel_command_b[env_id, 1] = vy
                cmd_term.vel_command_b[env_id, 2] = vz
                cmd_term.is_standing_env[env_id] = False
                if hasattr(cmd_term, 'is_heading_env'):
                    cmd_term.is_heading_env[env_id] = False
                    
    def set_velocity_for_all_envs(self, vx: float, vy: float, vz: float):
        """
        Set the same velocity command for all environments.
        
        Args:
            vx: forward/backward velocity (m/s)
            vy: left/right velocity (m/s)
            vz: rotation velocity (rad/s)
        """
        if hasattr(self._env.unwrapped, 'command_manager'):
            cmd_mgr = self._env.unwrapped.command_manager
            if 'base_velocity' in cmd_mgr._terms:
                cmd_term = cmd_mgr._terms['base_velocity']
                cmd_term.vel_command_b[:, 0] = vx
                cmd_term.vel_command_b[:, 1] = vy
                cmd_term.vel_command_b[:, 2] = vz
                cmd_term.is_standing_env[:] = False
                if hasattr(cmd_term, 'is_heading_env'):
                    cmd_term.is_heading_env[:] = False
    
    def get_current_velocity_commands(self):
        """
        Get the current velocity commands for all environments.
        
        Returns:
            numpy array of shape [num_envs, 3] with [vx, vy, vz] for each env
        """
        if hasattr(self._env.unwrapped, 'command_manager'):
            cmd_mgr = self._env.unwrapped.command_manager
            if 'base_velocity' in cmd_mgr._terms:
                cmd_term = cmd_mgr._terms['base_velocity']
                if hasattr(cmd_term, 'vel_command_b'):
                    return cmd_term.vel_command_b.detach().cpu().numpy()
        return None

    def close(self):
        """Close the environment."""
        try:
            self._env.close()
        except Exception:
            pass


class IsaacEnv(embodied.Env):
    """
    Wrapper for Isaac Lab environments that follows the embodied.Env interface.
    
    This wrapper handles the conversion from Isaac Lab's batched environment format
    to the single-environment interface expected by embodied's Driver.
    
    Since Isaac Lab environments are inherently vectorized, this wrapper exposes
    a single environment from the batch at a time (index-based access).
    """

    def __init__(
        self,
        task: str,
        num_envs: int = 1,
        device: str = "cuda:0",
        seed: int = 0,
        obs_key: str = "vector",
        act_key: str = "action",
        render_mode: str = None,
        **kwargs
    ):
        """
        Initialize the Isaac Lab environment wrapper.
        
        Args:
            task: Isaac Lab task name (e.g., "Isaac-Velocity-Rough-Unitree-Go2-v0")
            num_envs: Number of parallel environments in Isaac Lab
            device: Device to run simulation on
            seed: Random seed
            obs_key: Key name for observations in the output dict
            act_key: Key name for actions in the input dict  
            render_mode: Render mode for visualization
            **kwargs: Additional arguments passed to environment config
        """
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 - registers tasks
        from isaaclab_tasks.utils import parse_env_cfg
        
        self._task = task
        self._num_envs = num_envs
        self._device = device
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
        
        # Parse and create environment config
        env_cfg = parse_env_cfg(
            task,
            device=device,
            num_envs=num_envs,
            use_fabric=not kwargs.get("disable_fabric", False)
        )
        
        # Create the Isaac Lab environment
        self._env = gym.make(task, render_mode=render_mode, cfg=env_cfg)
        
        # Set seed
        if seed is not None:
            self._env.reset(seed=seed)
        
        # Get observation and action spaces
        # Isaac Lab uses batched spaces, we need single-env spaces
        if hasattr(self._env.unwrapped, 'single_observation_space'):
            self._single_obs_space = self._env.unwrapped.single_observation_space
        else:
            # Fallback: unbatch the space manually
            obs_space = self._env.observation_space
            if hasattr(obs_space, 'spaces'):
                self._single_obs_space = {
                    k: gym.spaces.Box(
                        low=v.low[0] if v.low.ndim > 1 else v.low,
                        high=v.high[0] if v.high.ndim > 1 else v.high,
                        dtype=v.dtype
                    )
                    for k, v in obs_space.spaces.items()
                }
            else:
                self._single_obs_space = obs_space
                
        if hasattr(self._env.unwrapped, 'single_action_space'):
            self._single_act_space = self._env.unwrapped.single_action_space
        else:
            act_space = self._env.action_space
            self._single_act_space = act_space
        
        # Episode tracking
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        # Current observation cache
        self._current_obs = None
        self._step_count = 0

    @property
    def env(self):
        """Return the underlying Isaac Lab environment."""
        return self._env

    @property
    def info(self):
        """Return the last info dict."""
        return self._info

    @functools.cached_property
    def obs_space(self):
        """Return the observation space in embodied format."""
        spaces = {}
        
        # Handle Isaac Lab's observation space format
        if isinstance(self._single_obs_space, dict):
            if "policy" in self._single_obs_space:
                space = self._single_obs_space["policy"]
                spaces[self._obs_key] = self._convert_space(space)
            else:
                for k, v in self._single_obs_space.items():
                    spaces[k] = self._convert_space(v)
        elif hasattr(self._single_obs_space, 'spaces'):
            if "policy" in self._single_obs_space.spaces:
                space = self._single_obs_space.spaces["policy"]
                spaces[self._obs_key] = self._convert_space(space)
            else:
                for k, v in self._single_obs_space.spaces.items():
                    spaces[k] = self._convert_space(v)
        else:
            spaces[self._obs_key] = self._convert_space(self._single_obs_space)
        
        # Add required embodied keys
        spaces['reward'] = elements.Space(np.float32)
        spaces['is_first'] = elements.Space(bool)
        spaces['is_last'] = elements.Space(bool)
        spaces['is_terminal'] = elements.Space(bool)
        
        return spaces

    @functools.cached_property
    def act_space(self):
        """Return the action space in embodied format."""
        spaces = {}
        spaces[self._act_key] = self._convert_space(self._single_act_space)
        spaces['reset'] = elements.Space(bool)
        return spaces

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Dict with 'action' and 'reset' keys
            
        Returns:
            Observation dict with obs, reward, is_first, is_last, is_terminal
        """
        if action['reset'] or self._done:
            self._done = False
            self._step_count = 0
            obs, info = self._env.reset()
            self._info = info
            self._episode_rewards = np.zeros(self._num_envs, dtype=np.float32)
            self._episode_lengths = np.zeros(self._num_envs, dtype=np.int32)
            return self._obs(obs, 0.0, is_first=True)
        
        # Get action tensor
        act = action[self._act_key]
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act, device=self._env.unwrapped.device)
        
        # Expand action to batch size if needed (single env case)
        if act.ndim == 1:
            act = act.unsqueeze(0).expand(self._num_envs, -1)
        
        # Step the environment
        obs, reward, terminated, truncated, info = self._env.step(act)
        self._info = info
        self._step_count += 1
        
        # Convert tensors to numpy
        if isinstance(reward, torch.Tensor):
            reward_np = reward.detach().cpu().numpy()
        else:
            reward_np = np.asarray(reward)
        
        if isinstance(terminated, torch.Tensor):
            terminated_np = terminated.detach().cpu().numpy()
        else:
            terminated_np = np.asarray(terminated)
            
        if isinstance(truncated, torch.Tensor):
            truncated_np = truncated.detach().cpu().numpy()
        else:
            truncated_np = np.asarray(truncated)
        
        # Flatten arrays
        reward_np = reward_np.flatten()
        terminated_np = terminated_np.flatten().astype(bool)
        truncated_np = truncated_np.flatten().astype(bool)
        
        # Update episode tracking
        self._episode_rewards += reward_np
        self._episode_lengths += 1
        
        # Check for done (any env done triggers is_last for single-env interface)
        done = terminated_np.any() or truncated_np.any()
        is_terminal = terminated_np.any()
        self._done = done
        
        # Use mean reward across parallel envs for single-env interface
        mean_reward = float(reward_np.mean())
        
        return self._obs(
            obs, 
            mean_reward,
            is_last=done,
            is_terminal=is_terminal
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        """Convert observation to embodied format."""
        result = {}
        
        # Handle Isaac Lab observation format
        if isinstance(obs, dict):
            if "policy" in obs:
                obs_tensor = obs["policy"]
            else:
                obs_tensor = list(obs.values())[0]
        else:
            obs_tensor = obs
            
        # Convert to numpy
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor)
        
        # Take mean across batch dimension for single-env interface
        if obs_np.ndim > 1 and obs_np.shape[0] == self._num_envs:
            obs_np = obs_np[0]  # Take first env's observation
        
        result[self._obs_key] = obs_np.astype(np.float32)
        result['reward'] = np.float32(reward)
        result['is_first'] = is_first
        result['is_last'] = is_last
        result['is_terminal'] = is_terminal
        
        return result

    def _convert_space(self, space):
        """Convert gym space to elements.Space."""
        if hasattr(space, 'n'):
            # Discrete space
            return elements.Space(np.int32, (), 0, space.n)
        elif hasattr(space, 'shape'):
            # Box space - handle batched dimensions
            shape = space.shape
            low = space.low
            high = space.high
            
            # If space is batched (first dim is num_envs), remove it
            if len(shape) > 1 and shape[0] == self._num_envs:
                shape = shape[1:]
                if low.ndim > 1:
                    low = low[0]
                if high.ndim > 1:
                    high = high[0]
            
            return elements.Space(np.float32, shape, low, high)
        else:
            raise ValueError(f"Unknown space type: {type(space)}")

    def render(self):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Close the environment."""
        try:
            self._env.close()
        except Exception:
            pass


class IsaacVecEnv(embodied.Env):
    """
    Vectorized wrapper for Isaac Lab environments that exposes all parallel
    environments to the embodied Driver.
    
    This is useful when you want to leverage Isaac Lab's internal parallelization
    directly with embodied's training loop.
    """

    def __init__(
        self,
        task: str,
        index: int = 0,
        num_envs: int = 1,
        device: str = "cuda:0",
        seed: int = 0,
        obs_key: str = "vector",
        act_key: str = "action",
        render_mode: str = None,
        _shared_env=None,
        **kwargs
    ):
        """
        Initialize a single-index view into a shared Isaac Lab environment.
        
        Args:
            task: Isaac Lab task name
            index: Index of this environment in the batch
            num_envs: Total number of parallel environments
            device: Device to run simulation on
            seed: Random seed
            obs_key: Key name for observations
            act_key: Key name for actions
            render_mode: Render mode
            _shared_env: Shared environment instance (internal use)
            **kwargs: Additional arguments
        """
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        
        self._task = task
        self._index = index
        self._num_envs = num_envs
        self._device = device
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        
        # Use shared environment or create new one
        if _shared_env is not None:
            self._env = _shared_env
            self._owns_env = False
        else:
            env_cfg = parse_env_cfg(
                task,
                device=device,
                num_envs=num_envs,
                use_fabric=not kwargs.get("disable_fabric", False)
            )
            self._env = gym.make(task, render_mode=render_mode, cfg=env_cfg)
            if seed is not None:
                self._env.reset(seed=seed)
            self._owns_env = True
        
        # Get single-env spaces
        if hasattr(self._env.unwrapped, 'single_observation_space'):
            self._single_obs_space = self._env.unwrapped.single_observation_space
        else:
            self._single_obs_space = self._env.observation_space
            
        if hasattr(self._env.unwrapped, 'single_action_space'):
            self._single_act_space = self._env.unwrapped.single_action_space
        else:
            self._single_act_space = self._env.action_space
        
        # Cached observation for this index
        self._current_obs = None
        self._current_reward = 0.0
        self._current_terminated = False
        self._current_truncated = False

    @functools.cached_property
    def obs_space(self):
        """Return observation space."""
        spaces = {}
        
        if hasattr(self._single_obs_space, 'spaces'):
            if "policy" in self._single_obs_space.spaces:
                space = self._single_obs_space.spaces["policy"]
                spaces[self._obs_key] = self._convert_space(space)
            else:
                for k, v in self._single_obs_space.spaces.items():
                    spaces[k] = self._convert_space(v)
        else:
            spaces[self._obs_key] = self._convert_space(self._single_obs_space)
        
        spaces['reward'] = elements.Space(np.float32)
        spaces['is_first'] = elements.Space(bool)
        spaces['is_last'] = elements.Space(bool)
        spaces['is_terminal'] = elements.Space(bool)
        
        return spaces

    @functools.cached_property
    def act_space(self):
        """Return action space."""
        spaces = {}
        spaces[self._act_key] = self._convert_space(self._single_act_space)
        spaces['reset'] = elements.Space(bool)
        return spaces

    def step(self, action):
        """Execute one step for this environment index."""
        if action['reset'] or self._done:
            self._done = False
            # Only reset if we own the env, otherwise use cached obs
            if self._owns_env or self._current_obs is None:
                obs, _ = self._env.reset()
                self._cache_obs(obs, 0.0, False, False)
            return self._get_obs(is_first=True)
        
        # Get action for this index
        act = action[self._act_key]
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act, device=self._env.unwrapped.device)
        
        # Create full action tensor (only for owned env)
        if self._owns_env:
            full_act = torch.zeros(
                (self._num_envs,) + act.shape, 
                dtype=act.dtype, 
                device=act.device
            )
            full_act[self._index] = act
            
            obs, reward, terminated, truncated, _ = self._env.step(full_act)
            
            # Cache for this index
            reward_np = reward.detach().cpu().numpy().flatten()[self._index]
            term = terminated.detach().cpu().numpy().flatten()[self._index]
            trunc = truncated.detach().cpu().numpy().flatten()[self._index]
            self._cache_obs(obs, reward_np, term, trunc)
        
        done = self._current_terminated or self._current_truncated
        self._done = done
        
        return self._get_obs(
            is_last=done,
            is_terminal=self._current_terminated
        )

    def _cache_obs(self, obs, reward, terminated, truncated):
        """Cache observation data for this index."""
        if isinstance(obs, dict):
            if "policy" in obs:
                obs_tensor = obs["policy"]
            else:
                obs_tensor = list(obs.values())[0]
        else:
            obs_tensor = obs
            
        if isinstance(obs_tensor, torch.Tensor):
            self._current_obs = obs_tensor[self._index].detach().cpu().numpy()
        else:
            self._current_obs = np.asarray(obs_tensor)[self._index]
        
        self._current_reward = float(reward)
        self._current_terminated = bool(terminated)
        self._current_truncated = bool(truncated)

    def _get_obs(self, is_first=False, is_last=False, is_terminal=False):
        """Get observation dict for this index."""
        return {
            self._obs_key: self._current_obs.astype(np.float32),
            'reward': np.float32(self._current_reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def _convert_space(self, space):
        """Convert gym space to elements.Space."""
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        elif hasattr(space, 'shape'):
            return elements.Space(np.float32, space.shape, space.low, space.high)
        else:
            raise ValueError(f"Unknown space type: {type(space)}")

    def close(self):
        """Close the environment if owned."""
        if self._owns_env:
            try:
                self._env.close()
            except Exception:
                pass

