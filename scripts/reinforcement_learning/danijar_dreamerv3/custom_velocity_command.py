# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom velocity command generator that allows predefined velocity sequences for reproducible testing."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg


class CustomVelocityCommand(UniformVelocityCommand):
    """
    Custom velocity command that allows setting predefined velocity vectors.
    
    This is useful for reproducible testing where you want to specify exact
    velocity commands rather than sampling them randomly.
    
    You can set custom velocities in two ways:
    1. Set `custom_velocities` - a list of [vx, vy, vz] commands that cycle
    2. Call `set_velocity_for_env(env_id, vx, vy, vz)` to set specific velocities
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Storage for custom velocities
        self.custom_velocities = None  # Will be a tensor of shape (N, 3) if set
        self.custom_velocity_index = 0  # Index for cycling through custom velocities
        self.use_custom_velocities = False
        
    def set_custom_velocities(self, velocities: list[tuple[float, float, float]]):
        """
        Set a list of custom velocity commands to cycle through.
        
        Args:
            velocities: List of (vx, vy, vz) tuples. When environments reset,
                       they will be assigned velocities from this list in sequence.
        
        Example:
            >>> cmd_term.set_custom_velocities([
            ...     (1.0, 0.0, 0.0),   # forward
            ...     (0.0, 1.0, 0.0),   # left
            ...     (-1.0, 0.0, 0.0),  # backward
            ...     (0.0, 0.0, 1.0),   # rotate
            ... ])
        """
        self.custom_velocities = torch.tensor(
            velocities, 
            dtype=torch.float32, 
            device=self.device
        )
        self.custom_velocity_index = 0
        self.use_custom_velocities = True
        print(f"[CustomVelocityCommand] Set {len(velocities)} custom velocity commands")
        
    def set_velocity_for_env(self, env_id: int, vx: float, vy: float, vz: float):
        """
        Set a specific velocity command for a single environment.
        
        Args:
            env_id: Environment index
            vx: Linear velocity in x direction (forward/backward)
            vy: Linear velocity in y direction (left/right)  
            vz: Angular velocity around z axis (rotation)
        """
        self.vel_command_b[env_id, 0] = vx
        self.vel_command_b[env_id, 1] = vy
        self.vel_command_b[env_id, 2] = vz
        # Mark as not standing/heading so velocity is actually used
        self.is_standing_env[env_id] = False
        if self.cfg.heading_command:
            self.is_heading_env[env_id] = False
            
    def set_velocity_for_all_envs(self, vx: float, vy: float, vz: float):
        """
        Set the same velocity command for all environments.
        
        Args:
            vx: Linear velocity in x direction (forward/backward)
            vy: Linear velocity in y direction (left/right)
            vz: Angular velocity around z axis (rotation)
        """
        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = vy
        self.vel_command_b[:, 2] = vz
        self.is_standing_env[:] = False
        if self.cfg.heading_command:
            self.is_heading_env[:] = False
    
    def disable_custom_velocities(self):
        """Disable custom velocities and return to random sampling."""
        self.use_custom_velocities = False
        print("[CustomVelocityCommand] Disabled custom velocities, returning to random sampling")
        
    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample velocity commands for specified environments.
        
        If custom velocities are enabled, cycle through the predefined list.
        Otherwise, fall back to random sampling.
        """
        if self.use_custom_velocities and self.custom_velocities is not None:
            # Cycle through custom velocities
            num_custom = len(self.custom_velocities)
            for env_id in env_ids:
                # Assign velocities from the custom list (cycling)
                vel_idx = self.custom_velocity_index % num_custom
                self.vel_command_b[env_id] = self.custom_velocities[vel_idx]
                self.custom_velocity_index += 1
                
            # Update standing envs (keep at 0 for reproducibility)
            self.is_standing_env[env_ids] = False
            
            if self.cfg.heading_command:
                # Disable heading control when using custom velocities
                self.is_heading_env[env_ids] = False
        else:
            # Fall back to parent's random sampling
            super()._resample_command(env_ids)


class FixedVelocityCommand(UniformVelocityCommand):
    """
    A simple fixed velocity command that always commands the same velocity.
    
    Useful for testing with a single fixed velocity target.
    """
    
    def __init__(
        self, 
        cfg: UniformVelocityCommandCfg, 
        env: ManagerBasedEnv,
        vx: float = 1.0,
        vy: float = 0.0,
        vz: float = 0.0
    ):
        super().__init__(cfg, env)
        self.fixed_vx = vx
        self.fixed_vy = vy
        self.fixed_vz = vz
        print(f"[FixedVelocityCommand] Using fixed velocity: vx={vx}, vy={vy}, vz={vz}")
        
    def _resample_command(self, env_ids: Sequence[int]):
        """Always set the same fixed velocity."""
        self.vel_command_b[env_ids, 0] = self.fixed_vx
        self.vel_command_b[env_ids, 1] = self.fixed_vy
        self.vel_command_b[env_ids, 2] = self.fixed_vz
        self.is_standing_env[env_ids] = False
        if self.cfg.heading_command:
            self.is_heading_env[env_ids] = False

