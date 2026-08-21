# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .platform_utils import platform_terrain_type_masks
from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects pit-like terrains (see
    :attr:`UniformThresholdVelocityCommandCfg.pit_terrain_names`) and applies restrictions:
    - For pit-like terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on any pit-like terrain (real-time check every step)
        on_pits = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for terrain_name in self.cfg.pit_terrain_names:
            on_pits |= is_robot_on_terrain(self._env, terrain_name)

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            if self.cfg.heading_command:
                # Actively steer toward world heading 0 (the pit-crossing direction) every step
                # instead of just zeroing the yaw command, so a robot that enters/spawns on the
                # pit off-heading actually turns to align with the crossing direction.
                self.heading_target[pit_env_ids] = 0.0
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[pit_env_ids] - self.robot.data.heading_w[pit_env_ids]
                )
                self.vel_command_b[pit_env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.cfg.ranges.ang_vel_z[0],
                    max=self.cfg.ranges.ang_vel_z[1],
                )
            else:
                self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    pit_terrain_names: tuple[str, ...] = ("pits", "rails", "boxes")
    """Sub-terrain names (keys in ``TerrainGeneratorCfg.sub_terrains``) that should restrict the
    velocity command to forward-only movement (see :meth:`UniformThresholdVelocityCommand._update_command`)."""


class TerrainAwareVelocityCommand(mdp.UniformVelocityCommand):
    """Use dedicated velocity ranges on platform terrain columns."""

    cfg: TerrainAwareVelocityCommandCfg

    def __init__(self, cfg: TerrainAwareVelocityCommandCfg, env: ManagerBasedEnv):
        if not 0.0 <= cfg.platform_terrain_start < 1.0:
            raise ValueError("platform_terrain_start must be in [0, 1).")
        if cfg.planar_deadzone < 0.0:
            raise ValueError("planar_deadzone must be non-negative.")
        if (cfg.platform_descent_terrain_start is None) != (cfg.platform_descent_ranges is None):
            raise ValueError("platform_descent_terrain_start and platform_descent_ranges must be set together.")
        if cfg.platform_descent_terrain_start is not None:
            if not cfg.platform_terrain_start < cfg.platform_descent_terrain_start < 1.0:
                raise ValueError("platform_descent_terrain_start must follow platform_terrain_start and be below 1.")
            if cfg.platform_descent_bidirectional:
                descent_x_range = cfg.platform_descent_ranges.lin_vel_x
                if not 0.0 < descent_x_range[0] <= descent_x_range[1]:
                    raise ValueError("Bidirectional descent requires a positive lin_vel_x magnitude range.")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)

        if self.cfg.planar_deadzone > 0.0:
            moving = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.planar_deadzone
            self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)

        terrain = self._env.scene.terrain
        terrain_generator = terrain.cfg.terrain_generator
        if terrain_generator is None or not hasattr(terrain, "terrain_types"):
            return

        platform_mask, descent_mask = platform_terrain_type_masks(
            terrain.terrain_types[env_ids],
            terrain_generator.num_cols,
            self.cfg.platform_terrain_start,
            self.cfg.platform_descent_terrain_start,
        )
        platform_env_ids = env_ids[platform_mask]
        if len(platform_env_ids) == 0:
            return

        samples = torch.empty(len(platform_env_ids), device=self.device)
        self.vel_command_b[platform_env_ids, 0] = samples.uniform_(*self.cfg.platform_ranges.lin_vel_x)
        self.vel_command_b[platform_env_ids, 1] = samples.uniform_(*self.cfg.platform_ranges.lin_vel_y)
        self.heading_target[platform_env_ids] = samples.uniform_(*self.cfg.platform_ranges.heading)
        self.is_heading_env[platform_env_ids] = True
        self.is_standing_env[platform_env_ids] = False

        if self.cfg.platform_descent_terrain_start is None:
            return

        descent_env_ids = env_ids[descent_mask]
        if len(descent_env_ids) == 0:
            return

        descent_ranges = self.cfg.platform_descent_ranges
        samples = torch.empty(len(descent_env_ids), device=self.device)
        descent_x = samples.uniform_(*descent_ranges.lin_vel_x)
        if self.cfg.platform_descent_bidirectional:
            signs = torch.randint(0, 2, (len(descent_env_ids),), device=self.device).mul_(2).sub_(1)
            descent_x *= signs
        self.vel_command_b[descent_env_ids, 0] = descent_x
        self.vel_command_b[descent_env_ids, 1] = samples.uniform_(*descent_ranges.lin_vel_y)
        self.heading_target[descent_env_ids] = samples.uniform_(*descent_ranges.heading)


@configclass
class TerrainAwareVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for terrain-conditioned velocity sampling."""

    @configclass
    class PlatformRanges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING

    class_type: type = TerrainAwareVelocityCommand
    platform_terrain_start: float = MISSING
    planar_deadzone: float = 0.0
    platform_ranges: PlatformRanges = MISSING
    platform_descent_terrain_start: float | None = None
    platform_descent_ranges: PlatformRanges | None = None
    platform_descent_bidirectional: bool = False


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
