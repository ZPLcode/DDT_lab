# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp
from isaaclab.utils import configclass

from .platform_utils import platform_terrain_type_start_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TerrainAwareVelocityCommand(mdp.UniformVelocityCommand):
    """Sample dedicated commands for selected terrain columns."""

    cfg: TerrainAwareVelocityCommandCfg

    def __init__(self, cfg: TerrainAwareVelocityCommandCfg, env: ManagerBasedEnv):
        if not 0.0 <= cfg.platform_terrain_start < 1.0:
            raise ValueError("platform_terrain_start must be in [0, 1).")
        if cfg.planar_deadzone < 0.0:
            raise ValueError("planar_deadzone must be non-negative.")
        if (cfg.flat_x_terrain_start is None) != (cfg.flat_x_terrain_end is None):
            raise ValueError("flat-X terrain bounds must be set together.")
        if cfg.flat_x_terrain_start is not None:
            if not 0.0 <= cfg.flat_x_terrain_start < cfg.flat_x_terrain_end <= cfg.platform_terrain_start:
                raise ValueError("flat-X terrain bounds are invalid.")
            if cfg.flat_x_min_speed < 0.0:
                raise ValueError("flat_x_min_speed must be non-negative.")
        if (cfg.platform_descent_terrain_start is None) != (cfg.platform_descent_ranges is None):
            raise ValueError("descent terrain start and ranges must be set together.")
        if cfg.platform_descent_terrain_start is not None:
            if not cfg.platform_terrain_start < cfg.platform_descent_terrain_start < 1.0:
                raise ValueError("platform descent must follow ascent terrain.")
            if cfg.platform_descent_bidirectional:
                x_range = cfg.platform_descent_ranges.lin_vel_x
                if not 0.0 < x_range[0] <= x_range[1]:
                    raise ValueError("bidirectional descent requires positive X magnitudes.")
        super().__init__(cfg, env)

    def _sample_ranges(
        self,
        env_ids: torch.Tensor,
        ranges: TerrainAwareVelocityCommandCfg.PlatformRanges,
        bidirectional_x: bool = False,
    ) -> None:
        if env_ids.numel() == 0:
            return

        samples = torch.empty(env_ids.numel(), device=self.device)
        x_command = samples.uniform_(*ranges.lin_vel_x)
        if bidirectional_x:
            x_command *= torch.randint(0, 2, (env_ids.numel(),), device=self.device).mul_(2).sub_(1)

        self.vel_command_b[env_ids, 0] = x_command
        self.vel_command_b[env_ids, 1] = samples.uniform_(*ranges.lin_vel_y)
        self.heading_target[env_ids] = samples.uniform_(*ranges.heading)
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = False

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)

        if self.cfg.planar_deadzone > 0.0:
            moving = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.planar_deadzone
            self.vel_command_b[env_ids, :2] *= moving.unsqueeze(1)

        terrain = self._env.scene.terrain
        generator = terrain.cfg.terrain_generator
        if generator is None or not hasattr(terrain, "terrain_types"):
            return

        terrain_types = terrain.terrain_types[env_ids]
        if self.cfg.flat_x_terrain_start is not None:
            first = platform_terrain_type_start_index(self.cfg.flat_x_terrain_start, generator.num_cols)
            last = platform_terrain_type_start_index(self.cfg.flat_x_terrain_end, generator.num_cols)
            flat_x_ids = env_ids[(terrain_types >= first) & (terrain_types < last)]
            if flat_x_ids.numel() > 0:
                x_command = self.vel_command_b[flat_x_ids, 0]
                x_sign = torch.where(x_command < 0.0, -torch.ones_like(x_command), torch.ones_like(x_command))
                self.vel_command_b[flat_x_ids, 0] = x_sign * torch.clamp(
                    torch.abs(x_command),
                    min=self.cfg.flat_x_min_speed,
                )
                self.vel_command_b[flat_x_ids, 1:] = 0.0
                self.is_heading_env[flat_x_ids] = False
                self.is_standing_env[flat_x_ids] = False

        platform_start = platform_terrain_type_start_index(self.cfg.platform_terrain_start, generator.num_cols)
        platform_ids = env_ids[terrain_types >= platform_start]
        self._sample_ranges(platform_ids, self.cfg.platform_ranges)

        if self.cfg.platform_descent_terrain_start is not None:
            descent_start = platform_terrain_type_start_index(
                self.cfg.platform_descent_terrain_start,
                generator.num_cols,
            )
            descent_ids = env_ids[terrain_types >= descent_start]
            self._sample_ranges(
                descent_ids,
                self.cfg.platform_descent_ranges,
                bidirectional_x=self.cfg.platform_descent_bidirectional,
            )


@configclass
class TerrainAwareVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for terrain-conditioned velocity commands."""

    @configclass
    class PlatformRanges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING

    class_type: type = TerrainAwareVelocityCommand
    platform_terrain_start: float = MISSING
    planar_deadzone: float = 0.0
    flat_x_terrain_start: float | None = None
    flat_x_terrain_end: float | None = None
    flat_x_min_speed: float = 0.0
    platform_ranges: PlatformRanges = MISSING
    platform_descent_terrain_start: float | None = None
    platform_descent_ranges: PlatformRanges | None = None
    platform_descent_bidirectional: bool = False
