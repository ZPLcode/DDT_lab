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

import torch
from isaaclab.envs import mdp
from isaaclab.utils import configclass

from .platform_utils import platform_terrain_type_masks

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TerrainAwareVelocityCommand(mdp.UniformVelocityCommand):
    """Use dedicated velocity ranges on platform terrain columns."""

    cfg: TerrainAwareVelocityCommandCfg

    def __init__(self, cfg: TerrainAwareVelocityCommandCfg, env: ManagerBasedEnv):
        if not 0.0 <= cfg.platform_terrain_start < 1.0:
            raise ValueError("platform_terrain_start must be in [0, 1).")
        if cfg.planar_deadzone < 0.0:
            raise ValueError("planar_deadzone must be non-negative.")
        if not cfg.platform_terrain_start < cfg.platform_descent_terrain_start < 1.0:
            raise ValueError("platform_descent_terrain_start must follow platform_terrain_start and be below 1.")
        if cfg.platform_descent_bidirectional:
            descent_x_range = cfg.platform_descent_ranges.lin_vel_x
            if not 0.0 < descent_x_range[0] <= descent_x_range[1]:
                raise ValueError("Bidirectional descent requires a positive lin_vel_x magnitude range.")
        super().__init__(cfg, env)

    def _sample_ranges(
        self,
        env_ids: torch.Tensor,
        ranges: TerrainAwareVelocityCommandCfg.PlatformRanges,
        bidirectional_x: bool = False,
    ):
        """Sample terrain-specific commands for the selected environments."""
        if env_ids.numel() == 0:
            return

        samples = torch.empty(env_ids.numel(), device=self.device)
        lin_vel_x = samples.uniform_(*ranges.lin_vel_x)
        if bidirectional_x:
            signs = torch.randint(0, 2, (env_ids.numel(),), device=self.device).mul_(2).sub_(1)
            lin_vel_x *= signs

        self.vel_command_b[env_ids, 0] = lin_vel_x
        self.vel_command_b[env_ids, 1] = samples.uniform_(*ranges.lin_vel_y)
        self.heading_target[env_ids] = samples.uniform_(*ranges.heading)
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = False

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

        ascent_mask, descent_mask = platform_terrain_type_masks(
            terrain.terrain_types[env_ids],
            terrain_generator.num_cols,
            self.cfg.platform_terrain_start,
            self.cfg.platform_descent_terrain_start,
        )
        self._sample_ranges(env_ids[ascent_mask], self.cfg.platform_ranges)
        self._sample_ranges(
            env_ids[descent_mask],
            self.cfg.platform_descent_ranges,
            bidirectional_x=self.cfg.platform_descent_bidirectional,
        )


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
    platform_descent_terrain_start: float = MISSING
    platform_descent_ranges: PlatformRanges = MISSING
    platform_descent_bidirectional: bool = False
