# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for commanded-height locomotion tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HeightVelocityCommand(UniformVelocityCommand):
    """Sample balanced velocity modes and track the position of zero commands.

    Args:
        cfg: Velocity sampling configuration.
        env: Environment that owns the command term.
    """

    cfg: HeightVelocityCommandCfg

    def __init__(self, cfg: HeightVelocityCommandCfg, env: ManagerBasedRLEnv):
        probabilities = (cfg.rel_pure_x_envs, cfg.rel_pure_y_envs, cfg.rel_pure_yaw_envs)
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("Pure-mode probabilities must be in [0, 1].")
        if sum(probabilities) > 1.0:
            raise ValueError("Pure-mode probabilities must sum to at most 1.")
        if cfg.min_linear_speed < 0.0:
            raise ValueError("min_linear_speed must be non-negative.")
        if not 0.0 < cfg.flat_terrain_end <= 1.0:
            raise ValueError("flat_terrain_end must be in (0, 1].")
        if cfg.flat_lin_vel_x[0] >= cfg.flat_lin_vel_x[1]:
            raise ValueError("flat_lin_vel_x must be an increasing range.")

        super().__init__(cfg, env)
        self.hold_position_w = torch.zeros((self.num_envs, 2), device=self.device)
        self.hold_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[
                slice(None) if env_ids is None else env_ids
            ]
        self.hold_active[env_ids] = False
        result = super().reset(env_ids)
        self._update_hold()
        return result

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)

        flat = self._flat_env_mask(env_ids)
        flat_ids = env_ids[flat & ~self.is_standing_env[env_ids]]
        if len(flat_ids) > 0:
            flat_x = torch.empty(len(flat_ids), device=self.device)
            self.vel_command_b[flat_ids, 0] = flat_x.uniform_(*self.cfg.flat_lin_vel_x)

        sample = torch.rand(len(env_ids), device=self.device)
        pure_x_end = self.cfg.rel_pure_x_envs
        pure_y_end = pure_x_end + self.cfg.rel_pure_y_envs
        pure_yaw_end = pure_y_end + self.cfg.rel_pure_yaw_envs
        moving = ~self.is_standing_env[env_ids]

        pure_x_ids = env_ids[moving & (sample < pure_x_end)]
        pure_y_ids = env_ids[moving & (sample >= pure_x_end) & (sample < pure_y_end)]
        pure_yaw_ids = env_ids[moving & (sample >= pure_y_end) & (sample < pure_yaw_end)]

        if len(pure_x_ids) > 0:
            x_command = self.vel_command_b[pure_x_ids, 0]
            x_sign = torch.where(x_command < 0.0, -torch.ones_like(x_command), torch.ones_like(x_command))
            self.vel_command_b[pure_x_ids, 0] = x_sign * torch.clamp(
                torch.abs(x_command), min=self.cfg.min_linear_speed
            )
            self.vel_command_b[pure_x_ids, 1:] = 0.0
            self.heading_target[pure_x_ids] = self.robot.data.heading_w[pure_x_ids]
            self.is_heading_env[pure_x_ids] = True

        if len(pure_y_ids) > 0:
            y_command = self.vel_command_b[pure_y_ids, 1]
            y_sign = torch.where(y_command < 0.0, -torch.ones_like(y_command), torch.ones_like(y_command))
            self.vel_command_b[pure_y_ids, 0] = 0.0
            self.vel_command_b[pure_y_ids, 1] = y_sign * torch.clamp(
                torch.abs(y_command), min=self.cfg.min_linear_speed
            )
            self.vel_command_b[pure_y_ids, 2] = 0.0
            self.heading_target[pure_y_ids] = self.robot.data.heading_w[pure_y_ids]
            self.is_heading_env[pure_y_ids] = True

        if len(pure_yaw_ids) > 0:
            self.vel_command_b[pure_yaw_ids, :2] = 0.0
            self.is_heading_env[pure_yaw_ids] = True

    def _flat_env_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the environments assigned to flat terrain columns."""
        terrain = self._env.scene.terrain
        terrain_generator = terrain.cfg.terrain_generator
        if terrain_generator is None or not hasattr(terrain, "terrain_types"):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        flat_col_max = math.ceil(
            (self.cfg.flat_terrain_end - 0.001) * terrain_generator.num_cols
        )
        return terrain.terrain_types[env_ids] < flat_col_max

    def _update_command(self):
        super()._update_command()
        self._update_hold()

    def _update_hold(self):
        standing = torch.linalg.vector_norm(self.command[:, :3], dim=1) < self.cfg.hold_threshold
        entering = standing & ~self.hold_active
        self.hold_position_w[entering] = self.robot.data.root_pos_w[entering, :2]
        self.hold_active.copy_(standing)


@configclass
class HeightVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`HeightVelocityCommand`."""

    class_type: type = HeightVelocityCommand

    rel_pure_x_envs: float = 0.0
    rel_pure_y_envs: float = 0.0
    rel_pure_yaw_envs: float = 0.0
    min_linear_speed: float = 0.0
    hold_threshold: float = 0.05
    flat_terrain_end: float = 0.10
    flat_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
