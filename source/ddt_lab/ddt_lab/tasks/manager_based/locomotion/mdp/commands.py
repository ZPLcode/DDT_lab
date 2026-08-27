# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command terms for the locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ModeBalancedVelocityCommand(UniformVelocityCommand):
    """Reserve part of the moving command batch for exact pure-X motion."""

    cfg: ModeBalancedVelocityCommandCfg

    def __init__(self, cfg: ModeBalancedVelocityCommandCfg, env: ManagerBasedRLEnv):
        if not 0.0 <= cfg.rel_pure_x_envs <= 1.0:
            raise ValueError(f"rel_pure_x_envs must be in [0, 1]; received {cfg.rel_pure_x_envs}.")
        if cfg.pure_x_min_speed < 0.0:
            raise ValueError(f"pure_x_min_speed must be non-negative; received {cfg.pure_x_min_speed}.")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)

        moving = ~self.is_standing_env[env_ids]
        selected = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_pure_x_envs
        pure_x_ids = env_ids[moving & selected]
        if len(pure_x_ids) == 0:
            return

        x_command = self.vel_command_b[pure_x_ids, 0]
        x_sign = torch.where(x_command < 0.0, -torch.ones_like(x_command), torch.ones_like(x_command))
        self.vel_command_b[pure_x_ids, 0] = x_sign * torch.clamp(torch.abs(x_command), min=self.cfg.pure_x_min_speed)
        self.vel_command_b[pure_x_ids, 1:] = 0.0
        self.is_heading_env[pure_x_ids] = False


@configclass
class ModeBalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity commands with a pure-X fraction among moving environments."""

    class_type: type = ModeBalancedVelocityCommand
    rel_pure_x_envs: float = 0.0
    pure_x_min_speed: float = 0.0


class UniformHeightCommand(CommandTerm):
    """Sample a target base height."""

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformHeightCommand: range={self.cfg.ranges.height}, resample={self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _resample_command(self, env_ids: Sequence[int]):
        samples = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = samples.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        pass

    def _update_metrics(self):
        height = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self.metrics["height_error"] = torch.abs(self.height_command[:, 0] - height)


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformHeightCommand`."""

    class_type: type = UniformHeightCommand
    asset_name: str = MISSING

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
