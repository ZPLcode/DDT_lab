# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for commanded-height all-terrain tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ModeBalancedVelocityCommand(UniformVelocityCommand):
    """Sample mutually exclusive axis modes with terrain-specific probabilities.

    Args:
        cfg: Velocity sampling configuration.
        env: Environment that owns the command term.
    """

    cfg: ModeBalancedVelocityCommandCfg

    def __init__(self, cfg: ModeBalancedVelocityCommandCfg, env: ManagerBasedRLEnv):
        if not 0.0 <= cfg.rel_pure_x_envs <= 1.0:
            raise ValueError(f"rel_pure_x_envs must be in [0, 1]; received {cfg.rel_pure_x_envs}.")
        flat_cfg_values = (
            cfg.flat_terrain_end,
            cfg.flat_lin_vel_x,
            cfg.flat_rel_pure_x_envs,
        )
        if any(value is not None for value in flat_cfg_values) and not all(
            value is not None for value in flat_cfg_values
        ):
            raise ValueError(
                "flat_terrain_end, flat_lin_vel_x, and flat_rel_pure_x_envs must all be set or all be None."
            )
        if cfg.flat_terrain_end is not None:
            if not 0.0 < cfg.flat_terrain_end <= 1.0:
                raise ValueError(
                    f"flat_terrain_end must be in (0, 1]; received {cfg.flat_terrain_end}."
                )
            if cfg.flat_lin_vel_x[0] >= cfg.flat_lin_vel_x[1]:
                raise ValueError(
                    f"flat_lin_vel_x must be an increasing range; received {cfg.flat_lin_vel_x}."
                )
            if not 0.0 <= cfg.flat_rel_pure_x_envs <= 1.0:
                raise ValueError(
                    "flat_rel_pure_x_envs must be in [0, 1]; "
                    f"received {cfg.flat_rel_pure_x_envs}."
                )
        if cfg.flat_rel_standing_envs is not None:
            if cfg.flat_terrain_end is None:
                raise ValueError("flat_rel_standing_envs requires flat_terrain_end.")
            if not cfg.rel_standing_envs <= cfg.flat_rel_standing_envs <= 1.0:
                raise ValueError(
                    "flat_rel_standing_envs must be in [rel_standing_envs, 1]; "
                    f"received {cfg.flat_rel_standing_envs}."
                )
        if (cfg.terrain_pure_x_start is None) != (cfg.terrain_rel_pure_x_envs is None):
            raise ValueError(
                "terrain_pure_x_start and terrain_rel_pure_x_envs must either both be set or both be None."
            )
        if cfg.terrain_pure_x_start is not None:
            if not 0.0 <= cfg.terrain_pure_x_start < 1.0:
                raise ValueError(
                    "terrain_pure_x_start must be in [0, 1); "
                    f"received {cfg.terrain_pure_x_start}."
                )
            if not 0.0 <= cfg.terrain_rel_pure_x_envs <= 1.0:
                raise ValueError(
                    "terrain_rel_pure_x_envs must be in [0, 1]; "
                    f"received {cfg.terrain_rel_pure_x_envs}."
                )
            if not 0.0 <= cfg.terrain_rel_pure_y_envs <= 1.0:
                raise ValueError(
                    "terrain_rel_pure_y_envs must be in [0, 1]; "
                    f"received {cfg.terrain_rel_pure_y_envs}."
                )
            if not 0.0 <= cfg.terrain_rel_pure_yaw_envs <= 1.0:
                raise ValueError(
                    "terrain_rel_pure_yaw_envs must be in [0, 1]; "
                    f"received {cfg.terrain_rel_pure_yaw_envs}."
                )
            terrain_mode_probability = (
                cfg.terrain_rel_pure_x_envs
                + cfg.terrain_rel_pure_y_envs
                + cfg.terrain_rel_pure_yaw_envs
            )
            if terrain_mode_probability > 1.0:
                raise ValueError(
                    "terrain pure-X, pure-Y, and pure-yaw probabilities must sum to at most 1; "
                    f"received {terrain_mode_probability}."
                )
            if cfg.flat_terrain_end is not None and cfg.flat_terrain_end > cfg.terrain_pure_x_start:
                raise ValueError(
                    "flat_terrain_end must not overlap terrain_pure_x_start; "
                    f"received {cfg.flat_terrain_end} > {cfg.terrain_pure_x_start}."
                )
        if not 0.0 <= cfg.terrain_pure_x_positive_fraction <= 1.0:
            raise ValueError(
                "terrain_pure_x_positive_fraction must be in [0, 1]; "
                f"received {cfg.terrain_pure_x_positive_fraction}."
            )
        if cfg.pure_x_min_speed < 0.0:
            raise ValueError(f"pure_x_min_speed must be non-negative; received {cfg.pure_x_min_speed}.")
        if cfg.pure_y_min_speed < 0.0:
            raise ValueError(f"pure_y_min_speed must be non-negative; received {cfg.pure_y_min_speed}.")
        if cfg.terrain_pure_x_start is None and (
            cfg.terrain_rel_pure_y_envs > 0.0 or cfg.terrain_rel_pure_yaw_envs > 0.0
        ):
            raise ValueError("terrain pure-Y/pure-yaw probabilities require terrain_pure_x_start.")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)

        pure_x_probability = torch.full(
            (len(env_ids),), self.cfg.rel_pure_x_envs, device=self.device
        )
        pure_y_probability = torch.zeros(len(env_ids), device=self.device)
        pure_yaw_probability = torch.zeros(len(env_ids), device=self.device)
        flat_override = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        terrain_override = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # Increase standing and pure-X coverage on flat terrain.
        if self.cfg.flat_terrain_end is not None:
            terrain = self._env.scene.terrain
            terrain_generator = terrain.cfg.terrain_generator
            if (
                terrain_generator is not None
                and terrain_generator.curriculum
                and hasattr(terrain, "terrain_types")
            ):
                flat_col_max = math.ceil(
                    (self.cfg.flat_terrain_end - 0.001) * terrain_generator.num_cols
                )
                flat_override = terrain.terrain_types[env_ids] < flat_col_max
                pure_x_probability[flat_override] = self.cfg.flat_rel_pure_x_envs
                if self.cfg.flat_rel_standing_envs is not None:
                    base_standing = self.cfg.rel_standing_envs
                    additional_probability = (
                        (self.cfg.flat_rel_standing_envs - base_standing) / (1.0 - base_standing)
                        if base_standing < 1.0
                        else 0.0
                    )
                    additional_standing = (
                        flat_override
                        & ~self.is_standing_env[env_ids]
                        & (torch.rand(len(env_ids), device=self.device) < additional_probability)
                    )
                    additional_standing_ids = env_ids[additional_standing]
                    self.is_standing_env[additional_standing_ids] = True
                    self.vel_command_b[additional_standing_ids] = 0.0

        moving = ~self.is_standing_env[env_ids]

        # Reserve exact-axis modes for stairs and other late terrain columns.
        if self.cfg.terrain_pure_x_start is not None:
            terrain = self._env.scene.terrain
            terrain_generator = terrain.cfg.terrain_generator
            if (
                terrain_generator is not None
                and terrain_generator.curriculum
                and hasattr(terrain, "terrain_types")
            ):
                terrain_col_min = math.ceil(
                    (self.cfg.terrain_pure_x_start - 0.001) * terrain_generator.num_cols
                )
                terrain_override = terrain.terrain_types[env_ids] >= terrain_col_min
                pure_x_probability[terrain_override] = self.cfg.terrain_rel_pure_x_envs
                pure_y_probability[terrain_override] = self.cfg.terrain_rel_pure_y_envs
                pure_yaw_probability[terrain_override] = self.cfg.terrain_rel_pure_yaw_envs

        # A shared sample keeps pure-X, pure-Y, and pure-yaw mutually exclusive.
        mode_sample = torch.rand(len(env_ids), device=self.device)
        pure_x_select = mode_sample < pure_x_probability
        pure_y_select = (mode_sample >= pure_x_probability) & (
            mode_sample < pure_x_probability + pure_y_probability
        )
        pure_yaw_select = (mode_sample >= pure_x_probability + pure_y_probability) & (
            mode_sample < pure_x_probability + pure_y_probability + pure_yaw_probability
        )
        pure_x_ids = env_ids[moving & pure_x_select]
        pure_y_ids = env_ids[moving & pure_y_select]
        pure_yaw_ids = env_ids[moving & pure_yaw_select]

        flat_pure_x_ids = env_ids[moving & pure_x_select & flat_override]
        if len(flat_pure_x_ids) > 0:
            flat_x = torch.empty(len(flat_pure_x_ids), device=self.device)
            self.vel_command_b[flat_pure_x_ids, 0] = flat_x.uniform_(
                *self.cfg.flat_lin_vel_x
            )

        if len(pure_x_ids) > 0:
            x_command = self.vel_command_b[pure_x_ids, 0]
            x_sign = torch.where(x_command < 0.0, -torch.ones_like(x_command), torch.ones_like(x_command))
            self.vel_command_b[pure_x_ids, 0] = x_sign * torch.clamp(
                torch.abs(x_command), min=self.cfg.pure_x_min_speed
            )
            self.vel_command_b[pure_x_ids, 1:] = 0.0
            self.heading_target[pure_x_ids] = self.robot.data.heading_w[pure_x_ids]
            self.is_heading_env[pure_x_ids] = True

        if len(pure_y_ids) > 0:
            y_command = self.vel_command_b[pure_y_ids, 1]
            y_sign = torch.where(y_command < 0.0, -torch.ones_like(y_command), torch.ones_like(y_command))
            self.vel_command_b[pure_y_ids, 0] = 0.0
            self.vel_command_b[pure_y_ids, 1] = y_sign * torch.clamp(
                torch.abs(y_command), min=self.cfg.pure_y_min_speed
            )
            self.vel_command_b[pure_y_ids, 2] = 0.0
            self.heading_target[pure_y_ids] = self.robot.data.heading_w[pure_y_ids]
            self.is_heading_env[pure_y_ids] = True

        if len(pure_yaw_ids) > 0:
            self.vel_command_b[pure_yaw_ids, :2] = 0.0
            self.is_heading_env[pure_yaw_ids] = True

        # Bias stair traversal forward while retaining reverse samples.
        terrain_pure_x_ids = env_ids[moving & pure_x_select & terrain_override]
        if len(terrain_pure_x_ids) > 0:
            positive = (
                torch.rand(len(terrain_pure_x_ids), device=self.device)
                < self.cfg.terrain_pure_x_positive_fraction
            )
            terrain_sign = torch.where(
                positive,
                torch.ones_like(self.vel_command_b[terrain_pure_x_ids, 0]),
                -torch.ones_like(self.vel_command_b[terrain_pure_x_ids, 0]),
            )
            self.vel_command_b[terrain_pure_x_ids, 0] = terrain_sign * torch.abs(
                self.vel_command_b[terrain_pure_x_ids, 0]
            )


class PositionHoldVelocityCommand(ModeBalancedVelocityCommand):
    """Latch the base position when the velocity command becomes zero.

    Args:
        cfg: Velocity sampling configuration.
        env: Environment that owns the command term.
    """

    command_threshold = 0.05

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_position_w = torch.zeros((self.num_envs, 2), device=self.device)
        self.hold_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[
                slice(None) if env_ids is None else env_ids
            ]
        self.hold_active[env_ids] = False
        result = super().reset(env_ids)
        self._update_hold()
        return result

    def _update_command(self):
        super()._update_command()
        self._update_hold()

    def _update_hold(self):
        standing = torch.linalg.vector_norm(self.command[:, :3], dim=1) < self.command_threshold
        entering = standing & ~self.hold_active
        self.hold_position_w[entering] = self.robot.data.root_pos_w[entering, :2]
        self.hold_active.copy_(standing)


@configclass
class ModeBalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity commands with reserved exact-axis moving modes."""

    class_type: type = ModeBalancedVelocityCommand

    rel_pure_x_envs: float = 0.0
    flat_terrain_end: float | None = None
    flat_lin_vel_x: tuple[float, float] | None = None
    flat_rel_pure_x_envs: float | None = None
    flat_rel_standing_envs: float | None = None
    terrain_pure_x_start: float | None = None
    terrain_rel_pure_x_envs: float | None = None
    terrain_rel_pure_y_envs: float = 0.0
    terrain_rel_pure_yaw_envs: float = 0.0
    terrain_pure_x_positive_fraction: float = 0.5
    pure_x_min_speed: float = 0.0
    pure_y_min_speed: float = 0.0
