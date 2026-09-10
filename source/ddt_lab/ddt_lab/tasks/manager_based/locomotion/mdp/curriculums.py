# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .platform_utils import platform_terrain_type_start_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_terrain_type_start: float | None = None,
    move_up_distance_override: float | None = None,
    clean_ascent_reward_term: str | None = None,
    clean_ascent_terrain_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Platform descent columns may use a shorter promotion distance.
    Selected ascent columns can additionally require both rear wheels to have
    landed cleanly, no wall-contact flag during the episode, and no termination.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    if (move_up_terrain_type_start is None) != (move_up_distance_override is None):
        raise ValueError("move_up_terrain_type_start and move_up_distance_override must be set together.")
    if move_up_terrain_type_start is not None and not 0.0 <= move_up_terrain_type_start < 1.0:
        raise ValueError("move_up_terrain_type_start must be in [0, 1).")
    if move_up_distance_override is not None and move_up_distance_override <= 0.0:
        raise ValueError("move_up_distance_override must be positive.")
    if (clean_ascent_reward_term is None) != (clean_ascent_terrain_range is None):
        raise ValueError("Clean ascent reward term and terrain range must be set together.")
    if clean_ascent_terrain_range is not None:
        if not 0.0 <= clean_ascent_terrain_range[0] < clean_ascent_terrain_range[1] < 1.0:
            raise ValueError("Clean ascent terrain range must be ordered within [0, 1).")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up_distance = torch.full_like(distance, terrain.cfg.terrain_generator.size[0] / 2)
    if move_up_terrain_type_start is not None:
        terrain_type_min = platform_terrain_type_start_index(
            move_up_terrain_type_start,
            terrain.cfg.terrain_generator.num_cols,
        )
        override_mask = terrain.terrain_types[env_ids] >= terrain_type_min
        move_up_distance = torch.where(
            override_mask,
            torch.full_like(move_up_distance, move_up_distance_override),
            move_up_distance,
        )
    move_up = distance > move_up_distance
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    if clean_ascent_reward_term is not None:
        first = platform_terrain_type_start_index(
            clean_ascent_terrain_range[0], terrain.cfg.terrain_generator.num_cols
        )
        last = platform_terrain_type_start_index(
            clean_ascent_terrain_range[1], terrain.cfg.terrain_generator.num_cols
        )
        ascent = (terrain.terrain_types[env_ids] >= first) & (terrain.terrain_types[env_ids] < last)
        term = env.reward_manager.get_term_cfg(clean_ascent_reward_term).func
        clean = term.episode_clean_landed[env_ids].all(dim=1) & ~term.episode_wall_touched[env_ids]
        clean &= ~env.termination_manager.terminated[env_ids]
        # Apply after the baseline demotion decision: a dirty traversal holds
        # its level, even when a large command would otherwise request demotion.
        move_up &= ~ascent | clean
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())


def terrain_level_statistics(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    terrain_type_groups: dict[str, Sequence[str]],
    statistic_names: Sequence[str] = ("mean", "median", "p90", "max"),
    include_level_fractions: bool = False,
) -> dict[str, torch.Tensor]:
    """Report curriculum levels for selected terrain groups."""
    del env_ids
    terrain: TerrainImporter = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    if generator is None:
        raise ValueError("terrain statistics require a terrain generator.")

    supported = {"mean", "median", "p90", "max"}
    statistic_names = tuple(statistic_names)
    unknown = set(statistic_names) - supported
    if unknown:
        raise ValueError(f"unsupported terrain statistics: {sorted(unknown)}")

    names = list(generator.sub_terrains)
    proportions = [float(cfg.proportion) for cfg in generator.sub_terrains.values()]
    total = sum(proportions)
    if total <= 0.0:
        raise ValueError("terrain proportions must sum to a positive value.")

    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    column_names = []
    for column in range(generator.num_cols):
        choice = column / generator.num_cols + 0.001
        index = next((i for i, boundary in enumerate(cumulative) if choice < boundary), len(names) - 1)
        column_names.append(names[index])

    result: dict[str, torch.Tensor] = {}
    available = set(names)
    for group, requested in terrain_type_groups.items():
        requested = tuple(requested)
        unknown = set(requested) - available
        if unknown:
            raise ValueError(f"unknown terrain types for {group!r}: {sorted(unknown)}")

        columns = [i for i, name in enumerate(column_names) if name in requested]
        if columns:
            column_ids = torch.tensor(columns, device=terrain.terrain_types.device)
            levels = terrain.terrain_levels[torch.isin(terrain.terrain_types, column_ids)].float()
        else:
            levels = torch.empty(0, device=terrain.terrain_levels.device)

        if levels.numel() == 0:
            nan = torch.tensor(float("nan"), device=terrain.terrain_levels.device)
            for statistic in statistic_names:
                result[f"{group}/{statistic}"] = nan
            if include_level_fractions:
                for level in range(terrain.max_terrain_level):
                    result[f"{group}/level_{level:02d}_fraction"] = nan
            continue

        if "mean" in statistic_names:
            result[f"{group}/mean"] = levels.mean()
        if "median" in statistic_names:
            result[f"{group}/median"] = levels.median()
        if "p90" in statistic_names:
            result[f"{group}/p90"] = torch.quantile(levels, 0.90)
        if "max" in statistic_names:
            result[f"{group}/max"] = levels.max()
        if include_level_fractions:
            for level in range(terrain.max_terrain_level):
                result[f"{group}/level_{level:02d}_fraction"] = (levels == level).float().mean()

    return result


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
