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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, RLTaskEnv


def terrain_levels_vel(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_level_statistics(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    terrain_type_groups: dict[str, Sequence[str]],
    statistic_names: Sequence[str] = ("mean", "median", "p90", "max"),
) -> dict[str, torch.Tensor]:
    """Report curriculum levels for selected generated-terrain groups."""
    del env_ids
    terrain: TerrainImporter = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        raise ValueError("Terrain-level statistics require a terrain generator.")

    terrain_names = list(terrain_generator.sub_terrains)
    proportions = [float(cfg.proportion) for cfg in terrain_generator.sub_terrains.values()]
    proportion_sum = sum(proportions)
    if proportion_sum <= 0.0:
        raise ValueError("Terrain proportions must sum to a positive value.")

    requested_statistics = tuple(statistic_names)
    supported_statistics = {"mean", "median", "p90", "max"}
    unknown_statistics = set(requested_statistics) - supported_statistics
    if unknown_statistics:
        raise ValueError(
            f"Unknown terrain-level statistics: {sorted(unknown_statistics)}. "
            f"Supported values: {sorted(supported_statistics)}."
        )

    cumulative_proportions: list[float] = []
    cumulative = 0.0
    for proportion in proportions:
        cumulative += proportion / proportion_sum
        cumulative_proportions.append(cumulative)

    # TerrainGenerator assigns curriculum terrain types by normalized column.
    column_terrain_names: list[str] = []
    for column in range(terrain_generator.num_cols):
        choice = column / terrain_generator.num_cols + 0.001
        terrain_index = next(
            (
                index
                for index, cumulative_proportion in enumerate(cumulative_proportions)
                if choice < cumulative_proportion
            ),
            len(terrain_names) - 1,
        )
        column_terrain_names.append(terrain_names[terrain_index])

    statistics: dict[str, torch.Tensor] = {}
    available_names = set(terrain_names)
    for group_name, requested_names in terrain_type_groups.items():
        requested_names = tuple(requested_names)
        unknown_names = set(requested_names) - available_names
        if unknown_names:
            raise ValueError(
                f"Unknown terrain types for group '{group_name}': {sorted(unknown_names)}. "
                f"Available types: {terrain_names}."
            )

        columns = [
            column for column, terrain_name in enumerate(column_terrain_names) if terrain_name in requested_names
        ]
        if columns:
            column_ids = torch.tensor(columns, device=terrain.terrain_types.device)
            levels = terrain.terrain_levels[torch.isin(terrain.terrain_types, column_ids)].float()
        else:
            levels = torch.empty(0, device=terrain.terrain_levels.device)

        if levels.numel() == 0:
            missing = torch.tensor(float("nan"), device=terrain.terrain_levels.device)
            for statistic_name in requested_statistics:
                statistics[f"{group_name}/{statistic_name}"] = missing
            continue

        if "mean" in requested_statistics:
            statistics[f"{group_name}/mean"] = levels.mean()
        if "median" in requested_statistics:
            statistics[f"{group_name}/median"] = levels.median()
        if "p90" in requested_statistics:
            statistics[f"{group_name}/p90"] = torch.quantile(levels, 0.90)
        if "max" in requested_statistics:
            statistics[f"{group_name}/max"] = levels.max()

    return statistics


def base_height_command_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    command_name: str = "base_height",
    start_floor: float = 0.35,
    end_floor: float = 0.15,
    full_steps: int = 144000,
) -> torch.Tensor:
    """Linearly lower the sampled base-height floor during early training."""
    del env_ids
    if full_steps <= 0:
        raise ValueError("full_steps must be positive.")

    progress = min(1.0, env.common_step_counter / float(full_steps))
    floor = start_floor + (end_floor - start_floor) * progress
    term = env.command_manager.get_term(command_name)
    ceiling = float(term.cfg.ranges.height[1])
    term.cfg.ranges.height = (float(floor), ceiling)
    return torch.tensor(floor, device=env.device)


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
