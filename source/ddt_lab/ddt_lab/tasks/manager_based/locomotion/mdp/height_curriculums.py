# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for commanded-height all-terrain tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, RLTaskEnv


def base_height_command_curriculum(
    env: RLTaskEnv,
    env_ids: Sequence[int],
    command_name: str = "base_height",
    start_floor: float = 0.35,
    end_floor: float = 0.15,
    full_steps: int = 144000,
) -> torch.Tensor:
    """Linearly lower the sampled base-height floor during early training.

    Args:
        env: Learning environment.
        env_ids: Reset environment IDs; unused because the range is global.
        command_name: Height command term name.
        start_floor: Initial minimum height.
        end_floor: Final minimum height.
        full_steps: Steps required to reach the final floor.

    Returns:
        Current minimum height as a scalar tensor.
    """
    del env_ids
    if full_steps <= 0:
        raise ValueError("full_steps must be positive.")

    progress = min(1.0, env.common_step_counter / float(full_steps))
    floor = start_floor + (end_floor - start_floor) * progress
    term = env.command_manager.get_term(command_name)
    ceiling = float(term.cfg.ranges.height[1])
    term.cfg.ranges.height = (float(floor), ceiling)
    return torch.tensor(floor, device=env.device)


def flat_command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    command_name: str = "base_velocity",
    terrain_type_end: float = 0.10,
    initial_max_speed: float = 1.0,
    final_max_speed: float = 2.0,
    increment: float = 0.25,
    success_threshold: float = 0.80,
    min_samples: int = 32,
) -> dict[str, torch.Tensor]:
    """Expand the flat-terrain X command range as tracking improves.

    Args:
        env: Learning environment.
        env_ids: Reset environment IDs used for the score update.
        reward_term_name: Tracking reward used to measure success.
        command_name: Velocity command term name.
        terrain_type_end: Exclusive end of the flat-terrain fraction.
        initial_max_speed: Initial absolute X-speed limit.
        final_max_speed: Final absolute X-speed limit.
        increment: Speed added after a successful score window.
        success_threshold: Required normalized tracking score.
        min_samples: Completed flat episodes per score window.

    Returns:
        Current speed limit, tracking score, and sample count.
    """
    if not 0.0 < terrain_type_end <= 1.0:
        raise ValueError("terrain_type_end must be in (0, 1].")
    if not 0.0 < initial_max_speed <= final_max_speed:
        raise ValueError("Expected 0 < initial_max_speed <= final_max_speed.")
    if increment <= 0.0:
        raise ValueError("increment must be positive.")
    if not 0.0 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be in (0, 1].")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term.cfg, "flat_lin_vel_x"):
        raise AttributeError(
            f"Command term '{command_name}' does not expose a flat_lin_vel_x range."
        )

    current_step = int(env.common_step_counter)
    cooldown_steps = int(env.max_episode_length)
    if not hasattr(env, "_flat_speed_curriculum_state"):
        env._flat_speed_curriculum_state = {
            "max_speed": float(initial_max_speed),
            "last_score": 0.0,
            "reward_sum": torch.zeros((), device=env.device),
            "sample_count": 0,
            "collect_after_step": current_step + cooldown_steps,
        }
    state = env._flat_speed_curriculum_state
    state["max_speed"] = min(max(float(state["max_speed"]), initial_max_speed), final_max_speed)
    command_term.cfg.flat_lin_vel_x = (-state["max_speed"], state["max_speed"])

    terrain: TerrainImporter = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if (
        terrain_generator is None
        or not terrain_generator.curriculum
        or not hasattr(terrain, "terrain_types")
    ):
        raise ValueError("Flat speed curriculum requires generated curriculum terrain.")

    reset_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    flat_col_max = math.ceil((terrain_type_end - 0.001) * terrain_generator.num_cols)
    flat_ids = reset_ids[terrain.terrain_types[reset_ids] < flat_col_max]
    is_full_reset = reset_ids.numel() == env.num_envs
    completed_flat_ids = flat_ids[env.episode_length_buf[flat_ids] > 0]

    # Ignore startup data until every environment has completed one episode.
    if is_full_reset:
        state["reward_sum"].zero_()
        state["sample_count"] = 0
        state["collect_after_step"] = current_step + cooldown_steps
    elif current_step >= state["collect_after_step"] and completed_flat_ids.numel() > 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        if reward_cfg.weight <= 0.0:
            raise ValueError(f"Reward term '{reward_term_name}' must have a positive weight.")
        state["reward_sum"].add_(episode_sums[completed_flat_ids].detach().sum())
        state["sample_count"] += completed_flat_ids.numel()

        if state["sample_count"] >= min_samples:
            score = state["reward_sum"] / (
                state["sample_count"] * env.max_episode_length_s * reward_cfg.weight
            )
            score_value = float(score.item())
            state["last_score"] = score_value
            state["reward_sum"].zero_()
            state["sample_count"] = 0

            if score_value >= success_threshold and state["max_speed"] < final_max_speed:
                state["max_speed"] = min(state["max_speed"] + increment, final_max_speed)
                command_term.cfg.flat_lin_vel_x = (-state["max_speed"], state["max_speed"])
                state["collect_after_step"] = current_step + cooldown_steps

    return {
        "max_abs_x": torch.tensor(state["max_speed"], device=env.device),
        "tracking_score": torch.tensor(state["last_score"], device=env.device),
        "sample_count": torch.tensor(state["sample_count"], device=env.device),
    }
