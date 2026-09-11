# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for commanded-height locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv


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
