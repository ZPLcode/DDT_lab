# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for platform terrain-column selection."""

from __future__ import annotations

import math


def platform_terrain_type_start_index(fraction: float, num_cols: int) -> int:
    """Map a terrain proportion to its first curriculum column."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1).")
    if num_cols <= 0:
        raise ValueError("num_cols must be positive.")
    return math.ceil((fraction - 0.001) * num_cols)
