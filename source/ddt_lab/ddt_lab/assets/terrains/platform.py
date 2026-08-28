# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain mix used by the D1 platform task."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import numpy as np
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

D1_FLAT_X_TERRAIN_START = 0.05
D1_FLAT_X_TERRAIN_END = 0.10
D1_PLATFORM_TERRAIN_START = 0.40
D1_PLATFORM_DESCENT_TERRAIN_START = 0.70
D1_HIGH_PLATFORM_HEIGHT_RANGE = (0.20, 1.00)


@height_field_to_mesh
def rough_pyramid_sloped_terrain(
    difficulty: float,
    cfg: HfRoughPyramidSlopedTerrainCfg,
) -> np.ndarray:
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    noise = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return slope + noise


@configclass
class HfRoughPyramidSlopedTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    """Pyramid slope with superimposed height noise."""

    function = rough_pyramid_sloped_terrain
    slope_range: tuple[float, float] = (0.10, 0.50)
    platform_width: float = 3.0
    inverted: bool = False
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float | None = 0.2


D1_PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=40,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "flat_x": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "smooth_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.02142857142857143,
            slope_range=(0.10, 0.50),
            platform_width=3.0,
            border_width=0.25,
        ),
        "smooth_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.02142857142857143,
            slope_range=(0.10, 0.50),
            platform_width=3.0,
            border_width=0.25,
        ),
        "rough_slope": HfRoughPyramidSlopedTerrainCfg(
            proportion=0.04285714285714286,
            slope_range=(0.10, 0.50),
            platform_width=3.0,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.08571428571428572,
            step_height_range=(0.05, 0.25),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.08571428571428572,
            step_height_range=(0.05, 0.25),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.04285714285714286,
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.25,
        ),
        "highplatform_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=D1_HIGH_PLATFORM_HEIGHT_RANGE,
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
        ),
        "highplatform_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=D1_HIGH_PLATFORM_HEIGHT_RANGE,
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)
