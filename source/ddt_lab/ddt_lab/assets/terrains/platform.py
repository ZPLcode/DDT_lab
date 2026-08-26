# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain generator configs for the D1 robot.

Add new ``TerrainGeneratorCfg`` presets here and import them from a robot/
algorithm ``*_env_cfg.py`` to swap ``SceneCfg.terrain.terrain_generator``.
See ``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`` for the stock preset
that ``base_env_cfg.py`` uses by default.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import numpy as np
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

# TODO: tune `proportion` (must sum to 1.0 across all sub-terrains), and the
# per-terrain parameters below to taste.
PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.1,
        #     gap_width_range=(0.05, 0.23),
        #     platform_width=3.0,
        # ),
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.05, 0.7),
            platform_width=2.0,
            double_box=True,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2,
            rail_thickness_range=(0.05, 0.10),
            rail_height_range=(0.7, 0.05),
            platform_width=3.0,
        ),
        "pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2,
            pit_depth_range=(0.05, 0.7),
            platform_width=3.0,
            double_pit=True,
        ),
    },
)


D1_SLOPE_TERRAIN_START = 0.10
D1_STAIRS_TERRAIN_START = 0.20
D1_PLATFORM_TERRAIN_START = 0.40
D1_PLATFORM_DESCENT_TERRAIN_START = 0.70


@height_field_to_mesh
def rough_pyramid_sloped_terrain(difficulty: float, cfg: HfRoughPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a pyramid slope with superimposed height noise."""
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    noise = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return slope + noise


@configclass
class HfRoughPyramidSlopedTerrainCfg(terrain_gen.HfTerrainBaseCfg):
    """Configuration for the combined rough-slope terrain."""

    function = rough_pyramid_sloped_terrain
    slope_range: tuple[float, float] = (0.10, 0.50)
    platform_width: float = 3.0
    inverted: bool = False
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float | None = 0.2


# With 20 columns, platform ascent occupies 8--13 and descent 14--19.
D1_HIGH_PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "smooth_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.10, 0.50),
            platform_width=3.0,
            border_width=0.25,
        ),
        "rough_slope": HfRoughPyramidSlopedTerrainCfg(
            proportion=0.05,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width=0.31,
            platform_width=3.0,
            border_width=0.25,
        ),
        "highplatform_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.20, 1.00),
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
        ),
        "highplatform_down": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.20, 1.00),
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)
