# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain mix used by the D1 platform task."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
import numpy as np
import trimesh
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

D1_FLAT_X_TERRAIN_START = 0.05
D1_FLAT_X_TERRAIN_END = 0.10
D1_PLATFORM_TERRAIN_START = 0.40
D1_PLATFORM_DESCENT_TERRAIN_START = 0.70
D1_HIGH_PLATFORM_HEIGHT_RANGE = (0.05, 1.00)
D1_HIGH_PLATFORM_ASCENT_HEIGHT_RANGE = (0.30, 1.00)


def _smooth_face_noise(shape: tuple[int, int], amplitude: float, smoothing_passes: int) -> np.ndarray:
    """Sample low-frequency face-normal displacement with fixed, gap-free edges."""
    if amplitude <= 0.0 or min(shape) <= 2:
        return np.zeros(shape)

    noise = np.random.uniform(-amplitude, amplitude, size=shape)
    for _ in range(smoothing_passes):
        padded = np.pad(noise, 1, mode="edge")
        noise = (
            4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 8.0
    noise[[0, -1], :] = 0.0
    noise[:, [0, -1]] = 0.0
    interior = noise[1:-1, 1:-1]
    interior -= np.mean(interior)
    positive = interior > 0.0
    negative = interior < 0.0
    if np.any(positive):
        interior[positive] *= amplitude / np.max(interior[positive])
    if np.any(negative):
        interior[negative] *= amplitude / -np.min(interior[negative])
    return noise


def _rough_face_geometry(
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    normal_axis: int,
    normal_sign: int,
    roughness: float,
    sample_spacing: float,
    smoothing_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one gridded box face whose displacement varies in both face directions."""
    tangent_axes = ((1, 2), (2, 0), (0, 1))[normal_axis]
    u_axis, v_axis = tangent_axes
    num_u = max(2, int(np.ceil(dimensions[u_axis] / sample_spacing)) + 1)
    num_v = max(2, int(np.ceil(dimensions[v_axis] / sample_spacing)) + 1)
    u = np.linspace(-0.5 * dimensions[u_axis], 0.5 * dimensions[u_axis], num_u)
    v = np.linspace(-0.5 * dimensions[v_axis], 0.5 * dimensions[v_axis], num_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    displacement = _smooth_face_noise((num_u, num_v), roughness, smoothing_passes)

    vertices = np.empty((num_u * num_v, 3), dtype=np.float64)
    vertices[:, normal_axis] = (
        center[normal_axis]
        + normal_sign * 0.5 * dimensions[normal_axis]
        + normal_sign * displacement.reshape(-1)
    )
    vertices[:, u_axis] = center[u_axis] + uu.reshape(-1)
    vertices[:, v_axis] = center[v_axis] + vv.reshape(-1)

    faces = []
    for i in range(num_u - 1):
        for j in range(num_v - 1):
            index_00 = i * num_v + j
            index_01 = index_00 + 1
            index_10 = (i + 1) * num_v + j
            index_11 = index_10 + 1
            if normal_sign > 0:
                faces.extend(((index_00, index_10, index_11), (index_00, index_11, index_01)))
            else:
                faces.extend(((index_00, index_11, index_10), (index_00, index_01, index_11)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _rough_box_mesh(
    dimensions: tuple[float, float, float],
    center: tuple[float, float, float],
    side_roughness: float,
    top_roughness: float,
    sample_spacing: float,
    smoothing_passes: int,
) -> trimesh.Trimesh:
    """Create a closed box mesh with true 3-D roughness on all sides and its top."""
    vertices_parts = []
    faces_parts = []
    vertex_offset = 0
    for normal_axis, normal_sign, roughness in (
        (0, -1, side_roughness),
        (0, 1, side_roughness),
        (1, -1, side_roughness),
        (1, 1, side_roughness),
        (2, -1, 0.0),
        (2, 1, top_roughness),
    ):
        vertices, faces = _rough_face_geometry(
            center,
            dimensions,
            normal_axis,
            normal_sign,
            roughness,
            sample_spacing,
            smoothing_passes,
        )
        vertices_parts.append(vertices)
        faces_parts.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices_parts),
        faces=np.concatenate(faces_parts),
        process=False,
    )


def rocky_mesh_pyramid_stairs_terrain(
    difficulty: float,
    cfg: MeshRockyPyramidStairsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate multi-level platforms with genuinely rough top and side meshes."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    side_roughness = cfg.side_roughness_range[0] + difficulty * (
        cfg.side_roughness_range[1] - cfg.side_roughness_range[0]
    )
    top_roughness = cfg.top_roughness_range[0] + difficulty * (
        cfg.top_roughness_range[1] - cfg.top_roughness_range[0]
    )
    if cfg.holes:
        raise ValueError("Rocky mesh pyramid stairs do not support holes.")

    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    inner_sizes = []
    current_size = terrain_size
    while current_size[0] > cfg.platform_width and current_size[1] > cfg.platform_width:
        current_size = (
            current_size[0] - 2 * cfg.step_width,
            current_size[1] - 2 * cfg.step_width,
        )
        inner_sizes.append(current_size)
    num_steps = len(inner_sizes)
    height_sign = -1.0 if cfg.inverted else 1.0
    bottom_z = min(0.0, height_sign * num_steps * step_height) - step_height
    meshes: list[trimesh.Trimesh] = []

    def append_box(dimensions: tuple[float, float, float], center: tuple[float, float, float]) -> None:
        meshes.append(
            _rough_box_mesh(
                dimensions,
                center,
                side_roughness,
                top_roughness,
                cfg.surface_sample_spacing,
                cfg.smoothing_passes,
            )
        )

    outer_size = cfg.size
    for level, inner_size in enumerate(inner_sizes):
        top_z = height_sign * level * step_height
        box_height = top_z - bottom_z
        box_z = 0.5 * (top_z + bottom_z)
        band_x = 0.5 * (outer_size[0] - inner_size[0])
        band_y = 0.5 * (outer_size[1] - inner_size[1])
        append_box(
            (outer_size[0], band_y, box_height),
            (terrain_center[0], terrain_center[1] + 0.5 * (inner_size[1] + band_y), box_z),
        )
        append_box(
            (outer_size[0], band_y, box_height),
            (terrain_center[0], terrain_center[1] - 0.5 * (inner_size[1] + band_y), box_z),
        )
        append_box(
            (band_x, inner_size[1], box_height),
            (terrain_center[0] + 0.5 * (inner_size[0] + band_x), terrain_center[1], box_z),
        )
        append_box(
            (band_x, inner_size[1], box_height),
            (terrain_center[0] - 0.5 * (inner_size[0] + band_x), terrain_center[1], box_z),
        )
        outer_size = inner_size

    center_top_z = height_sign * num_steps * step_height
    center_height = center_top_z - bottom_z
    append_box(
        (outer_size[0], outer_size[1], center_height),
        (terrain_center[0], terrain_center[1], 0.5 * (center_top_z + bottom_z)),
    )
    return meshes, np.array((terrain_center[0], terrain_center[1], center_top_z))


@configclass
class MeshRockyPyramidStairsTerrainCfg(terrain_gen.MeshPyramidStairsTerrainCfg):
    """Multi-level mesh platforms with roughness normal to every exposed face."""

    function = rocky_mesh_pyramid_stairs_terrain
    inverted: bool = False
    side_roughness_range: tuple[float, float] = (0.20, 0.20)
    top_roughness_range: tuple[float, float] = (0.0, 0.10)
    surface_sample_spacing: float = 0.25
    smoothing_passes: int = 1


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
        "highplatform_up": MeshRockyPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=D1_HIGH_PLATFORM_ASCENT_HEIGHT_RANGE,
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
            inverted=True,
        ),
        "highplatform_down": MeshRockyPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=D1_HIGH_PLATFORM_HEIGHT_RANGE,
            step_width=1.0,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)
