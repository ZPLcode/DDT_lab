# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-level tests for the rocky high-platform mesh."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TERRAIN_SOURCE = Path(__file__).resolve().parents[1] / "ddt_lab" / "assets" / "terrains" / "platform.py"


def _load_geometry_functions():
    tree = ast.parse(TERRAIN_SOURCE.read_text(encoding="utf-8"), filename=str(TERRAIN_SOURCE))
    function_names = {"_smooth_face_noise", "_rough_face_geometry"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    assert {node.name for node in selected} == function_names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(TERRAIN_SOURCE), "exec"), namespace)
    return namespace


def _face_vertices(normal_axis: int, roughness: float) -> np.ndarray:
    functions = _load_geometry_functions()
    np.random.seed(11 + normal_axis)
    vertices, faces = functions["_rough_face_geometry"](
        center=(0.0, 0.0, 0.0),
        dimensions=(2.0, 2.0, 1.0),
        normal_axis=normal_axis,
        normal_sign=1,
        roughness=roughness,
        sample_spacing=0.2,
        smoothing_passes=1,
    )
    assert faces.shape[1] == 3
    return vertices


def test_easy_platform_faces_remain_planar():
    for normal_axis, nominal_coordinate in ((0, 1.0), (1, 1.0), (2, 0.5)):
        vertices = _face_vertices(normal_axis, roughness=0.0)
        assert np.allclose(vertices[:, normal_axis], nominal_coordinate)


def test_rocky_mesh_displaces_both_pairs_of_vertical_walls():
    x_face = _face_vertices(normal_axis=0, roughness=0.20)
    y_face = _face_vertices(normal_axis=1, roughness=0.20)

    assert np.isclose(np.ptp(x_face[:, 0]), 0.40)
    assert np.isclose(np.ptp(y_face[:, 1]), 0.40)


def test_rocky_mesh_also_displaces_the_top_surface():
    top_face = _face_vertices(normal_axis=2, roughness=0.10)

    assert np.isclose(np.ptp(top_face[:, 2]), 0.20)


def test_ascent_curriculum_generates_at_least_30cm_risers_at_all_levels():
    tree = ast.parse(TERRAIN_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "rocky_mesh_pyramid_stairs_terrain"
    )
    height_range = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "D1_HIGH_PLATFORM_ASCENT_HEIGHT_RANGE" for t in node.targets)
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    namespace = {"np": np, "_rough_box_mesh": lambda *args: None}
    exec(compile(ast.fix_missing_locations(module), str(TERRAIN_SOURCE), "exec"), namespace)
    cfg = SimpleNamespace(
        step_height_range=height_range,
        side_roughness_range=(0.2, 0.2),
        top_roughness_range=(0.0, 0.1),
        holes=False,
        size=(8.0, 8.0),
        border_width=0.25,
        platform_width=3.0,
        step_width=1.0,
        inverted=True,
        surface_sample_spacing=0.25,
        smoothing_passes=1,
    )
    heights = []
    for level in range(40):
        for fraction in (0.0, 0.5, 1.0 - 1e-7):
            _, origin = namespace["rocky_mesh_pyramid_stairs_terrain"]((level + fraction) / 40, cfg)
            heights.append(-origin[2] / 3)
    assert np.isclose(min(heights), 0.30)
    assert max(heights) <= 1.0
    assert np.isclose(heights[2], 0.3175)
