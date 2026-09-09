# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor-level tests for the small platform reward helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

REWARD_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ddt_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "mdp"
    / "platform_rewards.py"
)


def _load_functions(*names: str) -> dict[str, object]:
    tree = ast.parse(REWARD_SOURCE.read_text(encoding="utf-8"), filename=str(REWARD_SOURCE))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(REWARD_SOURCE), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _contact_sensor(supported: torch.Tensor) -> SimpleNamespace:
    """Build ten force-history samples from a (batch, wheel) support mask."""
    forces = torch.zeros(supported.shape[0], 10, supported.shape[1], 3)
    forces[..., 2] = supported[:, None].float() * 50.0
    return SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))


class _Scene:
    def __init__(self, sensor: SimpleNamespace):
        self.sensors = {"contacts": sensor, "height": object()}

    def __getitem__(self, _name: str) -> object:
        return object()


def test_stable_support_requires_current_and_eight_of_ten_samples():
    stable_support = _load_functions("_stable_horizontal_support")["_stable_horizontal_support"]
    supported = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, True],
        ]
    )
    sensor = _contact_sensor(supported)
    result = stable_support(sensor, [0, 1, 2, 3], 10.0, 2.0, 10, 0.8)

    assert torch.equal(result, supported)

    sensor.data.net_forces_w_history[0, :3, 0, 2] = 0.0
    result = stable_support(sensor, [0, 1, 2, 3], 10.0, 2.0, 10, 0.8)
    assert not result[0, 0]


def test_ascent_rear_lateral_force_penalty_uses_peak_excess_and_terrain_gate():
    penalty = _load_functions("platform_ascent_rear_lateral_force_penalty")[
        "platform_ascent_rear_lateral_force_penalty"
    ]
    penalty.__globals__["_terrain_type_mask"] = lambda env, start: (
        env.ascent_or_later if start == 0.40 else env.descent_or_later
    )

    forces = torch.zeros(3, 10, 2, 3)
    forces[0, 4, :, 0] = torch.tensor([150.0, 250.0])
    forces[1, 7, :, 1] = torch.tensor([300.0, 400.0])
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    env = SimpleNamespace(
        scene=_Scene(sensor),
        ascent_or_later=torch.tensor([True, True, False]),
        descent_or_later=torch.tensor([False, True, False]),
    )
    sensor_cfg = SimpleNamespace(name="contacts", body_ids=[0, 1])

    result = penalty(env, sensor_cfg, 200.0, 0.40, 0.70)

    assert torch.equal(result, torch.tensor([50.0, 0.0, 0.0]))


def test_descent_front_impact_penalty_uses_peak_excess_and_forward_gate():
    penalty = _load_functions("platform_descent_front_impact_penalty")[
        "platform_descent_front_impact_penalty"
    ]
    penalty.__globals__["_terrain_type_mask"] = lambda env, _start: env.descent

    forces = torch.zeros(4, 10, 2, 3)
    forces[0, 4, :, 2] = torch.tensor([250.0, 500.0])
    forces[1, 7, :, 2] = torch.tensor([600.0, 700.0])
    forces[2, 3, :, 2] = torch.tensor([500.0, 500.0])
    forces[3, 5, :, 2] = torch.tensor([500.0, 500.0])
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    commands = SimpleNamespace(
        get_command=lambda _name: torch.tensor(
            [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.05, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    )
    env = SimpleNamespace(
        scene=_Scene(sensor),
        command_manager=commands,
        descent=torch.tensor([True, True, True, False]),
    )
    sensor_cfg = SimpleNamespace(name="contacts", body_ids=[0, 1])

    result = penalty(env, sensor_cfg, "base_velocity", 300.0, 0.70, 0.10)

    assert torch.equal(result, torch.tensor([200.0, 0.0, 0.0, 0.0]))


def test_descent_penalty_accepts_either_axle_but_not_diagonal_support():
    functions = _load_functions(
        "_stable_horizontal_support",
        "platform_descent_axle_support_penalty",
    )
    penalty = functions["platform_descent_axle_support_penalty"]
    penalty.__globals__["_stable_horizontal_support"] = functions["_stable_horizontal_support"]
    penalty.__globals__["_height_scan_transition"] = lambda *_args: (
        torch.tensor([True, True, True, False]),
        torch.zeros(4),
        torch.ones(4),
    )
    penalty.__globals__["_terrain_type_mask"] = lambda *_args: torch.ones(4, dtype=torch.bool)

    supported = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
            [True, False, False, True],
            [False, False, False, False],
        ]
    )
    sensor = _contact_sensor(supported)
    scene = _Scene(sensor)
    commands = SimpleNamespace(
        get_command=lambda _name: torch.tensor(
            [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]
        )
    )
    env = SimpleNamespace(scene=scene, command_manager=commands)
    sensor_cfg = SimpleNamespace(name="contacts", body_ids=[0, 1, 2, 3])
    result = penalty(
        env,
        SimpleNamespace(name="height"),
        sensor_cfg,
        SimpleNamespace(name="robot"),
        "base_velocity",
        0.70,
    )

    assert torch.equal(result, torch.tensor([0.0, 0.0, 1.0, 0.0]))
