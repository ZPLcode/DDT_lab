# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check ascent promotion decisions separately from simulator startup."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MDP_ROOT = Path(__file__).resolve().parents[1] / "ddt_lab/tasks/manager_based/locomotion/mdp"


def _load_curriculum():
    source = MDP_ROOT / "curriculums.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "terrain_levels_vel")
    namespace = {
        "torch": torch,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "platform_terrain_type_start_index": lambda start, columns: int(start * columns),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["terrain_levels_vel"]


class _Scene:
    def __init__(self, positions, terrain):
        self.env_origins = torch.zeros_like(positions)
        self.terrain = terrain
        self.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=positions))

    def __getitem__(self, name):
        return self.robot


def _fixture():
    # Non-ascent, descent, clean ascent, one-wheel-only, dirty ascent, toppled,
    # no clean landings, short-distance clean ascent, short-distance dirty ascent.
    positions = torch.tensor([
        [4.5, 0, 0],
        [3.5, 0, 0],
        [4.5, 0, 0],
        [4.5, 0, 0],
        [4.5, 0, 0],
        [4.5, 0, 0],
        [4.5, 0, 0],
        [1.0, 0, 0],
        [1.0, 0, 0],
    ])
    count = len(positions)
    updates = []
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0), num_cols=20)),
        terrain_types=torch.tensor([0, 14, 8, 9, 10, 11, 12, 13, 8]),
        terrain_levels=torch.full((count,), 5),
        update_env_origins=lambda ids, up, down: updates.append((torch.as_tensor(ids), up.clone(), down.clone())),
    )
    clean = torch.tensor([
        [False, False],
        [False, False],
        [True, True],
        [True, False],
        [True, True],
        [True, True],
        [False, False],
        [True, True],
        [False, False],
    ])
    term = SimpleNamespace(
        episode_clean_landed=clean,
        episode_wall_touched=torch.tensor([True, True, False, False, True, False, False, False, True]),
    )
    env = SimpleNamespace(
        scene=_Scene(positions, terrain),
        max_episode_length_s=20.0,
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.6, 0.0, 0.0]]).repeat(count, 1)),
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=term)),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False] * 5 + [True] + [False] * 3)),
    )
    params = dict(
        move_up_terrain_type_start=0.7,
        move_up_distance_override=3.0,
        clean_ascent_reward_term="ascent_rear_step",
        clean_ascent_terrain_range=(0.4, 0.7),
    )
    return env, updates, params


def test_only_clean_wall_free_two_wheel_ascent_is_promoted_and_dirty_traversal_holds_level():
    env, updates, params = _fixture()
    _load_curriculum()(env, list(range(9)), **params)
    _, up, down = updates[-1]
    assert up.tolist() == [True, True, True, False, False, False, False, False, False]
    # Blocked promotions do not become demotions just because command speed is high.
    assert down.tolist() == [False] * 7 + [True, True]


def test_curriculum_uses_only_reset_ids_without_clearing_episode_reward_state():
    env, updates, params = _fixture()
    before = env.reward_manager.get_term_cfg("ascent_rear_step").func.episode_clean_landed.clone()
    _load_curriculum()(env, torch.tensor([4, 2, 3]), **params)
    ids, up, down = updates[-1]
    assert ids.tolist() == [4, 2, 3]
    assert up.tolist() == [False, True, False]
    assert not down.any()
    assert torch.equal(env.reward_manager.get_term_cfg("ascent_rear_step").func.episode_clean_landed, before)


def test_curriculum_without_clean_gate_retains_distance_behavior():
    env, updates, params = _fixture()
    params.pop("clean_ascent_reward_term")
    params.pop("clean_ascent_terrain_range")
    _load_curriculum()(env, list(range(9)), **params)
    assert updates[-1][1].tolist() == [True] * 7 + [False, False]


@pytest.mark.parametrize(
    "changes",
    [
        {"clean_ascent_reward_term": None},
        {"clean_ascent_terrain_range": None},
        {"clean_ascent_terrain_range": (0.7, 0.4)},
        {"clean_ascent_terrain_range": (0.4, 1.0)},
    ],
)
def test_curriculum_rejects_incomplete_or_invalid_clean_gate(changes):
    env, _, params = _fixture()
    params.update(changes)
    with pytest.raises(ValueError):
        _load_curriculum()(env, [2], **params)
