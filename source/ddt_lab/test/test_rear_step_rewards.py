# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rear-wheel trajectories tested without starting Isaac Sim."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MDP_ROOT = Path(__file__).resolve().parents[1] / "ddt_lab" / "tasks" / "manager_based" / "locomotion" / "mdp"


class _ManagerTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def _load_reward():
    nodes = []
    for filename, names in (
        ("platform_rewards.py", {"_stable_horizontal_support"}),
        ("rear_step_rewards.py", {"_nearest_tread", "RearWheelStepReward"}),
    ):
        tree = ast.parse((MDP_ROOT / filename).read_text())
        nodes.extend(
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        )
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTerm,
        "_terrain_type_mask": lambda env, start: env.terrain_fraction >= start,
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(MDP_ROOT / "rear_step_rewards.py"), "exec"), namespace)
    return namespace["RearWheelStepReward"]


class _Scene:
    def __init__(self, asset, contacts, scanner):
        self.asset = asset
        self.sensors = {"contacts": contacts, "height": scanner}

    def __getitem__(self, name):
        return self.asset if name == "robot" else self.sensors[name]


class _Fixture:
    def __init__(self, num_envs=1, step_dt=0.02):
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, 21), torch.linspace(-0.5, 0.5, 11), indexing="ij")
        rays = torch.stack((grid_x, grid_y, (grid_x >= -1.0e-6).float() * 0.3), dim=-1).reshape(-1, 3)
        self.rays = rays[None].repeat(num_envs, 1, 1)
        self.positions = torch.tensor(
            [[0.35, 0.2, 0.387], [0.35, -0.2, 0.387], [-0.35, 0.2, 0.087], [-0.35, -0.2, 0.087]]
        )[None].repeat(num_envs, 1, 1)
        self.forces = torch.zeros(num_envs, 10, 4, 3)
        self.forces[..., 2] = 50.0
        self.data = SimpleNamespace(
            body_pos_w=self.positions,
            body_lin_vel_w=torch.zeros_like(self.positions),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
        )
        contacts = SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=self.forces, net_forces_w=self.forces[:, 0])
        )
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=self.rays))
        self.commands = torch.tensor([[0.3, 0.0, 0.0]]).repeat(num_envs, 1)
        self.env = SimpleNamespace(
            num_envs=num_envs,
            device="cpu",
            step_dt=step_dt,
            max_episode_length_s=20.0,
            terrain_fraction=torch.full((num_envs,), 0.5),
            command_manager=SimpleNamespace(get_command=lambda _: self.commands),
            reward_manager=SimpleNamespace(_episode_sums={}),
            scene=_Scene(SimpleNamespace(data=self.data), contacts, scanner),
        )
        self.params = {
            "terrain_sensor_cfg": SimpleNamespace(name="height"),
            "wheel_sensor_cfg": SimpleNamespace(name="contacts", body_ids=[0, 1, 2, 3]),
            "wheel_asset_cfg": SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3]),
            "command_name": "base_velocity",
            "terrain_type_start": 0.4,
            "terrain_type_end": 0.7,
        }
        self.reward = _load_reward()(SimpleNamespace(weight=1.0), self.env)

    def evaluate(self):
        return self.reward(self.env, **self.params) * self.env.step_dt

    def wheel(self, index, *, x=None, height=None, force=(0.0, 0.0, 0.0)):
        if x is not None:
            self.positions[:, index, 0] = x
        if height is not None:
            self.positions[:, index, 2] = height + 0.087
        self.forces[:, :, index] = torch.tensor(force)


def test_rear_wheels_can_clear_and_land_one_at_a_time_without_repeat_bonuses():
    f = _Fixture()
    assert f.evaluate().item() == 0.0
    for rear in (2, 3):
        f.wheel(rear, height=0.18)
        assert f.evaluate().item() == pytest.approx(0.125)
        assert f.evaluate().item() == 0.0
        f.wheel(rear, height=0.0)
        assert f.evaluate().item() == 0.0
        f.wheel(rear, height=0.18)
        assert f.evaluate().item() == 0.0
        f.wheel(rear, height=0.36)
        assert f.evaluate().item() == pytest.approx(0.125)
        f.wheel(rear, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
        assert f.evaluate().item() == pytest.approx(1.0)
        assert f.evaluate().item() == 0.0
        f.wheel(rear, height=0.5)
        assert f.evaluate().item() == 0.0
        f.wheel(rear, height=0.3, force=(0.0, 0.0, 50.0))
        assert f.evaluate().item() == 0.0


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.04])
def test_wall_recovery_rewards_new_free_lift_but_never_a_dirty_landing(step_dt):
    f = _Fixture(step_dt=step_dt)
    f.evaluate()
    f.wheel(2, x=-0.05, height=0.05, force=(-100.0, 0.0, 20.0))
    assert f.evaluate().item() < 0.0
    assert f.reward.wall_touched[0, 0]
    assert not f.reward.wall_touched[0, 1]
    f.wheel(2, x=-0.2, height=0.18)
    recovery = f.evaluate().item()
    assert recovery == pytest.approx(0.25 * (0.18 - 0.05) / 0.36)
    assert f.evaluate().item() == 0.0
    f.wheel(2, height=0.05)
    assert f.evaluate().item() == 0.0
    f.wheel(2, height=0.18)
    assert f.evaluate().item() == 0.0
    f.wheel(2, x=-0.2, height=0.36)
    recovery += f.evaluate().item()
    assert recovery == pytest.approx(0.25 * (0.36 - 0.05) / 0.36)
    sums = f.env.reward_manager._episode_sums
    assert sums["ascent_rear_step/recovery_progress"].item() == pytest.approx(recovery)
    assert sums["ascent_rear_step/clearance_progress"].item() == pytest.approx(recovery)
    assert not f.reward.cleared[0, 0]
    f.wheel(2, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0
    assert sums["ascent_rear_step/clean_landing"].item() == 0.0
    # The other rear wheel still qualifies for a clean, sequential landing.
    f.wheel(3, height=0.36)
    assert f.evaluate().item() == pytest.approx(0.25)
    f.wheel(3, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == pytest.approx(1.0)
    assert sums["ascent_rear_step/recovery_progress"].item() == pytest.approx(recovery)


def test_current_wall_contact_and_passive_wall_climbing_earn_no_lift_bonus():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.05, height=0.05, force=(-100.0, 0.0, 20.0))
    assert f.evaluate().item() < 0.0
    # A currently unloaded wheel with recent wall force is still gated out.
    f.positions[:, 2, 2] = 0.18 + 0.087
    f.forces[:, 0, 2] = 0.0
    assert f.evaluate().item() < 0.0
    sums = f.env.reward_manager._episode_sums
    assert sums["ascent_rear_step/clearance_progress"].item() == 0.0
    # Freeing the wheel cannot retroactively pay for height gained on the wall.
    f.wheel(2, height=0.18)
    assert f.evaluate().item() == 0.0
    f.wheel(2, height=0.27)
    assert f.evaluate().item() == pytest.approx(0.0625)


def test_recovery_still_requires_front_support_and_an_airborne_rear_wheel():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.05, height=0.05, force=(-100.0, 0.0, 20.0))
    f.evaluate()
    f.wheel(2, x=-0.35, height=0.10, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0
    f.forces[:, :, :2] = 0.0
    f.wheel(2, height=0.18)
    assert f.evaluate().item() == 0.0
    f.forces[:, :, :2, 2] = 50.0
    f.wheel(2, height=0.36)
    assert f.evaluate().item() == pytest.approx(0.125)


def test_ground_traction_and_front_wall_contacts_do_not_count_as_rear_wall_hits():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.05, height=0.0, force=(40.0, 0.0, 100.0))
    f.wheel(0, force=(-300.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0
    assert not f.reward.wall_touched.any()


def test_landing_requires_clearance_before_crossing_and_sustained_support():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=0.2, height=0.36)
    f.evaluate()
    f.wheel(2, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0

    f = _Fixture()
    f.evaluate()
    f.wheel(2, height=0.36)
    f.evaluate()
    f.wheel(2, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    f.forces[:, 1:, 2] = 0.0
    assert f.evaluate().item() == 0.0
    f.forces[:, :8, 2, 2] = 50.0
    assert f.evaluate().item() == pytest.approx(1.0)


@pytest.mark.parametrize("case", ["flat", "descent", "zero_command", "backward", "invalid_scan", "no_front_support"])
def test_shaping_is_inactive_outside_supported_forward_ascent(case):
    f = _Fixture()
    if case == "flat":
        f.rays[..., 2] = 0.0
    elif case == "descent":
        f.env.terrain_fraction[:] = 0.8
    elif case == "zero_command":
        f.commands[:] = 0.0
    elif case == "backward":
        f.commands[:, 0] = -0.3
    elif case == "invalid_scan":
        f.rays[:] = float("inf")
    else:
        f.forces[:, :, :2] = 0.0
    assert f.evaluate().item() == 0.0
    f.wheel(2, height=0.36)
    result = f.evaluate()
    assert torch.isfinite(result).all()
    assert result.item() == 0.0


def test_partial_reset_clears_only_finished_environment():
    f = _Fixture(num_envs=2)
    f.evaluate()
    f.wheel(2, height=0.36)
    f.evaluate()
    f.reward.wall_touched[:, 1] = True
    f.reward.completed_z[:] = 0.3
    f.reward.reset(torch.tensor([0]))
    assert not f.reward.active[0].any()
    assert not f.reward.wall_touched[0].any()
    assert not f.reward.cleared[0].any()
    assert (f.reward.completed_z[0] < 0.0).all()
    assert f.reward.active[1].all()
    assert f.reward.wall_touched[1, 1]
    assert f.reward.cleared[1, 0]
    assert (f.reward.completed_z[1] == 0.3).all()


def test_unsafe_approach_penalizes_before_contact_and_increases_towards_edge():
    f = _Fixture()
    f.wheel(2, x=-0.5, height=0.0, force=(0.0, 0.0, 50.0))
    f.data.body_lin_vel_w[:, 2, 0] = 0.4
    assert f.evaluate().item() == 0.0
    f.wheel(2, x=-0.3, force=(0.0, 0.0, 50.0))
    farther = f.evaluate().item()
    assert farther < 0.0
    f.wheel(2, x=-0.15, force=(0.0, 0.0, 50.0))
    nearer = f.evaluate().item()
    assert nearer < farther
    assert not f.reward.wall_touched.any()
    assert f.env.reward_manager._episode_sums["ascent_rear_step/wall_contact"].item() == 0.0
    assert f.env.reward_manager._episode_sums["ascent_rear_step/unsafe_approach"].item() == pytest.approx(
        farther + nearer
    )


@pytest.mark.parametrize("velocity", [(0.0, 0.0, 0.0), (-0.4, 0.0, 0.0), (0.0, 0.4, 0.0)])
def test_unsafe_approach_does_not_penalize_stopping_retreating_or_sideways_motion(velocity):
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.15, force=(0.0, 0.0, 50.0))
    f.data.body_lin_vel_w[:, 2] = torch.tensor(velocity)
    assert f.evaluate().item() == 0.0


def test_unsafe_approach_releases_after_clearance_including_clean_touchdown():
    f = _Fixture()
    f.evaluate()
    f.data.body_lin_vel_w[:, 2, 0] = 0.4
    f.wheel(2, x=-0.2, height=0.36)
    assert f.evaluate().item() == pytest.approx(0.25)
    f.wheel(2, x=0.1, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0
    f.wheel(2, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == pytest.approx(1.0)
    assert f.env.reward_manager._episode_sums["ascent_rear_step/unsafe_approach"].item() == 0.0


def test_dropping_before_crossing_cannot_bypass_unsafe_approach_penalty():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.2, height=0.36)
    assert f.evaluate().item() == pytest.approx(0.25)
    f.wheel(2, x=-0.15, height=0.0, force=(0.0, 0.0, 50.0))
    f.data.body_lin_vel_w[:, 2, 0] = 0.4
    assert f.evaluate().item() < 0.0
    assert not f.reward.wall_touched.any()


@pytest.mark.parametrize("case", ["descent", "zero_command", "invalid_scan", "no_front_support"])
def test_unsafe_approach_requires_a_valid_supported_ascent(case):
    f = _Fixture()
    f.evaluate()
    f.wheel(2, x=-0.15, force=(0.0, 0.0, 50.0))
    f.data.body_lin_vel_w[:, 2, 0] = 0.4
    if case == "descent":
        f.env.terrain_fraction[:] = 0.8
    elif case == "zero_command":
        f.commands[:] = 0.0
    elif case == "invalid_scan":
        f.rays[:] = float("inf")
    else:
        f.forces[:, :, :2] = 0.0
    assert f.evaluate().item() == 0.0


def test_unsafe_approach_uses_latched_heading_and_dt_scaled_velocity_cost():
    f = _Fixture(num_envs=2, step_dt=0.01)
    for tensor in (f.positions, f.rays):
        original = tensor[1].clone()
        tensor[1, :, 0] = -original[:, 1]
        tensor[1, :, 1] = original[:, 0]
    f.data.root_quat_w[1] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    f.data.body_lin_vel_w[0, 2, 0] = 0.4
    f.data.body_lin_vel_w[1, 2, 1] = 0.4
    penalty = f.evaluate()
    assert (penalty < 0.0).all()
    assert torch.allclose(penalty[0], penalty[1], atol=1e-6)
    f.env.step_dt = 0.02
    assert torch.allclose(f.evaluate(), penalty * 2.0)


def test_episode_clean_pair_survives_rearming_but_a_later_hit_blocks_promotion():
    f = _Fixture(num_envs=2)
    f.evaluate()
    for rear in (2, 3):
        f.wheel(rear, height=0.36)
        f.evaluate()
        f.wheel(rear, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
        f.evaluate()
    assert f.reward.episode_clean_landed.all()
    assert not f.reward.episode_wall_touched.any()
    key = "ascent_rear_step/wall_free_clean_pair"
    assert torch.equal(f.env.reward_manager._episode_sums[key], torch.full((2,), 20.0))
    f.rays[..., 2] = torch.where(f.rays[..., 0] >= 0.69, 0.6, f.rays[..., 2])
    for front in (0, 1):
        f.wheel(front, x=0.95, height=0.6, force=(0.0, 0.0, 50.0))
    f.evaluate()
    assert f.reward.episode_clean_landed.all()
    f.wheel(2, x=0.6, height=0.3, force=(-100.0, 0.0, 20.0))
    f.evaluate()
    assert f.reward.episode_wall_touched.all()
    assert not f.env.reward_manager._episode_sums[key].any()
    f.reward.reset(torch.tensor([0]))
    assert not f.reward.episode_clean_landed[0].any()
    assert not f.reward.episode_wall_touched[0]
    assert f.reward.episode_clean_landed[1].all()
    assert f.reward.episode_wall_touched[1]


@pytest.mark.parametrize("step_dt", [0.01, 0.02, 0.04])
def test_event_bonus_is_independent_of_control_rate(step_dt):
    f = _Fixture(step_dt=step_dt)
    f.evaluate()
    f.wheel(2, height=0.36)
    lift = f.evaluate().item()
    f.wheel(2, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    landing = f.evaluate().item()
    assert lift == pytest.approx(0.25)
    assert landing == pytest.approx(1.0)


def test_shaping_is_invariant_to_world_translation_and_heading():
    f = _Fixture(num_envs=2)
    offset = torch.tensor([30.0, -20.0, 5.0])
    for tensor in (f.positions, f.rays):
        old = tensor[1].clone()
        tensor[1, :, 0] = -old[:, 1]
        tensor[1, :, 1] = old[:, 0]
        tensor[1] += offset
    f.data.root_quat_w[1] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    assert torch.equal(f.evaluate(), torch.zeros(2))
    f.positions[:, 2, 2] += 0.36
    f.forces[:, :, 2] = 0.0
    assert torch.allclose(f.evaluate(), torch.full((2,), 0.25), atol=1.0e-5)


def test_next_higher_tread_rearms_after_a_completed_step():
    f = _Fixture()
    f.evaluate()
    f.wheel(2, height=0.36)
    f.evaluate()
    f.wheel(2, x=0.2, height=0.3, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == pytest.approx(1.0)
    f.rays[..., 2] = torch.where(f.rays[..., 0] >= 0.69, 0.6, f.rays[..., 2])
    for front in (0, 1):
        f.wheel(front, x=0.95, height=0.6, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == 0.0
    assert f.reward.target_z[0, 0].item() == pytest.approx(0.6)
    f.wheel(2, height=0.66)
    assert f.evaluate().item() == pytest.approx(0.25)
    f.wheel(2, x=0.9, height=0.6, force=(0.0, 0.0, 50.0))
    assert f.evaluate().item() == pytest.approx(1.0)


def test_task_preserves_rear_step_bonus_and_penalty_outside_ordinary_clipping():
    path = MDP_ROOT.parent / "robots" / "d1" / "np3o" / "platform_env_cfg.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "D1PlatformRLEnv")

    class _Env:
        def step(self, _action):
            return {}, self.raw_reward, self.terminated, torch.zeros(3, dtype=torch.bool), {}

    namespace = {"torch": torch, "ManagerBasedRLEnv": _Env}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    env = namespace["D1PlatformRLEnv"]()
    names = [*env._UNCLIPPED_REWARD_TERMS, "is_terminated", "ordinary"]
    rates = torch.zeros(3, len(names))
    rates[:, names.index("ascent_rear_step")] = torch.tensor([-1.0, 50.0, -1.0])
    rates[:, names.index("contact_forces")] = torch.tensor([-1.0, 0.0, -1.0])
    rates[:, names.index("ordinary")] = torch.tensor([-10.0, -10.0, 5.0])
    rates[:, names.index("is_terminated")] = torch.tensor([0.0, 0.0, -0.8])
    env.step_dt = 0.02
    env.raw_reward = rates.sum(dim=1) * env.step_dt
    env.terminated = torch.tensor([False, False, True])
    env.reward_manager = SimpleNamespace(
        active_terms=names, _step_reward=rates, get_term_cfg=lambda _: SimpleNamespace(weight=-0.8)
    )
    _, reward, _, _, _ = env.step(torch.zeros(3, 16))
    assert torch.allclose(reward, torch.tensor([-0.04, 1.0, 0.044]))
