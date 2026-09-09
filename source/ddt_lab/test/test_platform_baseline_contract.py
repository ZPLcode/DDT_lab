# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contracts for the command-tracking platform baseline."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "ddt_lab"
LOCOMOTION_ROOT = PACKAGE_ROOT / "tasks" / "manager_based" / "locomotion"
D1_ROOT = LOCOMOTION_ROOT / "robots" / "d1"
PLATFORM_CFG = D1_ROOT / "np3o" / "platform_env_cfg.py"
PLATFORM_TERRAIN = PACKAGE_ROOT / "assets" / "terrains" / "platform.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _class(node: ast.Module | ast.ClassDef, name: str) -> ast.ClassDef:
    return next(child for child in node.body if isinstance(child, ast.ClassDef) and child.name == name)


def _assigned_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.Assign):
            names.update(target.id for target in child.targets if isinstance(target, ast.Name))
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
    return names


def _module_assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"missing module assignment: {name}")


def _call_keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def test_platform_terrain_and_curriculum_contract_remain_unchanged():
    terrain_tree = _tree(PLATFORM_TERRAIN)
    terrain_call = _module_assignment(terrain_tree, "D1_PLATFORM_TERRAINS_CFG")
    assert isinstance(terrain_call, ast.Call)
    terrain_keywords = _call_keywords(terrain_call)

    assert ast.literal_eval(terrain_keywords["size"]) == (8.0, 8.0)
    assert ast.literal_eval(terrain_keywords["num_rows"]) == 40
    assert ast.literal_eval(terrain_keywords["num_cols"]) == 20
    assert ast.literal_eval(terrain_keywords["curriculum"]) is True

    sub_terrains = terrain_keywords["sub_terrains"]
    assert isinstance(sub_terrains, ast.Dict)
    names = [ast.literal_eval(key) for key in sub_terrains.keys]
    assert names[-2:] == ["highplatform_up", "highplatform_down"]
    for index, value in enumerate(sub_terrains.values[-2:]):
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        assert value.func.id == "MeshRockyPyramidStairsTerrainCfg"
        keywords = _call_keywords(value)
        assert ast.literal_eval(keywords["proportion"]) == 0.30
        assert isinstance(keywords["step_height_range"], ast.Name)
        assert keywords["step_height_range"].id == "D1_HIGH_PLATFORM_HEIGHT_RANGE"
        assert ast.literal_eval(keywords.get("inverted", ast.Constant(False))) is (index == 0)

    rocky_cfg = _class(terrain_tree, "MeshRockyPyramidStairsTerrainCfg")
    rocky_defaults = {
        node.target.id: node.value
        for node in rocky_cfg.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert ast.literal_eval(rocky_defaults["side_roughness_range"]) == (0.20, 0.20)
    assert ast.literal_eval(rocky_defaults["top_roughness_range"]) == (0.0, 0.10)

    assert ast.literal_eval(_module_assignment(terrain_tree, "D1_PLATFORM_TERRAIN_START")) == 0.40
    assert ast.literal_eval(_module_assignment(terrain_tree, "D1_PLATFORM_DESCENT_TERRAIN_START")) == 0.70
    assert ast.literal_eval(_module_assignment(terrain_tree, "D1_HIGH_PLATFORM_HEIGHT_RANGE")) == (0.05, 1.00)

    cfg_source = _read(PLATFORM_CFG)
    assert "self.scene.terrain.terrain_generator = D1_PLATFORM_TERRAINS_CFG.copy()" in cfg_source
    assert "self.scene.terrain.max_init_terrain_level = 5" in cfg_source
    assert '"move_up_terrain_type_start": D1_PLATFORM_DESCENT_TERRAIN_START' in cfg_source
    assert '"up": ("highplatform_up",)' in cfg_source
    assert '"down": ("highplatform_down",)' in cfg_source


def test_platform_commands_and_actor_observation_abi_remain_unchanged():
    platform_tree = _tree(PLATFORM_CFG)
    commands = _class(platform_tree, "PlatformCommandsCfg")
    command_call = next(
        node.value
        for node in commands.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "base_velocity" for target in node.targets)
    )
    assert isinstance(command_call, ast.Call)
    command_keywords = _call_keywords(command_call)
    assert isinstance(command_keywords["platform_terrain_start"], ast.Name)
    assert command_keywords["platform_terrain_start"].id == "D1_PLATFORM_TERRAIN_START"
    assert isinstance(command_keywords["platform_descent_terrain_start"], ast.Name)
    assert command_keywords["platform_descent_terrain_start"].id == "D1_PLATFORM_DESCENT_TERRAIN_START"
    assert ast.literal_eval(command_keywords["platform_descent_bidirectional"]) is True

    platform_cfg = _class(platform_tree, "D1PlatformNP3OEnvCfg")
    assert "observations" not in _assigned_names(platform_cfg)

    base_tree = _tree(D1_ROOT / "base_env_cfg.py")
    policy = _class(_class(base_tree, "ObservationsCfg"), "PolicyCfg")
    assert _assigned_names(policy) == {
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    }

    rough_source = _read(D1_ROOT / "np3o" / "rough_env_cfg.py")
    assert "self.observations.policy.history_length = 10" in rough_source
    assert "self.observations.policy.flatten_history_dim = False" in rough_source


def test_platform_guidance_stays_stateless_and_reward_only():
    platform_tree = _tree(PLATFORM_CFG)
    platform_env = _class(platform_tree, "D1PlatformRLEnv")
    rewards = _class(platform_tree, "PlatformRewardsCfg")
    costs = _class(platform_tree, "PlatformCostsCfg")
    forbidden_terms = {
        "highplatform_yaw",
        "highplatform_progress",
        "platform_traversal",
        "platform_safety",
    }

    assert not (_assigned_names(rewards) & forbidden_terms)
    assert _assigned_names(costs) == {"joint_pos_limit", "joint_vel_limit", "joint_torque_limit"}

    unclipped_terms = next(
        node.value
        for node in platform_env.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_UNCLIPPED_PENALTY_TERMS"
            for target in node.targets
        )
    )
    assert ast.literal_eval(unclipped_terms) == (
        "contact_forces",
        "descent_front_impact",
        "ascent_rear_lateral_force",
        "descent_axle_support",
    )
    step_source = ast.unparse(
        next(node for node in platform_env.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    )
    assert "reward_manager._step_reward[:, penalty_indices]" in step_source
    assert "regular_reward = reward - termination_reward - unclipped_penalty_reward" in step_source
    assert "torch.clamp(regular_reward, min=0.0)" in step_source

    contact_force_call = next(
        node.value
        for node in rewards.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "contact_forces" for target in node.targets)
    )
    assert isinstance(contact_force_call, ast.Call)
    contact_force_keywords = _call_keywords(contact_force_call)
    assert ast.literal_eval(contact_force_keywords["weight"]) == -4.0e-3
    contact_force_params = contact_force_keywords["params"]
    assert isinstance(contact_force_params, ast.Dict)
    contact_force_param_values = {
        ast.literal_eval(key): value for key, value in zip(contact_force_params.keys, contact_force_params.values)
    }
    assert ast.literal_eval(contact_force_param_values["threshold"]) == 400.0

    reward_tree = _tree(LOCOMOTION_ROOT / "mdp" / "platform_rewards.py")
    reward_functions = {node.name for node in reward_tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "platform_feet_stumble",
        "platform_landing_force_penalty",
        "platform_descent_front_impact_penalty",
        "platform_ascent_rear_lateral_force_penalty",
        "platform_ascent_diagnostics",
        "platform_descent_axle_support_penalty",
    } <= reward_functions

    assert {
        "ascent_rear_lateral_force",
        "ascent_diagnostics",
        "descent_front_impact",
        "descent_axle_support",
    } <= _assigned_names(rewards)
    reward_calls = {
        target.id: node.value
        for node in rewards.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    diagnostics_keywords = _call_keywords(reward_calls["ascent_diagnostics"])
    lateral_force_keywords = _call_keywords(reward_calls["ascent_rear_lateral_force"])
    front_impact_keywords = _call_keywords(reward_calls["descent_front_impact"])
    descent_keywords = _call_keywords(reward_calls["descent_axle_support"])
    assert ast.literal_eval(diagnostics_keywords["weight"]) == 1.0
    assert ast.literal_eval(lateral_force_keywords["weight"]) == -4.0e-3
    assert ast.literal_eval(front_impact_keywords["weight"]) == -8.0e-3
    assert ast.literal_eval(descent_keywords["weight"]) == -0.1

    diagnostics_params = {
        ast.literal_eval(key): value
        for key, value in zip(
            diagnostics_keywords["params"].keys,
            diagnostics_keywords["params"].values,
        )
    }
    descent_params = {
        ast.literal_eval(key): value
        for key, value in zip(
            descent_keywords["params"].keys,
            descent_keywords["params"].values,
        )
    }
    lateral_force_params = {
        ast.literal_eval(key): value
        for key, value in zip(
            lateral_force_keywords["params"].keys,
            lateral_force_keywords["params"].values,
        )
    }
    front_impact_params = {
        ast.literal_eval(key): value
        for key, value in zip(
            front_impact_keywords["params"].keys,
            front_impact_keywords["params"].values,
        )
    }
    assert ast.literal_eval(front_impact_params["threshold"]) == 400.0
    assert ast.literal_eval(front_impact_params["command_threshold"]) == 0.10
    assert ast.literal_eval(front_impact_params["command_name"]) == "base_velocity"
    assert isinstance(front_impact_params["terrain_type_start"], ast.Name)
    assert front_impact_params["terrain_type_start"].id == "D1_PLATFORM_DESCENT_TERRAIN_START"
    assert ast.literal_eval(lateral_force_params["threshold"]) == 200.0
    assert isinstance(lateral_force_params["terrain_type_start"], ast.Name)
    assert lateral_force_params["terrain_type_start"].id == "D1_PLATFORM_TERRAIN_START"
    assert isinstance(lateral_force_params["terrain_type_end"], ast.Name)
    assert lateral_force_params["terrain_type_end"].id == "D1_PLATFORM_DESCENT_TERRAIN_START"
    assert isinstance(diagnostics_params["terrain_type_start"], ast.Name)
    assert diagnostics_params["terrain_type_start"].id == "D1_PLATFORM_TERRAIN_START"
    assert isinstance(diagnostics_params["terrain_type_end"], ast.Name)
    assert diagnostics_params["terrain_type_end"].id == "D1_PLATFORM_DESCENT_TERRAIN_START"
    assert isinstance(descent_params["terrain_type_start"], ast.Name)
    assert descent_params["terrain_type_start"].id == "D1_PLATFORM_DESCENT_TERRAIN_START"
    assert ast.literal_eval(descent_params["support_history_steps"]) == 10
    assert ast.literal_eval(descent_params["stable_history_fraction"]) == 0.8

    scoped_source = _read(PLATFORM_CFG) + _read(LOCOMOTION_ROOT / "mdp" / "platform_rewards.py")
    assert "preserve_order=True" in scoped_source
    assert "_platform_phase_masks" not in scoped_source
    assert "descent_latch" not in scoped_source

    utils_tree = _tree(LOCOMOTION_ROOT / "mdp" / "platform_utils.py")
    assert {node.name for node in utils_tree.body if isinstance(node, ast.FunctionDef)} == {
        "platform_terrain_type_start_index"
    }
    assert not any(isinstance(node, ast.ClassDef) for node in utils_tree.body)

    scoped_source = _read(PLATFORM_CFG) + _read(LOCOMOTION_ROOT / "mdp" / "__init__.py")
    forbidden_symbols = {
        "_platform_term_params",
        "HighPlatformProgress",
        "PlatformTraversalSettings",
        "platform_traversal_reward",
        "platform_safety_cost",
    }
    assert not {symbol for symbol in forbidden_symbols if symbol in scoped_source}
