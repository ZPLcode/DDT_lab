# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation validity, conservative tuning, and persisted restart decisions."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/np3o"
sys.path.insert(0, str(SCRIPTS))
tuning = importlib.import_module("platform_tuning")
monitor = importlib.import_module("monitor_platform")
sys.path.remove(str(SCRIPTS))


def reports():
    episodes = {
        "finished_episode": [1] * 64,
        "full_traverse": [1] * 64,
        "clean_pair": [1] * 64,
        "attempts_wall_hit": [0] * 32 + [1] * 32,
        "terminated": [0] * 64,
        "max_abs_lateral_m": [0.3] * 64,
        "landings": [6] * 64,
        "clean_landings": [3] * 64,
        "dirty_landings": [2] * 64,
        "no_hit_but_no_clearance_landings": [1] * 64,
        "wall_contact_wheel_seconds": [0] * 32 + [2] * 32,
        "first_hit_airborne": [0] * 64,
    }
    report = {
        "protocol": "platform-monitor-v2-30plus",
        "training_ascent_height_range_m": [0.30, 1.0],
        "seed": 42,
        "command_forward_m_s": 0.4,
        "episode_seconds": 20,
        "reward_source_sha256": "reward",
        "evaluator_sha256": "evaluator",
        "environments_per_policy": 64,
        "policies": [{
            "checkpoint_sha256": "checkpoint",
            "per_episode": episodes,
            "totals": {k: sum(v) for k, v in episodes.items()},
            "mean_max_forward_m": 6.0,
        }],
    }
    return [dict(copy.deepcopy(report), step_height_m=height) for height in (0.3, 0.4)]


def test_success_requires_no_hit_both_rears_and_no_bypass():
    data = reports()
    episode = data[0]["policies"][0]["per_episode"]
    episode["clean_pair"][0] = 0
    episode["max_abs_lateral_m"][1] = 1.1
    episode["terminated"][2] = 1
    data[0]["policies"][0]["totals"] = {k: sum(v) for k, v in episode.items()}
    result = tuning.summarize_evaluations(data)
    assert [r["clean_successes"] for r in result["by_height"]] == [29, 32]


def test_old_20cm_protocol_cannot_affect_new_objective():
    data = reports()
    data[0]["step_height_m"] = 0.2
    with pytest.raises(ValueError, match="20 cm is outside"):
        tuning.summarize_evaluations(data)
    data = reports()
    data[0]["protocol"] = "platform-monitor-v1"
    with pytest.raises(ValueError, match="protocol"):
        tuning.summarize_evaluations(data)
    current = tuning.summarize_evaluations(reports())
    reference = copy.deepcopy(current)
    reference["by_height"][0]["height_m"] = 0.2
    with pytest.raises(ValueError, match="30/40 cm"):
        tuning.compare_results(current, reference)


@pytest.mark.parametrize(
    "fault", ["partial", "nan", "different_model", "different_source", "duplicate_height", "bad_totals"]
)
def test_invalid_evaluations_cannot_trigger_tuning(fault):
    data = reports()
    policy = data[0]["policies"][0]
    if fault == "partial":
        policy["totals"]["finished_episode"] = 63
    elif fault == "nan":
        policy["per_episode"]["max_abs_lateral_m"][0] = float("nan")
    elif fault == "different_model":
        policy["checkpoint_sha256"] = "other"
    elif fault == "different_source":
        data[0]["reward_source_sha256"] = "other"
    elif fault == "duplicate_height":
        data[0]["step_height_m"] = 0.4
    else:
        policy["totals"]["clean_landings"] += 1
    with pytest.raises(ValueError):
        tuning.summarize_evaluations(data)


def test_weak_changes_or_stalling_are_not_improvements():
    reference = tuning.summarize_evaluations(reports())
    current = copy.deepcopy(reference)
    current["clean_landing_fraction"] += 0.02
    current["wall_seconds_per_episode"] *= 0.95
    assert tuning.compare_results(current, reference)[0] == "stagnant"
    current["wall_seconds_per_episode"] *= 0.5
    assert tuning.compare_results(current, reference)[0] == "improved"
    current["by_height"][1]["traversal_rate"] = 0.8
    assert tuning.compare_results(current, reference)[0] == "stagnant"
    current["by_height"][1]["traversal_rate"] = 1.0
    current["by_height"][1]["clean_success_rate"] = 0.0
    assert tuning.compare_results(current, reference)[0] == "stagnant"


def test_clean_success_improvement_and_maintenance():
    reference = tuning.summarize_evaluations(reports())
    current = copy.deepcopy(reference)
    current["clean_success_rate"] += 0.10
    assert tuning.compare_results(current, reference)[0] == "improved"
    for row in current["by_height"]:
        row["clean_success_rate"] = 1.0
    assert tuning.compare_results(current, reference)[0] == "maintain"


def test_only_two_existing_weights_change_with_bounds_and_mobility_recovery():
    reference = tuning.summarize_evaluations(reports())
    params, _ = tuning.propose_tuning(reference, reference, tuning.DEFAULTS)
    assert set(params) == {"approach_penalty_scale", "clearance_bonus"}
    assert params["approach_penalty_scale"] > tuning.DEFAULTS["approach_penalty_scale"]
    stalled = dict(reference, traversal_rate=0.5)
    recovery, _ = tuning.propose_tuning(stalled, reference, params)
    assert recovery["approach_penalty_scale"] < params["approach_penalty_scale"]
    for _ in range(30):
        params, _ = tuning.propose_tuning(reference, reference, params)
    assert params == {key: bounds[1] for key, bounds in tuning.LIMITS.items()}


@pytest.mark.parametrize(
    "parameters",
    [
        dict(tuning.DEFAULTS, clearance_bonus=float("nan")),
        dict(tuning.DEFAULTS, clearance_bonus=2),
        dict(tuning.DEFAULTS, wall_force_threshold=999),
    ],
)
def test_tuning_rejects_out_of_bounds_or_detection_changes(parameters):
    with pytest.raises(ValueError):
        tuning.validate_tuning(parameters)


def test_apply_tuning_keeps_reward_structure_and_detection(tmp_path):
    reward = dict(tuning.DEFAULTS, clearance_margin=0.06, wall_force_threshold=20.0, landing_bonus=1.0)
    cfg = NS(
        rewards=NS(ascent_rear_step=NS(params=reward)),
        curriculum=NS(terrain_levels=NS(params={"clean_ascent_reward_term": "ascent_rear_step"})),
        scene=NS(terrain=NS(terrain_generator=NS(sub_terrains={"highplatform_up": NS(step_height_range=(0.30, 1.0))}))),
    )
    original = copy.deepcopy(reward)
    file = tmp_path / "tuning.json"
    file.write_text(
        json.dumps({"version": 1, "parameters": {"approach_penalty_scale": 2.7, "clearance_bonus": 0.3125}})
    )
    tuning.apply_tuning(cfg, file)
    assert set(reward) == set(original)
    assert {k for k in reward if reward[k] != original[k]} == set(tuning.DEFAULTS)
    ascent = cfg.scene.terrain.terrain_generator.sub_terrains["highplatform_up"]
    ascent.step_height_range = (0.05, 1.0)
    with pytest.raises(ValueError, match="30 cm to 1 m"):
        tuning.apply_tuning(cfg, file)
    ascent.step_height_range = (0.30, 1.0)
    reward["wall_force_threshold"] = 100.0
    with pytest.raises(ValueError):
        tuning.apply_tuning(cfg, file)


def test_resume_command_uses_remaining_budget_and_exact_model(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    checkpoint = tmp_path / "logs/np3o/test/old.run/model_3100.pt"
    command = monitor.train_command(
        {"experiment": "test", "num_envs": 4096, "seed": 42}, checkpoint, tmp_path / "tuning.json", 20000
    )
    assert command[command.index("--max_iterations") + 1] == "16900"
    assert command[command.index("--checkpoint") + 1] == r"^model_3100\.pt$"
    assert command[command.index("--load_run") + 1] == r"^old\.run$"
    with pytest.raises(ValueError):
        monitor.train_command(
            {"experiment": "test", "num_envs": 4096, "seed": 42}, checkpoint, tmp_path / "tuning.json", 3000
        )


def test_recovered_restart_finalizes_old_milestone_once(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "DIRECTORY", tmp_path)
    monkeypatch.setattr(monitor, "LATEST", tmp_path / "launch.json")
    monitor.write(monitor.LATEST, {"run_directory": str(tmp_path / "new_run_without_3000")})
    metrics = tuning.summarize_evaluations(reports())
    entry = {
        "iteration": 3000,
        "metrics": metrics,
        "decision": "stagnant",
        "reason": "flat",
        "parameters": tuning.DEFAULTS,
        "checkpoint": "/old/model_3000.pt",
    }
    new_params, _ = tuning.propose_tuning(metrics, metrics, tuning.DEFAULTS)
    monitor.write(
        tmp_path / "state.json",
        {
            "next_iteration": 3000,
            "target_iteration": 20000,
            "parameters": new_params,
            "reference": {"iteration": 1500, "metrics": metrics},
            "pending_entry": entry,
            "last_tuned_iteration": 3000,
            "history": [],
            "status": "waiting",
        },
    )
    instance = monitor.Monitor()
    monkeypatch.setattr(instance, "assert_sources", lambda: None)
    monkeypatch.setattr(instance, "prepare_restart", lambda *args: pytest.fail("Duplicate restart"))
    monkeypatch.setattr(instance, "evaluate", lambda *args: pytest.fail("Duplicate evaluation"))
    monkeypatch.setattr(instance, "record", lambda entry: None)
    instance.tick()
    assert instance.state["next_iteration"] == 6000
    assert instance.state["pending_entry"] is None
    assert len(instance.state["history"]) == 1
    assert instance.state["history"][0]["applied_parameters"] == new_params


def test_failed_evaluation_does_not_prepare_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "DIRECTORY", tmp_path)
    monkeypatch.setattr(monitor, "LATEST", tmp_path / "launch.json")
    source = tmp_path / "model_3000.pt"
    monitor.write(monitor.LATEST, {"run_directory": str(tmp_path)})
    monitor.write(tmp_path / "state.json", {"next_iteration": 3000, "target_iteration": 20000, "status": "waiting"})
    monkeypatch.setattr(monitor, "checkpoints", lambda run: [source])
    instance = monitor.Monitor()
    monkeypatch.setattr(instance, "assert_sources", lambda: None)
    monkeypatch.setattr(instance, "prepare_restart", lambda *args: pytest.fail("Invalid-data restart"))

    def fail(*args):
        raise ValueError("incomplete evaluation")

    monkeypatch.setattr(instance, "evaluate", fail)
    with pytest.raises(ValueError, match="incomplete"):
        instance.tick()
    assert instance.state["next_iteration"] == 3000


def test_user_stop_does_not_resume_evaluation_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "DIRECTORY", tmp_path)
    monitor.write(tmp_path / "state.json", {"status": "stopped_by_user"})
    instance = monitor.Monitor()
    monkeypatch.setattr(instance, "tick", lambda: pytest.fail("User-stopped monitor must stay stopped"))
    instance.run()
