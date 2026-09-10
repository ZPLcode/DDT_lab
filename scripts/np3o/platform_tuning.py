# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded platform tuning and fixed-protocol evaluation decisions (no simulator imports)."""

import json
import math
from pathlib import Path

EVALUATION_HEIGHTS = (0.30, 0.40)
EVALUATION_PROTOCOL = "platform-monitor-v2-30plus"

DEFAULTS = {
    "approach_penalty_scale": 2.0,
    "clearance_bonus": 0.25,
}
LIMITS = {
    "approach_penalty_scale": (0.5, 6.0),
    "clearance_bonus": (0.25, 1.0),
}


def validate_tuning(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(DEFAULTS):
        raise ValueError("Tuning must specify exactly the allowed platform parameters.")
    for key, value in parameters.items():
        low, high = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite or non-numeric tuning value: {key}")
        if not low <= value <= high:
            raise ValueError(f"{key} must be within [{low}, {high}].")
    return dict(parameters)


def apply_tuning(env_cfg, filename):
    document = json.loads(Path(filename).read_text())
    if document.get("version") != 1:
        raise ValueError("Unsupported platform tuning version.")
    params = validate_tuning(document["parameters"])
    reward = env_cfg.rewards.ascent_rear_step.params
    if reward["clearance_margin"] != 0.06 or reward["wall_force_threshold"] != 20.0:
        raise ValueError("Automatic tuning requires the frozen clearance and wall detection definitions.")
    curriculum = env_cfg.curriculum.terrain_levels.params
    if curriculum.get("clean_ascent_reward_term") != "ascent_rear_step":
        raise ValueError("Automatic tuning requires the clean-ascent curriculum gate.")
    ascent = env_cfg.scene.terrain.terrain_generator.sub_terrains["highplatform_up"]
    if tuple(ascent.step_height_range) != (0.30, 1.0):
        raise ValueError("Automatic tuning requires the 30 cm to 1 m ascending-platform curriculum.")
    for key, value in params.items():
        reward[key] = value
    return params


def summarize_evaluations(reports):
    if len(reports) != 2:
        raise ValueError("Exactly two height evaluations are required.")
    for key in (
        "seed",
        "command_forward_m_s",
        "episode_seconds",
        "reward_source_sha256",
        "evaluator_sha256",
        "training_ascent_height_range_m",
    ):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Height evaluation protocols differ: {key}")
    if reports[0]["seed"] != 42 or reports[0]["command_forward_m_s"] != 0.4 or reports[0]["episode_seconds"] != 20:
        raise ValueError("The fixed seed, duration and forward command must not change.")
    hashes = {p["checkpoint_sha256"] for r in reports for p in r["policies"]}
    if len(hashes) != 1:
        raise ValueError("The two heights must evaluate the same checkpoint.")
    rows = []
    for report in reports:
        if report["protocol"] != EVALUATION_PROTOCOL or len(report["policies"]) != 1:
            raise ValueError("Unexpected evaluation protocol or number of policies.")
        if report["training_ascent_height_range_m"] != [0.30, 1.0]:
            raise ValueError("Evaluation must use the 30 cm to 1 m ascent curriculum.")
        policy = report["policies"][0]
        n = report["environments_per_policy"]
        t, ep = policy["totals"], policy["per_episode"]
        if n != 64 or t["finished_episode"] != n:
            raise ValueError("Evaluation did not complete all 64 episodes.")
        for values in ep.values():
            if len(values) != n or any(not math.isfinite(x) for x in values):
                raise ValueError("Incomplete or non-finite evaluation data.")
        for key, value in t.items():
            if not math.isfinite(value) or abs(sum(ep[key]) - value) > 1e-3:
                raise ValueError("Evaluation totals disagree with episode data.")
        if any(x != 1 for x in ep["finished_episode"]):
            raise ValueError("Each environment must finish exactly one episode.")
        if not math.isfinite(policy["mean_max_forward_m"]):
            raise ValueError("Invalid forward progress.")
        if (
            abs(t["landings"] - t["clean_landings"] - t["dirty_landings"] - t["no_hit_but_no_clearance_landings"])
            > 1e-5
        ):
            raise ValueError("Landing event accounting is inconsistent.")
        clean_successes = sum(
            ep["full_traverse"][i] > 0
            and ep["clean_pair"][i] > 0
            and ep["attempts_wall_hit"][i] == 0
            and ep["terminated"][i] == 0
            and ep["max_abs_lateral_m"][i] <= 1.0
            for i in range(n)
        )
        rows.append({
            "height_m": report["step_height_m"],
            "episodes": n,
            "clean_successes": clean_successes,
            "clean_success_rate": clean_successes / n,
            "traversal_rate": t["full_traverse"] / n,
            "clean_landing_fraction": t["clean_landings"] / max(t["landings"], 1),
            "wall_seconds_per_episode": t["wall_contact_wheel_seconds"] / n,
            "mean_max_forward_m": policy["mean_max_forward_m"],
            "wall_attempts": t["attempts_wall_hit"],
            "first_hits_airborne": t["first_hit_airborne"],
            "no_clearance_fraction": t["no_hit_but_no_clearance_landings"] / max(t["landings"], 1),
        })
    rows.sort(key=lambda x: x["height_m"])
    if [r["height_m"] for r in rows] != list(EVALUATION_HEIGHTS):
        raise ValueError("Both frozen 30 cm and 40 cm evaluations are required; 20 cm is outside the objective.")
    summary = {
        key: sum(r[key] for r in rows) / len(rows)
        for key in [
            "clean_success_rate",
            "traversal_rate",
            "clean_landing_fraction",
            "wall_seconds_per_episode",
            "mean_max_forward_m",
            "no_clearance_fraction",
        ]
    }
    hits = sum(r["wall_attempts"] for r in rows)
    summary["grounded_first_hit_fraction"] = 1 - sum(r["first_hits_airborne"] for r in rows) / hits if hits else 0.0
    summary["by_height"] = rows
    return summary


def compare_results(current, reference):
    """Practical effect thresholds; these are not statistical significance tests."""
    for result in (current, reference):
        if [row["height_m"] for row in result["by_height"]] != list(EVALUATION_HEIGHTS):
            raise ValueError("Comparisons must use the same 30/40 cm protocol.")
    guard = all(
        c["traversal_rate"] >= r["traversal_rate"] - 0.05 and c["clean_success_rate"] >= r["clean_success_rate"] - 0.05
        for c, r in zip(current["by_height"], reference["by_height"])
    )
    if all(r["clean_success_rate"] >= 0.95 for r in current["by_height"]):
        return "maintain", "两种固定高度的无碰墙干净通过率均已达到 95%，保持当前设置。"
    if guard and current["clean_success_rate"] >= reference["clean_success_rate"] + 0.05:
        return "improved", "无碰墙干净通过率提高至少 5 个百分点，且各高度通过率无明显退步。"
    less_contact = (
        reference["wall_seconds_per_episode"] > 1e-6
        and current["wall_seconds_per_episode"] <= reference["wall_seconds_per_episode"] * 0.85
    )
    if guard and less_contact and current["clean_landing_fraction"] >= reference["clean_landing_fraction"] - 0.02:
        return "improved", "碰墙累计时长下降至少 15%，干净落台比例和通过率未明显退步。"
    if (
        guard
        and current["clean_landing_fraction"] >= reference["clean_landing_fraction"] + 0.05
        and current["wall_seconds_per_episode"] <= reference["wall_seconds_per_episode"] * 1.05 + 1e-6
    ):
        return "improved", "干净落台比例提高至少 5 个百分点，碰墙时长未明显增加。"
    return "stagnant", "固定协议下未达到改善门槛，或通过率出现退步。"


def propose_tuning(current, reference, parameters):
    params = validate_tuning(parameters)

    def change(key, value):
        low, high = LIMITS[key]
        params[key] = round(min(high, max(low, value)), 6)

    if (
        current["traversal_rate"] < reference["traversal_rate"] - 0.10
        or current["mean_max_forward_m"] < reference["mean_max_forward_m"] * 0.8
    ):
        change("approach_penalty_scale", params["approach_penalty_scale"] * 0.8)
        change("clearance_bonus", params["clearance_bonus"] * 1.25)
        reason = "移动能力下降：适度减轻接近惩罚，同时加强主动抬轮反馈。"
    elif current["grounded_first_hit_fraction"] >= 0.7:
        change("approach_penalty_scale", params["approach_penalty_scale"] * 1.35)
        change("clearance_bonus", params["clearance_bonus"] * 1.25)
        reason = "首碰时后轮大多尚未离地：加强提前避墙与抬轮反馈。"
    elif current["no_clearance_fraction"] >= 0.25:
        change("clearance_bonus", params["clearance_bonus"] * 1.3)
        reason = "未达到提前净空的落台较多：加强已有抬轮反馈。"
    else:
        change("approach_penalty_scale", params["approach_penalty_scale"] * 1.2)
        change("clearance_bonus", params["clearance_bonus"] * 1.2)
        reason = "碰墙仍持续：小幅加强已有的提前避墙与抬轮反馈。"
    if params == parameters:
        reason = "两个权重已达到本规则的边界，保持训练并标记需人工分析；不新增 reward 项。"
    return validate_tuning(params), reason
