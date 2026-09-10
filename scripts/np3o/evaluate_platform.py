# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Paired, fixed-height checkpoint diagnosis; does not update either policy."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--step_height", type=float, default=0.30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reuse_completed", action="store_true")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--record_trajectories", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import __future__

import ast
import hashlib
import importlib
import inspect
import json
import textwrap
import time

import ddt_lab.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from ddt_lab.algorithms.np3o import IsaacLabNP3OWrapper, OnConstraintPolicyRunner
from ddt_lab.tasks.manager_based.locomotion.mdp.rear_step_rewards import RearWheelStepReward
from isaaclab.utils.io import dump_yaml


def expose_reward_state():
    """Observe existing gate tensors in this evaluation process only."""
    original = RearWheelStepReward.__call__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    names = [
        "arm",
        "valid",
        "wall_contact",
        "front_ready",
        "airborne",
        "free_swing",
        "clean_swing",
        "landed",
        "clean_landing",
        "lift",
        "edge_progress",
    ]
    expression = (
        "self.evaluation_state = {"
        + ",".join(repr(name) + ": " + name + ".detach().clone()" for name in names)
        + ", 'cleared': self.cleared.clone(), 'wall_touched': self.wall_touched.clone(), 'clean_pair':"
        " self.episode_clean_landed.all(dim=1).clone()}"
    )
    tree.body[0].body.insert(-1, ast.parse(expression).body[0])
    namespace = dict(original.__globals__)
    code = compile(
        ast.fix_missing_locations(tree), "<reward-evaluation-probe>", "exec", flags=__future__.annotations.compiler_flag
    )
    exec(code, namespace)
    RearWheelStepReward.__call__ = namespace["__call__"]


def main():
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expose_reward_state()
    task = "DDT-Velocity-Platform-D1-NP3O-v0"
    spec = gym.spec(task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.max_init_terrain_level = 0
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 1
    generator.num_cols = 20
    generator.seed = args.seed
    training_height_range = generator.sub_terrains["highplatform_up"].step_height_range
    minimum_height, maximum_height = training_height_range
    if not minimum_height <= args.step_height <= maximum_height:
        raise ValueError("Evaluation height must be within the ascending-platform curriculum range.")
    difficulty = (args.step_height - minimum_height) / (maximum_height - minimum_height)
    generator.difficulty_range = (difficulty, difficulty)
    generator.sub_terrains["highplatform_up"].step_height_range = (args.step_height, args.step_height)
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.highplatform_levels = None
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.platform_ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.platform_ranges.heading = (0.0, 0.0)
    dump_yaml(str(output / "env.yaml"), cfg)
    env = IsaacLabNP3OWrapper(gym.make(task, cfg=cfg), device="cuda:0")
    module_name, attr = spec.kwargs["np3o_cfg_entry_point"].split(":")
    runner_cfg = getattr(importlib.import_module(module_name), attr)()
    runner = OnConstraintPolicyRunner(env, runner_cfg, log_dir=None, device="cuda:0")
    raw = env.unwrapped
    terrain = raw.scene.terrain
    terrain.terrain_types[:] = torch.arange(args.num_envs, device=raw.device) % 6 + 8
    terrain.terrain_levels[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    reward = raw.reward_manager.get_term_cfg("ascent_rear_step").func
    assert isinstance(reward, RearWheelStepReward)
    wheels = raw.scene["robot"].find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)[0]
    report = {
        "protocol": "platform-monitor-v2-30plus",
        "training_ascent_height_range_m": list(training_height_range),
        "step_height_m": args.step_height,
        "command_forward_m_s": 0.4,
        "environments_per_policy": args.num_envs,
        "seed": args.seed,
        "episode_seconds": raw.max_episode_length_s,
        "conditions": (
            "Fixed rough ascending platform; startup and reset randomization retained; pushes and observation noise"
            " disabled; deterministic policy mean; first episode only per environment."
        ),
        "wall_measurement": "The training reward's geometry-gated net-force estimate, not contact-pair ground truth.",
        "reward_source_sha256": hashlib.sha256(Path(inspect.getfile(RearWheelStepReward)).read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policies": [],
    }
    first_initial = None
    cached = {}
    if args.reuse_completed and (output / "results.json").exists():
        previous = json.loads((output / "results.json").read_text())
        for key in ["step_height_m", "command_forward_m_s", "environments_per_policy", "seed", "reward_source_sha256"]:
            assert previous[key] == report[key], "Cached evaluation settings differ"
        cached = {p["checkpoint_sha256"]: p for p in previous["policies"]}
    try:
        for checkpoint in args.checkpoints:
            path = Path(checkpoint).resolve()
            runner.alg.actor_critic.load_state_dict(
                torch.load(path, map_location="cuda:0", weights_only=False)["model_state_dict"]
            )
            policy = runner.get_inference_policy("cuda:0")
            raw.common_step_counter = 0
            with torch.inference_mode():
                obs_dict, _ = env.env.reset(seed=args.seed)
                obs = obs_dict["policy"]
                initial = torch.cat(
                    (
                        raw.scene["robot"].data.root_state_w,
                        raw.scene["robot"].data.joint_pos,
                        raw.scene["robot"].data.joint_vel,
                    ),
                    dim=-1,
                ).clone()
            if first_initial is None:
                first_initial = initial.clone()
            assert torch.allclose(initial, first_initial, atol=1e-6), "Paired initial states differ"
            assert torch.allclose(
                raw.command_manager.get_command("base_velocity")[:, 0],
                torch.full((args.num_envs,), 0.4, device=raw.device),
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in cached:
                assert cached[digest]["totals"]["finished_episode"] == args.num_envs
                report["policies"].append(cached[digest])
                print("REUSED " + str(path), flush=True)
                continue
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)

            def zeros():
                return torch.zeros(args.num_envs, device=raw.device)

            counts = {
                name: zeros()
                for name in [
                    "attempts",
                    "attempts_front_ready",
                    "attempts_airborne",
                    "attempts_cleared",
                    "attempts_wall_hit",
                    "first_hit_before_front_ready",
                    "first_hit_airborne",
                    "landings",
                    "clean_landings",
                    "dirty_landings",
                    "no_hit_but_no_clearance_landings",
                    "wall_contact_wheel_seconds",
                    "wall_first_lift_sum",
                    "wall_first_edge_progress_sum",
                    "terminated",
                    "finished_episode",
                    "full_traverse",
                    "max_forward_m",
                    "clean_pair",
                    "max_abs_lateral_m",
                ]
            }
            seen = {
                name: torch.zeros((args.num_envs, 2), dtype=torch.bool, device=raw.device)
                for name in ["front", "airborne", "cleared", "hit"]
            }
            trajectory = []
            started = time.monotonic()
            for step in range(int(raw.max_episode_length)):
                with torch.inference_mode():
                    obs, _, _, _, done, info = env.step(policy(obs))
                    st = reward.evaluation_state
                    mask = alive[:, None]
                    armed = st["arm"] & mask
                    counts["attempts"] += armed.sum(-1)
                    for flags in seen.values():
                        flags &= ~armed
                    for key, condition, metric in [
                        ("front", st["valid"] & st["front_ready"], "attempts_front_ready"),
                        ("airborne", st["valid"] & st["airborne"], "attempts_airborne"),
                        ("cleared", st["valid"] & st["cleared"], "attempts_cleared"),
                    ]:
                        event = condition & mask & ~seen[key]
                        counts[metric] += event.sum(-1)
                        seen[key] |= condition & mask
                    hit = st["wall_contact"] & mask & ~seen["hit"]
                    counts["attempts_wall_hit"] += hit.sum(-1)
                    counts["first_hit_before_front_ready"] += (hit & ~seen["front"]).sum(-1)
                    counts["first_hit_airborne"] += (hit & st["airborne"]).sum(-1)
                    counts["wall_first_lift_sum"] += torch.where(hit, st["lift"], 0).sum(-1)
                    counts["wall_first_edge_progress_sum"] += torch.where(hit, st["edge_progress"], 0).sum(-1)
                    seen["hit"] |= st["wall_contact"] & mask
                    landed = st["landed"] & mask
                    counts["landings"] += landed.sum(-1)
                    counts["clean_landings"] += (st["clean_landing"] & mask).sum(-1)
                    counts["dirty_landings"] += (landed & st["wall_touched"]).sum(-1)
                    counts["no_hit_but_no_clearance_landings"] += (landed & ~st["wall_touched"] & ~st["cleared"]).sum(
                        -1
                    )
                    counts["wall_contact_wheel_seconds"] += (st["wall_contact"] & mask).sum(-1) * raw.step_dt
                    robot = raw.scene["robot"]
                    relative = robot.data.root_pos_w - raw.scene.env_origins
                    counts["clean_pair"] = torch.maximum(counts["clean_pair"], (st["clean_pair"] & alive).float())
                    counts["max_abs_lateral_m"] = torch.where(
                        alive & ~done,
                        torch.maximum(counts["max_abs_lateral_m"], relative[:, 1].abs()),
                        counts["max_abs_lateral_m"],
                    )
                    counts["max_forward_m"] = torch.where(
                        alive & ~done, torch.maximum(counts["max_forward_m"], relative[:, 0]), counts["max_forward_m"]
                    )
                    # Traversal of all three rising treads, separate from reward milestones.
                    rear_bottom = robot.data.body_pos_w[:, wheels[2:], 2] - 0.087
                    traversed = (
                        (relative[:, 0] > 3.4)
                        & (rear_bottom.amin(-1) > -0.05)
                        & (-robot.data.projected_gravity_b[:, 2] > 0.8)
                    )
                    counts["full_traverse"] = torch.maximum(
                        counts["full_traverse"], (traversed & alive & ~done).float()
                    )
                    counts["terminated"] += (alive & done & ~info["time_outs"]).float()
                    counts["finished_episode"] += (alive & done).float()
                    if args.record_trajectories and step % 5 == 0:
                        trajectory.append(
                            torch.cat(
                                (
                                    relative,
                                    robot.data.body_pos_w[:, wheels].reshape(args.num_envs, -1),
                                    alive[:, None].float(),
                                ),
                                -1,
                            )
                            .cpu()
                            .numpy()
                        )
                    alive &= ~done
                if (step + 1) % 200 == 0:
                    print(
                        json.dumps({
                            "checkpoint": path.name,
                            "step": step + 1,
                            "alive": int(alive.sum()),
                            "wall_attempts": float(counts["attempts_wall_hit"].sum()),
                            "clean_landings": float(counts["clean_landings"].sum()),
                        }),
                        flush=True,
                    )
                if not alive.any():
                    break
            values = {name: value.cpu().numpy() for name, value in counts.items()}
            totals = {name: float(value.sum()) for name, value in values.items()}
            result = {
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "steps": step + 1,
                "elapsed_seconds": time.monotonic() - started,
                "totals": totals,
                "episodes_with_clean_landing": int((values["clean_landings"] > 0).sum()),
                "episodes_with_wall_hit": int((values["attempts_wall_hit"] > 0).sum()),
                "mean_max_forward_m": float(values["max_forward_m"].mean()),
                "per_episode": {name: value.tolist() for name, value in values.items()},
            }
            report["policies"].append(result)
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            if args.record_trajectories:
                np.savez_compressed(output / (path.stem + "_trajectories.npz"), trajectory=np.stack(trajectory))
            print("RESULT " + json.dumps({k: v for k, v in result.items() if k != "per_episode"}), flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
