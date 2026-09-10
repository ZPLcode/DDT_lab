# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent 3k-iteration platform evaluation and bounded, two-weight tuning."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from platform_tuning import (
    DEFAULTS,
    EVALUATION_HEIGHTS,
    EVALUATION_PROTOCOL,
    compare_results,
    propose_tuning,
    summarize_evaluations,
    validate_tuning,
)

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "logs/monitor/platform"
LATEST = ROOT / "logs/launches/rear_clearance_latest.json"
PYTHON = Path(sys.executable)
INTERVAL = 3000
STOP = False
CRITICAL = [
    "scripts/np3o/train.py",
    "scripts/np3o/evaluate_platform.py",
    "scripts/np3o/platform_tuning.py",
    "source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/mdp/rear_step_rewards.py",
    "source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/mdp/curriculums.py",
    "source/ddt_lab/ddt_lab/tasks/manager_based/locomotion/robots/d1/np3o/platform_env_cfg.py",
    "source/ddt_lab/ddt_lab/algorithms/np3o/runner.py",
    "ddt_ros2_control/urdfs/d1_description/urdf/robot.urdf",
    "source/ddt_lab/ddt_lab/assets/terrains/platform.py",
]


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    return {name: sha(ROOT / name) for name in CRITICAL}


def iteration(path):
    match = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
    if not match:
        raise ValueError(f"Invalid checkpoint filename: {path}")
    return int(match[1])


def checkpoints(run):
    # torch.save writes in place. Ignore fresh files until the writer has finished.
    return sorted((p for p in Path(run).glob("model_*.pt") if time.time() - p.stat().st_mtime > 15), key=iteration)


def process_matches(pid, script, extra=None):
    try:
        proc = Path(f"/proc/{int(pid)}")
        command = (proc / "cmdline").read_bytes().decode().split("\0")
        return (proc / "cwd").resolve() == ROOT and script in command and (extra is None or extra in command)
    except (OSError, ValueError, TypeError):
        return False


def stop_process(pid, script, extra=None):
    if not process_matches(pid, script, extra):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 25
    while process_matches(pid, script, extra) and time.monotonic() < deadline:
        time.sleep(1)
    if process_matches(pid, script, extra):
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)


def verify_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["iter"] != iteration(path) or not checkpoint.get("optimizer_state_dict"):
        raise ValueError("Checkpoint iteration or optimizer is missing.")

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        return True

    if not finite(checkpoint):
        raise ValueError("Non-finite checkpoint; training will not be interrupted.")


def train_command(metadata, checkpoint, tuning, target):
    checkpoint = Path(checkpoint).resolve()
    experiment_root = ROOT / "logs/np3o" / metadata["experiment"]
    if checkpoint.parent.parent != experiment_root or target <= iteration(checkpoint):
        raise ValueError("Resume path or training budget is invalid.")
    return [
        str(PYTHON),
        "-u",
        "scripts/np3o/train.py",
        "--task",
        "DDT-Velocity-Platform-D1-NP3O-v0",
        "--num_envs",
        str(metadata["num_envs"]),
        "--max_iterations",
        str(target - iteration(checkpoint)),
        "--experiment_name",
        metadata["experiment"],
        "--seed",
        str(metadata["seed"]),
        "--headless",
        "--resume",
        "--load_run",
        "^" + re.escape(checkpoint.parent.name) + "$",
        "--checkpoint",
        "^" + re.escape(checkpoint.name) + "$",
        "--platform_tuning",
        str(tuning),
    ]


class Monitor:
    def __init__(self):
        self.path = DIRECTORY / "state.json"
        self.state = read(self.path)

    def save(self, **changes):
        self.state.update(changes, updated_at=now())
        write(self.path, self.state)

    def log(self, message):
        print(f"{now()} {message}", flush=True)

    def assert_sources(self):
        if self.state.get("protocol") != EVALUATION_PROTOCOL or self.state.get("evaluation_heights_m") != list(
            EVALUATION_HEIGHTS
        ):
            raise ValueError("Monitor needs a matching 30/40 cm baseline before automatic decisions resume.")
        if fingerprint() != self.state["source_hashes"]:
            raise ValueError("Source files changed; automatic comparison/tuning paused until baseline is reviewed.")

    def evaluate(self, checkpoint, output):
        self.assert_sources()
        reports = []
        for height in EVALUATION_HEIGHTS:
            folder = output / f"h{round(height * 100):03d}"
            folder.mkdir(parents=True, exist_ok=True)
            result = folder / "results.json"
            if not result.exists():
                command = [
                    str(PYTHON),
                    "-u",
                    "scripts/np3o/evaluate_platform.py",
                    "--checkpoints",
                    str(checkpoint),
                    "--step_height",
                    str(height),
                    "--num_envs",
                    "64",
                    "--seed",
                    "42",
                    "--headless",
                    "--output_dir",
                    str(folder),
                ]
                with (folder / "console.log").open("w") as stream:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                self.save(evaluation_pid=process.pid, status=f"evaluating_{iteration(checkpoint)}_{height}")
                deadline = time.monotonic() + 1200
                while process.poll() is None and not STOP and time.monotonic() < deadline:
                    time.sleep(5)
                if STOP or process.poll() is None:
                    stop_process(process.pid, "scripts/np3o/evaluate_platform.py", str(folder))
                    raise RuntimeError("Evaluation interrupted or timed out.")
                self.save(evaluation_pid=None)
                if process.returncode or not result.exists():
                    raise RuntimeError(f"Evaluation failed; see {folder / 'console.log'}")
            report = read(result)
            if len(report["policies"]) != 1 or report["policies"][0]["checkpoint_sha256"] != sha(checkpoint):
                raise ValueError("Evaluation checkpoint does not match the scheduled checkpoint.")
            if report["evaluator_sha256"] != self.state["source_hashes"]["scripts/np3o/evaluate_platform.py"]:
                raise ValueError("Cached evaluator source differs.")
            reports.append(report)
        return summarize_evaluations(reports)

    def record(self, entry):
        folder = Path(
            entry.get(
                "output_directory",
                DIRECTORY / ("baseline" if entry["decision"] == "baseline" else f"iteration_{entry['iteration']:06d}"),
            )
        )
        write(folder / "summary.json", entry)
        metrics = entry["metrics"]
        lines = [
            f"# Platform 第 {entry['iteration']} 轮固定评估",
            "",
            entry["reason"],
            "",
            "当前目标：上台课程最低 30 cm；正式评估 30/40 cm，20 cm 不参与调参。",
            "新基线按当前课程的地形粗糙度生成；与旧 20/30 cm 协议的数据分开比较。",
            "",
            "| 台阶 | 无碰墙干净通过 | 通过率 | 干净落台占比 | 每回合累计碰墙秒数（两后轮相加） |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in metrics["by_height"]:
            lines.append(
                f"| {row['height_m'] * 100:.0f} cm | {row['clean_successes']}/64 | "
                f"{row['traversal_rate']:.1%} | {row['clean_landing_fraction']:.1%} | "
                f"{row['wall_seconds_per_episode']:.3f} |"
            )
        lines += [
            "",
            f"Checkpoint: `{entry['checkpoint']}`",
            "",
            "无碰墙干净通过要求：到达顶部、两只后轮均有干净落台、整回合无后轮碰墙标志、无失败终止、侧偏不超过 1 m。",
            (
                "碰墙沿用训练中的几何与净接触力估计；不是仿真接触对真值。每种高度固定 seed 42、64 回合、20 秒、前进指令"
                " 0.4 m/s。"
            ),
            "训练总 reward 不作为改善依据；这些是固定样本上的实用门槛，不能代替更多随机种子的泛化验证。",
            "",
        ]
        if entry.get("proposed_parameters"):
            lines += [
                "调参原因：" + entry["tuning_reason"],
                "",
                "仅调整已有两个权重：`" + json.dumps(entry["proposed_parameters"]) + "`。",
                "",
            ]
        body = "\n".join(lines)
        (folder / "report.md").write_text(body)
        (DIRECTORY / "latest_report.md").write_text(body)
        # A separate run in the user's existing TensorBoard experiment.
        from torch.utils.tensorboard import SummaryWriter

        experiment = read(LATEST)["experiment"]
        with SummaryWriter(str(ROOT / "logs/np3o" / experiment / "auto_monitor_30plus")) as writer:
            for row in metrics["by_height"]:
                for key in (
                    "clean_success_rate",
                    "traversal_rate",
                    "clean_landing_fraction",
                    "wall_seconds_per_episode",
                ):
                    writer.add_scalar(f"AutoEval/{round(row['height_m'] * 100)}cm/{key}", row[key], entry["iteration"])
        self.log(f"Iteration {entry['iteration']}: {entry['decision']}; {entry['reason']}")

    def prepare_restart(self, parameters, evaluated_iteration):
        self.assert_sources()
        metadata = read(LATEST)
        if not process_matches(metadata["pid"], "scripts/np3o/train.py", metadata["experiment"]):
            raise RuntimeError("Training is stopped; monitor will not restart a manually stopped run.")
        source = checkpoints(metadata["run_directory"])[-1]
        if iteration(source) >= self.state["target_iteration"]:
            return
        verify_checkpoint(source)
        launch = ROOT / "logs/launches" / datetime.now().strftime("rear_clearance_auto_%Y-%m-%d_%H-%M-%S")
        launch.mkdir(parents=True)
        with (launch / "tests.log").open("w") as stream:
            check = subprocess.run(
                [str(PYTHON), "-m", "pytest", "-q", "source/ddt_lab/test"],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        if check.returncode:
            raise RuntimeError(f"Pre-restart tests failed; current training kept running: {launch / 'tests.log'}")
        tuning = launch / "tuning.json"
        write(
            tuning,
            {"version": 1, "parameters": validate_tuning(parameters), "evaluated_iteration": evaluated_iteration},
        )
        write(launch / "previous_launch.json", metadata)
        write(launch / "source_sha256.json", fingerprint())
        with tarfile.open(launch / "source.tar.gz", "w:gz") as archive:
            for name in CRITICAL:
                archive.add(ROOT / name, arcname=name)
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=True)
        (launch / "working_tree.diff").write_bytes(diff.stdout)
        command = train_command(metadata, source, tuning, self.state["target_iteration"])
        self.save(
            restart={
                "previous": metadata,
                "checkpoint": str(source),
                "launch": str(launch),
                "command": command,
                "parameters": parameters,
                "evaluated_iteration": evaluated_iteration,
                "checkpoint_sha256": sha(source),
            },
            status="restart_prepared",
        )

    def finish_restart(self):
        plan = self.state["restart"]
        old, launch = plan["previous"], Path(plan["launch"])
        if "new_metadata" not in plan:
            self.assert_sources()
            # Adopt a child if the service was restarted just after Popen.
            adopted = next(
                (
                    int(p.name)
                    for p in Path("/proc").iterdir()
                    if p.name.isdigit() and process_matches(int(p.name), "scripts/np3o/train.py", plan["command"][-1])
                ),
                None,
            )
            if adopted is None:
                stop_process(old["pid"], "scripts/np3o/train.py", old["experiment"])
                if STOP:
                    return
                with (launch / "train.log").open("a") as stream:
                    process = subprocess.Popen(
                        plan["command"],
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                adopted = process.pid
            metadata = dict(old)
            for key in ("run_directory", "validation", "last_verified_iteration", "last_verified_at"):
                metadata.pop(key, None)
            metadata.update(
                pid=adopted,
                status="starting",
                command=plan["command"],
                started_at=now(),
                console_log=str(launch / "train.log"),
                source_archive=str(launch / "source.tar.gz"),
                resume_checkpoint=plan["checkpoint"],
                resume_iteration=iteration(plan["checkpoint"]),
                resume_checkpoint_sha256=plan["checkpoint_sha256"],
                supersedes_run=old["run_directory"],
                platform_tuning=plan["parameters"],
                approach_penalty_scale=plan["parameters"]["approach_penalty_scale"],
                max_iterations=self.state["target_iteration"] - iteration(plan["checkpoint"]),
                automatic_monitor=str(self.path),
                evaluated_iteration=plan["evaluated_iteration"],
            )
            plan["new_metadata"] = metadata
            self.save(restart=plan, status="verifying_restart")
        metadata = plan["new_metadata"]
        write(LATEST, metadata)
        write(launch / "launch.json", metadata)
        deadline = time.monotonic() + 240
        while not STOP and time.monotonic() < deadline:
            content = (launch / "train.log").read_text(errors="replace")
            run = re.search(r"Logging experiment in directory: (.+)", content)
            updates = [int(x) for x in re.findall(r"Learning iteration\s+(\d+)/", content)]
            if run and updates and max(updates) >= metadata["resume_iteration"] + 2:
                metadata.update(
                    run_directory=run[1].strip(),
                    status="running",
                    last_verified_at=now(),
                    last_verified_iteration=max(updates),
                )
                write(LATEST, metadata)
                write(launch / "launch.json", metadata)
                self.save(
                    parameters=plan["parameters"],
                    restart=None,
                    status="waiting",
                    last_tuned_iteration=plan["evaluated_iteration"],
                )
                self.log(f"Resumed at {metadata['resume_iteration']} with {plan['parameters']}; PID {metadata['pid']}")
                return
            if (
                not process_matches(metadata["pid"], "scripts/np3o/train.py", old["experiment"])
                or "Traceback (most recent call last)" in content
            ):
                break
            time.sleep(5)
        if STOP:
            return
        stop_process(metadata["pid"], "scripts/np3o/train.py", old["experiment"])
        if not plan.get("fallback"):
            # One bounded recovery with the previously working weights and original checkpoint.
            plan["fallback"] = True
            plan["parameters"] = self.state["parameters"]
            fallback = launch / "recovery_tuning.json"
            write(fallback, {"version": 1, "parameters": plan["parameters"]})
            plan["command"][-1] = str(fallback)
            plan.pop("new_metadata", None)
            (launch / "train.log").rename(launch / "failed_startup.log")
            self.save(restart=plan, status="recovering_previous_weights")
            self.finish_restart()
        else:
            self.save(
                restart=None,
                manual_attention=True,
                status="needs_attention",
                error="Both adjusted and recovery training failed; see launch logs.",
            )
            raise RuntimeError(self.state["error"])

    def tick(self):
        if self.state.get("manual_attention"):
            return
        if self.state.get("restart"):
            self.finish_restart()
            return
        self.assert_sources()
        metadata = read(LATEST)
        milestone = self.state["next_iteration"]
        entry = self.state.get("pending_entry")
        if entry is None:
            source = Path(metadata["run_directory"]) / f"model_{milestone}.pt"
            if source not in checkpoints(metadata["run_directory"]):
                status = (
                    "waiting"
                    if process_matches(metadata["pid"], "scripts/np3o/train.py", metadata["experiment"])
                    else "training_stopped"
                )
                self.save(
                    status=status,
                    latest_checkpoint_iteration=max(
                        map(iteration, checkpoints(metadata["run_directory"])), default=None
                    ),
                )
                return
            output = DIRECTORY / f"iteration_{milestone:06d}"
            metrics = self.evaluate(source, output)
            decision, reason = compare_results(metrics, self.state["reference"]["metrics"])
            entry = {
                "iteration": milestone,
                "checkpoint": str(source),
                "checkpoint_sha256": sha(source),
                "metrics": metrics,
                "decision": decision,
                "reason": reason,
                "at": now(),
                "reference_iteration": self.state["reference"]["iteration"],
                "parameters": self.state["parameters"],
                "protocol": EVALUATION_PROTOCOL,
            }
            self.save(pending_entry=entry)
        decision, metrics = entry["decision"], entry["metrics"]
        final = milestone >= self.state["target_iteration"]
        if decision == "stagnant" and not final:
            params, tuning_reason = propose_tuning(metrics, self.state["reference"]["metrics"], entry["parameters"])
            entry.update(proposed_parameters=params, tuning_reason=tuning_reason)
            self.save(pending_entry=entry)
            if params != self.state["parameters"] and self.state.get("last_tuned_iteration") != milestone:
                self.prepare_restart(params, milestone)
                if self.state.get("restart"):
                    self.finish_restart()
                if STOP:
                    return
                entry["applied_parameters"] = self.state["parameters"]
            elif self.state.get("last_tuned_iteration") == milestone:
                entry["applied_parameters"] = self.state["parameters"]
        self.record(entry)
        reference = entry if decision in ("improved", "maintain") else self.state["reference"]
        self.state["history"].append(entry)
        self.save(
            reference=reference,
            next_iteration=min(milestone + INTERVAL, self.state["target_iteration"]),
            status="complete" if final else "waiting",
            error=None,
            pending_entry=None,
        )

    def run(self):
        if self.state["status"] == "stopped_by_user":
            self.log("Monitor remains stopped at the user's request.")
            return
        self.log(
            f"Monitor started; next iteration {self.state['next_iteration']}; target {self.state['target_iteration']}."
        )
        orphan = self.state.get("evaluation_pid")
        if orphan:
            stop_process(orphan, "scripts/np3o/evaluate_platform.py")
            self.save(evaluation_pid=None)
        while not STOP and self.state["status"] != "complete":
            try:
                self.tick()
            except Exception as error:
                self.log(f"ERROR {type(error).__name__}: {error}")
                self.save(status="needs_attention", error=str(error))
            # Failures do not count as a flat result; retries use the same checkpoint.
            delay = 300 if self.state["status"] == "needs_attention" else 30
            for _ in range(delay):
                if STOP or self.state["status"] == "complete":
                    break
                time.sleep(1)


def initialize():
    if (DIRECTORY / "state.json").exists():
        raise RuntimeError("Monitor already initialized; existing history must not be overwritten.")
    bootstrap = read(DIRECTORY / "bootstrap.json")
    reports = [
        read(Path(bootstrap["output"]) / f"h{round(height * 100):03d}/results.json") for height in EVALUATION_HEIGHTS
    ]
    summary = summarize_evaluations(reports)
    checkpoint = Path(bootstrap["checkpoint"])
    if any(r["policies"][0]["checkpoint_sha256"] != sha(checkpoint) for r in reports):
        raise ValueError("Baseline checkpoint mismatch.")
    if any(
        r["evaluator_sha256"] != sha(ROOT / "scripts/np3o/evaluate_platform.py")
        or r["reward_source_sha256"] != sha(ROOT / CRITICAL[3])
        for r in reports
    ):
        raise ValueError("Baseline evaluation code differs from current sources.")
    entry = {
        "iteration": iteration(checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "metrics": summary,
        "decision": "baseline",
        "reason": "上台课程从 30 cm 开始；固定 30/40 cm 新协议基线，20 cm 不参与判断。",
        "protocol": EVALUATION_PROTOCOL,
        "output_directory": bootstrap["output"],
        "at": now(),
    }
    metadata = read(LATEST)
    write(
        DIRECTORY / "state.json",
        {
            "version": 2,
            "protocol": EVALUATION_PROTOCOL,
            "evaluation_heights_m": list(EVALUATION_HEIGHTS),
            "interval": INTERVAL,
            "target_iteration": metadata["target_total_iteration"],
            "next_iteration": (iteration(checkpoint) // INTERVAL + 1) * INTERVAL,
            "status": "waiting",
            "parameters": validate_tuning(metadata.get("platform_tuning", DEFAULTS)),
            "source_hashes": fingerprint(),
            "reference": entry,
            "history": [entry],
            "restart": None,
            "created_at": now(),
        },
    )
    Monitor().record(entry)


def stop_requested(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop_requested)
        signal.signal(signal.SIGINT, stop_requested)
        if args.initialize:
            initialize()
        else:
            Monitor().run()


if __name__ == "__main__":
    main()
