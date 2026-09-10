#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil


PROJECT = Path(__file__).resolve().parent


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _resolve(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _apply_self_limits(config: dict[str, Any]) -> dict[str, Any]:
    cpu = config["cpu"]
    process = psutil.Process()
    available = set(process.cpu_affinity())
    requested = [int(x) for x in cpu["affinity"]]
    selected = [x for x in requested if x in available]
    if len(selected) != int(cpu["threads"]):
        raise RuntimeError(
            f"요청 CPU affinity를 적용할 수 없습니다: requested={requested}, "
            f"available={sorted(available)}"
        )
    process.cpu_affinity(selected)
    priority = str(cpu.get("priority", "idle")).lower()
    if os.name == "nt":
        priority_map = {
            "idle": psutil.IDLE_PRIORITY_CLASS,
            "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "high": psutil.HIGH_PRIORITY_CLASS,
        }
        process.nice(priority_map.get(priority, psutil.NORMAL_PRIORITY_CLASS))
    else:
        process.nice(10 if priority == "idle" else 5 if priority == "below_normal" else 0)
    return {
        "pid": process.pid,
        "affinity": process.cpu_affinity(),
        "threads": int(cpu["threads"]),
        "physical_cores": int(cpu["physical_cores"]),
        "priority": priority,
    }


def _gpu_sealed(config: dict[str, Any]) -> bool:
    return bool(config.get("execution", {}).get("gpu_sealed", True))


def _sealed_environment(config: dict[str, Any], data_root: Path) -> dict[str, str]:
    threads = str(int(config["cpu"]["threads"]))
    model = config["model"]
    env = os.environ.copy()
    env.update(
        {
            "CRASHWATCH_DATA_DIR": str(data_root),
            "OMP_NUM_THREADS": threads,
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": threads,
            "MKL_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "LOKY_MAX_CPU_COUNT": threads,
            "CRASHWATCH_XGB_N_JOBS": threads,
            "CRASHWATCH_XGB_N_ESTIMATORS": str(model["n_estimators"]),
            "CRASHWATCH_XGB_MAX_DEPTH": str(model["max_depth"]),
            "CRASHWATCH_XGB_LEARNING_RATE": str(model["learning_rate"]),
            "CRASHWATCH_XGB_MAX_BIN": str(model["max_bin"]),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if _gpu_sealed(config):
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "CRASHWATCH_GPU_DEVICE": "cpu",
            }
        )
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
        env["CRASHWATCH_GPU_DEVICE"] = "cuda"
    return env


def _worker_command(
    project: Path,
    dataset: Path,
    job: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    exp = config["experiment"]
    seeds = job.get("seeds", exp["seeds"])
    command = [
        sys.executable,
        str(project / "03F_장기이탈테스트_워커.py"),
        "--dataset",
        str(dataset),
        "--run-tag",
        str(job["run_tag"]),
        "--modes",
        ",".join(job["modes"]),
        "--groups",
        ",".join(job["groups"]),
        "--seeds",
        ",".join(map(str, seeds)),
        "--folds",
        str(job.get("folds", exp["folds"])),
        "--validation-days",
        str(job.get("validation_days", exp["validation_days"])),
        "--purge-days",
        str(job.get("purge_days", exp["purge_days"])),
        "--min-train-days",
        str(job.get("min_train_days", exp["min_train_days"])),
        "--max-train-rows",
        str(job.get("max_train_rows", exp["max_train_rows"])),
        "--calibration",
        str(job.get("calibration", exp["calibration"])),
        "--calibration-days",
        str(job.get("calibration_days", exp["calibration_days"])),
    ]
    if _gpu_sealed(config):
        command.append("--cpu")
    return command


def _terminate_tree(pid: int, grace_seconds: int = 15) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    processes = [*root.children(recursive=True), root]
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def _finalize_partial_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "all_metrics_by_fold_seed_scope.csv"
    if not metrics_path.exists():
        return {"run_tag": run_dir.name, "status": "no_metrics"}
    import pandas as pd

    from dual_ablation.experiment.statistics import paired_delta_table, summarize_deltas
    from dual_ablation.io_utils import atomic_csv

    metrics = pd.read_csv(metrics_path)
    if metrics.empty or metrics["experiment"].nunique() < 2:
        return {
            "run_tag": run_dir.name,
            "status": "baseline_only",
            "metric_rows": int(len(metrics)),
        }
    deltas = paired_delta_table(metrics)
    summary = summarize_deltas(deltas)
    atomic_csv(deltas, run_dir / "paired_ablation_deltas.csv")
    atomic_csv(summary, run_dir / "ablation_statistical_summary.csv")

    bucket_source = summary.loc[
        summary["scope_type"].isin(["target_bucket", "bucket"])
        & summary["target_group"].notna()
    ]
    if not bucket_source.empty:
        matrix = bucket_source.pivot_table(
            index="scope_value",
            columns="target_group",
            values="pr_auc_loss_when_removed_mean",
            aggfunc="mean",
        )
        atomic_csv(matrix.reset_index(), run_dir / "bucket_group_sensitivity_matrix.csv")
    ticker_source = summary.loc[
        summary["scope_type"].isin(["target_ticker", "ticker"])
        & summary["target_group"].notna()
    ]
    if not ticker_source.empty:
        matrix = ticker_source.pivot_table(
            index="scope_value",
            columns="target_group",
            values="pr_auc_loss_when_removed_mean",
            aggfunc="mean",
        )
        atomic_csv(matrix.reset_index(), run_dir / "ticker_group_sensitivity_matrix.csv")
    return {
        "run_tag": run_dir.name,
        "status": "summarized",
        "metric_rows": int(len(metrics)),
        "experiments": int(metrics["experiment"].nunique()),
        "paired_rows": int(len(deltas)),
        "summary_rows": int(len(summary)),
    }


def _create_compact_zip(
    project: Path,
    root: Path,
    state_path: Path,
    config_path: Path,
    config: dict[str, Any],
) -> Path:
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = project
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = desktop / f"CrashWatch_V4_{config['profile']}_RESULTS_{stamp}.zip"
    allowed = {
        "ablation_statistical_summary.csv",
        "bucket_group_sensitivity_matrix.csv",
        "ticker_group_sensitivity_matrix.csv",
        "run_summary.json",
        "walk_forward_folds.json",
        "skipped_experiments.csv",
        "skipped_groups.csv",
        "valid_feature_audit.csv",
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        if state_path.exists():
            archive.write(state_path, f"state/{state_path.name}")
        archive.write(config_path, f"config/{config_path.name}")
        for job in config["jobs"]:
            run_dir = root / "runs" / str(job["run_tag"])
            if not run_dir.exists():
                continue
            for path in run_dir.iterdir():
                if path.is_file() and path.name in allowed and path.stat().st_size <= 12 * 1024 * 1024:
                    archive.write(path, f"runs/{run_dir.name}/{path.name}")
        log_dir = root / str(config.get("log_dir", "cpu_scout_3h_logs"))
        if log_dir.exists():
            for path in sorted(log_dir.glob("*.log")):
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-250:]
                archive.writestr(f"log_tails/{path.stem}.txt", "\n".join(lines))
    return output


def _initial_state(
    config: dict[str, Any],
    config_path: Path,
    dataset: Path,
    limits: dict[str, Any],
) -> dict[str, Any]:
    start = datetime.now().astimezone()
    sealed = _gpu_sealed(config)
    return {
        "profile": config["profile"],
        "status": "running",
        "started_at": start.isoformat(),
        "expected_end_at": (start + timedelta(hours=float(config["runtime_hours"]))).isoformat(),
        "config": str(config_path),
        "dataset": str(dataset),
        "gpu_sealed": sealed,
        "gpu_environment": (
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "worker_flag": "--cpu",
            }
            if sealed
            else {
                "CUDA_VISIBLE_DEVICES": "unrestricted",
                "NVIDIA_VISIBLE_DEVICES": "unrestricted",
                "worker_flag": "none",
            }
        ),
        "cpu_limits": limits,
        "jobs": [
            {
                "job_id": job["job_id"],
                "run_tag": job["run_tag"],
                "status": "pending",
                "attempts": 0,
                "return_code": None,
                "started_at": None,
                "ended_at": None,
            }
            for job in config["jobs"]
        ],
        "partial_summaries": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V4 3시간 CPU 전용 광역 얕은 이탈테스트")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "cpu_scout_3h.json",
    )
    parser.add_argument("--reset", action="store_true", help="scout 상태만 초기화; prediction cache는 보존")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(PROJECT, config["data_root"])
    dataset = _resolve(PROJECT, config["dataset"])
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    root = data_root / "ablation_longrun"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / config["state_file"]
    stop_path = root / config["stop_file"]
    limits = _apply_self_limits(config)
    environment = _sealed_environment(config, data_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(config_path),
                    "dataset": str(dataset),
                    "runtime_hours": config["runtime_hours"],
                    "limits": limits,
                    "gpu_sealed": _gpu_sealed(config),
                    "gpu_environment": {
                        key: environment.get(key, "unrestricted")
                        for key in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CRASHWATCH_GPU_DEVICE"]
                    },
                    "commands": [
                        _worker_command(PROJECT, dataset, job, config) for job in config["jobs"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.reset:
        if stop_path.exists():
            stop_path.unlink()
        state = _initial_state(config, config_path, dataset, limits)
        _atomic_json(state_path, state)
    elif state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") in {"completed", "budget_stopped", "stopped"}:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return
    else:
        state = _initial_state(config, config_path, dataset, limits)
        _atomic_json(state_path, state)

    started = datetime.fromisoformat(state["started_at"])
    deadline_epoch = started.timestamp() + float(config["runtime_hours"]) * 3600
    logs = root / str(config.get("log_dir", "cpu_scout_3h_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    job_by_id = {job["job_id"]: job for job in config["jobs"]}

    for job_state in state["jobs"]:
        if time.time() >= deadline_epoch:
            break
        if stop_path.exists():
            state["status"] = "stopped"
            break
        if job_state["status"] == "success":
            continue
        job = job_by_id[job_state["job_id"]]
        job_state["status"] = "running"
        job_state["attempts"] += 1
        job_state["started_at"] = _now_iso()
        _atomic_json(state_path, state)

        command = _worker_command(PROJECT, dataset, job, config)
        log_path = logs / f"{job['job_id']}.log"
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        with log_path.open("a", encoding="utf-8", buffering=1) as output:
            output.write("\n=== SCOUT COMMAND ===\n")
            output.write(subprocess.list2cmdline(command) + "\n")
            output.write(f"GPU_SEALED={_gpu_sealed(config)}\n")
            output.write("CPU_LIMITS=" + json.dumps(limits, ensure_ascii=False) + "\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            child = psutil.Process(process.pid)
            child.cpu_affinity(limits["affinity"])
            if os.name == "nt":
                child.nice(psutil.Process().nice())
            while process.poll() is None:
                if stop_path.exists() or time.time() >= deadline_epoch:
                    _terminate_tree(process.pid)
                    break
                time.sleep(5)
            code = process.poll()
            if code is None:
                code = -9
        job_state["return_code"] = int(code)
        job_state["ended_at"] = _now_iso()
        if time.time() >= deadline_epoch:
            job_state["status"] = "budget_stopped"
        elif stop_path.exists():
            job_state["status"] = "stopped"
        elif code == 0:
            job_state["status"] = "success"
        else:
            job_state["status"] = "failed"
        _atomic_json(state_path, state)

    summaries = []
    for job in config["jobs"]:
        run_dir = root / "runs" / job["run_tag"]
        if run_dir.exists():
            try:
                summaries.append(_finalize_partial_run(run_dir))
            except Exception as exc:  # noqa: BLE001
                summaries.append(
                    {"run_tag": job["run_tag"], "status": "summary_failed", "error": str(exc)}
                )
    state["partial_summaries"] = summaries
    if stop_path.exists():
        state["status"] = "stopped"
    elif time.time() >= deadline_epoch:
        state["status"] = "budget_stopped"
    elif all(job["status"] == "success" for job in state["jobs"]):
        state["status"] = "completed"
    else:
        state["status"] = "finished_with_failures"
    state["finished_at"] = _now_iso()
    state["compact_zip"] = str(_create_compact_zip(PROJECT, root, state_path, config_path, config))
    _atomic_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
