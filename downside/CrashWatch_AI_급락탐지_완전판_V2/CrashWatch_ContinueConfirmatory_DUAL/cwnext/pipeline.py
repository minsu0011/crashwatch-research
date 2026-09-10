from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cw7h.aggregate import aggregate_results
from cw7h.utils import atomic_json

from .common import (
    acquire_lock,
    bootstrap_checkpoints,
    locate_previous_output,
    prepare_context,
    release_lock,
    set_process_mode,
)
from .completion import run_cpu_completion
from .confirmatory import run_profile_benchmark, run_targeted_confirmatory
from .gpu_resume import run_gpu_resume
from .xgb_confirmatory import run_xgb_profile_confirmatory

LOGGER = logging.getLogger(__name__)


def _gpu_entry(q: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        q.put({"ok": True, "result": run_gpu_resume(**kwargs)})
    except Exception as exc:
        import traceback
        q.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


def _xgb_confirm_entry(q: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        q.put({"ok": True, "result": run_xgb_profile_confirmatory(**kwargs)})
    except Exception as exc:
        import traceback
        q.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


def _initial_eta(previous_output: Path, mode_cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = pd.read_csv(previous_output / "all_model_metrics.csv", low_memory=False)
    lgb = metrics[metrics.backend.eq("lightgbm_cpu")]
    xgb = metrics[metrics.backend.eq("xgboost_cuda")]
    lgb_seconds = float(lgb.elapsed_seconds.dropna().median()) if not lgb.empty else 10.0
    xgb_seconds = float(xgb[xgb.test_type.eq("single_feature_loo")].elapsed_seconds.dropna().median()) if not xgb.empty else 12.0
    missing_conditional = 63
    missing_cluster = 2496
    profile_tasks = 8 * 8 * 5
    targeted_tasks_est = 43 * 8 * 5
    cpu_tasks = missing_conditional + missing_cluster + profile_tasks + targeted_tasks_est
    cpu_parallel = max(1, int(mode_cfg["cpu"]["workers"]))
    cpu_hours = cpu_tasks * lgb_seconds / cpu_parallel / 3600
    missing_gpu = 1712
    gpu_workers = max(1, int(mode_cfg["gpu"]["workers"]))
    if mode_cfg["mode"] == "game":
        prior_util = 32.7
        target = float(mode_cfg["gpu"].get("target_average_gpu_percent", 10.0))
        throttle_factor = max(1.0, prior_util / max(1.0, target))
    else:
        throttle_factor = 1.0
    gpu_hours = missing_gpu * xgb_seconds * throttle_factor / gpu_workers / 3600
    total_hours = max(cpu_hours, gpu_hours) + (0.25 if mode_cfg["mode"] == "full" else 0.5)
    payload = {
        "lgb_median_model_seconds": lgb_seconds,
        "xgb_median_model_seconds": xgb_seconds,
        "estimated_cpu_hours": cpu_hours,
        "estimated_gpu_hours": gpu_hours,
        "estimated_total_hours": total_hours,
        "over_four_hours": total_hours > 4.0,
        "note": "실제 속도는 첫 체크포인트 이후 live_eta.json으로 갱신됩니다.",
    }
    atomic_json(payload, previous_output / f"initial_eta_{mode_cfg['mode']}.json")
    if payload["over_four_hours"]:
        (previous_output / "FULL_LOAD_RECOMMENDED.txt").write_text(
            "예상 소요시간이 4시간을 넘습니다. 게임 모드는 안전하게 계속 실행할 수 있지만, "
            "Ctrl+C로 종료 후 RUN_FULL_LOAD_7950X3D.bat를 실행하면 체크포인트에서 그대로 이어집니다.\n",
            encoding="utf-8",
        )
    return payload


def _write_human_summary(previous_output: Path, final: dict[str, Any]) -> None:
    out = previous_output / "confirmatory_v2"
    lines = [
        "CrashWatch 누락 실험 + 확증 실험 완료 요약",
        "=" * 54,
        f"실행 모드: {final['mode']}",
        f"총 경과시간: {final['elapsed_seconds']/3600:.2f}시간",
        f"CPU 누락 실험 상태: {final.get('cpu_completion',{}).get('aggregation',{}).get('status','unknown')}",
        f"GPU 누락 실험 상태: {final.get('gpu_completion',{}).get('status','unknown')}",
        f"확증 프로필 승자: {final.get('profile_winner','unknown')}",
        "",
        "주요 결과 파일:",
        str(out / "profile_summary.csv"),
        str(out / "profile_paired_deltas.csv"),
        str(out / "targeted_summary.csv"),
        str(previous_output / "cluster_ablation_summary.csv"),
        str(previous_output / "feature_master_decision.csv"),
    ]
    (out / "FINAL_RESULT_GUIDE_KO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    package_root: Path,
    project_root: Path,
    mode_cfg: dict[str, Any],
    *,
    dataset_override: str | None,
    previous_output_override: str | None,
) -> dict[str, Any]:
    started = time.time()
    seed_dir = package_root / "seed_results"
    previous_output = locate_previous_output(
        project_root,
        previous_output_override,
        str(mode_cfg.get("previous_output", "crashwatch_ai_data/all_feature_ablation_6h_full_5080_v1")),
    )
    previous_output.mkdir(parents=True, exist_ok=True)
    lock_path = previous_output / "continuation_pipeline.lock.json"
    acquire_lock(lock_path)
    gpu_process: mp.Process | None = None
    xgb_process: mp.Process | None = None
    try:
        bootstrap = bootstrap_checkpoints(seed_dir, previous_output)
        prepared, refs, folds, legacy_config, legacy_dataset_signature, run_signature, best_iterations = prepare_context(
            package_root, project_root, previous_output, dataset_override,
        )
        hardware = set_process_mode(mode_cfg["mode"], int(mode_cfg["cpu"]["total_threads"]))
        atomic_json(hardware, previous_output / f"hardware_mode_{mode_cfg['mode']}.json")
        # ProcessPool workers spawned on Windows do not reliably retain the
        # parent's priority class. Pass the intended game-mode limits through
        # the environment so every LightGBM/confirmatory child reapplies them.
        os.environ["CRASHWATCH_PROCESS_MODE"] = str(mode_cfg["mode"])
        os.environ["CRASHWATCH_CPU_AFFINITY"] = ",".join(str(x) for x in hardware.get("affinity", []))
        logical = int(hardware.get("logical_cpus", mode_cfg["cpu"]["total_threads"]))
        threads_per_worker = int(mode_cfg["cpu"]["threads_per_worker"])
        effective_cpu_workers = max(1, min(int(mode_cfg["cpu"]["workers"]), logical // max(1, threads_per_worker)))
        effective_total_threads = min(int(mode_cfg["cpu"]["total_threads"]), logical)
        effective_cfg = json.loads(json.dumps(mode_cfg))
        effective_cfg["cpu"]["workers"] = effective_cpu_workers
        effective_cfg["cpu"]["total_threads"] = effective_total_threads
        initial_eta = _initial_eta(previous_output, effective_cfg)

        gpu_kwargs = {
            "prepared": prepared,
            "folds": folds,
            "output_dir": previous_output,
            "legacy_dataset_signature": legacy_dataset_signature,
            "run_signature": run_signature,
            "gpu_config": legacy_config["gpu"] | mode_cfg["gpu"].get("overrides", {}),
            "mode": mode_cfg["mode"],
            "workers": int(mode_cfg["gpu"]["workers"]),
            "total_cpu_threads": effective_total_threads,
        }
        ctx = mp.get_context("spawn")
        gpu_queue: mp.Queue = ctx.Queue()
        gpu_process = ctx.Process(target=_gpu_entry, args=(gpu_queue, gpu_kwargs), name="cw-gpu-continuation", daemon=False)
        gpu_process.start()
        LOGGER.info("GPU continuation started PID=%s", gpu_process.pid)

        cpu_completion = run_cpu_completion(
            prepared, folds, previous_output, legacy_config, legacy_dataset_signature, run_signature, best_iterations,
            workers=effective_cpu_workers,
            threads_per_worker=threads_per_worker,
            block_size=int(mode_cfg["cpu"]["block_size"]),
        )
        gpu_process.join()
        try:
            gpu_message = gpu_queue.get(timeout=10)
        except queue.Empty:
            gpu_message = {"ok": False, "error": f"GPU process exited with code {gpu_process.exitcode} without result"}
        gpu_completion = gpu_message.get("result", gpu_message)
        aggregate_results(previous_output, len(prepared.feature_names), len(folds), legacy_dataset_signature, run_signature)

        confirm_cfg = dict(mode_cfg["confirmatory"])
        model_config = dict(legacy_config["lightgbm"])
        model_config.update(confirm_cfg.get("lightgbm_overrides", {}))
        seeds = [int(x) for x in confirm_cfg.get("seeds", [17, 43, 101, 211, 503])]
        profiles, winner, profile_result = run_profile_benchmark(
            prepared, folds, previous_output, seed_dir, legacy_dataset_signature, best_iterations, model_config,
            workers=effective_cpu_workers,
            threads=threads_per_worker,
            seeds=seeds,
        )

        xgb_queue: mp.Queue | None = None
        if bool(mode_cfg.get("xgb_confirmatory", {}).get("enabled", False)):
            xgb_queue = ctx.Queue()
            xgb_kwargs = {
                "prepared": prepared, "folds": folds, "previous_output": previous_output,
                "profiles": profiles,
                "config": legacy_config["gpu"] | mode_cfg["xgb_confirmatory"].get("overrides", {}),
                "workers": int(mode_cfg["xgb_confirmatory"].get("workers", 2)),
                "seeds": [int(x) for x in mode_cfg["xgb_confirmatory"].get("seeds", [17, 101, 503])],
                "profile_names": list(mode_cfg["xgb_confirmatory"].get("profiles", ["P0_FULL_439", "P1_EXACT_DEDUP", "P2_DEDUP_CLEAN", "P4_CORR095", "P7_CORR095_PLUS_CONDITIONAL"])),
            }
            xgb_process = ctx.Process(target=_xgb_confirm_entry, args=(xgb_queue, xgb_kwargs), name="cw-xgb-confirmatory", daemon=False)
            xgb_process.start()

        targeted_result = run_targeted_confirmatory(
            prepared, folds, previous_output, profiles, winner, legacy_dataset_signature, best_iterations, model_config,
            workers=effective_cpu_workers,
            threads=threads_per_worker,
            seeds=seeds,
        )
        xgb_result: dict[str, Any] = {"status": "sealed"}
        if xgb_process is not None and xgb_queue is not None:
            xgb_process.join()
            try:
                msg = xgb_queue.get(timeout=10)
            except queue.Empty:
                msg = {"ok": False, "error": f"XGB confirmatory exited code={xgb_process.exitcode}"}
            xgb_result = msg.get("result", msg)

        final = {
            "status": "completed",
            "mode": mode_cfg["mode"],
            "started_epoch": started,
            "elapsed_seconds": time.time() - started,
            "previous_output": str(previous_output),
            "dataset_cache": str(prepared.root),
            "bootstrap": bootstrap,
            "initial_eta": initial_eta,
            "cpu_completion": cpu_completion,
            "gpu_completion": gpu_completion,
            "profile_winner": winner,
            "profile_result": profile_result,
            "targeted_result": targeted_result,
            "xgb_confirmatory": xgb_result,
        }
        atomic_json(final, previous_output / "confirmatory_v2" / f"final_pipeline_{mode_cfg['mode']}.json")
        _write_human_summary(previous_output, final)
        return final
    finally:
        for proc in [gpu_process, xgb_process]:
            if proc is not None and proc.is_alive():
                proc.terminate(); proc.join(timeout=10)
        release_lock(lock_path)
