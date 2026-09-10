from __future__ import annotations

import ast
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from cw7h.data import References, PreparedData, discover_dataset, load_references, prepare_data
from cw7h.folds import FoldSlice, build_fold_slices
from cw7h.utils import atomic_json, canonical_hash, read_json

LOGGER = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    import sys
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)


def set_process_mode(mode: str, total_threads: int) -> dict[str, Any]:
    proc = psutil.Process(os.getpid())
    logical = psutil.cpu_count(logical=True) or 1
    total_threads = max(1, min(int(total_threads), logical))
    info: dict[str, Any] = {"mode": mode, "logical_cpus": logical, "requested_threads": total_threads}
    try:
        if mode == "game":
            # Keep the first logical CPUs free for the foreground game/Windows scheduler.
            affinity = list(range(max(0, logical - total_threads), logical))
        else:
            affinity = list(range(logical))
        proc.cpu_affinity(affinity)
        info["affinity"] = proc.cpu_affinity()
    except Exception as exc:
        info["affinity_error"] = repr(exc)
    try:
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if mode == "game" else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            info["priority"] = "below_normal" if mode == "game" else "above_normal"
        else:
            info["priority"] = "unchanged_non_windows"
    except Exception as exc:
        info["priority_error"] = repr(exc)
    return info


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            old = read_json(lock_path, {})
            pid = int(old.get("pid", -1))
            if pid > 0 and psutil.pid_exists(pid):
                raise RuntimeError(f"다른 실행이 이미 진행 중입니다. PID={pid}, lock={lock_path}")
        except RuntimeError:
            raise
        except Exception:
            pass
    atomic_json({"pid": os.getpid(), "started_epoch": time.time()}, lock_path)


def release_lock(lock_path: Path) -> None:
    try:
        current = read_json(lock_path, {})
        if int(current.get("pid", -1)) == os.getpid():
            lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def locate_previous_output(project_root: Path, explicit: str | None, default_rel: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = project_root / p
        return p.resolve()
    direct = (project_root / default_rel).resolve()
    if direct.exists():
        return direct
    candidates: list[Path] = []
    for root in [project_root, project_root.parent, Path.home() / "Downloads"]:
        if not root.exists():
            continue
        try:
            for p in root.rglob("all_feature_ablation_6h_full_5080_v1"):
                if p.is_dir() and (p / "run_summary.json").exists():
                    candidates.append(p.resolve())
        except (PermissionError, OSError):
            pass
    return sorted(set(candidates), key=lambda p: len(str(p)))[0] if candidates else direct


def _copy_seed_tree(seed_dir: Path, output_dir: Path) -> None:
    for src in seed_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(seed_dir)
        dst = output_dir / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
    return value


def _parse_identity(value: Any) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def bootstrap_checkpoints(seed_dir: Path, output_dir: Path) -> dict[str, int]:
    """Reconstruct completed task JSON files from the uploaded aggregate CSV.

    This makes the continuation package work even when only the compact result archive ZIP is
    available. Existing original checkpoints always take precedence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_seed_tree(seed_dir, output_dir)
    source = output_dir / "all_model_metrics.csv"
    if not source.exists():
        source = seed_dir / "all_model_metrics.csv"
    frame = pd.read_csv(source, low_memory=False)
    counts = {"lightgbm_cpu": 0, "xgboost_cuda": 0, "existing": 0, "failed_parse": 0}
    for row in frame.to_dict(orient="records"):
        backend = str(row.get("backend", ""))
        if backend not in {"lightgbm_cpu", "xgboost_cuda"}:
            continue
        fold = int(row.get("outer_fold", -1))
        task_id = row.get("task_id")
        if backend == "xgboost_cuda" and str(row.get("test_type")) == "baseline_all_valid":
            run_sig = str(row.get("run_signature") or row.get("_run_signature") or "")
            path = output_dir / "task_results" / backend / f"baseline_{run_sig}_fold_{fold}.json"
        else:
            if task_id is None or (isinstance(task_id, float) and not math.isfinite(task_id)):
                counts["failed_parse"] += 1
                continue
            path = output_dir / "task_results" / backend / f"{str(task_id)}.json"
        if path.exists():
            counts["existing"] += 1
            continue
        result: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"result_path", "_dataset_signature", "_run_signature"}:
                continue
            cleaned = _clean_value(value)
            if cleaned is not None:
                result[key] = cleaned
        identity = _parse_identity(row.get("identity"))
        if identity is not None:
            result["identity"] = identity
        result["status"] = "completed"
        result["prediction_path"] = ""
        atomic_json(result, path)
        counts[backend] += 1
    atomic_json(counts, output_dir / "bootstrap_checkpoint_summary.json")
    return counts


def load_legacy_context(seed_dir: Path, previous_output: Path) -> tuple[dict[str, Any], str, str, dict[int, int]]:
    resolved = read_json(previous_output / "resolved_run_config.json") or read_json(seed_dir / "resolved_run_config.json")
    if not resolved or "config" not in resolved:
        raise FileNotFoundError("resolved_run_config.json을 찾지 못했습니다.")
    run_summary = read_json(previous_output / "run_summary.json") or read_json(seed_dir / "run_summary.json")
    run_sig_obj = read_json(previous_output / "run_signature.json") or read_json(seed_dir / "run_signature.json")
    dataset_signature = str(run_summary["dataset_signature"])
    run_signature = str(run_sig_obj.get("run_signature", run_summary["run_signature"]))
    best_obj = read_json(previous_output / "effective_best_iterations.json") or read_json(seed_dir / "effective_best_iterations.json")
    best = {int(k): int(v) for k, v in best_obj.items()}
    return resolved["config"], dataset_signature, run_signature, best


def prepare_context(
    package_root: Path,
    project_root: Path,
    previous_output: Path,
    dataset_override: str | None,
) -> tuple[PreparedData, References, list[FoldSlice], dict[str, Any], str, str, dict[int, int]]:
    seed_dir = package_root / "seed_results"
    legacy_config, legacy_dataset_signature, run_signature, best_iterations = load_legacy_context(seed_dir, previous_output)
    refs = load_references(package_root / "reference", strict_counts=True)
    dataset = discover_dataset(
        project_root,
        str(legacy_config.get("dataset_path", "AUTO")),
        dataset_override,
        list(legacy_config.get("sealed_path_tokens", ["sealed", "untouched", "holdout"])),
    )
    cache_dir = Path(legacy_config["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    prepared = prepare_data(dataset, cache_dir, legacy_config, refs, force=False)
    dates = np.load(prepared.dates_path, mmap_mode="r")
    folds = build_fold_slices(
        dates,
        refs.folds,
        min_train_days=int(legacy_config["min_train_days"]),
        validation_days=int(legacy_config["validation_days"]),
        purge_days=int(legacy_config["purge_days"]),
        output_path=previous_output / "continuation_fold_manifest.json",
    )
    if any(not f.eligible for f in folds):
        raise RuntimeError("기존 8개 fold를 현재 데이터에서 재구성하지 못했습니다.")
    seed_summary = read_json(seed_dir / "run_summary.json", {})
    expected_rows = 91919
    expected_features = 439
    manifest = prepared.manifest
    compatible = (
        int(manifest.get("rows", -1)) == expected_rows
        and int(manifest.get("features", -1)) == expected_features
        and str(manifest.get("date_min", ""))[:10] == "2018-01-02"
        and str(manifest.get("date_max", ""))[:10] == "2026-06-22"
    )
    if prepared.signature != legacy_dataset_signature and not compatible:
        raise RuntimeError(
            "현재 데이터가 이전 실험 데이터와 호환되지 않습니다. "
            f"current={prepared.signature}, previous={legacy_dataset_signature}, manifest={manifest}"
        )
    atomic_json({
        "current_cache_signature": prepared.signature,
        "legacy_task_signature": legacy_dataset_signature,
        "compatible": compatible,
        "dataset_path": str(dataset),
        "cache_root": str(prepared.root),
        "seed_run_status": seed_summary.get("status"),
    }, previous_output / "continuation_dataset_compatibility.json")
    return prepared, refs, folds, legacy_config, legacy_dataset_signature, run_signature, best_iterations


def write_eta(output_dir: Path, stage: str, completed: int, total: int, elapsed: float, threshold_hours: float = 4.0) -> float:
    rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
    remaining = max(0, total - completed)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    payload = {
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds if math.isfinite(eta_seconds) else None,
        "eta_hours": eta_seconds / 3600 if math.isfinite(eta_seconds) else None,
        "full_load_recommended": math.isfinite(eta_seconds) and eta_seconds > threshold_hours * 3600,
        "updated_epoch": time.time(),
    }
    atomic_json(payload, output_dir / "live_eta.json")
    recommendation = output_dir / "FULL_LOAD_RECOMMENDED.txt"
    if payload["full_load_recommended"]:
        recommendation.write_text(
            "남은 예상시간이 4시간을 넘었습니다. 현재 실행을 Ctrl+C로 안전 종료한 뒤 "
            "RUN_FULL_LOAD_7950X3D.bat를 실행하면 같은 체크포인트에서 이어집니다.\n",
            encoding="utf-8",
        )
    return eta_seconds
