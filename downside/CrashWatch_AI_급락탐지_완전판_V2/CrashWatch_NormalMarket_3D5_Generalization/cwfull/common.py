from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from cw7h.data import discover_dataset, load_references, prepare_data
from cw7h.folds import build_fold_slices
from cw7h.utils import atomic_json, read_json

LOGGER = logging.getLogger(__name__)


def setup_logging(path: Path) -> None:
    import sys
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def set_full_load_mode(total_threads: int, priority: str = "above_normal") -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    logical = psutil.cpu_count(logical=True) or 1
    requested = max(1, min(int(total_threads), logical))
    info: dict[str, Any] = {
        "logical_cpus": logical,
        "physical_cpus": psutil.cpu_count(logical=False),
        "requested_threads": requested,
        "ram_gb": psutil.virtual_memory().total / 1024**3,
    }
    try:
        affinity = list(range(requested))
        process.cpu_affinity(affinity)
        info["affinity"] = process.cpu_affinity()
    except Exception as exc:
        info["affinity_error"] = repr(exc)
    try:
        if os.name == "nt":
            mapping = {
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
            }
            process.nice(mapping.get(str(priority).lower(), psutil.ABOVE_NORMAL_PRIORITY_CLASS))
            info["priority"] = str(priority)
    except Exception as exc:
        info["priority_error"] = repr(exc)
    return info


def set_worker_mode(threads: int, priority: str = "normal") -> None:
    value = str(max(1, int(threads)))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["LIGHTGBM_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        if os.name == "nt":
            mapping = {
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
            }
            psutil.Process(os.getpid()).nice(mapping.get(priority, psutil.NORMAL_PRIORITY_CLASS))
    except Exception:
        pass


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read_json(path, {})
        pid = int(old.get("pid", -1))
        if pid > 0 and psutil.pid_exists(pid):
            raise RuntimeError(f"이미 실행 중입니다. PID={pid}")
    atomic_json({"pid": os.getpid(), "started_epoch": time.time()}, path)


def release_lock(path: Path) -> None:
    try:
        old = read_json(path, {})
        if int(old.get("pid", -1)) == os.getpid():
            path.unlink(missing_ok=True)
    except Exception:
        pass


def resolve_output(project_root: Path, explicit: str | None, default_rel: str) -> Path:
    path = Path(explicit).expanduser() if explicit else project_root / default_rel
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_context(package_root: Path, project_root: Path, dataset_override: str | None, output_dir: Path):
    seed = package_root / "seed_results"
    resolved = read_json(seed / "resolved_run_config.json", {})
    if "config" not in resolved:
        raise FileNotFoundError("seed_results/resolved_run_config.json 누락")
    legacy = resolved["config"]
    legacy_summary = read_json(seed / "run_summary.json", {})
    refs = load_references(package_root / "reference", strict_counts=True)
    dataset = discover_dataset(
        project_root,
        str(legacy.get("dataset_path", "AUTO")),
        dataset_override,
        list(legacy.get("sealed_path_tokens", ["sealed", "holdout", "untouched"])),
    )
    cache_dir = Path(legacy.get("cache_dir", "crashwatch_ai_data/shared_cache/all_feature_ablation_7h_v1"))
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    prepared = prepare_data(dataset, cache_dir, legacy, refs, force=False)
    dates = np.load(prepared.dates_path, mmap_mode="r")
    folds = build_fold_slices(
        dates,
        refs.folds,
        int(legacy["min_train_days"]),
        int(legacy["validation_days"]),
        int(legacy["purge_days"]),
        output_dir / "fold_manifest.json",
    )
    if len(folds) != 8 or any(not fold.eligible for fold in folds):
        reasons = [fold.to_dict() for fold in folds if not fold.eligible]
        raise RuntimeError(f"기존 8개 outer fold 재구성 실패: {reasons}")
    old_signature = str(legacy_summary.get("dataset_signature", ""))
    manifest = prepared.manifest
    # Accept the original development set *or a chronological extension* of it.
    # This experiment intentionally supports adding newer development dates for
    # recent-drift diagnostics, but never accepts a truncated/older history.
    current_max = str(manifest.get("date_max", ""))[:10]
    compatible = (
        int(manifest.get("rows", -1)) >= 91919
        and int(manifest.get("features", -1)) == 439
        and str(manifest.get("date_min", ""))[:10] == "2018-01-02"
        and current_max >= "2026-06-22"
    )
    if prepared.signature != old_signature and not compatible:
        raise RuntimeError(
            "개발 데이터가 기존 실험과 일치하지 않습니다. "
            f"current={prepared.signature}, expected={old_signature}, manifest={manifest}"
        )
    best = {int(k): int(v) for k, v in read_json(seed / "effective_best_iterations.json", {}).items()}
    if set(best) != set(range(8)):
        raise RuntimeError(f"fold별 best iteration 누락: {best}")
    audit = {
        "dataset_path": str(dataset),
        "cache_root": str(prepared.root),
        "current_signature": prepared.signature,
        "legacy_signature": old_signature,
        "exact_or_compatible": prepared.signature == old_signature or compatible,
        "development_extension_accepted": bool(compatible and prepared.signature != old_signature),
        "manifest": manifest,
        "feature_count": len(prepared.feature_names),
    }
    atomic_json(audit, output_dir / "dataset_compatibility.json")
    return prepared, refs, folds, legacy, best


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def nvml_snapshot() -> dict[str, float | None]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return {
            "gpu_util_percent": float(util.gpu),
            "gpu_memory_util_percent": float(util.memory),
            "gpu_memory_used_gb": float(memory.used / 1024**3),
            "gpu_memory_total_gb": float(memory.total / 1024**3),
            "gpu_temperature_c": float(temp),
            "gpu_power_w": float(power),
        }
    except Exception:
        return {
            "gpu_util_percent": None,
            "gpu_memory_util_percent": None,
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
        }
