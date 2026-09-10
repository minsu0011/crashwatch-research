from __future__ import annotations

import concurrent.futures as cf
import gc
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, canonical_hash, hash_strings, read_json
from .common import set_worker_mode

ENGINE_VERSION = "cw_final_selection_lgb_v1"


def _params(config: dict[str, Any], y_train: np.ndarray, threads: int, seed: int) -> dict[str, Any]:
    positives = max(1, int(np.sum(y_train == 1)))
    negatives = max(1, int(np.sum(y_train == 0)))
    return {
        "objective": "binary",
        "metric": "None",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "feature_fraction": float(config["feature_fraction"]),
        "feature_fraction_bynode": 1.0,
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "lambda_l1": float(config["lambda_l1"]),
        "lambda_l2": float(config["lambda_l2"]),
        "max_bin": int(config["max_bin"]),
        "min_gain_to_split": float(config.get("min_gain_to_split", 1e-12)),
        "scale_pos_weight": float(negatives / positives),
        "num_threads": int(threads),
        "deterministic": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "drop_seed": int(seed),
    }


def _wait_for_memory(min_free_ram_gb: float) -> float:
    waited = 0.0
    while psutil.virtual_memory().available / 1024**3 < min_free_ram_gb:
        time.sleep(2.0)
        waited += 2.0
    return waited


def _bundle_worker(plan: dict[str, Any]) -> dict[str, Any]:
    set_worker_mode(int(plan["threads"]), str(plan.get("priority", "normal")))
    import lightgbm as lgb

    cache_root = Path(plan["cache_root"])
    X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
    y = np.load(cache_root / "target.npy", mmap_mode="r")
    dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
    feature_index = {feature: index for index, feature in enumerate(plan["feature_names"])}
    output_dir = Path(plan["task_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fold = plan["fold"]
    train_slice = slice(int(fold["train_start"]), int(fold["train_stop"]))
    valid_slice = slice(int(fold["validation_start"]), int(fold["validation_stop"]))
    features = list(plan["features"])
    indices = np.asarray([feature_index[feature] for feature in features], dtype=np.int32)
    profile_hash = hash_strings(features)
    memory_wait = _wait_for_memory(float(plan["min_free_ram_gb"]))

    x_train = np.ascontiguousarray(X[train_slice][:, indices], dtype=np.float32)
    x_valid = np.ascontiguousarray(X[valid_slice][:, indices], dtype=np.float32)
    y_train = np.asarray(y[train_slice], dtype=np.uint8)
    y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
    valid_dates = np.asarray(dates[valid_slice], dtype=np.int64)
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        free_raw_data=False,
        params={"max_bin": int(plan["config"]["max_bin"]), "feature_pre_filter": False},
    )
    dataset.construct()

    completed = cached = failed = 0
    elapsed_values: list[float] = []
    for seed in plan["seeds"]:
        identity = {
            "engine_version": ENGINE_VERSION,
            "dataset_signature": plan["dataset_signature"],
            "config_name": plan["config_name"],
            "config": plan["config"],
            "fold": fold,
            "profile_hash": profile_hash,
            "feature_count": len(features),
            "seed": int(seed),
            "best_iteration": int(plan["best_iteration"]),
        }
        task_id = canonical_hash(identity)
        task_path = output_dir / f"{task_id}.json"
        old = read_json(task_path, {})
        if old.get("status") == "completed" and old.get("identity") == identity:
            cached += 1
            continue
        started = time.perf_counter()
        try:
            model = lgb.train(
                _params(plan["config"], y_train, int(plan["threads"]), int(seed)),
                dataset,
                num_boost_round=int(plan["best_iteration"]),
                callbacks=[lgb.log_evaluation(0)],
            )
            prediction = np.asarray(model.predict(x_valid), dtype=np.float32)
            elapsed = time.perf_counter() - started
            elapsed_values.append(elapsed)
            result = {
                "status": "completed",
                "identity": identity,
                "task_id": task_id,
                "backend": "lightgbm_cpu",
                "config_name": plan["config_name"],
                "representative_profile": plan["representative_profile"],
                "profile_hash": profile_hash,
                "feature_count": len(features),
                "seed": int(seed),
                "outer_fold": int(fold["fold_id"]),
                "best_iteration": int(plan["best_iteration"]),
                "elapsed_seconds": elapsed,
                "memory_wait_seconds": memory_wait,
                **compute_metrics(y_valid, prediction, valid_dates),
            }
            atomic_json(result, task_path)
            completed += 1
            del model, prediction
        except Exception as exc:
            failed += 1
            atomic_json(
                {
                    "status": "failed",
                    "identity": identity,
                    "task_id": task_id,
                    "backend": "lightgbm_cpu",
                    "config_name": plan["config_name"],
                    "representative_profile": plan["representative_profile"],
                    "profile_hash": profile_hash,
                    "seed": int(seed),
                    "outer_fold": int(fold["fold_id"]),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                task_path,
            )
        gc.collect()

    del dataset, x_train, x_valid, y_train, y_valid, valid_dates
    gc.collect()
    return {
        "config_name": plan["config_name"],
        "representative_profile": plan["representative_profile"],
        "profile_hash": profile_hash,
        "fold": int(fold["fold_id"]),
        "completed": completed,
        "cached": cached,
        "failed": failed,
        "median_model_seconds": float(np.median(elapsed_values)) if elapsed_values else None,
    }


def _deduplicate_profiles(profiles: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    unique: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}
    hash_to_name: dict[str, str] = {}
    for name, features in profiles.items():
        digest = hash_strings(features)
        if digest in hash_to_name:
            aliases[name] = hash_to_name[digest]
        else:
            hash_to_name[digest] = name
            unique[name] = features
    return unique, aliases


def _run_once(plans: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(_bundle_worker, plan) for plan in plans]
        for future in cf.as_completed(futures):
            results.append(future.result())
    return results


def run_profiles(
    *,
    prepared,
    folds,
    profiles: dict[str, list[str]],
    seeds: list[int],
    best_iterations: dict[int, int],
    config: dict[str, Any],
    config_name: str,
    task_dir: Path,
    workers: int,
    threads_per_worker: int,
    min_free_ram_gb: float,
    priority: str,
    retries: int = 1,
) -> dict[str, Any]:
    unique, aliases = _deduplicate_profiles(profiles)
    fold_order = {fold_id: position for position, fold_id in enumerate([7, 6, 5, 4, 3, 2, 1, 0])}
    plans: list[dict[str, Any]] = []
    for fold in sorted(folds, key=lambda item: fold_order[item.fold_id]):
        for name, features in unique.items():
            plans.append(
                {
                    "cache_root": str(prepared.root),
                    "feature_names": prepared.feature_names,
                    "dataset_signature": prepared.signature,
                    "fold": fold.to_dict(),
                    "features": features,
                    "representative_profile": name,
                    "seeds": [int(seed) for seed in seeds],
                    "best_iteration": int(best_iterations[fold.fold_id]),
                    "config": config,
                    "config_name": config_name,
                    "task_dir": str(Path(task_dir) / config_name),
                    "threads": int(threads_per_worker),
                    "min_free_ram_gb": float(min_free_ram_gb),
                    "priority": priority,
                }
            )
    started = time.time()
    passes: list[list[dict[str, Any]]] = []
    for attempt in range(max(1, int(retries) + 1)):
        results = _run_once(plans, int(workers))
        passes.append(results)
        if sum(item["failed"] for item in results) == 0:
            break
    final_results = passes[-1] if passes else []
    expected = len(unique) * len(folds) * len(seeds)
    summary = {
        "status": "completed" if sum(item["failed"] for item in final_results) == 0 else "partial",
        "config_name": config_name,
        "input_profile_count": len(profiles),
        "unique_profile_count": len(unique),
        "aliases": aliases,
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "expected_model_count": expected,
        "completed_last_pass": sum(item["completed"] for item in final_results),
        "cached_last_pass": sum(item["cached"] for item in final_results),
        "failed_last_pass": sum(item["failed"] for item in final_results),
        "elapsed_seconds": time.time() - started,
        "workers": int(workers),
        "threads_per_worker": int(threads_per_worker),
        "passes": passes,
    }
    atomic_json(summary, Path(task_dir) / f"{config_name}_stage_summary.json")
    return summary
