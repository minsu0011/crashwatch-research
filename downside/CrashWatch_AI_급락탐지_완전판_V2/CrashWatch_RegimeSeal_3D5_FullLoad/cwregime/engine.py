from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from cw7h.utils import atomic_json, canonical_hash, read_json
from cwfull.common import set_worker_mode


def _slice_rows(dates: np.ndarray, start_ns: int, end_ns: int) -> tuple[int, int]:
    return int(np.searchsorted(dates, start_ns, side="left")), int(np.searchsorted(dates, end_ns, side="right"))


def _bundle_paths(output_dir: Path, stage: str, model: str, bundle_id: str) -> tuple[Path, Path]:
    root = Path(output_dir) / "checkpoints" / stage / model
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{bundle_id}.json", root / f"{bundle_id}.npz"


def _prepare_arrays(plan: dict[str, Any]):
    X = np.load(plan["matrix_path"], mmap_mode="r")
    y = np.load(plan["target_path"], mmap_mode="r")
    valid = np.load(plan["valid_path"], mmap_mode="r")
    dates = np.load(plan["dates_path"], mmap_mode="r")
    first_hit = np.load(plan["first_hit_path"], mmap_mode="r")
    train_end = int(np.searchsorted(dates, int(plan["train_cutoff_ns"]), side="right"))
    val_start, val_end = _slice_rows(dates, int(plan["val_start_ns"]), int(plan["val_end_ns"]))
    train_idx = np.flatnonzero(np.asarray(valid[:train_end], dtype=bool))
    val_local = np.flatnonzero(np.asarray(valid[val_start:val_end], dtype=bool))
    val_idx = val_local + val_start
    if len(train_idx) < 1000 or len(val_idx) < 100:
        raise RuntimeError(f"insufficient rows train={len(train_idx)} val={len(val_idx)}")
    x_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.uint8)
    x_val = np.asarray(X[val_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.uint8)
    hit_val = np.asarray(first_hit[val_idx], dtype=np.int8)
    return x_train, y_train, x_val, y_val, hit_val, np.asarray(val_idx, dtype=np.int64)


def _lgb_bundle_worker(plan: dict[str, Any]) -> dict[str, Any]:
    json_path = Path(plan["json_path"]); npz_path = Path(plan["npz_path"])
    cached = read_json(json_path, {})
    if cached.get("status") == "complete" and npz_path.exists() and npz_path.stat().st_size > 0:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(plan["threads"]), str(plan.get("priority", "above_normal")))
        import lightgbm as lgb
        x_train, y_train, x_val, y_val, hit_val, val_idx = _prepare_arrays(plan)
        pos = max(1, int(np.sum(y_train == 1))); neg = max(1, int(np.sum(y_train == 0)))
        cfg = dict(plan["model_config"])
        rounds = int(cfg.pop("rounds"))
        weight_multiplier = float(cfg.pop("class_weight_multiplier", 1.0))
        base = {
            "objective": "binary", "metric": "None", "verbosity": -1,
            "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
            "num_threads": int(plan["threads"]), "scale_pos_weight": (neg / pos) * weight_multiplier,
        }
        base.update(cfg)
        dataset = lgb.Dataset(x_train, label=y_train, free_raw_data=False, params={"max_bin": int(base.get("max_bin", 255)), "feature_pre_filter": False})
        preds = []
        for seed in plan["seeds"]:
            params = dict(base)
            params.update({"seed": int(seed), "feature_fraction_seed": int(seed), "bagging_seed": int(seed), "drop_seed": int(seed)})
            model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
            preds.append(np.asarray(model.predict(x_val), dtype=np.float32))
        pred_matrix = np.vstack(preds)
        np.savez_compressed(npz_path, val_idx=val_idx, y=y_val, first_hit=hit_val, predictions=pred_matrix, seeds=np.asarray(plan["seeds"], dtype=np.int32))
        result = {
            "status": "complete", "bundle_id": plan["bundle_id"], "stage": plan["stage"], "model": "lightgbm",
            "profile": plan["profile"], "config_id": plan["config_id"], "window_id": plan["window_id"],
            "seeds": list(map(int, plan["seeds"])), "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
            "positive_rate_train": float(np.mean(y_train)), "positive_rate_validation": float(np.mean(y_val)),
            "elapsed_seconds": float(time.time() - started), "npz_path": str(npz_path),
        }
        atomic_json(result, json_path)
        return result
    except Exception as exc:
        result = {"status": "failed", "bundle_id": plan.get("bundle_id"), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, json_path)
        return result


def _xgb_bundle_worker(plan: dict[str, Any]) -> dict[str, Any]:
    json_path = Path(plan["json_path"]); npz_path = Path(plan["npz_path"])
    cached = read_json(json_path, {})
    if cached.get("status") == "complete" and npz_path.exists() and npz_path.stat().st_size > 0:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(plan["threads"]), str(plan.get("priority", "above_normal")))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import xgboost as xgb
        x_train, y_train, x_val, y_val, hit_val, val_idx = _prepare_arrays(plan)
        cfg = dict(plan["model_config"])
        rounds = int(cfg.pop("rounds"))
        weight_multiplier = float(cfg.pop("class_weight_multiplier", 1.0))
        max_bin = int(cfg.get("max_bin", 256))
        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=max_bin, nthread=int(plan["threads"]))
        dval = xgb.QuantileDMatrix(x_val, ref=dtrain, max_bin=max_bin, nthread=int(plan["threads"]))
        pos = max(1, int(np.sum(y_train == 1))); neg = max(1, int(np.sum(y_train == 0)))
        preds = []
        for seed in plan["seeds"]:
            params = {
                "objective": "binary:logistic", "eval_metric": "aucpr", "device": "cuda", "tree_method": "hist",
                "scale_pos_weight": (neg / pos) * weight_multiplier, "seed": int(seed), "verbosity": 0,
                "nthread": int(plan["threads"]),
            }
            params.update(cfg)
            model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
            preds.append(np.asarray(model.predict(dval), dtype=np.float32))
        pred_matrix = np.vstack(preds)
        np.savez_compressed(npz_path, val_idx=val_idx, y=y_val, first_hit=hit_val, predictions=pred_matrix, seeds=np.asarray(plan["seeds"], dtype=np.int32))
        result = {
            "status": "complete", "bundle_id": plan["bundle_id"], "stage": plan["stage"], "model": "xgboost",
            "profile": plan["profile"], "config_id": plan["config_id"], "window_id": plan["window_id"],
            "seeds": list(map(int, plan["seeds"])), "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
            "positive_rate_train": float(np.mean(y_train)), "positive_rate_validation": float(np.mean(y_val)),
            "elapsed_seconds": float(time.time() - started), "npz_path": str(npz_path),
        }
        atomic_json(result, json_path)
        return result
    except Exception as exc:
        result = {"status": "failed", "bundle_id": plan.get("bundle_id"), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, json_path)
        return result


def make_bundle_plan(
    output_dir: Path, stage: str, model: str, profile: str, config_id: str, model_config: dict[str, Any],
    window: dict[str, Any], seeds: list[int], paths: dict[str, str], threads: int,
) -> dict[str, Any]:
    # Cache identity must change if a fixed config or chronology window changes.
    # This preserves resume speed while preventing stale checkpoints from being
    # accepted after an experiment-definition edit.
    identity = {
        "schema": "cw_regime_bundle_v2",
        "stage": stage,
        "model": model,
        "profile": profile,
        "config_id": config_id,
        "model_config": model_config,
        "window": window,
        "seeds": seeds,
    }
    bundle_id = canonical_hash(identity, length=28)
    json_path, npz_path = _bundle_paths(output_dir, stage, model, bundle_id)
    return {
        **identity, "window_id": str(window["window_id"]), "bundle_id": bundle_id,
        "json_path": str(json_path), "npz_path": str(npz_path),
        "model_config": model_config, "train_cutoff_ns": int(window["train_cutoff_ns"]),
        "val_start_ns": int(window["start_ns"]), "val_end_ns": int(window["end_ns"]), "threads": int(threads),
        **paths,
    }


def _run_xgb_pool_entry(plans: list[dict[str, Any]], workers: int, result_path: str) -> None:
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as ex:
        results = list(ex.map(_xgb_bundle_worker, plans))
    atomic_json(results, Path(result_path))


def run_parallel_bundles(
    lgb_plans: list[dict[str, Any]], xgb_plans: list[dict[str, Any]], *,
    lgb_workers: int, xgb_workers: int, scratch_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run CPU LightGBM and GPU XGBoost concurrently to saturate 7950X3D + RTX 5080."""
    scratch_dir = Path(scratch_dir); scratch_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    xgb_result_path = scratch_dir / f"xgb_batch_{canonical_hash([p['bundle_id'] for p in xgb_plans], 16)}.json"
    xgb_process = None
    lgb_results: list[dict[str, Any]] = []
    xgb_results: list[dict[str, Any]] = []
    try:
        if xgb_plans:
            xgb_process = context.Process(target=_run_xgb_pool_entry, args=(xgb_plans, int(xgb_workers), str(xgb_result_path)), name="cw-regime-xgb-pool")
            xgb_process.start()
        if lgb_plans:
            with cf.ProcessPoolExecutor(max_workers=int(lgb_workers), mp_context=context) as ex:
                lgb_results = list(ex.map(_lgb_bundle_worker, lgb_plans))
        if xgb_process is not None:
            xgb_process.join()
            if xgb_process.exitcode != 0:
                raise RuntimeError(f"XGBoost pool exited with code {xgb_process.exitcode}")
            xgb_results = read_json(xgb_result_path, [])
        if len(lgb_results) != len(lgb_plans) or len(xgb_results) != len(xgb_plans):
            raise RuntimeError(
                f"bundle result count mismatch: lgb={len(lgb_results)}/{len(lgb_plans)}, "
                f"xgb={len(xgb_results)}/{len(xgb_plans)}"
            )
        failed = [r for r in lgb_results + xgb_results if r.get("status") != "complete"]
        if failed:
            raise RuntimeError(f"model bundles failed: {failed[:3]}")
        return lgb_results, xgb_results
    finally:
        if xgb_process is not None and xgb_process.is_alive():
            xgb_process.terminate()
            xgb_process.join(timeout=15)
            if xgb_process.is_alive():
                xgb_process.kill()
