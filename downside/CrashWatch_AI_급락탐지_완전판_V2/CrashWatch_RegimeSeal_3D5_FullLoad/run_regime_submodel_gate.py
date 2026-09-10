from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from cw7h.utils import atomic_json, read_json
from cwfull.common import (
    acquire_lock,
    file_sha256,
    nvml_snapshot,
    release_lock,
    set_full_load_mode,
    set_worker_mode,
)
from cwregime.gating import (
    REGIME_KO,
    SUBMODELS,
    apply_gate,
    canonical_hash,
    evaluate_prediction,
    fit_gate,
    performance_table,
    submodel_label_table,
)
from run_regime_temporal_breadth import _merge_predictions


PROFILES = {
    "P2": "P2_DEDUP_CLEAN",
    "P7": "P7_CORR095_PLUS_CONDITIONAL",
}
MODEL_CONFIGS = {
    "P2_LGB": ("lightgbm", "P2_DEDUP_CLEAN", "L04_159_REG"),
    "P2_XGB": ("xgboost", "P2_DEDUP_CLEAN", "X08_CW075"),
    "P7_LGB": ("lightgbm", "P7_CORR095_PLUS_CONDITIONAL", "L10_LEAF63_CW125"),
    "P7_XGB": ("xgboost", "P7_CORR095_PLUS_CONDITIONAL", "X03_160_REG"),
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def _resource_monitor(stop: threading.Event, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    while not stop.wait(3.0):
        memory = psutil.virtual_memory()
        rows.append({
            "epoch": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": memory.used / 1024**3,
            "ram_available_gb": memory.available / 1024**3,
            **nvml_snapshot(),
        })
        if len(rows) >= 20:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
            rows.clear()
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_results(output: Path, model: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted((output / "checkpoints" / "temporal_breadth" / model).glob("*.json")):
        item = read_json(path, {})
        npz = Path(str(item.get("npz_path", "")))
        if item.get("status") == "complete" and npz.exists() and npz.stat().st_size > 0:
            results.append(item)
    return results


def _load_oof(selection: Path, temporal: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = read_json(selection / "RUNTIME_MANIFEST.json", {})
    if not meta:
        raise FileNotFoundError(selection / "RUNTIME_MANIFEST.json")
    calendar = pd.read_csv(selection / "regime_calendar.csv")
    windows = pd.read_csv(temporal / "TEMPORAL_BREADTH_WINDOWS.csv").to_dict("records")
    lightgbm = _load_results(temporal, "lightgbm")
    xgboost = _load_results(temporal, "xgboost")
    if len(lightgbm) != 64 or len(xgboost) != 64:
        raise RuntimeError(f"temporal cache incomplete: lightgbm={len(lightgbm)}/64, xgboost={len(xgboost)}/64")
    merged = _merge_predictions(lightgbm, xgboost, meta, calendar, windows)
    p2 = merged[PROFILES["P2"]].copy()
    p7 = merged[PROFILES["P7"]].copy()
    wide = p2[[
        "row_id", "date", "date_ns", "ticker", "target", "first_hit_day", "regime",
        "temporal_block", "window_id", "prediction_lgb", "prediction_xgb",
    ]].rename(columns={"prediction_lgb": "P2_LGB", "prediction_xgb": "P2_XGB"})
    wide = wide.merge(
        p7[["row_id", "window_id", "prediction_lgb", "prediction_xgb"]].rename(
            columns={"prediction_lgb": "P7_LGB", "prediction_xgb": "P7_XGB"}
        ),
        on=["row_id", "window_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(wide) != len(p2) or wide[list(SUBMODELS)].isna().any().any():
        raise RuntimeError("P2/P7 cached OOF merge is incomplete")
    if wide.duplicated(["row_id", "window_id"]).any():
        raise RuntimeError("duplicate OOF row/window detected")
    return wide, meta


def _candidate_grid() -> list[dict[str, Any]]:
    candidates = []
    for shrink_rows in (0, 200, 500, 1000, 2000):
        candidates.append({"method": "hard", "shrink_rows": shrink_rows, "n_clusters": 1})
        for clusters in (2, 3, 4):
            candidates.append({"method": "cluster", "shrink_rows": shrink_rows, "n_clusters": clusters})
    return candidates


def _prequential_gate_selection(wide: pd.DataFrame, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    regime_parts: list[pd.DataFrame] = []
    for candidate in _candidate_grid():
        candidate_id = canonical_hash(candidate, 12)
        for outer_number in (2, 3):
            train_blocks = [f"B{number}" for number in range(1, outer_number)]
            outer_block = f"B{outer_number}"
            train = wide[wide["temporal_block"].isin(train_blocks)]
            outer = wide[wide["temporal_block"] == outer_block]
            gate = fit_gate(train, **candidate)
            gated = apply_gate(outer, gate)
            row, regimes = evaluate_prediction(gated, f"GATE_{candidate_id}")
            row.update({
                "candidate_id": candidate_id,
                "outer_block": outer_block,
                "train_blocks": ",".join(train_blocks),
                **candidate,
                "mapping": json.dumps(gate.mapping, sort_keys=True, ensure_ascii=False),
            })
            rows.append(row)
            regimes.insert(0, "candidate_id", candidate_id)
            regimes.insert(1, "outer_block", outer_block)
            regime_parts.append(regimes)
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "GATE_CANDIDATE_PREQUENTIAL_METRICS.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "GATE_CANDIDATE_PREQUENTIAL_REGIME_METRICS.csv")

    summaries = []
    for candidate_id, part in metrics.groupby("candidate_id", sort=False):
        values = np.clip(part["selection_score"].to_numpy(dtype=float), 1e-9, None)
        first = part.iloc[0]
        summaries.append({
            "candidate_id": candidate_id,
            "method": first["method"],
            "shrink_rows": int(first["shrink_rows"]),
            "n_clusters": int(first["n_clusters"]),
            "B2_selection_score": float(part.loc[part.outer_block == "B2", "selection_score"].iloc[0]),
            "B3_selection_score": float(part.loc[part.outer_block == "B3", "selection_score"].iloc[0]),
            "geometric_mean_selection_score": float(np.exp(np.mean(np.log(values)))),
            "mean_worst_regime_pr_lift": float(part["worst_regime_pr_lift"].mean()),
            "mean_top3_precision_lift": float(part["top3_precision_lift"].mean()),
            "complexity_rank": 0 if first["method"] == "hard" else int(first["n_clusters"]),
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["geometric_mean_selection_score", "mean_worst_regime_pr_lift", "complexity_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    atomic_csv(summary, output / "GATE_CANDIDATE_SELECTION_SUMMARY.csv")
    winner = summary.iloc[0].to_dict()
    selected = {
        "selection_policy": "maximize geometric mean of honest prequential B2 and B3 selection scores; B4 excluded",
        "candidate_id": str(winner["candidate_id"]),
        "method": str(winner["method"]),
        "shrink_rows": int(winner["shrink_rows"]),
        "n_clusters": int(winner["n_clusters"]),
        "B2_selection_score": float(winner["B2_selection_score"]),
        "B3_selection_score": float(winner["B3_selection_score"]),
        "geometric_mean_selection_score": float(winner["geometric_mean_selection_score"]),
        "B4_used_for_selection": False,
    }
    atomic_json(selected, output / "GATE_SELECTION_BEFORE_B4.json")
    return selected


def _pseudo_sealed_b4(wide: pd.DataFrame, selected: dict[str, Any], output: Path) -> dict[str, Any]:
    params = {key: selected[key] for key in ("method", "shrink_rows", "n_clusters")}
    train = wide[wide["temporal_block"].isin(["B1", "B2", "B3"])]
    b4 = wide[wide["temporal_block"] == "B4"]
    gate = fit_gate(train, **params)
    gated = apply_gate(b4, gate)
    candidates: list[tuple[str, pd.DataFrame, str]] = [("FROZEN_GATE", gated, "prediction")]
    for model in SUBMODELS:
        candidates.append((model, b4, model))
    locked = b4.copy()
    locked["prediction"] = 0.1 * locked["P2_LGB"] + 0.9 * locked["P2_XGB"]
    candidates.append(("LOCKED_P2_LGB10_XGB90", locked, "prediction"))
    metric_rows = []
    regime_parts = []
    for name, frame, column in candidates:
        row, regimes = evaluate_prediction(frame, name, column)
        metric_rows.append(row)
        regime_parts.append(regimes)
    metrics = pd.DataFrame(metric_rows).sort_values("selection_score", ascending=False)
    atomic_csv(metrics, output / "PSEUDO_SEALED_B4_METRICS.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "PSEUDO_SEALED_B4_REGIME_METRICS.csv")
    gate.regime_table.to_csv(output / "PSEUDO_SEALED_B4_TRAINED_REGIME_MAP.csv", index=False, encoding="utf-8-sig")
    gate.cluster_table.to_csv(output / "PSEUDO_SEALED_B4_TRAINED_CLUSTERS.csv", index=False, encoding="utf-8-sig")
    gated[[
        "row_id", "date", "ticker", "target", "regime", "selected_submodel", "prediction",
    ]].to_parquet(output / "PSEUDO_SEALED_B4_PREDICTIONS.parquet", index=False)
    gate_row = metrics[metrics["candidate"] == "FROZEN_GATE"].iloc[0]
    locked_row = metrics[metrics["candidate"] == "LOCKED_P2_LGB10_XGB90"].iloc[0]
    result = {
        "status": "COMPLETED_ONCE",
        "scope": "development pseudo-seal B4; not the external final sealed interval",
        "train_blocks": ["B1", "B2", "B3"],
        "test_block": "B4",
        "mapping": gate.mapping,
        "gate_selection_score": float(gate_row["selection_score"]),
        "locked_selection_score": float(locked_row["selection_score"]),
        "gate_minus_locked": float(gate_row["selection_score"] - locked_row["selection_score"]),
        "gate_pr_auc": float(gate_row["overall_raw_pr_auc"]),
        "locked_pr_auc": float(locked_row["overall_raw_pr_auc"]),
        "gate_pr_auc_minus_locked": float(gate_row["overall_raw_pr_auc"] - locked_row["overall_raw_pr_auc"]),
        "interpretation": "positive delta supports gating; non-positive delta means the combined gate is not yet superior",
    }
    atomic_json(result, output / "PSEUDO_SEALED_B4_RESULT.json")
    return result


def _freeze_final_gate(wide: pd.DataFrame, selected: dict[str, Any], meta: dict[str, Any], output: Path) -> dict[str, Any]:
    params = {key: selected[key] for key in ("method", "shrink_rows", "n_clusters")}
    gate = fit_gate(wide, **params)
    atomic_csv(gate.regime_table, output / "FINAL_REGIME_GATE_TABLE.csv")
    atomic_csv(gate.cluster_table, output / "FINAL_RECLUSTERING.csv")
    atomic_csv(submodel_label_table(gate), output / "P2_P7_SUBMODEL_REGIME_LABELS.csv")

    profile_features = {profile: list(meta["profiles"][name]) for profile, name in PROFILES.items()}
    artifact_core = {
        "schema": "crashwatch_regime_submodel_gate_v1",
        "target": {"horizon_trading_days": 3, "drop_threshold": -0.05},
        "development_max": str(meta["dataset_date_max"])[:10],
        "temporal_blocks": ["B1", "B2", "B3", "B4"],
        "submodels": list(SUBMODELS),
        "model_configs": MODEL_CONFIGS,
        "gate_method": gate.method,
        "shrink_rows": gate.shrink_rows,
        "n_clusters": gate.n_clusters,
        "fallback_model": gate.fallback_model,
        "regime_mapping": gate.mapping,
        "regime_definitions": REGIME_KO,
        "profile_features": profile_features,
        "feature_hashes": {
            profile: canonical_hash(features, 24) for profile, features in profile_features.items()
        },
        "selection_source": "cached temporal OOF only; external sealed labels were not read",
    }
    artifact = {**artifact_core, "artifact_hash": canonical_hash(artifact_core, 32), "frozen": True}
    atomic_json(artifact, output / "FROZEN_REGIME_GATE.json")
    return artifact


def _config_params(config: dict[str, Any], algorithm: str, config_id: str) -> dict[str, Any]:
    for item in config["search_space"][algorithm]:
        if item["id"] == config_id:
            return dict(item["params"])
    raise KeyError(f"missing config {algorithm}/{config_id}")


def _cached_model(task: dict[str, Any]) -> dict[str, Any] | None:
    sidecar = Path(task["sidecar"])
    model_path = Path(task["model_path"])
    cached = read_json(sidecar, {})
    if cached.get("status") == "complete" and cached.get("task_hash") == task["task_hash"] and model_path.exists():
        if cached.get("sha256") == file_sha256(model_path):
            return {**cached, "cache_reused": True}
    return None


def _train_lightgbm(task: dict[str, Any]) -> dict[str, Any]:
    cached = _cached_model(task)
    if cached:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(task["threads"]), "above_normal")
        import lightgbm as lgb

        matrix = np.load(task["matrix_path"], mmap_mode="r")
        target = np.load(task["target_path"], mmap_mode="r")
        valid = np.load(task["valid_path"], mmap_mode="r").astype(bool)
        x = np.asarray(matrix[valid], dtype=np.float32)
        y = np.asarray(target[valid], dtype=np.uint8)
        params = dict(task["params"])
        rounds = int(params.pop("rounds"))
        multiplier = float(params.pop("class_weight_multiplier", 1.0))
        positives = max(1, int(y.sum()))
        negatives = max(1, int(len(y) - positives))
        params.update({
            "objective": "binary",
            "metric": "aucpr",
            "verbosity": -1,
            "num_threads": int(task["threads"]),
            "scale_pos_weight": (negatives / positives) * multiplier,
            "seed": int(task["seed"]),
            "feature_fraction_seed": int(task["seed"]),
            "bagging_seed": int(task["seed"]),
            "drop_seed": int(task["seed"]),
        })
        dataset = lgb.Dataset(x, label=y, free_raw_data=False)
        model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        model_path = Path(task["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # LightGBM's native Windows writer can fail on non-ASCII paths even
        # though Python itself handles them correctly.  Save under the ASCII
        # system temp directory, then let Python perform the Unicode-safe move.
        ascii_temp_dir = Path(tempfile.gettempdir()) / "cwregime_lgb_models"
        ascii_temp_dir.mkdir(parents=True, exist_ok=True)
        temp = ascii_temp_dir / f"{task['task_hash']}.{os.getpid()}.txt"
        model.save_model(str(temp))
        os.replace(temp, model_path)
        result = {
            "status": "complete", "submodel": task["submodel"], "algorithm": "lightgbm",
            "seed": int(task["seed"]), "rows": int(len(y)), "positives": int(y.sum()),
            "model_path": str(model_path), "sha256": file_sha256(model_path),
            "task_hash": task["task_hash"], "elapsed_seconds": time.time() - started,
            "cache_reused": False,
        }
        atomic_json(result, Path(task["sidecar"]))
        return result
    except Exception as exc:
        result = {"status": "failed", "submodel": task.get("submodel"), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, Path(task["sidecar"]))
        return result


def _train_xgboost(task: dict[str, Any]) -> dict[str, Any]:
    cached = _cached_model(task)
    if cached:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(task["threads"]), "above_normal")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import xgboost as xgb

        matrix = np.load(task["matrix_path"], mmap_mode="r")
        target = np.load(task["target_path"], mmap_mode="r")
        valid = np.load(task["valid_path"], mmap_mode="r").astype(bool)
        x = np.asarray(matrix[valid], dtype=np.float32)
        y = np.asarray(target[valid], dtype=np.uint8)
        params = dict(task["params"])
        rounds = int(params.pop("rounds"))
        multiplier = float(params.pop("class_weight_multiplier", 1.0))
        max_bin = int(params.get("max_bin", 256))
        positives = max(1, int(y.sum()))
        negatives = max(1, int(len(y) - positives))
        dtrain = xgb.QuantileDMatrix(x, label=y, max_bin=max_bin, nthread=int(task["threads"]))
        params.update({
            "objective": "binary:logistic", "eval_metric": "aucpr", "device": "cuda",
            "tree_method": "hist", "scale_pos_weight": (negatives / positives) * multiplier,
            "seed": int(task["seed"]), "verbosity": 0, "nthread": int(task["threads"]),
        })
        model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
        model_path = Path(task["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp = model_path.with_name(f".{model_path.stem}.{os.getpid()}.ubj")
        model.save_model(temp)
        os.replace(temp, model_path)
        result = {
            "status": "complete", "submodel": task["submodel"], "algorithm": "xgboost",
            "seed": int(task["seed"]), "rows": int(len(y)), "positives": int(y.sum()),
            "model_path": str(model_path), "sha256": file_sha256(model_path),
            "task_hash": task["task_hash"], "elapsed_seconds": time.time() - started,
            "cache_reused": False,
        }
        atomic_json(result, Path(task["sidecar"]))
        return result
    except Exception as exc:
        result = {"status": "failed", "submodel": task.get("submodel"), "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, Path(task["sidecar"]))
        return result


def _train_final_models(
    package: Path,
    output: Path,
    meta: dict[str, Any],
    gate_artifact: dict[str, Any],
    seeds: list[int],
    lgb_workers: int,
    lgb_threads: int,
    xgb_workers: int,
    xgb_threads: int,
) -> dict[str, Any]:
    config = read_json(package / "config_regime_3d5_full_load.json", {})
    tasks = {"lightgbm": [], "xgboost": []}
    models = output / "models"
    for submodel, (algorithm, profile, config_id) in MODEL_CONFIGS.items():
        params = _config_params(config, algorithm, config_id)
        for seed in seeds:
            suffix = ".txt" if algorithm == "lightgbm" else ".ubj"
            model_path = models / submodel / f"seed_{seed}{suffix}"
            identity = {
                "schema": "crashwatch_final_submodel_v1", "submodel": submodel,
                "algorithm": algorithm, "profile": profile, "config_id": config_id,
                "params": params, "seed": int(seed), "dataset_signature": meta["dataset_signature"],
                "gate_hash": gate_artifact["artifact_hash"],
            }
            task = {
                **identity, "task_hash": canonical_hash(identity, 32),
                "matrix_path": meta["matrix_paths"][profile], "target_path": meta["target_path"],
                "valid_path": meta["valid_path"], "params": params,
                "threads": lgb_threads if algorithm == "lightgbm" else xgb_threads,
                "model_path": str(model_path), "sidecar": str(model_path.with_suffix(model_path.suffix + ".json")),
            }
            tasks[algorithm].append(task)

    results: dict[str, list[dict[str, Any]]] = {"lightgbm": [], "xgboost": []}
    errors: list[str] = []

    def run_group(algorithm: str, worker, worker_count: int) -> None:
        try:
            context = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                results[algorithm] = list(executor.map(worker, tasks[algorithm]))
        except Exception:
            errors.append(traceback.format_exc())

    lgb_thread = threading.Thread(target=run_group, args=("lightgbm", _train_lightgbm, lgb_workers), daemon=False)
    xgb_thread = threading.Thread(target=run_group, args=("xgboost", _train_xgboost, xgb_workers), daemon=False)
    lgb_thread.start(); xgb_thread.start()
    lgb_thread.join(); xgb_thread.join()
    if errors:
        raise RuntimeError("final model pool failed: " + errors[0])
    all_results = results["lightgbm"] + results["xgboost"]
    failed = [item for item in all_results if item.get("status") != "complete"]
    if len(all_results) != len(tasks["lightgbm"]) + len(tasks["xgboost"]) or failed:
        raise RuntimeError(f"final model training incomplete: results={len(all_results)}, failed={failed[:2]}")
    manifest = {
        "status": "complete", "gate_hash": gate_artifact["artifact_hash"], "seeds": seeds,
        "expected_models": len(all_results), "completed_models": len(all_results),
        "cache_reused": int(sum(bool(item.get("cache_reused")) for item in all_results)),
        "hardware_plan": {
            "lightgbm_workers": lgb_workers, "threads_per_lightgbm": lgb_threads,
            "xgboost_gpu_workers": xgb_workers, "threads_per_xgboost": xgb_threads,
        },
        "models": all_results,
    }
    atomic_json(manifest, output / "FINAL_MODEL_MANIFEST.json")
    return manifest


def _future_readiness(project: Path, meta: dict[str, Any], gate: dict[str, Any], output: Path) -> dict[str, Any]:
    candidates = {
        "ticker_features": project / "CrashWatch_AI_V4_LongRun_5Day/crashwatch_ai_data/features/dual_ablation/ticker_features.parquet",
        "universe_features": project / "CrashWatch_AI_V4_LongRun_5Day/crashwatch_ai_data/features/dual_ablation/universe_features.parquet",
        "finance_ticker_features": project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3/crashwatch_ai_data/features/dual_ablation/finance11h_ticker_features.parquet",
        "finance_universe_features": project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3/crashwatch_ai_data/features/dual_ablation/finance11h_universe_features.parquet",
        "future_price_source": project / "CrashWatch_AI_V4_LongRun_5Day/crashwatch_ai_data/raw/dual_ablation/krx_ticker_timeseries.parquet",
        "legacy_label_tail": project / "crashwatch_ai_data/sealed/label_tail.parquet",
    }
    source_rows = []
    union: set[str] = set()
    for name, path in candidates.items():
        if not path.exists():
            source_rows.append({"source": name, "path": str(path), "exists": False})
            continue
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        union.update(columns)
        date_max = None
        if "date" in columns:
            dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"], errors="coerce")
            date_max = str(dates.max().date()) if dates.notna().any() else None
        source_rows.append({
            "source": name, "path": str(path), "exists": True,
            "rows": int(parquet.metadata.num_rows), "columns": len(columns), "date_max": date_max,
        })
    atomic_csv(pd.DataFrame(source_rows), output / "FUTURE_SOURCE_AUDIT.csv")
    required = set().union(*[set(features) for features in gate["profile_features"].values()])
    missing = sorted(required - union)
    future_dates = 0
    price_path = candidates["future_price_source"]
    if price_path.exists():
        dates = pd.to_datetime(pd.read_parquet(price_path, columns=["date"])["date"], errors="coerce")
        future_dates = int(dates[dates > pd.Timestamp(str(meta["dataset_date_max"])[:10])].nunique())
    readiness = {
        "status": "BLOCKED_BEFORE_FINAL_SEALED",
        "gate_frozen": True,
        "gate_hash": gate["artifact_hash"],
        "development_max": str(meta["dataset_date_max"])[:10],
        "future_trading_dates_available": future_dates,
        "valid_target_dates_after_3d_horizon": max(0, future_dates - 3),
        "theoretical_dates_needed_for_8_regimes_at_8_dates_each": 64,
        "required_feature_count": len(required),
        "required_features_present_across_sources": len(required) - len(missing),
        "missing_features_across_sources": missing,
        "feature_reconstruction_possible": not missing,
        "blocking_reasons": [
            "finance feature parquet currently ends at development max and must be regenerated through the future interval",
            "only 20 future trading dates exist locally; this cannot validate all 8 regimes with 8 dates each",
            "one-time external sealed evaluation requires the exact confirmation phrase after preflight is ready",
        ],
        "sealed_targets_read": False,
        "sealed_predictions_made": False,
        "original_S00_S03_usable": False,
        "original_S00_S03_reason": "they overlap the 3D5 development interval and use a different 20-day target schema",
    }
    atomic_json(readiness, output / "FINAL_SEALED_READINESS.json")
    return readiness


def _export(package: Path, output: Path, status: dict[str, Any]) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_P2_P7_RegimeGate_RESULTS_{stamp}.zip"
    important = {
        "RUN_STATUS.json", "OOF_CACHE_AUDIT.json", "GATE_SELECTION_BEFORE_B4.json",
        "GATE_CANDIDATE_SELECTION_SUMMARY.csv", "PSEUDO_SEALED_B4_RESULT.json",
        "PSEUDO_SEALED_B4_METRICS.csv", "PSEUDO_SEALED_B4_REGIME_METRICS.csv",
        "FINAL_REGIME_GATE_TABLE.csv", "FINAL_RECLUSTERING.csv", "P2_P7_SUBMODEL_REGIME_LABELS.csv",
        "FROZEN_REGIME_GATE.json", "FINAL_MODEL_MANIFEST.json", "FINAL_SEALED_READINESS.json",
        "FUTURE_SOURCE_AUDIT.csv", "SUBMODEL_METRICS_BY_BLOCK_REGIME.csv",
    }
    start = "\n".join([
        "# CrashWatch P2/P7 Regime Submodel Gate",
        "",
        f"- status: {status.get('status')}",
        "- target: next 3 trading days, cumulative drop <= -5%",
        "- gate selection: prequential B2/B3 only; B4 used once as development pseudo-seal",
        "- external sealed labels consumed: false",
        "",
        "Read results/PSEUDO_SEALED_B4_RESULT.json, results/FROZEN_REGIME_GATE.json,",
        "results/P2_P7_SUBMODEL_REGIME_LABELS.csv, and results/FINAL_SEALED_READINESS.json first.",
    ])
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start.encode("utf-8"))
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name in important:
                zipped.write(path, f"results/{path.name}")
        for relative in [
            "run_regime_submodel_gate.py", "cwregime/gating.py", "cwregime/regimes.py",
            "cwregime/scoring.py", "config_regime_3d5_full_load.json",
        ]:
            path = package / relative
            if path.exists():
                zipped.write(path, f"code_context/{relative}")
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        entries = len(zipped.namelist())
    if bad:
        raise RuntimeError(f"Result archive validation failed: {bad}")
    result = {
        "archive": str(archive), "size_bytes": archive.stat().st_size,
        "sha256": file_sha256(archive), "entries": entries, "verified": True,
    }
    archive.with_suffix(archive.suffix + ".sha256.txt").write_text(
        f"{result['sha256']}  {archive.name}\n", encoding="utf-8"
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    project = package.parent
    selection = Path(args.selection_output).expanduser().resolve()
    temporal = Path(args.temporal_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "regime_submodel_gate.lock.json"
    acquire_lock(lock)
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=_resource_monitor, args=(monitor_stop, output / "resource_usage.csv"), daemon=True)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        hardware["gpu"] = nvml_snapshot()
        hardware["mode"] = "full_load_7950x3d_96gb_rtx5080"
        atomic_json(hardware, output / "HARDWARE_FULL_LOAD.json")
        monitor.start()

        wide, meta = _load_oof(selection, temporal)
        cache_audit = {
            "status": "complete", "rows": len(wide), "dates": int(wide["date"].nunique()),
            "date_min": str(wide["date"].min()), "date_max": str(wide["date"].max()),
            "blocks": wide["temporal_block"].value_counts().sort_index().to_dict(),
            "submodels": list(SUBMODELS), "sealed_rows": 0, "sealed_targets_read": False,
        }
        atomic_json(cache_audit, output / "OOF_CACHE_AUDIT.json")
        metrics = pd.concat(
            [performance_table(wide[wide.temporal_block == block], block) for block in ("B1", "B2", "B3", "B4")]
            + [performance_table(wide, "POOLED")],
            ignore_index=True,
        )
        atomic_csv(metrics, output / "SUBMODEL_METRICS_BY_BLOCK_REGIME.csv")

        selected = _prequential_gate_selection(wide, output)
        pseudo = _pseudo_sealed_b4(wide, selected, output)
        gate = _freeze_final_gate(wide, selected, meta, output)
        readiness = _future_readiness(project, meta, gate, output)
        model_manifest = {"status": "skipped_by_cli"}
        if not args.analysis_only:
            seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
            model_manifest = _train_final_models(
                package, output, meta, gate, seeds,
                int(args.lgb_workers), int(args.lgb_threads), int(args.xgb_workers), int(args.xgb_threads),
            )

        status = {
            "status": "DEVELOPMENT_GATE_AND_FINAL_MODELS_COMPLETE",
            "elapsed_seconds": time.time() - started,
            "output": str(output),
            "selected_gate": selected,
            "pseudo_sealed_B4": pseudo,
            "gate_hash": gate["artifact_hash"],
            "final_models": model_manifest.get("status"),
            "final_model_count": model_manifest.get("completed_models", 0),
            "external_final_sealed": readiness["status"],
            "external_sealed_consumed": False,
        }
        atomic_json(status, output / "RUN_STATUS.json")
        export = _export(package, output, status)
        status["result_export"] = export
        atomic_json(status, output / "RUN_STATUS.json")
        return status
    except Exception as exc:
        failed = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
        atomic_json(failed, output / "RUN_STATUS.json")
        raise
    finally:
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=10)
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="P2/P7 regime submodel labeling, reclustering, and full-load final training")
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    parser.add_argument("--total-threads", type=int, default=32)
    parser.add_argument("--lgb-workers", type=int, default=3)
    parser.add_argument("--lgb-threads", type=int, default=8)
    parser.add_argument("--xgb-workers", type=int, default=2)
    parser.add_argument("--xgb-threads", type=int, default=4)
    parser.add_argument("--seeds", default="17,43,101,211,503")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
