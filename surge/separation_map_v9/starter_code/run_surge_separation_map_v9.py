from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gc
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from surge_separation_common_v9 import (
    ERROR_A_TOP_TP,
    ERROR_B_TOP_FP,
    ERROR_C_LOW_TP,
    ERROR_D_LOW_TN,
    ERROR_GROUP_NAMES,
    FoldSpec,
    RunStatus,
    aggregate_fold_metrics,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    build_error_group_codes,
    build_horizon_pairs,
    build_matched_pairs,
    cluster_innovation_values,
    compute_interaction_moment_screen,
    compute_output_inventory,
    datewise_percentile_rank,
    fit_forward_logistic_probe,
    graphml_escape,
    groupwise_median_delta,
    groupwise_percentile_rank,
    groupwise_robust_z,
    hash_strings,
    horizon_rank_spread,
    join_source_and_target,
    load_correlation_matrix,
    load_feature_profile,
    load_folds,
    load_json,
    log,
    pair_transform_values,
    peer_weighted_innovation_values,
    precision_curve_metrics,
    read_table,
    role_for_fold,
    safe_binary_metrics,
    separation_metrics,
    sha256_file,
    table_columns,
    utc_now,
    verify_feature_names,
    write_graphml,
)


DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_data_in_leaf": 55,
    "lambda_l1": 0.25,
    "lambda_l2": 1.8,
    "max_bin": 255,
    "min_gain_to_split": 1e-12,
    "feature_pre_filter": False,
    "force_col_wise": True,
    "verbosity": -1,
    "deterministic": True,
}

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "learning_rate": 0.035,
    "max_depth": 0,
    "max_leaves": 96,
    "grow_policy": "lossguide",
    "min_child_weight": 7.0,
    "reg_alpha": 0.2,
    "reg_lambda": 1.8,
    "max_bin": 256,
    "verbosity": 0,
}

ROLE_DEFAULTS = {
    "selection": [0, 1, 2, 3, 4],
    "confirmation": [5, 6],
    "recent_audit": [7],
}

TRANSFORM_TYPES = (
    "raw",
    "date_rank",
    "date_z",
    "date_market_rank",
    "date_bucket_rank",
    "date_industry_rank",
    "missing_indicator",
    "cluster_innovation",
    "peer_innovation_k3",
)


def analysis_workers(args: argparse.Namespace, item_count: int) -> int:
    """Bound independent map work without changing calculation order."""
    configured = max(1, int(getattr(args, "analysis_workers", 1)))
    return max(1, min(configured, int(item_count)))


def build_fold_evaluation_contexts(
    oof: pd.DataFrame,
    ab_pairs: pd.DataFrame,
    cd_pairs: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Prepare immutable fold indices and local matched-pair positions once.

    The original implementation rebuilt these DataFrames for every feature,
    transform and pair.  Reusing them is mathematically identical and removes
    the dominant Python/Pandas bookkeeping cost from the full-439 run.
    """

    def local_pairs(pair_table: pd.DataFrame, fold_id: int, local_position: Mapping[int, int]) -> pd.DataFrame:
        if pair_table.empty or "fold_id" not in pair_table.columns:
            return pd.DataFrame(columns=["case_index", "control_index"])
        pairs = pair_table[pair_table["fold_id"].eq(int(fold_id))][["case_index", "control_index"]].copy()
        if pairs.empty:
            return pairs
        pairs["case_index"] = pairs["case_index"].map(local_position)
        pairs["control_index"] = pairs["control_index"].map(local_position)
        pairs.dropna(subset=["case_index", "control_index"], inplace=True)
        pairs[["case_index", "control_index"]] = pairs[["case_index", "control_index"]].astype(np.int64)
        return pairs

    contexts: list[dict[str, Any]] = []
    for fold_id, part in oof.groupby("fold_id", sort=True):
        indices = part.index.to_numpy(dtype=np.int64)
        local_position = {int(global_index): position for position, global_index in enumerate(indices)}
        contexts.append(
            {
                "fold_id": int(fold_id),
                "indices": indices,
                "codes": part["error_code"].to_numpy(dtype=np.int8),
                "ab_pairs": local_pairs(ab_pairs, int(fold_id), local_position),
                "cd_pairs": local_pairs(cd_pairs, int(fold_id), local_position),
            }
        )
    return contexts


_PAIR_WORKER_STATE: dict[str, Any] = {}


def init_pair_process_worker(
    rank_matrix: np.ndarray,
    feature_index: Mapping[str, int],
    dates: np.ndarray,
    fold_contexts: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Sequence[int]],
) -> None:
    global _PAIR_WORKER_STATE
    _PAIR_WORKER_STATE = {
        "rank_matrix": rank_matrix,
        "feature_index": dict(feature_index),
        "dates": dates,
        "fold_contexts": list(fold_contexts),
        "roles": {key: list(values) for key, values in roles.items()},
    }


def compute_pair_process_worker(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    state = _PAIR_WORKER_STATE
    rank_matrix = state["rank_matrix"]
    feature_index = state["feature_index"]
    values_a = rank_matrix[:, feature_index[str(row["feature_a"])]].astype(np.float64)
    values_b = rank_matrix[:, feature_index[str(row["feature_b"])]].astype(np.float64)
    transforms = pair_transform_values(values_a, values_b, state["dates"])
    pair_records: list[dict[str, Any]] = []
    pair_values: dict[str, np.ndarray] = {}
    axis = str(row["axis"])
    for transform_name, values in transforms.items():
        node_id = f"PAIR::{transform_name}::{row['feature_a']}::{row['feature_b']}"
        pair_values[node_id] = values.astype(np.float32)
        for context in state["fold_contexts"]:
            fold_id = int(context["fold_id"])
            indices = context["indices"]
            local_codes = context["codes"]
            local_values = values[indices]
            fold_pairs = context["ab_pairs"] if axis == "AB" else context["cd_pairs"]
            positive_code = ERROR_A_TOP_TP if axis == "AB" else ERROR_C_LOW_TP
            negative_code = ERROR_B_TOP_FP if axis == "AB" else ERROR_D_LOW_TN
            metrics = separation_metrics(local_values, local_codes, positive_code, negative_code, fold_pairs)
            pair_records.append(
                {
                    "fold_id": fold_id,
                    "fold_role": role_for_fold(fold_id, state["roles"]),
                    "axis": axis,
                    "feature_a": str(row["feature_a"]),
                    "feature_b": str(row["feature_b"]),
                    "pair_transform": transform_name,
                    "node_id": node_id,
                    **metrics,
                }
            )
    return pair_records, pair_values


def resolve_default_paths(package_root: Path, args: argparse.Namespace) -> None:
    if args.dataset is None:
        args.dataset = package_root / "data" / "training_dataset_finance11h.parquet"
    if args.target_sidecar is None:
        args.target_sidecar = package_root / "data" / "surge_target_3d5.parquet"
    if args.folds is None:
        candidates = [
            package_root / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json",
            package_root / "references" / "feature_metadata" / "outer_walk_forward_folds.json",
        ]
        args.folds = next((path for path in candidates if path.exists()), candidates[-1])
    if args.feature_profile_manifest is None:
        candidates = [
            package_root / "references" / "feature_metadata" / "profile_manifest.json",
            package_root / "outputs" / "surge_pre_model_gate_v3" / "surge_feature_profiles_corrected.json",
        ]
        args.feature_profile_manifest = next((path for path in candidates if path.exists()), candidates[0])
    if args.correlation_matrix is None:
        candidates = [
            package_root / "outputs" / "surge_correlation_map_complete" / "cluster_basis_combined_abs.csv.gz",
            package_root / "outputs" / "surge_correlation_map_complete" / "combined_abs_correlation.csv.gz",
            package_root / "references" / "correlation_map_legacy_crash" / "cluster_basis_combined_abs.csv.gz",
        ]
        args.correlation_matrix = next((path for path in candidates if path.exists()), candidates[0])
    if args.v8_dir is None:
        args.v8_dir = package_root / "outputs" / "surge_organic_ablation_v8"
    if args.output is None:
        args.output = package_root / "outputs" / "surge_separation_map_v9"


def require_paths(paths: Mapping[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))


def parse_int_tokens(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_float_tokens(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(item) for item in value]


def resolve_roles(args: argparse.Namespace, folds: Sequence[FoldSpec]) -> dict[str, list[int]]:
    roles = {
        "selection": parse_int_tokens(args.selection_folds),
        "confirmation": parse_int_tokens(args.confirmation_folds),
        "recent_audit": parse_int_tokens(args.recent_folds),
    }
    known = {fold.fold_id for fold in folds}
    ownership: dict[int, str] = {}
    for role, fold_ids in roles.items():
        for fold_id in fold_ids:
            if fold_id not in known:
                raise ValueError(f"Unknown fold ID in role {role}: {fold_id}")
            if fold_id in ownership:
                raise ValueError(f"Fold {fold_id} assigned to {ownership[fold_id]} and {role}")
            ownership[fold_id] = role
    if not roles["selection"]:
        raise ValueError("At least one selection fold is required")
    return roles


def load_feature_universe(args: argparse.Namespace, dataset_columns: Sequence[str]) -> list[str]:
    features = load_feature_profile(Path(args.feature_profile_manifest), args.feature_profile)
    if args.limit_features > 0:
        features = features[: int(args.limit_features)]
    verify_feature_names(features, dataset_columns)
    if args.require_full_439 and len(features) != 439:
        raise RuntimeError(f"--require-full-439 is enabled but feature count is {len(features)}")
    return features


def load_dataset_frame(args: argparse.Namespace, features: Sequence[str]) -> pd.DataFrame:
    source_columns = set(features) | {
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.market_column,
        args.bucket_column,
        args.industry_column,
    }
    available = set(table_columns(Path(args.dataset)))
    source_columns = [column for column in source_columns if column in available]
    source = read_table(Path(args.dataset), columns=source_columns)
    target_columns = [
        column
        for column in [
            "source_row_id",
            args.date_column,
            args.ticker_column,
            args.target_column,
            args.target_valid_column,
            "first_hit_day",
            "best_forward_return_3d",
        ]
        if column in set(table_columns(Path(args.target_sidecar)))
    ]
    target = read_table(Path(args.target_sidecar), columns=target_columns)
    joined = join_source_and_target(
        source,
        target,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    joined[args.date_column] = pd.to_datetime(joined[args.date_column], errors="coerce")
    valid = joined[args.target_valid_column].astype(bool) & joined[args.target_column].isin([0, 1])
    joined = joined.loc[valid].reset_index(drop=True)
    for column in features:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in [args.market_column, args.bucket_column, args.industry_column, args.ticker_column]:
        if column not in joined.columns:
            joined[column] = "UNKNOWN"
        joined[column] = joined[column].astype("string").fillna("UNKNOWN")
    if joined[args.date_column].isna().any():
        raise ValueError("Date parsing failed in valid rows")
    return joined


def fold_indices(frame: pd.DataFrame, folds: Sequence[FoldSpec], date_column: str) -> dict[int, dict[str, np.ndarray]]:
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    result: dict[int, dict[str, np.ndarray]] = {}
    for fold in folds:
        train = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        validation = np.flatnonzero((dates >= fold.validation_start) & (dates <= fold.validation_end))
        if len(train) == 0 or len(validation) == 0:
            raise RuntimeError(f"fold {fold.fold_id} has empty train/validation rows")
        result[fold.fold_id] = {"train": train.astype(np.int64), "validation": validation.astype(np.int64)}
    return result


def resolve_xgboost_device(args: argparse.Namespace) -> str:
    if args.device == "cpu":
        return "cpu"
    try:
        import xgboost as xgb
    except ImportError:
        if args.allow_cpu_fallback:
            return "cpu"
        raise
    build_info = dict(xgb.build_info()) if hasattr(xgb, "build_info") else {}
    cuda_available = bool(build_info.get("USE_CUDA", False)) and shutil.which("nvidia-smi") is not None
    if args.device == "cuda" and not cuda_available:
        if args.allow_cpu_fallback:
            log("CUDA requested but unavailable; using CPU because --allow-cpu-fallback is set")
            return "cpu"
        raise RuntimeError("CUDA XGBoost requested but unavailable")
    return "cuda" if cuda_available else "cpu"


def train_lightgbm_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    iterations: int,
    seed: int,
    threads: int,
) -> np.ndarray:
    import lightgbm as lgb

    params = dict(DEFAULT_LGB_PARAMS)
    params.update({"seed": int(seed), "num_threads": int(threads)})
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=int(iterations))
    return booster.predict(x_validation, num_iteration=int(iterations)).astype(np.float64)


def train_xgboost_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    iterations: int,
    seed: int,
    threads: int,
    device: str,
) -> np.ndarray:
    import xgboost as xgb

    params = dict(DEFAULT_XGB_PARAMS)
    params.update({"seed": int(seed), "nthread": int(threads), "device": str(device)})
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=int(params["max_bin"]))
    dvalidation = xgb.QuantileDMatrix(x_validation, ref=dtrain, max_bin=int(params["max_bin"]))
    booster = xgb.train(params, dtrain, num_boost_round=int(iterations), verbose_eval=False)
    return booster.predict(dvalidation, iteration_range=(0, int(iterations))).astype(np.float64)


def load_best_iterations(v8_dir: Path, folds: Sequence[FoldSpec]) -> dict[str, dict[int, int]]:
    path = v8_dir / "best_iterations.json"
    defaults = {
        "lightgbm_cpu": {fold.fold_id: 75 for fold in folds},
        "xgboost_gpu": {fold.fold_id: 75 for fold in folds},
        "xgboost_cpu": {fold.fold_id: 75 for fold in folds},
    }
    if not path.exists():
        return defaults
    payload = load_json(path)
    effective = payload.get("effective_iterations", {})
    for backend in ["lightgbm_cpu", "xgboost_gpu", "xgboost_cpu"]:
        source_key = backend if backend in effective else "xgboost_gpu" if backend.startswith("xgboost") else backend
        if source_key in effective:
            defaults[backend] = {
                fold.fold_id: int(effective[source_key].get(str(fold.fold_id), defaults[backend][fold.fold_id]))
                for fold in folds
            }
    return defaults


def discover_v8_baseline_oof(
    v8_dir: Path,
    frame: pd.DataFrame,
    folds: Sequence[FoldSpec],
    fold_index: Mapping[int, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> pd.DataFrame | None:
    task_root = v8_dir / "task_results"
    if not task_root.exists():
        return None
    payloads: list[dict[str, Any]] = []
    for path in task_root.rglob("*.json"):
        with contextlib.suppress(Exception):
            payload = load_json(path)
            if payload.get("status") == "completed" and payload.get("test_type") == "baseline":
                payload["_result_path"] = str(path)
                payloads.append(payload)
    if not payloads:
        return None
    records: list[dict[str, Any]] = []
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        by_fold[int(payload["fold_id"])].append(payload)
    for fold in folds:
        validation_indices = np.asarray(fold_index[fold.fold_id]["validation"], dtype=np.int64)
        dates = frame.iloc[validation_indices][args.date_column].to_numpy(dtype="datetime64[ns]")
        rank_vectors: list[np.ndarray] = []
        raw_vectors: list[np.ndarray] = []
        backend_names: list[str] = []
        for payload in by_fold.get(fold.fold_id, []):
            prediction_path = payload.get("prediction_path")
            candidate_paths: list[Path] = []
            if prediction_path:
                candidate_paths.append(Path(str(prediction_path)))
            identity_hash = str(payload.get("identity_hash", ""))
            if identity_hash:
                candidate_paths.append(v8_dir / "prediction_cache" / f"{identity_hash}.npz")
            local_path = next((path for path in candidate_paths if path.exists()), None)
            if local_path is None:
                continue
            prediction = np.load(local_path, allow_pickle=False)
            indices = np.asarray(prediction["validation_indices"], dtype=np.int64)
            score = np.asarray(prediction["prediction"], dtype=np.float64)
            if not np.array_equal(indices, validation_indices):
                lookup = pd.Series(np.arange(len(indices), dtype=np.int64), index=indices)
                try:
                    order = lookup.loc[validation_indices].to_numpy(dtype=np.int64)
                except KeyError:
                    continue
                score = score[order]
            rank_vectors.append(datewise_percentile_rank(score, dates))
            raw_vectors.append(score)
            backend_names.append(str(payload.get("backend", "unknown")))
        if not rank_vectors:
            return None
        rank_matrix = np.vstack(rank_vectors)
        raw_matrix = np.vstack(raw_vectors)
        ensemble_rank = np.nanmean(rank_matrix, axis=0)
        ensemble_raw = np.nanmean(raw_matrix, axis=0)
        for local_index, global_index in enumerate(validation_indices):
            records.append(
                {
                    "row_index": int(global_index),
                    "source_row_id": int(frame.at[global_index, "source_row_id"]),
                    "fold_id": int(fold.fold_id),
                    "base_score_raw": float(ensemble_raw[local_index]),
                    "base_rank": float(ensemble_rank[local_index]),
                    "base_model_count": int(len(rank_vectors)),
                    "base_backends": "|".join(sorted(backend_names)),
                }
            )
    return pd.DataFrame.from_records(records)


def load_external_base_oof(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".npz"):
        payload = np.load(path, allow_pickle=False)
        if "row_index" in payload:
            row_index = np.asarray(payload["row_index"], dtype=np.int64)
        elif "validation_indices" in payload:
            row_index = np.asarray(payload["validation_indices"], dtype=np.int64)
        elif "validation_index" in payload:
            row_index = np.asarray(payload["validation_index"], dtype=np.int64)
        else:
            raise ValueError("NPZ base OOF needs row_index, validation_indices, or validation_index")
        score_key = next(
            (key for key in ["base_score_raw", "prediction", "score", "ensemble_score"] if key in payload),
            None,
        )
        if score_key is None:
            raise ValueError("NPZ base OOF has no supported score array")
        score = np.asarray(payload[score_key], dtype=np.float64)
        result = pd.DataFrame({"row_index": row_index, "base_score_raw": score})
        if "fold_id" in payload:
            result["fold_id"] = np.asarray(payload["fold_id"], dtype=np.int64)
    else:
        result = read_table(path)
    if "row_index" not in result.columns:
        if "source_row_id" not in result.columns:
            raise ValueError("External base OOF needs row_index or source_row_id")
        lookup = pd.Series(frame.index.to_numpy(dtype=np.int64), index=frame["source_row_id"].to_numpy(dtype=np.int64))
        result["row_index"] = lookup.loc[result["source_row_id"].to_numpy(dtype=np.int64)].to_numpy(dtype=np.int64)
    if "base_score_raw" not in result.columns:
        score_candidates = [column for column in ["score", "prediction", "surge_score", "ensemble_score"] if column in result.columns]
        if not score_candidates:
            raise ValueError("External base OOF has no score column")
        result["base_score_raw"] = pd.to_numeric(result[score_candidates[0]], errors="coerce")
    result = result.drop_duplicates("row_index", keep="last").copy()
    dates = frame.loc[result["row_index"].to_numpy(dtype=np.int64), "date"].to_numpy(dtype="datetime64[ns]")
    if "base_rank" not in result.columns:
        result["base_rank"] = datewise_percentile_rank(result["base_score_raw"].to_numpy(dtype=np.float64), dates)
    return result


def build_base_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    fold_index: Mapping[int, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    backends = [token.strip() for token in args.base_backends.split(",") if token.strip()]
    iterations = load_best_iterations(Path(args.v8_dir), folds)
    x = frame[list(features)].to_numpy(dtype=np.float32)
    y = frame[args.target_column].to_numpy(dtype=np.int8)
    records: list[dict[str, Any]] = []
    device = resolve_xgboost_device(args)
    for fold in folds:
        train_indices = np.asarray(fold_index[fold.fold_id]["train"], dtype=np.int64)
        validation_indices = np.asarray(fold_index[fold.fold_id]["validation"], dtype=np.int64)
        dates = frame.iloc[validation_indices][args.date_column].to_numpy(dtype="datetime64[ns]")
        rank_vectors: list[np.ndarray] = []
        raw_vectors: list[np.ndarray] = []
        used_backends: list[str] = []
        for backend in backends:
            log(f"Base OOF fold={fold.fold_id} backend={backend} train={len(train_indices)} valid={len(validation_indices)}")
            if backend == "lightgbm_cpu":
                score = train_lightgbm_predict(
                    x[train_indices],
                    y[train_indices],
                    x[validation_indices],
                    iterations=iterations[backend][fold.fold_id],
                    seed=int(args.seed),
                    threads=int(args.threads),
                )
            elif backend in {"xgboost_gpu", "xgboost_cpu"}:
                backend_device = "cpu" if backend == "xgboost_cpu" else device
                score = train_xgboost_predict(
                    x[train_indices],
                    y[train_indices],
                    x[validation_indices],
                    iterations=iterations[backend][fold.fold_id],
                    seed=int(args.seed),
                    threads=int(args.xgboost_threads),
                    device=backend_device,
                )
            else:
                raise ValueError(f"Unsupported base backend: {backend}")
            raw_vectors.append(score)
            rank_vectors.append(datewise_percentile_rank(score, dates))
            used_backends.append(backend)
        ensemble_raw = np.nanmean(np.vstack(raw_vectors), axis=0)
        ensemble_rank = np.nanmean(np.vstack(rank_vectors), axis=0)
        for local_index, global_index in enumerate(validation_indices):
            records.append(
                {
                    "row_index": int(global_index),
                    "source_row_id": int(frame.at[global_index, "source_row_id"]),
                    "fold_id": int(fold.fold_id),
                    "base_score_raw": float(ensemble_raw[local_index]),
                    "base_rank": float(ensemble_rank[local_index]),
                    "base_model_count": len(used_backends),
                    "base_backends": "|".join(used_backends),
                }
            )
    return pd.DataFrame.from_records(records)


def load_or_build_base_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    fold_index: Mapping[int, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str]:
    cache_path = Path(args.output) / "base_oof_predictions.csv"
    if args.resume and cache_path.exists():
        cached = pd.read_csv(cache_path)
        required = {"row_index", "fold_id", "base_score_raw", "base_rank"}
        if required.issubset(cached.columns):
            return cached, "v9_cache"
    if args.base_oof is not None:
        result = load_external_base_oof(Path(args.base_oof), frame)
        return result, "external"
    # Prefer the strongest available forward OOF score. V6's frozen candidate
    # prediction is usually a stronger error-stratification base than the V8
    # full-model baseline. The map still works without it and then falls back
    # to V8 or a clean rebuild.
    v6_candidate = Path(args.package_root).resolve() / "outputs" / "surge_precision70_v6" / "precision_candidate_predictions.npz"
    if v6_candidate.exists():
        with contextlib.suppress(Exception):
            result = load_external_base_oof(v6_candidate, frame)
            if "fold_id" in result.columns:
                return result, "v6_precision_candidate_oof"
    discovered = discover_v8_baseline_oof(Path(args.v8_dir), frame, folds, fold_index, args)
    if discovered is not None:
        return discovered, "v8_prediction_cache"
    return build_base_oof(frame, features, folds, fold_index, args), "rebuilt"


def attach_oof_metadata(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    oof = oof.copy()
    required = {"row_index", "fold_id", "base_score_raw", "base_rank"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"Base OOF missing columns: {sorted(missing)}")
    oof["row_index"] = pd.to_numeric(oof["row_index"], errors="raise").astype(np.int64)
    if oof["row_index"].duplicated().any():
        raise ValueError("Base OOF row_index duplicated")
    metadata_columns = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.market_column,
        args.bucket_column,
        args.industry_column,
        args.target_column,
    ]
    metadata = frame.loc[oof["row_index"].to_numpy(dtype=np.int64), metadata_columns].reset_index(drop=True)
    oof = oof.reset_index(drop=True)
    for column in metadata_columns:
        oof[column] = metadata[column].to_numpy()
    oof["fold_role"] = oof["fold_id"].map(lambda fold_id: role_for_fold(int(fold_id), roles))
    error_codes = np.zeros(len(oof), dtype=np.int8)
    reranked = np.full(len(oof), np.nan, dtype=np.float64)
    for fold_id, part in oof.groupby("fold_id", sort=True):
        indices = part.index.to_numpy(dtype=np.int64)
        codes, ranks = build_error_group_codes(
            part[args.target_column].to_numpy(dtype=np.int8),
            part["base_rank"].to_numpy(dtype=np.float64),
            part[args.date_column].to_numpy(dtype="datetime64[ns]"),
            top_quantile=float(args.error_top_quantile),
            low_quantile=float(args.error_low_quantile),
        )
        error_codes[indices] = codes
        reranked[indices] = ranks
    oof["error_code"] = error_codes
    oof["error_group"] = pd.Series(error_codes).map(ERROR_GROUP_NAMES).to_numpy()
    oof["base_rank"] = reranked
    return oof


def build_match_tables(oof: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    ab_parts: list[pd.DataFrame] = []
    cd_parts: list[pd.DataFrame] = []
    for fold_id, part in oof.groupby("fold_id", sort=True):
        ab = build_matched_pairs(
            part,
            int(fold_id),
            "AB",
            ERROR_A_TOP_TP,
            ERROR_B_TOP_FP,
            controls_per_case=int(args.controls_per_case),
            date_column=args.date_column,
            score_column="base_rank",
            market_column=args.market_column,
            bucket_column=args.bucket_column,
            industry_column=args.industry_column,
        )
        cd = build_matched_pairs(
            part,
            int(fold_id),
            "CD",
            ERROR_C_LOW_TP,
            ERROR_D_LOW_TN,
            controls_per_case=int(args.controls_per_case),
            date_column=args.date_column,
            score_column="base_rank",
            market_column=args.market_column,
            bucket_column=args.bucket_column,
            industry_column=args.industry_column,
        )
        ab_parts.append(ab)
        cd_parts.append(cd)
    ab_pairs = pd.concat(ab_parts, ignore_index=True) if ab_parts else pd.DataFrame()
    cd_pairs = pd.concat(cd_parts, ignore_index=True) if cd_parts else pd.DataFrame()
    return ab_pairs, cd_pairs


def load_primary_clusters(v8_dir: Path, features: Sequence[str], threshold: float) -> tuple[dict[str, list[str]], pd.DataFrame]:
    path = v8_dir / "correlation_cluster_assignments_v8.csv"
    if not path.exists():
        return {feature: [feature] for feature in features}, pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[np.isclose(pd.to_numeric(frame["threshold"], errors="coerce"), float(threshold))].copy()
    frame = frame[frame["feature"].isin(features)]
    mapping: dict[str, list[str]] = {}
    for _, part in frame.groupby("cluster_id", sort=True):
        members = sorted(part["feature"].astype(str).tolist())
        for feature in members:
            mapping[feature] = members
    for feature in features:
        mapping.setdefault(feature, [feature])
    return mapping, frame


def v8_annotations(v8_dir: Path) -> pd.DataFrame:
    path = v8_dir / "organic_feature_consensus.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature"])
    frame = pd.read_csv(path)
    keep = [
        column
        for column in [
            "feature",
            "consensus_decision",
            "consensus_rank",
            "mean_single_utility",
            "mean_cluster_utility",
            "mean_single_best_precision_utility",
            "mean_neighborhood_k1_utility",
            "mean_neighborhood_k3_utility",
            "mean_neighborhood_k5_utility",
        ]
        if column in frame.columns
    ]
    return frame[keep].drop_duplicates("feature")


def compute_feature_transform_values(
    oof: pd.DataFrame,
    frame: pd.DataFrame,
    feature: str,
    correlation: pd.DataFrame,
    cluster_members: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
    transform_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute only the requested transforms for one feature.

    V9 deliberately avoids the old all-feature x all-transform expansion. Every
    feature receives the cheap Stage-0 audit (raw and missingness); relative and
    innovation transforms are created only after an A/B or C/D prefilter selects
    the source feature in past selection folds.
    """
    requested = set(transform_names or TRANSFORM_TYPES)
    unknown = requested - set(TRANSFORM_TYPES)
    if unknown:
        raise ValueError(f"Unknown transform names for {feature}: {sorted(unknown)}")
    row_indices = oof["row_index"].to_numpy(dtype=np.int64)
    raw = frame.loc[row_indices, feature].to_numpy(dtype=np.float64)
    dates = oof[args.date_column].to_numpy(dtype="datetime64[ns]")
    transforms: dict[str, np.ndarray] = {}
    if "raw" in requested:
        transforms["raw"] = raw
    if "missing_indicator" in requested:
        transforms["missing_indicator"] = (~np.isfinite(raw)).astype(np.float64)
    if "date_rank" in requested:
        transforms["date_rank"] = datewise_percentile_rank(raw, dates)
    if "date_z" in requested:
        transforms["date_z"] = groupwise_robust_z(raw, [dates])
    if "date_market_rank" in requested:
        market = oof[args.market_column].astype(str).to_numpy()
        transforms["date_market_rank"] = groupwise_percentile_rank(raw, [dates, market])
    if "date_bucket_rank" in requested:
        bucket = oof[args.bucket_column].astype(str).to_numpy()
        transforms["date_bucket_rank"] = groupwise_percentile_rank(raw, [dates, bucket])
    if "date_industry_rank" in requested:
        industry = oof[args.industry_column].astype(str).to_numpy()
        transforms["date_industry_rank"] = groupwise_percentile_rank(raw, [dates, industry])
    if "cluster_innovation" in requested:
        peers = [peer for peer in cluster_members.get(feature, []) if peer != feature]
        peer_values = [frame.loc[row_indices, peer].to_numpy(dtype=np.float64) for peer in peers]
        transforms["cluster_innovation"] = cluster_innovation_values(raw, peer_values, dates)
    if "peer_innovation_k3" in requested:
        if feature in correlation.index:
            row = correlation.loc[feature].drop(feature, errors="ignore").sort_values(ascending=False)
            top_peers = [peer for peer in row.index[:3] if peer in frame.columns]
            weighted_peers = [
                (frame.loc[row_indices, peer].to_numpy(dtype=np.float64), float(max(row.loc[peer], 1e-6)))
                for peer in top_peers
            ]
            transforms["peer_innovation_k3"] = peer_weighted_innovation_values(raw, weighted_peers, dates)
        else:
            transforms["peer_innovation_k3"] = np.full(len(raw), np.nan, dtype=np.float64)
    return transforms

def compute_univariate_map(
    oof: pd.DataFrame,
    frame: pd.DataFrame,
    features: Sequence[str],
    correlation: pd.DataFrame,
    cluster_members: Mapping[str, Sequence[str]],
    ab_pairs: pd.DataFrame,
    cd_pairs: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    transform_plan: Mapping[str, Sequence[str]] | None = None,
    stage_name: str = "univariate",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    candidate_values: dict[str, np.ndarray] = {}
    active_features = [feature for feature in features if transform_plan is None or feature in transform_plan]
    fold_contexts = build_fold_evaluation_contexts(oof, ab_pairs, cd_pairs)
    oof_row_indices = oof["row_index"].to_numpy(dtype=np.int64)

    def compute_feature(feature: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        feature_records: list[dict[str, Any]] = []
        feature_values: dict[str, np.ndarray] = {}
        requested = None if transform_plan is None else transform_plan.get(feature, [])
        source_raw_values = frame.loc[oof_row_indices, feature].to_numpy(dtype=np.float64)
        transforms = compute_feature_transform_values(
            oof,
            frame,
            feature,
            correlation,
            cluster_members,
            args,
            transform_names=requested,
        )
        for transform_name, values in transforms.items():
            node_id = f"{transform_name}::{feature}"
            feature_values[node_id] = values.astype(np.float32)
            for context in fold_contexts:
                fold_id = int(context["fold_id"])
                indices = context["indices"]
                local_codes = context["codes"]
                local_values = values[indices]
                for axis, positive_code, negative_code, pairs in [
                    ("AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP, context["ab_pairs"]),
                    ("CD", ERROR_C_LOW_TP, ERROR_D_LOW_TN, context["cd_pairs"]),
                ]:
                    metrics = separation_metrics(local_values, local_codes, positive_code, negative_code, pairs)
                    source_local = source_raw_values[indices]
                    source_positive = source_local[local_codes == int(positive_code)]
                    source_negative = source_local[local_codes == int(negative_code)]
                    source_positive_valid = int(np.isfinite(source_positive).sum())
                    source_negative_valid = int(np.isfinite(source_negative).sum())
                    source_positive_coverage = (
                        float(source_positive_valid / len(source_positive)) if len(source_positive) else float("nan")
                    )
                    source_negative_coverage = (
                        float(source_negative_valid / len(source_negative)) if len(source_negative) else float("nan")
                    )
                    if transform_name == "missing_indicator":
                        effective_positive_valid = source_positive_valid
                        effective_negative_valid = source_negative_valid
                        effective_positive_coverage = source_positive_coverage
                        effective_negative_coverage = source_negative_coverage
                    else:
                        effective_positive_valid = int(metrics["positive_valid_n"])
                        effective_negative_valid = int(metrics["negative_valid_n"])
                        effective_positive_coverage = float(metrics["positive_coverage"])
                        effective_negative_coverage = float(metrics["negative_coverage"])
                    feature_records.append(
                        {
                            "fold_id": int(fold_id),
                            "fold_role": role_for_fold(int(fold_id), roles),
                            "axis": axis,
                            "feature": feature,
                            "transform": transform_name,
                            "node_id": node_id,
                            "map_stage": stage_name,
                            "source_positive_valid_n": source_positive_valid,
                            "source_negative_valid_n": source_negative_valid,
                            "source_positive_coverage": source_positive_coverage,
                            "source_negative_coverage": source_negative_coverage,
                            "effective_positive_valid_n": effective_positive_valid,
                            "effective_negative_valid_n": effective_negative_valid,
                            "effective_positive_coverage": effective_positive_coverage,
                            "effective_negative_coverage": effective_negative_coverage,
                            **metrics,
                        }
                    )
        return feature_records, feature_values

    workers = analysis_workers(args, len(active_features))
    log(f"{stage_name}: analysis workers={workers}")
    if workers == 1:
        results = map(compute_feature, active_features)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-feature")
        results = executor.map(compute_feature, active_features)
    try:
        for feature_index, (feature, result) in enumerate(zip(active_features, results), start=1):
            feature_records, feature_values = result
            records.extend(feature_records)
            candidate_values.update(feature_values)
            if feature_index == 1 or feature_index % 25 == 0 or feature_index == len(active_features):
                log(f"{stage_name} separation map {feature_index}/{len(active_features)}: {feature}")
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return pd.DataFrame.from_records(records), candidate_values

def summarize_univariate_map(
    fold_map: pd.DataFrame,
    v8_meta: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for role_name in ["selection", "confirmation", "recent_audit"]:
        part = fold_map[fold_map["fold_id"].isin(roles.get(role_name, []))]
        if part.empty:
            continue
        summary = aggregate_fold_metrics(part, ["axis", "feature", "transform", "node_id"])
        summary = summary.add_prefix(f"{role_name}__")
        summary.rename(
            columns={
                f"{role_name}__axis": "axis",
                f"{role_name}__feature": "feature",
                f"{role_name}__transform": "transform",
                f"{role_name}__node_id": "node_id",
            },
            inplace=True,
        )
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    merged = summaries[0]
    for summary in summaries[1:]:
        merged = merged.merge(summary, on=["axis", "feature", "transform", "node_id"], how="outer")
    if not v8_meta.empty:
        merged = merged.merge(v8_meta, on="feature", how="left")
    positive_coverage_column = (
        "selection__mean_effective_positive_coverage"
        if "selection__mean_effective_positive_coverage" in merged.columns
        else "selection__mean_positive_coverage"
    )
    negative_coverage_column = (
        "selection__mean_effective_negative_coverage"
        if "selection__mean_effective_negative_coverage" in merged.columns
        else "selection__mean_negative_coverage"
    )
    coverage = np.minimum(
        pd.to_numeric(merged[positive_coverage_column], errors="coerce").fillna(0.0),
        pd.to_numeric(merged[negative_coverage_column], errors="coerce").fillna(0.0),
    )
    auc_edge = np.clip(
        (pd.to_numeric(merged.get("selection__mean_oriented_auc"), errors="coerce").fillna(0.5) - 0.5) * 2.0,
        0.0,
        1.0,
    )
    matched_edge = np.clip(
        (pd.to_numeric(merged.get("selection__mean_matched_concordance"), errors="coerce").fillna(0.5) - 0.5) * 2.0,
        0.0,
        1.0,
    )
    worst_edge = np.clip(
        (pd.to_numeric(merged.get("selection__min_oriented_auc"), errors="coerce").fillna(0.5) - 0.5) * 2.0,
        0.0,
        1.0,
    )
    ks = pd.to_numeric(merged.get("selection__mean_ks"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    js = pd.to_numeric(merged.get("selection__mean_js_divergence"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    orientation_stability = pd.to_numeric(merged.get("selection__orientation_consistency"), errors="coerce").fillna(0.0)
    matched_stability = pd.to_numeric(merged.get("selection__matched_orientation_consistency"), errors="coerce").fillna(0.0)
    sparse_penalty = np.sqrt(np.clip(coverage, 0.0, 1.0))
    # Candidate ranking is selection-only. Confirmation and recent metrics remain
    # diagnostic columns and must never influence feature selection.
    merged["separation_score"] = (
        0.30 * auc_edge
        + 0.24 * matched_edge
        + 0.18 * worst_edge
        + 0.10 * ks
        + 0.05 * js
        + 0.08 * orientation_stability
        + 0.05 * matched_stability
    ) * sparse_penalty
    merged["coverage_gate"] = coverage >= float(args.minimum_coverage)
    positive_valid_column = (
        "selection__min_effective_positive_valid_n"
        if "selection__min_effective_positive_valid_n" in merged.columns
        else "selection__min_positive_valid_n"
    )
    negative_valid_column = (
        "selection__min_effective_negative_valid_n"
        if "selection__min_effective_negative_valid_n" in merged.columns
        else "selection__min_negative_valid_n"
    )
    merged["sample_gate"] = (
        pd.to_numeric(merged[positive_valid_column], errors="coerce").fillna(0)
        >= int(args.minimum_positive_rows)
    ) & (
        pd.to_numeric(merged[negative_valid_column], errors="coerce").fillna(0)
        >= int(args.minimum_negative_rows)
    )
    merged["stability_gate"] = (
        pd.to_numeric(merged.get("selection__orientation_consistency"), errors="coerce").fillna(0.0)
        >= float(args.minimum_orientation_consistency)
    )
    merged["separation_gate"] = merged["coverage_gate"] & merged["sample_gate"] & merged["stability_gate"]
    merged.sort_values(["axis", "separation_gate", "separation_score", "node_id"], ascending=[True, False, False, True], inplace=True)
    merged["axis_rank"] = merged.groupby("axis", sort=False).cumcount() + 1
    return merged


def compute_horizon_map(
    oof: pd.DataFrame,
    frame: pd.DataFrame,
    horizon_pairs: pd.DataFrame,
    ab_pairs: pd.DataFrame,
    cd_pairs: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if horizon_pairs.empty:
        return pd.DataFrame(), {}
    row_indices = oof["row_index"].to_numpy(dtype=np.int64)
    dates = oof[args.date_column].to_numpy(dtype="datetime64[ns]")
    records: list[dict[str, Any]] = []
    values_by_name: dict[str, np.ndarray] = {}
    fold_contexts = build_fold_evaluation_contexts(oof, ab_pairs, cd_pairs)
    pair_rows = list(horizon_pairs.itertuples(index=False))

    def compute_pair(row: Any) -> tuple[list[dict[str, Any]], str, np.ndarray]:
        pair_records: list[dict[str, Any]] = []
        short_values = frame.loc[row_indices, row.short_feature].to_numpy(dtype=np.float64)
        long_values = frame.loc[row_indices, row.long_feature].to_numpy(dtype=np.float64)
        values = horizon_rank_spread(short_values, long_values, dates)
        for context in fold_contexts:
            fold_id = int(context["fold_id"])
            indices = context["indices"]
            local_values = values[indices]
            local_codes = context["codes"]
            for axis, positive_code, negative_code, pairs in [
                ("AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP, context["ab_pairs"]),
                ("CD", ERROR_C_LOW_TP, ERROR_D_LOW_TN, context["cd_pairs"]),
            ]:
                metrics = separation_metrics(local_values, local_codes, positive_code, negative_code, pairs)
                pair_records.append(
                    {
                        "fold_id": int(fold_id),
                        "fold_role": role_for_fold(int(fold_id), roles),
                        "axis": axis,
                        "stem": row.stem,
                        "short_horizon": int(row.short_horizon),
                        "long_horizon": int(row.long_horizon),
                        "short_feature": row.short_feature,
                        "long_feature": row.long_feature,
                        "transform_name": row.transform_name,
                        **metrics,
                    }
                )
        return pair_records, str(row.transform_name), values.astype(np.float32)

    workers = analysis_workers(args, len(pair_rows))
    log(f"Horizon map: analysis workers={workers}")
    if workers == 1:
        results = map(compute_pair, pair_rows)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-horizon")
        results = executor.map(compute_pair, pair_rows)
    try:
        for pair_index, result in enumerate(results, start=1):
            pair_records, transform_name, values = result
            records.extend(pair_records)
            values_by_name[transform_name] = values
            if pair_index == 1 or pair_index % 50 == 0 or pair_index == len(pair_rows):
                log(f"Horizon separation map {pair_index}/{len(pair_rows)}")
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return pd.DataFrame.from_records(records), values_by_name


def summarize_horizon_map(fold_map: pd.DataFrame, roles: Mapping[str, Sequence[int]], args: argparse.Namespace) -> pd.DataFrame:
    if fold_map.empty:
        return pd.DataFrame()
    summaries: list[pd.DataFrame] = []
    keys = ["axis", "stem", "short_horizon", "long_horizon", "short_feature", "long_feature", "transform_name"]
    for role in ["selection", "confirmation", "recent_audit"]:
        part = fold_map[fold_map["fold_id"].isin(roles.get(role, []))]
        summary = aggregate_fold_metrics(part, keys).add_prefix(f"{role}__")
        summary.rename(columns={f"{role}__{key}": key for key in keys}, inplace=True)
        summaries.append(summary)
    merged = summaries[0]
    for summary in summaries[1:]:
        merged = merged.merge(summary, on=keys, how="outer")
    coverage = np.minimum(
        pd.to_numeric(merged["selection__mean_positive_coverage"], errors="coerce").fillna(0),
        pd.to_numeric(merged["selection__mean_negative_coverage"], errors="coerce").fillna(0),
    )
    auc_edge = np.clip((pd.to_numeric(merged["selection__mean_oriented_auc"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    matched_edge = np.clip((pd.to_numeric(merged["selection__mean_matched_concordance"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    worst_edge = np.clip((pd.to_numeric(merged["selection__min_oriented_auc"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    merged["separation_score"] = (0.45 * auc_edge + 0.35 * matched_edge + 0.20 * worst_edge) * np.sqrt(coverage)
    merged["separation_gate"] = (
        coverage >= float(args.minimum_coverage)
    ) & (
        pd.to_numeric(merged["selection__min_positive_valid_n"], errors="coerce").fillna(0) >= int(args.minimum_positive_rows)
    ) & (
        pd.to_numeric(merged["selection__min_negative_valid_n"], errors="coerce").fillna(0) >= int(args.minimum_negative_rows)
    )
    merged.sort_values(["axis", "separation_gate", "separation_score"], ascending=[True, False, False], inplace=True)
    merged["axis_rank"] = merged.groupby("axis", sort=False).cumcount() + 1
    return merged


def build_rank_matrix(oof: pd.DataFrame, frame: pd.DataFrame, features: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    row_indices = oof["row_index"].to_numpy(dtype=np.int64)
    dates = oof[args.date_column].to_numpy(dtype="datetime64[ns]")
    matrix = np.empty((len(oof), len(features)), dtype=np.float32)
    def rank_feature(feature: str) -> np.ndarray:
        values = frame.loc[row_indices, feature].to_numpy(dtype=np.float64)
        return datewise_percentile_rank(values, dates).astype(np.float32)

    workers = analysis_workers(args, len(features))
    log(f"OOF rank matrix: analysis workers={workers}")
    if workers == 1:
        results = map(rank_feature, features)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-rank")
        results = executor.map(rank_feature, features)
    try:
        for feature_index, values in enumerate(results):
            if feature_index == 0 or (feature_index + 1) % 50 == 0:
                log(f"Building OOF date-rank matrix {feature_index + 1}/{len(features)}")
            matrix[:, feature_index] = values
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return matrix


def compute_interaction_map(
    oof: pd.DataFrame,
    rank_matrix: np.ndarray,
    features: Sequence[str],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for fold_id, fold_part in oof.groupby("fold_id", sort=True):
        indices = fold_part.index.to_numpy(dtype=np.int64)
        local_matrix = rank_matrix[indices]
        local_codes = fold_part["error_code"].to_numpy(dtype=np.int8)
        for axis, positive_code, negative_code in [
            ("AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP),
            ("CD", ERROR_C_LOW_TP, ERROR_D_LOW_TN),
        ]:
            screen = compute_interaction_moment_screen(
                local_matrix,
                features,
                local_codes,
                positive_code,
                negative_code,
                minimum_valid_rows=max(10, int(args.minimum_positive_rows)),
            )
            if screen.empty:
                continue
            screen["fold_id"] = int(fold_id)
            screen["fold_role"] = role_for_fold(int(fold_id), roles)
            screen["axis"] = axis
            screen["interaction_sign"] = np.sign(screen["delta_covariance"]).astype(int)
            screen.sort_values("interaction_excess_score", ascending=False, inplace=True)
            parts.append(screen.head(int(args.interaction_top_per_fold)))
    fold_map = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if fold_map.empty:
        return fold_map, pd.DataFrame()
    selection = fold_map[fold_map["fold_id"].isin(roles["selection"])]
    keys = ["axis", "feature_a", "feature_b"]
    summary = aggregate_fold_metrics(selection, keys)
    sign_frame = (
        selection.groupby(keys, sort=True)["interaction_sign"]
        .apply(lambda series: max(float(np.mean(series > 0)), float(np.mean(series < 0))) if len(series) else np.nan)
        .rename("sign_consistency")
        .reset_index()
    )
    summary = summary.merge(sign_frame, on=keys, how="left")
    summary["interaction_score"] = (
        pd.to_numeric(summary["mean_interaction_excess_score"], errors="coerce").fillna(0).clip(lower=0)
        * pd.to_numeric(summary["sign_consistency"], errors="coerce").fillna(0)
        * np.sqrt(pd.to_numeric(summary["fold_count"], errors="coerce").fillna(0) / max(1, len(roles["selection"])))
    )
    summary.sort_values(["axis", "interaction_score", "feature_a", "feature_b"], ascending=[True, False, True, True], inplace=True)
    summary["axis_rank"] = summary.groupby("axis", sort=False).cumcount() + 1
    return fold_map, summary



def _interaction_pair_key(axis: str, feature_a: str, feature_b: str) -> str:
    first, second = sorted([str(feature_a), str(feature_b)])
    return f"{axis}::{first}::{second}"


def rank_prior_interaction_pairs(
    interaction_fold: pd.DataFrame,
    prior_folds: Sequence[int],
    pair_count: int,
    minimum_fold_presence: float,
) -> pd.DataFrame:
    part = interaction_fold[interaction_fold["fold_id"].isin([int(value) for value in prior_folds])].copy()
    if part.empty or not prior_folds:
        return pd.DataFrame(columns=["axis", "feature_a", "feature_b", "pair_key", "score"])
    records: list[dict[str, Any]] = []
    for (axis, feature_a, feature_b), group in part.groupby(["axis", "feature_a", "feature_b"], sort=True):
        values = pd.to_numeric(group["interaction_excess_score"], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        signs = pd.to_numeric(group["interaction_sign"], errors="coerce").to_numpy(dtype=np.float64)
        signs = signs[np.isfinite(signs) & (signs != 0)]
        observed = int(group["fold_id"].nunique())
        presence = float(observed / max(1, len(prior_folds)))
        sign_consistency = max(float(np.mean(signs > 0)), float(np.mean(signs < 0))) if len(signs) else 0.0
        if len(values) == 0 or presence < float(minimum_fold_presence):
            continue
        score = float(np.mean(np.clip(values, 0.0, None))) * sign_consistency * math.sqrt(presence)
        records.append(
            {
                "axis": str(axis),
                "feature_a": str(feature_a),
                "feature_b": str(feature_b),
                "pair_key": _interaction_pair_key(str(axis), str(feature_a), str(feature_b)),
                "observed_fold_count": observed,
                "fold_presence": presence,
                "sign_consistency": sign_consistency,
                "score": score,
            }
        )
    ranked = pd.DataFrame.from_records(records)
    if ranked.empty:
        return ranked
    ranked.sort_values(["axis", "score", "feature_a", "feature_b"], ascending=[True, False, True, True], inplace=True)
    return ranked.groupby("axis", sort=False, group_keys=False).head(int(pair_count)).reset_index(drop=True)


def build_forward_interaction_prefilter(
    interaction_fold: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selection_folds = sorted(int(value) for value in roles["selection"])
    pair_keys_by_fold: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    union_parts: list[pd.DataFrame] = []
    for fold_id in selection_folds:
        prior = [value for value in selection_folds if value < fold_id]
        ranked = rank_prior_interaction_pairs(
            interaction_fold,
            prior,
            int(args.interaction_prefilter_count),
            float(args.minimum_interaction_fold_presence),
        )
        pair_keys_by_fold[str(fold_id)] = ranked.get("pair_key", pd.Series(dtype=str)).astype(str).tolist()
        if not ranked.empty:
            ranked = ranked.copy()
            ranked["probe_fold"] = str(fold_id)
            ranked["prior_fold_ids"] = "|".join(map(str, prior))
            rows.extend(ranked.to_dict(orient="records"))
            union_parts.append(ranked[["axis", "feature_a", "feature_b", "pair_key"]])
    global_ranked = rank_prior_interaction_pairs(
        interaction_fold,
        selection_folds,
        int(args.interaction_validate_count),
        float(args.minimum_interaction_fold_presence),
    )
    pair_keys_by_fold["GLOBAL_SELECTION"] = global_ranked.get("pair_key", pd.Series(dtype=str)).astype(str).tolist()
    if not global_ranked.empty:
        global_copy = global_ranked.copy()
        global_copy["probe_fold"] = "GLOBAL_SELECTION"
        global_copy["prior_fold_ids"] = "|".join(map(str, selection_folds))
        rows.extend(global_copy.to_dict(orient="records"))
        union_parts.append(global_ranked[["axis", "feature_a", "feature_b", "pair_key"]])
    union = (
        pd.concat(union_parts, ignore_index=True).drop_duplicates(["axis", "feature_a", "feature_b"])
        if union_parts
        else pd.DataFrame(columns=["axis", "feature_a", "feature_b", "pair_key"])
    )
    manifest = {
        "schema": "surge_separation_forward_interaction_manifest_v9",
        "generated_at": utc_now(),
        "pair_keys_by_probe_fold": pair_keys_by_fold,
        "selection_rows": rows,
        "validated_pair_union_count": int(len(union)),
        "interaction_prefilter_count_per_axis": int(args.interaction_prefilter_count),
        "global_validate_count_per_axis": int(args.interaction_validate_count),
        "minimum_fold_presence": float(args.minimum_interaction_fold_presence),
    }
    return manifest, union

def validate_pair_transforms(
    oof: pd.DataFrame,
    rank_matrix: np.ndarray,
    features: Sequence[str],
    selected_pairs: pd.DataFrame,
    ab_pairs: pd.DataFrame,
    cd_pairs: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if selected_pairs.empty:
        return pd.DataFrame(), {}
    feature_index = {feature: index for index, feature in enumerate(features)}
    selected = selected_pairs.drop_duplicates(["axis", "feature_a", "feature_b"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    values_by_name: dict[str, np.ndarray] = {}
    dates = oof[args.date_column].to_numpy(dtype="datetime64[ns]")
    fold_contexts = build_fold_evaluation_contexts(oof, ab_pairs, cd_pairs)
    pair_rows = list(selected.itertuples(index=False))

    def compute_pair(row: Any) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        pair_records: list[dict[str, Any]] = []
        pair_values: dict[str, np.ndarray] = {}
        values_a = rank_matrix[:, feature_index[row.feature_a]].astype(np.float64)
        values_b = rank_matrix[:, feature_index[row.feature_b]].astype(np.float64)
        transforms = pair_transform_values(values_a, values_b, dates)
        for transform_name, values in transforms.items():
            node_id = f"PAIR::{transform_name}::{row.feature_a}::{row.feature_b}"
            pair_values[node_id] = values.astype(np.float32)
            for context in fold_contexts:
                fold_id = int(context["fold_id"])
                indices = context["indices"]
                local_codes = context["codes"]
                local_values = values[indices]
                fold_pairs = context["ab_pairs"] if row.axis == "AB" else context["cd_pairs"]
                positive_code = ERROR_A_TOP_TP if row.axis == "AB" else ERROR_C_LOW_TP
                negative_code = ERROR_B_TOP_FP if row.axis == "AB" else ERROR_D_LOW_TN
                metrics = separation_metrics(local_values, local_codes, positive_code, negative_code, fold_pairs)
                pair_records.append(
                    {
                        "fold_id": int(fold_id),
                        "fold_role": role_for_fold(int(fold_id), roles),
                        "axis": row.axis,
                        "feature_a": row.feature_a,
                        "feature_b": row.feature_b,
                        "pair_transform": transform_name,
                        "node_id": node_id,
                        **metrics,
                    }
                )
        return pair_records, pair_values

    workers = max(1, min(int(getattr(args, "pair_workers", 1)), len(pair_rows)))
    log(f"Pair validation: process workers={workers}")
    if workers == 1:
        results = map(compute_pair, pair_rows)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_pair_process_worker,
            initargs=(rank_matrix, feature_index, dates, fold_contexts, roles),
        )
        results = executor.map(compute_pair_process_worker, [row._asdict() for row in pair_rows], chunksize=1)
    try:
        for row_index, result in enumerate(results, start=1):
            pair_records, pair_values = result
            records.extend(pair_records)
            values_by_name.update(pair_values)
            if row_index == 1 or row_index % 50 == 0 or row_index == len(pair_rows):
                log(f"Pair interaction validation {row_index}/{len(pair_rows)}")
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return pd.DataFrame.from_records(records), values_by_name


def summarize_pair_validation(fold_map: pd.DataFrame, roles: Mapping[str, Sequence[int]], args: argparse.Namespace) -> pd.DataFrame:
    if fold_map.empty:
        return pd.DataFrame()
    keys = ["axis", "feature_a", "feature_b", "pair_transform", "node_id"]
    summaries: list[pd.DataFrame] = []
    for role in ["selection", "confirmation", "recent_audit"]:
        part = fold_map[fold_map["fold_id"].isin(roles.get(role, []))]
        summary = aggregate_fold_metrics(part, keys).add_prefix(f"{role}__")
        summary.rename(columns={f"{role}__{key}": key for key in keys}, inplace=True)
        summaries.append(summary)
    merged = summaries[0]
    for summary in summaries[1:]:
        merged = merged.merge(summary, on=keys, how="outer")
    coverage = np.minimum(
        pd.to_numeric(merged["selection__mean_positive_coverage"], errors="coerce").fillna(0),
        pd.to_numeric(merged["selection__mean_negative_coverage"], errors="coerce").fillna(0),
    )
    auc_edge = np.clip((pd.to_numeric(merged["selection__mean_oriented_auc"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    matched_edge = np.clip((pd.to_numeric(merged["selection__mean_matched_concordance"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    worst_edge = np.clip((pd.to_numeric(merged["selection__min_oriented_auc"], errors="coerce").fillna(0.5) - 0.5) * 2.0, 0, 1)
    merged["separation_score"] = (0.45 * auc_edge + 0.35 * matched_edge + 0.20 * worst_edge) * np.sqrt(coverage)
    merged["separation_gate"] = (
        coverage >= float(args.minimum_coverage)
    ) & (
        pd.to_numeric(merged["selection__min_positive_valid_n"], errors="coerce").fillna(0) >= int(args.minimum_positive_rows)
    ) & (
        pd.to_numeric(merged["selection__min_negative_valid_n"], errors="coerce").fillna(0) >= int(args.minimum_negative_rows)
    )
    merged.sort_values(["axis", "separation_gate", "separation_score"], ascending=[True, False, False], inplace=True)
    merged["axis_rank"] = merged.groupby("axis", sort=False).cumcount() + 1
    return merged


def candidate_manifest(
    univariate_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    staged_transform_manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates: dict[str, list[str]] = {"AB": [], "CD": [], "INTERACTION_AB": [], "INTERACTION_CD": []}
    global_axis_features = staged_transform_manifest.get("features_by_probe_fold_and_axis", {}).get("GLOBAL_SELECTION", {})
    for axis in ["AB", "CD"]:
        allowed = set(str(value) for value in global_axis_features.get(axis, []))
        part = univariate_summary[(univariate_summary["axis"].eq(axis)) & (univariate_summary["separation_gate"])].copy()
        if allowed:
            part = part[part["feature"].isin(allowed)]
        else:
            part = part.iloc[0:0]
        # Preserve transform diversity and prevent one raw feature from occupying every slot.
        selected: list[str] = []
        source_counts: dict[str, int] = defaultdict(int)
        for row in part.sort_values("separation_score", ascending=False).itertuples(index=False):
            if source_counts[str(row.feature)] >= int(args.max_transforms_per_feature):
                continue
            selected.append(str(row.node_id))
            source_counts[str(row.feature)] += 1
            if len(selected) >= int(args.axis_candidate_count):
                break
        if not horizon_summary.empty:
            horizon = horizon_summary[(horizon_summary["axis"].eq(axis)) & (horizon_summary["separation_gate"])]
            selected.extend(horizon.head(int(args.horizon_candidate_count))["transform_name"].astype(str).tolist())
        candidates[axis] = list(dict.fromkeys(selected))
    if not pair_summary.empty:
        for axis in ["AB", "CD"]:
            part = pair_summary[(pair_summary["axis"].eq(axis)) & (pair_summary["separation_gate"])]
            candidates[f"INTERACTION_{axis}"] = part.head(int(args.pair_candidate_count))["node_id"].astype(str).tolist()
    return {
        "schema": "surge_separation_candidate_manifest_v9",
        "generated_at": utc_now(),
        "analysis_universe": "all input features",
        "target_precision": float(args.target_precision),
        "minimum_alerts": int(args.minimum_alerts),
        "candidates": candidates,
    }


def build_candidate_matrix(
    oof: pd.DataFrame,
    univariate_values: Mapping[str, np.ndarray],
    horizon_values: Mapping[str, np.ndarray],
    pair_values: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    columns = ["base_rank"]
    matrix = pd.DataFrame({"base_rank": pd.to_numeric(oof["base_rank"], errors="coerce").to_numpy(dtype=np.float64)})
    all_values: dict[str, np.ndarray] = {}
    all_values.update(univariate_values)
    all_values.update(horizon_values)
    all_values.update(pair_values)
    requested: list[str] = []
    for values in manifest.get("candidates", {}).values():
        requested.extend([str(value) for value in values])
    for node_id in dict.fromkeys(requested):
        if node_id not in all_values:
            continue
        safe_name = f"f_{len(columns):03d}"
        matrix[safe_name] = np.asarray(all_values[node_id], dtype=np.float32)
        columns.append(safe_name)
        matrix.attrs[safe_name] = node_id
    return matrix, columns


def score_prior_univariate_candidates(
    fold_map: pd.DataFrame,
    prior_folds: Sequence[int],
    axis_candidate_count: int,
    max_transforms_per_feature: int,
    minimum_positive_rows: int,
    minimum_negative_rows: int,
    minimum_coverage: float,
    minimum_orientation_consistency: float,
    allowed_features: Sequence[str] | None = None,
) -> list[str]:
    part = fold_map[fold_map["fold_id"].isin([int(value) for value in prior_folds])].copy()
    if allowed_features is not None:
        part = part[part["feature"].isin(set(str(value) for value in allowed_features))]
    if part.empty:
        return []
    records: list[dict[str, Any]] = []
    for keys, group in part.groupby(["axis", "feature", "transform", "node_id"], sort=True):
        axis, feature, transform, node_id = keys
        auc = pd.to_numeric(group["oriented_auc"], errors="coerce").to_numpy(dtype=np.float64)
        matched = pd.to_numeric(group["matched_concordance"], errors="coerce").to_numpy(dtype=np.float64)
        positive_coverage_name = "effective_positive_coverage" if "effective_positive_coverage" in group.columns else "positive_coverage"
        negative_coverage_name = "effective_negative_coverage" if "effective_negative_coverage" in group.columns else "negative_coverage"
        coverage = np.minimum(
            pd.to_numeric(group[positive_coverage_name], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(group[negative_coverage_name], errors="coerce").to_numpy(dtype=np.float64),
        )
        auc = auc[np.isfinite(auc)]
        matched = matched[np.isfinite(matched)]
        coverage = coverage[np.isfinite(coverage)]
        if len(auc) == 0:
            continue
        orientation = pd.to_numeric(group["orientation"], errors="coerce").to_numpy(dtype=np.float64)
        orientation = orientation[np.isfinite(orientation) & (orientation != 0)]
        stability = max(float(np.mean(orientation > 0)), float(np.mean(orientation < 0))) if len(orientation) else 0.0
        positive_valid_name = "effective_positive_valid_n" if "effective_positive_valid_n" in group.columns else "positive_valid_n"
        negative_valid_name = "effective_negative_valid_n" if "effective_negative_valid_n" in group.columns else "negative_valid_n"
        positive_n = pd.to_numeric(group[positive_valid_name], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        negative_n = pd.to_numeric(group[negative_valid_name], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        minimum_group_coverage = float(np.min(coverage)) if len(coverage) else 0.0
        if (
            len(group) < len(prior_folds)
            or len(positive_n) == 0
            or len(negative_n) == 0
            or float(np.min(positive_n)) < int(minimum_positive_rows)
            or float(np.min(negative_n)) < int(minimum_negative_rows)
            or minimum_group_coverage < float(minimum_coverage)
            or stability < float(minimum_orientation_consistency)
        ):
            continue
        score = (
            0.45 * max(0.0, (float(np.mean(auc)) - 0.5) * 2.0)
            + 0.25 * max(0.0, (float(np.min(auc)) - 0.5) * 2.0)
        )
        if len(matched):
            score += 0.20 * max(0.0, (float(np.mean(matched)) - 0.5) * 2.0)
        score += 0.10 * stability
        score *= math.sqrt(max(0.0, float(np.mean(coverage))) if len(coverage) else 0.0)
        records.append(
            {
                "axis": axis,
                "feature": feature,
                "transform": transform,
                "node_id": node_id,
                "score": score,
            }
        )
    ranking = pd.DataFrame.from_records(records)
    selected: list[str] = []
    for axis in ["AB", "CD"]:
        axis_part = ranking[ranking["axis"].eq(axis)].sort_values("score", ascending=False)
        source_counts: dict[str, int] = defaultdict(int)
        axis_selected: list[str] = []
        for row in axis_part.itertuples(index=False):
            if source_counts[str(row.feature)] >= int(max_transforms_per_feature):
                continue
            axis_selected.append(str(row.node_id))
            source_counts[str(row.feature)] += 1
            if len(axis_selected) >= int(axis_candidate_count):
                break
        selected.extend(axis_selected)
    return list(dict.fromkeys(selected))


def score_prior_horizon_candidates(
    fold_map: pd.DataFrame,
    prior_folds: Sequence[int],
    candidate_count: int,
    minimum_positive_rows: int,
    minimum_negative_rows: int,
    minimum_coverage: float,
    minimum_orientation_consistency: float,
) -> list[str]:
    part = fold_map[fold_map["fold_id"].isin([int(value) for value in prior_folds])].copy()
    if part.empty:
        return []
    records: list[dict[str, Any]] = []
    keys = ["axis", "transform_name"]
    for (axis, transform_name), group in part.groupby(keys, sort=True):
        auc = pd.to_numeric(group["oriented_auc"], errors="coerce").to_numpy(dtype=np.float64)
        matched = pd.to_numeric(group["matched_concordance"], errors="coerce").to_numpy(dtype=np.float64)
        auc = auc[np.isfinite(auc)]
        matched = matched[np.isfinite(matched)]
        if len(auc) == 0:
            continue
        orientation = pd.to_numeric(group["orientation"], errors="coerce").to_numpy(dtype=np.float64)
        orientation = orientation[np.isfinite(orientation) & (orientation != 0)]
        stability = max(float(np.mean(orientation > 0)), float(np.mean(orientation < 0))) if len(orientation) else 0.0
        positive_n = pd.to_numeric(group["positive_valid_n"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        negative_n = pd.to_numeric(group["negative_valid_n"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        coverage = np.minimum(
            pd.to_numeric(group["positive_coverage"], errors="coerce").fillna(0).to_numpy(dtype=np.float64),
            pd.to_numeric(group["negative_coverage"], errors="coerce").fillna(0).to_numpy(dtype=np.float64),
        )
        if (
            len(group) < len(prior_folds)
            or len(positive_n) == 0
            or len(negative_n) == 0
            or float(np.min(positive_n)) < int(minimum_positive_rows)
            or float(np.min(negative_n)) < int(minimum_negative_rows)
            or float(np.min(coverage)) < float(minimum_coverage)
            or stability < float(minimum_orientation_consistency)
        ):
            continue
        score = 0.65 * max(0.0, (float(np.mean(auc)) - 0.5) * 2.0)
        score += 0.25 * max(0.0, (float(np.min(auc)) - 0.5) * 2.0)
        score += 0.10 * max(0.0, (float(np.mean(matched)) - 0.5) * 2.0) if len(matched) else 0.0
        records.append({"axis": axis, "transform_name": transform_name, "score": score})
    ranking = pd.DataFrame.from_records(records)
    selected: list[str] = []
    for axis in ["AB", "CD"]:
        selected.extend(
            ranking[ranking["axis"].eq(axis)]
            .sort_values("score", ascending=False)
            .head(int(candidate_count))["transform_name"]
            .astype(str)
            .tolist()
        )
    return list(dict.fromkeys(selected))



def _feature_from_node_id(node_id: str) -> str:
    value = str(node_id)
    return value.split("::", 1)[1] if "::" in value else value


def build_staged_transform_prefilter(
    stage0_fold: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    """Select 12-24 A/B and C/D source features before relative expansion.

    The union contains the global selection shortlist plus every forward-only
    shortlist needed by a selection probe fold. Availability for fold k is still
    restricted to the shortlist learned from folds < k, so precomputing the union
    does not leak future membership into that fold.
    """
    selection_folds = sorted(int(value) for value in roles["selection"])
    rows: list[dict[str, Any]] = []
    features_by_fold: dict[str, list[str]] = {}
    features_by_fold_axis: dict[str, dict[str, list[str]]] = {}

    def select(prior_folds: Sequence[int], label: str) -> tuple[list[str], dict[str, list[str]]]:
        axis_features: dict[str, list[str]] = {}
        union: list[str] = []
        for axis in ["AB", "CD"]:
            nodes = score_prior_univariate_candidates(
                stage0_fold[stage0_fold["axis"].eq(axis)],
                prior_folds,
                int(args.prefilter_feature_count),
                1,
                int(args.minimum_positive_rows),
                int(args.minimum_negative_rows),
                float(args.minimum_coverage),
                float(args.minimum_orientation_consistency),
            )
            selected_features = list(dict.fromkeys(_feature_from_node_id(node) for node in nodes))
            selected_features = selected_features[: int(args.prefilter_feature_count)]
            axis_features[axis] = selected_features
            union.extend(selected_features)
            for rank, (node, feature) in enumerate(zip(nodes, selected_features), start=1):
                rows.append(
                    {
                        "probe_fold": label,
                        "prior_fold_ids": "|".join(map(str, prior_folds)),
                        "axis": axis,
                        "rank": rank,
                        "node_id": str(node),
                        "feature": str(feature),
                    }
                )
        return list(dict.fromkeys(union)), axis_features

    for fold_id in selection_folds:
        prior = [value for value in selection_folds if value < fold_id]
        selected, selected_by_axis = select(prior, str(fold_id)) if prior else ([], {"AB": [], "CD": []})
        features_by_fold[str(fold_id)] = selected
        features_by_fold_axis[str(fold_id)] = selected_by_axis
    global_selected, global_by_axis = select(selection_folds, "GLOBAL_SELECTION")
    features_by_fold["GLOBAL_SELECTION"] = global_selected
    features_by_fold_axis["GLOBAL_SELECTION"] = global_by_axis

    extended_union: list[str] = []
    for values in features_by_fold.values():
        extended_union.extend(values)
    extended_union = list(dict.fromkeys(extended_union))
    manifest = {
        "schema": "surge_separation_staged_transform_manifest_v9",
        "generated_at": utc_now(),
        "stage0_transforms": ["raw", "missing_indicator"],
        "extended_transforms": [
            "date_rank",
            "date_z",
            "date_market_rank",
            "date_bucket_rank",
            "date_industry_rank",
            "cluster_innovation",
            "peer_innovation_k3",
        ],
        "prefilter_feature_count_per_axis": int(args.prefilter_feature_count),
        "features_by_probe_fold": features_by_fold,
        "features_by_probe_fold_and_axis": features_by_fold_axis,
        "extended_feature_union": extended_union,
        "extended_feature_union_count": len(extended_union),
        "selection_rows": rows,
    }
    return manifest, extended_union


def score_prior_pair_candidates(
    pair_fold: pd.DataFrame,
    prior_folds: Sequence[int],
    candidate_count: int,
    allowed_pair_keys: Sequence[str],
    minimum_positive_rows: int,
    minimum_negative_rows: int,
    minimum_coverage: float,
    minimum_orientation_consistency: float,
) -> list[str]:
    if pair_fold.empty or not prior_folds or not allowed_pair_keys:
        return []
    part = pair_fold[pair_fold["fold_id"].isin([int(value) for value in prior_folds])].copy()
    part["pair_key"] = [
        _interaction_pair_key(axis, feature_a, feature_b)
        for axis, feature_a, feature_b in zip(part["axis"], part["feature_a"], part["feature_b"])
    ]
    part = part[part["pair_key"].isin(set(str(value) for value in allowed_pair_keys))]
    if part.empty:
        return []
    records: list[dict[str, Any]] = []
    keys = ["axis", "feature_a", "feature_b", "pair_transform", "node_id", "pair_key"]
    for group_key, group in part.groupby(keys, sort=True):
        axis, feature_a, feature_b, pair_transform, node_id, pair_key = group_key
        auc = pd.to_numeric(group["oriented_auc"], errors="coerce").to_numpy(dtype=np.float64)
        auc = auc[np.isfinite(auc)]
        matched = pd.to_numeric(group["matched_concordance"], errors="coerce").to_numpy(dtype=np.float64)
        matched = matched[np.isfinite(matched)]
        orientation = pd.to_numeric(group["orientation"], errors="coerce").to_numpy(dtype=np.float64)
        orientation = orientation[np.isfinite(orientation) & (orientation != 0)]
        stability = max(float(np.mean(orientation > 0)), float(np.mean(orientation < 0))) if len(orientation) else 0.0
        coverage = np.minimum(
            pd.to_numeric(group["positive_coverage"], errors="coerce").fillna(0).to_numpy(dtype=np.float64),
            pd.to_numeric(group["negative_coverage"], errors="coerce").fillna(0).to_numpy(dtype=np.float64),
        )
        positive_n = pd.to_numeric(group["positive_valid_n"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        negative_n = pd.to_numeric(group["negative_valid_n"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        if (
            len(group) < len(prior_folds)
            or len(auc) == 0
            or float(np.min(positive_n)) < int(minimum_positive_rows)
            or float(np.min(negative_n)) < int(minimum_negative_rows)
            or float(np.min(coverage)) < float(minimum_coverage)
            or stability < float(minimum_orientation_consistency)
        ):
            continue
        score = 0.45 * max(0.0, (float(np.mean(auc)) - 0.5) * 2.0)
        score += 0.30 * max(0.0, (float(np.min(auc)) - 0.5) * 2.0)
        if len(matched):
            score += 0.20 * max(0.0, (float(np.mean(matched)) - 0.5) * 2.0)
        score += 0.05 * stability
        score *= math.sqrt(float(np.mean(coverage)))
        records.append(
            {
                "axis": str(axis),
                "feature_a": str(feature_a),
                "feature_b": str(feature_b),
                "pair_transform": str(pair_transform),
                "node_id": str(node_id),
                "pair_key": str(pair_key),
                "score": score,
            }
        )
    ranking = pd.DataFrame.from_records(records)
    if ranking.empty:
        return []
    selected: list[str] = []
    for axis in ["AB", "CD"]:
        selected.extend(
            ranking[ranking["axis"].eq(axis)]
            .sort_values(["score", "node_id"], ascending=[False, True])
            .head(int(candidate_count))["node_id"]
            .astype(str)
            .tolist()
        )
    return list(dict.fromkeys(selected))

def build_dynamic_probe_frame(
    oof: pd.DataFrame,
    selected_nodes: Sequence[str],
    value_sources: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    matrix = pd.DataFrame({"base_rank": pd.to_numeric(oof["base_rank"], errors="coerce").to_numpy(dtype=np.float64)})
    columns = ["base_rank"]
    column_map = {"base_rank": "base_rank"}
    for node_id in selected_nodes:
        if node_id not in value_sources:
            continue
        column = f"f_{len(columns):03d}"
        matrix[column] = np.asarray(value_sources[node_id], dtype=np.float32)
        columns.append(column)
        column_map[column] = str(node_id)
    return matrix, columns, column_map


def forward_precision_probe(
    oof: pd.DataFrame,
    univariate_fold: pd.DataFrame,
    horizon_fold: pd.DataFrame,
    pair_fold: pd.DataFrame,
    univariate_values: Mapping[str, np.ndarray],
    horizon_values: Mapping[str, np.ndarray],
    pair_values: Mapping[str, np.ndarray],
    global_manifest: Mapping[str, Any],
    staged_transform_manifest: Mapping[str, Any],
    interaction_prefilter_manifest: Mapping[str, Any],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_folds = sorted(int(value) for value in roles["selection"])
    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    target = oof[args.target_column].to_numpy(dtype=np.int8)
    all_values: dict[str, np.ndarray] = {}
    all_values.update(univariate_values)
    all_values.update(horizon_values)
    all_values.update(pair_values)
    for fold_id in sorted(oof["fold_id"].unique()):
        fold_id = int(fold_id)
        role = role_for_fold(fold_id, roles)
        validation_mask = oof["fold_id"].eq(fold_id).to_numpy()
        if role == "selection":
            prior_folds = [prior for prior in selection_folds if prior < fold_id]
            allowed_features = staged_transform_manifest.get("features_by_probe_fold", {}).get(str(fold_id), [])
            selected_nodes = score_prior_univariate_candidates(
                univariate_fold,
                prior_folds,
                int(args.axis_candidate_count),
                int(args.max_transforms_per_feature),
                int(args.minimum_positive_rows),
                int(args.minimum_negative_rows),
                float(args.minimum_coverage),
                float(args.minimum_orientation_consistency),
                allowed_features=allowed_features,
            )
            selected_nodes.extend(
                score_prior_horizon_candidates(
                    horizon_fold,
                    prior_folds,
                    int(args.horizon_candidate_count),
                    int(args.minimum_positive_rows),
                    int(args.minimum_negative_rows),
                    float(args.minimum_coverage),
                    float(args.minimum_orientation_consistency),
                )
            )
            allowed_pair_keys = interaction_prefilter_manifest.get("pair_keys_by_probe_fold", {}).get(str(fold_id), [])
            selected_nodes.extend(
                score_prior_pair_candidates(
                    pair_fold,
                    prior_folds,
                    int(args.pair_candidate_count),
                    allowed_pair_keys,
                    int(args.minimum_positive_rows),
                    int(args.minimum_negative_rows),
                    float(args.minimum_coverage),
                    float(args.minimum_orientation_consistency),
                )
            )
        else:
            prior_folds = selection_folds
            selected_nodes = []
            for values in global_manifest.get("candidates", {}).values():
                selected_nodes.extend([str(value) for value in values])
        selected_nodes = list(dict.fromkeys(selected_nodes))
        probe_matrix, probe_columns, column_map = build_dynamic_probe_frame(oof, selected_nodes, all_values)
        train_mask = oof["fold_id"].isin(prior_folds).to_numpy()
        base_scores = probe_matrix.loc[validation_mask, "base_rank"].to_numpy(dtype=np.float64)
        if (
            int(train_mask.sum()) < int(args.minimum_probe_train_rows)
            or len(np.unique(target[train_mask])) < 2
            or len(probe_columns) <= 1
        ):
            probe_scores = base_scores.copy()
            status = "BASE_FALLBACK_WARMUP"
        else:
            train_frame = probe_matrix.loc[train_mask, probe_columns].copy()
            validation_frame = probe_matrix.loc[validation_mask, probe_columns].copy()
            train_frame[args.target_column] = target[train_mask]
            probe_scores = fit_forward_logistic_probe(
                train_frame,
                validation_frame,
                probe_columns,
                args.target_column,
                negative_weight=float(args.probe_negative_weight),
                regularization_c=float(args.probe_regularization_c),
                seed=int(args.seed + fold_id),
            )
            status = "PROBE"
        y_validation = target[validation_mask]
        base_metrics = safe_binary_metrics(y_validation, base_scores)
        base_tail = precision_curve_metrics(
            y_validation,
            base_scores,
            minimum_alerts=int(args.minimum_alerts),
            target_precision=float(args.target_precision),
        )
        probe_metrics = safe_binary_metrics(y_validation, probe_scores)
        probe_tail = precision_curve_metrics(
            y_validation,
            probe_scores,
            minimum_alerts=int(args.minimum_alerts),
            target_precision=float(args.target_precision),
        )
        records.append(
            {
                "fold_id": fold_id,
                "fold_role": role,
                "status": status,
                "prior_fold_ids": "|".join(map(str, prior_folds)),
                "train_oof_rows": int(train_mask.sum()),
                "candidate_feature_count": int(len(probe_columns)),
                **{f"base_{key}": value for key, value in {**base_metrics, **base_tail}.items()},
                **{f"probe_{key}": value for key, value in {**probe_metrics, **probe_tail}.items()},
                "utility_pr_auc": float(probe_metrics["pr_auc"] - base_metrics["pr_auc"])
                if np.isfinite(probe_metrics["pr_auc"]) and np.isfinite(base_metrics["pr_auc"])
                else float("nan"),
                "utility_best_precision": float(probe_tail["best_precision_min_alerts"] - base_tail["best_precision_min_alerts"])
                if np.isfinite(probe_tail["best_precision_min_alerts"]) and np.isfinite(base_tail["best_precision_min_alerts"])
                else float("nan"),
                "probe_precision70_gate": bool(probe_tail["precision_target_alerts"] >= int(args.minimum_alerts)),
            }
        )
        for column in probe_columns:
            candidate_records.append(
                {
                    "fold_id": fold_id,
                    "fold_role": role,
                    "prior_fold_ids": "|".join(map(str, prior_folds)),
                    "column": column,
                    "node_id": column_map.get(column, column),
                }
            )
    fold_results = pd.DataFrame.from_records(records)
    role_records: list[dict[str, Any]] = []
    for role, part in fold_results.groupby("fold_role", sort=True):
        role_records.append(
            {
                "fold_role": role,
                "fold_count": int(len(part)),
                "mean_base_pr_auc": float(pd.to_numeric(part["base_pr_auc"], errors="coerce").mean()),
                "mean_probe_pr_auc": float(pd.to_numeric(part["probe_pr_auc"], errors="coerce").mean()),
                "mean_utility_pr_auc": float(pd.to_numeric(part["utility_pr_auc"], errors="coerce").mean()),
                "mean_base_best_precision": float(pd.to_numeric(part["base_best_precision_min_alerts"], errors="coerce").mean()),
                "mean_probe_best_precision": float(pd.to_numeric(part["probe_best_precision_min_alerts"], errors="coerce").mean()),
                "mean_utility_best_precision": float(pd.to_numeric(part["utility_best_precision"], errors="coerce").mean()),
                "precision70_fold_pass_rate": float(part["probe_precision70_gate"].mean()),
                "minimum_probe_best_precision": float(pd.to_numeric(part["probe_best_precision_min_alerts"], errors="coerce").min()),
                "maximum_probe_best_precision": float(pd.to_numeric(part["probe_best_precision_min_alerts"], errors="coerce").max()),
            }
        )
    return fold_results, pd.DataFrame.from_records(role_records), pd.DataFrame.from_records(candidate_records)

def build_graph(
    univariate_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = set()
    for values in manifest.get("candidates", {}).values():
        requested.update(str(value) for value in values)
    nodes = univariate_summary[univariate_summary["node_id"].isin(requested)].copy()
    node_records: list[dict[str, Any]] = []
    for row in nodes.itertuples(index=False):
        node_records.append(
            {
                "node_id": str(row.node_id),
                "node_type": "UNIVARIATE",
                "feature": str(row.feature),
                "transform": str(row.transform),
                "axis": str(row.axis),
                "separation_score": float(row.separation_score),
                "selection_auc": float(getattr(row, "selection__mean_oriented_auc", np.nan)),
                "selection_matched": float(getattr(row, "selection__mean_matched_concordance", np.nan)),
                "recent_auc": float(getattr(row, "recent_audit__mean_oriented_auc", np.nan)),
            }
        )
    if not horizon_summary.empty:
        selected_horizons = horizon_summary[horizon_summary["transform_name"].isin(requested)]
        for row in selected_horizons.itertuples(index=False):
            node_records.append(
                {
                    "node_id": str(row.transform_name),
                    "node_type": "HORIZON_SPREAD",
                    "feature": f"{row.short_feature}|{row.long_feature}",
                    "transform": "horizon_rank_spread",
                    "axis": str(row.axis),
                    "separation_score": float(row.separation_score),
                    "selection_auc": float(getattr(row, "selection__mean_oriented_auc", np.nan)),
                    "selection_matched": float(getattr(row, "selection__mean_matched_concordance", np.nan)),
                    "recent_auc": float(getattr(row, "recent_audit__mean_oriented_auc", np.nan)),
                }
            )
    if not pair_summary.empty:
        selected_pairs = pair_summary[pair_summary["node_id"].isin(requested)]
        for row in selected_pairs.itertuples(index=False):
            node_records.append(
                {
                    "node_id": str(row.node_id),
                    "node_type": "PAIR_TRANSFORM",
                    "feature": f"{row.feature_a}|{row.feature_b}",
                    "transform": str(row.pair_transform),
                    "axis": str(row.axis),
                    "separation_score": float(row.separation_score),
                    "selection_auc": float(getattr(row, "selection__mean_oriented_auc", np.nan)),
                    "selection_matched": float(getattr(row, "selection__mean_matched_concordance", np.nan)),
                    "recent_auc": float(getattr(row, "recent_audit__mean_oriented_auc", np.nan)),
                }
            )
    node_frame = pd.DataFrame.from_records(node_records).drop_duplicates("node_id") if node_records else pd.DataFrame(columns=["node_id"])
    feature_to_nodes: dict[str, list[str]] = defaultdict(list)
    for row in node_frame.itertuples(index=False):
        if row.node_type == "UNIVARIATE":
            feature_to_nodes[str(row.feature)].append(str(row.node_id))
    edge_records: list[dict[str, Any]] = []
    features = list(correlation.index)
    values = correlation.to_numpy(dtype=np.float64)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr = float(values[i, j])
            if corr < float(args.graph_correlation_threshold):
                continue
            for source in feature_to_nodes.get(features[i], []):
                for target in feature_to_nodes.get(features[j], []):
                    edge_records.append(
                        {
                            "source": source,
                            "target": target,
                            "edge_type": "REDUNDANCY_CORRELATION",
                            "weight": corr,
                            "detail": f"abs_corr={corr:.6f}",
                        }
                    )
    if not horizon_summary.empty:
        selected_horizons = horizon_summary[horizon_summary["transform_name"].isin(requested)]
        for row in selected_horizons.itertuples(index=False):
            horizon_node = str(row.transform_name)
            for feature in [str(row.short_feature), str(row.long_feature)]:
                for raw_node in feature_to_nodes.get(feature, []):
                    edge_records.append(
                        {
                            "source": horizon_node,
                            "target": raw_node,
                            "edge_type": "HORIZON_COMPONENT",
                            "weight": float(row.separation_score),
                            "detail": str(row.axis),
                        }
                    )
    if not pair_summary.empty:
        selected_pairs = pair_summary[pair_summary["node_id"].isin(requested)]
        for row in selected_pairs.itertuples(index=False):
            pair_node = str(row.node_id)
            for feature in [str(row.feature_a), str(row.feature_b)]:
                for raw_node in feature_to_nodes.get(feature, []):
                    edge_records.append(
                        {
                            "source": pair_node,
                            "target": raw_node,
                            "edge_type": "INTERACTION_COMPONENT",
                            "weight": float(row.separation_score),
                            "detail": str(row.axis),
                        }
                    )
    edge_frame = pd.DataFrame.from_records(edge_records)
    if not edge_frame.empty:
        edge_frame.drop_duplicates(["source", "target", "edge_type"], inplace=True)
    return node_frame, edge_frame


def build_report(
    args: argparse.Namespace,
    oof: pd.DataFrame,
    univariate_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    probe_role: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> str:
    group_counts = oof.groupby(["fold_role", "error_group"], sort=True).size().unstack(fill_value=0)
    lines = [
        "# CrashWatch Surge Separation Map V9 결과",
        "",
        "## 목표",
        "",
        f"- 최소 경보 수: {int(args.minimum_alerts)}",
        f"- 목표 경보 Precision: {float(args.target_precision):.1%}",
        "- 분석 단위: 439개 Stage-0 + A/B·C/D shortlist 상대·잔차 확장 + 전체 interaction screen",
        "",
        "## 오류집단",
        "",
        "```text",
        group_counts.to_string(),
        "```",
        "",
        "## 분리력 지도 설계",
        "",
        "- A/B는 같은 고점수 구간 안에서 TP와 FP를 비교합니다.",
        "- C/D는 같은 저점수 구간 안에서 missed positive와 true negative를 비교합니다.",
        "- 같은 날짜의 base-score 근접 대조군을 매칭해 기존 score 차이를 통제합니다.",
        "- 원본값 외에 날짜/시장/bucket/industry 상대순위, cluster innovation, peer innovation을 계산합니다.",
        "- 모든 피처쌍의 class-conditional covariance 차이를 계산한 뒤 상위 interaction을 forward 검증합니다.",
        "- 최종 후보는 base OOF에 추가해 forward-only 30경보 Precision probe를 수행합니다.",
        "",
        "## 지도 규모",
        "",
        f"- Combined univariate map rows: {len(univariate_summary):,}",
        f"- Stage-0 source features: {int(manifest.get('feature_count', 0)):,}",
        f"- Extended shortlist source features: {int(manifest.get('extended_feature_union_count', 0)):,}",
        f"- Horizon map rows: {len(horizon_summary):,}",
        f"- Validated pair-transform rows: {len(pair_summary):,}",
        f"- AB candidate count: {len(manifest.get('candidates', {}).get('AB', []))}",
        f"- CD candidate count: {len(manifest.get('candidates', {}).get('CD', []))}",
        "",
        "## Forward precision probe",
        "",
        "```text",
        probe_role.to_string(index=False) if not probe_role.empty else "No probe results",
        "```",
        "",
    ]
    if not probe_role.empty and "precision70_fold_pass_rate" in probe_role.columns:
        selection = probe_role[probe_role["fold_role"].eq("selection")]
        confirmation = probe_role[probe_role["fold_role"].eq("confirmation")]
        recent = probe_role[probe_role["fold_role"].eq("recent_audit")]
        passed = (
            not selection.empty
            and not confirmation.empty
            and not recent.empty
            and float(selection["precision70_fold_pass_rate"].iloc[0]) >= 1.0
            and float(confirmation["precision70_fold_pass_rate"].iloc[0]) >= 1.0
            and float(recent["precision70_fold_pass_rate"].iloc[0]) >= 1.0
        )
    else:
        passed = False
    lines.extend(
        [
            "## 판정",
            "",
            "`READY_FOR_FROZEN_CONFIRMATION`" if passed else "`STOP_PRECISION30_SEPARATION_GATE`",
            "",
            "이 지도는 상관계수가 높은 피처를 찾는 지도가 아니라, 기존 점수가 설명하지 못한 A/B와 C/D의 조건부 분리력을 지도화합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    package_root = Path(args.package_root).resolve()
    resolve_default_paths(package_root, args)
    args.dataset = Path(args.dataset).resolve()
    args.target_sidecar = Path(args.target_sidecar).resolve()
    args.folds = Path(args.folds).resolve()
    args.feature_profile_manifest = Path(args.feature_profile_manifest).resolve()
    args.correlation_matrix = Path(args.correlation_matrix).resolve()
    args.v8_dir = Path(args.v8_dir).resolve()
    args.output = Path(args.output).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    status = RunStatus(args.output, "CrashWatch Surge Separation Map V9")
    started = time.monotonic()
    try:
        require_paths(
            {
                "dataset": args.dataset,
                "target_sidecar": args.target_sidecar,
                "folds": args.folds,
                "feature_profile_manifest": args.feature_profile_manifest,
                "correlation_matrix": args.correlation_matrix,
                "V8 output": args.v8_dir,
            }
        )
        folds = load_folds(args.folds)
        roles = resolve_roles(args, folds)
        dataset_columns = table_columns(args.dataset)
        features = load_feature_universe(args, dataset_columns)
        status.stage("inputs", "SUCCESS", feature_count=len(features))

        log("Loading source data and target sidecar")
        frame = load_dataset_frame(args, features)
        indices = fold_indices(frame, folds, args.date_column)
        correlation = load_correlation_matrix(args.correlation_matrix, features)
        cluster_members, primary_cluster_frame = load_primary_clusters(args.v8_dir, features, float(args.primary_cluster_threshold))
        v8_meta = v8_annotations(args.v8_dir)
        status.stage("data", "SUCCESS", rows=len(frame), positives=int(frame[args.target_column].sum()))

        log("Loading or rebuilding full-439 base OOF")
        base_oof, base_source = load_or_build_base_oof(frame, features, folds, indices, args)
        oof = attach_oof_metadata(frame, base_oof, folds, roles, args)
        atomic_write_csv(args.output / "base_oof_predictions.csv", oof)
        base_metrics_records: list[dict[str, Any]] = []
        for fold_id, part in oof.groupby("fold_id", sort=True):
            metrics = safe_binary_metrics(part[args.target_column].to_numpy(dtype=np.int8), part["base_rank"].to_numpy(dtype=np.float64))
            tail = precision_curve_metrics(
                part[args.target_column].to_numpy(dtype=np.int8),
                part["base_rank"].to_numpy(dtype=np.float64),
                minimum_alerts=int(args.minimum_alerts),
                target_precision=float(args.target_precision),
            )
            base_metrics_records.append(
                {
                    "fold_id": int(fold_id),
                    "fold_role": role_for_fold(int(fold_id), roles),
                    **metrics,
                    **tail,
                }
            )
        base_metrics = pd.DataFrame.from_records(base_metrics_records)
        atomic_write_csv(args.output / "base_oof_metrics_by_fold.csv", base_metrics)
        status.stage("base_oof", "SUCCESS", source=base_source, rows=len(oof))

        log("Building matched A/B and C/D controls")
        ab_pairs, cd_pairs = build_match_tables(oof, args)
        atomic_write_csv(args.output / "matched_pairs_ab.csv", ab_pairs)
        atomic_write_csv(args.output / "matched_pairs_cd.csv", cd_pairs)
        group_counts = (
            oof.groupby(["fold_id", "fold_role", "error_group"], sort=True)
            .size()
            .rename("rows")
            .reset_index()
        )
        atomic_write_csv(args.output / "error_group_counts_by_fold.csv", group_counts)
        status.stage("matching", "SUCCESS", ab_pairs=len(ab_pairs), cd_pairs=len(cd_pairs))

        log("Stage 0: auditing all features with raw and missingness transforms")
        stage0_plan = {feature: ["raw", "missing_indicator"] for feature in features}
        stage0_fold, stage0_values = compute_univariate_map(
            oof,
            frame,
            features,
            correlation,
            cluster_members,
            ab_pairs,
            cd_pairs,
            roles,
            args,
            transform_plan=stage0_plan,
            stage_name="STAGE0_ALL_FEATURES",
        )
        stage0_summary = summarize_univariate_map(stage0_fold, v8_meta, roles, args)
        atomic_write_csv(args.output / "separation_stage0_by_fold.csv", stage0_fold)
        atomic_write_csv(args.output / "separation_stage0_summary.csv", stage0_summary)

        log("Building forward-only A/B and C/D prefilter for staged relative transforms")
        staged_transform_manifest, extended_features = build_staged_transform_prefilter(stage0_fold, roles, args)
        atomic_write_json(args.output / "STAGED_TRANSFORM_MANIFEST_V9.json", staged_transform_manifest)
        selection_rows = pd.DataFrame.from_records(staged_transform_manifest.get("selection_rows", []))
        atomic_write_csv(args.output / "staged_prefilter_selection.csv", selection_rows)

        extended_transforms = [
            "date_rank",
            "date_z",
            "date_market_rank",
            "date_bucket_rank",
            "date_industry_rank",
            "cluster_innovation",
            "peer_innovation_k3",
        ]
        extended_plan = {feature: extended_transforms for feature in extended_features}
        if extended_features:
            log(f"Stages 1-3: expanding relative/innovation transforms for {len(extended_features)} shortlisted features")
            extended_fold, extended_values = compute_univariate_map(
                oof,
                frame,
                extended_features,
                correlation,
                cluster_members,
                ab_pairs,
                cd_pairs,
                roles,
                args,
                transform_plan=extended_plan,
                stage_name="STAGE1_3_SHORTLISTED_RELATIVE",
            )
            extended_summary = summarize_univariate_map(extended_fold, v8_meta, roles, args)
        else:
            extended_fold = pd.DataFrame(columns=stage0_fold.columns)
            extended_values = {}
            extended_summary = pd.DataFrame(columns=stage0_summary.columns)
        atomic_write_csv(args.output / "separation_extended_by_fold.csv", extended_fold)
        atomic_write_csv(args.output / "separation_extended_summary.csv", extended_summary)

        univariate_fold = pd.concat([stage0_fold, extended_fold], ignore_index=True)
        univariate_values = dict(stage0_values)
        univariate_values.update(extended_values)
        univariate_summary = summarize_univariate_map(univariate_fold, v8_meta, roles, args)
        atomic_write_csv(args.output / "separation_univariate_by_fold.csv", univariate_fold)
        atomic_write_csv(args.output / "separation_univariate_summary.csv", univariate_summary)
        status.stage(
            "univariate_map",
            "SUCCESS",
            stage0_features=len(features),
            extended_features=len(extended_features),
            rows=len(univariate_summary),
        )

        log("Computing staged horizon-shape separation map")
        all_horizon_pairs = build_horizon_pairs(features)
        extended_set = set(extended_features)
        horizon_pairs = all_horizon_pairs[
            all_horizon_pairs["short_feature"].isin(extended_set)
            | all_horizon_pairs["long_feature"].isin(extended_set)
        ].reset_index(drop=True) if extended_set and not all_horizon_pairs.empty else all_horizon_pairs.iloc[0:0].copy()
        horizon_fold, horizon_values = compute_horizon_map(
            oof,
            frame,
            horizon_pairs,
            ab_pairs,
            cd_pairs,
            roles,
            args,
        )
        horizon_summary = summarize_horizon_map(horizon_fold, roles, args)
        atomic_write_csv(args.output / "horizon_shape_by_fold.csv", horizon_fold)
        atomic_write_csv(args.output / "horizon_shape_summary.csv", horizon_summary)
        status.stage("horizon_map", "SUCCESS", pairs=len(horizon_pairs), rows=len(horizon_summary))

        log("Building full 439-feature OOF date-rank matrix")
        rank_matrix = build_rank_matrix(oof, frame, features, args)
        atomic_write_npz(
            args.output / "oof_rank_matrix_v9.npz",
            matrix=rank_matrix,
            row_index=oof["row_index"].to_numpy(dtype=np.int64),
            feature_names=np.asarray(features, dtype="U128"),
        )
        log("Computing pairwise interaction-moment screen")
        interaction_fold, interaction_summary = compute_interaction_map(oof, rank_matrix, features, roles, args)
        atomic_write_csv(args.output / "interaction_moment_by_fold.csv", interaction_fold)
        atomic_write_csv(args.output / "interaction_moment_summary.csv", interaction_summary)
        interaction_prefilter_manifest, interaction_pair_union = build_forward_interaction_prefilter(
            interaction_fold,
            roles,
            args,
        )
        atomic_write_json(args.output / "FORWARD_INTERACTION_MANIFEST_V9.json", interaction_prefilter_manifest)
        atomic_write_csv(
            args.output / "forward_interaction_prefilter_selection.csv",
            pd.DataFrame.from_records(interaction_prefilter_manifest.get("selection_rows", [])),
        )
        status.stage(
            "interaction_screen",
            "SUCCESS",
            rows=len(interaction_summary),
            validated_pair_union=len(interaction_pair_union),
        )

        log("Validating forward-screened nonlinear pair transforms")
        pair_fold, pair_values = validate_pair_transforms(
            oof,
            rank_matrix,
            features,
            interaction_pair_union,
            ab_pairs,
            cd_pairs,
            roles,
            args,
        )
        pair_summary = summarize_pair_validation(pair_fold, roles, args)
        atomic_write_csv(args.output / "pair_transform_by_fold.csv", pair_fold)
        atomic_write_csv(args.output / "pair_transform_summary.csv", pair_summary)
        status.stage("pair_validation", "SUCCESS", rows=len(pair_summary))

        manifest = candidate_manifest(
            univariate_summary,
            horizon_summary,
            pair_summary,
            staged_transform_manifest,
            args,
        )
        manifest.update(
            {
                "dataset_sha256": sha256_file(args.dataset),
                "target_sha256": sha256_file(args.target_sidecar),
                "correlation_sha256": sha256_file(args.correlation_matrix),
                "feature_count": len(features),
                "feature_hash": hash_strings(features),
                "base_oof_source": base_source,
                "fold_roles": roles,
                "staged_transform_manifest": "STAGED_TRANSFORM_MANIFEST_V9.json",
                "forward_interaction_manifest": "FORWARD_INTERACTION_MANIFEST_V9.json",
                "extended_feature_union_count": len(extended_features),
                "validated_interaction_pair_union_count": len(interaction_pair_union),
            }
        )
        atomic_write_json(args.output / "SEPARATION_FEATURE_MANIFEST_V9.json", manifest)

        candidate_matrix, candidate_columns = build_candidate_matrix(
            oof,
            univariate_values,
            horizon_values,
            pair_values,
            manifest,
        )
        candidate_name_map = {column: candidate_matrix.attrs.get(column, column) for column in candidate_columns}
        atomic_write_json(args.output / "candidate_column_map.json", candidate_name_map)
        atomic_write_npz(
            args.output / "candidate_feature_matrix_v9.npz",
            matrix=candidate_matrix[candidate_columns].to_numpy(dtype=np.float32),
            columns=np.asarray(candidate_columns, dtype="U32"),
            row_index=oof["row_index"].to_numpy(dtype=np.int64),
        )
        log("Running forward-only 30-alert precision probe")
        probe_fold, probe_role, probe_candidates = forward_precision_probe(
            oof,
            univariate_fold,
            horizon_fold,
            pair_fold,
            univariate_values,
            horizon_values,
            pair_values,
            manifest,
            staged_transform_manifest,
            interaction_prefilter_manifest,
            roles,
            args,
        )
        atomic_write_csv(args.output / "precision30_probe_by_fold.csv", probe_fold)
        atomic_write_csv(args.output / "precision30_probe_by_role.csv", probe_role)
        atomic_write_csv(args.output / "precision30_probe_candidates_by_fold.csv", probe_candidates)
        non_warmup_selection = probe_fold[(probe_fold["fold_role"].eq("selection")) & (probe_fold["status"].eq("PROBE"))]
        confirmation_probe = probe_fold[probe_fold["fold_role"].eq("confirmation")]
        recent_probe = probe_fold[probe_fold["fold_role"].eq("recent_audit")]
        selection_pass = bool(len(non_warmup_selection) and non_warmup_selection["probe_precision70_gate"].all())
        confirmation_pass = bool(len(confirmation_probe) and confirmation_probe["probe_precision70_gate"].all())
        recent_pass = bool(len(recent_probe) and recent_probe["probe_precision70_gate"].all())
        gate_status = (
            "READY_FOR_FROZEN_CONFIRMATION"
            if selection_pass and confirmation_pass and recent_pass
            else "STOP_PRECISION30_SEPARATION_GATE"
        )
        recommendation = {
            "schema": "surge_separation_recommendation_v9",
            "generated_at": utc_now(),
            "status": gate_status,
            "target_precision": float(args.target_precision),
            "minimum_alerts": int(args.minimum_alerts),
            "selection_pass": selection_pass,
            "confirmation_pass": confirmation_pass,
            "recent_pass": recent_pass,
            "production_action": "FREEZE_AND_CONFIRM" if gate_status.startswith("READY") else "NO_ALERT_AND_CONTINUE_RESEARCH",
            "candidate_manifest": "SEPARATION_FEATURE_MANIFEST_V9.json",
            "probe_metrics": "precision30_probe_by_fold.csv",
        }
        atomic_write_json(args.output / "FINAL_RECOMMENDATION_V9.json", recommendation)
        status.stage("precision_probe", "SUCCESS", candidate_count=len(candidate_columns), gate_status=gate_status)

        node_frame, edge_frame = build_graph(univariate_summary, horizon_summary, pair_summary, correlation, manifest, args)
        atomic_write_csv(args.output / "separation_graph_nodes.csv", node_frame)
        atomic_write_csv(args.output / "separation_graph_edges.csv", edge_frame)
        if not node_frame.empty:
            write_graphml(args.output / "separation_graph_v9.graphml", node_frame, edge_frame)
        status.stage("graph", "SUCCESS", nodes=len(node_frame), edges=len(edge_frame))

        report = build_report(args, oof, univariate_summary, horizon_summary, pair_summary, probe_role, manifest)
        atomic_write_text(args.output / "SEPARATION_MAP_REPORT_KO.md", report)
        run_config = {
            "schema": "surge_separation_map_v9_config",
            "generated_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "package_root": str(package_root),
            "dataset": str(args.dataset),
            "target_sidecar": str(args.target_sidecar),
            "folds": [fold.to_dict() for fold in folds],
            "fold_roles": roles,
            "feature_count": len(features),
            "feature_hash": hash_strings(features),
            "base_oof_source": base_source,
            "target_precision": float(args.target_precision),
            "minimum_alerts": int(args.minimum_alerts),
            "error_top_quantile": float(args.error_top_quantile),
            "error_low_quantile": float(args.error_low_quantile),
            "transform_types": list(TRANSFORM_TYPES),
            "prefilter_feature_count_per_axis": int(args.prefilter_feature_count),
            "extended_feature_union_count": int(len(extended_features)),
            "staged_transform_manifest": "STAGED_TRANSFORM_MANIFEST_V9.json",
            "forward_interaction_manifest": "FORWARD_INTERACTION_MANIFEST_V9.json",
            "validated_interaction_pair_union_count": int(len(interaction_pair_union)),
        }
        atomic_write_json(args.output / "resolved_run_config_v9.json", run_config)
        inventory = compute_output_inventory(args.output, exclude=["RUN_STATUS.json", "SEPARATION_MAP_MANIFEST_V9.json"])
        final_manifest = {
            "schema": "surge_separation_map_manifest_v9",
            "status": "SUCCESS",
            "generated_at": utc_now(),
            "run_config": run_config,
            "output_inventory": inventory,
        }
        atomic_write_json(args.output / "SEPARATION_MAP_MANIFEST_V9.json", final_manifest)
        status.success(
            feature_count=len(features),
            oof_rows=len(oof),
            univariate_nodes=len(univariate_summary),
            pair_nodes=len(pair_summary),
            candidate_count=len(candidate_columns),
            output_inventory_count=len(inventory),
            model_gate_status=gate_status,
        )
        log(f"Completed: {args.output}")
    except BaseException as exc:
        status.failure(exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CrashWatch Surge V9: conditional separation correlation map for 30-alert Precision 70%",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--feature-profile-manifest", type=Path)
    parser.add_argument("--feature-profile", default="P0_FULL_439")
    parser.add_argument("--correlation-matrix", type=Path)
    parser.add_argument("--v8-dir", type=Path)
    parser.add_argument("--base-oof", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-column", default="label_abs_surge_3d_5pct")
    parser.add_argument("--target-valid-column", default="target_valid")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--market-column", default="market")
    parser.add_argument("--bucket-column", default="bucket")
    parser.add_argument("--industry-column", default="industry_name")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--base-backends", default="lightgbm_cpu,xgboost_gpu")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--xgboost-threads", type=int, default=8)
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=1,
        help="Parallel workers for independent feature, horizon, rank, and pair-map calculations",
    )
    parser.add_argument(
        "--pair-workers",
        type=int,
        default=16,
        help="Spawned processes for nonlinear pair validation (uses separate memory per worker)",
    )
    parser.add_argument("--error-top-quantile", type=float, default=0.80)
    parser.add_argument("--error-low-quantile", type=float, default=0.50)
    parser.add_argument("--controls-per-case", type=int, default=3)
    parser.add_argument("--primary-cluster-threshold", type=float, default=0.92)
    parser.add_argument("--minimum-positive-rows", type=int, default=30)
    parser.add_argument("--minimum-negative-rows", type=int, default=60)
    parser.add_argument("--minimum-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-orientation-consistency", type=float, default=0.80)
    parser.add_argument("--interaction-top-per-fold", type=int, default=2500)
    parser.add_argument("--interaction-validate-count", type=int, default=256)
    parser.add_argument(
        "--interaction-prefilter-count",
        type=int,
        default=64,
        help="Pair sources selected per axis from prior folds for forward selection probes",
    )
    parser.add_argument("--minimum-interaction-fold-presence", type=float, default=0.50)
    parser.add_argument(
        "--prefilter-feature-count",
        type=int,
        default=18,
        help="A/B and C/D source features selected per axis before relative-transform expansion (recommended 12-24)",
    )
    parser.add_argument("--axis-candidate-count", type=int, default=24)
    parser.add_argument("--horizon-candidate-count", type=int, default=12)
    parser.add_argument("--pair-candidate-count", type=int, default=16)
    parser.add_argument("--max-transforms-per-feature", type=int, default=2)
    parser.add_argument("--minimum-alerts", type=int, default=30)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--minimum-probe-train-rows", type=int, default=300)
    parser.add_argument("--probe-negative-weight", type=float, default=3.0)
    parser.add_argument("--probe-regularization-c", type=float, default=0.05)
    parser.add_argument("--graph-correlation-threshold", type=float, default=0.80)
    parser.add_argument("--limit-features", type=int, default=0)
    parser.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
