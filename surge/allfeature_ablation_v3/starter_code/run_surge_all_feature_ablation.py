from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gc
import json
import math
import multiprocessing
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from surge_ablation_common import (
    FileLock,
    FoldSpec,
    RunStatus,
    TaskSpec,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    benjamini_hochberg,
    bootstrap_mean_ci,
    compute_output_inventory,
    deterministic_seed,
    ensure_free_disk,
    evaluate_prediction_metrics,
    exact_sign_flip_p,
    fraction_label,
    hash_strings,
    join_source_and_target,
    leakage_reason,
    load_folds,
    load_json,
    log,
    model_versions,
    parse_bool_series,
    parse_float_list,
    parse_int_list,
    payload_checksum_is_valid,
    read_table,
    result_file_is_valid,
    role_for_fold,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    table_columns,
    task_identity_hash,
    utc_now,
    validate_folds,
    validate_role_assignments,
    verify_feature_names,
    verify_output_inventory,
    with_payload_checksum,
)


_WORKER_CONTEXT: dict[str, Any] | None = None
_WORKER_MATRIX: np.ndarray | None = None
_WORKER_TARGET: np.ndarray | None = None
_WORKER_DATES: np.ndarray | None = None
_WORKER_FOLD_INDICES: Any = None
_WORKER_SCOPE_CODES: dict[str, np.ndarray] = {}


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
    if args.correlation_dir is None:
        candidates = [
            package_root / "outputs" / "surge_correlation_map_complete",
            package_root / "outputs" / "surge_correlation_map",
        ]
        args.correlation_dir = next((path for path in candidates if path.exists()), candidates[0])
    if args.pre_model_dir is None:
        args.pre_model_dir = package_root / "outputs" / "surge_pre_model_gate_v3"
    if args.output is None:
        args.output = package_root / "outputs" / "surge_all_feature_ablation_v3"


def require_paths(paths: Mapping[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"필수 입력 누락: {json.dumps(missing, ensure_ascii=False, indent=2)}")


def load_pre_model_gate(pre_model_dir: Path, dataset: Path, target_sidecar: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = pre_model_dir / "PRE_MODEL_GATE_MANIFEST.json"
    membership_path = pre_model_dir / "surge_feature_profile_membership_corrected.csv"
    audit_path = pre_model_dir / "surge_pre_model_feature_audit.csv"
    status_path = pre_model_dir / "RUN_STATUS.json"
    require_paths(
        {
            "pre-model manifest": manifest_path,
            "corrected membership": membership_path,
            "pre-model feature audit": audit_path,
            "pre-model status": status_path,
        }
    )
    status = load_json(status_path)
    if status.get("status") != "SUCCESS":
        raise RuntimeError(f"pre-model gate가 SUCCESS가 아닙니다: {status.get('status')}")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "SUCCESS":
        raise RuntimeError("PRE_MODEL_GATE_MANIFEST status가 SUCCESS가 아닙니다")
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("PRE_MODEL_GATE_MANIFEST에 검증 가능한 output_inventory가 없습니다")
    inventory_valid, inventory_reasons = verify_output_inventory(pre_model_dir, inventory)
    if not inventory_valid:
        raise RuntimeError(f"pre-model gate 산출물 무결성 실패: {inventory_reasons[:20]}")
    dataset_hash = sha256_file(dataset)
    target_hash = sha256_file(target_sidecar)
    if dataset_hash != manifest.get("dataset_sha256"):
        raise ValueError("pre-model gate와 현재 dataset SHA-256이 다릅니다")
    if target_hash != manifest.get("target_sha256"):
        raise ValueError("pre-model gate와 현재 target SHA-256이 다릅니다")
    membership = pd.read_csv(membership_path)
    audit = pd.read_csv(audit_path)
    if "feature" not in membership.columns or "feature" not in audit.columns:
        raise ValueError("pre-model 산출물에 feature 열이 없습니다")
    return manifest, membership, audit


def select_feature_universe(membership: pd.DataFrame, audit: pd.DataFrame, profile: str, limit_features: int) -> list[str]:
    if profile not in membership.columns:
        raise ValueError(f"baseline profile이 corrected membership에 없습니다: {profile}")
    mask = parse_bool_series(membership[profile])
    features = membership.loc[mask, "feature"].astype(str).tolist()
    if not features:
        raise ValueError(f"baseline profile이 비어 있습니다: {profile}")
    priority_map: dict[str, float] = {}
    if "surge_priority_rank" in audit.columns:
        priority_map = dict(zip(audit["feature"].astype(str), pd.to_numeric(audit["surge_priority_rank"], errors="coerce")))
    features.sort(key=lambda feature: (priority_map.get(feature, math.inf), feature))
    if limit_features > 0:
        features = features[:limit_features]
    return features


def verify_backend_availability(backends: Sequence[str]) -> dict[str, Any]:
    """Fail early instead of silently falling back to another compute device."""
    audit: dict[str, Any] = {}
    if "lightgbm_cpu" in backends:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm_cpu 실행에는 lightgbm이 필요합니다") from exc
        audit["lightgbm_cpu"] = {"version": str(getattr(lgb, "__version__", "unknown"))}

    if {"xgboost_cpu", "xgboost_gpu"} & set(backends):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("XGBoost backend 실행에는 xgboost가 필요합니다") from exc
        build_info = dict(xgb.build_info()) if hasattr(xgb, "build_info") else {}
        audit["xgboost"] = {
            "version": str(getattr(xgb, "__version__", "unknown")),
            "build_info": build_info,
        }
        if "xgboost_gpu" in backends:
            use_cuda = bool(build_info.get("USE_CUDA", False))
            if not use_cuda:
                raise RuntimeError(
                    "설치된 xgboost가 CUDA 지원 빌드가 아닙니다. GPU 실행을 CPU로 조용히 대체하지 않습니다."
                )
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi is None:
                raise RuntimeError("xgboost_gpu를 요청했지만 nvidia-smi를 찾지 못했습니다")
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                raise RuntimeError(
                    "NVIDIA GPU preflight 실패: "
                    + (completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}")
                )
            audit["xgboost_gpu"] = {
                "nvidia_smi_path": nvidia_smi,
                "devices": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
                "silent_cpu_fallback_allowed": False,
            }
    return audit


def build_matrix_cache(
    args: argparse.Namespace,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    pre_model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    cache_dir = args.output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "matrix_cache_manifest.json"
    matrix_path = cache_dir / "features.float32.npy"
    arrays_path = cache_dir / "row_metadata.npz"
    fold_indices_path = cache_dir / "fold_indices.npz"
    scope_codebook_path = cache_dir / "scope_codebooks.json"

    dataset_hash = pre_model_manifest["dataset_sha256"]
    target_hash = pre_model_manifest["target_sha256"]
    cache_identity = {
        "schema": "surge_matrix_cache_v3",
        "dataset_sha256": dataset_hash,
        "target_sha256": target_hash,
        "feature_hash": hash_strings(features),
        "features": list(features),
        "target_column": args.target_column,
        "target_valid_column": args.target_valid_column,
        "date_column": args.date_column,
        "ticker_column": args.ticker_column,
        "scope_columns": args.scope_columns,
        "folds": [fold.to_dict() for fold in folds],
    }
    cache_hash = sha256_bytes(stable_json_bytes(cache_identity))
    if args.resume and metadata_path.exists() and matrix_path.exists() and arrays_path.exists() and fold_indices_path.exists():
        existing = load_json(metadata_path)
        if existing.get("cache_hash") == cache_hash:
            expected = existing.get("artifacts", {})
            valid = True
            for name, path in {
                "matrix": matrix_path,
                "arrays": arrays_path,
                "fold_indices": fold_indices_path,
                "scope_codebooks": scope_codebook_path,
            }.items():
                if not path.exists() or expected.get(name, {}).get("sha256") != sha256_file(path):
                    valid = False
                    break
            if valid:
                log("검증된 matrix cache 재사용")
                return existing

    dataset_columns = table_columns(args.dataset)
    verify_feature_names(features, dataset_columns)
    required_source_columns = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        "sealed_do_not_train_or_tune",
        *args.scope_columns,
        *features,
    ]
    required_source_columns = [column for column in dict.fromkeys(required_source_columns) if column in dataset_columns]
    source = read_table(args.dataset, columns=required_source_columns)
    target_columns = table_columns(args.target_sidecar)
    side_columns = [
        column
        for column in ["source_row_id", args.target_column, args.target_valid_column, args.date_column, args.ticker_column]
        if column in target_columns
    ]
    sidecar = read_table(args.target_sidecar, columns=side_columns)
    joined = join_source_and_target(
        source,
        sidecar,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    if "sealed_do_not_train_or_tune" in joined.columns:
        sealed = pd.to_numeric(joined["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        if sealed.ne(0).any():
            raise ValueError(f"sealed_do_not_train_or_tune 비영 행: {int(sealed.ne(0).sum())}")
    joined[args.date_column] = pd.to_datetime(joined[args.date_column], errors="coerce")
    valid = parse_bool_series(joined[args.target_valid_column])
    valid &= joined[args.target_column].isin([0, 1])
    valid &= joined[args.date_column].notna()
    data = joined.loc[valid].reset_index(drop=True)
    if not len(data):
        raise ValueError("target-valid 학습 행이 없습니다")
    validate_folds(folds, minimum_purge_trading_days=args.minimum_purge_trading_days, all_dates=data[args.date_column])

    log(f"float32 matrix 생성: {len(data):,}행 × {len(features):,}피처")
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(data), len(features)))
    block_size = max(1, int(args.cache_column_block))
    for start in range(0, len(features), block_size):
        block_features = list(features[start : start + block_size])
        block = data[block_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        block[~np.isfinite(block)] = np.nan
        matrix[:, start : start + len(block_features)] = block
        matrix.flush()
        log(f"matrix columns {start + 1}-{start + len(block_features)} / {len(features)}")
    del matrix

    target = pd.to_numeric(data[args.target_column], errors="coerce").to_numpy(dtype=np.uint8)
    dates = data[args.date_column].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    source_row_id = pd.to_numeric(data.get("source_row_id", pd.Series(np.arange(len(data)))), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    scope_arrays: dict[str, np.ndarray] = {}
    scope_codebooks: dict[str, list[str]] = {}
    for column in args.scope_columns:
        if column not in data.columns:
            continue
        values = data[column].astype("string").fillna("<NA>")
        codes, uniques = pd.factorize(values, sort=True)
        scope_arrays[f"scope__{column}"] = codes.astype(np.int32)
        scope_codebooks[column] = [str(value) for value in uniques.tolist()]
    atomic_write_npz(
        arrays_path,
        target=target,
        dates=dates,
        source_row_id=source_row_id,
        **scope_arrays,
    )
    atomic_write_json(scope_codebook_path, scope_codebooks)

    date_series = pd.to_datetime(pd.Series(dates))
    fold_arrays: dict[str, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        train = date_series.between(fold.train_start, fold.train_end).to_numpy()
        validation = date_series.between(fold.validation_start, fold.validation_end).to_numpy()
        train_indices = np.flatnonzero(train).astype(np.int64)
        validation_indices = np.flatnonzero(validation).astype(np.int64)
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"fold {fold.fold_id}: train 또는 validation 행이 없습니다")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ValueError(f"fold {fold.fold_id}: train/validation row overlap")
        fold_arrays[f"train_{fold.fold_id}"] = train_indices
        fold_arrays[f"validation_{fold.fold_id}"] = validation_indices
        fold_records.append(
            {
                **fold.to_dict(),
                "fold_role": role_for_fold(fold.fold_id, roles),
                "train_rows": int(len(train_indices)),
                "validation_rows": int(len(validation_indices)),
                "train_positive_rate": float(np.mean(target[train_indices])),
                "validation_positive_rate": float(np.mean(target[validation_indices])),
            }
        )
    atomic_write_npz(fold_indices_path, **fold_arrays)
    atomic_write_json(args.output / "walk_forward_folds_ablation.json", fold_records)

    manifest = {
        **cache_identity,
        "cache_hash": cache_hash,
        "created_at": utc_now(),
        "row_count": len(data),
        "feature_count": len(features),
        "target_positive_count": int(target.sum()),
        "target_positive_rate": float(target.mean()),
        "date_min": pd.Timestamp(dates.min()).strftime("%Y-%m-%d"),
        "date_max": pd.Timestamp(dates.max()).strftime("%Y-%m-%d"),
        "matrix_shape": [len(data), len(features)],
        "matrix_dtype": "float32",
        "scope_codebooks": scope_codebooks,
        "artifacts": {
            "matrix": {"path": str(matrix_path), "sha256": sha256_file(matrix_path), "bytes": matrix_path.stat().st_size},
            "arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path), "bytes": arrays_path.stat().st_size},
            "fold_indices": {"path": str(fold_indices_path), "sha256": sha256_file(fold_indices_path), "bytes": fold_indices_path.stat().st_size},
            "scope_codebooks": {"path": str(scope_codebook_path), "sha256": sha256_file(scope_codebook_path), "bytes": scope_codebook_path.stat().st_size},
        },
    }
    atomic_write_json(metadata_path, manifest)
    del data, joined, source, sidecar
    gc.collect()
    return manifest


def _init_worker(context_path: str) -> None:
    global _WORKER_CONTEXT, _WORKER_MATRIX, _WORKER_TARGET, _WORKER_DATES, _WORKER_FOLD_INDICES, _WORKER_SCOPE_CODES
    _WORKER_CONTEXT = load_json(Path(context_path))
    os.environ["OMP_NUM_THREADS"] = str(_WORKER_CONTEXT["threads_per_worker"])
    os.environ["MKL_NUM_THREADS"] = str(_WORKER_CONTEXT["threads_per_worker"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(_WORKER_CONTEXT["threads_per_worker"])
    _WORKER_MATRIX = np.load(_WORKER_CONTEXT["matrix_path"], mmap_mode="r")
    arrays = np.load(_WORKER_CONTEXT["arrays_path"], mmap_mode="r", allow_pickle=False)
    _WORKER_TARGET = arrays["target"]
    _WORKER_DATES = arrays["dates"]
    _WORKER_SCOPE_CODES = {
        name.removeprefix("scope__"): arrays[name]
        for name in arrays.files
        if name.startswith("scope__")
    }
    _WORKER_FOLD_INDICES = np.load(_WORKER_CONTEXT["fold_indices_path"], mmap_mode="r", allow_pickle=False)


def _load_task_arrays(task: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _WORKER_MATRIX is None or _WORKER_TARGET is None or _WORKER_DATES is None or _WORKER_FOLD_INDICES is None:
        raise RuntimeError("worker context가 초기화되지 않았습니다")
    fold_id = int(task["fold_id"])
    train_indices = np.asarray(_WORKER_FOLD_INDICES[f"train_{fold_id}"], dtype=np.int64)
    validation_indices = np.asarray(_WORKER_FOLD_INDICES[f"validation_{fold_id}"], dtype=np.int64)
    feature_indices = np.asarray(task["enabled_feature_indices"], dtype=np.int64)
    if feature_indices.size == 0:
        raise ValueError("enabled feature가 0개입니다")
    x_train = np.asarray(_WORKER_MATRIX[np.ix_(train_indices, feature_indices)], dtype=np.float32)
    x_validation = np.asarray(_WORKER_MATRIX[np.ix_(validation_indices, feature_indices)], dtype=np.float32)
    y_train = np.asarray(_WORKER_TARGET[train_indices], dtype=np.uint8)
    y_validation = np.asarray(_WORKER_TARGET[validation_indices], dtype=np.uint8)
    dates_validation = np.asarray(_WORKER_DATES[validation_indices], dtype=np.int64).astype("datetime64[ns]")
    return x_train, y_train, x_validation, y_validation, dates_validation, validation_indices


def train_lightgbm_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    iterations: int,
    seed: int,
    params: Mapping[str, Any],
    threads: int,
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm이 설치되어 있지 않습니다") from exc
    resolved = dict(DEFAULT_LGB_PARAMS)
    resolved.update(params)
    resolved.update(
        {
            "metric": "None",
            "num_threads": int(threads),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
        }
    )
    dataset = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    booster = lgb.train(
        resolved,
        dataset,
        num_boost_round=int(iterations),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    prediction = booster.predict(x_validation, num_iteration=int(iterations))
    return np.asarray(prediction, dtype=np.float64)


def train_xgboost_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    iterations: int,
    seed: int,
    params: Mapping[str, Any],
    threads: int,
    device: str,
) -> np.ndarray:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost가 설치되어 있지 않습니다") from exc
    resolved = dict(DEFAULT_XGB_PARAMS)
    resolved.update(params)
    resolved.update({"seed": int(seed), "nthread": int(threads), "device": device})
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=int(resolved.get("max_bin", 256)))
    dvalidation = xgb.QuantileDMatrix(x_validation, ref=dtrain, max_bin=int(resolved.get("max_bin", 256)))
    booster = xgb.train(resolved, dtrain, num_boost_round=int(iterations), verbose_eval=False)
    prediction = booster.predict(dvalidation, iteration_range=(0, int(iterations)))
    return np.asarray(prediction, dtype=np.float64)


def _scope_metrics(
    target: np.ndarray,
    predictions: np.ndarray,
    validation_indices: np.ndarray,
) -> list[dict[str, Any]]:
    from surge_ablation_common import safe_binary_metrics

    records: list[dict[str, Any]] = []
    if _WORKER_CONTEXT is None:
        return records
    codebooks = _WORKER_CONTEXT.get("scope_codebooks", {})
    minimum_rows = int(_WORKER_CONTEXT.get("minimum_scope_rows", 20))
    for scope_name, codes_all in _WORKER_SCOPE_CODES.items():
        codes = np.asarray(codes_all[validation_indices], dtype=np.int32)
        labels = codebooks.get(scope_name, [])
        for code in np.unique(codes):
            mask = codes == code
            if int(mask.sum()) < minimum_rows:
                continue
            metrics = safe_binary_metrics(target[mask], predictions[mask])
            label = labels[int(code)] if 0 <= int(code) < len(labels) else str(code)
            records.append({"scope_type": scope_name, "scope_value": label, **metrics})
    return records


def _run_worker_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context 누락")
    result_path = Path(task["result_path"])
    identity_hash = str(task["identity_hash"])
    if result_file_is_valid(result_path, identity_hash):
        return {"status": "cached", "identity_hash": identity_hash, "result_path": str(result_path)}
    started = time.monotonic()
    try:
        x_train, y_train, x_validation, y_validation, validation_dates, validation_indices = _load_task_arrays(task)
        backend = str(task["backend"])
        if backend == "lightgbm_cpu":
            predictions = train_lightgbm_predict(
                x_train,
                y_train,
                x_validation,
                iterations=int(task["best_iteration"]),
                seed=int(task["seed"]),
                params=_WORKER_CONTEXT.get("lightgbm_params", {}),
                threads=int(_WORKER_CONTEXT["threads_per_worker"]),
            )
        elif backend in {"xgboost_gpu", "xgboost_cpu"}:
            device = "cuda" if backend == "xgboost_gpu" else "cpu"
            predictions = train_xgboost_predict(
                x_train,
                y_train,
                x_validation,
                iterations=int(task["best_iteration"]),
                seed=int(task["seed"]),
                params=_WORKER_CONTEXT.get("xgboost_params", {}),
                threads=int(_WORKER_CONTEXT.get("xgboost_threads", 1)),
                device=device,
            )
        else:
            raise ValueError(f"지원하지 않는 backend: {backend}")
        metrics = evaluate_prediction_metrics(
            y_validation,
            predictions,
            validation_dates,
            _WORKER_CONTEXT["top_fractions"],
        )
        scope_metrics = _scope_metrics(y_validation, predictions, validation_indices)
        prediction_path: str | None = None
        if bool(task.get("save_prediction")):
            prediction_dir = Path(_WORKER_CONTEXT["prediction_dir"])
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_file = prediction_dir / f"{identity_hash}.npz"
            atomic_write_npz(
                prediction_file,
                validation_indices=validation_indices.astype(np.int64),
                target=y_validation.astype(np.uint8),
                prediction=predictions.astype(np.float32),
                dates=validation_dates.astype("datetime64[ns]").astype(np.int64),
            )
            prediction_path = str(prediction_file)
        payload = {
            "status": "completed",
            "identity_hash": identity_hash,
            "task_id": identity_hash,
            "run_signature": task.get("run_signature"),
            "backend": backend,
            "stage": task["stage"],
            "test_type": task["test_type"],
            "condition_id": task["condition_id"],
            "fold_id": int(task["fold_id"]),
            "fold_role": task["fold_role"],
            "seed": int(task["seed"]),
            "best_iteration": int(task["best_iteration"]),
            "feature_count_total": int(task["feature_count_total"]),
            "enabled_feature_count": int(len(task["enabled_feature_indices"])),
            "dropped_features": list(task["dropped_features"]),
            "representative_feature": task.get("representative_feature"),
            "cluster_id": task.get("cluster_id"),
            "feature_group": task.get("feature_group"),
            "profile_name": task.get("profile_name"),
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "elapsed_seconds": time.monotonic() - started,
            "metrics": metrics,
            "scope_metrics": scope_metrics,
            "prediction_path": prediction_path,
        }
        payload = with_payload_checksum(payload)
        atomic_write_json(result_path, payload)
        return {"status": "completed", "identity_hash": identity_hash, "result_path": str(result_path)}
    except BaseException as exc:
        failed = {
            "status": "failed",
            "identity_hash": identity_hash,
            "run_signature": task.get("run_signature"),
            "backend": task.get("backend"),
            "stage": task.get("stage"),
            "test_type": task.get("test_type"),
            "condition_id": task.get("condition_id"),
            "fold_id": task.get("fold_id"),
            "seed": task.get("seed"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(result_path, failed)
        return {"status": "failed", "identity_hash": identity_hash, "result_path": str(result_path), "error": str(exc)}
    finally:
        with contextlib.suppress(Exception):
            del x_train, y_train, x_validation, y_validation
        gc.collect()


def build_inner_windows(
    all_dates: np.ndarray,
    outer_train_indices: np.ndarray,
    validation_days: int,
    purge_days: int,
    rolling_windows: int,
    rolling_step_days: int,
    minimum_train_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    train_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(all_dates[outer_train_indices]).unique())).sort_values()
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    date_values = pd.to_datetime(pd.Series(all_dates))
    for window_id in range(max(1, rolling_windows)):
        validation_end_pos = len(train_dates) - 1 - window_id * rolling_step_days
        validation_start_pos = validation_end_pos - validation_days + 1
        train_end_pos = validation_start_pos - purge_days - 1
        if validation_start_pos < 0 or train_end_pos < minimum_train_days - 1:
            continue
        inner_train_end = train_dates[train_end_pos]
        inner_validation_start = train_dates[validation_start_pos]
        inner_validation_end = train_dates[validation_end_pos]
        inner_train = outer_train_indices[date_values.iloc[outer_train_indices].le(inner_train_end).to_numpy()]
        inner_validation = outer_train_indices[
            date_values.iloc[outer_train_indices].between(inner_validation_start, inner_validation_end).to_numpy()
        ]
        if len(inner_train) and len(inner_validation):
            windows.append((inner_train.astype(np.int64), inner_validation.astype(np.int64)))
    return windows


def tune_lightgbm_iterations(
    matrix: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    outer_train_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[int, list[int]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm이 필요합니다") from exc
    windows = build_inner_windows(
        dates,
        outer_train_indices,
        validation_days=args.inner_validation_days,
        purge_days=args.inner_purge_days,
        rolling_windows=args.tuning_windows,
        rolling_step_days=args.tuning_step_days,
        minimum_train_days=args.minimum_inner_train_days,
    )
    if not windows:
        return int(args.fallback_iterations), []
    best: list[int] = []
    params = dict(DEFAULT_LGB_PARAMS)
    params.update(args.lightgbm_params)
    params.update(
        {
            "metric": "average_precision",
            "num_threads": int(args.threads_per_worker),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
        }
    )
    for inner_train, inner_validation in windows:
        x_train = np.asarray(matrix[inner_train, :], dtype=np.float32)
        x_valid = np.asarray(matrix[inner_validation, :], dtype=np.float32)
        train_set = lgb.Dataset(x_train, label=target[inner_train], free_raw_data=True)
        valid_set = lgb.Dataset(x_valid, label=target[inner_validation], reference=train_set, free_raw_data=True)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=int(args.max_tuning_rounds),
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        iteration = int(booster.best_iteration or args.fallback_iterations)
        best.append(iteration)
        del x_train, x_valid, train_set, valid_set, booster
        gc.collect()
    effective = int(np.median(best)) if best else int(args.fallback_iterations)
    effective = max(int(args.minimum_iterations), min(int(args.maximum_effective_iterations), effective))
    return effective, best


def tune_xgboost_iterations(
    matrix: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
    outer_train_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> tuple[int, list[int]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost가 필요합니다") from exc
    windows = build_inner_windows(
        dates,
        outer_train_indices,
        validation_days=args.inner_validation_days,
        purge_days=args.inner_purge_days,
        rolling_windows=args.tuning_windows,
        rolling_step_days=args.tuning_step_days,
        minimum_train_days=args.minimum_inner_train_days,
    )
    if not windows:
        return int(args.xgb_fallback_iterations), []
    best: list[int] = []
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(args.xgboost_params)
    params.update({"device": device, "seed": int(seed), "nthread": int(args.xgboost_threads)})
    for inner_train, inner_validation in windows:
        x_train = np.asarray(matrix[inner_train, :], dtype=np.float32)
        x_valid = np.asarray(matrix[inner_validation, :], dtype=np.float32)
        dtrain = xgb.QuantileDMatrix(x_train, label=target[inner_train], max_bin=int(params.get("max_bin", 256)))
        dvalid = xgb.QuantileDMatrix(x_valid, label=target[inner_validation], ref=dtrain, max_bin=int(params.get("max_bin", 256)))
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(args.xgb_max_tuning_rounds),
            evals=[(dvalid, "validation")],
            early_stopping_rounds=int(args.xgb_early_stopping_rounds),
            verbose_eval=False,
        )
        iteration = int((booster.best_iteration + 1) if booster.best_iteration is not None else args.xgb_fallback_iterations)
        best.append(iteration)
        del x_train, x_valid, dtrain, dvalid, booster
        gc.collect()
    effective = int(np.median(best)) if best else int(args.xgb_fallback_iterations)
    effective = max(int(args.xgb_minimum_iterations), min(int(args.xgb_maximum_effective_iterations), effective))
    return effective, best


def resolve_best_iterations(
    args: argparse.Namespace,
    cache_manifest: Mapping[str, Any],
    folds: Sequence[FoldSpec],
    backends: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, dict[str, int]]:
    path = args.output / "best_iterations.json"
    starter_dir = Path(__file__).resolve().parent
    identity = {
        "schema": "surge_ablation_best_iterations_v3",
        "cache_hash": cache_manifest["cache_hash"],
        "backends": list(backends),
        "seeds": list(seeds),
        "folds": [fold.to_dict() for fold in folds],
        "tuning": {
            "windows": int(args.tuning_windows),
            "step_days": int(args.tuning_step_days),
            "inner_validation_days": int(args.inner_validation_days),
            "inner_purge_days": int(args.inner_purge_days),
            "minimum_inner_train_days": int(args.minimum_inner_train_days),
            "lightgbm_max_tuning_rounds": int(args.max_tuning_rounds),
            "lightgbm_early_stopping_rounds": int(args.early_stopping_rounds),
            "lightgbm_minimum_iterations": int(args.minimum_iterations),
            "lightgbm_maximum_effective_iterations": int(args.maximum_effective_iterations),
            "lightgbm_fallback_iterations": int(args.fallback_iterations),
            "xgboost_max_tuning_rounds": int(args.xgb_max_tuning_rounds),
            "xgboost_early_stopping_rounds": int(args.xgb_early_stopping_rounds),
            "xgboost_minimum_iterations": int(args.xgb_minimum_iterations),
            "xgboost_maximum_effective_iterations": int(args.xgb_maximum_effective_iterations),
            "xgboost_fallback_iterations": int(args.xgb_fallback_iterations),
        },
        "threads": {
            "lightgbm": int(args.threads_per_worker),
            "xgboost": int(args.xgboost_threads),
        },
        "lightgbm_params": args.lightgbm_params,
        "xgboost_params": args.xgboost_params,
        "model_versions": model_versions(),
        "code_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "common": sha256_file(starter_dir / "surge_ablation_common.py"),
        },
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and path.exists():
        existing = load_json(path)
        if existing.get("identity_hash") == identity_hash and payload_checksum_is_valid(existing):
            return {
                backend: {str(key): int(value) for key, value in values.items()}
                for backend, values in existing["effective_iterations"].items()
            }

    matrix = np.load(cache_manifest["artifacts"]["matrix"]["path"], mmap_mode="r")
    arrays = np.load(cache_manifest["artifacts"]["arrays"]["path"], mmap_mode="r")
    fold_indices = np.load(cache_manifest["artifacts"]["fold_indices"]["path"], mmap_mode="r")
    target = arrays["target"]
    dates = arrays["dates"].astype("datetime64[ns]")
    effective: dict[str, dict[str, int]] = {backend: {} for backend in backends}
    details: list[dict[str, Any]] = []
    for backend in backends:
        for fold in folds:
            fold_seed_iterations: list[int] = []
            for seed in seeds:
                outer_train = np.asarray(fold_indices[f"train_{fold.fold_id}"], dtype=np.int64)
                if backend == "lightgbm_cpu":
                    iteration, windows = tune_lightgbm_iterations(matrix, target, dates, outer_train, seed, args)
                elif backend == "xgboost_gpu":
                    iteration, windows = tune_xgboost_iterations(matrix, target, dates, outer_train, seed, args, "cuda")
                elif backend == "xgboost_cpu":
                    iteration, windows = tune_xgboost_iterations(matrix, target, dates, outer_train, seed, args, "cpu")
                else:
                    raise ValueError(f"지원하지 않는 backend: {backend}")
                fold_seed_iterations.append(iteration)
                details.append(
                    {
                        "backend": backend,
                        "fold_id": fold.fold_id,
                        "seed": seed,
                        "effective_iteration": iteration,
                        "window_best_iterations": windows,
                    }
                )
                log(f"best iteration: {backend} fold={fold.fold_id} seed={seed} -> {iteration}")
            effective[backend][str(fold.fold_id)] = int(np.median(fold_seed_iterations))
    best_iteration_payload = with_payload_checksum(
        {
            "identity": identity,
            "identity_hash": identity_hash,
            "effective_iterations": effective,
            "details": details,
        }
    )
    atomic_write_json(path, best_iteration_payload)
    return effective


def build_task_payload(
    spec: TaskSpec,
    features: Sequence[str],
    best_iteration: int,
    roles: Mapping[str, Sequence[int]],
    run_signature: str,
    result_root: Path,
    save_prediction: bool,
) -> dict[str, Any]:
    identity = {
        "schema": "surge_ablation_task_v3",
        "run_signature": run_signature,
        "backend": spec.backend,
        "stage": spec.stage,
        "test_type": spec.test_type,
        "condition_id": spec.condition_id,
        "fold_id": spec.fold_id,
        "seed": spec.seed,
        "best_iteration": best_iteration,
        "enabled_feature_indices": list(spec.enabled_feature_indices),
        "dropped_features": list(spec.dropped_features),
        "representative_feature": spec.representative_feature,
        "cluster_id": spec.cluster_id,
        "feature_group": spec.feature_group,
        "profile_name": spec.profile_name,
        "save_prediction": bool(save_prediction),
    }
    identity_hash = task_identity_hash(identity)
    result_path = result_root / spec.backend / spec.stage / f"{identity_hash}.json"
    return {
        **identity,
        "identity_hash": identity_hash,
        "result_path": str(result_path),
        "fold_role": role_for_fold(spec.fold_id, roles),
        "feature_count_total": len(features),
        "save_prediction": save_prediction,
    }


def generate_task_specs(
    args: argparse.Namespace,
    features: Sequence[str],
    membership: pd.DataFrame,
    audit: pd.DataFrame,
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    backends: Sequence[str],
    seeds: Sequence[int],
    stages: Sequence[str],
) -> list[TaskSpec]:
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    all_indices = tuple(range(len(features)))
    selected_fold_ids = set(args.fold_ids) if args.fold_ids else {fold.fold_id for fold in folds}
    selected_folds = [fold for fold in folds if fold.fold_id in selected_fold_ids]
    if not selected_folds:
        raise ValueError("실행할 fold가 없습니다")

    specs: list[TaskSpec] = []
    for backend in backends:
        for seed in seeds:
            for fold in selected_folds:
                specs.append(
                    TaskSpec(
                        backend=backend,
                        stage="baseline",
                        test_type="baseline",
                        condition_id=f"BASELINE::{args.baseline_profile}",
                        fold_id=fold.fold_id,
                        seed=seed,
                        enabled_feature_indices=all_indices,
                        dropped_features=(),
                        profile_name=args.baseline_profile,
                    )
                )

    if "profiles" in stages:
        requested_profiles = [token.strip() for token in args.profile_benchmarks.split(",") if token.strip()]
        for profile in requested_profiles:
            if profile == args.baseline_profile:
                continue
            if profile not in membership.columns:
                log(f"profile benchmark 건너뜀(열 없음): {profile}")
                continue
            profile_mask = parse_bool_series(membership[profile])
            profile_lookup = dict(zip(membership["feature"].astype(str), profile_mask))
            profile_features = [feature for feature in features if bool(profile_lookup.get(feature, False))]
            if not profile_features:
                log(f"profile benchmark 건너뜀(비어 있음): {profile}")
                continue
            enabled = tuple(feature_to_index[feature] for feature in profile_features)
            dropped = tuple(feature for feature in features if feature not in set(profile_features))
            for backend in backends:
                for seed in seeds:
                    for fold in selected_folds:
                        specs.append(
                            TaskSpec(
                                backend=backend,
                                stage="profiles",
                                test_type="profile_benchmark",
                                condition_id=f"PROFILE::{profile}",
                                fold_id=fold.fold_id,
                                seed=seed,
                                enabled_feature_indices=enabled,
                                dropped_features=dropped,
                                profile_name=profile,
                            )
                        )

    if "feature_loo" in stages:
        for feature_index, feature in enumerate(features):
            enabled = tuple(index for index in all_indices if index != feature_index)
            for backend in backends:
                for seed in seeds:
                    for fold in selected_folds:
                        specs.append(
                            TaskSpec(
                                backend=backend,
                                stage="feature_loo",
                                test_type="single_feature_loo",
                                condition_id=f"LOO::{feature}",
                                fold_id=fold.fold_id,
                                seed=seed,
                                enabled_feature_indices=enabled,
                                dropped_features=(feature,),
                                representative_feature=feature,
                            )
                        )

    if "group_loo" in stages:
        audit_index = audit.drop_duplicates("feature").set_index("feature")
        groups: dict[str, list[str]] = defaultdict(list)
        for feature in features:
            group = str(audit_index.at[feature, "group"]) if "group" in audit_index.columns and pd.notna(audit_index.at[feature, "group"]) else feature.split("_")[1] if "_" in feature else "other"
            groups[group].append(feature)
        for group, group_features in sorted(groups.items()):
            dropped_set = set(group_features)
            enabled = tuple(index for index, feature in enumerate(features) if feature not in dropped_set)
            if not enabled:
                continue
            for backend in backends:
                for seed in seeds:
                    for fold in selected_folds:
                        specs.append(
                            TaskSpec(
                                backend=backend,
                                stage="group_loo",
                                test_type="feature_group_loo",
                                condition_id=f"GROUP_LOO::{group}",
                                fold_id=fold.fold_id,
                                seed=seed,
                                enabled_feature_indices=enabled,
                                dropped_features=tuple(sorted(group_features)),
                                feature_group=group,
                            )
                        )

    if "cluster_loo" in stages:
        cluster_column = "primary_cluster_id" if "primary_cluster_id" in audit.columns else "cluster_id"
        if cluster_column not in audit.columns:
            log("cluster_loo 건너뜀: pre-model audit에 cluster ID 없음")
        else:
            audit_index = audit.drop_duplicates("feature").set_index("feature")
            clusters: dict[int, list[str]] = defaultdict(list)
            for feature in features:
                value = audit_index.at[feature, cluster_column]
                if pd.isna(value):
                    continue
                clusters[int(value)].append(feature)
            for cluster_id, cluster_features in sorted(clusters.items()):
                if len(cluster_features) < args.cluster_min_size:
                    continue
                dropped_set = set(cluster_features)
                enabled = tuple(index for index, feature in enumerate(features) if feature not in dropped_set)
                if not enabled:
                    continue
                representative = None
                rep_candidates = audit_index.loc[cluster_features]
                if "is_primary_representative" in rep_candidates.columns:
                    matches = rep_candidates[parse_bool_series(rep_candidates["is_primary_representative"])]
                    if len(matches):
                        representative = str(matches.index[0])
                for backend in backends:
                    for seed in seeds:
                        for fold in selected_folds:
                            specs.append(
                                TaskSpec(
                                    backend=backend,
                                    stage="cluster_loo",
                                    test_type="cluster_loo",
                                    condition_id=f"CLUSTER_LOO::{cluster_id}",
                                    fold_id=fold.fold_id,
                                    seed=seed,
                                    enabled_feature_indices=enabled,
                                    dropped_features=tuple(sorted(cluster_features)),
                                    representative_feature=representative,
                                    cluster_id=cluster_id,
                                )
                            )
    return specs


def execute_tasks(
    args: argparse.Namespace,
    payloads: Sequence[dict[str, Any]],
    context_path: Path,
    status: RunStatus,
) -> tuple[int, int, int]:
    pending = [payload for payload in payloads if not (args.resume and result_file_is_valid(Path(payload["result_path"]), payload["identity_hash"]))]
    cached = len(payloads) - len(pending)
    completed = 0
    failed = 0
    if not pending:
        return completed, failed, cached
    lightgbm_payloads = [payload for payload in pending if payload["backend"] == "lightgbm_cpu"]
    gpu_payloads = [payload for payload in pending if payload["backend"] == "xgboost_gpu"]
    xgb_cpu_payloads = [payload for payload in pending if payload["backend"] == "xgboost_cpu"]
    total = len(pending)
    started = time.monotonic()

    def update_progress(result: Mapping[str, Any]) -> None:
        nonlocal completed, failed
        if result.get("status") in {"completed", "cached"}:
            completed += 1
        else:
            failed += 1
        done = completed + failed
        if done == 1 or done % args.progress_every == 0 or done == total:
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            status.stage(
                "model_tasks",
                "running",
                total=total,
                completed=completed,
                failed=failed,
                cached=cached,
                rate_tasks_per_second=rate,
            )
            log(f"model tasks {done}/{total} complete={completed} failed={failed} cached={cached}")

    if lightgbm_payloads:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(context_path),),
        ) as executor:
            futures = {executor.submit(_run_worker_task, payload): payload for payload in lightgbm_payloads}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except BaseException as exc:
                    result = {"status": "failed", "error": str(exc)}
                update_progress(result)

    for group in (gpu_payloads, xgb_cpu_payloads):
        if not group:
            continue
        _init_worker(str(context_path))
        for payload in group:
            result = _run_worker_task(payload)
            update_progress(result)

    return completed, failed, cached


def collect_results(
    result_root: Path,
    expected_identity_hashes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_records: list[dict[str, Any]] = []
    scope_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    for path in sorted(result_root.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception as exc:
            if expected_identity_hashes is None or path.stem in expected_identity_hashes:
                failed_records.append(
                    {
                        "result_path": str(path),
                        "status": "invalid_result_json",
                        "identity_hash": path.stem,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            continue
        identity_hash = str(payload.get("identity_hash", ""))
        if expected_identity_hashes is not None and identity_hash not in expected_identity_hashes:
            continue
        if payload.get("status") != "completed":
            failed_records.append({"result_path": str(path), **payload})
            continue
        if not payload_checksum_is_valid(payload):
            failed_records.append(
                {
                    "result_path": str(path),
                    **payload,
                    "status": "invalid_payload_checksum",
                }
            )
            continue
        base = {
            key: payload.get(key)
            for key in [
                "status",
                "identity_hash",
                "task_id",
                "backend",
                "stage",
                "test_type",
                "condition_id",
                "fold_id",
                "fold_role",
                "seed",
                "best_iteration",
                "feature_count_total",
                "enabled_feature_count",
                "representative_feature",
                "cluster_id",
                "feature_group",
                "profile_name",
                "train_rows",
                "validation_rows",
                "elapsed_seconds",
                "prediction_path",
            ]
        }
        dropped = payload.get("dropped_features") or []
        base["dropped_feature_count"] = len(dropped)
        base["dropped_features"] = "|".join(str(value) for value in dropped)
        base.update(payload.get("metrics", {}))
        model_records.append(base)
        for scope in payload.get("scope_metrics", []):
            scope_records.append({**base, **scope})
    return (
        pd.DataFrame.from_records(model_records),
        pd.DataFrame.from_records(scope_records),
        pd.DataFrame.from_records(failed_records),
    )


def build_paired_deltas(metrics: pd.DataFrame, top_fractions: Sequence[float]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    baseline = metrics[metrics["test_type"].eq("baseline")].copy()
    baseline_keys = ["backend", "fold_id", "seed"]
    if baseline.duplicated(baseline_keys).any():
        duplicated = baseline[baseline.duplicated(baseline_keys, keep=False)][baseline_keys]
        raise ValueError(f"baseline 중복: {duplicated.head().to_dict('records')}")
    higher_better = ["pr_auc", "roc_auc", "pr_auc_lift"]
    lower_better = ["brier", "logloss", "ece_10"]
    for fraction in top_fractions:
        label = fraction_label(fraction)
        for prefix in ("daily", "pooled"):
            higher_better.extend(
                [
                    f"{prefix}_top_{label}_precision",
                    f"{prefix}_top_{label}_recall",
                    f"{prefix}_top_{label}_lift",
                ]
            )
    metric_columns = [column for column in higher_better + lower_better if column in metrics.columns]
    baseline_columns = baseline_keys + metric_columns
    base = baseline[baseline_columns].rename(columns={column: f"baseline_{column}" for column in metric_columns})
    ablated = metrics[~metrics["test_type"].eq("baseline")].copy()
    paired = ablated.merge(base, on=baseline_keys, how="left", validate="many_to_one")
    for column in higher_better:
        if column in paired.columns:
            paired[f"utility_{column}"] = paired[f"baseline_{column}"] - paired[column]
    for column in lower_better:
        if column in paired.columns:
            paired[f"utility_{column}"] = paired[column] - paired[f"baseline_{column}"]
    paired["primary_utility"] = paired.get("utility_pr_auc")
    return paired


def summarize_ablation(
    paired: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    top_fractions: Sequence[float],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    identity_columns = [
        "backend",
        "stage",
        "test_type",
        "condition_id",
        "representative_feature",
        "cluster_id",
        "feature_group",
        "profile_name",
        "dropped_features",
        "dropped_feature_count",
    ]
    utility_columns = [column for column in paired.columns if column.startswith("utility_")]
    per_fold = (
        paired.groupby(identity_columns + ["fold_id", "fold_role"], dropna=False, sort=False)[utility_columns]
        .mean()
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    group_keys = identity_columns
    for key, part in per_fold.groupby(group_keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_keys, key))
        primary = part["utility_pr_auc"].to_numpy(dtype=np.float64)
        finite = primary[np.isfinite(primary)]
        record.update(
            {
                "fold_count": int(len(finite)),
                "mean_pr_auc_utility": float(np.mean(finite)) if len(finite) else np.nan,
                "median_pr_auc_utility": float(np.median(finite)) if len(finite) else np.nan,
                "std_pr_auc_utility": float(np.std(finite, ddof=0)) if len(finite) else np.nan,
                "positive_fold_ratio": float(np.mean(finite > 0)) if len(finite) else np.nan,
                "negative_fold_ratio": float(np.mean(finite < 0)) if len(finite) else np.nan,
                "worst_fold_utility": float(np.min(finite)) if len(finite) else np.nan,
                "best_fold_utility": float(np.max(finite)) if len(finite) else np.nan,
                "exact_sign_flip_p": exact_sign_flip_p(finite),
            }
        )
        ci_low, ci_high = bootstrap_mean_ci(
            finite,
            seed=deterministic_seed(*[str(value) for value in key], base=bootstrap_seed),
            repetitions=bootstrap_repetitions,
        )
        record["ci95_low"] = ci_low
        record["ci95_high"] = ci_high
        for role in ("selection", "confirmation", "recent_audit"):
            role_part = part[part["fold_id"].isin({int(value) for value in roles.get(role, [])})]
            values = role_part["utility_pr_auc"].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"{role}_fold_count"] = int(len(values))
            record[f"{role}_mean_pr_auc_utility"] = float(np.mean(values)) if len(values) else np.nan
            record[f"{role}_median_pr_auc_utility"] = float(np.median(values)) if len(values) else np.nan
            record[f"{role}_positive_fold_ratio"] = float(np.mean(values > 0)) if len(values) else np.nan
            record[f"{role}_worst_fold_utility"] = float(np.min(values)) if len(values) else np.nan
            record[f"{role}_sign_flip_p"] = exact_sign_flip_p(values)
            role_ci_low, role_ci_high = bootstrap_mean_ci(
                values,
                seed=deterministic_seed(*[str(value) for value in key], role, base=bootstrap_seed),
                repetitions=bootstrap_repetitions,
            )
            record[f"{role}_ci95_low"] = role_ci_low
            record[f"{role}_ci95_high"] = role_ci_high
        for column in utility_columns:
            values = part[column].to_numpy(dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            record[f"mean_{column}"] = float(np.mean(finite_values)) if len(finite_values) else np.nan
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        summary["selection_bh_q"] = (
            summary.groupby(["backend", "test_type"], dropna=False)["selection_sign_flip_p"]
            .transform(benjamini_hochberg)
        )
        summary["allfold_bh_q"] = (
            summary.groupby(["backend", "test_type"], dropna=False)["exact_sign_flip_p"]
            .transform(benjamini_hochberg)
        )
        summary["selection_evidence_class"] = summary.apply(classify_selection_evidence, axis=1)
        summary["confirmation_result"] = summary.apply(lambda row: classify_holdout_result(row, "confirmation"), axis=1)
        summary["recent_result"] = summary.apply(lambda row: classify_holdout_result(row, "recent_audit"), axis=1)
        summary["final_review_tier"] = summary.apply(classify_final_tier, axis=1)
    return summary


def classify_selection_evidence(row: pd.Series) -> str:
    mean = float(row.get("selection_mean_pr_auc_utility", np.nan))
    positive = float(row.get("selection_positive_fold_ratio", np.nan))
    ci_low = float(row.get("selection_ci95_low", np.nan))
    if not np.isfinite(mean) or not np.isfinite(positive):
        return "INSUFFICIENT"
    if mean >= 0.001 and positive >= 0.8 and (not np.isfinite(ci_low) or ci_low > -0.001):
        return "USEFUL_SELECTION"
    if mean <= -0.001 and positive <= 0.2:
        return "HARMFUL_SELECTION"
    if mean > 0 and positive >= 0.6:
        return "WEAK_USEFUL_SELECTION"
    if mean < 0 and positive <= 0.4:
        return "WEAK_HARMFUL_SELECTION"
    return "INCONCLUSIVE_SELECTION"


def classify_holdout_result(row: pd.Series, role: str) -> str:
    count = int(row.get(f"{role}_fold_count", 0) or 0)
    mean = float(row.get(f"{role}_mean_pr_auc_utility", np.nan))
    positive = float(row.get(f"{role}_positive_fold_ratio", np.nan))
    if count == 0 or not np.isfinite(mean):
        return "NOT_AVAILABLE"
    if mean > 0 and positive >= 1.0 - 1e-12:
        return "SUPPORTED"
    if mean < 0 and positive <= 1e-12:
        return "CONTRADICTED"
    return "MIXED"


def classify_final_tier(row: pd.Series) -> str:
    selection = str(row.get("selection_evidence_class"))
    confirmation = str(row.get("confirmation_result"))
    recent = str(row.get("recent_result"))
    if selection == "USEFUL_SELECTION" and confirmation == "SUPPORTED" and recent == "SUPPORTED":
        return "KEEP_STRONG"
    if selection in {"USEFUL_SELECTION", "WEAK_USEFUL_SELECTION"} and confirmation != "CONTRADICTED":
        return "KEEP_CONDITIONAL"
    if selection == "HARMFUL_SELECTION" and confirmation == "CONTRADICTED":
        return "DROP_HARMFUL"
    if selection in {"HARMFUL_SELECTION", "WEAK_HARMFUL_SELECTION"} and recent == "CONTRADICTED":
        return "DROP_CANDIDATE"
    return "HOLD_FOR_REVIEW"


def merge_feature_metadata(feature_summary: pd.DataFrame, pre_model_audit: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return feature_summary
    feature_rows = feature_summary[feature_summary["test_type"].eq("single_feature_loo")].copy()
    feature_rows["feature"] = feature_rows["representative_feature"].astype("string")
    metadata_columns = [
        "feature",
        "surge_priority_rank",
        "strict_stable_rank",
        "selection_direction_source",
        "selection_support_metric",
        "selection_abs_strength",
        "selection_min_fold_abs_corr",
        "strict_stable_supported",
        "train_validation_aligned",
        "train_validation_fold_sign_match_ratio",
        "selection_train_selected_metric_mean",
        "selection_validation_selected_metric_mean",
        "metric_consistent_target_relation_class",
        "metric_consistent_directional_score",
        "selection_daily_top_1pct_lift_mean",
        "selection_daily_top_3pct_lift_mean",
        "selection_daily_top_5pct_lift_mean",
        "selection_daily_top_10pct_lift_mean",
        "selection_oriented_roc_auc_mean",
        "selection_oriented_average_precision_mean",
        "primary_cluster_id",
        "primary_cluster_size",
        "is_primary_representative",
        "group",
        "missing_ratio",
    ]
    available = [column for column in metadata_columns if column in pre_model_audit.columns]
    metadata = pre_model_audit[available].drop_duplicates("feature")
    return feature_rows.merge(metadata, on="feature", how="left", validate="many_to_one")


def build_ticker_sensitivity(scope_metrics: pd.DataFrame, model_metrics: pd.DataFrame) -> pd.DataFrame:
    if scope_metrics.empty or "scope_type" not in scope_metrics.columns:
        return pd.DataFrame()
    ticker = scope_metrics[scope_metrics["scope_type"].eq("ticker")].copy()
    if ticker.empty:
        return pd.DataFrame()
    baselines = ticker[ticker["test_type"].eq("baseline")][
        ["backend", "fold_id", "seed", "scope_value", "pr_auc"]
    ].rename(columns={"pr_auc": "baseline_pr_auc"})
    ablated = ticker[ticker["test_type"].eq("single_feature_loo")].copy()
    paired = ablated.merge(
        baselines,
        on=["backend", "fold_id", "seed", "scope_value"],
        how="left",
        validate="many_to_one",
    )
    paired["pr_auc_utility"] = paired["baseline_pr_auc"] - paired["pr_auc"]
    paired["feature"] = paired["representative_feature"]
    summary = (
        paired.groupby(["backend", "feature", "scope_value"], dropna=False)["pr_auc_utility"]
        .mean()
        .reset_index()
    )
    return summary.pivot_table(index=["backend", "feature"], columns="scope_value", values="pr_auc_utility", aggfunc="first").reset_index()


def validate_completion(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    backends: Sequence[str],
    seeds: Sequence[int],
    stages: Sequence[str],
) -> dict[str, Any]:
    selected_fold_ids = args.fold_ids or [fold.fold_id for fold in folds]
    audit: dict[str, Any] = {}
    if "feature_loo" in stages:
        expected_per_backend = len(features) * len(selected_fold_ids) * len(seeds)
        for backend in backends:
            completed = int(
                len(
                    metrics[
                        metrics["backend"].eq(backend)
                        & metrics["test_type"].eq("single_feature_loo")
                    ]
                )
            )
            audit[backend] = {
                "expected_feature_loo_tasks": expected_per_backend,
                "completed_feature_loo_tasks": completed,
                "complete": completed == expected_per_backend,
            }
            if not args.allow_partial and completed != expected_per_backend:
                raise RuntimeError(
                    f"{backend} 전체 피처 LOO 미완료: {completed}/{expected_per_backend}"
                )
    return audit


def write_report(
    output: Path,
    run_summary: Mapping[str, Any],
    feature_master: pd.DataFrame,
    profile_summary: pd.DataFrame,
) -> None:
    lines = [
        "# CrashWatch Surge 전체 피처 이탈 결과",
        "",
        f"- 상태: {run_summary.get('status')}",
        f"- 피처 수: {run_summary.get('feature_count')}",
        f"- fold 수: {run_summary.get('fold_count')}",
        f"- backends: {', '.join(run_summary.get('backends', []))}",
        f"- 완료 모델: {run_summary.get('completed_model_count')}",
        f"- 실패 모델: {run_summary.get('failed_model_count')}",
        "",
        "## 효과 부호",
        "",
        "`utility_pr_auc = baseline PR-AUC - ablated PR-AUC`이며 양수일수록 제거된 피처가 유용했다는 뜻이다.",
        "Brier/logloss/ECE는 `ablated - baseline`으로 계산해 역시 양수가 유용한 방향이다.",
        "",
        "## Pre-model gate",
        "",
        "모든 모델은 train-validation 방향 정렬, 동일 metric 급등/급락 방향성 분류, train-fixed univariate top-k lift가 완료된 후 실행됐다.",
        "",
    ]
    if not feature_master.empty:
        lines.extend(["## LightGBM 기준 상위 유지 후보", ""])
        subset = feature_master.sort_values("selection_mean_pr_auc_utility", ascending=False).head(20)
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: selection ΔPR-AUC={getattr(row, 'selection_mean_pr_auc_utility', float('nan')):.6f}, "
                f"confirmation={getattr(row, 'confirmation_result', '')}, recent={getattr(row, 'recent_result', '')}, "
                f"tier={getattr(row, 'final_review_tier', '')}"
            )
        lines.append("")
    if not profile_summary.empty:
        lines.extend(["## 프로필 비교", ""])
        for row in profile_summary.sort_values("mean_pr_auc_utility", ascending=False).itertuples(index=False):
            lines.append(
                f"- `{getattr(row, 'profile_name', '')}`: baseline 대비 평균 ΔPR-AUC={getattr(row, 'mean_pr_auc_utility', float('nan')):.6f}"
            )
    (output / "SURGE_FEATURE_ABLATION_GUIDE_KO.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    package_root = args.package_root.resolve()
    resolve_default_paths(package_root, args)
    for name in ("dataset", "target_sidecar", "folds", "correlation_dir", "pre_model_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_free_disk(args.output, args.minimum_free_disk_gb)
    require_paths(
        {
            "dataset": args.dataset,
            "target sidecar": args.target_sidecar,
            "folds": args.folds,
            "correlation dir": args.correlation_dir,
            "pre-model dir": args.pre_model_dir,
        }
    )
    with FileLock(args.output / ".surge_ablation.lock"):
        status = RunStatus(args.output, "surge_all_feature_ablation_v3")
        started = time.monotonic()
        try:
            status.stage("pre_model_gate_validation", "running")
            pre_model_manifest, membership, pre_model_audit = load_pre_model_gate(
                args.pre_model_dir,
                args.dataset,
                args.target_sidecar,
            )
            roles = {str(key): [int(value) for value in values] for key, values in pre_model_manifest["fold_roles"].items()}
            folds = load_folds(args.folds)
            validate_role_assignments(roles, folds, require_selection=True)
            requested_fold_ids = parse_int_list(args.fold_ids_text)
            args.fold_ids = requested_fold_ids
            if requested_fold_ids:
                unknown = set(requested_fold_ids) - {fold.fold_id for fold in folds}
                if unknown:
                    raise ValueError(f"존재하지 않는 fold ID: {sorted(unknown)}")
            active_fold_ids = set(requested_fold_ids) if requested_fold_ids else {fold.fold_id for fold in folds}
            active_folds = [fold for fold in folds if fold.fold_id in active_fold_ids]
            if not active_folds:
                raise ValueError("실행할 fold가 없습니다")
            args.scope_columns = [token.strip() for token in args.scope_columns_text.split(",") if token.strip()]
            args.top_fractions = sorted(set(parse_float_list(args.top_fractions_text)))
            args.lightgbm_params = json.loads(args.lightgbm_params_json)
            args.xgboost_params = json.loads(args.xgboost_params_json)
            features = select_feature_universe(membership, pre_model_audit, args.baseline_profile, args.limit_features)
            if args.baseline_profile == "P0_ALL_VALID" and args.limit_features <= 0 and len(features) != pre_model_manifest["feature_count"]:
                raise RuntimeError(
                    f"전체 피처 universe 불일치: {len(features)} != {pre_model_manifest['feature_count']}"
                )
            status.stage(
                "pre_model_gate_validation",
                "complete",
                feature_count=len(features),
                gate_manifest_sha256=sha256_file(args.pre_model_dir / "PRE_MODEL_GATE_MANIFEST.json"),
            )

            status.stage("matrix_cache", "running")
            cache_manifest = build_matrix_cache(args, features, folds, roles, pre_model_manifest)
            status.stage("matrix_cache", "complete", cache_hash=cache_manifest["cache_hash"])

            backends = [token.strip() for token in args.backends.split(",") if token.strip()]
            allowed_backends = {"lightgbm_cpu", "xgboost_gpu", "xgboost_cpu"}
            if not backends or not set(backends).issubset(allowed_backends):
                raise ValueError(f"지원 backend: {sorted(allowed_backends)}, 입력={backends}")
            backend_audit = verify_backend_availability(backends)
            atomic_write_json(args.output / "backend_preflight.json", backend_audit)
            seeds = parse_int_list(args.seeds)
            if not seeds:
                raise ValueError("seed가 비어 있습니다")
            stages = [token.strip() for token in args.stages.split(",") if token.strip()]
            allowed_stages = {"baseline", "profiles", "feature_loo", "group_loo", "cluster_loo"}
            if not set(stages).issubset(allowed_stages):
                raise ValueError(f"지원 stage: {sorted(allowed_stages)}")
            if "baseline" not in stages:
                stages.insert(0, "baseline")

            status.stage("best_iterations", "running")
            if args.dry_run:
                best_iterations = {
                    backend: {
                        str(fold.fold_id): int(
                            args.xgb_fallback_iterations
                            if backend.startswith("xgboost")
                            else args.fallback_iterations
                        )
                        for fold in active_folds
                    }
                    for backend in backends
                }
                status.stage(
                    "best_iterations",
                    "skipped_dry_run",
                    note="dry-run에서는 모델 tuning을 실행하지 않고 fallback iteration으로 task plan만 생성",
                    best_iterations=best_iterations,
                )
            else:
                best_iterations = resolve_best_iterations(args, cache_manifest, active_folds, backends, seeds)
                status.stage("best_iterations", "complete", best_iterations=best_iterations)

            starter_dir = Path(__file__).resolve().parent
            run_identity = {
                "schema": "surge_all_feature_ablation_v3",
                "dataset_sha256": pre_model_manifest["dataset_sha256"],
                "target_sha256": pre_model_manifest["target_sha256"],
                "pre_model_gate_sha256": sha256_file(args.pre_model_dir / "PRE_MODEL_GATE_MANIFEST.json"),
                "matrix_cache_hash": cache_manifest["cache_hash"],
                "feature_hash": hash_strings(features),
                "feature_count": len(features),
                "folds": [fold.to_dict() for fold in folds],
                "fold_roles": roles,
                "fold_ids": requested_fold_ids,
                "backends": backends,
                "backend_audit": backend_audit,
                "model_versions": model_versions(),
                "seeds": seeds,
                "stages": stages,
                "best_iterations": best_iterations,
                "lightgbm_params": args.lightgbm_params,
                "xgboost_params": args.xgboost_params,
                "top_fractions": args.top_fractions,
                "baseline_profile": args.baseline_profile,
                "profile_benchmarks": [
                    token.strip() for token in args.profile_benchmarks.split(",") if token.strip()
                ],
                "cluster_min_size": int(args.cluster_min_size),
                "scope_columns": list(args.scope_columns),
                "minimum_scope_rows": int(args.minimum_scope_rows),
                "save_all_predictions": bool(args.save_all_predictions),
                "workers": int(args.workers),
                "threads_per_worker": int(args.threads_per_worker),
                "xgboost_threads": int(args.xgboost_threads),
                "retry_failed": int(args.retry_failed),
                "allow_partial": bool(args.allow_partial),
                "bootstrap_repetitions": int(args.bootstrap_repetitions),
                "bootstrap_seed": int(args.bootstrap_seed),
                "columns": {
                    "target": args.target_column,
                    "target_valid": args.target_valid_column,
                    "date": args.date_column,
                    "ticker": args.ticker_column,
                },
                "minimum_purge_trading_days": int(args.minimum_purge_trading_days),
                "tuning": {
                    "windows": int(args.tuning_windows),
                    "step_days": int(args.tuning_step_days),
                    "inner_validation_days": int(args.inner_validation_days),
                    "inner_purge_days": int(args.inner_purge_days),
                    "minimum_inner_train_days": int(args.minimum_inner_train_days),
                    "lightgbm_max_tuning_rounds": int(args.max_tuning_rounds),
                    "lightgbm_early_stopping_rounds": int(args.early_stopping_rounds),
                    "lightgbm_minimum_iterations": int(args.minimum_iterations),
                    "lightgbm_maximum_effective_iterations": int(args.maximum_effective_iterations),
                    "lightgbm_fallback_iterations": int(args.fallback_iterations),
                    "xgboost_max_tuning_rounds": int(args.xgb_max_tuning_rounds),
                    "xgboost_early_stopping_rounds": int(args.xgb_early_stopping_rounds),
                    "xgboost_minimum_iterations": int(args.xgb_minimum_iterations),
                    "xgboost_maximum_effective_iterations": int(args.xgb_maximum_effective_iterations),
                    "xgboost_fallback_iterations": int(args.xgb_fallback_iterations),
                },
                "code_sha256": {
                    "runner": sha256_file(Path(__file__).resolve()),
                    "common": sha256_file(starter_dir / "surge_ablation_common.py"),
                },
            }
            task_model_identity = {
                "schema": "surge_ablation_model_task_context_v3",
                "dataset_sha256": run_identity["dataset_sha256"],
                "target_sha256": run_identity["target_sha256"],
                "pre_model_gate_sha256": run_identity["pre_model_gate_sha256"],
                "matrix_cache_hash": run_identity["matrix_cache_hash"],
                "feature_hash": run_identity["feature_hash"],
                "feature_count": run_identity["feature_count"],
                "folds": run_identity["folds"],
                "fold_roles": run_identity["fold_roles"],
                "lightgbm_params": run_identity["lightgbm_params"],
                "xgboost_params": run_identity["xgboost_params"],
                "top_fractions": run_identity["top_fractions"],
                "scope_columns": run_identity["scope_columns"],
                "minimum_scope_rows": run_identity["minimum_scope_rows"],
                "threads_per_worker": run_identity["threads_per_worker"],
                "xgboost_threads": run_identity["xgboost_threads"],
                "columns": run_identity["columns"],
                "minimum_purge_trading_days": run_identity["minimum_purge_trading_days"],
                "model_versions": run_identity["model_versions"],
                "code_sha256": run_identity["code_sha256"],
            }
            task_run_signature = sha256_bytes(stable_json_bytes(task_model_identity))[:24]
            run_identity["task_model_identity"] = task_model_identity
            run_identity["task_run_signature"] = task_run_signature
            run_signature = sha256_bytes(stable_json_bytes(run_identity))[:24]
            atomic_write_json(
                args.output / "resolved_run_config.json",
                {**run_identity, "run_signature": run_signature},
            )

            specs = generate_task_specs(
                args,
                features,
                membership,
                pre_model_audit,
                folds,
                roles,
                backends,
                seeds,
                stages,
            )
            result_root = args.output / "task_results"
            prediction_dir = args.output / "prediction_cache"
            payloads: list[dict[str, Any]] = []
            for spec in specs:
                best_iteration = int(best_iterations[spec.backend][str(spec.fold_id)])
                save_prediction = spec.test_type == "baseline" or args.save_all_predictions
                payloads.append(
                    build_task_payload(
                        spec,
                        features,
                        best_iteration,
                        roles,
                        task_run_signature,
                        result_root,
                        save_prediction,
                    )
                )
            expected = pd.DataFrame(
                [
                    {
                        "identity_hash": payload["identity_hash"],
                        "backend": payload["backend"],
                        "stage": payload["stage"],
                        "test_type": payload["test_type"],
                        "condition_id": payload["condition_id"],
                        "fold_id": payload["fold_id"],
                        "fold_role": payload["fold_role"],
                        "seed": payload["seed"],
                        "enabled_feature_count": len(payload["enabled_feature_indices"]),
                        "dropped_feature_count": len(payload["dropped_features"]),
                        "result_path": payload["result_path"],
                    }
                    for payload in payloads
                ]
            )
            atomic_write_csv(args.output / "task_manifest.csv", expected)
            if args.dry_run:
                status.success(dry_run=True, planned_task_count=len(payloads))
                log(f"dry-run 완료: planned tasks={len(payloads)}")
                return

            context = {
                "matrix_path": cache_manifest["artifacts"]["matrix"]["path"],
                "arrays_path": cache_manifest["artifacts"]["arrays"]["path"],
                "fold_indices_path": cache_manifest["artifacts"]["fold_indices"]["path"],
                "prediction_dir": str(prediction_dir),
                "threads_per_worker": args.threads_per_worker,
                "xgboost_threads": args.xgboost_threads,
                "lightgbm_params": args.lightgbm_params,
                "xgboost_params": args.xgboost_params,
                "top_fractions": args.top_fractions,
                "scope_codebooks": cache_manifest.get("scope_codebooks", {}),
                "minimum_scope_rows": args.minimum_scope_rows,
            }
            context_path = args.output / "worker_context.json"
            atomic_write_json(context_path, context)

            status.stage("model_tasks", "running", planned=len(payloads))
            completed, failed, cached = execute_tasks(args, payloads, context_path, status)
            if failed and args.retry_failed > 0:
                log(f"실패 {failed}건을 최대 {args.retry_failed}회 재시도")
                for retry in range(args.retry_failed):
                    retry_payloads = [
                        payload
                        for payload in payloads
                        if not result_file_is_valid(
                            Path(payload["result_path"]),
                            str(payload["identity_hash"]),
                        )
                    ]
                    if not retry_payloads:
                        failed = 0
                        break
                    retry_completed, retry_failed, retry_cached = execute_tasks(
                        args,
                        retry_payloads,
                        context_path,
                        status,
                    )
                    completed += retry_completed
                    failed = retry_failed
                    cached += retry_cached
                    log(
                        f"retry {retry + 1}: attempted={len(retry_payloads)}, "
                        f"completed={retry_completed}, failed={retry_failed}, cached={retry_cached}"
                    )
            status.stage("model_tasks", "complete", completed=completed, failed=failed, cached=cached)

            status.stage("aggregation", "running")
            expected_identity_hashes = {str(payload["identity_hash"]) for payload in payloads}
            model_metrics, scope_metrics, failed_tasks = collect_results(
                result_root, expected_identity_hashes=expected_identity_hashes
            )
            if not model_metrics.empty:
                model_metrics.sort_values(["backend", "stage", "condition_id", "fold_id", "seed"], inplace=True)
            atomic_write_csv(args.output / "all_model_metrics.csv", model_metrics)
            atomic_write_csv(args.output / "all_scope_metrics.csv", scope_metrics)
            atomic_write_csv(args.output / "failed_tasks.csv", failed_tasks)

            completed_identity_hashes = set(model_metrics.get("identity_hash", pd.Series(dtype="string")).astype(str))
            missing_identity_hashes = expected_identity_hashes - completed_identity_hashes
            if missing_identity_hashes and not args.allow_partial:
                raise RuntimeError(
                    f"계획 task 중 결과가 없는 항목 {len(missing_identity_hashes)}건: "
                    f"{sorted(missing_identity_hashes)[:10]}"
                )
            completion_audit = validate_completion(args, model_metrics, features, folds, backends, seeds, stages)
            selected_fold_count = len(args.fold_ids or [fold.fold_id for fold in folds])
            expected_baseline_tasks = len(backends) * len(seeds) * selected_fold_count
            completed_baseline_tasks = int(model_metrics["test_type"].eq("baseline").sum()) if not model_metrics.empty else 0
            primary_completion_payload = {
                "schema": "crashwatch_surge_primary_completion_audit_v3",
                "status": "COMPLETE" if all(
                    bool(record.get("complete", True)) for record in completion_audit.values()
                ) and completed_baseline_tasks == expected_baseline_tasks else "INCOMPLETE",
                "feature_count": len(features),
                "fold_count": selected_fold_count,
                "seed_count": len(seeds),
                "backends": backends,
                "expected_baseline_tasks": expected_baseline_tasks,
                "completed_baseline_tasks": completed_baseline_tasks,
                "feature_loo": completion_audit,
            }
            atomic_write_json(args.output / "primary_completion_audit.json", primary_completion_payload)
            if not args.allow_partial and primary_completion_payload["status"] != "COMPLETE":
                raise RuntimeError(f"필수 baseline/feature LOO 완결성 실패: {primary_completion_payload}")

            paired = build_paired_deltas(model_metrics, args.top_fractions)
            atomic_write_csv(args.output / "paired_ablation_deltas.csv", paired)
            expected_paired_rows = int(len(model_metrics) - completed_baseline_tasks)
            baseline_missing_rows = int(
                paired["baseline_pr_auc"].isna().sum()
                if not paired.empty and "baseline_pr_auc" in paired.columns
                else expected_paired_rows
            )
            pairing_audit = {
                "schema": "crashwatch_surge_baseline_pairing_audit_v3",
                "status": "COMPLETE"
                if len(paired) == expected_paired_rows and baseline_missing_rows == 0
                else "INCOMPLETE",
                "expected_ablated_rows": expected_paired_rows,
                "paired_rows": int(len(paired)),
                "missing_baseline_pr_auc_rows": baseline_missing_rows,
                "pairing_keys": ["backend", "fold_id", "seed"],
            }
            atomic_write_json(args.output / "baseline_pairing_audit.json", pairing_audit)
            if not args.allow_partial and pairing_audit["status"] != "COMPLETE":
                raise RuntimeError(f"baseline pairing 완결성 실패: {pairing_audit}")
            summary = summarize_ablation(
                paired,
                roles,
                args.top_fractions,
                bootstrap_repetitions=args.bootstrap_repetitions,
                bootstrap_seed=args.bootstrap_seed,
            )
            atomic_write_csv(args.output / "all_ablation_summary.csv", summary)
            feature_summary = summary[summary["test_type"].eq("single_feature_loo")].copy() if not summary.empty else pd.DataFrame()
            cluster_summary = summary[summary["test_type"].eq("cluster_loo")].copy() if not summary.empty else pd.DataFrame()
            group_summary = summary[summary["test_type"].eq("feature_group_loo")].copy() if not summary.empty else pd.DataFrame()
            profile_summary = summary[summary["test_type"].eq("profile_benchmark")].copy() if not summary.empty else pd.DataFrame()
            atomic_write_csv(args.output / "feature_ablation_summary.csv", feature_summary)
            atomic_write_csv(args.output / "cluster_ablation_summary.csv", cluster_summary)
            atomic_write_csv(args.output / "group_ablation_summary.csv", group_summary)
            atomic_write_csv(args.output / "profile_benchmark_summary.csv", profile_summary)
            feature_master = merge_feature_metadata(summary, pre_model_audit)
            if not feature_master.empty:
                feature_master["selection_loo_rank"] = (
                    feature_master.groupby("backend", dropna=False)["selection_mean_pr_auc_utility"]
                    .rank(method="min", ascending=False)
                    .astype("Int64")
                )
                feature_master.sort_values(
                    ["backend", "selection_loo_rank", "feature"],
                    ascending=[True, True, True],
                    kind="mergesort",
                    inplace=True,
                )
            atomic_write_csv(args.output / "feature_master_decision.csv", feature_master)
            selection_identity_columns = {
                "backend", "stage", "test_type", "condition_id", "feature",
                "representative_feature", "feature_group", "group", "dropped_features",
                "dropped_feature_count", "selection_loo_rank", "surge_priority_rank",
                "missing_ratio", "primary_cluster_id", "primary_cluster_size",
                "is_primary_representative",
            }
            selection_only_columns = [
                column
                for column in feature_master.columns
                if column in selection_identity_columns
                or column.startswith("selection_")
                or column.startswith("train_validation_")
                or column.startswith("metric_consistent_")
            ]
            selection_ranking = feature_master[selection_only_columns].copy() if selection_only_columns else pd.DataFrame()
            atomic_write_csv(args.output / "feature_selection_ranking.csv", selection_ranking)
            holdout_columns = [
                column
                for column in [
                    "backend", "feature", "selection_loo_rank", "selection_evidence_class",
                    "selection_mean_pr_auc_utility", "selection_positive_fold_ratio",
                    "confirmation_result", "confirmation_mean_pr_auc_utility",
                    "recent_result", "recent_audit_mean_pr_auc_utility",
                    "final_review_tier", "metric_consistent_target_relation_class",
                    "train_validation_aligned", "selection_daily_top_3pct_lift_mean",
                ]
                if column in feature_master.columns
            ]
            atomic_write_csv(
                args.output / "feature_holdout_review.csv",
                feature_master[holdout_columns].copy() if holdout_columns else pd.DataFrame(),
            )
            ticker_sensitivity = build_ticker_sensitivity(scope_metrics, model_metrics)
            atomic_write_csv(args.output / "ticker_feature_sensitivity_matrix.csv", ticker_sensitivity)

            run_summary = {
                "schema": "crashwatch_surge_all_feature_ablation_v3",
                "status": "SUCCESS" if failed_tasks.empty else "PARTIAL_WITH_FAILURES",
                "created_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "run_signature": run_signature,
                "task_run_signature": task_run_signature,
                "feature_count": len(features),
                "fold_count": len(requested_fold_ids or folds),
                "backends": backends,
                "backend_preflight": backend_audit,
                "seeds": seeds,
                "stages": stages,
                "planned_model_count": len(payloads),
                "completed_model_count": int(len(model_metrics)),
                "failed_model_count": int(len(failed_tasks)),
                "completion_audit": completion_audit,
                "primary_completion_audit": primary_completion_payload,
                "baseline_pairing_audit": pairing_audit,
                "effect_sign": {
                    "higher_better_metrics": "baseline - ablated",
                    "lower_better_metrics": "ablated - baseline",
                    "positive_means_removed_input_was_useful": True,
                },
                "pre_model_gate_enforced": {
                    "train_validation_direction_alignment": True,
                    "metric_consistent_directional_class": True,
                    "train_fixed_univariate_topk_lift": True,
                },
            }
            atomic_write_json(args.output / "run_summary.json", run_summary)
            write_report(args.output, run_summary, feature_master, profile_summary)

            manifest = {
                **run_summary,
                "package_root": str(package_root),
                "dataset_path": str(args.dataset),
                "target_sidecar_path": str(args.target_sidecar),
                "dataset_sha256": pre_model_manifest["dataset_sha256"],
                "target_sha256": pre_model_manifest["target_sha256"],
                "pre_model_gate_manifest_sha256": sha256_file(args.pre_model_dir / "PRE_MODEL_GATE_MANIFEST.json"),
                "feature_hash": hash_strings(features),
                "features": list(features),
                "fold_roles": roles,
                "best_iterations": best_iterations,
                "resolved_run_config": run_identity,
                "model_versions": model_versions(),
            }
            atomic_write_json(args.output / "MODEL_FREEZE_MANIFEST.json", manifest)
            manifest["output_inventory"] = compute_output_inventory(
                args.output,
                exclude_relative_paths=[
                    "MODEL_FREEZE_MANIFEST.json",
                    "RUN_STATUS.json",
                    ".surge_ablation.lock",
                ],
            )
            atomic_write_json(args.output / "MODEL_FREEZE_MANIFEST.json", manifest)
            status.stage("aggregation", "complete", completion_audit=completion_audit)
            if not args.allow_partial and not failed_tasks.empty:
                raise RuntimeError(f"실패 task가 {len(failed_tasks)}건 남았습니다")
            status.success(
                run_signature=run_signature,
                completed_model_count=int(len(model_metrics)),
                failed_model_count=int(len(failed_tasks)),
                completion_audit=completion_audit,
                primary_completion_status=primary_completion_payload["status"],
                baseline_pairing_status=pairing_audit["status"],
            )
            log("전체 피처 이탈 파이프라인 완료")
        except BaseException as exc:
            status.failure(exc)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrashWatch Surge 439개 전체 피처 leave-one-out 이탈 실험. "
            "검증된 pre-model gate가 없으면 모델을 시작하지 않습니다."
        )
    )
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--correlation-dir", type=Path)
    parser.add_argument("--pre-model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-column", default="label_abs_surge_3d_5pct")
    parser.add_argument("--target-valid-column", default="target_valid")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--minimum-purge-trading-days", type=int, default=3)
    parser.add_argument("--baseline-profile", default="P0_ALL_VALID")
    parser.add_argument("--profile-benchmarks", default=(
        "P1_SELECTION_TOP,"
        "P2S_SELECTION_TRAIN_ALIGNED,P3S_SELECTION_CLUSTER_REP,"
        "P3S_ALLFOLD_MINABS_CLUSTER_REP,P4S_METRIC_CONSISTENT_DIRECTIONAL,"
        "P5S_COMMON_LARGE_MOVE,P6S_BALANCED_SELECTION_ONLY,P7S_UNIVARIATE_TOP3_LIFT"
    ))
    parser.add_argument("--stages", default="baseline,profiles,feature_loo")
    parser.add_argument("--backends", default="lightgbm_cpu")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--fold-ids", dest="fold_ids_text", default="")
    parser.add_argument("--limit-features", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads-per-worker", type=int, default=3)
    parser.add_argument("--xgboost-threads", type=int, default=1)
    parser.add_argument("--scope-columns", dest="scope_columns_text", default="ticker,industry_name,market,bucket")
    parser.add_argument("--minimum-scope-rows", type=int, default=20)
    parser.add_argument("--top-fractions", dest="top_fractions_text", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--cluster-min-size", type=int, default=2)
    parser.add_argument("--cache-column-block", type=int, default=64)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=4.0)

    parser.add_argument("--tuning-windows", type=int, default=3)
    parser.add_argument("--tuning-step-days", type=int, default=120)
    parser.add_argument("--inner-validation-days", type=int, default=60)
    parser.add_argument("--inner-purge-days", type=int, default=20)
    parser.add_argument("--minimum-inner-train-days", type=int, default=500)
    parser.add_argument("--max-tuning-rounds", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--minimum-iterations", type=int, default=50)
    parser.add_argument("--maximum-effective-iterations", type=int, default=320)
    parser.add_argument("--fallback-iterations", type=int, default=220)
    parser.add_argument("--xgb-max-tuning-rounds", type=int, default=500)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--xgb-minimum-iterations", type=int, default=40)
    parser.add_argument("--xgb-maximum-effective-iterations", type=int, default=300)
    parser.add_argument("--xgb-fallback-iterations", type=int, default=260)
    parser.add_argument("--lightgbm-params-json", default="{}")
    parser.add_argument("--xgboost-params-json", default="{}")

    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--retry-failed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--save-all-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
