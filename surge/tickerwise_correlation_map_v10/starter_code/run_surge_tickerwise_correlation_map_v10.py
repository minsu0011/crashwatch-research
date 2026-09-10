from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


from surge_ticker_hierarchy_v10 import (
    HierarchyConfig,
    build_hierarchical_effect_map,
    build_ticker_driver_profiles,
    build_ticker_similarity_map,
    save_ticker_map_visualizations,
)

from surge_ticker_common_v10 import (
    ERROR_A_TOP_TP,
    ERROR_B_TOP_FP,
    ERROR_C_LOW_TP,
    ERROR_D_LOW_TN,
    ERROR_GROUP_NAMES,
    FoldSpec,
    RunStatus,
    TickerFoldEligibility,
    aggregate_fold_metrics,
    apply_ticker_time_transform,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    build_ticker_error_group_codes,
    build_ticker_temporal_matched_pairs,
    cluster_from_correlation,
    compute_output_inventory,
    compute_ticker_correlation_matrix,
    datewise_percentile_rank,
    fixed_direction_auc,
    groupwise_percentile_rank,
    hash_strings,
    infer_direction_from_auc,
    join_source_and_target,
    load_feature_profile,
    load_folds,
    load_json,
    log,
    make_ticker_transform_name,
    pairwise_valid_correlation,
    percentile_rank_1d,
    probability_metrics,
    read_table,
    role_for_fold,
    safe_binary_metrics,
    safe_pearson,
    safe_roc_auc,
    safe_spearman,
    select_threshold_for_precision,
    separation_metrics,
    sha256_file,
    shrink_auc_to_half,
    stable_correlation_edges,
    summarize_fixed_direction,
    table_columns,
    ticker_precision_policy_metrics,
    utc_now,
    verify_feature_names,
    write_graphml,
)


SCHEMA_VERSION = "crashwatch_surge_tickerwise_correlation_map_v10"
ROLE_DEFAULTS = {
    "selection": [0, 1, 2, 3, 4],
    "confirmation": [5, 6],
    "recent_audit": [7],
}

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
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
    "max_leaves": 31,
    "grow_policy": "lossguide",
    "min_child_weight": 5.0,
    "reg_alpha": 0.3,
    "reg_lambda": 2.0,
    "max_bin": 256,
    "verbosity": 0,
}

TIME_TRANSFORMS = (
    "z20",
    "z60",
    "rank60",
    "change5",
    "change20",
)

CROSS_SECTIONAL_TRANSFORMS = (
    "date_rank",
    "date_market_rank",
    "date_bucket_rank",
    "date_industry_rank",
)

PROFILE_NAMES = (
    "BASE_TICKER",
    "TARGET_TOP",
    "AB_PRECISION",
    "CD_RECOVERY",
    "CLUSTER_REP",
    "COMBINED",
)


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
    if args.output is None:
        args.output = package_root / "outputs" / "surge_tickerwise_correlation_map_v10"


def require_paths(paths: Mapping[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not Path(path).exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))


def parse_int_tokens(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    return [int(item) for item in value]


def parse_str_tokens(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


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
                raise ValueError(f"Fold {fold_id} belongs to {ownership[fold_id]} and {role}")
            ownership[fold_id] = role
    if not roles["selection"]:
        raise ValueError("At least one selection fold is required")
    return roles


def load_feature_universe(args: argparse.Namespace, dataset_columns: Sequence[str]) -> list[str]:
    features = load_feature_profile(Path(args.feature_profile_manifest), args.feature_profile)
    if int(args.limit_features) > 0:
        features = features[: int(args.limit_features)]
    verify_feature_names(features, dataset_columns)
    if args.require_full_439 and len(features) != 439:
        raise RuntimeError(f"--require-full-439 is enabled but feature count is {len(features)}")
    return features


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    """Normalize numeric stock codes without altering non-numeric synthetic tickers."""

    def normalize_one(value: object) -> str:
        if pd.isna(value):
            return "UNKNOWN"
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if text.isdigit():
            return text.zfill(6)
        return text or "UNKNOWN"

    return series.map(normalize_one).astype("string")


def load_dataset_frame(args: argparse.Namespace, features: Sequence[str]) -> pd.DataFrame:
    available_source = set(table_columns(Path(args.dataset)))
    source_columns = list(
        dict.fromkeys(
            [
                "source_row_id",
                args.date_column,
                args.ticker_column,
                args.name_column,
                args.market_column,
                args.bucket_column,
                args.industry_column,
                *features,
            ]
        )
    )
    source_columns = [column for column in source_columns if column in available_source]
    source = read_table(Path(args.dataset), columns=source_columns)
    if args.ticker_column in source.columns:
        source[args.ticker_column] = normalize_ticker_series(source[args.ticker_column])
    if "source_row_id" not in source.columns:
        source = source.reset_index(drop=True)
        source["source_row_id"] = np.arange(len(source), dtype=np.int64)
    available_target = set(table_columns(Path(args.target_sidecar)))
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
        if column in available_target
    ]
    target = read_table(Path(args.target_sidecar), columns=target_columns)
    if args.ticker_column in target.columns:
        target[args.ticker_column] = normalize_ticker_series(target[args.ticker_column])
    frame = join_source_and_target(
        source,
        target,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    valid = frame[args.target_valid_column].astype(bool) & frame[args.target_column].isin([0, 1])
    frame = frame.loc[valid].copy()
    frame = frame.sort_values([args.ticker_column, args.date_column, "source_row_id"], kind="mergesort").reset_index(drop=True)
    limit_tickers = int(getattr(args, "limit_tickers", 0))
    if limit_tickers > 0:
        selected_tickers = sorted(frame[args.ticker_column].astype(str).unique().tolist())[:limit_tickers]
        frame = frame.loc[frame[args.ticker_column].astype(str).isin(selected_tickers)].reset_index(drop=True)
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in [args.ticker_column, args.name_column, args.market_column, args.bucket_column, args.industry_column]:
        if column not in frame.columns:
            frame[column] = "UNKNOWN"
        if column == args.ticker_column:
            frame[column] = normalize_ticker_series(frame[column])
        else:
            frame[column] = frame[column].astype("string").fillna("UNKNOWN")
    if frame[args.date_column].isna().any():
        raise ValueError("Date parsing produced NaT")
    if frame["source_row_id"].duplicated().any():
        raise ValueError("source_row_id must be unique after joining")
    return frame


def build_global_fold_index(
    frame: pd.DataFrame,
    folds: Sequence[FoldSpec],
    date_column: str,
) -> dict[int, dict[str, np.ndarray]]:
    dates = frame[date_column].to_numpy(dtype="datetime64[ns]")
    result: dict[int, dict[str, np.ndarray]] = {}
    for fold in folds:
        train = np.flatnonzero((dates >= fold.train_start.to_datetime64()) & (dates <= fold.train_end.to_datetime64()))
        validation = np.flatnonzero(
            (dates >= fold.validation_start.to_datetime64()) & (dates <= fold.validation_end.to_datetime64())
        )
        if train.size == 0 or validation.size == 0:
            raise RuntimeError(f"Fold {fold.fold_id} has empty train or validation rows")
        result[fold.fold_id] = {"train": train.astype(np.int64), "validation": validation.astype(np.int64)}
    return result


def build_ticker_fold_index(
    frame: pd.DataFrame,
    folds: Sequence[FoldSpec],
    global_fold_index: Mapping[int, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], pd.DataFrame]:
    ticker_values = frame[args.ticker_column].astype(str).to_numpy()
    y = frame[args.target_column].to_numpy(dtype=np.int8)
    role_lookup = {
        fold_id: role_for_fold(fold_id, {
            "selection": parse_int_tokens(args.selection_folds),
            "confirmation": parse_int_tokens(args.confirmation_folds),
            "recent_audit": parse_int_tokens(args.recent_folds),
        })
        for fold_id in [fold.fold_id for fold in folds]
    }
    result: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    for ticker in sorted(pd.unique(ticker_values).tolist()):
        ticker_mask = ticker_values == str(ticker)
        for fold in folds:
            train = np.asarray(global_fold_index[fold.fold_id]["train"], dtype=np.int64)
            validation = np.asarray(global_fold_index[fold.fold_id]["validation"], dtype=np.int64)
            train = train[ticker_mask[train]]
            validation = validation[ticker_mask[validation]]
            train_pos = int(np.sum(y[train] == 1))
            train_neg = int(np.sum(y[train] == 0))
            valid_pos = int(np.sum(y[validation] == 1))
            valid_neg = int(np.sum(y[validation] == 0))
            map_ok = bool(
                len(validation) >= int(args.minimum_validation_rows_map)
                and valid_pos >= int(args.minimum_validation_positive_map)
                and valid_neg >= int(args.minimum_validation_negative_map)
            )
            model_ok = bool(
                len(train) >= int(args.minimum_train_rows_model)
                and train_pos >= int(args.minimum_train_positive_model)
                and train_neg >= int(args.minimum_train_negative_model)
                and len(validation) >= int(args.minimum_validation_rows_model)
                and valid_pos >= int(args.minimum_validation_positive_model)
                and valid_neg >= int(args.minimum_validation_negative_model)
            )
            reasons: list[str] = []
            if len(train) < int(args.minimum_train_rows_model):
                reasons.append("TRAIN_ROWS")
            if train_pos < int(args.minimum_train_positive_model):
                reasons.append("TRAIN_POSITIVE")
            if train_neg < int(args.minimum_train_negative_model):
                reasons.append("TRAIN_NEGATIVE")
            if len(validation) < int(args.minimum_validation_rows_model):
                reasons.append("VALIDATION_ROWS")
            if valid_pos < int(args.minimum_validation_positive_model):
                reasons.append("VALIDATION_POSITIVE")
            if valid_neg < int(args.minimum_validation_negative_model):
                reasons.append("VALIDATION_NEGATIVE")
            result[str(ticker)][fold.fold_id] = {"train": train, "validation": validation}
            eligibility = TickerFoldEligibility(
                ticker=str(ticker),
                fold_id=int(fold.fold_id),
                role=str(role_lookup[fold.fold_id]),
                train_rows=int(len(train)),
                train_positive=train_pos,
                train_negative=train_neg,
                validation_rows=int(len(validation)),
                validation_positive=valid_pos,
                validation_negative=valid_neg,
                eligible_map=map_ok,
                eligible_model=model_ok,
                reason="|".join(reasons) if reasons else "OK",
            )
            rows.append(eligibility.to_dict())
    return dict(result), pd.DataFrame(rows)


def ticker_target_summary(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, part in frame.groupby(args.ticker_column, sort=True):
        y = part[args.target_column].to_numpy(dtype=int)
        rows.append(
            {
                "ticker": str(ticker),
                "rows": int(len(part)),
                "positive": int(np.sum(y == 1)),
                "negative": int(np.sum(y == 0)),
                "positive_rate": float(np.mean(y == 1)) if len(y) else float("nan"),
                "date_min": pd.Timestamp(part[args.date_column].min()).strftime("%Y-%m-%d"),
                "date_max": pd.Timestamp(part[args.date_column].max()).strftime("%Y-%m-%d"),
                "name": str(part[args.name_column].mode(dropna=True).iloc[0]) if not part[args.name_column].mode(dropna=True).empty else "UNKNOWN",
                "market": str(part[args.market_column].mode(dropna=True).iloc[0]) if not part[args.market_column].mode(dropna=True).empty else "UNKNOWN",
                "bucket": str(part[args.bucket_column].mode(dropna=True).iloc[0]) if not part[args.bucket_column].mode(dropna=True).empty else "UNKNOWN",
                "industry": str(part[args.industry_column].mode(dropna=True).iloc[0]) if not part[args.industry_column].mode(dropna=True).empty else "UNKNOWN",
            }
        )
    return pd.DataFrame(rows)


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
            log("CUDA XGBoost requested but unavailable; using CPU")
            return "cpu"
        raise RuntimeError("CUDA XGBoost requested but unavailable")
    return "cuda" if cuda_available else "cpu"


def train_lightgbm_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
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
    *,
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


def train_logistic_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10, 90))),
            (
                "logistic",
                LogisticRegression(
                    C=0.25,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_validation)[:, 1].astype(np.float64)


def train_backend_predict(
    backend: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> np.ndarray:
    if backend == "lightgbm_cpu":
        return train_lightgbm_predict(
            x_train,
            y_train,
            x_validation,
            iterations=int(args.base_iterations),
            seed=seed,
            threads=int(args.threads),
        )
    if backend in {"xgboost_gpu", "xgboost_cpu"}:
        backend_device = "cpu" if backend == "xgboost_cpu" else device
        return train_xgboost_predict(
            x_train,
            y_train,
            x_validation,
            iterations=int(args.base_iterations),
            seed=seed,
            threads=int(args.xgboost_threads),
            device=backend_device,
        )
    if backend == "logistic":
        return train_logistic_predict(x_train, y_train, x_validation, seed=seed)
    raise ValueError(f"Unsupported backend: {backend}")


def compute_train_feature_scores(
    frame: pd.DataFrame,
    indices: np.ndarray,
    features: Sequence[str],
    target_column: str,
    *,
    minimum_rows: int,
    prior_strength: float,
) -> pd.DataFrame:
    y = frame.iloc[indices][target_column].to_numpy(dtype=int)
    feature_list = list(features)
    # Materializing the ticker-fold slice once avoids hundreds of repeated
    # pandas indexing operations while preserving the per-feature formulas.
    feature_matrix = frame.iloc[indices][feature_list].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(feature_list):
        values = feature_matrix[:, feature_index]
        valid = np.isfinite(values)
        valid_y = y[valid]
        valid_values = values[valid]
        positive_n = int(np.sum(valid_y == 1))
        negative_n = int(np.sum(valid_y == 0))
        raw_auc = safe_roc_auc(valid_y, valid_values) if positive_n > 0 and negative_n > 0 else float("nan")
        oriented = max(raw_auc, 1.0 - raw_auc) if math.isfinite(raw_auc) else float("nan")
        shrunk, reliability = shrink_auc_to_half(oriented, positive_n, negative_n, prior_strength=prior_strength)
        spearman, spearman_n = safe_spearman(values, y, minimum_rows=minimum_rows)
        coverage = float(np.mean(valid)) if len(valid) else 0.0
        score = 0.0
        if math.isfinite(shrunk):
            score += max(shrunk - 0.5, 0.0) * 5.0
        if math.isfinite(spearman):
            score += abs(spearman) * 0.75
        score *= math.sqrt(max(coverage, 0.0))
        rows.append(
            {
                "feature": feature,
                "valid_n": int(valid.sum()),
                "positive_n": positive_n,
                "negative_n": negative_n,
                "coverage": coverage,
                "raw_auc": raw_auc,
                "oriented_auc": oriented,
                "shrunk_oriented_auc": shrunk,
                "auc_reliability": reliability,
                "spearman": spearman,
                "spearman_n": spearman_n,
                "train_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["train_score", "coverage", "feature"], ascending=[False, False, True], kind="mergesort"
    )


def greedy_deduplicate_features(
    frame: pd.DataFrame,
    indices: np.ndarray,
    ranked: pd.DataFrame,
    *,
    maximum_features: int,
    correlation_threshold: float,
    candidate_multiplier: int = 4,
) -> list[str]:
    candidate_count = max(int(maximum_features), int(maximum_features) * int(candidate_multiplier))
    candidates = ranked.head(candidate_count)["feature"].astype(str).tolist()
    if not candidates:
        return []
    candidate_matrix = frame.iloc[indices][candidates].to_numpy(dtype=float)
    selected: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for feature_index, feature in enumerate(candidates):
        values = candidate_matrix[:, feature_index]
        keep = True
        for peer_values in selected_arrays:
            correlation, _ = pairwise_valid_correlation(values, peer_values, method="spearman", minimum_rows=30)
            if math.isfinite(correlation) and abs(correlation) >= float(correlation_threshold):
                keep = False
                break
        if keep:
            selected.append(feature)
            selected_arrays.append(values)
        if len(selected) >= int(maximum_features):
            break
    if not selected:
        selected = candidates[: int(maximum_features)]
    return selected


def build_ticker_base_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    ticker_fold_index: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    eligibility: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = Path(args.output) / "ticker_base_oof_predictions.csv"
    manifest_path = Path(args.output) / "ticker_base_feature_manifest.csv"
    if args.resume and cache_path.exists() and manifest_path.exists():
        cached = pd.read_csv(cache_path)
        manifest = pd.read_csv(manifest_path)
        required = {"row_index", "ticker", "fold_id", "base_score_raw", "base_rank"}
        if required.issubset(cached.columns):
            return cached, manifest
    backends = parse_str_tokens(args.base_backends)
    device = resolve_xgboost_device(args)
    eligibility_lookup = eligibility.set_index(["ticker", "fold_id"])
    records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    y_all = frame[args.target_column].to_numpy(dtype=np.int8)
    jobs = [(str(ticker), fold) for ticker in sorted(ticker_fold_index) for fold in folds]

    def compute_job(job: tuple[str, FoldSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        ticker, fold = job
        job_records: list[dict[str, Any]] = []
        job_features: list[dict[str, Any]] = []
        row = eligibility_lookup.loc[(ticker, int(fold.fold_id))]
        train_indices = np.asarray(ticker_fold_index[ticker][fold.fold_id]["train"], dtype=np.int64)
        validation_indices = np.asarray(ticker_fold_index[ticker][fold.fold_id]["validation"], dtype=np.int64)

        def append_no_model(status_name: str) -> None:
            for index in validation_indices:
                job_records.append(
                    {
                        "row_index": int(index),
                        "source_row_id": int(frame.at[index, "source_row_id"]),
                        "ticker": ticker,
                        "fold_id": int(fold.fold_id),
                        "role": role_for_fold(fold.fold_id, roles),
                        "base_score_raw": float("nan"),
                        "base_rank": float("nan"),
                        "base_model_count": 0,
                        "base_backends": "",
                        "model_status": status_name,
                    }
                )

        if not bool(row["eligible_model"]):
            append_no_model("LOW_EVIDENCE_NO_MODEL")
            return job_records, job_features, f"Ticker base OOF ticker={ticker} fold={fold.fold_id} skipped=LOW_EVIDENCE"

        ranked = compute_train_feature_scores(
            frame,
            train_indices,
            features,
            args.target_column,
            minimum_rows=int(args.minimum_feature_valid_rows),
            prior_strength=float(args.auc_prior_strength),
        )
        ranked = ranked.loc[
            (ranked["valid_n"] >= int(args.minimum_feature_valid_rows))
            & (ranked["positive_n"] >= int(args.minimum_feature_positive_rows))
            & (ranked["negative_n"] >= int(args.minimum_feature_negative_rows))
            & (ranked["coverage"] >= float(args.minimum_feature_coverage))
        ]
        selected = greedy_deduplicate_features(
            frame,
            train_indices,
            ranked,
            maximum_features=int(args.base_feature_count),
            correlation_threshold=float(args.base_dedup_threshold),
        )
        if len(selected) < int(args.minimum_base_feature_count):
            selected = ranked.head(int(args.base_feature_count))["feature"].astype(str).tolist()
        if len(selected) < int(args.minimum_base_feature_count):
            append_no_model("INSUFFICIENT_FEATURES")
            return job_records, job_features, f"Ticker base OOF ticker={ticker} fold={fold.fold_id} skipped=INSUFFICIENT_FEATURES"

        for rank_position, feature in enumerate(selected, start=1):
            feature_row = ranked.loc[ranked["feature"].eq(feature)].iloc[0]
            job_features.append(
                {
                    "ticker": ticker,
                    "fold_id": int(fold.fold_id),
                    "role": role_for_fold(fold.fold_id, roles),
                    "feature": feature,
                    "rank": rank_position,
                    "train_score": float(feature_row["train_score"]),
                    "raw_auc": float(feature_row["raw_auc"]),
                    "shrunk_oriented_auc": float(feature_row["shrunk_oriented_auc"]),
                    "spearman": float(feature_row["spearman"]),
                    "coverage": float(feature_row["coverage"]),
                }
            )
        x_train = frame.iloc[train_indices][selected].to_numpy(dtype=np.float32)
        x_validation = frame.iloc[validation_indices][selected].to_numpy(dtype=np.float32)
        y_train = y_all[train_indices]
        backend_scores: list[np.ndarray] = []
        used_backends: list[str] = []

        def train_one_backend(backend: str) -> tuple[str, np.ndarray] | None:
            try:
                score = train_backend_predict(
                    backend,
                    x_train,
                    y_train,
                    x_validation,
                    args=args,
                    device=device,
                    seed=int(args.seed) + int(fold.fold_id) * 1009,
                )
            except Exception as exc:
                if args.strict_backend:
                    raise
                log(f"Ticker base backend failed ticker={ticker} fold={fold.fold_id} backend={backend}: {exc}")
                return None
            return backend, score

        if bool(getattr(args, "parallel_backends", False)) and len(backends) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(backends), thread_name_prefix="v10-backend"
            ) as backend_executor:
                backend_results = backend_executor.map(train_one_backend, backends)
                for result in backend_results:
                    if result is not None:
                        used_backend, score = result
                        used_backends.append(used_backend)
                        backend_scores.append(score)
        else:
            for backend in backends:
                result = train_one_backend(backend)
                if result is not None:
                    used_backend, score = result
                    used_backends.append(used_backend)
                    backend_scores.append(score)
        if not backend_scores:
            raise RuntimeError(f"No backend succeeded for ticker={ticker} fold={fold.fold_id}")
        raw_score = np.nanmean(np.vstack(backend_scores), axis=0)
        rank_score = percentile_rank_1d(raw_score)
        for local, index in enumerate(validation_indices):
            job_records.append(
                {
                    "row_index": int(index),
                    "source_row_id": int(frame.at[index, "source_row_id"]),
                    "ticker": ticker,
                    "fold_id": int(fold.fold_id),
                    "role": role_for_fold(fold.fold_id, roles),
                    "base_score_raw": float(raw_score[local]),
                    "base_rank": float(rank_score[local]),
                    "base_model_count": len(used_backends),
                    "base_backends": "|".join(used_backends),
                    "model_status": "OK",
                }
            )
        message = (
            f"Ticker base OOF ticker={ticker} fold={fold.fold_id} train={len(train_indices)} "
            f"valid={len(validation_indices)} features={len(selected)} backends={len(used_backends)}"
        )
        return job_records, job_features, message

    model_workers = max(1, min(int(getattr(args, "model_workers", 1)), len(jobs)))
    log(f"Ticker base OOF jobs={len(jobs)} workers={model_workers}")
    if model_workers == 1:
        results = map(compute_job, jobs)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=model_workers, thread_name_prefix="v10-base")
        results = executor.map(compute_job, jobs)
    try:
        for job_records, job_features, message in results:
            records.extend(job_records)
            feature_records.extend(job_features)
            log(message)
    finally:
        if model_workers != 1:
            executor.shutdown(wait=True)
    oof = pd.DataFrame(records).sort_values(["fold_id", "ticker", "row_index"], kind="mergesort")
    manifest = pd.DataFrame(feature_records).sort_values(["ticker", "fold_id", "rank"], kind="mergesort")
    atomic_write_csv(cache_path, oof)
    atomic_write_csv(manifest_path, manifest)
    return oof, manifest


def attach_oof_metadata(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    metadata_columns = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.name_column,
        args.market_column,
        args.bucket_column,
        args.industry_column,
        args.target_column,
    ]
    metadata = frame[metadata_columns].copy()
    metadata["row_index"] = frame.index.to_numpy(dtype=np.int64)
    merge_keys = ["row_index", "source_row_id"]
    if args.ticker_column in oof.columns and args.ticker_column in metadata.columns:
        merge_keys.append(args.ticker_column)
    merged = oof.merge(metadata, on=merge_keys, how="left", validate="one_to_one")
    if merged[args.target_column].isna().any():
        raise RuntimeError("OOF metadata join produced missing target")
    return merged


def add_ticker_error_groups(
    oof: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_parts: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    for (ticker, fold_id), part in oof.groupby([args.ticker_column, "fold_id"], sort=True):
        current = part.copy()
        codes, rank = build_ticker_error_group_codes(
            current[args.target_column].to_numpy(dtype=int),
            current["base_score_raw"].to_numpy(dtype=float),
            top_quantile=float(args.error_top_quantile),
            low_quantile=float(args.error_low_quantile),
        )
        current["base_rank"] = rank
        current["error_code"] = codes
        current["error_group"] = pd.Series(codes).map(ERROR_GROUP_NAMES).to_numpy()
        result_parts.append(current)
        counts = pd.Series(codes).value_counts().to_dict()
        count_rows.append(
            {
                "ticker": str(ticker),
                "fold_id": int(fold_id),
                "role": str(current["role"].iloc[0]),
                "rows": int(len(current)),
                "A_TOP_TRUE_POSITIVE": int(counts.get(ERROR_A_TOP_TP, 0)),
                "B_TOP_FALSE_POSITIVE": int(counts.get(ERROR_B_TOP_FP, 0)),
                "C_LOW_MISSED_POSITIVE": int(counts.get(ERROR_C_LOW_TP, 0)),
                "D_LOW_TRUE_NEGATIVE": int(counts.get(ERROR_D_LOW_TN, 0)),
                "OTHER": int(counts.get(0, 0)),
            }
        )
    return pd.concat(result_parts, ignore_index=True), pd.DataFrame(count_rows)


def choose_regime_columns(features: Sequence[str], maximum: int = 6) -> list[str]:
    prefixes = ("u_global_", "u_finmarket_", "u_credit_", "t_link_kospi_", "t_link_kosdaq_")
    candidates = [feature for feature in features if feature.startswith(prefixes)]
    preferred_tokens = ("vol", "drawdown", "credit", "spread", "usdkrw", "rate", "return")
    candidates.sort(key=lambda value: (not any(token in value for token in preferred_tokens), value))
    return candidates[: int(maximum)]


def build_all_ticker_matched_pairs(
    frame: pd.DataFrame,
    oof_groups: pd.DataFrame,
    features: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regime_columns = choose_regime_columns(features, maximum=int(args.match_regime_columns))
    metadata = frame[["source_row_id", *regime_columns]].copy()
    pair_frame = oof_groups.merge(metadata, on="source_row_id", how="left", validate="one_to_one")
    ab_parts: list[pd.DataFrame] = []
    cd_parts: list[pd.DataFrame] = []
    for (ticker, fold_id), part in pair_frame.groupby([args.ticker_column, "fold_id"], sort=True):
        indexed = part.set_index("row_index", drop=False)
        ab = build_ticker_temporal_matched_pairs(
            indexed,
            fold_id=int(fold_id),
            ticker=str(ticker),
            axis="AB",
            case_code=ERROR_A_TOP_TP,
            control_code=ERROR_B_TOP_FP,
            controls_per_case=int(args.controls_per_case),
            date_column=args.date_column,
            score_column="base_rank",
            regime_columns=regime_columns,
            maximum_day_distance=int(args.maximum_match_day_distance),
        )
        cd = build_ticker_temporal_matched_pairs(
            indexed,
            fold_id=int(fold_id),
            ticker=str(ticker),
            axis="CD",
            case_code=ERROR_C_LOW_TP,
            control_code=ERROR_D_LOW_TN,
            controls_per_case=int(args.controls_per_case),
            date_column=args.date_column,
            score_column="base_rank",
            regime_columns=regime_columns,
            maximum_day_distance=int(args.maximum_match_day_distance),
        )
        if not ab.empty:
            ab_parts.append(ab)
        if not cd.empty:
            cd_parts.append(cd)
    ab_frame = pd.concat(ab_parts, ignore_index=True) if ab_parts else pd.DataFrame()
    cd_frame = pd.concat(cd_parts, ignore_index=True) if cd_parts else pd.DataFrame()
    return ab_frame, cd_frame


def compute_ticker_stage0_maps(
    frame: pd.DataFrame,
    oof_groups: pd.DataFrame,
    features: Sequence[str],
    matched_ab: pd.DataFrame,
    matched_cd: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = Path(args.output) / "ticker_feature_map_by_fold.csv"
    if args.resume and cache.exists():
        by_fold = pd.read_csv(cache)
    else:
        source = frame[["source_row_id", *features]].copy()
        combined = oof_groups.merge(source, on="source_row_id", how="left", validate="one_to_one")
        ab_lookup = {
            (str(ticker), int(fold_id)): part
            for (ticker, fold_id), part in matched_ab.groupby(["ticker", "fold_id"], sort=False)
        } if not matched_ab.empty else {}
        cd_lookup = {
            (str(ticker), int(fold_id)): part
            for (ticker, fold_id), part in matched_cd.groupby(["ticker", "fold_id"], sort=False)
        } if not matched_cd.empty else {}
        group_jobs = list(combined.groupby([args.ticker_column, "fold_id"], sort=True))

        def compute_group(job: tuple[tuple[Any, Any], pd.DataFrame]) -> tuple[list[dict[str, Any]], str]:
            (ticker, fold_id), part = job
            group_rows: list[dict[str, Any]] = []
            codes = part["error_code"].to_numpy(dtype=np.int8)
            y = part[args.target_column].to_numpy(dtype=np.int8)
            ab_pairs = ab_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
            cd_pairs = cd_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
            for feature in features:
                values = part[feature].to_numpy(dtype=float)
                target_auc = safe_roc_auc(y[np.isfinite(values)], values[np.isfinite(values)])
                pearson, pearson_n = safe_pearson(values, y, minimum_rows=int(args.minimum_map_valid_rows))
                spearman, spearman_n = safe_spearman(values, y, minimum_rows=int(args.minimum_map_valid_rows))
                valid_y = y[np.isfinite(values)]
                valid_n = int(np.isfinite(values).sum())
                pos_n = int(np.sum(valid_y == 1))
                neg_n = int(np.sum(valid_y == 0))
                group_rows.append(
                    {
                        "ticker": str(ticker),
                        "fold_id": int(fold_id),
                        "role": str(part["role"].iloc[0]),
                        "axis": "TARGET",
                        "feature": feature,
                        "transform": "raw",
                        "raw_auc": target_auc,
                        "valid_rows": valid_n,
                        "positive_n": pos_n,
                        "negative_n": neg_n,
                        "pearson": pearson,
                        "pearson_n": pearson_n,
                        "spearman": spearman,
                        "spearman_n": spearman_n,
                        "coverage": float(valid_n / len(part)) if len(part) else float("nan"),
                    }
                )
                for axis, positive_code, negative_code, pairs in (
                    ("AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP, ab_pairs),
                    ("CD", ERROR_C_LOW_TP, ERROR_D_LOW_TN, cd_pairs),
                ):
                    metrics = separation_metrics(
                        values,
                        codes,
                        positive_code,
                        negative_code,
                        pairs,
                        row_indices=part["row_index"].to_numpy(dtype=np.int64),
                    )
                    group_rows.append(
                        {
                            "ticker": str(ticker),
                            "fold_id": int(fold_id),
                            "role": str(part["role"].iloc[0]),
                            "axis": axis,
                            "feature": feature,
                            "transform": "raw",
                            "valid_rows": int(metrics["positive_valid_n"] + metrics["negative_valid_n"]),
                            "positive_n": int(metrics["positive_valid_n"]),
                            "negative_n": int(metrics["negative_valid_n"]),
                            **metrics,
                        }
                    )
            return group_rows, f"Ticker stage0 map ticker={ticker} fold={fold_id} features={len(features)}"

        rows: list[dict[str, Any]] = []
        analysis_workers = max(1, min(int(getattr(args, "analysis_workers", 1)), len(group_jobs)))
        if analysis_workers == 1:
            group_results = map(compute_group, group_jobs)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=analysis_workers, thread_name_prefix="v10-stage0"
            )
            group_results = executor.map(compute_group, group_jobs)
        try:
            for group_rows, message in group_results:
                rows.extend(group_rows)
                log(message)
        finally:
            if analysis_workers != 1:
                executor.shutdown(wait=True)
        by_fold = pd.DataFrame(rows)
        atomic_write_csv(cache, by_fold)
    roles = {
        "selection": parse_int_tokens(args.selection_folds),
        "confirmation": parse_int_tokens(args.confirmation_folds),
        "recent_audit": parse_int_tokens(args.recent_folds),
    }
    summaries: list[pd.DataFrame] = []
    for axis in ["TARGET", "AB", "CD"]:
        subset = by_fold.loc[by_fold["axis"].eq(axis)].copy()
        if subset.empty:
            continue
        summary = summarize_fixed_direction(
            subset,
            group_columns=["ticker", "axis", "feature", "transform"],
            selection_fold_ids=roles["selection"],
            confirmation_fold_ids=roles["confirmation"],
            recent_fold_ids=roles["recent_audit"],
            raw_auc_column="raw_auc",
            valid_rows_column="valid_rows",
            positive_n_column="positive_n",
            negative_n_column="negative_n",
            prior_strength=float(args.auc_prior_strength),
        )
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary = assign_map_evidence_grades(summary, args)
    feature_quality = summarize_ticker_feature_quality(frame, features, args)
    return by_fold, summary, feature_quality


def assign_map_evidence_grades(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return summary
    result = summary.copy()
    selection_auc = pd.to_numeric(result["selection_mean_fixed_auc"], errors="coerce")
    selection_min = pd.to_numeric(result["selection_min_fixed_auc"], errors="coerce")
    selection_consistency = pd.to_numeric(result["selection_direction_consistency"], errors="coerce")
    confirmation_auc = pd.to_numeric(result.get("confirmation_mean_fixed_auc"), errors="coerce")
    recent_auc = pd.to_numeric(result.get("recent_mean_fixed_auc"), errors="coerce")
    result["selection_gate"] = (
        (selection_auc >= float(args.minimum_selection_mean_auc))
        & (selection_min >= float(args.minimum_selection_min_auc))
        & (selection_consistency >= float(args.minimum_direction_consistency))
        & (pd.to_numeric(result["selection_fold_count"], errors="coerce") >= int(args.minimum_selection_fold_count))
    )
    result["confirmation_direction_support"] = confirmation_auc >= 0.5
    result["recent_direction_support"] = recent_auc >= 0.5
    result["time_stable"] = (
        result["selection_gate"]
        & result["confirmation_direction_support"].fillna(False)
        & result["recent_direction_support"].fillna(False)
    )
    grades: list[str] = []
    for row in result.itertuples(index=False):
        if not bool(getattr(row, "selection_gate")):
            grade = "D_WEAK"
        elif bool(getattr(row, "time_stable")) and float(getattr(row, "selection_mean_fixed_auc")) >= 0.60:
            grade = "A_STABLE_STRONG"
        elif bool(getattr(row, "time_stable")):
            grade = "B_STABLE"
        elif bool(getattr(row, "confirmation_direction_support")) or bool(getattr(row, "recent_direction_support")):
            grade = "C_REGIME_DEPENDENT"
        else:
            grade = "D_SELECTION_ONLY"
        grades.append(grade)
    result["evidence_grade"] = grades
    result["ticker_signal_score"] = (
        np.maximum(selection_auc - 0.5, 0.0) * 5.0
        + np.maximum(selection_min - 0.5, 0.0) * 2.0
        + np.maximum(selection_consistency - 0.5, 0.0) * 0.5
        + result["time_stable"].astype(float) * 0.25
    )
    return result.sort_values(
        ["ticker", "axis", "ticker_signal_score", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_ticker_feature_quality(
    frame: pd.DataFrame,
    features: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, part in frame.groupby(args.ticker_column, sort=True):
        for feature in features:
            values = part[feature].to_numpy(dtype=float)
            finite = np.isfinite(values)
            valid = values[finite]
            rows.append(
                {
                    "ticker": str(ticker),
                    "feature": feature,
                    "rows": int(len(values)),
                    "valid_n": int(finite.sum()),
                    "coverage": float(np.mean(finite)) if len(values) else float("nan"),
                    "missing_rate": float(np.mean(~finite)) if len(values) else float("nan"),
                    "n_unique": int(pd.Series(valid).nunique(dropna=True)) if valid.size else 0,
                    "mean": float(np.mean(valid)) if valid.size else float("nan"),
                    "std": float(np.std(valid)) if valid.size else float("nan"),
                    "p05": float(np.quantile(valid, 0.05)) if valid.size else float("nan"),
                    "p50": float(np.quantile(valid, 0.50)) if valid.size else float("nan"),
                    "p95": float(np.quantile(valid, 0.95)) if valid.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_ticker_correlation_maps(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    map_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    edge_parts: list[pd.DataFrame] = []
    cluster_parts: list[pd.DataFrame] = []
    representative_rows: list[dict[str, Any]] = []
    matrix_payload: dict[str, dict[str, np.ndarray]] = {}
    selection_end = max(fold.validation_end for fold in folds if fold.fold_id in set(roles["selection"]))
    fold_role_lookup = {fold.fold_id: role_for_fold(fold.fold_id, roles) for fold in folds}
    signal_lookup = map_summary.loc[map_summary["axis"].isin(["TARGET", "AB", "CD"])].copy()
    if not signal_lookup.empty:
        signal_lookup["axis_weight"] = signal_lookup["axis"].map({"AB": 1.0, "TARGET": 0.7, "CD": 0.5}).fillna(0.5)
        signal_lookup["weighted_signal"] = signal_lookup["ticker_signal_score"] * signal_lookup["axis_weight"]
        signal_lookup = (
            signal_lookup.groupby(["ticker", "feature"], as_index=False)["weighted_signal"].max()
            .rename(columns={"weighted_signal": "map_signal"})
        )
    for ticker, part in frame.groupby(args.ticker_column, sort=True):
        development = part.loc[part[args.date_column] <= selection_end].copy()
        full_corr, valid_n = compute_ticker_correlation_matrix(
            development,
            features,
            method=args.correlation_method,
            minimum_rows=int(args.minimum_correlation_rows),
            shrinkage_strength=float(args.correlation_shrinkage_strength),
        )
        fold_correlations: dict[int, pd.DataFrame] = {}
        for fold in folds:
            if fold.fold_id not in set(roles["selection"]):
                continue
            fold_frame = part.loc[
                (part[args.date_column] >= fold.train_start) & (part[args.date_column] <= fold.train_end)
            ]
            fold_corr, _ = compute_ticker_correlation_matrix(
                fold_frame,
                features,
                method=args.correlation_method,
                minimum_rows=int(args.minimum_correlation_rows),
                shrinkage_strength=float(args.correlation_shrinkage_strength),
            )
            fold_correlations[fold.fold_id] = fold_corr
        edges = stable_correlation_edges(
            full_corr,
            fold_correlations,
            fold_role_lookup,
            threshold=float(args.correlation_edge_threshold),
            minimum_fold_abs_correlation=float(args.minimum_fold_abs_correlation),
            minimum_sign_consistency=float(args.minimum_correlation_sign_consistency),
            maximum_edges=int(args.maximum_edges_per_ticker),
        )
        if not edges.empty:
            edges.insert(0, "ticker", str(ticker))
            edge_parts.append(edges)
        clusters = cluster_from_correlation(full_corr, threshold=float(args.cluster_threshold))
        clusters.insert(0, "ticker", str(ticker))
        cluster_parts.append(clusters)
        ticker_signals = signal_lookup.loc[signal_lookup["ticker"].astype(str).eq(str(ticker))] if not signal_lookup.empty else pd.DataFrame()
        score_lookup = dict(zip(ticker_signals["feature"], ticker_signals["map_signal"])) if not ticker_signals.empty else {}
        corr_values = np.abs(full_corr.to_numpy(dtype=float))
        for cluster_id, cluster in clusters.groupby("cluster_id", sort=True):
            members = cluster["feature"].astype(str).tolist()
            member_indices = [features.index(feature) for feature in members]
            if len(member_indices) > 1:
                submatrix = corr_values[np.ix_(member_indices, member_indices)]
                centrality = np.nanmean(submatrix, axis=1)
            else:
                centrality = np.ones(1, dtype=float)
            candidates: list[tuple[tuple[float, float, str], str, float, float]] = []
            for local_index, feature in enumerate(members):
                map_signal = float(score_lookup.get(feature, 0.0))
                center = float(centrality[local_index])
                candidates.append(((map_signal, center, feature), feature, map_signal, center))
            candidates.sort(key=lambda item: (-item[0][0], -item[0][1], item[0][2]))
            _, representative, map_signal, center = candidates[0]
            representative_rows.append(
                {
                    "ticker": str(ticker),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "representative": representative,
                    "representative_map_signal": map_signal,
                    "representative_centrality": center,
                    "members": "|".join(sorted(members)),
                }
            )
        matrix_payload[str(ticker)] = {
            "feature_names": np.asarray(features, dtype="U"),
            "correlation": full_corr.to_numpy(dtype=np.float32),
            "valid_n": valid_n.to_numpy(dtype=np.int32),
        }
        log(f"Ticker organic map ticker={ticker} clusters={clusters['cluster_id'].nunique()} edges={len(edges)}")
    edges_all = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    clusters_all = pd.concat(cluster_parts, ignore_index=True) if cluster_parts else pd.DataFrame()
    representatives = pd.DataFrame(representative_rows)
    return edges_all, clusters_all, representatives, matrix_payload


def select_ticker_sources(
    map_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rep_lookup: dict[str, set[str]] = defaultdict(set)
    if not representatives.empty:
        for ticker, part in representatives.groupby("ticker"):
            rep_lookup[str(ticker)] = set(part["representative"].astype(str))
    for ticker, ticker_map in map_summary.groupby("ticker", sort=True):
        for axis, maximum in (
            ("TARGET", int(args.target_source_count)),
            ("AB", int(args.ab_source_count)),
            ("CD", int(args.cd_source_count)),
        ):
            part = ticker_map.loc[ticker_map["axis"].eq(axis)].copy()
            part = part.loc[part["selection_fold_count"] >= int(args.minimum_selection_fold_count)]
            part = part.sort_values(
                ["selection_evidence_score", "ticker_signal_score", "feature"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            selected: list[str] = []
            for row in part.itertuples(index=False):
                feature = str(row.feature)
                if feature in selected:
                    continue
                selected.append(feature)
                rows.append(
                    {
                        "ticker": str(ticker),
                        "axis": axis,
                        "feature": feature,
                        "source_rank": len(selected),
                        "selection_evidence_score": float(row.selection_evidence_score),
                        "selection_mean_fixed_auc": float(row.selection_mean_fixed_auc),
                        "selection_min_fixed_auc": float(row.selection_min_fixed_auc),
                        "selection_direction": int(row.selection_direction),
                        "evidence_grade": str(row.evidence_grade),
                        "is_cluster_representative": feature in rep_lookup.get(str(ticker), set()),
                    }
                )
                if len(selected) >= maximum:
                    break
    return pd.DataFrame(rows)


def build_selected_transforms(
    frame: pd.DataFrame,
    selected_sources: pd.DataFrame,
    clusters: pd.DataFrame,
    matrix_payload: Mapping[str, Mapping[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    metadata_columns = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.market_column,
        args.bucket_column,
        args.industry_column,
        args.target_column,
    ]
    transformed = frame[metadata_columns].copy()
    manifest: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_union = sorted(selected_sources["feature"].astype(str).unique().tolist()) if not selected_sources.empty else []
    dates = frame[args.date_column].to_numpy(dtype="datetime64[ns]")
    market_groups = [dates, frame[args.market_column].astype(str).to_numpy()]
    bucket_groups = [dates, frame[args.bucket_column].astype(str).to_numpy()]
    industry_groups = [dates, frame[args.industry_column].astype(str).to_numpy()]
    cross_cache: dict[tuple[str, str], np.ndarray] = {}
    for feature in source_union:
        values = frame[feature].to_numpy(dtype=float)
        cross_cache[(feature, "date_rank")] = datewise_percentile_rank(values, dates)
        cross_cache[(feature, "date_market_rank")] = groupwise_percentile_rank(values, market_groups)
        cross_cache[(feature, "date_bucket_rank")] = groupwise_percentile_rank(values, bucket_groups)
        cross_cache[(feature, "date_industry_rank")] = groupwise_percentile_rank(values, industry_groups)
    cluster_lookup = clusters.groupby(["ticker", "cluster_id"])["feature"].apply(list).to_dict() if not clusters.empty else {}
    feature_cluster_lookup = {
        (str(row.ticker), str(row.feature)): int(row.cluster_id)
        for row in clusters.itertuples(index=False)
    } if not clusters.empty else {}
    for ticker, ticker_sources in selected_sources.groupby("ticker", sort=True):
        ticker_mask = frame[args.ticker_column].astype(str).eq(str(ticker)).to_numpy()
        ticker_indices = np.flatnonzero(ticker_mask)
        ticker_indices = ticker_indices[np.argsort(frame.iloc[ticker_indices][args.date_column].to_numpy(dtype="datetime64[ns]"), kind="mergesort")]
        ticker_features = ticker_sources["feature"].astype(str).unique().tolist()
        feature_names = list(matrix_payload.get(str(ticker), {}).get("feature_names", []))
        correlation = np.asarray(matrix_payload.get(str(ticker), {}).get("correlation", np.empty((0, 0))), dtype=float)
        feature_position = {str(feature): index for index, feature in enumerate(feature_names)}
        for feature in ticker_features:
            for transform in CROSS_SECTIONAL_TRANSFORMS:
                column = f"{feature}__{transform}"
                if column not in transformed.columns:
                    transformed[column] = cross_cache[(feature, transform)]
                manifest[str(ticker)][feature].append(column)
            ticker_values = frame.iloc[ticker_indices][feature].to_numpy(dtype=float)
            for transform in TIME_TRANSFORMS:
                column = make_ticker_transform_name(feature, transform)
                if column not in transformed.columns:
                    transformed[column] = np.nan
                transformed.loc[ticker_indices, column] = apply_ticker_time_transform(ticker_values, transform)
                manifest[str(ticker)][feature].append(column)
            cluster_id = feature_cluster_lookup.get((str(ticker), feature))
            peers: list[str] = []
            if cluster_id is not None:
                peers = [peer for peer in cluster_lookup.get((str(ticker), cluster_id), []) if str(peer) != feature]
            if not peers and feature in feature_position and correlation.size:
                position = feature_position[feature]
                order = np.argsort(-np.abs(correlation[position]), kind="mergesort")
                peers = [str(feature_names[index]) for index in order if str(feature_names[index]) != feature][:3]
            if peers:
                column = f"{feature}__ticker_cluster_innovation"
                if column not in transformed.columns:
                    transformed[column] = np.nan
                base_rank = percentile_rank_1d(ticker_values)
                peer_ranks = []
                for peer in peers[:5]:
                    peer_values = frame.iloc[ticker_indices][peer].to_numpy(dtype=float)
                    peer_ranks.append(percentile_rank_1d(peer_values))
                peer_median = np.nanmedian(np.vstack(peer_ranks), axis=0)
                transformed.loc[ticker_indices, column] = base_rank - peer_median
                manifest[str(ticker)][feature].append(column)
    return transformed, {ticker: dict(value) for ticker, value in manifest.items()}


def compute_ticker_extended_maps(
    transformed: pd.DataFrame,
    oof_groups: pd.DataFrame,
    selected_sources: pd.DataFrame,
    transform_manifest: Mapping[str, Mapping[str, Sequence[str]]],
    matched_ab: pd.DataFrame,
    matched_cd: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_columns = sorted(
        {
            column
            for ticker_map in transform_manifest.values()
            for columns in ticker_map.values()
            for column in columns
        }
    )
    if not source_columns:
        return pd.DataFrame(), pd.DataFrame()
    merged = oof_groups.merge(
        transformed[["source_row_id", *source_columns]],
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    ab_lookup = {
        (str(ticker), int(fold_id)): part
        for (ticker, fold_id), part in matched_ab.groupby(["ticker", "fold_id"], sort=False)
    } if not matched_ab.empty else {}
    cd_lookup = {
        (str(ticker), int(fold_id)): part
        for (ticker, fold_id), part in matched_cd.groupby(["ticker", "fold_id"], sort=False)
    } if not matched_cd.empty else {}
    source_axis_lookup = (
        selected_sources.groupby(["ticker", "feature"])["axis"].apply(lambda values: sorted(set(values))).to_dict()
        if not selected_sources.empty else {}
    )
    group_jobs = list(merged.groupby([args.ticker_column, "fold_id"], sort=True))

    def compute_group(job: tuple[tuple[Any, Any], pd.DataFrame]) -> tuple[list[dict[str, Any]], str]:
        (ticker, fold_id), part = job
        group_rows: list[dict[str, Any]] = []
        codes = part["error_code"].to_numpy(dtype=np.int8)
        y = part[args.target_column].to_numpy(dtype=np.int8)
        ab_pairs = ab_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
        cd_pairs = cd_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
        ticker_manifest = transform_manifest.get(str(ticker), {})
        for source_feature, columns in ticker_manifest.items():
            axes = source_axis_lookup.get((str(ticker), str(source_feature)), ["TARGET", "AB", "CD"])
            for column in columns:
                if column not in part.columns:
                    continue
                values = part[column].to_numpy(dtype=float)
                transform = column.split("__", 1)[1] if "__" in column else "unknown"
                for axis in sorted(set(axes) | {"TARGET"}):
                    if axis == "TARGET":
                        valid = np.isfinite(values)
                        valid_y = y[valid]
                        valid_values = values[valid]
                        pos_n = int(np.sum(valid_y == 1))
                        neg_n = int(np.sum(valid_y == 0))
                        group_rows.append(
                            {
                                "ticker": str(ticker),
                                "fold_id": int(fold_id),
                                "role": str(part["role"].iloc[0]),
                                "axis": axis,
                                "feature": str(source_feature),
                                "transformed_feature": column,
                                "transform": transform,
                                "raw_auc": safe_roc_auc(valid_y, valid_values) if pos_n > 0 and neg_n > 0 else float("nan"),
                                "valid_rows": int(valid.sum()),
                                "positive_n": pos_n,
                                "negative_n": neg_n,
                            }
                        )
                    elif axis == "AB":
                        metrics = separation_metrics(
                            values,
                            codes,
                            ERROR_A_TOP_TP,
                            ERROR_B_TOP_FP,
                            ab_pairs,
                            row_indices=part["row_index"].to_numpy(dtype=np.int64),
                        )
                        group_rows.append(
                            {
                                "ticker": str(ticker),
                                "fold_id": int(fold_id),
                                "role": str(part["role"].iloc[0]),
                                "axis": axis,
                                "feature": str(source_feature),
                                "transformed_feature": column,
                                "transform": transform,
                                "valid_rows": int(metrics["positive_valid_n"] + metrics["negative_valid_n"]),
                                "positive_n": int(metrics["positive_valid_n"]),
                                "negative_n": int(metrics["negative_valid_n"]),
                                **metrics,
                            }
                        )
                    elif axis == "CD":
                        metrics = separation_metrics(
                            values,
                            codes,
                            ERROR_C_LOW_TP,
                            ERROR_D_LOW_TN,
                            cd_pairs,
                            row_indices=part["row_index"].to_numpy(dtype=np.int64),
                        )
                        group_rows.append(
                            {
                                "ticker": str(ticker),
                                "fold_id": int(fold_id),
                                "role": str(part["role"].iloc[0]),
                                "axis": axis,
                                "feature": str(source_feature),
                                "transformed_feature": column,
                                "transform": transform,
                                "valid_rows": int(metrics["positive_valid_n"] + metrics["negative_valid_n"]),
                                "positive_n": int(metrics["positive_valid_n"]),
                                "negative_n": int(metrics["negative_valid_n"]),
                                **metrics,
                            }
                        )
        return group_rows, f"Ticker extended map ticker={ticker} fold={fold_id}"

    rows: list[dict[str, Any]] = []
    analysis_workers = max(1, min(int(getattr(args, "analysis_workers", 1)), len(group_jobs)))
    if analysis_workers == 1:
        group_results = map(compute_group, group_jobs)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=analysis_workers, thread_name_prefix="v10-extended"
        )
        group_results = executor.map(compute_group, group_jobs)
    try:
        for group_rows, message in group_results:
            rows.extend(group_rows)
            log(message)
    finally:
        if analysis_workers != 1:
            executor.shutdown(wait=True)
    by_fold = pd.DataFrame(rows)
    if by_fold.empty:
        return by_fold, pd.DataFrame()
    roles = {
        "selection": parse_int_tokens(args.selection_folds),
        "confirmation": parse_int_tokens(args.confirmation_folds),
        "recent_audit": parse_int_tokens(args.recent_folds),
    }
    summaries: list[pd.DataFrame] = []
    for axis in ["TARGET", "AB", "CD"]:
        subset = by_fold.loc[by_fold["axis"].eq(axis)]
        if subset.empty:
            continue
        summary = summarize_fixed_direction(
            subset,
            group_columns=["ticker", "axis", "feature", "transformed_feature", "transform"],
            selection_fold_ids=roles["selection"],
            confirmation_fold_ids=roles["confirmation"],
            recent_fold_ids=roles["recent_audit"],
            raw_auc_column="raw_auc",
            valid_rows_column="valid_rows",
            positive_n_column="positive_n",
            negative_n_column="negative_n",
            prior_strength=float(args.auc_prior_strength),
        )
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary = assign_map_evidence_grades(summary, args)
    return by_fold, summary


def build_ticker_profile_manifest(
    stage0_summary: pd.DataFrame,
    extended_summary: pd.DataFrame,
    representatives: pd.DataFrame,
    selected_sources: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame]:
    profiles: dict[str, dict[str, list[str]]] = defaultdict(dict)
    membership_rows: list[dict[str, Any]] = []
    rep_lookup = {
        str(ticker): part["representative"].astype(str).tolist()
        for ticker, part in representatives.groupby("ticker")
    } if not representatives.empty else {}
    for ticker in sorted(stage0_summary["ticker"].astype(str).unique().tolist()):
        ticker_stage0 = stage0_summary.loc[stage0_summary["ticker"].astype(str).eq(ticker)].copy()
        ticker_extended = extended_summary.loc[extended_summary["ticker"].astype(str).eq(ticker)].copy() if not extended_summary.empty else pd.DataFrame()
        target_raw = (
            ticker_stage0.loc[ticker_stage0["axis"].eq("TARGET")]
            .sort_values(["ticker_signal_score", "feature"], ascending=[False, True])
            .head(int(args.profile_target_count))["feature"].astype(str).tolist()
        )
        ab_raw = (
            ticker_stage0.loc[ticker_stage0["axis"].eq("AB")]
            .sort_values(["ticker_signal_score", "feature"], ascending=[False, True])
            .head(int(args.profile_ab_count))["feature"].astype(str).tolist()
        )
        cd_raw = (
            ticker_stage0.loc[ticker_stage0["axis"].eq("CD")]
            .sort_values(["ticker_signal_score", "feature"], ascending=[False, True])
            .head(int(args.profile_cd_count))["feature"].astype(str).tolist()
        )
        extended_ab = []
        extended_cd = []
        extended_target = []
        if not ticker_extended.empty:
            extended_ab = (
                ticker_extended.loc[ticker_extended["axis"].eq("AB")]
                .sort_values(["ticker_signal_score", "transformed_feature"], ascending=[False, True])
                .head(int(args.profile_extended_ab_count))["transformed_feature"].astype(str).tolist()
            )
            extended_cd = (
                ticker_extended.loc[ticker_extended["axis"].eq("CD")]
                .sort_values(["ticker_signal_score", "transformed_feature"], ascending=[False, True])
                .head(int(args.profile_extended_cd_count))["transformed_feature"].astype(str).tolist()
            )
            extended_target = (
                ticker_extended.loc[ticker_extended["axis"].eq("TARGET")]
                .sort_values(["ticker_signal_score", "transformed_feature"], ascending=[False, True])
                .head(int(args.profile_extended_target_count))["transformed_feature"].astype(str).tolist()
            )
        profiles[ticker]["TARGET_TOP"] = list(dict.fromkeys(target_raw + extended_target))
        profiles[ticker]["AB_PRECISION"] = list(dict.fromkeys(ab_raw + extended_ab))
        profiles[ticker]["CD_RECOVERY"] = list(dict.fromkeys(cd_raw + extended_cd))
        profiles[ticker]["CLUSTER_REP"] = rep_lookup.get(ticker, [])[: int(args.profile_cluster_rep_count)]
        profiles[ticker]["COMBINED"] = list(
            dict.fromkeys(
                profiles[ticker]["AB_PRECISION"]
                + profiles[ticker]["TARGET_TOP"]
                + profiles[ticker]["CD_RECOVERY"]
                + profiles[ticker]["CLUSTER_REP"]
            )
        )[: int(args.profile_combined_max_count)]
        for profile, columns in profiles[ticker].items():
            for position, column in enumerate(columns, start=1):
                membership_rows.append(
                    {
                        "ticker": ticker,
                        "profile": profile,
                        "feature": column,
                        "position": position,
                    }
                )
    return dict(profiles), pd.DataFrame(membership_rows)



def hierarchy_config_from_args(args: argparse.Namespace) -> HierarchyConfig:
    return HierarchyConfig(
        prior_strength=float(args.hierarchy_prior_strength),
        minimum_peer_tickers=int(args.minimum_peer_tickers),
        weak_effect=float(args.hierarchy_weak_effect),
        strong_effect=float(args.hierarchy_strong_effect),
        unique_delta=float(args.hierarchy_unique_delta),
        amplified_delta=float(args.hierarchy_amplified_delta),
        reversal_effect=float(args.hierarchy_reversal_effect),
        minimum_reliability=float(args.hierarchy_minimum_reliability),
        confirmation_margin=float(args.hierarchy_confirmation_margin),
        recent_margin=float(args.hierarchy_recent_margin),
        industry_weight=float(args.hierarchy_industry_weight),
        bucket_weight=float(args.hierarchy_bucket_weight),
        market_weight=float(args.hierarchy_market_weight),
        global_weight=float(args.hierarchy_global_weight),
        similarity_min_common=int(args.ticker_similarity_min_common),
        similarity_edge_threshold=float(args.ticker_similarity_edge_threshold),
        similarity_top_k=int(args.ticker_similarity_top_k),
        similarity_cluster_threshold=float(args.ticker_similarity_cluster_threshold),
        similarity_feature_count=int(args.ticker_similarity_feature_count),
    )


def merge_hierarchical_profiles(
    base_profiles: Mapping[str, Mapping[str, Sequence[str]]],
    base_membership: pd.DataFrame,
    hierarchical_profiles: Mapping[str, Mapping[str, Sequence[str]]],
    hierarchical_membership: pd.DataFrame,
    *,
    combined_max_count: int,
) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame]:
    merged: dict[str, dict[str, list[str]]] = {
        str(ticker): {str(profile): list(map(str, columns)) for profile, columns in profiles.items()}
        for ticker, profiles in base_profiles.items()
    }
    for ticker, profiles in hierarchical_profiles.items():
        ticker_key = str(ticker)
        merged.setdefault(ticker_key, {})
        for profile, columns in profiles.items():
            merged[ticker_key][str(profile)] = list(dict.fromkeys(map(str, columns)))
        old_combined = merged[ticker_key].get("COMBINED", [])
        hierarchy_combined = merged[ticker_key].get("HIERARCHICAL_COMBINED", [])
        merged[ticker_key]["COMBINED"] = list(dict.fromkeys(hierarchy_combined + old_combined))[: int(combined_max_count)]
    rows: list[dict[str, Any]] = []
    for ticker, profiles in sorted(merged.items()):
        for profile, columns in sorted(profiles.items()):
            for position, column in enumerate(columns, start=1):
                rows.append(
                    {
                        "ticker": str(ticker),
                        "profile": str(profile),
                        "feature": str(column),
                        "position": int(position),
                        "profile_source": "HIERARCHICAL" if profile in {
                            "SURGE_ASSOCIATION",
                            "PRECISION_SEPARATOR",
                            "MISSED_POSITIVE_RECOVERY",
                            "TICKER_SPECIFIC",
                            "PEER_SHARED",
                            "DIRECTION_REVERSAL",
                            "HIERARCHICAL_COMBINED",
                        } else "ORGANIC",
                    }
                )
    return merged, pd.DataFrame(rows)


def write_ticker_similarity_graph(
    output: Path,
    ticker_metadata: pd.DataFrame,
    similarity_edges: pd.DataFrame,
    ticker_clusters: pd.DataFrame,
) -> None:
    metadata = ticker_metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str)
    nodes = metadata.merge(ticker_clusters, on="ticker", how="left")
    nodes = nodes.rename(columns={"ticker": "node_id"})
    nodes["node_type"] = "ticker"
    edges = similarity_edges.copy()
    if not edges.empty:
        edges = edges.rename(columns={"ticker_a": "source", "ticker_b": "target", "map_spearman": "weight"})
        edges["edge_type"] = "ticker_map_similarity"
    else:
        edges = pd.DataFrame(columns=["source", "target", "weight", "edge_type"])
    write_graphml(output / "ticker_similarity_graph_v10.graphml", nodes, edges)



def write_per_ticker_hierarchy_exports(
    output: Path,
    effect_map: pd.DataFrame,
    driver_membership: pd.DataFrame,
    similarity_edges: pd.DataFrame,
    ticker_clusters: pd.DataFrame,
) -> None:
    root = output / "per_ticker"
    root.mkdir(parents=True, exist_ok=True)
    tickers = sorted(effect_map["ticker"].astype(str).unique().tolist()) if not effect_map.empty else []
    for ticker in tickers:
        ticker_dir = root / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        current = effect_map.loc[effect_map["ticker"].astype(str).eq(ticker)].copy()
        atomic_write_csv(ticker_dir / "hierarchical_driver_map.csv", current)
        atomic_write_csv(
            ticker_dir / "top_precision_drivers.csv",
            current.loc[current["axis"].eq("AB")].sort_values(
                ["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort"
            ).head(50),
        )
        atomic_write_csv(
            ticker_dir / "top_surge_association.csv",
            current.loc[current["axis"].eq("TARGET")].sort_values(
                ["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort"
            ).head(50),
        )
        atomic_write_csv(
            ticker_dir / "ticker_specific_drivers.csv",
            current.loc[current["is_ticker_specific"]].sort_values(
                ["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort"
            ),
        )
        if not driver_membership.empty:
            atomic_write_csv(
                ticker_dir / "driver_profiles.csv",
                driver_membership.loc[driver_membership["ticker"].astype(str).eq(ticker)],
            )
        if not similarity_edges.empty:
            neighbors = similarity_edges.loc[
                similarity_edges["ticker_a"].astype(str).eq(ticker)
                | similarity_edges["ticker_b"].astype(str).eq(ticker)
            ].copy()
            atomic_write_csv(ticker_dir / "similar_tickers.csv", neighbors)
        if not ticker_clusters.empty:
            atomic_write_csv(
                ticker_dir / "ticker_map_cluster.csv",
                ticker_clusters.loc[ticker_clusters["ticker"].astype(str).eq(ticker)],
            )


def append_hierarchy_report(
    report_path: Path,
    effect_map: pd.DataFrame,
    similarity_edges: pd.DataFrame,
    ticker_clusters: pd.DataFrame,
) -> None:
    if not report_path.exists():
        return
    taxonomy = effect_map["ticker_effect_class"].value_counts().to_dict() if not effect_map.empty else {}
    unique_tickers = int(
        effect_map.loc[
            effect_map["ticker_effect_class"].isin(
                ["TICKER_UNIQUE_DRIVER", "TICKER_AMPLIFIED_DRIVER", "TICKER_DIRECTION_REVERSAL"]
            ),
            "ticker",
        ].nunique()
    ) if not effect_map.empty else 0
    lines = [
        "",
        "## 종목별 차이 지도",
        "",
        "- 공통 예측 모델을 학습하지 않고, 종목별 원시 효과를 종목을 제외한 industry/bucket/market/global 지도와 비교했다.",
        "- peer reference는 설명 지도 안정화 용도이며 예측 모델의 학습 행을 합치지 않는다.",
        f"- 종목 특화·증폭·방향반전 신호가 하나 이상 있는 종목: {unique_tickers:,}",
        f"- 종목 유사도 edge: {len(similarity_edges):,}",
        f"- 종목 지도 cluster: {ticker_clusters['ticker_map_cluster'].nunique() if not ticker_clusters.empty else 0:,}",
        "",
        "| 분류 | 노드 수 |",
        "|---|---:|",
    ]
    for label, count in sorted(taxonomy.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {label} | {int(count):,} |")
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_model_frame(
    frame: pd.DataFrame,
    transformed: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base_columns = ["source_row_id", *[column for column in transformed.columns if "__" in column]]
    merged = frame.merge(transformed[base_columns], on="source_row_id", how="left", validate="one_to_one")
    return merged


def fit_per_ticker_probe_models(
    model_frame: pd.DataFrame,
    base_oof: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Sequence[str]]],
    folds: Sequence[FoldSpec],
    ticker_fold_index: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    eligibility: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    features: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = Path(args.output) / "ticker_probe_predictions.csv"
    metric_path = Path(args.output) / "ticker_probe_metrics_by_ticker_fold.csv"
    if args.resume and cache_path.exists() and metric_path.exists():
        return pd.read_csv(cache_path), pd.read_csv(metric_path)
    backends = parse_str_tokens(args.probe_backends)
    device = resolve_xgboost_device(args)
    eligibility_lookup = eligibility.set_index(["ticker", "fold_id"])
    base_lookup = base_oof.set_index("row_index")
    y_all = model_frame[args.target_column].to_numpy(dtype=np.int8)
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    requested_profiles = parse_str_tokens(args.probe_profiles)
    for ticker in sorted(ticker_fold_index):
        ticker_profiles = profiles.get(str(ticker), {})
        for fold in folds:
            row = eligibility_lookup.loc[(str(ticker), int(fold.fold_id))]
            train_indices = np.asarray(ticker_fold_index[ticker][fold.fold_id]["train"], dtype=np.int64)
            validation_indices = np.asarray(ticker_fold_index[ticker][fold.fold_id]["validation"], dtype=np.int64)
            if validation_indices.size == 0:
                continue
            y_validation = y_all[validation_indices]
            for profile in requested_profiles:
                if profile == "BASE_TICKER":
                    score = base_lookup.loc[validation_indices, "base_score_raw"].to_numpy(dtype=float)
                    used_features: list[str] = []
                    status = "OK" if np.isfinite(score).any() else "NO_BASE_SCORE"
                else:
                    columns = [column for column in ticker_profiles.get(profile, []) if column in model_frame.columns]
                    if not bool(row["eligible_model"]) or len(columns) < int(args.minimum_probe_feature_count):
                        score = np.full(len(validation_indices), np.nan, dtype=float)
                        used_features = columns
                        status = "LOW_EVIDENCE_NO_MODEL"
                    else:
                        x_train = model_frame.iloc[train_indices][columns].to_numpy(dtype=np.float32)
                        x_validation = model_frame.iloc[validation_indices][columns].to_numpy(dtype=np.float32)
                        y_train = y_all[train_indices]
                        backend_scores: list[np.ndarray] = []
                        for backend in backends:
                            try:
                                backend_score = train_backend_predict(
                                    backend,
                                    x_train,
                                    y_train,
                                    x_validation,
                                    args=args,
                                    device=device,
                                    seed=int(args.seed) + int(fold.fold_id) * 1009 + int(hash_strings([str(ticker), str(profile)])[:8], 16) % 100000,
                                )
                            except Exception as exc:
                                if args.strict_backend:
                                    raise
                                log(f"Probe backend failed ticker={ticker} fold={fold.fold_id} profile={profile} backend={backend}: {exc}")
                                continue
                            backend_scores.append(backend_score)
                        score = np.nanmean(np.vstack(backend_scores), axis=0) if backend_scores else np.full(len(validation_indices), np.nan)
                        used_features = columns
                        status = "OK" if backend_scores else "ALL_BACKENDS_FAILED"
                metrics = safe_binary_metrics(y_validation, score)
                probability = probability_metrics(y_validation, score)
                practical = select_threshold_for_precision(
                    y_validation,
                    score,
                    target_precision=float(args.target_precision),
                    minimum_alerts=int(args.minimum_alerts_per_ticker_diagnostic),
                    minimum_recall=0.0,
                    minimum_wilson_lcb=0.0,
                )
                metric_rows.append(
                    {
                        "ticker": str(ticker),
                        "fold_id": int(fold.fold_id),
                        "role": role_for_fold(fold.fold_id, roles),
                        "profile": profile,
                        "status": status,
                        "feature_count": len(used_features),
                        **metrics,
                        **probability,
                        **{f"precision_policy_{key}": value for key, value in practical.items()},
                    }
                )
                rank_score = percentile_rank_1d(score)
                for local, index in enumerate(validation_indices):
                    prediction_rows.append(
                        {
                            "row_index": int(index),
                            "source_row_id": int(model_frame.at[index, "source_row_id"]),
                            "ticker": str(ticker),
                            "fold_id": int(fold.fold_id),
                            "role": role_for_fold(fold.fold_id, roles),
                            "profile": profile,
                            "target": int(y_validation[local]),
                            "score_raw": float(score[local]),
                            "score_rank_ticker_fold": float(rank_score[local]),
                            "status": status,
                        }
                    )
            log(f"Ticker probe ticker={ticker} fold={fold.fold_id}")
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    atomic_write_csv(cache_path, predictions)
    atomic_write_csv(metric_path, metrics)
    return predictions, metrics


def select_ticker_profile_forward(
    predictions: pd.DataFrame,
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_order = sorted(int(value) for value in roles["selection"])
    rows: list[dict[str, Any]] = []
    routed_parts: list[pd.DataFrame] = []
    for fold in folds:
        if fold.fold_id in selection_order:
            prior_folds = [value for value in selection_order if value < fold.fold_id]
        else:
            prior_folds = selection_order
        fold_part = predictions.loc[predictions["fold_id"].eq(fold.fold_id)].copy()
        for ticker, ticker_current in fold_part.groupby("ticker", sort=True):
            prior = predictions.loc[
                predictions["ticker"].astype(str).eq(str(ticker))
                & predictions["fold_id"].isin(prior_folds)
                & predictions["status"].eq("OK")
            ].copy()
            if prior.empty:
                champion = "BASE_TICKER"
                evidence_rows = 0
                reason = "WARMUP_BASE"
            else:
                candidates: list[tuple[tuple[float, float, float, float, str], str]] = []
                for profile, part in prior.groupby("profile", sort=True):
                    y = part["target"].to_numpy(dtype=int)
                    score = part["score_raw"].to_numpy(dtype=float)
                    metrics = safe_binary_metrics(y, score)
                    practical = select_threshold_for_precision(
                        y,
                        score,
                        target_precision=float(args.target_precision),
                        minimum_alerts=int(args.minimum_alerts_per_ticker_policy),
                        minimum_recall=0.0,
                        minimum_wilson_lcb=0.0,
                    )
                    precision = float(practical.get("best_practical_precision", float("nan")))
                    recall = float(practical.get("best_practical_recall", 0.0))
                    pr_auc = float(metrics.get("pr_auc", float("nan")))
                    key = (
                        1.0 if bool(practical.get("gate_pass", False)) else 0.0,
                        precision if math.isfinite(precision) else -1.0,
                        recall,
                        pr_auc if math.isfinite(pr_auc) else -1.0,
                        str(profile),
                    )
                    candidates.append((key, str(profile)))
                candidates.sort(key=lambda item: (-item[0][0], -item[0][1], -item[0][2], -item[0][3], item[0][4]))
                champion = candidates[0][1]
                evidence_rows = int(len(prior))
                reason = "PRIOR_TICKER_ONLY"
            selected = ticker_current.loc[ticker_current["profile"].eq(champion)].copy()
            if selected.empty:
                selected = ticker_current.loc[ticker_current["profile"].eq("BASE_TICKER")].copy()
                champion = "BASE_TICKER"
                reason = f"{reason}|FALLBACK_BASE"
            selected["selected_profile"] = champion
            routed_parts.append(selected)
            rows.append(
                {
                    "ticker": str(ticker),
                    "fold_id": int(fold.fold_id),
                    "role": role_for_fold(fold.fold_id, roles),
                    "selected_profile": champion,
                    "prior_fold_count": len(prior_folds),
                    "prior_prediction_rows": evidence_rows,
                    "selection_reason": reason,
                }
            )
    routed = pd.concat(routed_parts, ignore_index=True) if routed_parts else pd.DataFrame()
    return pd.DataFrame(rows), routed


def calibrate_scores_per_ticker_forward(
    routed: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression

    selection_order = sorted(int(value) for value in roles["selection"])
    parts: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    for (ticker, fold_id), current in routed.groupby(["ticker", "fold_id"], sort=True):
        fold_id = int(fold_id)
        if fold_id in selection_order:
            prior_folds = [value for value in selection_order if value < fold_id]
        else:
            prior_folds = selection_order
        prior = routed.loc[
            routed["ticker"].astype(str).eq(str(ticker))
            & routed["fold_id"].isin(prior_folds)
            & np.isfinite(routed["score_raw"])
        ].copy()
        current = current.copy()
        raw = current["score_raw"].to_numpy(dtype=float)
        if (
            len(prior) >= int(args.minimum_calibration_rows)
            and prior["target"].nunique() == 2
            and prior["target"].sum() >= int(args.minimum_calibration_positive)
        ):
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            x_prior = prior[["score_raw"]].to_numpy(dtype=float)
            y_prior = prior["target"].to_numpy(dtype=int)
            model.fit(x_prior, y_prior)
            calibrated = model.predict_proba(raw.reshape(-1, 1))[:, 1]
            method = "PLATT_TICKER_ONLY"
        elif len(prior) >= int(args.minimum_calibration_rows_rank):
            history = np.sort(prior["score_raw"].to_numpy(dtype=float))
            calibrated = np.searchsorted(history, raw, side="right") / max(len(history), 1)
            method = "EMPIRICAL_CDF_TICKER_ONLY"
        else:
            calibrated = percentile_rank_1d(raw)
            method = "CURRENT_FOLD_RANK_WARMUP"
        current["score_calibrated_ticker"] = calibrated
        current["calibration_method"] = method
        parts.append(current)
        calibration_rows.append(
            {
                "ticker": str(ticker),
                "fold_id": fold_id,
                "role": str(current["role"].iloc[0]),
                "method": method,
                "prior_rows": int(len(prior)),
                "prior_positive": int(prior["target"].sum()) if len(prior) else 0,
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(calibration_rows)


def evaluate_forward_ticker_policies(
    calibrated: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_order = sorted(int(value) for value in roles["selection"])
    policy_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for fold_id in sorted(calibrated["fold_id"].unique().tolist()):
        current = calibrated.loc[calibrated["fold_id"].eq(fold_id)].copy()
        if int(fold_id) in selection_order:
            prior_folds = [value for value in selection_order if value < int(fold_id)]
        else:
            prior_folds = selection_order
        for ticker, ticker_current in current.groupby("ticker", sort=True):
            prior = calibrated.loc[
                calibrated["ticker"].astype(str).eq(str(ticker))
                & calibrated["fold_id"].isin(prior_folds)
            ].copy()
            if prior.empty:
                threshold = float("inf")
                policy = {"gate_pass": False, "threshold": threshold, "alerts": 0, "precision": float("nan"), "recall": 0.0}
                reason = "NO_PRIOR_TICKER_EVIDENCE"
            else:
                policy = select_threshold_for_precision(
                    prior["target"].to_numpy(dtype=int),
                    prior["score_calibrated_ticker"].to_numpy(dtype=float),
                    target_precision=float(args.target_precision),
                    minimum_alerts=int(args.minimum_alerts_per_ticker_policy),
                    minimum_recall=float(args.minimum_recall_per_ticker_policy),
                    minimum_wilson_lcb=float(args.minimum_precision_lcb_per_ticker),
                )
                threshold = float(policy.get("threshold", float("inf")))
                reason = "TICKER_PRIOR_POLICY" if bool(policy.get("gate_pass", False)) else "NO_SAFE_TICKER_THRESHOLD"
            ticker_current["ticker_threshold"] = threshold
            ticker_current["ticker_alert"] = np.isfinite(ticker_current["score_calibrated_ticker"]) & (
                ticker_current["score_calibrated_ticker"] >= threshold
            )
            ticker_current["ticker_policy_reason"] = reason
            scored_parts.append(ticker_current)
            policy_rows.append(
                {
                    "ticker": str(ticker),
                    "fold_id": int(fold_id),
                    "role": str(ticker_current["role"].iloc[0]),
                    "prior_fold_count": len(prior_folds),
                    "prior_rows": int(len(prior)),
                    "policy_reason": reason,
                    **policy,
                }
            )
        scored_fold = pd.concat(scored_parts[-current["ticker"].nunique():], ignore_index=True)
        alerts = scored_fold["ticker_alert"].to_numpy(dtype=bool)
        y = scored_fold["target"].to_numpy(dtype=int)
        tp = int(np.sum(alerts & (y == 1)))
        fp = int(np.sum(alerts & (y == 0)))
        positives = int(np.sum(y == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(positives, 1)
        rank_metrics = safe_binary_metrics(y, scored_fold["score_calibrated_ticker"].to_numpy(dtype=float))
        fold_metric_rows.append(
            {
                "fold_id": int(fold_id),
                "role": str(scored_fold["role"].iloc[0]),
                "rows": int(len(scored_fold)),
                "alerts": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "precision": float(precision),
                "recall": float(recall),
                "precision_gate": bool(
                    (tp + fp) >= int(args.minimum_portfolio_alerts)
                    and precision >= float(args.target_precision)
                ),
                **rank_metrics,
            }
        )
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    return pd.DataFrame(policy_rows), scored, pd.DataFrame(fold_metric_rows)


def aggregate_probe_results(
    ticker_fold_metrics: pd.DataFrame,
    portfolio_fold_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ticker_fold_metrics.empty:
        ticker_summary = pd.DataFrame()
    else:
        ticker_summary = (
            ticker_fold_metrics.groupby(["ticker", "profile", "role"], as_index=False)
            .agg(
                fold_count=("fold_id", "nunique"),
                mean_pr_auc=("pr_auc", "mean"),
                mean_roc_auc=("roc_auc", "mean"),
                mean_best_precision=("precision_policy_best_practical_precision", "mean"),
                max_best_precision=("precision_policy_best_practical_precision", "max"),
                mean_best_recall=("precision_policy_best_practical_recall", "mean"),
            )
        )
    if portfolio_fold_metrics.empty:
        role_summary = pd.DataFrame()
    else:
        role_summary = (
            portfolio_fold_metrics.groupby("role", as_index=False)
            .agg(
                fold_count=("fold_id", "nunique"),
                mean_alerts=("alerts", "mean"),
                total_alerts=("alerts", "sum"),
                total_tp=("tp", "sum"),
                total_fp=("fp", "sum"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                minimum_precision=("precision", "min"),
                minimum_recall=("recall", "min"),
                gate_pass_rate=("precision_gate", "mean"),
                mean_pr_auc=("pr_auc", "mean"),
                mean_roc_auc=("roc_auc", "mean"),
            )
        )
        role_summary["pooled_precision"] = role_summary["total_tp"] / (
            role_summary["total_tp"] + role_summary["total_fp"]
        ).replace(0, np.nan)
    return ticker_summary, role_summary


def build_graph_outputs(
    map_summary: pd.DataFrame,
    edges: pd.DataFrame,
    clusters: pd.DataFrame,
    representatives: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_set = set(
        zip(representatives["ticker"].astype(str), representatives["representative"].astype(str))
    ) if not representatives.empty else set()
    cluster_lookup = {
        (str(row.ticker), str(row.feature)): (int(row.cluster_id), int(row.cluster_size))
        for row in clusters.itertuples(index=False)
    } if not clusters.empty else {}
    node_rows: list[dict[str, Any]] = []
    for row in map_summary.itertuples(index=False):
        key = (str(row.ticker), str(row.feature))
        cluster_id, cluster_size = cluster_lookup.get(key, (-1, 1))
        node_rows.append(
            {
                "node_id": f"{row.ticker}::{row.axis}::{row.feature}::{row.transform}",
                "ticker": str(row.ticker),
                "axis": str(row.axis),
                "feature": str(row.feature),
                "transform": str(row.transform),
                "selection_direction": int(row.selection_direction),
                "selection_mean_fixed_auc": float(row.selection_mean_fixed_auc),
                "selection_min_fixed_auc": float(row.selection_min_fixed_auc),
                "confirmation_mean_fixed_auc": float(row.confirmation_mean_fixed_auc),
                "recent_mean_fixed_auc": float(row.recent_mean_fixed_auc),
                "ticker_signal_score": float(row.ticker_signal_score),
                "evidence_grade": str(row.evidence_grade),
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "is_representative": key in rep_set,
            }
        )
    nodes = pd.DataFrame(node_rows)
    edge_rows: list[dict[str, Any]] = []
    if not edges.empty:
        for row in edges.itertuples(index=False):
            edge_rows.append(
                {
                    "source": f"{row.ticker}::TARGET::{row.feature_a}::raw",
                    "target": f"{row.ticker}::TARGET::{row.feature_b}::raw",
                    "ticker": str(row.ticker),
                    "edge_type": "ticker_correlation",
                    "weight": float(row.full_abs_correlation),
                    "correlation": float(row.full_correlation),
                    "stable_edge": bool(row.stable_edge),
                }
            )
    graph_edges = pd.DataFrame(edge_rows)
    if not nodes.empty:
        write_graphml(output / "ticker_separation_graph_v10.graphml", nodes, graph_edges)
    return nodes, graph_edges


def write_ticker_reports(
    output: Path,
    ticker_summary: pd.DataFrame,
    map_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
    eligibility: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# CrashWatch Surge 종목별 상관관계 지도 V10 결과 보고서")
    lines.append("")
    lines.append("## 핵심 판정")
    lines.append("")
    if role_summary.empty:
        lines.append("종목별 모델의 포트폴리오 정책 결과가 생성되지 않았다.")
    else:
        all_pass = bool(
            (role_summary["gate_pass_rate"] >= float(args.required_fold_pass_rate)).all()
            and (role_summary["pooled_precision"] >= float(args.target_precision)).all()
            and (role_summary["total_alerts"] >= int(args.minimum_portfolio_alerts)).all()
        )
        lines.append(
            "`READY_FOR_FROZEN_TICKER_CONFIRMATION`" if all_pass else "`STOP_TICKER_PRECISION_GATE`"
        )
        lines.append("")
        lines.append(role_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 종목별 증거 부족")
    lines.append("")
    low = eligibility.loc[~eligibility["eligible_model"]]
    lines.append(f"독립 모델 기준 미충족 ticker-fold: {len(low):,}건")
    lines.append("")
    lines.append("표본 부족 종목은 공통 모델로 대체하지 않고 `LOW_EVIDENCE_NO_MODEL` 또는 `NO_ALERT`로 남겼다.")
    lines.append("")
    lines.append("## 종목별 지도 해석")
    lines.append("")
    if not map_summary.empty:
        top = (
            map_summary.loc[map_summary["axis"].eq("AB")]
            .sort_values(["ticker_signal_score"], ascending=False)
            .groupby("ticker", as_index=False)
            .head(3)
        )
        lines.append(top[[
            "ticker", "feature", "transform", "selection_mean_fixed_auc", "selection_min_fixed_auc",
            "confirmation_mean_fixed_auc", "recent_mean_fixed_auc", "evidence_grade"
        ]].to_markdown(index=False))
    atomic_write_text(output / "TICKER_SEPARATION_MAP_REPORT_KO.md", "\n".join(lines) + "\n")


def write_per_ticker_exports(
    output: Path,
    map_summary: pd.DataFrame,
    edges: pd.DataFrame,
    clusters: pd.DataFrame,
    representatives: pd.DataFrame,
    ticker_probe_summary: pd.DataFrame,
) -> None:
    root = output / "per_ticker"
    root.mkdir(parents=True, exist_ok=True)
    tickers = sorted(map_summary["ticker"].astype(str).unique().tolist()) if not map_summary.empty else []
    for ticker in tickers:
        ticker_dir = root / str(ticker)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(ticker_dir / "separation_map.csv", map_summary.loc[map_summary["ticker"].astype(str).eq(ticker)])
        if not edges.empty:
            atomic_write_csv(ticker_dir / "correlation_edges.csv", edges.loc[edges["ticker"].astype(str).eq(ticker)])
        if not clusters.empty:
            atomic_write_csv(ticker_dir / "clusters.csv", clusters.loc[clusters["ticker"].astype(str).eq(ticker)])
        if not representatives.empty:
            atomic_write_csv(
                ticker_dir / "cluster_representatives.csv",
                representatives.loc[representatives["ticker"].astype(str).eq(ticker)],
            )
        if not ticker_probe_summary.empty:
            atomic_write_csv(
                ticker_dir / "probe_metrics.csv",
                ticker_probe_summary.loc[ticker_probe_summary["ticker"].astype(str).eq(ticker)],
            )


def create_manifest(
    output: Path,
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    final_status: str,
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": str(Path(args.dataset)),
        "dataset_sha256": sha256_file(Path(args.dataset)),
        "target_sidecar": str(Path(args.target_sidecar)),
        "target_sidecar_sha256": sha256_file(Path(args.target_sidecar)),
        "folds": str(Path(args.folds)),
        "folds_sha256": sha256_file(Path(args.folds)),
        "feature_profile_manifest": str(Path(args.feature_profile_manifest)),
        "feature_profile_manifest_sha256": sha256_file(Path(args.feature_profile_manifest)),
        "feature_profile": args.feature_profile,
        "feature_count": len(features),
        "feature_hash": hash_strings(list(features)),
        "row_count": int(len(frame)),
        "ticker_count": int(frame[args.ticker_column].nunique()),
        "date_min": pd.Timestamp(frame[args.date_column].min()).strftime("%Y-%m-%d"),
        "date_max": pd.Timestamp(frame[args.date_column].max()).strftime("%Y-%m-%d"),
        "positive_count": int(frame[args.target_column].sum()),
        "positive_rate": float(frame[args.target_column].mean()),
        "roles": {key: list(map(int, value)) for key, value in roles.items()},
        "folds_definition": [fold.to_dict() for fold in folds],
        "final_gate_status": final_status,
        "config": vars(args),
    }
    atomic_write_json(output / "TICKER_SEPARATION_MAP_MANIFEST_V10.json", manifest)
    return manifest


def save_matrix_payload(output: Path, payload: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    root = output / "ticker_correlation_matrices"
    root.mkdir(parents=True, exist_ok=True)
    for ticker, arrays in payload.items():
        atomic_write_npz(
            root / f"{ticker}.npz",
            feature_names=np.asarray(arrays["feature_names"]),
            correlation=np.asarray(arrays["correlation"], dtype=np.float32),
            valid_n=np.asarray(arrays["valid_n"], dtype=np.int32),
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.package_root).resolve()
    resolve_default_paths(package_root, args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = RunStatus(output, "CrashWatch Surge Tickerwise Correlation Map V10")
    try:
        require_paths(
            {
                "dataset": Path(args.dataset),
                "target_sidecar": Path(args.target_sidecar),
                "folds": Path(args.folds),
                "feature_profile_manifest": Path(args.feature_profile_manifest),
            }
        )
        dataset_columns = table_columns(Path(args.dataset))
        features = load_feature_universe(args, dataset_columns)
        frame = load_dataset_frame(args, features)
        folds = load_folds(Path(args.folds))
        roles = resolve_roles(args, folds)
        global_fold_index = build_global_fold_index(frame, folds, args.date_column)
        ticker_fold_index, eligibility = build_ticker_fold_index(frame, folds, global_fold_index, args)
        target_summary = ticker_target_summary(frame, args)
        atomic_write_csv(output / "ticker_target_summary.csv", target_summary)
        atomic_write_csv(output / "ticker_fold_eligibility.csv", eligibility)
        status.stage("load", "SUCCESS", rows=len(frame), tickers=frame[args.ticker_column].nunique(), features=len(features))

        base_oof, base_feature_manifest = build_ticker_base_oof(
            frame, features, folds, ticker_fold_index, eligibility, roles, args
        )
        oof_metadata = attach_oof_metadata(frame, base_oof, args)
        oof_groups, error_counts = add_ticker_error_groups(oof_metadata, args)
        atomic_write_csv(output / "ticker_error_group_oof.csv", oof_groups)
        atomic_write_csv(output / "ticker_error_group_counts_by_fold.csv", error_counts)
        status.stage("ticker_base_oof", "SUCCESS", rows=len(base_oof))

        matched_ab, matched_cd = build_all_ticker_matched_pairs(frame, oof_groups, features, args)
        atomic_write_csv(output / "ticker_matched_pairs_ab.csv", matched_ab)
        atomic_write_csv(output / "ticker_matched_pairs_cd.csv", matched_cd)
        status.stage("matching", "SUCCESS", ab_pairs=len(matched_ab), cd_pairs=len(matched_cd))

        stage0_by_fold, stage0_summary, feature_quality = compute_ticker_stage0_maps(
            frame, oof_groups, features, matched_ab, matched_cd, args
        )
        atomic_write_csv(output / "ticker_feature_map_by_fold.csv", stage0_by_fold)
        atomic_write_csv(output / "ticker_feature_map_summary.csv", stage0_summary)
        atomic_write_csv(output / "ticker_feature_quality.csv", feature_quality)
        status.stage("stage0_map", "SUCCESS", rows=len(stage0_summary))

        edges, clusters, representatives, matrix_payload = build_ticker_correlation_maps(
            frame, features, folds, roles, stage0_summary, args
        )
        atomic_write_csv(output / "ticker_correlation_edges.csv", edges)
        atomic_write_csv(output / "ticker_cluster_assignments.csv", clusters)
        atomic_write_csv(output / "ticker_cluster_representatives.csv", representatives)
        if args.save_full_matrices:
            save_matrix_payload(output, matrix_payload)
        status.stage("ticker_organic_map", "SUCCESS", edges=len(edges), clusters=clusters["cluster_id"].nunique() if not clusters.empty else 0)

        selected_sources = select_ticker_sources(stage0_summary, representatives, args)
        atomic_write_csv(output / "ticker_selected_source_features.csv", selected_sources)
        transformed, transform_manifest = build_selected_transforms(
            frame, selected_sources, clusters, matrix_payload, args
        )
        atomic_write_json(output / "TICKER_TRANSFORM_MANIFEST_V10.json", transform_manifest)
        extended_by_fold, extended_summary = compute_ticker_extended_maps(
            transformed, oof_groups, selected_sources, transform_manifest, matched_ab, matched_cd, args
        )
        atomic_write_csv(output / "ticker_extended_map_by_fold.csv", extended_by_fold)
        atomic_write_csv(output / "ticker_extended_map_summary.csv", extended_summary)
        combined_map = pd.concat([stage0_summary, extended_summary], ignore_index=True, sort=False)
        combined_by_fold = pd.concat([stage0_by_fold, extended_by_fold], ignore_index=True, sort=False)
        atomic_write_csv(output / "ticker_separation_map_complete.csv", combined_map)
        status.stage("extended_map", "SUCCESS", rows=len(extended_summary))

        hierarchy_config = hierarchy_config_from_args(args)
        hierarchy_map, peer_reference_map = build_hierarchical_effect_map(
            combined_map,
            combined_by_fold,
            target_summary,
            roles["selection"],
            hierarchy_config,
        )
        hierarchy_profiles, hierarchy_membership = build_ticker_driver_profiles(
            hierarchy_map,
            per_axis_count=int(args.hierarchy_profile_per_axis_count),
            unique_count=int(args.hierarchy_profile_unique_count),
            shared_count=int(args.hierarchy_profile_shared_count),
            reversal_count=int(args.hierarchy_profile_reversal_count),
        )
        similarity_edges, ticker_clusters, similarity_matrix, similarity_overlap = build_ticker_similarity_map(
            hierarchy_map, hierarchy_config
        )
        taxonomy_summary = (
            hierarchy_map.groupby(["axis", "ticker_effect_class"], as_index=False)
            .agg(nodes=("node_id", "size"), tickers=("ticker", "nunique"), mean_driver_score=("ticker_driver_score", "mean"))
            .sort_values(["axis", "nodes", "ticker_effect_class"], ascending=[True, False, True], kind="mergesort")
        )
        atomic_write_csv(output / "ticker_hierarchical_effect_map.csv", hierarchy_map)
        atomic_write_csv(output / "ticker_peer_reference_map.csv", peer_reference_map)
        atomic_write_csv(output / "ticker_effect_taxonomy_summary.csv", taxonomy_summary)
        atomic_write_json(output / "TICKER_DRIVER_PROFILES_V10.json", hierarchy_profiles)
        atomic_write_csv(output / "ticker_driver_profile_membership.csv", hierarchy_membership)
        atomic_write_csv(output / "ticker_similarity_edges.csv", similarity_edges)
        atomic_write_csv(output / "ticker_map_clusters.csv", ticker_clusters)
        atomic_write_csv(output / "ticker_similarity_matrix.csv", similarity_matrix.reset_index().rename(columns={"index": "ticker"}))
        atomic_write_csv(output / "ticker_similarity_overlap.csv", similarity_overlap.reset_index().rename(columns={"index": "ticker"}))
        write_ticker_similarity_graph(output, target_summary, similarity_edges, ticker_clusters)
        visualization_files: list[str] = []
        if bool(args.create_plots):
            visualization_files = save_ticker_map_visualizations(
                hierarchy_map,
                similarity_matrix,
                output,
                per_ticker_top_n=int(args.plot_top_n),
                heatmap_node_count=int(args.plot_heatmap_nodes),
            )
            atomic_write_json(output / "TICKER_VISUALIZATION_MANIFEST_V10.json", visualization_files)
        status.stage(
            "hierarchical_ticker_map",
            "SUCCESS",
            nodes=len(hierarchy_map),
            ticker_specific_nodes=int(hierarchy_map["is_ticker_specific"].sum()) if not hierarchy_map.empty else 0,
            similarity_edges=len(similarity_edges),
            visualizations=len(visualization_files),
        )

        base_profiles, base_profile_membership = build_ticker_profile_manifest(
            stage0_summary, extended_summary, representatives, selected_sources, args
        )
        profiles, profile_membership = merge_hierarchical_profiles(
            base_profiles,
            base_profile_membership,
            hierarchy_profiles,
            hierarchy_membership,
            combined_max_count=int(args.profile_combined_max_count),
        )
        atomic_write_json(output / "TICKER_FEATURE_PROFILES_V10.json", profiles)
        atomic_write_csv(output / "ticker_profile_membership.csv", profile_membership)

        if bool(args.run_probe):
            model_frame = build_model_frame(frame, transformed, args)
            probe_predictions, ticker_fold_metrics = fit_per_ticker_probe_models(
                model_frame,
                base_oof,
                profiles,
                folds,
                ticker_fold_index,
                eligibility,
                roles,
                features,
                args,
            )
            profile_router, routed = select_ticker_profile_forward(probe_predictions, folds, roles, args)
            calibrated, calibration_audit = calibrate_scores_per_ticker_forward(routed, roles, args)
            ticker_policies, scored_alerts, portfolio_fold_metrics = evaluate_forward_ticker_policies(
                calibrated, roles, args
            )
            ticker_probe_summary, role_summary = aggregate_probe_results(
                ticker_fold_metrics, portfolio_fold_metrics, args
            )
            atomic_write_csv(output / "ticker_profile_router_by_fold.csv", profile_router)
            atomic_write_csv(output / "ticker_calibration_audit.csv", calibration_audit)
            atomic_write_csv(output / "ticker_threshold_policy_by_fold.csv", ticker_policies)
            atomic_write_csv(output / "ticker_alert_predictions.csv", scored_alerts)
            atomic_write_csv(output / "ticker_portfolio_metrics_by_fold.csv", portfolio_fold_metrics)
            atomic_write_csv(output / "ticker_probe_metrics_summary.csv", ticker_probe_summary)
            atomic_write_csv(output / "ticker_portfolio_metrics_by_role.csv", role_summary)
            status.stage("ticker_probe", "SUCCESS", prediction_rows=len(probe_predictions))
        else:
            ticker_probe_summary = pd.DataFrame()
            role_summary = pd.DataFrame()
            status.stage("ticker_probe", "SKIPPED", reason="--no-run-probe")

        nodes, graph_edges = build_graph_outputs(combined_map, edges, clusters, representatives, output)
        atomic_write_csv(output / "ticker_separation_graph_nodes.csv", nodes)
        atomic_write_csv(output / "ticker_separation_graph_edges.csv", graph_edges)
        write_per_ticker_exports(output, combined_map, edges, clusters, representatives, ticker_probe_summary)
        write_per_ticker_hierarchy_exports(
            output,
            hierarchy_map,
            hierarchy_membership,
            similarity_edges,
            ticker_clusters,
        )

        if not bool(args.run_probe):
            final_gate = False
            final_status = "TICKER_MAP_COMPLETE_PROBE_NOT_RUN"
        elif role_summary.empty:
            final_gate = False
            final_status = "STOP_TICKER_PRECISION_GATE"
        else:
            final_gate = bool(
                (role_summary["gate_pass_rate"] >= float(args.required_fold_pass_rate)).all()
                and (role_summary["pooled_precision"] >= float(args.target_precision)).all()
                and (role_summary["total_alerts"] >= int(args.minimum_portfolio_alerts)).all()
            )
            final_status = "READY_FOR_FROZEN_TICKER_CONFIRMATION" if final_gate else "STOP_TICKER_PRECISION_GATE"
        recommendation = {
            "schema": SCHEMA_VERSION,
            "status": final_status,
            "target_precision": float(args.target_precision),
            "minimum_portfolio_alerts": int(args.minimum_portfolio_alerts),
            "common_prediction_model_used": False,
            "ticker_specific_models": bool(args.run_probe),
            "hierarchical_peer_reference_used_for_map_only": True,
            "peer_reference_is_leave_one_ticker_out": True,
            "low_evidence_action": "NO_ALERT",
            "production_action": (
                "FREEZE_AND_USE_NEW_FUTURE_HOLDOUT"
                if final_gate
                else "MAP_ONLY_RESEARCH" if not bool(args.run_probe) else "NO_ALERT_AND_CONTINUE_TICKER_RESEARCH"
            ),
        }
        atomic_write_json(output / "FINAL_RECOMMENDATION_V10.json", recommendation)
        write_ticker_reports(output, ticker_probe_summary, combined_map, role_summary, eligibility, args)
        append_hierarchy_report(
            output / "TICKER_SEPARATION_MAP_REPORT_KO.md",
            hierarchy_map,
            similarity_edges,
            ticker_clusters,
        )
        manifest = create_manifest(output, frame, features, folds, roles, args, final_status)
        inventory = compute_output_inventory(output, exclude=["RUN_STATUS.json", "OUTPUT_INVENTORY_V10.json"])
        atomic_write_json(output / "OUTPUT_INVENTORY_V10.json", inventory)
        status.success(
            final_gate_status=final_status,
            rows=len(frame),
            tickers=int(frame[args.ticker_column].nunique()),
            features=len(features),
            output_files=len(inventory),
        )
        return {"status": final_status, "manifest": manifest, "recommendation": recommendation}
    except BaseException as exc:
        status.failure(exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build independent ticker-wise surge correlation/separation maps with hierarchical peer references."
    )
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--feature-profile-manifest", type=Path)
    parser.add_argument("--feature-profile", default="P0_FULL_439")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--market-column", default="market")
    parser.add_argument("--bucket-column", default="bucket")
    parser.add_argument("--industry-column", default="industry_name")
    parser.add_argument("--target-column", default="label_abs_surge_3d_5pct")
    parser.add_argument("--target-valid-column", default="target_valid")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--limit-features", type=int, default=0)
    parser.add_argument("--limit-tickers", type=int, default=0, help="Development benchmark only; 0 keeps all tickers")
    parser.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--base-backends", default="lightgbm_cpu,xgboost_gpu")
    parser.add_argument("--probe-backends", default="lightgbm_cpu,xgboost_gpu")
    parser.add_argument("--strict-backend", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--xgboost-threads", type=int, default=8)
    parser.add_argument("--model-workers", type=int, default=1, help="Concurrent independent ticker-fold base models")
    parser.add_argument(
        "--parallel-backends",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlap CPU and CUDA backends within each ticker-fold job",
    )
    parser.add_argument("--analysis-workers", type=int, default=1, help="Concurrent ticker-fold map calculations")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-iterations", type=int, default=160)
    parser.add_argument("--base-feature-count", type=int, default=64)
    parser.add_argument("--minimum-base-feature-count", type=int, default=12)
    parser.add_argument("--base-dedup-threshold", type=float, default=0.98)

    parser.add_argument("--minimum-train-rows-model", type=int, default=400)
    parser.add_argument("--minimum-train-positive-model", type=int, default=35)
    parser.add_argument("--minimum-train-negative-model", type=int, default=100)
    parser.add_argument("--minimum-validation-rows-model", type=int, default=60)
    parser.add_argument("--minimum-validation-positive-model", type=int, default=5)
    parser.add_argument("--minimum-validation-negative-model", type=int, default=20)
    parser.add_argument("--minimum-validation-rows-map", type=int, default=40)
    parser.add_argument("--minimum-validation-positive-map", type=int, default=3)
    parser.add_argument("--minimum-validation-negative-map", type=int, default=10)
    parser.add_argument("--minimum-feature-valid-rows", type=int, default=80)
    parser.add_argument("--minimum-feature-positive-rows", type=int, default=10)
    parser.add_argument("--minimum-feature-negative-rows", type=int, default=30)
    parser.add_argument("--minimum-feature-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-map-valid-rows", type=int, default=20)
    parser.add_argument("--auc-prior-strength", type=float, default=80.0)

    parser.add_argument("--error-top-quantile", type=float, default=0.80)
    parser.add_argument("--error-low-quantile", type=float, default=0.50)
    parser.add_argument("--controls-per-case", type=int, default=3)
    parser.add_argument("--maximum-match-day-distance", type=int, default=756)
    parser.add_argument("--match-regime-columns", type=int, default=6)

    parser.add_argument("--correlation-method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--minimum-correlation-rows", type=int, default=120)
    parser.add_argument("--correlation-shrinkage-strength", type=float, default=60.0)
    parser.add_argument("--correlation-edge-threshold", type=float, default=0.92)
    parser.add_argument("--minimum-fold-abs-correlation", type=float, default=0.80)
    parser.add_argument("--minimum-correlation-sign-consistency", type=float, default=0.80)
    parser.add_argument("--maximum-edges-per-ticker", type=int, default=2000)
    parser.add_argument("--cluster-threshold", type=float, default=0.92)
    parser.add_argument("--save-full-matrices", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--minimum-selection-mean-auc", type=float, default=0.56)
    parser.add_argument("--minimum-selection-min-auc", type=float, default=0.51)
    parser.add_argument("--minimum-direction-consistency", type=float, default=0.75)
    parser.add_argument("--minimum-selection-fold-count", type=int, default=2)
    parser.add_argument("--target-source-count", type=int, default=12)
    parser.add_argument("--ab-source-count", type=int, default=16)
    parser.add_argument("--cd-source-count", type=int, default=16)

    parser.add_argument("--profile-target-count", type=int, default=16)
    parser.add_argument("--profile-ab-count", type=int, default=16)
    parser.add_argument("--profile-cd-count", type=int, default=12)
    parser.add_argument("--profile-extended-target-count", type=int, default=8)
    parser.add_argument("--profile-extended-ab-count", type=int, default=12)
    parser.add_argument("--profile-extended-cd-count", type=int, default=8)
    parser.add_argument("--profile-cluster-rep-count", type=int, default=32)
    parser.add_argument("--profile-combined-max-count", type=int, default=64)
    parser.add_argument("--probe-profiles", default="BASE_TICKER,TARGET_TOP,AB_PRECISION,CD_RECOVERY,CLUSTER_REP,HIERARCHICAL_COMBINED,COMBINED")
    parser.add_argument("--minimum-probe-feature-count", type=int, default=4)

    parser.add_argument("--minimum-calibration-rows", type=int, default=80)
    parser.add_argument("--minimum-calibration-positive", type=int, default=8)
    parser.add_argument("--minimum-calibration-rows-rank", type=int, default=30)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--minimum-alerts-per-ticker-diagnostic", type=int, default=3)
    parser.add_argument("--minimum-alerts-per-ticker-policy", type=int, default=3)
    parser.add_argument("--minimum-recall-per-ticker-policy", type=float, default=0.0)
    parser.add_argument("--minimum-precision-lcb-per-ticker", type=float, default=0.0)
    parser.add_argument("--minimum-portfolio-alerts", type=int, default=30)
    parser.add_argument("--required-fold-pass-rate", type=float, default=1.0)
    parser.add_argument("--run-probe", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--hierarchy-prior-strength", type=float, default=120.0)
    parser.add_argument("--minimum-peer-tickers", type=int, default=2)
    parser.add_argument("--hierarchy-weak-effect", type=float, default=0.015)
    parser.add_argument("--hierarchy-strong-effect", type=float, default=0.035)
    parser.add_argument("--hierarchy-unique-delta", type=float, default=0.025)
    parser.add_argument("--hierarchy-amplified-delta", type=float, default=0.020)
    parser.add_argument("--hierarchy-reversal-effect", type=float, default=0.020)
    parser.add_argument("--hierarchy-minimum-reliability", type=float, default=0.10)
    parser.add_argument("--hierarchy-confirmation-margin", type=float, default=0.0)
    parser.add_argument("--hierarchy-recent-margin", type=float, default=0.0)
    parser.add_argument("--hierarchy-industry-weight", type=float, default=0.35)
    parser.add_argument("--hierarchy-bucket-weight", type=float, default=0.35)
    parser.add_argument("--hierarchy-market-weight", type=float, default=0.20)
    parser.add_argument("--hierarchy-global-weight", type=float, default=0.10)
    parser.add_argument("--ticker-similarity-min-common", type=int, default=20)
    parser.add_argument("--ticker-similarity-edge-threshold", type=float, default=0.35)
    parser.add_argument("--ticker-similarity-top-k", type=int, default=5)
    parser.add_argument("--ticker-similarity-cluster-threshold", type=float, default=0.30)
    parser.add_argument("--ticker-similarity-feature-count", type=int, default=120)
    parser.add_argument("--hierarchy-profile-per-axis-count", type=int, default=12)
    parser.add_argument("--hierarchy-profile-unique-count", type=int, default=16)
    parser.add_argument("--hierarchy-profile-shared-count", type=int, default=16)
    parser.add_argument("--hierarchy-profile-reversal-count", type=int, default=8)
    parser.add_argument("--create-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-top-n", type=int, default=20)
    parser.add_argument("--plot-heatmap-nodes", type=int, default=40)

    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
