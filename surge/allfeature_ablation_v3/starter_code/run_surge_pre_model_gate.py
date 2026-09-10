from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from surge_ablation_common import (
    FileLock,
    RunStatus,
    atomic_write_csv,
    atomic_write_json,
    compute_output_inventory,
    deterministic_seed,
    daily_top_fraction_metrics,
    fraction_label,
    hash_strings,
    join_source_and_target,
    load_folds,
    load_json,
    load_roles_from_manifest,
    log,
    model_versions,
    parse_bool_series,
    parse_float_list,
    pooled_top_fraction_metrics,
    read_table,
    role_for_fold,
    sha256_bytes,
    sha256_file,
    sign_consistency,
    stable_json_bytes,
    target_metric_for_feature,
    table_columns,
    utc_now,
    validate_folds,
    validate_role_assignments,
    verify_feature_names,
    verify_output_inventory,
)


METRIC_COLUMNS = {
    "pearson": ("train_target_pearson", "validation_target_pearson"),
    "spearman": ("train_target_spearman", "validation_target_spearman"),
    "within_date_pearson": (
        "train_target_within_date_pearson",
        "validation_target_within_date_pearson",
    ),
    "within_ticker_pearson": (
        "train_target_within_ticker_pearson",
        "validation_target_within_ticker_pearson",
    ),
}

DIRECTIONAL_CLASSES = {
    "SURGE_SPECIFIC",
    "OPPOSITE_DIRECTION",
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
    if args.output is None:
        args.output = package_root / "outputs" / "surge_pre_model_gate_v3"


def require_files(paths: Mapping[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"필수 파일 누락: {json.dumps(missing, ensure_ascii=False, indent=2)}")


def load_feature_universe(correlation_dir: Path) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = correlation_dir / "surge_feature_correlation_summary.csv"
    fold_path = correlation_dir / "surge_feature_target_correlation_by_fold.csv"
    support_path = correlation_dir / "surge_support_audit.csv"
    membership_path = correlation_dir / "surge_feature_profile_membership.csv"
    clusters_path = correlation_dir / "primary_clusters.csv"
    require_files(
        {
            "summary": summary_path,
            "fold correlations": fold_path,
            "support audit": support_path,
            "profile membership": membership_path,
            "primary clusters": clusters_path,
        }
    )
    summary = pd.read_csv(summary_path)
    fold_correlations = pd.read_csv(fold_path)
    support = pd.read_csv(support_path)
    membership = pd.read_csv(membership_path)
    clusters = pd.read_csv(clusters_path)
    if "feature" not in summary.columns:
        raise ValueError("surge_feature_correlation_summary.csv에 feature 열이 없습니다")

    
    if "primary_cluster_id" not in summary.columns and "feature" in clusters.columns:
        primary = clusters.copy()
        if "threshold" in primary.columns and primary["threshold"].notna().any():
            thresholds = pd.to_numeric(primary["threshold"], errors="coerce")
            available = sorted(float(value) for value in thresholds.dropna().unique())
            if available:
                selected_threshold = min(available, key=lambda value: abs(value - 0.92))
                primary = primary[np.isclose(thresholds, selected_threshold)]
        primary = primary.drop_duplicates("feature")
        rename = {
            "cluster_id": "primary_cluster_id",
            "cluster_size": "primary_cluster_size",
            "representative": "primary_representative",
            "is_representative": "is_primary_representative",
        }
        primary.rename(columns={key: value for key, value in rename.items() if key in primary.columns}, inplace=True)
        carry = [
            column
            for column in [
                "feature", "primary_cluster_id", "primary_cluster_size",
                "primary_representative", "is_primary_representative",
            ]
            if column in primary.columns
        ]
        if len(carry) > 1:
            summary = summary.merge(primary[carry], on="feature", how="left", validate="one_to_one")

    if "selected_for_correlation" in summary.columns:
        selected = parse_bool_series(summary["selected_for_correlation"])
        features = summary.loc[selected, "feature"].astype(str).tolist()
    else:
        features = summary["feature"].astype(str).tolist()
    if len(features) != len(set(features)):
        raise ValueError("상관지도 feature universe에 중복 피처가 있습니다")
    return features, summary, fold_correlations, support, membership


def metric_name_from_support(row: pd.Series) -> str:
    source = str(row.get("selection_direction_source", "pearson"))
    if source not in METRIC_COLUMNS:
        support_metric = str(row.get("selection_support_metric", ""))
        for metric, (_, validation_column) in METRIC_COLUMNS.items():
            if support_metric == validation_column:
                return metric
        return "pearson"
    return source


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "mean": np.nan,
            "abs_mean": np.nan,
            "mean_abs": np.nan,
            "min_abs": np.nan,
            "sign_consistency": np.nan,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "abs_mean": float(abs(np.mean(finite))),
        "mean_abs": float(np.mean(np.abs(finite))),
        "min_abs": float(np.min(np.abs(finite))),
        "sign_consistency": float(sign_consistency(finite)),
    }


def compute_train_validation_alignment(
    features: Sequence[str],
    support: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    selection_fold_ids: Sequence[int],
    minimum_match_ratio: float,
    minimum_train_abs_mean: float,
) -> pd.DataFrame:
    support_index = support.drop_duplicates("feature").set_index("feature")
    fold_index = {
        str(feature): part.set_index("outer_fold", drop=False)
        for feature, part in fold_correlations.groupby("feature", sort=False)
    }
    expected = sorted({int(value) for value in selection_fold_ids})
    records: list[dict[str, Any]] = []
    for feature in features:
        if feature not in support_index.index:
            raise ValueError(f"support audit에 피처 누락: {feature}")
        support_row = support_index.loc[feature]
        metric = metric_name_from_support(support_row)
        train_column, validation_column = METRIC_COLUMNS[metric]
        part = fold_index.get(feature)
        if part is None:
            raise ValueError(f"fold correlation에 피처 누락: {feature}")
        selection = part.reindex(expected)
        train_values = selection[train_column].to_numpy(dtype=np.float64)
        validation_values = selection[validation_column].to_numpy(dtype=np.float64)
        paired = np.isfinite(train_values) & np.isfinite(validation_values)
        nonzero = paired & (np.abs(train_values) > 1e-12) & (np.abs(validation_values) > 1e-12)
        sign_matches = np.zeros(len(expected), dtype=bool)
        sign_matches[nonzero] = np.sign(train_values[nonzero]) == np.sign(validation_values[nonzero])
        paired_count = int(np.sum(nonzero))
        match_count = int(np.sum(sign_matches))
        match_ratio = float(match_count / len(expected)) if expected else np.nan
        train_stats = finite_stats(train_values)
        validation_stats = finite_stats(validation_values)
        train_mean = float(train_stats["mean"])
        validation_mean = float(validation_stats["mean"])
        mean_sign_match = bool(
            np.isfinite(train_mean)
            and np.isfinite(validation_mean)
            and abs(train_mean) > 1e-12
            and abs(validation_mean) > 1e-12
            and np.sign(train_mean) == np.sign(validation_mean)
        )
        reasons: list[str] = []
        if paired_count != len(expected):
            reasons.append("missing_or_zero_train_validation_fold")
        if not mean_sign_match:
            reasons.append("train_validation_mean_sign_mismatch")
        if not np.isfinite(match_ratio) or match_ratio < minimum_match_ratio:
            reasons.append("train_validation_fold_sign_match_ratio_below_min")
        if not np.isfinite(float(train_stats["mean_abs"])) or float(train_stats["mean_abs"]) < minimum_train_abs_mean:
            reasons.append("train_metric_abs_mean_below_min")
        aligned = not reasons
        record: dict[str, Any] = {
            "feature": feature,
            "selected_metric": metric,
            "train_metric_column": train_column,
            "validation_metric_column": validation_column,
            "selection_fold_count_expected": len(expected),
            "selection_fold_count_paired_nonzero": paired_count,
            "train_validation_fold_sign_match_count": match_count,
            "train_validation_fold_sign_match_ratio": match_ratio,
            "train_validation_mean_sign_match": mean_sign_match,
            "train_validation_aligned": aligned,
            "train_validation_alignment_reason": "|".join(reasons),
        }
        for prefix, stats in (("selection_train_selected_metric", train_stats), ("selection_validation_selected_metric", validation_stats)):
            for key, value in stats.items():
                record[f"{prefix}_{key}"] = value
        for index, fold_id in enumerate(expected):
            record[f"fold_{fold_id}_train_selected_metric"] = train_values[index]
            record[f"fold_{fold_id}_validation_selected_metric"] = validation_values[index]
            record[f"fold_{fold_id}_train_validation_sign_match"] = bool(sign_matches[index])
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_target_correlations_for_frame(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
) -> pd.DataFrame:
    target_series = pd.Series(target, index=feature_frame.index, dtype="float64")
    pearson = feature_frame.corrwith(target_series, axis=0, method="pearson")
    spearman = feature_frame.corrwith(target_series, axis=0, method="spearman")

    date_groups = pd.Series(dates, index=feature_frame.index)
    ticker_groups = pd.Series(tickers, index=feature_frame.index)
    date_residual_features = feature_frame - feature_frame.groupby(date_groups, sort=False).transform("mean")
    date_residual_target = target_series - target_series.groupby(date_groups, sort=False).transform("mean")
    within_date = date_residual_features.corrwith(date_residual_target, axis=0, method="pearson")
    del date_residual_features

    ticker_residual_features = feature_frame - feature_frame.groupby(ticker_groups, sort=False).transform("mean")
    ticker_residual_target = target_series - target_series.groupby(ticker_groups, sort=False).transform("mean")
    within_ticker = ticker_residual_features.corrwith(ticker_residual_target, axis=0, method="pearson")
    del ticker_residual_features

    return pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "pearson": pearson.reindex(feature_frame.columns).to_numpy(dtype=np.float64),
            "spearman": spearman.reindex(feature_frame.columns).to_numpy(dtype=np.float64),
            "within_date_pearson": within_date.reindex(feature_frame.columns).to_numpy(dtype=np.float64),
            "within_ticker_pearson": within_ticker.reindex(feature_frame.columns).to_numpy(dtype=np.float64),
        }
    )


def compute_crash_metric_consistent_map(
    data: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[Any],
    roles: Mapping[str, Sequence[int]],
    crash_target_column: str,
    date_column: str,
    ticker_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if crash_target_column not in data.columns:
        raise ValueError(f"급락 비교 target 열이 데이터에 없습니다: {crash_target_column}")
    crash_target = pd.to_numeric(data[crash_target_column], errors="coerce").to_numpy(dtype=np.float64)
    dates = pd.to_datetime(data[date_column], errors="coerce")
    tickers = data[ticker_column].astype("string").fillna("<NA>").to_numpy()
    records: list[pd.DataFrame] = []
    for fold in folds:
        validation_mask = dates.between(fold.validation_start, fold.validation_end).to_numpy()
        validation_mask &= np.isfinite(crash_target) & np.isin(crash_target, [0.0, 1.0])
        indices = np.flatnonzero(validation_mask)
        if not len(indices):
            raise ValueError(f"fold {fold.fold_id}: crash target validation 행이 없습니다")
        feature_frame = data.iloc[indices][list(features)].apply(pd.to_numeric, errors="coerce").astype("float64")
        feature_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        result = compute_target_correlations_for_frame(
            feature_frame,
            crash_target[indices],
            dates.iloc[indices].to_numpy(),
            tickers[indices],
        )
        result.insert(0, "fold_role", role_for_fold(fold.fold_id, roles))
        result.insert(0, "outer_fold", int(fold.fold_id))
        result["validation_rows"] = int(len(indices))
        result["validation_positives"] = int(np.sum(crash_target[indices] == 1))
        result["validation_positive_rate"] = float(np.mean(crash_target[indices] == 1))
        records.append(result)
        log(f"crash 동일 metric 상관 계산 완료: fold {fold.fold_id}")
    fold_map = pd.concat(records, ignore_index=True)
    selection_ids = {int(value) for value in roles.get("selection", [])}
    selection = fold_map[fold_map["outer_fold"].isin(selection_ids)]
    summary_records: list[dict[str, Any]] = []
    for feature, part in selection.groupby("feature", sort=False):
        record: dict[str, Any] = {"feature": feature}
        for metric in METRIC_COLUMNS:
            stats = finite_stats(part[metric].to_numpy(dtype=np.float64))
            for key, value in stats.items():
                record[f"crash_selection_{metric}_{key}"] = value
        summary_records.append(record)
    return fold_map, pd.DataFrame.from_records(summary_records)


def classify_direction_relation(
    surge_value: float,
    crash_value: float,
    directional_min_abs: float,
    crash_weak_abs: float,
) -> str:
    if not np.isfinite(surge_value) or abs(surge_value) < directional_min_abs:
        return "WEAK_OR_UNCLEAR"
    if not np.isfinite(crash_value):
        return "CRASH_REFERENCE_UNAVAILABLE"
    if abs(crash_value) < crash_weak_abs:
        return "SURGE_SPECIFIC"
    if np.sign(surge_value) != np.sign(crash_value):
        return "OPPOSITE_DIRECTION"
    if min(abs(surge_value), abs(crash_value)) >= directional_min_abs:
        return "COMMON_LARGE_MOVE"
    return "MIXED_OVERLAP"


def compute_metric_consistent_directional_map(
    features: Sequence[str],
    support: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    crash_summary: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    directional_min_abs: float,
    crash_weak_abs: float,
) -> pd.DataFrame:
    support_index = support.drop_duplicates("feature").set_index("feature")
    crash_index = crash_summary.drop_duplicates("feature").set_index("feature")
    selection_ids = {int(value) for value in roles.get("selection", [])}
    surge_selection = fold_correlations[fold_correlations["outer_fold"].isin(selection_ids)]
    surge_parts = {str(feature): part for feature, part in surge_selection.groupby("feature", sort=False)}
    records: list[dict[str, Any]] = []
    for feature in features:
        support_row = support_index.loc[feature]
        metric = metric_name_from_support(support_row)
        _, validation_column = METRIC_COLUMNS[metric]
        surge_values = surge_parts[feature][validation_column].to_numpy(dtype=np.float64)
        surge_stats = finite_stats(surge_values)
        crash_column = f"crash_selection_{metric}_mean"
        crash_value = float(crash_index.loc[feature, crash_column]) if feature in crash_index.index else np.nan
        surge_value = float(surge_stats["mean"])
        relation = classify_direction_relation(surge_value, crash_value, directional_min_abs, crash_weak_abs)
        same_direction = (
            bool(np.sign(surge_value) == np.sign(crash_value))
            if np.isfinite(surge_value) and np.isfinite(crash_value) and abs(surge_value) > 1e-12 and abs(crash_value) > 1e-12
            else pd.NA
        )
        overlap = (
            min(abs(surge_value), abs(crash_value))
            if same_direction is True
            else 0.0
        )
        specific = (
            max(0.0, abs(surge_value) - overlap)
            if np.isfinite(surge_value) and np.isfinite(crash_value)
            else np.nan
        )
        if same_direction is False and np.isfinite(surge_value):
            specific = abs(surge_value)
        records.append(
            {
                "feature": feature,
                "selected_metric": metric,
                "surge_selected_metric_selection_mean": surge_value,
                "surge_selected_metric_selection_abs_mean": float(surge_stats["mean_abs"]),
                "surge_selected_metric_selection_min_abs": float(surge_stats["min_abs"]),
                "surge_selected_metric_selection_sign_consistency": float(surge_stats["sign_consistency"]),
                "crash_same_metric_selection_mean": crash_value,
                "same_direction_same_metric": same_direction,
                "same_metric_abs_difference": abs(surge_value - crash_value) if np.isfinite(surge_value) and np.isfinite(crash_value) else np.nan,
                "same_metric_common_large_move_strength": overlap,
                "same_metric_surge_specific_strength": specific,
                "metric_consistent_target_relation_class": relation,
            }
        )
    frame = pd.DataFrame.from_records(records)
    stability = frame["surge_selected_metric_selection_sign_consistency"].fillna(0.0).clip(0.0, 1.0)
    specific_rank = frame["same_metric_surge_specific_strength"].fillna(0.0).rank(pct=True, method="average")
    contrast_rank = frame["same_metric_abs_difference"].fillna(0.0).rank(pct=True, method="average")
    frame["metric_consistent_directional_score"] = (
        0.50 * specific_rank + 0.30 * contrast_rank + 0.20 * stability
    ).clip(0.0, 1.0)
    order = frame.sort_values(
        ["metric_consistent_directional_score", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    rank = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    rank.loc[order] = np.arange(1, len(frame) + 1, dtype=np.int32)
    frame["metric_consistent_directional_rank"] = rank
    return frame


def univariate_rank_metrics(target: np.ndarray, oriented_scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    score = np.asarray(oriented_scores, dtype=np.float64)
    valid = np.isfinite(score) & np.isin(y, [0, 1])
    y = y[valid]
    score = score[valid]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    base_rate = float(positives / len(y)) if len(y) else np.nan
    if positives == 0 or negatives == 0:
        return {
            "oriented_roc_auc": np.nan,
            "oriented_average_precision": np.nan,
            "oriented_pr_lift": np.nan,
            "finite_rows": int(len(y)),
            "positive_rows": positives,
            "positive_rate": base_rate,
        }
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    return {
        "oriented_roc_auc": auc,
        "oriented_average_precision": ap,
        "oriented_pr_lift": float(ap / base_rate) if base_rate > 0 else np.nan,
        "finite_rows": int(len(y)),
        "positive_rows": positives,
        "positive_rate": base_rate,
    }


def compute_univariate_topk_map(
    data: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[Any],
    roles: Mapping[str, Sequence[int]],
    support: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    target_column: str,
    target_valid_column: str,
    date_column: str,
    top_fractions: Sequence[float],
) -> pd.DataFrame:
    support_index = support.drop_duplicates("feature").set_index("feature")
    fold_index = fold_correlations.set_index(["outer_fold", "feature"])
    target = pd.to_numeric(data[target_column], errors="coerce").to_numpy(dtype=np.float64)
    target_valid = parse_bool_series(data[target_valid_column]).to_numpy(dtype=bool)
    dates = pd.to_datetime(data[date_column], errors="coerce")
    records: list[dict[str, Any]] = []
    for fold in folds:
        mask = target_valid & dates.between(fold.validation_start, fold.validation_end).to_numpy()
        mask &= np.isfinite(target) & np.isin(target, [0.0, 1.0])
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"fold {fold.fold_id}: surge target validation 행이 없습니다")
        y = target[indices].astype(np.int8)
        validation_dates = dates.iloc[indices].to_numpy()
        values = data.iloc[indices][list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        for feature_index, feature in enumerate(features):
            support_row = support_index.loc[feature]
            metric = metric_name_from_support(support_row)
            train_column, _ = METRIC_COLUMNS[metric]
            fold_row = fold_index.loc[(fold.fold_id, feature)]
            train_metric = float(fold_row[train_column])
            direction = int(np.sign(train_metric)) if np.isfinite(train_metric) and abs(train_metric) > 1e-12 else 0
            oriented = values[:, feature_index] * direction if direction else np.full(len(indices), np.nan)
            record: dict[str, Any] = {
                "outer_fold": int(fold.fold_id),
                "fold_role": role_for_fold(fold.fold_id, roles),
                "feature": feature,
                "selected_metric": metric,
                "train_selected_metric": train_metric,
                "train_direction": direction,
            }
            record.update(univariate_rank_metrics(y, oriented))
            for fraction in top_fractions:
                label = fraction_label(fraction)
                pooled = pooled_top_fraction_metrics(y, oriented, fraction)
                daily = daily_top_fraction_metrics(y, oriented, validation_dates, fraction)
                for key, value in pooled.items():
                    if key != "fraction":
                        record[f"pooled_top_{label}_{key}"] = value
                for key, value in daily.items():
                    if key != "fraction":
                        record[f"daily_top_{label}_{key}"] = value
            records.append(record)
        log(f"univariate train-fixed top-k 완료: fold {fold.fold_id}")
    return pd.DataFrame.from_records(records)


def summarize_univariate_map(
    fold_map: pd.DataFrame,
    features: Sequence[str],
    roles: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in fold_map.columns
        if column not in {"outer_fold", "fold_role", "feature", "selected_metric"}
        and pd.api.types.is_numeric_dtype(fold_map[column])
        and column not in {"train_direction"}
    ]
    records: list[dict[str, Any]] = []
    grouped = {str(feature): part for feature, part in fold_map.groupby("feature", sort=False)}
    for feature in features:
        part = grouped[feature]
        record: dict[str, Any] = {
            "feature": feature,
            "selected_metric": str(part["selected_metric"].iloc[0]),
        }
        for role in ("selection", "confirmation", "recent_audit"):
            role_part = part[part["outer_fold"].isin({int(value) for value in roles.get(role, [])})]
            for column in metric_columns:
                values = pd.to_numeric(role_part[column], errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                record[f"{role}_{column}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
                record[f"{role}_{column}_median"] = float(np.median(finite)) if len(finite) else np.nan
                record[f"{role}_{column}_min"] = float(np.min(finite)) if len(finite) else np.nan
                record[f"{role}_{column}_max"] = float(np.max(finite)) if len(finite) else np.nan
                record[f"{role}_{column}_folds"] = int(len(finite))
        selection_directions = part[
            part["outer_fold"].isin({int(value) for value in roles.get("selection", [])})
        ]["train_direction"].to_numpy(dtype=np.float64)
        record["selection_train_direction_sign_consistency"] = sign_consistency(selection_directions)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_corrected_profiles(
    features: Sequence[str],
    summary: pd.DataFrame,
    support: pd.DataFrame,
    old_membership: pd.DataFrame,
    alignment: pd.DataFrame,
    directional: pd.DataFrame,
    univariate_summary: pd.DataFrame,
    minimum_selection_fold_abs: float,
    minimum_train_fold_abs: float,
    univariate_min_top3_lift: float,
    univariate_min_auc: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = pd.DataFrame({"feature": list(features)})
    merged = base.merge(summary, on="feature", how="left", validate="one_to_one", suffixes=("", "__summary"))
    merged = merged.merge(support, on="feature", how="left", validate="one_to_one", suffixes=("", "__support"))
    merged = merged.merge(alignment, on="feature", how="left", validate="one_to_one")
    merged = merged.merge(directional, on="feature", how="left", validate="one_to_one")
    merged = merged.merge(univariate_summary, on="feature", how="left", validate="one_to_one", suffixes=("", "__univariate"))

    metric_columns = [
        column
        for column in ["selected_metric", "selected_metric_x", "selected_metric_y", "selected_metric__univariate"]
        if column in merged.columns
    ]
    if metric_columns:
        canonical_metric = merged[metric_columns[0]].astype("string")
        for column in metric_columns[1:]:
            candidate = merged[column].astype("string")
            mismatch = canonical_metric.notna() & candidate.notna() & canonical_metric.ne(candidate)
            if mismatch.any():
                examples = merged.loc[mismatch, ["feature", metric_columns[0], column]].head(10).to_dict("records")
                raise RuntimeError(f"selected metric 병합 불일치: {examples}")
            canonical_metric = canonical_metric.fillna(candidate)
        merged["selected_metric"] = canonical_metric
        drop_metric_columns = [column for column in metric_columns if column != "selected_metric"]
        if drop_metric_columns:
            merged.drop(columns=drop_metric_columns, inplace=True)

    representative_column = "is_primary_representative"
    if representative_column not in merged.columns:
        if "is_representative" in merged.columns:
            representative_column = "is_representative"
        else:
            merged[representative_column] = False
    strict = parse_bool_series(merged["strict_stable_supported"])
    selection_supported = parse_bool_series(
        merged.get("selection_supported", pd.Series(False, index=merged.index))
    )
    aligned = parse_bool_series(merged["train_validation_aligned"])
    representative = parse_bool_series(merged[representative_column])
    all_selection_sign = parse_bool_series(merged.get("selection_all_fold_sign_match", pd.Series(False, index=merged.index)))
    validation_min_abs = pd.to_numeric(merged.get("selection_min_fold_abs_corr"), errors="coerce")
    train_min_abs = pd.to_numeric(merged.get("selection_train_selected_metric_min_abs"), errors="coerce")
    direction_class = merged["metric_consistent_target_relation_class"].astype("string")

    top3_lift_column = "selection_daily_top_3pct_lift_mean"
    if top3_lift_column not in merged.columns:
        top3_lift_column = next(
            (column for column in merged.columns if column.startswith("selection_daily_top_3pct_lift") and column.endswith("_mean")),
            "",
        )
    auc_column = "selection_oriented_roc_auc_mean"
    top3_lift = pd.to_numeric(merged[top3_lift_column], errors="coerce") if top3_lift_column else pd.Series(np.nan, index=merged.index)
    univariate_auc = pd.to_numeric(merged.get(auc_column), errors="coerce")

    # S profiles use only selection-era evidence. They are the correct inputs for model selection.
    p2s = selection_supported & aligned
    p3s = p2s & representative
    p3s_strict = (
        p3s
        & all_selection_sign
        & validation_min_abs.ge(minimum_selection_fold_abs)
        & train_min_abs.ge(minimum_train_fold_abs)
    )
    p4s = p2s & direction_class.isin(DIRECTIONAL_CLASSES)
    p5s = p2s & direction_class.eq("COMMON_LARGE_MOVE")
    p6s = p3s_strict | p4s
    p7s = p3s & top3_lift.ge(univariate_min_top3_lift) & univariate_auc.ge(univariate_min_auc)

    # A profiles additionally require the old strict gate, which already looked at confirmation/recent.
    # They are retained as holdout-confirmed diagnostics and must not replace selection-only profiles.
    p2a = strict & aligned
    p3a = p2a & representative
    p3b = (
        p3a
        & all_selection_sign
        & validation_min_abs.ge(minimum_selection_fold_abs)
        & train_min_abs.ge(minimum_train_fold_abs)
    )
    p4a = p2a & direction_class.isin(DIRECTIONAL_CLASSES)
    p5a = p2a & direction_class.eq("COMMON_LARGE_MOVE")
    p6a = p3b | p4a
    p7a = p3a & top3_lift.ge(univariate_min_top3_lift) & univariate_auc.ge(univariate_min_auc)

    old_profile_columns = [
        column for column in old_membership.columns
        if column == "feature" or column.startswith("P")
    ]
    membership = old_membership[old_profile_columns].drop_duplicates("feature").copy()
    membership = base.merge(membership, on="feature", how="left", validate="one_to_one")
    for column in membership.columns:
        if column != "feature":
            membership[column] = parse_bool_series(membership[column])
    membership["P2S_SELECTION_TRAIN_ALIGNED"] = p2s.to_numpy()
    membership["P3S_SELECTION_CLUSTER_REP"] = p3s.to_numpy()
    membership["P3S_ALLFOLD_MINABS_CLUSTER_REP"] = p3s_strict.to_numpy()
    membership["P4S_METRIC_CONSISTENT_DIRECTIONAL"] = p4s.to_numpy()
    membership["P5S_COMMON_LARGE_MOVE"] = p5s.to_numpy()
    membership["P6S_BALANCED_SELECTION_ONLY"] = p6s.to_numpy()
    membership["P7S_UNIVARIATE_TOP3_LIFT"] = p7s.to_numpy()

    membership["P2A_TRAIN_ALIGNED_STRICT"] = p2a.to_numpy()
    membership["P3A_TRAIN_ALIGNED_CLUSTER_REP"] = p3a.to_numpy()
    membership["P3B_ALLFOLD_MINABS_CLUSTER_REP"] = p3b.to_numpy()
    membership["P4A_METRIC_CONSISTENT_DIRECTIONAL"] = p4a.to_numpy()
    membership["P5A_ALIGNED_COMMON_LARGE_MOVE"] = p5a.to_numpy()
    membership["P6A_BALANCED_CORRECTED"] = p6a.to_numpy()
    membership["P7A_UNIVARIATE_TOP3_LIFT"] = p7a.to_numpy()

    priority = pd.to_numeric(merged.get("surge_priority_rank"), errors="coerce")
    directional_rank = pd.to_numeric(merged.get("metric_consistent_directional_rank"), errors="coerce")
    lift_rank = top3_lift.rank(method="min", ascending=False)
    profiles: dict[str, Any] = {
        "schema": "crashwatch_surge_corrected_profiles_v3",
        "warning": (
            "피처 확정 결과가 아니라 이탈 실험 입력 프로필입니다. "
            "P*S는 selection-only이며, P*A는 confirmation/recent까지 본 holdout-confirmed 진단 프로필입니다."
        ),
        "criteria": {
            "selection_only_profile_prefix": "P*S",
            "holdout_confirmed_diagnostic_prefix": "P*A",
            "train_validation_alignment_required": True,
            "minimum_selection_fold_abs": minimum_selection_fold_abs,
            "minimum_train_fold_abs": minimum_train_fold_abs,
            "univariate_min_top3_lift": univariate_min_top3_lift,
            "univariate_min_auc": univariate_min_auc,
            "directional_class_uses_same_metric_for_surge_and_crash": True,
        },
        "profiles": {},
    }
    for profile in [column for column in membership.columns if column != "feature"]:
        selected_features = membership.loc[membership[profile], "feature"].astype(str).tolist()
        if profile in {"P4S_METRIC_CONSISTENT_DIRECTIONAL", "P4A_METRIC_CONSISTENT_DIRECTIONAL"}:
            rank_map = dict(zip(merged["feature"], directional_rank))
        elif profile in {"P7S_UNIVARIATE_TOP3_LIFT", "P7A_UNIVARIATE_TOP3_LIFT"}:
            rank_map = dict(zip(merged["feature"], lift_rank))
        else:
            rank_map = dict(zip(merged["feature"], priority))
        selected_features.sort(key=lambda feature: (rank_map.get(feature, math.inf), feature))
        profiles["profiles"][profile] = {
            "feature_count": len(selected_features),
            "features": selected_features,
        }
    return membership, merged, profiles


def verify_source_hashes(
    dataset: Path,
    target_sidecar: Path,
    correlation_manifest: Mapping[str, Any],
    allow_mismatch: bool,
) -> dict[str, Any]:
    dataset_hash = sha256_file(dataset)
    target_hash = sha256_file(target_sidecar)
    expected_dataset = correlation_manifest.get("source_dataset_sha256")
    expected_target = correlation_manifest.get("target_sha256")
    dataset_match = expected_dataset in (None, dataset_hash)
    target_match = expected_target in (None, target_hash)
    if not allow_mismatch and (not dataset_match or not target_match):
        raise ValueError(
            "상관지도와 현재 입력 hash가 다릅니다. "
            f"dataset_match={dataset_match}, target_match={target_match}"
        )
    return {
        "dataset_sha256": dataset_hash,
        "target_sha256": target_hash,
        "correlation_expected_dataset_sha256": expected_dataset,
        "correlation_expected_target_sha256": expected_target,
        "dataset_hash_match": dataset_match,
        "target_hash_match": target_match,
    }


def build_gate_config_identity(
    args: argparse.Namespace,
    hashes: Mapping[str, Any],
    features: Sequence[str],
    folds: Sequence[Any],
    roles: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    return {
        "schema": "crashwatch_surge_pre_model_gate_config_v3",
        "dataset_sha256": hashes["dataset_sha256"],
        "target_sha256": hashes["target_sha256"],
        "correlation_manifest_sha256": sha256_file(args.correlation_dir / "surge_correlation_manifest.json"),
        "folds_sha256": sha256_file(args.folds),
        "feature_hash": hash_strings(features),
        "feature_count": len(features),
        "folds": [fold.to_dict() for fold in folds],
        "fold_roles": {str(key): [int(value) for value in values] for key, values in roles.items()},
        "target_column": args.target_column,
        "target_valid_column": args.target_valid_column,
        "crash_target_column": args.crash_target_column,
        "date_column": args.date_column,
        "ticker_column": args.ticker_column,
        "minimum_purge_trading_days": args.minimum_purge_trading_days,
        "train_validation_min_match_ratio": args.train_validation_min_match_ratio,
        "train_validation_min_train_abs_mean": args.train_validation_min_train_abs_mean,
        "directional_min_abs": args.directional_min_abs,
        "directional_crash_weak_abs": args.directional_crash_weak_abs,
        "top_fractions": sorted(set(parse_float_list(args.top_fractions))),
        "profile_min_selection_fold_abs": args.profile_min_selection_fold_abs,
        "profile_min_train_fold_abs": args.profile_min_train_fold_abs,
        "univariate_min_top3_lift": args.univariate_min_top3_lift,
        "univariate_min_auc": args.univariate_min_auc,
        "code_sha256": {
            "run_surge_pre_model_gate.py": sha256_file(Path(__file__)),
            "surge_ablation_common.py": sha256_file(Path(__file__).with_name("surge_ablation_common.py")),
        },
    }


def validated_gate_resume_available(
    args: argparse.Namespace,
    config_hash: str,
) -> tuple[bool, list[str]]:
    manifest_path = args.output / "PRE_MODEL_GATE_MANIFEST.json"
    status_path = args.output / "RUN_STATUS.json"
    if not manifest_path.exists() or not status_path.exists():
        return False, ["manifest_or_status_missing"]
    try:
        manifest = load_json(manifest_path)
        status = load_json(status_path)
    except Exception as exc:
        return False, [f"json_read_error:{type(exc).__name__}"]
    reasons: list[str] = []
    if status.get("status") != "SUCCESS":
        reasons.append(f"run_status:{status.get('status')}")
    if manifest.get("status") != "SUCCESS":
        reasons.append(f"manifest_status:{manifest.get('status')}")
    if manifest.get("config_hash") != config_hash:
        reasons.append("config_hash_mismatch")
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, list) or not inventory:
        reasons.append("output_inventory_missing")
    else:
        valid, inventory_reasons = verify_output_inventory(args.output, inventory)
        if not valid:
            reasons.extend(inventory_reasons)
    return not reasons, reasons


def run(args: argparse.Namespace) -> None:
    package_root = args.package_root.resolve()
    resolve_default_paths(package_root, args)
    args.dataset = args.dataset.resolve()
    args.target_sidecar = args.target_sidecar.resolve()
    args.folds = args.folds.resolve()
    args.correlation_dir = args.correlation_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    required = {
        "dataset": args.dataset,
        "target sidecar": args.target_sidecar,
        "folds": args.folds,
        "correlation manifest": args.correlation_dir / "surge_correlation_manifest.json",
    }
    require_files(required)

    with FileLock(args.output / ".pre_model_gate.lock"):
        correlation_manifest = load_json(args.correlation_dir / "surge_correlation_manifest.json")
        roles = load_roles_from_manifest(args.correlation_dir / "surge_correlation_manifest.json")
        folds = load_folds(args.folds)
        validate_role_assignments(roles, folds, require_selection=True)
        hashes = verify_source_hashes(args.dataset, args.target_sidecar, correlation_manifest, args.allow_hash_mismatch)
        features, summary, fold_correlations, support, old_membership = load_feature_universe(args.correlation_dir)
        config_identity = build_gate_config_identity(args, hashes, features, folds, roles)
        config_hash = sha256_bytes(stable_json_bytes(config_identity))
        if args.resume:
            reusable, reasons = validated_gate_resume_available(args, config_hash)
            if reusable:
                log("검증된 pre-model gate 재사용")
                return
            log(f"pre-model gate 재계산: {reasons}")

        status = RunStatus(args.output, "surge_pre_model_gate_v3")
        try:
            started = time.monotonic()
            status.stage("input_integrity", "running")
            dataset_columns = table_columns(args.dataset)
            verify_feature_names(features, dataset_columns)
            needed_columns = list(dict.fromkeys([
                "source_row_id",
                args.date_column,
                args.ticker_column,
                args.crash_target_column,
                "sealed_do_not_train_or_tune",
                *features,
            ]))
            needed_columns = [column for column in needed_columns if column in dataset_columns]
            source = read_table(args.dataset, columns=needed_columns)
            target_columns = table_columns(args.target_sidecar)
            side_columns = [column for column in ["source_row_id", args.target_column, args.target_valid_column, args.date_column, args.ticker_column] if column in target_columns]
            target_sidecar = read_table(args.target_sidecar, columns=side_columns)
            data = join_source_and_target(
                source,
                target_sidecar,
                target_column=args.target_column,
                target_valid_column=args.target_valid_column,
                date_column=args.date_column,
                ticker_column=args.ticker_column,
            )
            data[args.date_column] = pd.to_datetime(data[args.date_column], errors="coerce")
            if data[args.date_column].isna().any():
                raise ValueError(f"date 파싱 실패: {int(data[args.date_column].isna().sum())}행")
            if "sealed_do_not_train_or_tune" in data.columns:
                sealed = pd.to_numeric(data["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
                if sealed.ne(0).any():
                    raise ValueError(f"sealed_do_not_train_or_tune 비영 행 발견: {int(sealed.ne(0).sum())}")
            validate_folds(folds, minimum_purge_trading_days=args.minimum_purge_trading_days, all_dates=data[args.date_column])
            status.stage(
                "input_integrity",
                "complete",
                row_count=len(data),
                feature_count=len(features),
                **hashes,
            )

            status.stage("train_validation_alignment", "running")
            alignment = compute_train_validation_alignment(
                features,
                support,
                fold_correlations,
                roles.get("selection", []),
                minimum_match_ratio=args.train_validation_min_match_ratio,
                minimum_train_abs_mean=args.train_validation_min_train_abs_mean,
            )
            atomic_write_csv(args.output / "surge_train_validation_direction_alignment.csv", alignment)
            status.stage(
                "train_validation_alignment",
                "complete",
                aligned_count=int(alignment["train_validation_aligned"].sum()),
            )

            status.stage("metric_consistent_directional", "running")
            crash_fold_map, crash_summary = compute_crash_metric_consistent_map(
                data,
                features,
                folds,
                roles,
                crash_target_column=args.crash_target_column,
                date_column=args.date_column,
                ticker_column=args.ticker_column,
            )
            directional = compute_metric_consistent_directional_map(
                features,
                support,
                fold_correlations,
                crash_summary,
                roles,
                directional_min_abs=args.directional_min_abs,
                crash_weak_abs=args.directional_crash_weak_abs,
            )
            atomic_write_csv(args.output / "crash_feature_target_correlation_by_fold_metric_consistent.csv", crash_fold_map)
            atomic_write_csv(args.output / "crash_feature_selection_metric_summary.csv", crash_summary)
            atomic_write_csv(args.output / "surge_metric_consistent_directional_map.csv", directional)
            status.stage(
                "metric_consistent_directional",
                "complete",
                class_counts={str(key): int(value) for key, value in directional["metric_consistent_target_relation_class"].value_counts().items()},
            )

            status.stage("univariate_topk", "running")
            top_fractions = sorted(set(parse_float_list(args.top_fractions)))
            if not top_fractions or any(value <= 0 or value > 0.5 for value in top_fractions):
                raise ValueError(f"top fractions는 (0, 0.5] 범위여야 합니다: {top_fractions}")
            univariate_fold = compute_univariate_topk_map(
                data,
                features,
                folds,
                roles,
                support,
                fold_correlations,
                target_column=args.target_column,
                target_valid_column=args.target_valid_column,
                date_column=args.date_column,
                top_fractions=top_fractions,
            )
            univariate_summary = summarize_univariate_map(univariate_fold, features, roles)
            expected_rows = len(features) * len(folds)
            if len(univariate_fold) != expected_rows:
                raise RuntimeError(f"univariate row 수 불일치: {len(univariate_fold)} != {expected_rows}")
            atomic_write_csv(args.output / "surge_feature_univariate_by_fold.csv", univariate_fold)
            atomic_write_csv(args.output / "surge_feature_univariate_summary.csv", univariate_summary)
            status.stage("univariate_topk", "complete", row_count=len(univariate_fold))

            status.stage("corrected_profiles", "running")
            membership, merged_audit, profile_payload = build_corrected_profiles(
                features,
                summary,
                support,
                old_membership,
                alignment,
                directional,
                univariate_summary,
                minimum_selection_fold_abs=args.profile_min_selection_fold_abs,
                minimum_train_fold_abs=args.profile_min_train_fold_abs,
                univariate_min_top3_lift=args.univariate_min_top3_lift,
                univariate_min_auc=args.univariate_min_auc,
            )
            atomic_write_csv(args.output / "surge_feature_profile_membership_corrected.csv", membership)
            atomic_write_csv(args.output / "surge_pre_model_feature_audit.csv", merged_audit)
            atomic_write_json(args.output / "surge_feature_profiles_corrected.json", profile_payload)

            profile_counts = {
                profile: int(membership[profile].fillna(False).sum())
                for profile in membership.columns
                if profile != "feature"
            }
            required_profiles = [
                "P2S_SELECTION_TRAIN_ALIGNED",
                "P3S_SELECTION_CLUSTER_REP",
                "P3S_ALLFOLD_MINABS_CLUSTER_REP",
                "P4S_METRIC_CONSISTENT_DIRECTIONAL",
                "P2A_TRAIN_ALIGNED_STRICT",
                "P3A_TRAIN_ALIGNED_CLUSTER_REP",
                "P3B_ALLFOLD_MINABS_CLUSTER_REP",
                "P4A_METRIC_CONSISTENT_DIRECTIONAL",
            ]
            for profile in required_profiles:
                if profile not in membership.columns:
                    raise RuntimeError(f"필수 corrected profile 생성 실패: {profile}")
            status.stage("corrected_profiles", "complete", profile_counts=profile_counts)

            target_valid_mask = parse_bool_series(data[args.target_valid_column])
            manifest = {
                "schema": "crashwatch_surge_pre_model_gate_v3",
                "status": "SUCCESS",
                "created_at": utc_now(),
                "config_hash": config_hash,
                "config_identity": config_identity,
                "elapsed_seconds": time.monotonic() - started,
                "package_root": str(package_root),
                "dataset_path": str(args.dataset),
                "target_sidecar_path": str(args.target_sidecar),
                "correlation_dir": str(args.correlation_dir),
                "folds_path": str(args.folds),
                "dataset_sha256": hashes["dataset_sha256"],
                "target_sha256": hashes["target_sha256"],
                "correlation_manifest_sha256": sha256_file(args.correlation_dir / "surge_correlation_manifest.json"),
                "feature_hash": hash_strings(features),
                "feature_count": len(features),
                "row_count": len(data),
                "target_valid_rows": int(target_valid_mask.sum()),
                "target_positive_rows": int(pd.to_numeric(data.loc[target_valid_mask, args.target_column], errors="coerce").fillna(0).sum()),
                "fold_roles": roles,
                "folds": [fold.to_dict() for fold in folds],
                "train_validation_alignment": {
                    "minimum_match_ratio": args.train_validation_min_match_ratio,
                    "minimum_train_abs_mean": args.train_validation_min_train_abs_mean,
                    "aligned_feature_count": int(alignment["train_validation_aligned"].sum()),
                },
                "directional": {
                    "crash_target_column": args.crash_target_column,
                    "minimum_surge_abs": args.directional_min_abs,
                    "crash_weak_abs": args.directional_crash_weak_abs,
                    "same_metric_enforced": True,
                    "class_counts": {str(key): int(value) for key, value in directional["metric_consistent_target_relation_class"].value_counts().items()},
                },
                "univariate": {
                    "direction_fixed_from_each_fold_train_selected_metric": True,
                    "top_fractions": top_fractions,
                    "fold_rows": len(univariate_fold),
                },
                "corrected_profile_counts": profile_counts,
                "model_versions": model_versions(),
            }
            atomic_write_json(args.output / "PRE_MODEL_GATE_MANIFEST.json", manifest)
            inventory = compute_output_inventory(
                args.output,
                names=[
                    "surge_train_validation_direction_alignment.csv",
                    "crash_feature_target_correlation_by_fold_metric_consistent.csv",
                    "crash_feature_selection_metric_summary.csv",
                    "surge_metric_consistent_directional_map.csv",
                    "surge_feature_univariate_by_fold.csv",
                    "surge_feature_univariate_summary.csv",
                    "surge_feature_profile_membership_corrected.csv",
                    "surge_pre_model_feature_audit.csv",
                    "surge_feature_profiles_corrected.json",
                ],
            )
            manifest["output_inventory"] = inventory
            atomic_write_json(args.output / "PRE_MODEL_GATE_MANIFEST.json", manifest)
            status.success(
                feature_count=len(features),
                aligned_feature_count=int(alignment["train_validation_aligned"].sum()),
                profile_counts=profile_counts,
            )
            log("Pre-model gate 완료")
            log(json.dumps(profile_counts, ensure_ascii=False, indent=2))
        except BaseException as exc:
            status.failure(exc)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrashWatch Surge 모델 실행 전 필수 gate: train-validation 방향 정렬, "
            "동일 metric 급등/급락 방향성 분류, train-fixed univariate top-k lift를 생성합니다."
        )
    )
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--correlation-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-column", default="label_abs_surge_3d_5pct")
    parser.add_argument("--target-valid-column", default="target_valid")
    parser.add_argument("--crash-target-column", default="label_abs_crash_20")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--ticker-column", default="ticker")
    parser.add_argument("--minimum-purge-trading-days", type=int, default=3)
    parser.add_argument("--train-validation-min-match-ratio", type=float, default=0.80)
    parser.add_argument("--train-validation-min-train-abs-mean", type=float, default=0.0)
    parser.add_argument("--directional-min-abs", type=float, default=0.03)
    parser.add_argument("--directional-crash-weak-abs", type=float, default=0.015)
    parser.add_argument("--top-fractions", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--profile-min-selection-fold-abs", type=float, default=0.01)
    parser.add_argument("--profile-min-train-fold-abs", type=float, default=0.005)
    parser.add_argument("--univariate-min-top3-lift", type=float, default=1.10)
    parser.add_argument("--univariate-min-auc", type=float, default=0.52)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
