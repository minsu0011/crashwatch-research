from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_csv, atomic_json, read_json

LOGGER = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "task_id", "family", "backend", "phase", "profile", "condition", "outer_fold", "seed",
    "scope_type", "scope_value", "feature_count", "feature_hash", "best_iteration", "elapsed_seconds",
    "train_rows", "validation_rows", "train_date_min", "train_date_max", "validation_date_min", "validation_date_max",
    "rows", "positives", "positive_rate", "raw_pr_auc", "raw_roc_auc", "raw_brier", "raw_logloss",
    "raw_mean_prediction", "balanced_accuracy", "raw_pr_auc_lift", "top_1pct_precision", "top_1pct_recall",
    "top_3pct_precision", "top_3pct_recall", "top_5pct_precision", "top_5pct_recall", "prediction_path"
]


TASK_COLUMNS = [
    "task_id", "status", "metric_path", "prediction_path", "elapsed_seconds", "error",
    "family", "backend", "profile", "condition", "seed", "outer_fold"
]

DELTA_COLUMNS = [
    "family", "backend", "profile", "condition", "scope_type", "scope_value", "outer_fold", "seed",
    "baseline_task_id", "ablated_task_id", "raw_pr_auc_loss", "raw_roc_auc_loss", "raw_brier_loss",
    "raw_logloss_loss", "top_3pct_precision_loss", "top_3pct_recall_loss", "balanced_accuracy_loss",
    "baseline_positive_rate", "baseline_raw_pr_auc", "ablated_raw_pr_auc"
]


def _collect_metric_jsons(cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    base = cache_root / "task_results"
    if not base.exists():
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame(columns=TASK_COLUMNS)
    for metric_path in sorted(base.glob("*/metrics/*.json")):
        try:
            payload = read_json(metric_path)
        except Exception as exc:
            task_rows.append({"metric_path": str(metric_path), "status": "corrupt", "error": repr(exc)})
            continue
        task_row = {
            "task_id": payload.get("task_id"),
            "status": payload.get("status"),
            "metric_path": str(metric_path),
            "prediction_path": payload.get("prediction_path", ""),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "error": payload.get("error", ""),
        }
        identity = payload.get("identity", {})
        for key in ["family", "backend", "profile", "condition", "seed"]:
            task_row[key] = identity.get(key)
        fold = identity.get("fold", {})
        task_row["outer_fold"] = fold.get("fold_id")
        task_rows.append(task_row)
        if payload.get("status") == "completed":
            for row in payload.get("metrics", []):
                metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    for col in METRIC_COLUMNS:
        if col not in metrics.columns:
            metrics[col] = pd.Series(dtype="object")
    metrics = metrics[METRIC_COLUMNS]
    tasks = pd.DataFrame(task_rows)
    for col in TASK_COLUMNS:
        if col not in tasks.columns:
            tasks[col] = pd.Series(dtype="object")
    return metrics, tasks[TASK_COLUMNS]


def _paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=DELTA_COLUMNS)
    key = ["family", "backend", "profile", "outer_fold", "seed", "scope_type", "scope_value"]
    baseline = metrics[metrics["condition"].eq("B0")].copy()
    baseline = baseline.rename(columns={
        "task_id": "baseline_task_id",
        "raw_pr_auc": "baseline_raw_pr_auc",
        "raw_roc_auc": "baseline_raw_roc_auc",
        "raw_brier": "baseline_raw_brier",
        "raw_logloss": "baseline_raw_logloss",
        "top_3pct_precision": "baseline_top_3pct_precision",
        "top_3pct_recall": "baseline_top_3pct_recall",
        "balanced_accuracy": "baseline_balanced_accuracy",
        "positive_rate": "baseline_positive_rate",
    })
    ablated = metrics[~metrics["condition"].eq("B0")].copy()
    ablated = ablated.rename(columns={
        "task_id": "ablated_task_id",
        "raw_pr_auc": "ablated_raw_pr_auc",
        "raw_roc_auc": "ablated_raw_roc_auc",
        "raw_brier": "ablated_raw_brier",
        "raw_logloss": "ablated_raw_logloss",
        "top_3pct_precision": "ablated_top_3pct_precision",
        "top_3pct_recall": "ablated_top_3pct_recall",
        "balanced_accuracy": "ablated_balanced_accuracy",
    })
    keep_base = key + [
        "baseline_task_id", "baseline_raw_pr_auc", "baseline_raw_roc_auc", "baseline_raw_brier",
        "baseline_raw_logloss", "baseline_top_3pct_precision", "baseline_top_3pct_recall",
        "baseline_balanced_accuracy", "baseline_positive_rate"
    ]
    keep_ab = key + [
        "condition", "ablated_task_id", "ablated_raw_pr_auc", "ablated_raw_roc_auc", "ablated_raw_brier",
        "ablated_raw_logloss", "ablated_top_3pct_precision", "ablated_top_3pct_recall", "ablated_balanced_accuracy"
    ]
    merged = ablated[keep_ab].merge(baseline[keep_base], on=key, how="inner", validate="many_to_one")
    merged["raw_pr_auc_loss"] = merged["baseline_raw_pr_auc"] - merged["ablated_raw_pr_auc"]
    merged["raw_roc_auc_loss"] = merged["baseline_raw_roc_auc"] - merged["ablated_raw_roc_auc"]
    merged["raw_brier_loss"] = merged["ablated_raw_brier"] - merged["baseline_raw_brier"]
    merged["raw_logloss_loss"] = merged["ablated_raw_logloss"] - merged["baseline_raw_logloss"]
    merged["top_3pct_precision_loss"] = merged["baseline_top_3pct_precision"] - merged["ablated_top_3pct_precision"]
    merged["top_3pct_recall_loss"] = merged["baseline_top_3pct_recall"] - merged["ablated_top_3pct_recall"]
    merged["balanced_accuracy_loss"] = merged["baseline_balanced_accuracy"] - merged["ablated_balanced_accuracy"]
    for col in DELTA_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[DELTA_COLUMNS]


def _exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    if n <= 16:
        count = 0
        total = 2**n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            if statistic >= observed - 1e-15:
                count += 1
        return count / total
    rng = np.random.default_rng(17)
    signs = rng.choice([-1.0, 1.0], size=(20000, n))
    permuted = np.abs(np.mean(signs * values, axis=1))
    return float((np.sum(permuted >= observed) + 1) / (len(permuted) + 1))


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return pd.Series(q, index=p_values.index)
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        running = min(running, p[idx] * m / rank)
        adjusted[rank - 1] = running
    for rank, idx in enumerate(order):
        q[idx] = min(1.0, adjusted[rank])
    return pd.Series(q, index=p_values.index)


def _summary(deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if deltas.empty:
        return pd.DataFrame(), pd.DataFrame()
    fold_level = deltas.groupby(
        ["family", "backend", "profile", "condition", "scope_type", "scope_value", "outer_fold"],
        dropna=False,
        as_index=False,
    ).agg(
        raw_pr_auc_loss=("raw_pr_auc_loss", "mean"),
        raw_roc_auc_loss=("raw_roc_auc_loss", "mean"),
        raw_brier_loss=("raw_brier_loss", "mean"),
        raw_logloss_loss=("raw_logloss_loss", "mean"),
        top_3pct_precision_loss=("top_3pct_precision_loss", "mean"),
        top_3pct_recall_loss=("top_3pct_recall_loss", "mean"),
        balanced_accuracy_loss=("balanced_accuracy_loss", "mean"),
        seed_count=("seed", "nunique"),
    )
    rows: list[dict[str, Any]] = []
    group_cols = ["family", "backend", "profile", "condition", "scope_type", "scope_value"]
    for keys, frame in fold_level.groupby(group_cols, dropna=False):
        values = frame["raw_pr_auc_loss"].to_numpy(dtype=float)
        early = frame[frame["outer_fold"] <= 3]["raw_pr_auc_loss"].to_numpy(dtype=float)
        recent = frame[frame["outer_fold"] >= 4]["raw_pr_auc_loss"].to_numpy(dtype=float)
        ranks = frame["outer_fold"].rank().to_numpy(dtype=float)
        value_ranks = frame["raw_pr_auc_loss"].rank().to_numpy(dtype=float)
        spearman = float(np.corrcoef(ranks, value_ranks)[0, 1]) if len(frame) >= 3 and np.std(value_ranks) > 0 else float("nan")
        row = dict(zip(group_cols, keys))
        row.update({
            "fold_count": int(frame["outer_fold"].nunique()),
            "seed_count_min": int(frame["seed_count"].min()),
            "raw_pr_auc_loss_mean": float(np.nanmean(values)),
            "raw_pr_auc_loss_median": float(np.nanmedian(values)),
            "raw_pr_auc_loss_std": float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else float("nan"),
            "raw_pr_auc_loss_q25": float(np.nanquantile(values, 0.25)),
            "raw_pr_auc_loss_q75": float(np.nanquantile(values, 0.75)),
            "raw_pr_auc_loss_worst_fold": float(np.nanmin(values)),
            "positive_fold_ratio": float(np.mean(values > 0)),
            "recent_raw_pr_auc_loss_mean": float(np.nanmean(recent)) if len(recent) else float("nan"),
            "recent_positive_fold_ratio": float(np.mean(recent > 0)) if len(recent) else float("nan"),
            "early_raw_pr_auc_loss_mean": float(np.nanmean(early)) if len(early) else float("nan"),
            "recent_minus_early": float(np.nanmean(recent) - np.nanmean(early)) if len(recent) and len(early) else float("nan"),
            "fold_effect_spearman": spearman,
            "raw_roc_auc_loss_mean": float(frame["raw_roc_auc_loss"].mean()),
            "raw_brier_loss_mean": float(frame["raw_brier_loss"].mean()),
            "raw_logloss_loss_mean": float(frame["raw_logloss_loss"].mean()),
            "top_3pct_precision_loss_mean": float(frame["top_3pct_precision_loss"].mean()),
            "top_3pct_recall_loss_mean": float(frame["top_3pct_recall_loss"].mean()),
            "balanced_accuracy_loss_mean": float(frame["balanced_accuracy_loss"].mean()),
            "sign_flip_p": _exact_sign_flip_p(values),
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q"] = summary.groupby(["family", "scope_type"])["sign_flip_p"].transform(_bh_qvalues)
    return fold_level, summary


def _synergy(fold_level: pd.DataFrame) -> pd.DataFrame:
    if fold_level.empty:
        return pd.DataFrame()
    base = fold_level[fold_level["scope_type"].eq("all")].copy()
    piv = base.pivot_table(
        index=["family", "backend", "profile", "outer_fold"],
        columns="condition",
        values="raw_pr_auc_loss",
        aggfunc="first",
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for row in piv.itertuples(index=False):
        data = row._asdict()
        common = {"family": data["family"], "backend": data["backend"], "profile": data["profile"], "outer_fold": data["outer_fold"]}
        if all(name in data and pd.notna(data[name]) for name in ["A1", "A2", "A3"]):
            rows.append({**common, "contrast": "A3_over_best_A1_A2", "incremental_loss": float(data["A3"] - max(data["A1"], data["A2"]))})
        if all(name in data and pd.notna(data[name]) for name in ["A2", "A5"]):
            rows.append({**common, "contrast": "A5_minus_A2", "incremental_loss": float(data["A5"] - data["A2"])})
        if all(name in data and pd.notna(data[name]) for name in ["A3", "A6"]):
            rows.append({**common, "contrast": "A6_minus_A3", "incremental_loss": float(data["A6"] - data["A3"])})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summaries: list[dict[str, Any]] = []
    for keys, frame in detail.groupby(["family", "backend", "profile", "contrast"]):
        values = frame["incremental_loss"].to_numpy(dtype=float)
        summaries.append({
            "family": keys[0], "backend": keys[1], "profile": keys[2], "contrast": keys[3],
            "fold_count": int(frame["outer_fold"].nunique()),
            "incremental_loss_mean": float(np.mean(values)),
            "incremental_loss_median": float(np.median(values)),
            "positive_fold_ratio": float(np.mean(values > 0)),
            "worst_fold": float(np.min(values)),
            "sign_flip_p": _exact_sign_flip_p(values),
        })
    summary = pd.DataFrame(summaries)
    summary["bh_q"] = summary.groupby("family")["sign_flip_p"].transform(_bh_qvalues)
    return summary


def _profile_consistency(fold_level: pd.DataFrame) -> pd.DataFrame:
    if fold_level.empty:
        return pd.DataFrame()
    data = fold_level[(fold_level["family"] == "lightgbm") & (fold_level["scope_type"] == "all")]
    pivot = data.pivot_table(index=["condition", "outer_fold"], columns="profile", values="raw_pr_auc_loss", aggfunc="first").reset_index()
    if not {"full_reduced", "common_period"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot["direction_agreement"] = np.sign(pivot["full_reduced"]) == np.sign(pivot["common_period"])
    pivot["common_minus_full_effect"] = pivot["common_period"] - pivot["full_reduced"]
    summary = pivot.groupby("condition", as_index=False).agg(
        paired_fold_count=("outer_fold", "nunique"),
        direction_agreement_ratio=("direction_agreement", "mean"),
        common_minus_full_effect_mean=("common_minus_full_effect", "mean"),
        full_effect_mean=("full_reduced", "mean"),
        common_effect_mean=("common_period", "mean"),
    )
    return summary


def _decision_table(summary: pd.DataFrame, synergy: pd.DataFrame, profile_consistency: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    thresholds = config["decision_thresholds"]
    official = summary[(summary["family"] == "lightgbm")].copy()
    consistency_map = profile_consistency.set_index("condition").to_dict("index") if not profile_consistency.empty else {}
    rows: list[dict[str, Any]] = []
    for row in official.itertuples(index=False):
        if row.scope_type == "all" and row.condition in {"A1", "A2"}:
            consistent = consistency_map.get(row.condition, {}).get("direction_agreement_ratio", 0.0) >= 0.75
            passed = (
                row.raw_pr_auc_loss_mean >= thresholds["global_min_mean_loss"]
                and row.raw_pr_auc_loss_median > 0
                and row.positive_fold_ratio >= thresholds["global_min_positive_fold_ratio"]
                and row.recent_positive_fold_ratio >= 0.75
                and row.raw_pr_auc_loss_worst_fold >= thresholds["global_worst_fold_floor"]
                and consistent
            )
            rows.append({
                "hypothesis": f"{row.condition}_{row.profile}",
                "condition": row.condition,
                "profile": row.profile,
                "scope": "all_validation",
                "effect_mean": row.raw_pr_auc_loss_mean,
                "effect_median": row.raw_pr_auc_loss_median,
                "positive_fold_ratio": row.positive_fold_ratio,
                "recent_positive_fold_ratio": row.recent_positive_fold_ratio,
                "worst_fold": row.raw_pr_auc_loss_worst_fold,
                "q_value": row.bh_q,
                "profile_direction_consistency": consistent,
                "decision": "KEEP" if passed else "HOLD",
            })
        if row.scope_type == "target_bucket" and row.condition == "S1" and row.profile == "common_period":
            passed = (
                row.raw_pr_auc_loss_mean >= thresholds["battery_min_mean_loss"]
                and row.raw_pr_auc_loss_median > 0
                and row.positive_fold_ratio >= thresholds["global_min_positive_fold_ratio"]
                and row.recent_positive_fold_ratio >= 0.75
            )
            rows.append({
                "hypothesis": "battery_shorting_S1",
                "condition": "S1",
                "profile": row.profile,
                "scope": "battery_materials",
                "effect_mean": row.raw_pr_auc_loss_mean,
                "effect_median": row.raw_pr_auc_loss_median,
                "positive_fold_ratio": row.positive_fold_ratio,
                "recent_positive_fold_ratio": row.recent_positive_fold_ratio,
                "worst_fold": row.raw_pr_auc_loss_worst_fold,
                "q_value": row.bh_q,
                "profile_direction_consistency": np.nan,
                "decision": "CONDITIONAL_KEEP" if passed else "HOLD",
            })
    if not synergy.empty:
        for contrast, threshold, label in [
            ("A3_over_best_A1_A2", thresholds["joint_incremental_min"], "joint_shorting_market"),
            ("A5_minus_A2", thresholds["etf_incremental_min"], "etf_with_market"),
            ("A6_minus_A3", thresholds["etf_incremental_min"], "etf_with_shorting_market"),
        ]:
            for row in synergy[(synergy["family"] == "lightgbm") & (synergy["contrast"] == contrast)].itertuples(index=False):
                passed = row.incremental_loss_mean >= threshold and row.positive_fold_ratio >= 0.75
                rows.append({
                    "hypothesis": f"{label}_{row.profile}",
                    "condition": contrast,
                    "profile": row.profile,
                    "scope": "all_validation",
                    "effect_mean": row.incremental_loss_mean,
                    "effect_median": row.incremental_loss_median,
                    "positive_fold_ratio": row.positive_fold_ratio,
                    "recent_positive_fold_ratio": np.nan,
                    "worst_fold": row.worst_fold,
                    "q_value": row.bh_q,
                    "profile_direction_consistency": np.nan,
                    "decision": "KEEP" if passed else "HOLD",
                })
    return pd.DataFrame(rows)


def aggregate_results(cache_root: Path, output_dir: Path, config: dict[str, Any], run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, tasks = _collect_metric_jsons(cache_root)
    atomic_csv(metrics, output_dir / "all_metrics.csv", METRIC_COLUMNS)
    atomic_csv(tasks, output_dir / "task_manifest.csv", TASK_COLUMNS)
    failed = tasks[~tasks["status"].isin(["completed"])] if not tasks.empty else pd.DataFrame()
    atomic_csv(failed, output_dir / "failed_tasks.csv")

    deltas = _paired_deltas(metrics)
    atomic_csv(deltas, output_dir / "paired_ablation_deltas.csv", DELTA_COLUMNS)
    fold_level, summary = _summary(deltas)
    atomic_csv(fold_level, output_dir / "fold_level_effects.csv")
    atomic_csv(summary, output_dir / "global_feature_utility_summary.csv")

    synergy = _synergy(fold_level)
    atomic_csv(synergy, output_dir / "composite_group_synergy.csv")
    consistency = _profile_consistency(fold_level)
    atomic_csv(consistency, output_dir / "profile_consistency.csv")
    decision = _decision_table(summary, synergy, consistency, config)
    atomic_csv(decision, output_dir / "decision_table.csv")

    drift_cols = [
        "family", "backend", "profile", "condition", "scope_type", "scope_value",
        "early_raw_pr_auc_loss_mean", "recent_raw_pr_auc_loss_mean", "recent_minus_early",
        "recent_positive_fold_ratio", "fold_effect_spearman"
    ]
    drift = summary[[c for c in drift_cols if c in summary.columns]].copy() if not summary.empty else pd.DataFrame(columns=drift_cols)
    atomic_csv(drift, output_dir / "recent_drift_summary.csv", drift_cols)

    worst = fold_level.sort_values("raw_pr_auc_loss").groupby(
        ["family", "profile", "condition", "scope_type", "scope_value"], as_index=False
    ).first() if not fold_level.empty else pd.DataFrame()
    atomic_csv(worst, output_dir / "worst_fold_diagnostics.csv")

    battery = summary[(summary["scope_type"] == "target_bucket") & (summary["scope_value"] == "battery_materials")].copy() if not summary.empty else pd.DataFrame()
    atomic_csv(battery, output_dir / "battery_sector_confirmatory.csv")
    multiple = summary[["family", "profile", "condition", "scope_type", "scope_value", "sign_flip_p", "bh_q"]].copy() if not summary.empty else pd.DataFrame()
    atomic_csv(multiple, output_dir / "multiple_testing_summary.csv")

    completed_tasks = int(tasks.loc[tasks["status"].eq("completed"), "task_id"].nunique()) if not tasks.empty else 0
    failed_tasks = int(tasks["status"].eq("failed").sum()) if not tasks.empty else 0
    incomplete_tasks = int((~tasks["status"].eq("completed")).sum()) if not tasks.empty else 0
    family_counts = metrics.groupby("family")["task_id"].nunique().to_dict() if not metrics.empty else {}
    summary_payload: dict[str, Any] = {
        "status": "completed" if incomplete_tasks == 0 and completed_tasks > 0 else ("partial" if completed_tasks > 0 else "no_results"),
        "completed_task_count": completed_tasks,
        "failed_task_count": failed_tasks,
        "incomplete_task_count": incomplete_tasks,
        "family_task_counts": family_counts,
        "metric_row_count": int(len(metrics)),
        "paired_delta_row_count": int(len(deltas)),
        "fold_effect_row_count": int(len(fold_level)),
        "decision_count": int(len(decision)),
        "keep_count": int(decision["decision"].isin(["KEEP", "CONDITIONAL_KEEP"]).sum()) if not decision.empty else 0,
        "output_dir": str(output_dir),
    }
    if run_metadata:
        summary_payload.update(run_metadata)
    atomic_json(summary_payload, output_dir / "run_summary.json")
    return summary_payload
