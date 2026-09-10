from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_json, bh_adjust, bootstrap_mean_ci, exact_sign_flip_p

LOGGER = logging.getLogger(__name__)
METRICS = [
    "raw_pr_auc", "raw_roc_auc", "raw_brier", "raw_logloss", "raw_pr_auc_lift",
    "top_1pct_precision", "top_1pct_recall", "top_3pct_precision", "top_3pct_recall",
    "top_5pct_precision", "top_5pct_recall",
]


def _read_result_jsons(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    if not directory.exists():
        return pd.DataFrame(), pd.DataFrame()
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            row["result_path"] = str(path)
            identity = row.get("identity") if isinstance(row.get("identity"), dict) else {}
            row["_dataset_signature"] = row.get("dataset_signature") or identity.get("dataset_signature", "")
            row["_run_signature"] = row.get("run_signature") or identity.get("run_signature", "")
            if row.get("status") == "completed":
                completed.append(row)
            else:
                failed.append(row)
        except Exception as exc:
            failed.append({"status": "unreadable", "result_path": str(path), "error": repr(exc)})
    return pd.DataFrame(completed), pd.DataFrame(failed)


def _baseline_map(metrics: pd.DataFrame, backend: str, test_type: str) -> pd.DataFrame:
    base = metrics[(metrics["backend"] == backend) & (metrics["test_type"] == test_type)].copy()
    cols = ["outer_fold"] + METRICS
    base = base[cols].drop_duplicates("outer_fold")
    return base.rename(columns={m: f"baseline_{m}" for m in METRICS})


def _paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for backend in sorted(metrics["backend"].dropna().unique()):
        backend_df = metrics[metrics["backend"] == backend].copy()
        baseline = _baseline_map(backend_df, backend, "baseline_all_valid")
        tests = backend_df[backend_df["test_type"].isin(["single_feature_loo", "cluster_loo"])].copy()
        if not tests.empty and not baseline.empty:
            merged = tests.merge(baseline, on="outer_fold", how="inner", validate="many_to_one")
            for metric in METRICS:
                if metric in merged and f"baseline_{metric}" in merged:
                    if metric in {"raw_brier", "raw_logloss"}:
                        # Lower is better: positive means the ablated model became worse.
                        merged[f"delta_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]
                    else:
                        merged[f"delta_{metric}"] = merged[f"baseline_{metric}"] - merged[metric]
            rows.append(merged)
        pruned = backend_df[backend_df["test_type"] == "baseline_pruned"].copy()
        conditional = backend_df[backend_df["test_type"].isin(["conditional_drop_rep", "conditional_add_nonrep"])].copy()
        if not conditional.empty and not pruned.empty:
            pcols = ["outer_fold"] + METRICS
            p = pruned[pcols].drop_duplicates("outer_fold").rename(columns={m: f"baseline_{m}" for m in METRICS})
            merged = conditional.merge(p, on="outer_fold", how="inner", validate="many_to_one")
            for metric in METRICS:
                if metric not in merged or f"baseline_{metric}" not in merged:
                    continue
                drop_mask = merged["test_type"].eq("conditional_drop_rep")
                if metric in {"raw_brier", "raw_logloss"}:
                    values = np.where(
                        drop_mask,
                        merged[metric] - merged[f"baseline_{metric}"],
                        merged[f"baseline_{metric}"] - merged[metric],
                    )
                else:
                    values = np.where(
                        drop_mask,
                        merged[f"baseline_{metric}"] - merged[metric],
                        merged[metric] - merged[f"baseline_{metric}"],
                    )
                merged[f"delta_{metric}"] = values
            rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _summarize_effects(paired: pd.DataFrame, test_type: str) -> pd.DataFrame:
    part = paired[paired["test_type"] == test_type].copy()
    if part.empty:
        return pd.DataFrame()
    key = "feature" if test_type != "cluster_loo" else "cluster_id"
    output: list[dict[str, Any]] = []
    for (backend, item), group in part.groupby(["backend", key], dropna=False, sort=False):
        values = group["delta_raw_pr_auc"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        low, high = bootstrap_mean_ci(values) if len(values) else (float("nan"), float("nan"))
        row: dict[str, Any] = {
            "backend": backend,
            key: item,
            "feature": group["feature"].iloc[0] if "feature" in group else "",
            "test_type": test_type,
            "fold_count": int(len(values)),
            "mean_pr_auc_utility": float(np.mean(values)) if len(values) else float("nan"),
            "median_pr_auc_utility": float(np.median(values)) if len(values) else float("nan"),
            "std_pr_auc_utility": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "ci95_low": low,
            "ci95_high": high,
            "positive_fold_ratio": float(np.mean(values > 0)) if len(values) else float("nan"),
            "worst_fold_utility": float(np.min(values)) if len(values) else float("nan"),
            "best_fold_utility": float(np.max(values)) if len(values) else float("nan"),
            "exact_sign_flip_p": exact_sign_flip_p(values),
        }
        recent = group[group["outer_fold"].isin([4, 5, 6, 7])]["delta_raw_pr_auc"].to_numpy(dtype=np.float64)
        recent = recent[np.isfinite(recent)]
        row["recent_mean_pr_auc_utility"] = float(np.mean(recent)) if len(recent) else float("nan")
        row["recent_positive_fold_ratio"] = float(np.mean(recent > 0)) if len(recent) else float("nan")
        for metric in ["raw_roc_auc", "raw_pr_auc_lift", "top_3pct_precision", "top_3pct_recall", "raw_brier", "raw_logloss"]:
            col = f"delta_{metric}"
            if col in group:
                arr = group[col].to_numpy(dtype=np.float64)
                row[f"mean_{metric}_utility"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        output.append(row)
    summary = pd.DataFrame(output)
    if not summary.empty:
        summary["bh_q"] = np.nan
        for backend, idx in summary.groupby("backend").groups.items():
            positions = list(idx)
            summary.loc[positions, "bh_q"] = bh_adjust(summary.loc[positions, "exact_sign_flip_p"].to_numpy(dtype=np.float64))
    return summary


def _gpu_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "xgboost_cuda" not in set(metrics.get("backend", pd.Series(dtype=str)).dropna()):
        return pd.DataFrame()
    paired = _paired_deltas(metrics[metrics["backend"] == "xgboost_cuda"].copy())
    return _summarize_effects(paired, "single_feature_loo")


def _summary_schema(test_type: str) -> list[str]:
    key = "cluster_id" if test_type == "cluster_loo" else "feature"
    return [
        "backend", key, "feature", "test_type", "fold_count", "mean_pr_auc_utility",
        "median_pr_auc_utility", "std_pr_auc_utility", "ci95_low", "ci95_high",
        "positive_fold_ratio", "worst_fold_utility", "best_fold_utility",
        "exact_sign_flip_p", "recent_mean_pr_auc_utility", "recent_positive_fold_ratio",
        "mean_raw_roc_auc_utility", "mean_raw_pr_auc_lift_utility",
        "mean_top_3pct_precision_utility", "mean_top_3pct_recall_utility",
        "mean_raw_brier_utility", "mean_raw_logloss_utility", "bh_q",
    ]


def _with_schema(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def aggregate_results(output_dir: Path, feature_count: int, fold_count: int, dataset_signature: str, run_signature: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    lgb_metrics, lgb_failed = _read_result_jsons(output_dir / "task_results" / "lightgbm_cpu")
    gpu_metrics, gpu_failed = _read_result_jsons(output_dir / "task_results" / "xgboost_cuda")
    # GPU baseline files use a different deterministic filename but still live in the same directory.
    if not lgb_metrics.empty:
        lgb_metrics = lgb_metrics[
            lgb_metrics["_dataset_signature"].astype(str).eq(dataset_signature)
            & lgb_metrics["_run_signature"].astype(str).eq(run_signature)
        ].copy()
    if not gpu_metrics.empty:
        gpu_metrics = gpu_metrics[
            gpu_metrics["_dataset_signature"].astype(str).eq(dataset_signature)
            & gpu_metrics["_run_signature"].astype(str).eq(run_signature)
        ].copy()
    if not lgb_failed.empty and "_dataset_signature" in lgb_failed:
        ds = lgb_failed["_dataset_signature"].fillna("").astype(str)
        rs = lgb_failed.get("_run_signature", pd.Series("", index=lgb_failed.index)).fillna("").astype(str)
        lgb_failed = lgb_failed[(ds.eq("") | ds.eq(dataset_signature)) & (rs.eq("") | rs.eq(run_signature))].copy()
    if not gpu_failed.empty and "_dataset_signature" in gpu_failed:
        ds = gpu_failed["_dataset_signature"].fillna("").astype(str)
        rs = gpu_failed.get("_run_signature", pd.Series("", index=gpu_failed.index)).fillna("").astype(str)
        gpu_failed = gpu_failed[(ds.eq("") | ds.eq(dataset_signature)) & (rs.eq("") | rs.eq(run_signature))].copy()
    metrics = pd.concat([lgb_metrics, gpu_metrics], ignore_index=True, sort=False)
    failed = pd.concat([lgb_failed, gpu_failed], ignore_index=True, sort=False)
    failed = _with_schema(failed, [
        "status", "backend", "dataset_signature", "run_signature", "stage", "test_type",
        "condition_id", "feature", "outer_fold", "error", "traceback", "result_path",
    ])
    if metrics.empty:
        metrics = pd.DataFrame(columns=["backend", "test_type", "feature", "outer_fold", *METRICS])
    metrics.to_csv(output_dir / "all_model_metrics.csv", index=False)
    failed.to_csv(output_dir / "failed_tasks.csv", index=False)

    paired = _paired_deltas(metrics)
    paired.to_csv(output_dir / "paired_ablation_deltas.csv", index=False)
    loo = _with_schema(_summarize_effects(paired, "single_feature_loo"), _summary_schema("single_feature_loo"))
    conditional_drop = _summarize_effects(paired, "conditional_drop_rep")
    conditional_add = _summarize_effects(paired, "conditional_add_nonrep")
    conditional = pd.concat([conditional_drop, conditional_add], ignore_index=True, sort=False)
    conditional = _with_schema(conditional, _summary_schema("conditional_drop_rep"))
    cluster = _with_schema(_summarize_effects(paired, "cluster_loo"), _summary_schema("cluster_loo"))
    loo.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    conditional.to_csv(output_dir / "conditional_ablation_summary.csv", index=False)
    cluster.to_csv(output_dir / "cluster_ablation_summary.csv", index=False)

    corr_path = output_dir / "correlation" / "feature_correlation_summary.csv"
    corr = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame({"feature": sorted(metrics.get("feature", pd.Series(dtype=str)).dropna().unique())})
    cpu_loo = loo[loo["backend"] == "lightgbm_cpu"].copy() if not loo.empty else pd.DataFrame()
    if not cpu_loo.empty:
        cpu_loo = cpu_loo.drop(columns=["backend", "test_type"], errors="ignore").add_prefix("loo_").rename(columns={"loo_feature": "feature"})
    cpu_cond = conditional[conditional["backend"] == "lightgbm_cpu"].copy() if not conditional.empty else pd.DataFrame()
    if not cpu_cond.empty:
        cpu_cond = cpu_cond.sort_values(["feature", "mean_pr_auc_utility"], ascending=[True, False]).drop_duplicates("feature")
        cpu_cond = cpu_cond.drop(columns=["backend", "test_type"], errors="ignore").add_prefix("conditional_").rename(columns={"conditional_feature": "feature"})
    gpu = loo[loo["backend"] == "xgboost_cuda"].copy() if not loo.empty else pd.DataFrame()
    if not gpu.empty:
        gpu = gpu.drop(columns=["backend", "test_type"], errors="ignore").add_prefix("gpu_loo_").rename(columns={"gpu_loo_feature": "feature"})
    master = corr.copy()
    for frame in [cpu_loo, cpu_cond, gpu]:
        if not frame.empty:
            master = master.merge(frame, on="feature", how="left")

    primary_clusters_path = output_dir / "correlation" / "primary_clusters.csv"
    primary_clusters = pd.read_csv(primary_clusters_path) if primary_clusters_path.exists() else pd.DataFrame()
    if not cluster.empty and not primary_clusters.empty:
        cluster_cpu = cluster[cluster["backend"] == "lightgbm_cpu"].copy()
        cluster_cpu["cluster_id"] = pd.to_numeric(cluster_cpu["cluster_id"], errors="coerce")
        cluster_map = cluster_cpu[["cluster_id", "mean_pr_auc_utility", "positive_fold_ratio", "bh_q"]].rename(columns={
            "mean_pr_auc_utility": "cluster_mean_pr_auc_utility",
            "positive_fold_ratio": "cluster_positive_fold_ratio",
            "bh_q": "cluster_bh_q",
        })
        member_map = primary_clusters[["feature", "cluster_id"]].copy()
        member_map["cluster_id"] = pd.to_numeric(member_map["cluster_id"], errors="coerce")
        member_map = member_map.merge(cluster_map, on="cluster_id", how="left")
        master = master.merge(member_map.drop(columns=["cluster_id"]), on="feature", how="left")

    def decide(row: pd.Series) -> tuple[str, str]:
        loo_mean = row.get("loo_mean_pr_auc_utility", np.nan)
        loo_pos = row.get("loo_positive_fold_ratio", np.nan)
        loo_worst = row.get("loo_worst_fold_utility", np.nan)
        loo_q = row.get("loo_bh_q", np.nan)
        cond_mean = row.get("conditional_mean_pr_auc_utility", np.nan)
        cond_pos = row.get("conditional_positive_fold_ratio", np.nan)
        cluster_mean = row.get("cluster_mean_pr_auc_utility", np.nan)
        max_corr = row.get("max_abs_correlation", np.nan)
        missing = row.get("missing_ratio", np.nan)
        gpu_mean = row.get("gpu_loo_mean_pr_auc_utility", np.nan)
        gpu_agrees = not np.isfinite(gpu_mean) or not np.isfinite(loo_mean) or np.sign(gpu_mean) == np.sign(loo_mean) or abs(gpu_mean) < 0.001
        if np.isfinite(loo_mean) and loo_mean >= 0.003 and loo_pos >= 0.75 and loo_worst >= -0.01 and gpu_agrees:
            strength = "STRONG" if np.isfinite(loo_q) and loo_q <= 0.10 else "MODERATE"
            return "KEEP_DIRECT", strength
        if np.isfinite(cond_mean) and cond_mean >= 0.003 and cond_pos >= 0.75:
            return "KEEP_CONDITIONAL", "MODERATE"
        if np.isfinite(cluster_mean) and cluster_mean >= 0.005 and np.isfinite(max_corr) and max_corr >= 0.90 and (not np.isfinite(loo_mean) or abs(loo_mean) < 0.003):
            return "REDUNDANT_CLUSTER_SIGNAL", "MODERATE"
        if np.isfinite(loo_mean) and loo_mean <= -0.003 and loo_pos <= 0.25:
            return "DROP_HARMFUL", "MODERATE"
        if np.isfinite(missing) and missing >= 0.80 and (not np.isfinite(loo_mean) or abs(loo_mean) < 0.001) and (not np.isfinite(cond_mean) or abs(cond_mean) < 0.001):
            return "DROP_LOW_VALUE", "WEAK"
        return "HOLD", "WEAK"

    decisions = master.apply(decide, axis=1, result_type="expand")
    master["decision"] = decisions[0]
    master["evidence_strength"] = decisions[1]
    sort_cols = [c for c in ["decision", "loo_mean_pr_auc_utility", "conditional_mean_pr_auc_utility", "quality_score"] if c in master]
    ascending = [True] + [False] * (len(sort_cols) - 1)
    if sort_cols:
        master.sort_values(sort_cols, ascending=ascending, inplace=True)
    master.to_csv(output_dir / "feature_master_decision.csv", index=False)

    expected_primary = feature_count * fold_count
    completed_primary = 0
    if not lgb_metrics.empty:
        completed_primary = int(len(lgb_metrics[lgb_metrics["test_type"] == "single_feature_loo"]))
    cpu_failed_count = int(len(lgb_failed))
    optional_gpu_failed_count = int(len(gpu_failed))
    summary = {
        "status": "completed" if completed_primary >= expected_primary and cpu_failed_count == 0 else "partial",
        "feature_count": feature_count,
        "fold_count": fold_count,
        "expected_primary_single_feature_tasks": expected_primary,
        "completed_primary_single_feature_tasks": completed_primary,
        "primary_completion_ratio": completed_primary / expected_primary if expected_primary else 0.0,
        "completed_metric_rows": int(len(metrics)),
        "failed_task_rows": int(len(failed)),
        "cpu_failed_task_rows": cpu_failed_count,
        "optional_gpu_failed_task_rows": optional_gpu_failed_count,
        "dataset_signature": dataset_signature,
        "run_signature": run_signature,
        "master_decision_rows": int(len(master)),
        "decision_counts": master["decision"].value_counts(dropna=False).to_dict() if "decision" in master else {},
        "gpu_completed_feature_tasks": int(len(gpu_metrics[gpu_metrics.get("test_type", pd.Series(dtype=str)) == "single_feature_loo"])) if not gpu_metrics.empty else 0,
    }
    atomic_json(summary, output_dir / "aggregation_summary.json")
    return summary
