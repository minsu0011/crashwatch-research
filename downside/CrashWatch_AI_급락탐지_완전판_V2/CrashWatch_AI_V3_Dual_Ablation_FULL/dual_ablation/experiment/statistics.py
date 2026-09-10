from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p.index, dtype=float)
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    out.loc[adjusted.index] = adjusted
    return out


def normalize_pair_scope(values: pd.Series) -> pd.Series:
    return values.replace({"target_bucket": "bucket", "target_ticker": "ticker"})


def paired_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    data = metrics.copy()
    data["pair_scope_type"] = normalize_pair_scope(data["scope_type"])
    key = ["fold", "seed", "pair_scope_type", "scope_value"]
    metric_cols = ["pr_auc", "roc_auc", "brier", "logloss", "alert_precision", "alert_recall"]
    baseline = data.loc[data["experiment"].eq("baseline")].copy()
    if baseline.duplicated(key).any():
        raise ValueError("baseline has duplicate normalized pairing keys")
    base = baseline[key + metric_cols + ["validation_row_hash"]].rename(
        columns={**{c: f"baseline_{c}" for c in metric_cols}, "validation_row_hash": "baseline_row_hash"},
    )
    compared = data.loc[~data["experiment"].eq("baseline")].copy()
    if compared.duplicated(["experiment"] + key).any():
        raise ValueError("ablation has duplicate normalized pairing keys")
    compared = compared.merge(base, on=key, how="left", validate="many_to_one")
    missing = compared["baseline_row_hash"].isna()
    if missing.any():
        raise ValueError(f"unpaired baseline scopes: {int(missing.sum())}")
    mismatch = compared["validation_row_hash"].ne(compared["baseline_row_hash"])
    if mismatch.any():
        sample = compared.loc[mismatch, ["experiment"] + key].head().to_dict("records")
        raise AssertionError(f"baseline/ablation validation row ids differ: {sample}")
    compared["pr_auc_loss_when_removed"] = compared["baseline_pr_auc"] - compared["pr_auc"]
    compared["roc_auc_loss_when_removed"] = compared["baseline_roc_auc"] - compared["roc_auc"]
    compared["brier_increase_when_removed"] = compared["brier"] - compared["baseline_brier"]
    compared["logloss_increase_when_removed"] = compared["logloss"] - compared["baseline_logloss"]
    compared["alert_precision_loss_when_removed"] = compared["baseline_alert_precision"] - compared["alert_precision"]
    compared["alert_recall_loss_when_removed"] = compared["baseline_alert_recall"] - compared["alert_recall"]
    return compared


def _bootstrap_fold_means(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float, float]:
    if not len(values):
        return np.nan, np.nan, np.nan
    boot = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    # An empty empirical tail means "below the Monte Carlo resolution", not
    # mathematical certainty.  The add-one correction keeps p-values (and
    # therefore BH q-values) strictly positive with finite bootstrap samples.
    lower_count = int(np.count_nonzero(boot <= 0))
    upper_count = int(np.count_nonzero(boot >= 0))
    tail_probability = (min(lower_count, upper_count) + 1) / (samples + 1)
    pvalue = min(1.0, float(2 * tail_probability))
    return float(low), float(high), pvalue


def summarize_deltas(deltas: pd.DataFrame, bootstrap_samples: int = 4000, seed: int = 20260722) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    value_cols = [
        "pr_auc_loss_when_removed", "roc_auc_loss_when_removed", "brier_increase_when_removed",
        "logloss_increase_when_removed", "alert_precision_loss_when_removed", "alert_recall_loss_when_removed",
    ]
    group_cols = [
        c for c in ["experiment", "namespace", "ablation_mode", "target_group", "target_bucket",
                    "target_ticker", "pair_scope_type", "scope_value"] if c in deltas.columns
    ]
    fold_keys = group_cols + ["fold"]
    # Seeds are repeated fits of the same market fold.  Average them first;
    # folds, not seeds, are the statistical units.
    fold_level = deltas.groupby(fold_keys, dropna=False, as_index=False)[value_cols].mean()
    seed_counts = deltas.groupby(group_cols, dropna=False, as_index=False).agg(seed_count=("seed", "nunique"))
    fold_level = fold_level.merge(seed_counts, on=group_cols, how="left", validate="many_to_one")
    rows: list[dict] = []
    for keys, block in fold_level.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["fold_count"] = int(block["fold"].nunique())
        row["seed_count"] = int(block["seed_count"].max())
        for col in value_cols:
            values = pd.to_numeric(block[col], errors="coerce").dropna().to_numpy()
            low, high, pvalue = _bootstrap_fold_means(values, rng, bootstrap_samples)
            row[f"{col}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{col}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{col}_ci_lower"] = low
            row[f"{col}_ci_upper"] = high
            row[f"{col}_positive_fold_ratio"] = float(np.mean(values > 0)) if len(values) else np.nan
            row[f"{col}_negative_fold_ratio"] = float(np.mean(values < 0)) if len(values) else np.nan
            row[f"{col}_raw_p_value"] = pvalue
        # Requested generic columns use PR-AUC loss as the primary sensitivity.
        primary = "pr_auc_loss_when_removed"
        row.update({
            "mean_delta": row[f"{primary}_mean"], "median_delta": row[f"{primary}_median"],
            "std_delta": row[f"{primary}_std"], "ci_lower": row[f"{primary}_ci_lower"],
            "ci_upper": row[f"{primary}_ci_upper"],
            "positive_fold_ratio": row[f"{primary}_positive_fold_ratio"],
            "negative_fold_ratio": row[f"{primary}_negative_fold_ratio"],
            "raw_p_value": row[f"{primary}_raw_p_value"],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out["fdr_q_value"] = benjamini_hochberg(out["raw_p_value"]) if len(out) else np.nan
    out["pr_auc_fdr_q"] = out["fdr_q_value"]
    return out
