from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p.index)
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    out.loc[adjusted.index] = adjusted
    return out


def _pair_scope_type(scope: pd.Series) -> pd.Series:
    return scope.replace({"target_bucket": "bucket", "target_ticker": "ticker"})


def paired_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.copy()
    work["pair_scope_type"] = _pair_scope_type(work["scope_type"].astype(str))
    key = ["fold", "seed", "pair_scope_type", "scope_value"]
    cols = ["pr_auc", "roc_auc", "brier", "logloss", "alert_precision", "alert_recall"]

    baseline = work.loc[work["experiment"].eq("baseline")].copy()
    base = baseline[key + cols].drop_duplicates(key, keep="last")
    # 빈 종목/단일 클래스 scope는 baseline 지표가 전부 NaN일 수 있다. 이 경우도
    # baseline 행 자체는 존재하므로 "짝 없음"으로 오인하면 안 된다.
    base["_baseline_present"] = True
    base = base.rename(columns={c: f"baseline_{c}" for c in cols})
    compared = work.loc[~work["experiment"].eq("baseline")].merge(
        base, on=key, how="left", validate="many_to_one"
    )
    missing_baseline = compared["_baseline_present"].isna()
    if missing_baseline.any():
        sample = compared.loc[missing_baseline, ["experiment", "fold", "seed", "scope_type", "scope_value"]].head(10)
        raise ValueError(f"baseline pairing 실패:\n{sample.to_string(index=False)}")
    compared = compared.drop(columns=["_baseline_present"])

    compared["pr_auc_loss_when_removed"] = compared["baseline_pr_auc"] - compared["pr_auc"]
    compared["roc_auc_loss_when_removed"] = compared["baseline_roc_auc"] - compared["roc_auc"]
    compared["brier_increase_when_removed"] = compared["brier"] - compared["baseline_brier"]
    compared["logloss_increase_when_removed"] = compared["logloss"] - compared["baseline_logloss"]
    compared["alert_precision_loss_when_removed"] = compared["baseline_alert_precision"] - compared["alert_precision"]
    compared["alert_recall_loss_when_removed"] = compared["baseline_alert_recall"] - compared["alert_recall"]
    return compared


def summarize_deltas(deltas: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 20260722) -> pd.DataFrame:
    """Summarize ablation effects using folds, not seeds, as statistical units.

    Seeds share the same validation dates and are first averaged within each fold. The
    bootstrap and sign consistency are then calculated from fold-level means.
    """
    rng = np.random.default_rng(seed)
    value_cols = [
        "pr_auc_loss_when_removed", "roc_auc_loss_when_removed", "brier_increase_when_removed",
        "logloss_increase_when_removed", "alert_precision_loss_when_removed", "alert_recall_loss_when_removed",
    ]
    group_cols = [c for c in [
        "experiment", "namespace", "ablation_mode", "target_group", "target_bucket",
        "target_ticker", "scope_type", "pair_scope_type", "scope_value",
    ] if c in deltas.columns]

    rows: list[dict] = []
    for keys, block in deltas.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        fold_means = block.groupby("fold", as_index=False)[value_cols].mean(numeric_only=True)
        row["fold_count"] = int(fold_means["fold"].nunique())
        row["seed_count"] = int(block["seed"].nunique())
        row["paired_units"] = row["fold_count"]
        for col in value_cols:
            values = pd.to_numeric(fold_means[col], errors="coerce").dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{col}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{col}_sign_consistency"] = float(np.mean(values > 0)) if len(values) else np.nan
            row[f"{col}_valid_fold_count"] = int(len(values))
            if len(values):
                samples = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
                row[f"{col}_ci_low"] = float(np.quantile(samples, 0.025))
                row[f"{col}_ci_high"] = float(np.quantile(samples, 0.975))
                # 유한 bootstrap 표본에서 p=0이 되는 것을 방지하는 add-one 보정.
                # 이는 downstream FDR가 허위로 0이 되는 문제도 막는다.
                non_positive = (int(np.sum(samples <= 0)) + 1) / (len(samples) + 1)
                non_negative = (int(np.sum(samples >= 0)) + 1) / (len(samples) + 1)
                row[f"{col}_p_two_sided"] = float(min(1.0, 2 * min(non_positive, non_negative)))
            else:
                row[f"{col}_ci_low"] = row[f"{col}_ci_high"] = row[f"{col}_p_two_sided"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    pcol = "pr_auc_loss_when_removed_p_two_sided"
    if pcol in out.columns:
        # FDR within comparable experiment/scope families, not across all heterogeneous tests.
        family_cols = [c for c in ["namespace", "ablation_mode", "pair_scope_type"] if c in out.columns]
        if family_cols:
            out["pr_auc_fdr_q"] = out.groupby(family_cols, dropna=False)[pcol].transform(benjamini_hochberg)
        else:
            out["pr_auc_fdr_q"] = benjamini_hochberg(out[pcol])
    return out
