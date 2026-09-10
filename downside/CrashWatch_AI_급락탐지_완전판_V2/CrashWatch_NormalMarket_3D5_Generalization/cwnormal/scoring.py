from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cw7h.metrics import compute_metrics


def safe_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return compute_metrics(np.array([], dtype=np.uint8), np.array([], dtype=float), np.array([], dtype=np.int64))
    return compute_metrics(
        frame["target"].to_numpy(dtype=np.uint8),
        frame["prediction"].to_numpy(dtype=float),
        frame["date_ns"].to_numpy(dtype=np.int64),
    )


def ticker_metrics(frame: pd.DataFrame, min_pos: int = 2) -> pd.DataFrame:
    rows = []
    for ticker, part in frame.groupby("ticker", sort=True):
        pos = int(part["target"].sum())
        m = safe_metrics(part) if pos >= min_pos and pos < len(part) else {"raw_pr_auc": np.nan, "raw_pr_auc_lift": np.nan, "raw_roc_auc": np.nan}
        rows.append({
            "ticker": str(ticker), "rows": int(len(part)), "dates": int(part["date_ns"].nunique()),
            "positives": pos, "positive_rate": float(part["target"].mean()) if len(part) else np.nan,
            "pr_auc": m.get("raw_pr_auc", np.nan), "pr_auc_lift": m.get("raw_pr_auc_lift", np.nan),
            "roc_auc": m.get("raw_roc_auc", np.nan),
            "status": "EVALUABLE" if pos >= min_pos and pos < len(part) else "NOT_EVALUABLE",
        })
    return pd.DataFrame(rows)


def sector_metrics(frame: pd.DataFrame, min_pos: int = 2) -> pd.DataFrame:
    rows = []
    for sector, part in frame.groupby("sector", sort=True):
        pos = int(part["target"].sum())
        m = safe_metrics(part) if pos >= min_pos and pos < len(part) else {"raw_pr_auc": np.nan, "raw_pr_auc_lift": np.nan, "raw_roc_auc": np.nan}
        rows.append({"sector": str(sector), "rows": len(part), "dates": part["date_ns"].nunique(), "positives": pos,
                     "pr_auc_lift": m.get("raw_pr_auc_lift", np.nan), "roc_auc": m.get("raw_roc_auc", np.nan)})
    return pd.DataFrame(rows)


def score_candidate(frame: pd.DataFrame, *, recent_fold_id: int = 7) -> dict[str, Any]:
    overall = safe_metrics(frame)
    by_fold = []
    for fold_id, part in frame.groupby("fold_id", sort=True):
        m = safe_metrics(part)
        by_fold.append({"fold_id": int(fold_id), **m, "dates": int(part["date_ns"].nunique())})
    fold_df = pd.DataFrame(by_fold)
    valid = fold_df[np.isfinite(fold_df.get("raw_roc_auc", np.nan))].copy() if not fold_df.empty else pd.DataFrame()
    mean_roc = float(valid["raw_roc_auc"].mean()) if len(valid) else np.nan
    median_roc = float(valid["raw_roc_auc"].median()) if len(valid) else np.nan
    worst_roc = float(valid["raw_roc_auc"].min()) if len(valid) else np.nan
    recent = valid[valid["fold_id"] == int(recent_fold_id)] if len(valid) else pd.DataFrame()
    recent_roc = float(recent["raw_roc_auc"].iloc[0]) if len(recent) else np.nan
    pr_lift = float(overall.get("raw_pr_auc_lift", np.nan))
    # Selection objective intentionally prioritizes ROC/generalization on NORMAL dates.
    pieces = [mean_roc, median_roc, worst_roc, recent_roc, min(pr_lift / 2.0, 1.0) if np.isfinite(pr_lift) else np.nan]
    if all(np.isfinite(x) for x in pieces):
        selection_score = 0.35 * mean_roc + 0.15 * median_roc + 0.15 * worst_roc + 0.20 * recent_roc + 0.15 * pieces[4]
    else:
        selection_score = -np.inf
    return {
        "selection_score": float(selection_score), "overall": overall,
        "mean_fold_roc": mean_roc, "median_fold_roc": median_roc, "worst_fold_roc": worst_roc,
        "recent_fold_roc": recent_roc, "fold_metrics": fold_df,
    }


def block_bootstrap_delta(base: pd.DataFrame, challenger: pd.DataFrame, *, block_dates: int = 20, n_boot: int = 500, seed: int = 17) -> dict[str, float]:
    merged = base[["row_id", "date_ns", "target", "prediction"]].merge(
        challenger[["row_id", "prediction"]], on="row_id", suffixes=("_base", "_challenger"), how="inner"
    )
    if merged.empty or merged["target"].nunique() < 2:
        return {"delta_roc": np.nan, "delta_pr_lift": np.nan, "roc_ci_low": np.nan, "roc_ci_high": np.nan, "pr_lift_ci_low": np.nan, "pr_lift_ci_high": np.nan}
    dates = np.sort(merged["date_ns"].unique())
    rng = np.random.default_rng(seed)
    blocks = [dates[i:i+block_dates] for i in range(0, len(dates), block_dates)]
    deltas_roc, deltas_pr = [], []
    for _ in range(int(n_boot)):
        sampled = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        part = pd.concat([merged[merged["date_ns"].isin(b)] for b in [sampled]], ignore_index=True)
        # Duplicated blocks are intentionally represented via repeated date selection by rebuilding indices below.
        pieces = []
        for d in sampled:
            pieces.append(merged[merged["date_ns"] == d])
        part = pd.concat(pieces, ignore_index=True) if pieces else merged.iloc[:0]
        y = part["target"].to_numpy(dtype=np.uint8)
        if np.unique(y).size < 2:
            continue
        b = part["prediction_base"].to_numpy(float); c = part["prediction_challenger"].to_numpy(float)
        roc_b, roc_c = roc_auc_score(y, b), roc_auc_score(y, c)
        rate = float(np.mean(y))
        pr_b, pr_c = average_precision_score(y, b) / rate, average_precision_score(y, c) / rate
        deltas_roc.append(roc_c - roc_b); deltas_pr.append(pr_c - pr_b)
    def ci(values):
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (np.nan, np.nan)
    rlo, rhi = ci(deltas_roc); plo, phi = ci(deltas_pr)
    y = merged["target"].to_numpy(dtype=np.uint8)
    base_roc = roc_auc_score(y, merged["prediction_base"]); chal_roc = roc_auc_score(y, merged["prediction_challenger"])
    rate = float(np.mean(y)); base_pr = average_precision_score(y, merged["prediction_base"]) / rate; chal_pr = average_precision_score(y, merged["prediction_challenger"]) / rate
    return {"delta_roc": float(chal_roc-base_roc), "delta_pr_lift": float(chal_pr-base_pr), "roc_ci_low": rlo, "roc_ci_high": rhi, "pr_lift_ci_low": plo, "pr_lift_ci_high": phi}


def permutation_sanity(frame: pd.DataFrame, *, n_perm: int = 99, block_dates: int = 20, seed: int = 503) -> dict[str, Any]:
    if frame.empty or frame["target"].nunique() < 2:
        return {"status": "NOT_EVALUABLE"}
    dates = np.sort(frame["date_ns"].unique())
    y = frame["target"].to_numpy(dtype=np.uint8); p = frame["prediction"].to_numpy(float)
    observed = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    blocks = [dates[i:i+int(block_dates)] for i in range(0, len(dates), int(block_dates))]
    by_date = {int(d): frame.loc[frame["date_ns"] == d, "target"].to_numpy(dtype=np.uint8) for d in dates}
    vals=[]
    for _ in range(int(n_perm)):
        order = rng.permutation(len(blocks))
        source_dates = np.concatenate([blocks[i] for i in order])[:len(dates)]
        yp=[]; pp=[]
        for dst, src in zip(dates, source_dates):
            dst_part = frame[frame["date_ns"] == dst]
            src_y = by_date[int(src)]
            if len(src_y) != len(dst_part):
                src_y = rng.choice(y, size=len(dst_part), replace=True)
            yp.append(src_y); pp.append(dst_part["prediction"].to_numpy(float))
        yp=np.concatenate(yp); pp=np.concatenate(pp)
        if np.unique(yp).size >= 2:
            vals.append(float(roc_auc_score(yp, pp)))
    pval = float((1 + sum(v >= observed for v in vals)) / (1 + len(vals))) if vals else np.nan
    return {"status":"PASS" if np.isfinite(pval) and pval <= 0.05 else "REVIEW", "observed_roc": observed, "permutation_mean_roc": float(np.mean(vals)) if vals else np.nan, "p_value": pval, "n_permutations": len(vals), "block_dates": int(block_dates)}
