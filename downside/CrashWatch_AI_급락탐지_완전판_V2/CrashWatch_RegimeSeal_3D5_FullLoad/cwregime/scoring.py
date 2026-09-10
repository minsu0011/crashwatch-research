from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cw7h.metrics import compute_metrics
from .regimes import REGIMES


def metrics_by_regime(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    y = predictions["target"].to_numpy(dtype=np.uint8)
    pred = predictions["prediction"].to_numpy(dtype=float)
    dates_ns = pd.to_datetime(predictions["date"]).astype("int64").to_numpy(dtype=np.int64)
    overall = compute_metrics(y, pred, dates_ns)
    rows = []
    for regime in REGIMES:
        part = predictions[predictions["regime"] == regime]
        if part.empty:
            metrics = compute_metrics(np.array([], dtype=np.uint8), np.array([], dtype=float), np.array([], dtype=np.int64))
        else:
            metrics = compute_metrics(
                part["target"].to_numpy(dtype=np.uint8),
                part["prediction"].to_numpy(dtype=float),
                pd.to_datetime(part["date"]).astype("int64").to_numpy(dtype=np.int64),
            )
        rows.append({"regime": regime, **metrics, "unique_dates": int(part["date"].nunique())})
    regime_df = pd.DataFrame(rows)
    valid = regime_df[np.isfinite(regime_df["raw_pr_auc_lift"]) & (regime_df["positives"] > 0)].copy()
    if valid.empty:
        score = float("-inf")
        components = {}
    else:
        lifts = np.clip(valid["raw_pr_auc_lift"].to_numpy(dtype=float), 1e-6, 10.0)
        regime_gmean = float(np.exp(np.mean(np.log(lifts))))
        worst = float(np.min(lifts))
        overall_lift = float(overall.get("raw_pr_auc_lift", np.nan))
        rate = float(overall.get("positive_rate", np.nan))
        top3 = float(overall.get("top_3pct_precision", np.nan))
        top3_lift = float(top3 / rate) if rate > 0 and np.isfinite(top3) else 0.0
        score = 0.50 * regime_gmean + 0.20 * worst + 0.20 * overall_lift + 0.10 * top3_lift
        components = {
            "regime_pr_lift_geometric_mean": regime_gmean,
            "worst_regime_pr_lift": worst,
            "overall_pr_lift": overall_lift,
            "top3_precision_lift": top3_lift,
        }
    return {"selection_score": float(score), "overall": overall, **components}, regime_df


def horizon_positive_recall(predictions: pd.DataFrame, fraction: float = 0.03) -> pd.DataFrame:
    """Recall among positives split by first crash-hit day, using daily top-k selection."""
    frame = predictions.copy()
    frame["selected"] = False
    for _, idx in frame.groupby("date", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        k = max(1, int(np.ceil(len(idx) * fraction)))
        local_pred = frame.loc[idx, "prediction"].to_numpy(dtype=float)
        selected_local = np.argpartition(local_pred, -k)[-k:] if k < len(idx) else np.arange(len(idx))
        frame.loc[idx[selected_local], "selected"] = True
    out = []
    for day in [1, 2, 3]:
        part = frame[(frame["target"] == 1) & (frame["first_hit_day"] == day)]
        out.append({
            "first_hit_day": day,
            "positives": int(len(part)),
            "top3_recall": float(part["selected"].mean()) if len(part) else float("nan"),
        })
    return pd.DataFrame(out)
