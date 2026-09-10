from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def add_daily_alerts(predictions: pd.DataFrame, top_fraction: float = 0.03) -> pd.DataFrame:
    out = predictions.copy()
    out["alert"] = 0
    for _, idx in out.groupby("date", sort=False).groups.items():
        block = out.loc[idx]
        count = max(1, int(math.ceil(len(block) * top_fraction)))
        chosen = block.nlargest(count, "prediction").index
        out.loc[chosen, "alert"] = 1
    return out


def safe_metrics(block: pd.DataFrame) -> dict[str, float]:
    if block.empty:
        return {k: np.nan for k in ["rows", "positives", "positive_rate", "pr_auc", "roc_auc", "brier", "logloss", "mean_prediction", "alert_count", "alert_precision", "alert_recall", "annualized_alerts"]}
    y = pd.to_numeric(block["target"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(block["prediction"], errors="coerce").clip(1e-7, 1 - 1e-7).to_numpy()
    alerts = pd.to_numeric(block["alert"], errors="coerce").fillna(0).astype(int).to_numpy()
    positives = int(y.sum())
    alert_count = int(alerts.sum())
    true_alerts = int(((alerts == 1) & (y == 1)).sum())
    unique = np.unique(y)
    return {
        "rows": int(len(block)), "positives": positives, "positive_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, p)) if len(unique) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(unique) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "mean_prediction": float(np.mean(p)),
        "alert_count": alert_count,
        "alert_precision": true_alerts / alert_count if alert_count else np.nan,
        "alert_recall": true_alerts / positives if positives else np.nan,
        "annualized_alerts": alert_count / max(1, block["date"].nunique()) * 250.0,
    }


def evaluate_scopes(predictions: pd.DataFrame, baskets: pd.DataFrame, experiment: str, fold: int, seed: int, target_bucket: str | None = None, target_ticker: str | None = None) -> pd.DataFrame:
    pred = add_daily_alerts(predictions)
    rows: list[dict] = []

    def add(scope_type: str, scope_value: str, block: pd.DataFrame):
        rows.append({"experiment": experiment, "fold": fold, "seed": seed, "scope_type": scope_type, "scope_value": scope_value, **safe_metrics(block)})

    add("all", "all_validation", pred)
    sentinel = pred.loc[pred["ticker"].isin(baskets["ticker"])]
    add("sentinel", "sentinel_pooled", sentinel)
    for bucket, tickers in baskets.groupby("bucket")["ticker"]:
        add("bucket", str(bucket), pred.loc[pred["ticker"].isin(tickers)])
    if target_bucket:
        add("target_bucket", target_bucket, pred.loc[pred["bucket"].eq(target_bucket)])
    if target_ticker:
        add("target_ticker", target_ticker, pred.loc[pred["ticker"].eq(target_ticker)])
    for ticker in baskets["ticker"]:
        add("ticker", ticker, pred.loc[pred["ticker"].eq(ticker)])
    return pd.DataFrame(rows)
