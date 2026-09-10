from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ..io_utils import stable_hash


def add_daily_alerts(predictions: pd.DataFrame, top_fraction: float = 0.03, prediction_col: str = "prediction") -> pd.DataFrame:
    out = predictions.copy()
    if out.empty:
        out["alert"] = pd.Series(dtype="int8")
        return out
    group = out.groupby("date", sort=False)
    ranks = group[prediction_col].rank(method="first", ascending=False)
    counts = np.ceil(group[prediction_col].transform("size") * top_fraction).clip(lower=1)
    out["alert"] = ranks.le(counts).astype("int8")
    return out


def _top_precision(block: pd.DataFrame, fraction: float) -> float:
    ranked = add_daily_alerts(block.drop(columns="alert", errors="ignore"), fraction)
    selected = ranked.loc[ranked["alert"].eq(1)]
    return float(selected["target"].mean()) if len(selected) else np.nan


def calibration_errors(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)
    weighted, maximum = 0.0, 0.0
    for bucket in range(bins):
        mask = assignments == bucket
        if not mask.any():
            continue
        gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
        weighted += gap * float(mask.mean())
        maximum = max(maximum, gap)
    return weighted, maximum


def safe_metrics(block: pd.DataFrame) -> dict[str, float | str]:
    names = [
        "rows", "positives", "positive_rate", "pr_auc", "roc_auc", "brier", "logloss",
        "mean_prediction", "alert_count", "alert_precision", "alert_recall", "annualized_alerts",
        "top_1_percent_precision", "top_3_percent_precision", "top_5_percent_precision",
        "uncalibrated_brier", "calibrated_brier", "uncalibrated_logloss", "calibrated_logloss",
        "expected_calibration_error", "maximum_calibration_error", "validation_row_hash", "metric_status",
    ]
    if block.empty:
        out = {name: np.nan for name in names}
        out.update({
            "rows": 0, "positives": 0, "alert_count": 0,
            "validation_row_hash": stable_hash([]), "metric_status": "skipped_no_rows",
        })
        return out
    y = pd.to_numeric(block["target"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(block["prediction"], errors="coerce").clip(1e-7, 1 - 1e-7).to_numpy()
    raw_p = pd.to_numeric(block.get("prediction_uncalibrated", block["prediction"]), errors="coerce").clip(1e-7, 1 - 1e-7).to_numpy()
    alerts = pd.to_numeric(block["alert"], errors="coerce").fillna(0).astype(int).to_numpy()
    positives = int(y.sum())
    alert_count = int(alerts.sum())
    true_alerts = int(((alerts == 1) & (y == 1)).sum())
    unique = np.unique(y)
    ece, mce = calibration_errors(y, p)
    row_ids = sorted(block.get("row_id", block["ticker"].astype(str) + "|" + block["date"].astype(str)).astype(str))
    return {
        "rows": int(len(block)), "positives": positives, "positive_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, p)) if len(unique) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(unique) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])),
        "mean_prediction": float(np.mean(p)), "alert_count": alert_count,
        "alert_precision": true_alerts / alert_count if alert_count else np.nan,
        "alert_recall": true_alerts / positives if positives else np.nan,
        "annualized_alerts": alert_count / max(1, block["date"].nunique()) * 250.0,
        "top_1_percent_precision": _top_precision(block, 0.01),
        "top_3_percent_precision": _top_precision(block, 0.03),
        "top_5_percent_precision": _top_precision(block, 0.05),
        "uncalibrated_brier": float(brier_score_loss(y, raw_p)), "calibrated_brier": float(brier_score_loss(y, p)),
        "uncalibrated_logloss": float(log_loss(y, raw_p, labels=[0, 1])),
        "calibrated_logloss": float(log_loss(y, p, labels=[0, 1])),
        "expected_calibration_error": ece, "maximum_calibration_error": mce,
        "validation_row_hash": stable_hash(row_ids),
        "metric_status": "ok" if len(unique) == 2 else "single_class_auc_nan",
    }


def evaluate_scopes(
    predictions: pd.DataFrame, baskets: pd.DataFrame, experiment: str, fold: int, seed: int,
    target_bucket: str | None = None, target_ticker: str | None = None,
) -> pd.DataFrame:
    pred = add_daily_alerts(predictions)
    rows: list[dict] = []

    def add(scope_type: str, scope_value: str, block: pd.DataFrame) -> None:
        rows.append({"experiment": experiment, "fold": fold, "seed": seed,
                     "scope_type": scope_type, "scope_value": scope_value, **safe_metrics(block)})

    add("all", "all_validation", pred)
    sentinel = pred.loc[pred["ticker"].isin(baskets["ticker"])]
    add("sentinel", "sentinel_pooled", sentinel)
    for bucket, tickers in baskets.groupby("bucket")["ticker"]:
        add("bucket", str(bucket), pred.loc[pred["ticker"].isin(tickers)])
    for ticker in baskets["ticker"]:
        add("ticker", ticker, pred.loc[pred["ticker"].eq(ticker)])
    return pd.DataFrame(rows)
