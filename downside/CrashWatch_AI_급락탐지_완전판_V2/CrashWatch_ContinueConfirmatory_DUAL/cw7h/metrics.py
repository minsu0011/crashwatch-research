from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _safe_auc(metric, y: np.ndarray, pred: np.ndarray) -> float:
    return float(metric(y, pred)) if np.unique(y).size >= 2 else float("nan")


def _daily_top(y: np.ndarray, pred: np.ndarray, dates_ns: np.ndarray, fraction: float) -> tuple[float, float]:
    selected_pos = 0
    selected_total = 0
    total_pos = int(np.sum(y == 1))
    for date in np.unique(dates_ns):
        idx = np.flatnonzero(dates_ns == date)
        if len(idx) == 0:
            continue
        k = max(1, int(math.ceil(len(idx) * fraction)))
        if k >= len(idx):
            selected = idx
        else:
            local = np.argpartition(pred[idx], -k)[-k:]
            selected = idx[local]
        selected_pos += int(np.sum(y[selected] == 1))
        selected_total += int(len(selected))
    precision = selected_pos / selected_total if selected_total else float("nan")
    recall = selected_pos / total_pos if total_pos else float("nan")
    return float(precision), float(recall)


def compute_metrics(y: np.ndarray, pred: np.ndarray, dates_ns: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.float64)
    dates_ns = np.asarray(dates_ns, dtype=np.int64)
    if len(y) == 0:
        return {k: float("nan") for k in [
            "raw_pr_auc", "raw_roc_auc", "raw_brier", "raw_logloss", "raw_mean_prediction",
            "raw_pr_auc_lift", "top_1pct_precision", "top_1pct_recall", "top_3pct_precision",
            "top_3pct_recall", "top_5pct_precision", "top_5pct_recall"
        ]} | {"rows": 0, "positives": 0, "positive_rate": float("nan")}
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    rate = float(np.mean(y))
    pr = _safe_auc(average_precision_score, y, pred)
    roc = _safe_auc(roc_auc_score, y, pred)
    p1, r1 = _daily_top(y, pred, dates_ns, 0.01)
    p3, r3 = _daily_top(y, pred, dates_ns, 0.03)
    p5, r5 = _daily_top(y, pred, dates_ns, 0.05)
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "positive_rate": rate,
        "raw_pr_auc": pr,
        "raw_roc_auc": roc,
        "raw_brier": float(brier_score_loss(y, pred)),
        "raw_logloss": float(log_loss(y, pred, labels=[0, 1])),
        "raw_mean_prediction": float(np.mean(pred)),
        "raw_pr_auc_lift": float(pr / rate) if rate > 0 and np.isfinite(pr) else float("nan"),
        "top_1pct_precision": p1,
        "top_1pct_recall": r1,
        "top_3pct_precision": p3,
        "top_3pct_recall": r3,
        "top_5pct_precision": p5,
        "top_5pct_recall": r5,
    }
