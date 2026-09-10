from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _safe_pr_auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(average_precision_score(y, pred)) if np.unique(y).size >= 2 else float("nan")


def _safe_roc_auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred)) if np.unique(y).size >= 2 else float("nan")


def _daily_top_metrics(y: np.ndarray, pred: np.ndarray, dates_ns: np.ndarray, fraction: float) -> tuple[float, float, int, int]:
    selected_positive = 0
    selected_total = 0
    total_positive = int(np.sum(y == 1))
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
        selected_positive += int(np.sum(y[selected] == 1))
        selected_total += int(len(selected))
    precision = selected_positive / selected_total if selected_total else float("nan")
    recall = selected_positive / total_positive if total_positive else float("nan")
    return float(precision), float(recall), selected_positive, selected_total


def compute_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    dates_ns: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.float64)
    dates_ns = np.asarray(dates_ns, dtype=np.int64)
    if len(y) == 0:
        return {
            "rows": 0,
            "positives": 0,
            "positive_rate": float("nan"),
            "raw_pr_auc": float("nan"),
            "raw_roc_auc": float("nan"),
            "raw_brier": float("nan"),
            "raw_logloss": float("nan"),
            "raw_mean_prediction": float("nan"),
            "balanced_accuracy": float("nan"),
            "raw_pr_auc_lift": float("nan"),
            "top_1pct_precision": float("nan"),
            "top_1pct_recall": float("nan"),
            "top_3pct_precision": float("nan"),
            "top_3pct_recall": float("nan"),
            "top_5pct_precision": float("nan"),
            "top_5pct_recall": float("nan"),
        }
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    positive_rate = float(np.mean(y))
    pr_auc = _safe_pr_auc(y, pred)
    roc_auc = _safe_roc_auc(y, pred)
    brier = float(brier_score_loss(y, pred))
    loss = float(log_loss(y, pred, labels=[0, 1]))
    binary = (pred >= threshold).astype(np.uint8)
    balanced = float(balanced_accuracy_score(y, binary)) if np.unique(y).size >= 2 else float("nan")
    top1p, top1r, _, _ = _daily_top_metrics(y, pred, dates_ns, 0.01)
    top3p, top3r, _, _ = _daily_top_metrics(y, pred, dates_ns, 0.03)
    top5p, top5r, _, _ = _daily_top_metrics(y, pred, dates_ns, 0.05)
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "positive_rate": positive_rate,
        "raw_pr_auc": pr_auc,
        "raw_roc_auc": roc_auc,
        "raw_brier": brier,
        "raw_logloss": loss,
        "raw_mean_prediction": float(np.mean(pred)),
        "balanced_accuracy": balanced,
        "raw_pr_auc_lift": float(pr_auc / positive_rate) if positive_rate > 0 and np.isfinite(pr_auc) else float("nan"),
        "top_1pct_precision": top1p,
        "top_1pct_recall": top1r,
        "top_3pct_precision": top3p,
        "top_3pct_recall": top3r,
        "top_5pct_precision": top5p,
        "top_5pct_recall": top5r,
    }


def scoped_metric_rows(
    y: np.ndarray,
    pred: np.ndarray,
    dates_ns: np.ndarray,
    buckets: np.ndarray,
    include_battery_scope: bool,
) -> list[dict[str, Any]]:
    rows = [{"scope_type": "all", "scope_value": "all_validation", **compute_metrics(y, pred, dates_ns)}]
    if include_battery_scope:
        mask = np.asarray(buckets).astype(str) == "battery_materials"
        rows.append({
            "scope_type": "target_bucket",
            "scope_value": "battery_materials",
            **compute_metrics(y[mask], pred[mask], dates_ns[mask]),
        })
    return rows
