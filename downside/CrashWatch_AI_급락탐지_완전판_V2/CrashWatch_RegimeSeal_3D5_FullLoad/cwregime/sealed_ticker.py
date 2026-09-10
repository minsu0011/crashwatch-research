from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def apply_platt_blend(raw_score: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    """Apply the already-frozen global Platt head without fitting on sealed data."""
    raw = np.asarray(raw_score, dtype=float)
    intercept = float(calibration["intercept"][0])
    coefficient = float(calibration["coefficients"][0][0])
    blend = float(calibration.get("blend", 0.5))
    linear = np.clip(intercept + coefficient * raw, -35.0, 35.0)
    platt = 1.0 / (1.0 + np.exp(-linear))
    return np.clip((1.0 - blend) * raw + blend * platt, 1e-6, 1.0 - 1e-6)


def add_daily_alert_flag(
    frame: pd.DataFrame,
    score_column: str,
    *,
    fraction: float = 0.03,
    output_column: str = "is_top3_alert",
) -> pd.DataFrame:
    """Flag daily universe-wide Top-N scores, with at least one alert per day."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    out = frame.copy()
    out[output_column] = False
    for _, block in out.groupby("date", sort=False):
        count = max(1, int(math.ceil(len(block) * float(fraction))))
        selected = block.nlargest(count, score_column, keep="first").index
        out.loc[selected, output_column] = True
    return out


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def scope_metrics(
    frame: pd.DataFrame,
    *,
    raw_column: str,
    risk_column: str,
    alert_column: str,
    benchmark_probability: float,
) -> dict[str, Any]:
    y = frame["target"].to_numpy(dtype=np.uint8)
    raw = np.clip(frame[raw_column].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    risk = np.clip(frame[risk_column].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    rows = int(len(frame))
    positives = int(y.sum()) if rows else 0
    negatives = rows - positives
    positive_rate = float(y.mean()) if rows else float("nan")
    two_class = bool(positives and negatives)
    pr_auc = float(average_precision_score(y, raw)) if two_class else float("nan")
    roc_auc = float(roc_auc_score(y, raw)) if two_class else float("nan")
    brier = float(brier_score_loss(y, risk)) if rows else float("nan")
    loss = float(log_loss(y, risk, labels=[0, 1])) if rows else float("nan")
    benchmark = float(np.mean((y.astype(float) - float(benchmark_probability)) ** 2)) if rows else float("nan")
    brier_skill = 1.0 - brier / benchmark if benchmark > 0.0 and np.isfinite(brier) else float("nan")
    alerts = frame[frame[alert_column].astype(bool)] if rows else frame.iloc[:0]
    alert_count = int(len(alerts))
    alert_positives = int(alerts["target"].sum()) if alert_count else 0
    alert_precision = float(alert_positives / alert_count) if alert_count else float("nan")
    alert_recall = float(alert_positives / positives) if positives else float("nan")
    alert_precision_lift = (
        float(alert_precision / positive_rate)
        if np.isfinite(alert_precision) and np.isfinite(positive_rate) and positive_rate > 0.0
        else float("nan")
    )
    dates = int(frame["date"].nunique()) if rows else 0
    return {
        "rows": rows,
        "dates": dates,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positive_rate,
        "pr_auc": pr_auc,
        "pr_auc_lift": float(pr_auc / positive_rate) if two_class and positive_rate > 0.0 else float("nan"),
        "roc_auc": roc_auc,
        "brier": brier,
        "logloss": loss,
        "benchmark_probability_development": float(benchmark_probability),
        "benchmark_brier": benchmark,
        "brier_skill_vs_development_prevalence": _safe_float(brier_skill),
        "mean_raw_score": float(raw.mean()) if rows else float("nan"),
        "mean_risk_probability": float(risk.mean()) if rows else float("nan"),
        "alert_count": alert_count,
        "alert_positives": alert_positives,
        "alert_precision": alert_precision,
        "alert_recall": alert_recall,
        "alert_precision_lift": alert_precision_lift,
        "annualized_alert_count_250d": float(alert_count / dates * 250.0) if dates else float("nan"),
    }


def evaluate_locked_p2_by_ticker(
    frame: pd.DataFrame,
    goal_policy: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    criteria = goal_policy["primary_ranking_goal"]
    support = goal_policy["minimum_evaluation_support"]
    identity = goal_policy["ticker_identity"]
    ticker_prevalence = goal_policy["development_prevalence_by_ticker"]
    tickers = sorted(identity)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        part = frame[frame["ticker"] == ticker]
        metrics = scope_metrics(
            part,
            raw_column="locked_p2_raw",
            risk_column="locked_p2_risk_probability",
            alert_column="is_top3_alert",
            benchmark_probability=float(ticker_prevalence[ticker]),
        )
        enough_support = bool(
            metrics["rows"] >= int(support["min_rows"])
            and metrics["positives"] >= int(support["min_positives"])
            and metrics["negatives"] >= int(support["min_negatives"])
        )
        pr_met = bool(
            enough_support
            and np.isfinite(metrics["pr_auc_lift"])
            and metrics["pr_auc_lift"] >= float(criteria["pr_auc_lift_min"])
        )
        roc_met = bool(
            enough_support
            and np.isfinite(metrics["roc_auc"])
            and metrics["roc_auc"] >= float(criteria["roc_auc_min"])
        )
        if enough_support:
            status = "MET" if pr_met and roc_met else "NOT_MET"
        elif metrics["rows"] < int(support["min_rows"]):
            status = "NOT_EVALUABLE_TOO_FEW_ROWS"
        elif metrics["positives"] < int(support["min_positives"]):
            status = "NOT_EVALUABLE_TOO_FEW_POSITIVES"
        else:
            status = "NOT_EVALUABLE_TOO_FEW_NEGATIVES"
        skill = metrics["brier_skill_vs_development_prevalence"]
        rows.append({
            "ticker": ticker,
            "name": identity[ticker].get("name", ""),
            "market": identity[ticker].get("market", ""),
            **metrics,
            "primary_pr_lift_goal_met": pr_met if enough_support else None,
            "primary_roc_goal_met": roc_met if enough_support else None,
            "primary_goal_status": status,
            "calibration_brier_skill_nonnegative": bool(skill >= 0.0) if np.isfinite(skill) else None,
            "sealed_scope_limit": "PARTIAL_17_EVALUABLE_DATES_EXPECTED",
        })
    table = pd.DataFrame(rows)
    global_metrics = scope_metrics(
        frame,
        raw_column="locked_p2_raw",
        risk_column="locked_p2_risk_probability",
        alert_column="is_top3_alert",
        benchmark_probability=float(goal_policy["development_global_prevalence"]),
    )
    status_counts = {str(k): int(v) for k, v in table["primary_goal_status"].value_counts().to_dict().items()}
    summary = {
        "model": "LOCKED_P2_LGB10_XGB90",
        "ranking_formula": "0.1*P2_LGB + 0.9*P2_XGB",
        "risk_formula": "0.5*raw + 0.5*frozen_global_platt(raw)",
        "goal_policy_hash": goal_policy["policy_hash"],
        "overall": global_metrics,
        "ticker_status_counts": status_counts,
        "ticker_count": int(len(table)),
        "statistical_warning": (
            "Per-ticker results have at most about 17 dates. NOT_EVALUABLE is not a failure, "
            "and MET/NOT_MET remains provisional until a longer full-regime sealed interval exists."
        ),
    }
    return table, summary
