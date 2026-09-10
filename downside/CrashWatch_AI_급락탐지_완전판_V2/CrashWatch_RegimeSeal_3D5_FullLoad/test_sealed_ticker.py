from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cwregime.sealed_ticker import (
    add_daily_alert_flag,
    apply_platt_blend,
    evaluate_locked_p2_by_ticker,
    scope_metrics,
)


class SealedTickerTest(unittest.TestCase):
    def test_platt_blend_is_bounded_and_monotonic(self) -> None:
        raw = np.array([0.01, 0.2, 0.8, 0.99])
        calibration = {"intercept": [-1.2], "coefficients": [[0.5]], "blend": 0.5}
        result = apply_platt_blend(raw, calibration)
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))
        self.assertTrue(np.all(np.diff(result) > 0.0))

    def test_top_three_percent_keeps_minimum_and_uses_ceiling(self) -> None:
        rows = []
        for date in pd.date_range("2026-01-01", periods=2):
            for number in range(48):
                rows.append({"date": date, "ticker": f"{number:06d}", "score": number})
        result = add_daily_alert_flag(pd.DataFrame(rows), "score")
        self.assertEqual(result.groupby("date")["is_top3_alert"].sum().tolist(), [2, 2])
        one = add_daily_alert_flag(pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "ticker": "A", "score": 1.0}]), "score")
        self.assertTrue(bool(one.iloc[0]["is_top3_alert"]))

    def test_single_class_metrics_remain_nan(self) -> None:
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=12),
            "target": np.zeros(12, dtype=np.uint8),
            "raw": np.linspace(0.1, 0.2, 12),
            "risk": np.linspace(0.1, 0.2, 12),
            "alert": [True] + [False] * 11,
        })
        metrics = scope_metrics(
            frame, raw_column="raw", risk_column="risk", alert_column="alert", benchmark_probability=0.1
        )
        self.assertTrue(np.isnan(metrics["pr_auc"]))
        self.assertTrue(np.isnan(metrics["roc_auc"]))
        self.assertTrue(np.isfinite(metrics["brier"]))
        self.assertTrue(np.isfinite(metrics["logloss"]))

    def test_goal_statuses_distinguish_met_not_met_and_not_evaluable(self) -> None:
        rows = []
        dates = pd.date_range("2026-01-01", periods=12)
        for ticker in ("A", "B", "C"):
            if ticker == "C":
                target = np.zeros(12, dtype=np.uint8)
            else:
                target = np.array([0] * 8 + [1] * 4, dtype=np.uint8)
            if ticker == "A":
                score = np.array([0.1] * 8 + [0.9] * 4)
            elif ticker == "B":
                score = np.array([0.9] * 8 + [0.1] * 4)
            else:
                score = np.linspace(0.1, 0.2, 12)
            for date, y, value in zip(dates, target, score):
                rows.append({
                    "date": date, "ticker": ticker, "target": y,
                    "locked_p2_raw": value, "locked_p2_risk_probability": value,
                })
        frame = add_daily_alert_flag(pd.DataFrame(rows), "locked_p2_raw")
        policy = {
            "policy_hash": "test",
            "primary_ranking_goal": {"pr_auc_lift_min": 1.0, "roc_auc_min": 0.5},
            "minimum_evaluation_support": {"min_rows": 12, "min_positives": 2, "min_negatives": 2},
            "ticker_identity": {t: {"name": t, "market": "TEST"} for t in ("A", "B", "C")},
            "development_prevalence_by_ticker": {t: 0.2 for t in ("A", "B", "C")},
            "development_global_prevalence": 0.2,
        }
        table, _ = evaluate_locked_p2_by_ticker(frame, policy)
        statuses = table.set_index("ticker")["primary_goal_status"].to_dict()
        self.assertEqual(statuses["A"], "MET")
        self.assertEqual(statuses["B"], "NOT_MET")
        self.assertEqual(statuses["C"], "NOT_EVALUABLE_TOO_FEW_POSITIVES")


if __name__ == "__main__":
    unittest.main()
