from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_surge_hardfp_v7 import scope_level_mask
from surge_hardfp_common_v7 import (
    ErrorGroupConfig,
    RelativeFeatureSpec,
    apply_rule,
    assign_error_groups,
    build_output_inventory,
    build_pairwise_training_frame,
    build_relative_feature_frame,
    contrast_feature_stats,
    fit_lgb_ranker,
    fit_xgb_ranker,
    predict_ranker,
    event_structure,
    evaluate_alert_mask,
    search_precision_cascade,
    select_error_features,
    select_precision_threshold,
    summarize_error_map,
    wilson_lower_bound,
)


class TestHardFPV7(unittest.TestCase):
    def test_output_inventory_excludes_mutable_status_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "result.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "RUN_STATUS.json").write_text("{}", encoding="utf-8")
            (root / "VERIFICATION_REPORT_V7.json").write_text("{}", encoding="utf-8")
            inventory = build_output_inventory(root)
        paths = {item["path"] for item in inventory["files"]}
        self.assertEqual(paths, {"result.csv"})

    def test_scope_level_mask_treats_missing_as_non_member(self) -> None:
        values = pd.Series(["KOSPI", None, pd.NA, "KOSDAQ"], dtype="string")
        mask = scope_level_mask(values, "KOSPI")
        np.testing.assert_array_equal(mask, np.array([True, False, False, False]))

    def test_event_structure_balances_positive_runs(self) -> None:
        y = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
        ticker = ["A"] * len(y)
        dates = pd.date_range("2026-01-01", periods=len(y), freq="D")
        out = event_structure(y, ticker, dates)
        self.assertEqual(int(out["event_start"].sum()), 2)
        self.assertAlmostEqual(float(out.loc[1:3, "event_weight"].sum()), 1.0)
        self.assertAlmostEqual(float(out.loc[5:6, "event_weight"].sum()), 1.0)

    def test_error_group_definition(self) -> None:
        dates = np.repeat(pd.date_range("2026-01-01", periods=2), 10)
        score = np.tile(np.arange(10), 2)
        y = np.zeros(20, dtype=np.uint8)
        y[[9, 10]] = 1
        groups = assign_error_groups(y, score, dates, ErrorGroupConfig(0.8, 0.5, 1))
        self.assertEqual(groups.loc[9, "error_group"], "A_TOP_TRUE_POSITIVE")
        self.assertEqual(groups.loc[10, "error_group"], "C_LOW_MISSED_POSITIVE")
        self.assertIn("B_TOP_FALSE_POSITIVE", set(groups["error_group"]))

    def test_contrast_map_finds_separator(self) -> None:
        rng = np.random.default_rng(7)
        n = 160
        groups = np.array(["A_TOP_TRUE_POSITIVE"] * 80 + ["B_TOP_FALSE_POSITIVE"] * 80, dtype=object)
        raw_good = np.r_[rng.normal(2.0, 0.4, 80), rng.normal(-1.0, 0.4, 80)]
        raw_bad = rng.normal(0, 1, n)
        rank_good = pd.Series(raw_good).rank(pct=True).to_numpy()
        rank_bad = pd.Series(raw_bad).rank(pct=True).to_numpy()
        rows = pd.DataFrame([
            contrast_feature_stats("good", raw_good, rank_good, groups, "A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE", 0, "A_VS_B"),
            contrast_feature_stats("bad", raw_bad, rank_bad, groups, "A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE", 0, "A_VS_B"),
        ])
        summary = summarize_error_map(rows)
        selected = select_error_features(summary, 1, 0.0)
        self.assertEqual(selected, ["good"])

    def test_relative_features_are_cross_sectional(self) -> None:
        matrix = np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]], dtype=np.float32)
        meta = pd.DataFrame({
            "date": np.repeat(pd.to_datetime(["2026-01-01", "2026-01-02"]), 3),
            "market": ["KOSPI"] * 6,
            "bucket": ["A"] * 6,
        })
        out = build_relative_feature_frame(matrix, ["f"], meta, RelativeFeatureSpec(("f",)))
        self.assertAlmostEqual(float(out.loc[2, "date_rank__f"]), 1.0)
        self.assertAlmostEqual(float(out.loc[5, "date_rank__f"]), 1.0)
        self.assertAlmostEqual(float(out.loc[0, "market_delta__f"]), -1.0)

    def test_pairwise_frame_prioritizes_missed_positive(self) -> None:
        features = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8]})
        groups = ["B_TOP_FALSE_POSITIVE", "C_LOW_MISSED_POSITIVE", "B_TOP_FALSE_POSITIVE", "A_TOP_TRUE_POSITIVE"]
        dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"])
        x, y, qid, audit = build_pairwise_training_frame(features, groups, dates)
        self.assertEqual(len(x), 4)
        self.assertIn(2.0, set(y.tolist()))
        self.assertEqual(len(np.unique(qid)), 2)


    def test_lightgbm_pairwise_ranker_runs(self) -> None:
        x = np.array([[0.0], [1.0], [0.2], [1.2], [0.1], [1.1]], dtype=np.float32)
        y = np.array([0, 2, 0, 2, 0, 2], dtype=np.float32)
        qid = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        model = fit_lgb_ranker(x, y, qid, {"min_data_in_leaf": 1, "num_leaves": 3}, 20, 7, 1)
        pred = predict_ranker(model, "lightgbm", x)
        self.assertGreater(float(np.mean(pred[y > 0])), float(np.mean(pred[y == 0])))

    def test_xgboost_pairwise_ranker_runs_cpu(self) -> None:
        x = np.array([[0.0], [1.0], [0.2], [1.2], [0.1], [1.1]], dtype=np.float32)
        y = np.array([0, 2, 0, 2, 0, 2], dtype=np.float32)
        qid = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        model = fit_xgb_ranker(x, y, qid, {"max_depth": 2, "min_child_weight": 1}, 20, 7, 1, "cpu")
        pred = predict_ranker(model, "xgboost", x)
        self.assertGreater(float(np.mean(pred[y > 0])), float(np.mean(pred[y == 0])))

    def test_wilson_small_sample_is_conservative(self) -> None:
        self.assertLess(wilson_lower_bound(3, 3), 0.70)
        self.assertGreater(wilson_lower_bound(90, 100), 0.80)

    def test_no_safe_threshold_returns_no_alert(self) -> None:
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({
            "target": rng.binomial(1, 0.2, 500),
            "score": rng.random(500),
            "date": np.repeat(pd.date_range("2026-01-01", periods=50), 10),
            "event_id": -1,
        })
        rule, search, diagnostic = select_precision_threshold(frame, "target", "score", "date", "event_id", 0.70, 0.60, 30, 10, 0.03)
        self.assertFalse(rule.gate_pass)
        self.assertTrue(math.isinf(rule.score_threshold))
        self.assertFalse(apply_rule(frame, rule, "score").any())

    def test_safe_threshold_prefers_coverage(self) -> None:
        dates = np.repeat(pd.date_range("2026-01-01", periods=20), 10)
        target = np.tile([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 20)
        score = np.tile([0.99, 0.97, 0.95, 0.90, 0.85, 0.3, 0.2, 0.1, 0.05, 0.01], 20)
        frame = pd.DataFrame({"target": target, "score": score, "date": dates, "event_id": -1})
        rule, _, _ = select_precision_threshold(frame, "target", "score", "date", "event_id", 0.70, 0.60, 30, 10, 0.03)
        self.assertTrue(rule.gate_pass)
        metrics = evaluate_alert_mask(target, apply_rule(frame, rule, "score"), dates)
        self.assertGreaterEqual(metrics["precision"], 0.70)
        self.assertGreater(metrics["recall"], 0.5)

    def test_cascade_can_reject_hard_false_positives(self) -> None:
        dates = np.repeat(pd.date_range("2026-01-01", periods=30), 10)
        y = np.tile([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 30)
        base = np.tile([0.95, 0.94, 0.93, 0.92, 0.91, 0.4, 0.3, 0.2, 0.1, 0.0], 30)
        verifier = np.tile([0.95, 0.94, 0.93, 0.2, 0.1, 0.3, 0.3, 0.2, 0.1, 0.1], 30)
        pairwise = verifier.copy()
        recent = verifier.copy()
        frame = pd.DataFrame({"target": y, "score": base, "verifier_mean": verifier, "pairwise_mean": pairwise, "recent_mean": recent, "date": dates, "event_id": -1})
        rule, _, _ = search_precision_cascade(frame, "target", "score", "date", "event_id", 0.70, 0.60, 30, 10, 0.03)
        self.assertTrue(rule.gate_pass)
        metrics = evaluate_alert_mask(y, apply_rule(frame, rule, "score"), dates)
        self.assertGreaterEqual(metrics["precision"], 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
