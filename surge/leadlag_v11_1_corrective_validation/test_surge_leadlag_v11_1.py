from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from surge_leadlag_common_v11_1 import alignment_offsets, select_frozen_portfolio_threshold
from surge_leadlag_corrective_v11_1 import (
    _estimate_maxstat_empirical_p,
    block_permutation_matrix,
    build_target_aligned_feature_frame,
    run_matched_probe,
)


class V111CorrectiveTests(unittest.TestCase):
    def test_alignment_offsets(self) -> None:
        self.assertEqual(alignment_offsets(1), [(1, 0)])
        self.assertEqual(alignment_offsets(2), [(1, -1), (2, 0)])
        self.assertEqual(alignment_offsets(3), [(1, -2), (2, -1), (3, 0)])
        self.assertEqual(alignment_offsets(4), [(1, -3), (2, -2), (3, -1)])
        self.assertEqual(alignment_offsets(5), [(1, -4), (2, -3), (3, -2)])
        for lag in range(1, 6):
            self.assertTrue(all(offset <= 0 for _, offset in alignment_offsets(lag)))

    def test_block_permutation_preserves_values(self) -> None:
        y = np.arange(17, dtype=float)
        rng = np.random.default_rng(1)
        result = block_permutation_matrix(y, 10, 4, rng)
        self.assertEqual(result.shape, (10, 17))
        for row in result:
            self.assertEqual(sorted(row.tolist()), sorted(y.tolist()))

    def test_maxstat_null_detects_obvious_signal(self) -> None:
        rng = np.random.default_rng(7)
        segments = []
        observed_values = []
        for _ in range(3):
            x = rng.normal(size=80)
            y = np.r_[rng.normal(scale=0.05, size=2), x[:-2] + rng.normal(scale=0.05, size=78)]
            segments.append((x, y))
            observed_values.append(np.corrcoef(x[:-2], y[2:])[0, 1])
        observed = abs(float(np.mean(observed_values)))
        p, n, _ = _estimate_maxstat_empirical_p(
            segments, observed, permutations=199, block_size=5, minimum=30, seed=11, batch_size=64
        )
        self.assertEqual(n, 199)
        self.assertLessEqual(p, 0.05)

    def test_target_alignment_lag5_uses_t_minus_4_to_t_minus_2(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=20)
        universe = ["000001", "000002"]
        raw = np.column_stack([np.arange(20, dtype=float) / 100.0, np.zeros(20)])
        target = pd.DataFrame({"000001": np.zeros(20), "000002": np.zeros(20)}, index=dates)
        source = pd.DataFrame({"000001": np.arange(20), "000002": np.arange(100, 120)}, index=dates)
        payload = {
            3: {
                "indices": np.arange(10, 15),
                "bucket_residual_all": raw.copy(),
            }
        }
        edges = pd.DataFrame([
            {
                "leader": "000001",
                "follower": "000002",
                "directed_lag": 5,
                "discovery_best_lag_correlation": 0.5,
            }
        ])
        features, manifest = build_target_aligned_feature_frame(edges, universe, dates, raw, target, source, payload)
        row = features.loc[features["date"].eq(dates[10])].iloc[0]
        self.assertAlmostEqual(row["ll_000001_h5_raw_d1"], raw[6, 0])
        self.assertAlmostEqual(row["ll_000001_h5_raw_d2"], raw[7, 0])
        self.assertAlmostEqual(row["ll_000001_h5_raw_d3"], raw[8, 0])
        self.assertNotIn("ll_000001_h5_raw_d4", features.columns)
        self.assertFalse(manifest["future_offsets_used"])
        self.assertTrue(manifest["edges"][0]["all_offsets_nonpositive"])

    def test_negative_edge_pressure_is_direction_flipped(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=10)
        universe = ["000001", "000002"]
        raw = np.column_stack([np.arange(10, dtype=float) / 100.0, np.zeros(10)])
        target = pd.DataFrame(0.0, index=dates, columns=universe)
        source = pd.DataFrame({"000001": np.arange(10), "000002": np.arange(100, 110)}, index=dates)
        payload = {3: {"indices": np.arange(5, 8), "bucket_residual_all": raw.copy()}}
        edges = pd.DataFrame([{
            "leader": "000001", "follower": "000002", "directed_lag": 3,
            "discovery_best_lag_correlation": -0.4,
        }])
        features, _ = build_target_aligned_feature_frame(edges, universe, dates, raw, target, source, payload)
        row = features.loc[features["date"].eq(dates[5])].iloc[0]
        expected = -np.mean(raw[[3, 4, 5], 0])
        self.assertAlmostEqual(row["ll_000001_h3_raw_pressure"], expected)

    def test_frozen_threshold_does_not_use_future_folds(self) -> None:
        rows = []
        for fold_id in [3, 4, 5]:
            for i in range(40):
                rows.append({
                    "fold_id": fold_id,
                    "score": (40 - i) / 40.0 if fold_id < 5 else i / 40.0,
                    "target": int(i < 20),
                })
        frame = pd.DataFrame(rows)
        first = select_frozen_portfolio_threshold(frame, target_precision=0.5, minimum_alerts_per_fold=5)
        mutated = frame.copy()
        mutated.loc[mutated["fold_id"].eq(5), "score"] = np.linspace(0, 1000, 40)
        second = select_frozen_portfolio_threshold(mutated, target_precision=0.5, minimum_alerts_per_fold=5)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["selection_status"], second["selection_status"])

    def test_matched_probe_base_plus_have_identical_validation_rows(self) -> None:
        rng = np.random.default_rng(10)
        rows = []
        base_rows = []
        labels = []
        sid = 0
        dates = pd.bdate_range("2025-01-01", periods=8 * 25)
        for fold_id in range(8):
            for j in range(25):
                y = int(rng.random() < 0.3)
                rows.append({
                    "source_row_id": sid,
                    "date": dates[fold_id * 25 + j],
                    "ticker": "000001",
                    "fold_id": fold_id,
                    "role": "x",
                    "ll_000002_h2_resid_pressure": rng.normal() + y * 0.3,
                })
                labels.append({"source_row_id": sid, "ticker": "000001", "label_abs_surge_3d_5pct": y})
                base_rows.append({
                    "source_row_id": sid,
                    "ticker": "000001",
                    "fold_id": fold_id,
                    "base_score_raw": float(np.clip(0.2 + 0.3 * y + rng.normal(scale=0.1), 0, 1)),
                })
                sid += 1
        # One validation row has missing base; both profiles must drop it.
        base_rows[3 * 25 + 7]["base_score_raw"] = np.nan
        features = pd.DataFrame(rows)
        frame = pd.DataFrame(labels)
        base = pd.DataFrame(base_rows)
        predictions, metrics, _, _, _, _ = run_matched_probe(
            "TEST",
            features,
            frame,
            base,
            target_precision=0.7,
            minimum_portfolio_alerts=3,
            minimum_training_rows=40,
            minimum_validation_base_rows=10,
        )
        for fold_id in [3, 4, 5, 6, 7]:
            part = predictions.loc[predictions["fold_id"].eq(fold_id)]
            base_ids = set(part.loc[part["profile"].eq("BASE_TICKER"), "source_row_id"])
            plus_ids = set(part.loc[part["profile"].eq("BASE_PLUS_ALIGNED_LEAD"), "source_row_id"])
            if plus_ids:
                self.assertEqual(base_ids, plus_ids)
        fold3_base = metrics.loc[
            metrics["fold_id"].eq(3) & metrics["profile"].eq("BASE_TICKER") & metrics["status"].eq("OK")
        ].iloc[0]
        self.assertEqual(int(fold3_base["rows"]), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
