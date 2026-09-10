from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from surge_ticker_hierarchy_v10_1 import (
    RobustHierarchyConfig,
    aggregate_hierarchy_sensitivity,
    audit_hierarchy_levels,
    auc_standard_error,
    build_adaptive_probe_eligibility,
    build_fixed_signature_similarity,
    build_hierarchy_for_strength,
    build_precision_separator_map,
    normalize_map_frames,
    _prepare_base_effect_frame,
)


class TestTickerwiseV101(unittest.TestCase):
    def test_bad_industry_is_disabled(self):
        tickers = [f"T{i:02d}" for i in range(12)]
        meta = pd.DataFrame({
            "ticker": tickers,
            "name": [f"Company{i}" for i in range(12)],
            "industry": ["UNKNOWN"] * 8 + [f"Company{i}" for i in range(4)],
            "bucket": ["A"] * 6 + ["B"] * 6,
            "market": ["KOSPI"] * 8 + ["KOSDAQ"] * 4,
        })
        audit, weights = audit_hierarchy_levels(meta, RobustHierarchyConfig())
        industry = audit.loc[audit.level.eq("industry")].iloc[0]
        self.assertFalse(bool(industry.valid))
        self.assertEqual(weights["industry"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)
        self.assertGreater(weights["bucket"], 0.0)

    def test_valid_industry_is_kept(self):
        meta = pd.DataFrame({
            "ticker": [f"T{i:02d}" for i in range(12)],
            "name": [f"Company{i}" for i in range(12)],
            "industry": ["Semi"] * 4 + ["Auto"] * 4 + ["Bio"] * 4,
            "bucket": ["Tech"] * 4 + ["Industrial"] * 4 + ["Health"] * 4,
            "market": ["KOSPI"] * 8 + ["KOSDAQ"] * 4,
        })
        audit, weights = audit_hierarchy_levels(meta, RobustHierarchyConfig())
        industry = audit.loc[audit.level.eq("industry")].iloc[0]
        self.assertTrue(bool(industry.valid))
        self.assertGreater(weights["industry"], 0.0)

    def test_auc_standard_error_decreases_with_sample(self):
        small = auc_standard_error(0.65, 10, 30)
        large = auc_standard_error(0.65, 100, 300)
        self.assertTrue(math.isfinite(small))
        self.assertLess(large, small)

    def _synthetic_maps(self):
        tickers = ["A", "B", "C", "D", "E", "F"]
        summary_rows = []
        fold_rows = []
        for ti, ticker in enumerate(tickers):
            for axis in ["TARGET", "AB", "CD"]:
                for feature in ["f1", "f2", "f3"]:
                    raw_auc = 0.62 if (ticker == "A" and feature == "f1" and axis == "AB") else 0.55
                    if ticker != "A" and feature == "f1" and axis == "AB":
                        raw_auc = 0.51
                    summary_rows.append({
                        "ticker": ticker, "axis": axis, "feature": feature, "transform": "raw",
                        "selection_direction": 1, "selection_fold_count": 3,
                        "selection_mean_fixed_auc": raw_auc, "selection_min_fixed_auc": raw_auc - 0.01,
                        "selection_mean_shrunk_auc": 0.505, "selection_direction_consistency": 1.0,
                        "selection_evidence_score": 1.0,
                        "confirmation_mean_fixed_auc": raw_auc, "recent_mean_fixed_auc": raw_auc,
                    })
                    for fold in [0, 1, 2]:
                        fold_rows.append({
                            "ticker": ticker, "axis": axis, "feature": feature, "transform": "raw", "fold_id": fold,
                            "positive_n": 20, "negative_n": 40, "valid_rows": 60,
                            "matched_concordance": raw_auc if axis in ["AB", "CD"] else np.nan,
                        })
        meta = pd.DataFrame({
            "ticker": tickers, "name": tickers, "market": ["KOSPI"] * 3 + ["KOSDAQ"] * 3,
            "bucket": ["X", "X", "Y", "Y", "Z", "Z"], "industry": ["UNKNOWN"] * 6,
        })
        return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), meta

    def test_double_shrinkage_is_removed(self):
        summary, folds, meta = self._synthetic_maps()
        base = _prepare_base_effect_frame(summary, folds, meta, [0, 1, 2])
        a = base.loc[(base.ticker.eq("A")) & (base.axis.eq("AB")) & (base.feature.eq("f1"))].iloc[0]
        self.assertAlmostEqual(float(a.ticker_signed_effect_raw), 0.12, places=8)
        # V10's pre-shrunk value would have implied only 0.005.
        self.assertGreater(float(a.ticker_signed_effect_raw), 0.1)

    def test_sensitivity_finds_ticker_heterogeneity(self):
        summary, folds, meta = self._synthetic_maps()
        config = RobustHierarchyConfig(
            prior_strength_grid=(10.0, 20.0, 40.0), minimum_specific_z=0.5, robust_specific_z=0.75,
            precision_min_selection_auc=0.57, precision_min_selection_folds=2,
            precision_min_effective_n=5.0, precision_min_matched_concordance=0.54,
        )
        audit, weights = audit_hierarchy_levels(meta, config)
        base = _prepare_base_effect_frame(summary, folds, meta, [0, 1, 2])
        parts = [build_hierarchy_for_strength(base, [0, 1, 2], s, weights, config) for s in config.prior_strength_grid]
        robust = aggregate_hierarchy_sensitivity(pd.concat(parts, ignore_index=True), config)
        row = robust.loc[(robust.ticker.eq("A")) & (robust.axis.eq("AB")) & (robust.node_id.eq("f1"))].iloc[0]
        self.assertGreater(float(row.specific_z_median), 0.5)
        self.assertTrue(bool(row.ticker_specific_robust))
        precision = build_precision_separator_map(robust, config)
        p = precision.loc[(precision.ticker.eq("A")) & (precision.node_id.eq("f1"))].iloc[0]
        self.assertTrue(bool(p.precision_separator_selection_candidate))

    def test_fixed_signature_similarity_has_real_overlap(self):
        summary, folds, meta = self._synthetic_maps()
        config = RobustHierarchyConfig(prior_strength_grid=(20.0, 40.0), similarity_feature_count=6, similarity_min_node_coverage=0.5)
        _, weights = audit_hierarchy_levels(meta, config)
        base = _prepare_base_effect_frame(summary, folds, meta, [0, 1, 2])
        long = pd.concat([build_hierarchy_for_strength(base, [0, 1, 2], s, weights, config) for s in config.prior_strength_grid], ignore_index=True)
        robust = aggregate_hierarchy_sensitivity(long, config)
        edges, clusters, matrix, overlap, signature = build_fixed_signature_similarity(robust, config)
        self.assertGreater(len(signature), 0)
        self.assertEqual(matrix.shape, (6, 6))
        self.assertTrue((overlap.to_numpy()[np.triu_indices(6, 1)] >= 0).all())
        self.assertFalse((matrix.to_numpy()[np.triu_indices(6, 1)] == 0).all())

    def test_recent_row_threshold_adapts_only_when_impossible(self):
        rows = []
        for fold, n in [(0, 60), (7, 57)]:
            for ticker in ["A", "B"]:
                rows.append({
                    "ticker": ticker, "fold_id": fold, "role": "recent_audit" if fold == 7 else "selection",
                    "validation_rows": n, "validation_positive": 5, "validation_negative": n - 5,
                })
        out = build_adaptive_probe_eligibility(pd.DataFrame(rows), configured_min_rows=60)
        sel = out.loc[out.fold_id.eq(0)]
        rec = out.loc[out.fold_id.eq(7)]
        self.assertTrue((sel.v10_1_required_validation_rows == 60).all())
        self.assertFalse(sel.v10_1_adaptive_row_threshold.any())
        self.assertTrue(rec.v10_1_adaptive_row_threshold.all())
        self.assertTrue((rec.v10_1_required_validation_rows <= 57).all())
        self.assertTrue(rec.v10_1_eligible_model.all())


if __name__ == "__main__":
    unittest.main()
