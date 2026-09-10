from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from surge_precision_gate_v12 import (
    apply_frozen_threshold,
    build_discovery_separator_specs,
    combine_base_and_gate,
    empirical_rank,
    fit_evidence_transformer,
    fit_gate_bundle,
    predict_gate_bundle,
    select_frozen_dev_threshold,
    transform_evidence,
)


class TestV12(unittest.TestCase):
    def test_empirical_rank_is_reference_only(self) -> None:
        ref = np.array([0.1, 0.2, 0.3, 0.4])
        got = empirical_rank(ref, np.array([0.05, 0.2, 0.35, 0.9]))
        np.testing.assert_allclose(got, [0.0, 0.5, 0.75, 1.0])

    def test_discovery_specs_deduplicate_source(self) -> None:
        df = pd.DataFrame([
            {"ticker":"000001","axis":"AB","node_id":"f1","source_feature":"src","selection_direction":1,"precision_separator_score_v10_2":2.0,"precision_separator_selection_candidate":True},
            {"ticker":"000001","axis":"AB","node_id":"f1z","source_feature":"src","selection_direction":1,"precision_separator_score_v10_2":1.0,"precision_separator_selection_candidate":True},
            {"ticker":"000001","axis":"AB","node_id":"f2","source_feature":"src2","selection_direction":-1,"precision_separator_score_v10_2":1.5,"precision_separator_selection_candidate":True},
        ])
        specs = build_discovery_separator_specs(df, ["f1","f1z","f2"])
        self.assertEqual([x.node_id for x in specs["000001"]], ["f1","f2"])
        self.assertEqual(specs["000001"][1].direction, -1)

    def test_oriented_evidence_direction(self) -> None:
        df = pd.DataFrame([
            {"ticker":"000001","base_score_raw":0.4,"f_up":0.0,"f_down":10.0},
            {"ticker":"000001","base_score_raw":0.5,"f_up":1.0,"f_down":9.0},
            {"ticker":"000001","base_score_raw":0.6,"f_up":2.0,"f_down":8.0},
        ])
        mp = pd.DataFrame([
            {"ticker":"000001","axis":"AB","node_id":"f_up","source_feature":"a","selection_direction":1,"precision_separator_score_v10_2":2.0,"precision_separator_selection_candidate":True},
            {"ticker":"000001","axis":"AB","node_id":"f_down","source_feature":"b","selection_direction":-1,"precision_separator_score_v10_2":2.0,"precision_separator_selection_candidate":True},
        ])
        specs = build_discovery_separator_specs(mp, df.columns)
        tr = fit_evidence_transformer(df, specs, max_slots=2)
        out = transform_evidence(df, tr)
        # Third row is A-like on both features after orientation.
        self.assertGreater(out.iloc[2]["ab_weighted_mean"], out.iloc[0]["ab_weighted_mean"])

    def test_frozen_threshold_requires_each_dev_fold_support(self) -> None:
        rows = []
        for fold in [3,4]:
            for i in range(40):
                rows.append({"fold_id":fold,"target":1 if i < 28 else 0,"v12_score":1.0-i/100.0,"candidate_eligible":True})
        df = pd.DataFrame(rows)
        result = select_frozen_dev_threshold(df, minimum_alerts=30, target_precision=0.70)
        self.assertTrue(result["safe"])
        self.assertEqual(len(result["per_fold"]), 2)

    def test_frozen_threshold_fails_when_one_fold_lacks_support(self) -> None:
        rows = []
        for i in range(40):
            rows.append({"fold_id":3,"target":1 if i < 30 else 0,"v12_score":1.0-i/100.0,"candidate_eligible":True})
        for i in range(20):
            rows.append({"fold_id":4,"target":1,"v12_score":1.0-i/100.0,"candidate_eligible":True})
        df = pd.DataFrame(rows)
        result = select_frozen_dev_threshold(df, minimum_alerts=30, target_precision=0.70)
        self.assertFalse(result["safe"])
        self.assertTrue(math.isinf(result["threshold"]))

    def test_meta_gate_improves_synthetic_hard_fp(self) -> None:
        rng = np.random.default_rng(123)
        tickers = ["000001","000002","000003"]
        rows = []
        for fold in range(8):
            for ticker_i, ticker in enumerate(tickers):
                for i in range(60):
                    y = int(rng.random() < 0.30)
                    base = 0.18 + 0.42*y + rng.normal(0,0.10)
                    if y == 0 and rng.random() < 0.30:
                        base += 0.55  # hard false positive
                    base = float(np.clip(base,0.01,0.99))
                    sep = (2.4 if y else -1.2) + rng.normal(0,0.65) + ticker_i*0.05
                    rows.append({"ticker":ticker,"fold_id":fold,"target":y,"base_score_raw":base,"sep":sep})
        df = pd.DataFrame(rows)
        mp = pd.DataFrame([
            {"ticker":t,"axis":"AB","node_id":"sep","source_feature":"sep","selection_direction":1,"precision_separator_score_v10_2":2.0,"precision_separator_selection_candidate":True}
            for t in tickers
        ])
        specs = build_discovery_separator_specs(mp, df.columns)
        discovery = df.loc[df.fold_id.isin([0,1,2])].copy()
        evaluation = df.loc[df.fold_id.isin([3,4,5,6,7])].copy()
        tr = fit_evidence_transformer(discovery, specs, max_slots=1)
        dm = pd.concat([discovery.reset_index(drop=True), transform_evidence(discovery,tr).reset_index(drop=True)],axis=1)
        em = pd.concat([evaluation.reset_index(drop=True), transform_evidence(evaluation,tr).reset_index(drop=True)],axis=1)
        bundle = fit_gate_bundle(dm, backend="logit", candidate_quantile=0.50, max_slots=1, min_ticker_rows=20, min_ticker_class=3)
        gp = predict_gate_bundle(bundle, em)
        pred = em[["ticker","fold_id","target","base_score_raw"]].copy()
        pred["candidate_eligible"] = gp["candidate_eligible"].to_numpy(bool)
        pred["v12_score"] = combine_base_and_gate(pred["base_score_raw"], gp["gate_prob"], 1.0)
        policy = select_frozen_dev_threshold(pred, minimum_alerts=30, target_precision=0.70)
        self.assertTrue(policy["safe"])
        forward = apply_frozen_threshold(pred, policy["threshold"], folds=[5,6,7])
        self.assertTrue((forward["alerts"] >= 30).all())
        self.assertGreater(float(forward["precision"].mean()), 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
