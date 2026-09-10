from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from surge_precision_gate_v12_7h import (
    TrialConfig,
    add_discovery_ticker_prior,
    build_prediction_frame,
    config_for_family,
    dev_summary,
    family_sort_key,
    fit_trial_model,
    predict_trial_model,
    rank_hard_fp_features,
    stratified_bootstrap_indices, fast_select_frozen_dev_threshold,
)


def make_meta(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(8):
        for ticker in ["000001", "000002", "000003"]:
            for i in range(120):
                f1 = rng.normal()
                f2 = rng.normal()
                base = 1 / (1 + np.exp(-(-1.7 + 0.55 * f1 + rng.normal(scale=0.7))))
                y = int(rng.random() < 1 / (1 + np.exp(-(-2.0 + 0.8*f1 - 0.5*f2 + 1.0*base))))
                rows.append({
                    "ticker": ticker, "fold_id": fold, "row_index": len(rows), "source_row_id": len(rows),
                    "target": y, "base_score_raw": base, "base_hist_rank": base,
                    "base_logit": np.log(np.clip(base,1e-6,1-1e-6)/np.clip(1-base,1e-6,1)),
                    "ab_evidence_count": 2.0, "ab_weighted_mean": f1, "ab_mean": f1,
                    "ab_min": min(f1,f2), "ab_max": max(f1,f2), "ab_std": abs(f1-f2)/2,
                    "ab_favorable_fraction": float(f1>0), "ab_adverse_fraction": float(f1<0),
                    "ab_strong_adverse_fraction": float(f1<-1.5),
                    "ab_slot_01": f1, "ab_slot_present_01": 1.0,
                    "raw_good": f1, "raw_bad": rng.normal(), "raw_inv": -f1 + rng.normal(scale=0.2),
                })
    df = pd.DataFrame(rows)
    d = df[df.fold_id <= 2].reset_index(drop=True)
    e = df[df.fold_id >= 3].reset_index(drop=True)
    d, e = add_discovery_ticker_prior(d, e)
    # convert synthetic base score to a discovery-only reference percentile
    ref = np.sort(d.base_score_raw.to_numpy())
    for frame in (d,e):
        frame["base_hist_rank"] = np.searchsorted(ref, frame.base_score_raw.to_numpy(), side="right")/len(ref)
    return d,e


class V127HTests(unittest.TestCase):
    def test_config_is_deterministic(self):
        self.assertEqual(config_for_family(12, 99), config_for_family(12, 99))
        self.assertNotEqual(config_for_family(12, 99), config_for_family(13, 99))

    def test_hard_fp_ranking_finds_signal(self):
        d,_ = make_meta()
        r = rank_hard_fp_features(d, ["raw_good","raw_bad","raw_inv"], candidate_quantile=0.4)
        self.assertIn(r.iloc[0].feature, {"raw_good","raw_inv"})
        self.assertGreater(r.iloc[0].score, 0)

    def test_ticker_prior_discovery_only(self):
        d,e = make_meta()
        self.assertTrue(np.isfinite(d.ticker_prior_logit).all())
        self.assertTrue(np.isfinite(e.ticker_prior_logit).all())

    def test_logit_fit_predict(self):
        d,e = make_meta()
        cfg = config_for_family(0, 1)
        cfg = TrialConfig(**{**vars(cfg), "backend":"logit", "feature_k":2, "candidate_quantile":0.4})
        b = fit_trial_model(d, ["raw_good","raw_inv"], cfg, seed=1, cpu_threads=2, gpu_available=False)
        p = predict_trial_model(b, e)
        self.assertEqual(len(p), len(e))
        self.assertTrue(((p>0)&(p<1)).all())

    def test_prediction_and_dev_policy(self):
        d,e = make_meta()
        cfg = config_for_family(0, 1)
        cfg = TrialConfig(**{**vars(cfg), "backend":"logit", "feature_k":2, "candidate_quantile":0.4})
        b = fit_trial_model(d, ["raw_good","raw_inv"], cfg, seed=2, cpu_threads=2, gpu_available=False)
        p = predict_trial_model(b, e)
        pred = build_prediction_frame(e,p,cfg,minimum_evidence_count=1)
        sm = dev_summary(pred,dev_folds=[3,4],minimum_alerts=10,target_precision=0.5)
        self.assertIn("threshold", sm)
        self.assertIn("mean_dev_pr_auc_delta", sm)

    def test_bootstrap_is_stratified_size_preserving(self):
        d,_ = make_meta()
        idx = stratified_bootstrap_indices(d, np.random.default_rng(7))
        self.assertEqual(len(idx), len(d))


    def test_fast_threshold_matches_original(self):
        from surge_precision_gate_v12 import select_frozen_dev_threshold
        d,e = make_meta()
        cfg = config_for_family(0, 1)
        cfg = TrialConfig(**{**vars(cfg), "backend":"logit", "feature_k":2, "candidate_quantile":0.4})
        b = fit_trial_model(d, ["raw_good","raw_inv"], cfg, seed=3, cpu_threads=2, gpu_available=False)
        pred = build_prediction_frame(e, predict_trial_model(b,e), cfg, minimum_evidence_count=1)
        fast = fast_select_frozen_dev_threshold(pred, dev_folds=[3,4], minimum_alerts=10, target_precision=0.5)
        old = select_frozen_dev_threshold(pred.rename(columns={"score":"v12_score"}), dev_folds=[3,4], minimum_alerts=10, target_precision=0.5)
        self.assertEqual(fast["safe"], old["safe"])
        self.assertAlmostEqual(fast["threshold"], old["threshold"], places=12)

    def test_sort_key_prefers_safe(self):
        a={"dev_safe":True,"min_dev_precision":0.7,"mean_dev_precision":0.7,"min_dev_pr_auc_delta":0,"mean_dev_pr_auc_delta":0,"feature_k":64}
        b={"dev_safe":False,"min_dev_precision":0.9,"mean_dev_precision":0.9,"min_dev_pr_auc_delta":1,"mean_dev_pr_auc_delta":1,"feature_k":1}
        self.assertGreater(family_sort_key(a),family_sort_key(b))


if __name__ == "__main__":
    unittest.main()
