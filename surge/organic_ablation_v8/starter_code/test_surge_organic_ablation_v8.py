from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from surge_organic_common_v8 import (
    ERROR_A_TOP_TP,
    ERROR_B_TOP_FP,
    ERROR_C_LOW_FN,
    ERROR_D_LOW_TN,
    build_cluster_synergy,
    build_error_group_codes,
    build_organic_conditions,
    build_organic_feature_map,
    build_pair_nonadditivity,
    classify_organic_role,
    datewise_percentile_rank,
    error_strata_metrics,
    hierarchical_clusters,
    selective_precision_metrics,
)
from surge_ablation_common import leakage_reason, verify_feature_names


class OrganicAblationCoreTests(unittest.TestCase):
    def test_audited_historical_lead_features_are_not_false_positive_leaks(self) -> None:
        audited = [
            "t_network_market_lead_beta_60",
            "t_network_peer_lead_beta_60",
            "t_peer_lead_lag_1",
            "t_peer_lead_lag_3",
        ]
        verify_feature_names(audited, audited)
        self.assertTrue(all(leakage_reason(feature) is None for feature in audited))
        self.assertEqual(leakage_reason("future_lead_return"), "name_contains:future")

    def test_error_groups_have_abcd(self) -> None:
        dates = np.repeat(np.arange(2), 10)
        score = np.tile(np.linspace(0.0, 1.0, 10), 2)
        y = np.array([1,0,1,0,1,0,0,0,1,1] * 2, dtype=np.uint8)
        codes, ranks = build_error_group_codes(y, score, dates, top_quantile=0.8, low_quantile=0.5)
        self.assertTrue(np.any(codes == ERROR_A_TOP_TP))
        self.assertTrue(np.any(codes == ERROR_B_TOP_FP))
        self.assertTrue(np.any(codes == ERROR_C_LOW_FN))
        self.assertTrue(np.any(codes == ERROR_D_LOW_TN))
        self.assertTrue(np.isfinite(ranks).all())

    def test_error_metrics_reward_ab_separation(self) -> None:
        codes = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], dtype=np.int8)
        good = np.array([.95,.9,.85,.8,.4,.3,.2,.1,.95,.85,.75,.7,.4,.3,.2,.1])
        bad = 1.0 - good
        mg = error_strata_metrics(codes, good)
        mb = error_strata_metrics(codes, bad)
        self.assertGreater(mg["ab_auc"], mb["ab_auc"])
        self.assertGreater(mg["cd_auc"], mb["cd_auc"])

    def test_selective_precision_frontier(self) -> None:
        y=np.array([1,1,1,0,0,0,1,0,0,0],dtype=np.int8)
        s=np.array([.99,.98,.97,.5,.4,.3,.2,.1,.05,.01])
        m=selective_precision_metrics(y,s,target_precision=.70,minimum_alerts=3)
        self.assertTrue(m["precision_target_reachable"])
        self.assertGreaterEqual(m["best_precision_min_alerts"],.70)
        self.assertGreater(m["max_recall_at_precision_target"],0)

    def test_average_linkage_clusters(self) -> None:
        features = ["a","b","c","d"]
        values = np.array([
            [1,.95,.1,.1],
            [.95,1,.1,.1],
            [.1,.1,1,.94],
            [.1,.1,.94,1],
        ])
        corr = pd.DataFrame(values, index=features, columns=features)
        clusters = hierarchical_clusters(corr, [0.92])
        sizes = sorted(clusters["cluster_size"].tolist())
        self.assertEqual(sizes, [2,2,2,2])

    def test_plan_covers_every_feature_single_and_neighborhood(self) -> None:
        features = [f"f{i}" for i in range(6)]
        values = np.eye(6)
        values[0,1]=values[1,0]=.96
        values[2,3]=values[3,2]=.93
        values[4,5]=values[5,4]=.85
        corr = pd.DataFrame(values, index=features, columns=features)
        audit = pd.DataFrame({"feature":features,"group":["g1","g1","g2","g2","g3","g3"]})
        conditions, _, pairs, _ = build_organic_conditions(
            features, corr, audit,
            cluster_thresholds=[.8,.92,.95], primary_cluster_threshold=.92,
            pair_threshold=.92, neighborhood_ks=[1,3],
        )
        single = [c for c in conditions if c.test_type == "single_feature_loo"]
        neigh = [c for c in conditions if c.test_type == "correlation_neighborhood_loo"]
        self.assertEqual(len(single), 6)
        self.assertEqual(len(neigh), 12)
        self.assertEqual(len(pairs), 2)

    def test_cluster_solo_means_keep_feature_drop_peers(self) -> None:
        features=["a","b","c"]
        corr=pd.DataFrame([[1,.95,.1],[.95,1,.1],[.1,.1,1]],index=features,columns=features)
        audit=pd.DataFrame({"feature":features,"group":["g","g","h"]})
        conditions, _, _, _ = build_organic_conditions(
            features,corr,audit,cluster_thresholds=[.92],primary_cluster_threshold=.92,
            pair_threshold=.92,neighborhood_ks=[],include_pair=False,include_groups=False,include_neighborhoods=False,
        )
        solo = next(c for c in conditions if c.test_type=="cluster_solo_keep_feature" and c.representative_feature=="a")
        self.assertEqual(solo.dropped_features,("b",))

    def test_role_redundant_backup(self) -> None:
        role=classify_organic_role(0.0,0.01,0.009,0.01,epsilon=.0005)
        self.assertEqual(role,"REDUNDANT_BACKUP_CORE")

    def test_datewise_rank_is_scale_invariant(self) -> None:
        dates=np.array([1,1,1,2,2,2])
        a=np.array([1,2,3,10,20,30],float)
        b=100*a+7
        np.testing.assert_allclose(datewise_percentile_rank(a,dates),datewise_percentile_rank(b,dates))

    def test_pair_nonadditivity_formula(self) -> None:
        summary=pd.DataFrame([
            {"backend":"b","test_type":"single_feature_loo","condition_id":"LOO::a","selection_mean_pr_auc_utility":.001},
            {"backend":"b","test_type":"single_feature_loo","condition_id":"LOO::b","selection_mean_pr_auc_utility":.001},
            {"backend":"b","test_type":"correlated_pair_loo","condition_id":"PAIR_LOO::a::b","selection_mean_pr_auc_utility":.01},
        ])
        plan=pd.DataFrame([{"test_type":"correlated_pair_loo","condition_id":"PAIR_LOO::a::b"}])
        pairs=pd.DataFrame([{"feature_a":"a","feature_b":"b","combined_abs_corr":.95}])
        result=build_pair_nonadditivity(summary,plan,pairs)
        self.assertAlmostEqual(float(result.iloc[0]["pair_nonadditivity"]),.008)
        self.assertGreater(float(result.iloc[0]["joint_unmasking_over_best_single"]),0)

    def test_cluster_hidden_value_formula(self) -> None:
        summary=pd.DataFrame([
            {"backend":"b","test_type":"single_feature_loo","condition_id":"LOO::a","selection_mean_pr_auc_utility":.0},
            {"backend":"b","test_type":"single_feature_loo","condition_id":"LOO::b","selection_mean_pr_auc_utility":.0},
            {"backend":"b","test_type":"correlation_cluster_loo","condition_id":"CL","selection_mean_pr_auc_utility":.01},
        ])
        plan=pd.DataFrame([{
            "test_type":"correlation_cluster_loo","condition_id":"CL","dropped_features":"a|b",
            "relation_threshold":.92,"relation_set_hash":"x"
        }])
        result=build_cluster_synergy(summary,plan)
        self.assertAlmostEqual(float(result.iloc[0]["hidden_cluster_value_vs_best"]),.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
