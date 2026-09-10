from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from surge_ticker_hierarchy_v10_2 import (
    ConservativeHierarchyConfig,
    build_disjoint_peer_rings,
    combine_disjoint_prior,
    aggregate_hierarchy_sensitivity,
    build_precision_separator_map,
    build_residual_similarity,
)
from run_surge_tickerwise_validation_v10_2 import adaptive_eligibility, audit_industry_metadata, drop_invalid_industry_transforms, run_probe
import run_surge_tickerwise_correlation_map_v10 as v10
from surge_ticker_common_v10 import FoldSpec
from argparse import Namespace


class V102Tests(unittest.TestCase):
    def test_adaptive_recent_rows(self) -> None:
        df = pd.DataFrame([
            {"ticker":"A","fold_id":7,"role":"recent_audit","train_rows":1000,"train_positive":100,"train_negative":900,"validation_rows":57,"validation_positive":10,"validation_negative":47,"eligible_model":False},
            {"ticker":"B","fold_id":7,"role":"recent_audit","train_rows":1000,"train_positive":100,"train_negative":900,"validation_rows":57,"validation_positive":38,"validation_negative":19,"eligible_model":False},
        ])
        out = adaptive_eligibility(df, [7], 60, 0.85, 45, 3, 20)
        self.assertEqual(int(out.loc[out.ticker.eq("A"), "adaptive_min_validation_rows"].iloc[0]), 49)
        self.assertTrue(bool(out.loc[out.ticker.eq("A"), "eligible_model_v10_2"].iloc[0]))
        self.assertFalse(bool(out.loc[out.ticker.eq("B"), "eligible_model_v10_2"].iloc[0]))

    def test_industry_audit_rejects_company_names(self) -> None:
        frame = pd.DataFrame({
            "ticker":["A","B","C","D"],
            "name":["CorpA","CorpB","CorpC","CorpD"],
            "industry":["CorpA","CorpB","UNKNOWN","UNKNOWN"],
        })
        audit = audit_industry_metadata(frame)
        self.assertFalse(audit["valid"])

    def test_invalid_industry_transform_removed(self) -> None:
        f = pd.DataFrame({"transform":["raw","date_industry_rank","ticker_z20"],"x":[1,2,3]})
        a,b = drop_invalid_industry_transforms(f, f, False)
        self.assertNotIn("date_industry_rank", set(a["transform"]))
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)

    def _base_peer_frame(self) -> pd.DataFrame:
        rows=[]
        tickers=[("A","M1","B1",0.12,10), ("B","M1","B1",0.10,1), ("C","M1","B2",0.02,1), ("D","M2","B3",-0.03,1), ("E","M2","B3",-0.04,1)]
        for t,m,b,e,w in tickers:
            rows.append({"ticker":t,"market":m,"bucket":b,"axis":"AB","node_id":"f","source_feature":"f","transform":"raw","ticker_signed_effect_raw":e,"prior_weight":w})
        return pd.DataFrame(rows)

    def test_disjoint_rings_do_not_double_count(self) -> None:
        base=self._base_peer_frame()
        rings=build_disjoint_peer_rings(base)
        a=rings.loc[rings.ticker.eq("A")].iloc[0]
        # bucket ring contains B only; market-outer contains C; global-outer contains D,E.
        self.assertEqual(int(a["bucket_count"]),1)
        self.assertEqual(int(a["market_outer_count"]),1)
        self.assertEqual(int(a["global_outer_count"]),2)

    def test_kish_neff_reflects_weight_concentration(self) -> None:
        base=self._base_peer_frame()
        rings=build_disjoint_peer_rings(base)
        # For C, market-outer is bucket B1 containing weights 10 and 1 => n_eff < 2.
        c=rings.loc[rings.ticker.eq("C")].iloc[0]
        self.assertGreater(float(c["market_outer_neff"]),1.0)
        self.assertLess(float(c["market_outer_neff"]),2.0)

    def test_prior_heterogeneity_inflates_se(self) -> None:
        row=pd.Series({
            "bucket_effect":0.10,"bucket_se":0.01,"bucket_neff":5,
            "market_outer_effect":-0.10,"market_outer_se":0.01,"market_outer_neff":5,
            "global_outer_effect":0.0,"global_outer_se":0.01,"global_outer_neff":5,
        })
        cfg=ConservativeHierarchyConfig()
        mean,se,src,n=combine_disjoint_prior(row,cfg)
        self.assertTrue(math.isfinite(mean))
        self.assertGreater(se,0.05)
        self.assertEqual(int(n),3)

    def _sensitivity_frame(self) -> pd.DataFrame:
        rows=[]
        for strength,z in [(20,2.2),(40,2.0),(80,1.9),(120,1.8)]:
            rows.append({
                "ticker":"A","axis":"AB","node_id":"f","source_feature":"f","transform":"raw","name":"A","market":"M1","bucket":"B1","industry":"UNKNOWN",
                "prior_strength":strength,"ticker_signed_effect_raw":0.10,"peer_prior_signed_effect":0.02,"posterior_signed_effect":0.05,
                "ticker_specific_delta_raw":0.08,"ticker_specific_z":z,"ticker_map_reliability":0.30,"peer_prior_sources":"bucket+market_outer",
                "selection_direction":1,"selection_fold_count":5,"selection_mean_fixed_auc":0.65,"selection_min_fixed_auc":0.58,
                "selection_direction_consistency":1.0,"selection_mean_matched_concordance":0.62,"selection_effective_n":30,
                "confirmation_mean_fixed_auc":0.60,"confirmation_min_fixed_auc":0.55,"confirmation_direction_consistency":1.0,
                "recent_mean_fixed_auc":0.59,"recent_min_fixed_auc":0.59,"recent_direction_consistency":1.0,"selection_evidence_score":1.0,
            })
        return pd.DataFrame(rows)

    def test_conservative_z_uses_weakest_prior_strength(self) -> None:
        out=aggregate_hierarchy_sensitivity(self._sensitivity_frame(), ConservativeHierarchyConfig())
        self.assertAlmostEqual(float(out.iloc[0]["specific_z_conservative"]),1.8,places=6)
        self.assertTrue(math.isfinite(float(out.iloc[0]["specific_q_ticker_axis"])))

    def test_precision_confirmation_requires_direction_consistency(self) -> None:
        robust=aggregate_hierarchy_sensitivity(self._sensitivity_frame(), ConservativeHierarchyConfig(specific_fdr_ticker_axis=1.0,specific_fdr_global_axis=1.0))
        robust.loc[:,"confirmation_direction_consistency"]=0.5
        precision=build_precision_separator_map(robust, ConservativeHierarchyConfig(specific_fdr_ticker_axis=1.0,specific_fdr_global_axis=1.0))
        self.assertEqual(str(precision.iloc[0]["confirmation_status"]),"NOT_SUPPORTED")
        self.assertFalse(bool(precision.iloc[0]["precision_separator_confirmed"]))

    def test_recent_nan_is_untested_not_failed(self) -> None:
        robust=aggregate_hierarchy_sensitivity(self._sensitivity_frame(), ConservativeHierarchyConfig(specific_fdr_ticker_axis=1.0,specific_fdr_global_axis=1.0))
        robust.loc[:,"recent_mean_fixed_auc"]=np.nan
        precision=build_precision_separator_map(robust, ConservativeHierarchyConfig(specific_fdr_ticker_axis=1.0,specific_fdr_global_axis=1.0))
        self.assertEqual(str(precision.iloc[0]["recent_status"]),"UNTESTED")

    def test_residual_similarity_does_not_use_posterior(self) -> None:
        rows=[]
        for ticker,sign in [("A",1),("B",1),("C",-1)]:
            for k in range(25):
                rows.append({"ticker":ticker,"axis":"TARGET","node_id":f"f{k}","source_feature":f"f{k}","transform":"raw",
                             "specific_delta_raw_median":sign*(0.01+k*0.001),"posterior_signed_effect_median":0.20,
                             "reliability_median":0.5})
        robust=pd.DataFrame(rows)
        cfg=ConservativeHierarchyConfig(similarity_feature_count=25,similarity_min_coverage=1.0,similarity_min_common=10)
        sim,edges,clusters,nodes=build_residual_similarity(robust,cfg)
        self.assertGreater(float(sim.loc["A","B"]),0.9)
        self.assertLess(float(sim.loc["A","C"]),-0.9)

    def test_small_probe_execution(self) -> None:
        rng=np.random.default_rng(7)
        n=260
        f1=rng.normal(size=n); f2=rng.normal(size=n)
        y=(f1+0.3*f2+rng.normal(scale=0.7,size=n)>0.5).astype(np.int8)
        frame=pd.DataFrame({"source_row_id":np.arange(n),"ticker":"A","label_abs_surge_3d_5pct":y})
        transformed=pd.DataFrame({"source_row_id":np.arange(n),"f1__ticker_z20":f1,"f2__ticker_z20":f2})
        folds=[
            FoldSpec(3,pd.Timestamp("2020-01-01"),pd.Timestamp("2020-06-01"),pd.Timestamp("2020-06-02"),pd.Timestamp("2020-07-01")),
            FoldSpec(4,pd.Timestamp("2020-01-01"),pd.Timestamp("2020-07-01"),pd.Timestamp("2020-07-02"),pd.Timestamp("2020-08-01")),
        ]
        ticker_index={"A":{3:{"train":np.arange(0,160),"validation":np.arange(160,200)},4:{"train":np.arange(0,200),"validation":np.arange(200,240)}}}
        elig=pd.DataFrame([
            {"ticker":"A","fold_id":3,"eligible_model_v10_2":True},
            {"ticker":"A","fold_id":4,"eligible_model_v10_2":True},
        ])
        base_rows=[]
        for fid,idx in [(3,np.arange(160,200)),(4,np.arange(200,240))]:
            for i in idx:
                base_rows.append({"row_index":int(i),"base_score_raw":float(0.5+0.1*rng.normal())})
        base=pd.DataFrame(base_rows)
        profiles={"A":{"AB_CONSERVATIVE":["f1__ticker_z20","f2__ticker_z20"]}}
        v10args=v10.build_parser().parse_args([])
        v10args.target_column="label_abs_surge_3d_5pct"; v10args.ticker_column="ticker"; v10args.device="cpu"; v10args.allow_cpu_fallback=True; v10args.strict_backend=False; v10args.threads=1; v10args.xgboost_threads=1; v10args.base_iterations=20
        args=Namespace(probe_backends="lightgbm_cpu",minimum_probe_features=2,strict_backend=False,seed=17,target_precision=0.70,minimum_alerts_per_ticker=2,minimum_portfolio_alerts=4)
        preds,metrics,champions,portfolio=run_probe(frame,transformed,profiles,base,folds,ticker_index,elig,v10args,[3,4],args)
        self.assertFalse(preds.empty)
        self.assertFalse(metrics.empty)
        self.assertEqual(len(champions),1)
        self.assertFalse(portfolio.empty)


if __name__ == "__main__":
    unittest.main()
