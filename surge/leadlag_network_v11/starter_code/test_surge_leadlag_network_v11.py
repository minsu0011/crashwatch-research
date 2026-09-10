from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from surge_leadlag_common_v11 import bh_fdr, granger_incremental, lag_align, quantile_mutual_information, safe_corr
from surge_leadlag_network_v11 import classify_state, fit_apply_residual, leave_one_out_factor, summarize_lags


class LeadLagV11Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(17)

    def test_a_leads_b_by_two_days(self) -> None:
        a = self.rng.normal(size=400); b = np.r_[np.zeros(2), a[:-2]] + self.rng.normal(0, .02, 400)
        corr, _, _ = safe_corr(*lag_align(a, b, 2))
        self.assertGreater(corr, .95)

    def test_reverse_direction(self) -> None:
        b = self.rng.normal(size=400); a = np.r_[0, b[:-1]]
        positive, _, _ = safe_corr(*lag_align(a, b, 1)); negative, _, _ = safe_corr(*lag_align(a, b, -1))
        self.assertGreater(abs(negative), abs(positive))

    def test_synchronous_dominates_lag_zero(self) -> None:
        a = self.rng.normal(size=300); b = a + self.rng.normal(0, .05, 300)
        values = {lag: abs(safe_corr(*lag_align(a, b, lag))[0]) for lag in range(-3, 4)}
        self.assertEqual(max(values, key=values.get), 0)

    def test_negative_synchronous_edge(self) -> None:
        a = self.rng.normal(size=300); b = -a + self.rng.normal(0, .03, 300)
        self.assertLess(safe_corr(a, b)[0], -.95)

    def test_market_factor_removal_reduces_spurious_corr(self) -> None:
        market = self.rng.normal(size=500)
        values = np.column_stack([market + self.rng.normal(0, .3, 500) for _ in range(4)])
        factors = leave_one_out_factor(values, ["M"] * 4)
        train, test = np.arange(350), np.arange(350, 500)
        raw = safe_corr(values[test, 0], values[test, 1])[0]
        r0 = fit_apply_residual(values[:, 0], factors[:, [0]], train, test)
        r1 = fit_apply_residual(values[:, 1], factors[:, [1]], train, test)
        self.assertLess(abs(safe_corr(r0, r1)[0]), abs(raw))

    def test_rolling_regime_decoupling(self) -> None:
        self.assertEqual(classify_state(.05, 0, .05, .2, .1), "UNSTABLE")

    def test_rolling_regime_diverging(self) -> None:
        self.assertEqual(classify_state(.4, 1, .3, 2.2, 1.0), "DIVERGING")

    def test_shock_delayed_response_alignment(self) -> None:
        a = np.zeros(100); a[[10, 30, 50]] = .03; b = np.zeros(100); b[[12, 32, 52]] = .02
        x, y = lag_align(a, b, 2)
        self.assertAlmostEqual(float(y[x >= .02].mean()), .02)

    def test_self_momentum(self) -> None:
        x = np.zeros(500)
        noise = self.rng.normal(size=500)
        for index in range(1, len(x)): x[index] = .65 * x[index - 1] + noise[index]
        self.assertGreater(safe_corr(*lag_align(x, x, 1))[0], .4)

    def test_self_reversal(self) -> None:
        x = np.zeros(500); noise = self.rng.normal(size=500)
        for index in range(1, len(x)): x[index] = -.65 * x[index - 1] + noise[index]
        self.assertLess(safe_corr(*lag_align(x, x, 1))[0], -.4)

    def test_discovery_direction_is_frozen(self) -> None:
        rows = []
        for fold in range(8):
            for lag in [-1, 0, 1, 2]:
                corr = .6 if lag == 2 else .05
                if fold >= 5 and lag == 2: corr = -.2
                rows.append({"ticker_a":"A","ticker_b":"B","variant":"bucket_residual","fold_id":fold,
                             "role":"discovery" if fold <= 2 else "development" if fold <= 4 else "confirmation" if fold <= 6 else "recent_audit",
                             "lag":lag,"correlation":corr,"p_value":.001,"observations":60})
        result = summarize_lags(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result.discovery_best_lag, 2)
        self.assertTrue(result.development_same_direction)
        self.assertEqual(result.development_valid_folds, 2)
        self.assertFalse(result.confirmation_same_direction)

    def test_train_only_residual_parameters(self) -> None:
        y = self.rng.normal(size=200); factor = self.rng.normal(size=(200, 1)); train=np.arange(100); test=np.arange(100,200)
        first=fit_apply_residual(y,factor,train,test); changed=y.copy(); changed[test]+=5; second=fit_apply_residual(changed,factor,train,test)
        np.testing.assert_allclose(second-first,5,atol=1e-10)

    def test_missing_pair_data(self) -> None:
        x=self.rng.normal(size=100); y=x.copy(); y[:50]=np.nan
        corr,_,n=safe_corr(x,y,minimum=20); self.assertGreater(corr,.99); self.assertEqual(n,50)

    def test_bh_fdr(self) -> None:
        q=bh_fdr([.001,.01,.20,np.nan]); self.assertTrue(q[0] <= q[1] <= q[2]); self.assertTrue(np.isnan(q[3]))

    def test_mutual_information_nonlinear(self) -> None:
        x=self.rng.normal(size=1000); y=x*x+self.rng.normal(0,.05,1000); mi,_=quantile_mutual_information(x,y)
        self.assertGreater(mi,.3)

    def test_granger_incremental(self) -> None:
        a=self.rng.normal(size=800); b=np.zeros(800)
        for i in range(1,800): b[i]=.2*b[i-1]+.7*a[i-1]+self.rng.normal(0,.3)
        result=granger_incremental(b,a,1); self.assertGreater(result["incremental_r2"],.2); self.assertLess(result["p_value"],.001)

    def test_graphml_generation(self) -> None:
        graph=nx.DiGraph(); graph.add_edge("A","B",lag=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"x.graphml"; nx.write_graphml(graph,path); loaded=nx.read_graphml(path)
            self.assertTrue(loaded.has_edge("A","B"))


if __name__ == "__main__":
    unittest.main()
