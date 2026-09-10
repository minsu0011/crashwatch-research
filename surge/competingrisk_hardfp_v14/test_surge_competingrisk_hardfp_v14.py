from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from surge_v13_data import TARGET_COLUMN, MOVE_COLUMN
from surge_v14_models import (
    DOWN_COLUMN, FIRST_UP_DAY_COLUMN,
    add_competing_risk_labels, crossfit_rank_features, deduplicate_ranked_features,
    fit_nonnegative_stacker, predict_nonnegative_stacker,
    build_ranked_policy_score, select_threshold, policy_metrics,
)
from run_surge_competingrisk_hardfp_v14 import concatenate_expert_folds


class V14Tests(unittest.TestCase):
    def test_screening_concatenates_disjoint_folds(self):
        fold0=pd.DataFrame({'source_row_id':[10,11],'fold_id':[0,0],'score':[.1,.2]})
        fold1=pd.DataFrame({'source_row_id':[20,21,22],'fold_id':[1,1,1],'score':[.3,.4,.5]})
        out=concatenate_expert_folds([fold1,fold0])
        self.assertEqual(len(out),5)
        self.assertEqual(out['fold_id'].tolist(),[0,0,1,1,1])
        self.assertEqual(out['source_row_id'].tolist(),[10,11,20,21,22])

    def test_competing_labels(self):
        df=pd.DataFrame({
            TARGET_COLUMN:[1,0,1,0],
            'future_cumret_d1':[.06,-.01,.01,-.06],
            'future_cumret_d2':[.04,-.06,.03,-.04],
            'future_cumret_d3':[.08,-.03,.07,-.02],
        })
        out=add_competing_risk_labels(df,.05)
        self.assertEqual(out[FIRST_UP_DAY_COLUMN].tolist(),[1,0,3,0])
        self.assertEqual(out[DOWN_COLUMN].tolist(),[0,1,0,1])

    def test_official_target_controls_d3_hazard(self):
        df=pd.DataFrame({
            TARGET_COLUMN:[1],
            'future_cumret_d1':[.01],'future_cumret_d2':[.02],'future_cumret_d3':[.03],
        })
        out=add_competing_risk_labels(df,.05)
        self.assertEqual(int(out[FIRST_UP_DAY_COLUMN].iloc[0]),3)

    def test_crossfit_ranking_rejects_sparse(self):
        rng=np.random.default_rng(4)
        rows=[]
        for f in range(3):
            for i in range(200):
                y=rng.integers(0,2)
                rows.append({'fold_id':f,TARGET_COLUMN:y,'t_price_signal':y+rng.normal(0,.5),'t_price_sparse':(y+rng.normal(0,.1) if i<15 else np.nan)})
        df=pd.DataFrame(rows)
        rank=crossfit_rank_features(
            df,['t_price_signal','t_price_sparse'],target_column=TARGET_COLUMN,folds=[0,1,2],
            minimum_total_rows=200,minimum_holdout_rows=40,minimum_coverage=.5,minimum_fold_coverage=.4,minimum_worst_auc=.49,
        )
        self.assertIn('t_price_signal',rank.feature.tolist())
        self.assertNotIn('t_price_sparse',rank.feature.tolist())

    def test_crossfit_uses_heldout_direction(self):
        rng=np.random.default_rng(5)
        rows=[]
        # folds 0/1 positive association, fold2 reverse. Worst heldout gate should reject.
        for f in range(3):
            for i in range(180):
                y=rng.integers(0,2)
                x=(y if f<2 else 1-y)+rng.normal(0,.15)
                rows.append({'fold_id':f,TARGET_COLUMN:y,'t_price_unstable':x})
        df=pd.DataFrame(rows)
        rank=crossfit_rank_features(
            df,['t_price_unstable'],target_column=TARGET_COLUMN,folds=[0,1,2],
            minimum_total_rows=200,minimum_holdout_rows=40,minimum_coverage=.8,minimum_fold_coverage=.8,minimum_worst_auc=.49,
        )
        self.assertTrue(rank.empty)

    def test_dedup_removes_correlated_variant(self):
        x=np.linspace(-1,1,500)
        df=pd.DataFrame({'a':x,'a__ticker_z60':x*2+1e-8,'b':np.sin(np.arange(500))})
        ranking=pd.DataFrame({'feature':['a','a__ticker_z60','b'],'crossfit_score':[3,2,1]})
        selected,dec=deduplicate_ranked_features(df,ranking,max_features=3,corr_threshold=.95,max_per_stem=2)
        self.assertIn('a',selected); self.assertIn('b',selected)
        self.assertNotIn('a__ticker_z60',selected)

    def test_nonnegative_stacker(self):
        rng=np.random.default_rng(2); n=1000
        a=rng.random(n); b=rng.random(n); y=(a+.7*b+rng.normal(0,.2,n)>1.0).astype(int)
        df=pd.DataFrame({'a':a,'b':b,TARGET_COLUMN:y})
        m=fit_nonnegative_stacker(df,columns=['a','b'],l2=.05)
        self.assertTrue(np.all(m.weights>=0))
        p=predict_nonnegative_stacker(m,df)
        self.assertGreater(float(np.corrcoef(p,y)[0,1]),.5)

    def test_ranked_policy_score(self):
        ref=pd.DataFrame({'ticker':['000001']*40+['000002']*40,'p':np.r_[np.linspace(.1,.8,40),np.linspace(.2,.9,40)]})
        cur=pd.DataFrame({'ticker':['000001','000002'],'date':pd.to_datetime(['2026-01-01']*2),'p':[.7,.8]})
        score,hist,dr=build_ranked_policy_score(ref,cur,probability_column='p',raw_weight=.5,historical_weight=.4,date_weight=.1)
        self.assertTrue(np.all((score>=0)&(score<=1)))
        self.assertTrue(np.all(np.isfinite(hist)))

    def test_threshold_minimum_alert_contract(self):
        rows=[]
        for f in range(3):
            y=np.r_[np.ones(35),np.zeros(65)]
            s=np.r_[np.linspace(.99,.8,35),np.linspace(.7,.01,65)]
            rows.append(pd.DataFrame({'fold_id':f,TARGET_COLUMN:y,'policy_score':s}))
        df=pd.concat(rows,ignore_index=True)
        pol=select_threshold(df,folds=[0,1,2],score_column='policy_score',minimum_alerts=30,target_precision=.7)
        self.assertTrue(pol['safe'])
        met=policy_metrics(df,pol['threshold'])
        self.assertTrue((met.alerts>=30).all())
        self.assertTrue((met.precision>=.7).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
