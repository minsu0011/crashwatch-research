from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

REQ=[
 'RUN_STATUS.json','DATA_AUDIT_V14.json','FINAL_RECOMMENDATION_V14.json','V14_CHAMPION_CONFIG.json',
 'V14_CONFIG_SCREENING.csv','V14_DEVELOPMENT_CONFIG_VALIDATION.csv',
 'V14_DIRECT_FEATURE_RANKING_CROSSFIT.csv','V14_DIRECTION_FEATURE_RANKING_CROSSFIT.csv','V14_DOWN_FEATURE_RANKING_CROSSFIT.csv',
 'V14_DIRECT_DEDUP_DECISIONS.csv','V14_DIRECTION_DEDUP_DECISIONS.csv','V14_DOWN_DEDUP_DECISIONS.csv',
 'v14_oof_predictions.csv.gz','v14_frozen_policy_by_fold.csv','v14_oracle_topk_by_fold.csv','v14_expert_metrics_by_fold.csv'
]
REAL_ONLY_REQ=[
 'v14_expert_tail_diagnostics.csv','v14_expert_pairwise_spearman.csv',
 'v14_hardfp_slice_diagnostics.csv','v14_ensemble_size_stability.csv','EXPERIMENT_BUDGET_V14.json'
]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--allow-synthetic',action='store_true'); a=p.parse_args()
 root=Path(a.output); checks=[]
 def check(name,ok,detail=None): checks.append({'check':name,'passed':bool(ok),'detail':detail})
 for f in REQ: check('file_exists:'+f,(root/f).exists(),str(root/f))
 if not a.allow_synthetic:
  for f in REAL_ONLY_REQ: check('file_exists:'+f,(root/f).exists(),str(root/f))
 if not all(x['passed'] for x in checks):
  print(json.dumps({'status':'FAIL','checks':checks},indent=2)); return 1
 status=json.loads((root/'RUN_STATUS.json').read_text(encoding='utf8'))
 audit=json.loads((root/'DATA_AUDIT_V14.json').read_text(encoding='utf8'))
 rec=json.loads((root/'FINAL_RECOMMENDATION_V14.json').read_text(encoding='utf8'))
 champ=json.loads((root/'V14_CHAMPION_CONFIG.json').read_text(encoding='utf8'))
 check('run_success',status.get('status')=='SUCCESS',status.get('status'))
 if not a.allow_synthetic:
  check('rows_91775',int(audit.get('rows_full_target_valid',-1))==91775,audit.get('rows_full_target_valid'))
  check('tickers_48',int(audit.get('tickers',-1))==48,audit.get('tickers'))
  check('raw_features_439',int(audit.get('raw_features',-1))==439,audit.get('raw_features'))
 check('v11_self_not_primary',audit.get('v13_v11_self_primary') is False,audit.get('v13_v11_self_primary'))
 check('v11_lag0_hardfp_only',audit.get('v11_lag0_hardfp_only') is True,audit.get('v11_lag0_hardfp_only'))
 if a.allow_synthetic:
  check('stable_direct_nontrivial',int(audit.get('stable_direct_pool',0))>=1,audit.get('stable_direct_pool'))
  check('stable_direction_nontrivial',int(audit.get('stable_direction_pool',0))>=1,audit.get('stable_direction_pool'))
  check('stable_down_nontrivial',int(audit.get('stable_down_pool',0))>=1,audit.get('stable_down_pool'))
 else:
  check('stable_direct_nontrivial',int(audit.get('stable_direct_pool',0))>=16,audit.get('stable_direct_pool'))
  check('stable_direction_nontrivial',int(audit.get('stable_direction_pool',0))>=8,audit.get('stable_direction_pool'))
  check('stable_down_nontrivial',int(audit.get('stable_down_pool',0))>=16,audit.get('stable_down_pool'))
 for name in ['V14_DIRECT_FEATURE_RANKING_CROSSFIT.csv','V14_DIRECTION_FEATURE_RANKING_CROSSFIT.csv','V14_DOWN_FEATURE_RANKING_CROSSFIT.csv']:
  df=pd.read_csv(root/name)
  check(name+':crossfit_columns',{'minimum_fold_coverage','median_holdout_auc','worst_holdout_auc','crossfit_score'}.issubset(df.columns),list(df.columns))
  if not df.empty:
   check(name+':coverage_gate',(pd.to_numeric(df.minimum_fold_coverage,errors='coerce')>=0.39).all(),float(pd.to_numeric(df.minimum_fold_coverage,errors='coerce').min()))
 pred=pd.read_csv(root/'v14_oof_predictions.csv.gz',dtype={'ticker':str})
 required={'source_row_id','fold_id','ticker','policy_score','meta_probability','p_direct','p_hazard_up','p_down','p_hardfp','v13_stage_probability','v13_p_move','v13_p_up_given_move'}
 check('prediction_columns',required.issubset(pred.columns),sorted(required-set(pred.columns)))
 check('prediction_folds',sorted(pred.fold_id.astype(int).unique().tolist())==list(range(8)),sorted(pred.fold_id.astype(int).unique().tolist()))
 check('prediction_unique',not pred.duplicated(['source_row_id','fold_id']).any(),int(pred.duplicated(['source_row_id','fold_id']).sum()))
 for c in ['policy_score','meta_probability','p_direct','p_hazard_up','p_down','p_hardfp']:
  x=pd.to_numeric(pred[c],errors='coerce')
  check('finite:'+c,x.notna().all() and np.isfinite(x).all(),int((~np.isfinite(x)).sum()))
  check('bounded:'+c,((x>=-1e-9)&(x<=1+1e-9)).all(),[float(x.min()),float(x.max())])
 metrics=pd.read_csv(root/'v14_frozen_policy_by_fold.csv')
 check('policy_all_folds',sorted(metrics.fold_id.astype(int).tolist())==list(range(8)),metrics.fold_id.tolist())
 check('minimum_alert_target_not_lowered',int(rec.get('policy_target',{}).get('minimum_alerts',0))>=30,rec.get('policy_target'))
 check('precision_target_not_lowered',float(rec.get('policy_target',{}).get('precision',0))>=.70,rec.get('policy_target'))
 check('production_no_alert',rec.get('production_decision')=='NO_ALERT',rec.get('production_decision'))
 weights=champ.get('stacker_weights',{})
 check('stacker_nonnegative',bool(weights) and all(float(v)>=0 for v in weights.values()),weights)
 check('direct_expert_present','p_direct' in weights,weights)
 check('v13_product_only_expert','v13_stage_probability' in weights,weights)
 check('downside_veto_present','one_minus_p_down' in weights,weights)
 check('v11_self_primary_false',champ.get('v11_self_primary') is False,champ.get('v11_self_primary'))
 result={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','passed':sum(x['passed'] for x in checks),'total':len(checks),'checks':checks}
 (root/'VERIFIER_RESULTS_V14.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
 return 0 if result['status']=='PASS' else 1

if __name__=='__main__': sys.exit(main())
