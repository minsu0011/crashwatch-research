from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from cw7h.utils import atomic_json, bh_adjust, bootstrap_mean_ci, exact_sign_flip_p, read_json

METRICS=['raw_pr_auc','raw_roc_auc','raw_pr_auc_lift','top_3pct_precision','top_3pct_recall','raw_brier','raw_logloss']

def read_tasks(path:Path)->pd.DataFrame:
  rows=[]
  if path.exists():
    for p in path.glob('*.json'):
      d=read_json(p,{})
      if d.get('status')=='completed': rows.append(d)
  return pd.DataFrame(rows)

def add_alias_rows(df:pd.DataFrame,aliases:dict[str,str])->pd.DataFrame:
  pieces=[df]
  for alias,source in aliases.items():
    g=df[df.profile.eq(source)].copy()
    if not g.empty: g['profile']=alias; g['aliased_from']=source; pieces.append(g)
  return pd.concat(pieces,ignore_index=True) if pieces else df

def paired_summary(metrics:pd.DataFrame,base_name:str,profiles:list[str],recent_folds:list[int]):
  if metrics.empty: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
  cols=['outer_fold','seed']+METRICS
  base=metrics[metrics.profile.eq(base_name)][cols].rename(columns={m:f'base_{m}' for m in METRICS})
  paired=metrics.merge(base,on=['outer_fold','seed'],how='inner')
  for m in METRICS:
    paired[f'delta_{m}']=paired[m]-paired[f'base_{m}']
  fold=paired.groupby(['profile','outer_fold'],as_index=False)[[f'delta_{m}' for m in METRICS]].mean()
  rows=[]
  for profile,g in fold.groupby('profile'):
    vals=g.delta_raw_pr_auc.to_numpy(float); recent=g[g.outer_fold.isin(recent_folds)].delta_raw_pr_auc.to_numpy(float)
    lo,hi=bootstrap_mean_ci(vals) if len(vals) else (np.nan,np.nan)
    rows.append({'profile':profile,'fold_count':len(vals),'mean_delta':float(np.mean(vals)) if len(vals) else np.nan,'median_delta':float(np.median(vals)) if len(vals) else np.nan,'ci95_low':lo,'ci95_high':hi,'positive_fold_ratio':float(np.mean(vals>0)) if len(vals) else np.nan,'worst_fold':float(np.min(vals)) if len(vals) else np.nan,'recent_mean_delta':float(np.mean(recent)) if len(recent) else np.nan,'recent_positive_ratio':float(np.mean(recent>0)) if len(recent) else np.nan,'exact_sign_flip_p':exact_sign_flip_p(vals) if len(vals) else np.nan})
  s=pd.DataFrame(rows)
  if not s.empty: s['bh_q']=bh_adjust(s.exact_sign_flip_p.to_numpy(float))
  return paired,fold,s

def direct_compare(metrics:pd.DataFrame,a:str,b:str,recent_folds:list[int]):
  if metrics.empty: return {}
  cols=['outer_fold','seed']+METRICS
  x=metrics[metrics.profile.eq(a)][cols].rename(columns={m:f'a_{m}' for m in METRICS})
  y=metrics[metrics.profile.eq(b)][cols].rename(columns={m:f'b_{m}' for m in METRICS})
  z=x.merge(y,on=['outer_fold','seed'],how='inner')
  if z.empty: return {}
  for m in METRICS: z[f'delta_{m}']=z[f'a_{m}']-z[f'b_{m}']
  fold=z.groupby('outer_fold',as_index=False)[[f'delta_{m}' for m in METRICS]].mean(); vals=fold.delta_raw_pr_auc.to_numpy(float); recent=fold[fold.outer_fold.isin(recent_folds)].delta_raw_pr_auc.to_numpy(float); lo,hi=bootstrap_mean_ci(vals)
  return {'a':a,'b':b,'paired_models':len(z),'fold_count':len(vals),'mean_delta_raw_pr_auc':float(np.mean(vals)),'ci95_low':lo,'ci95_high':hi,'positive_fold_ratio':float(np.mean(vals>0)),'recent_mean_delta':float(np.mean(recent)) if len(recent) else None,'recent_positive_ratio':float(np.mean(recent>0)) if len(recent) else None,'mean_delta_raw_roc_auc':float(fold.delta_raw_roc_auc.mean()),'mean_delta_top3_precision':float(fold.delta_top_3pct_precision.mean()),'mean_delta_top3_recall':float(fold.delta_top_3pct_recall.mean()),'mean_delta_brier':float(fold.delta_raw_brier.mean()),'mean_delta_logloss':float(fold.delta_raw_logloss.mean())}

def aggregate(output_dir:Path,aliases:dict[str,str],cfg:dict,profile_manifest:dict):
  out=output_dir/'results'; out.mkdir(parents=True,exist_ok=True)
  lgb_raw=read_tasks(output_dir/'task_results/lightgbm'); lgb_unique_complete=len(lgb_raw); lgb=add_alias_rows(lgb_raw,aliases); xgb=read_tasks(output_dir/'task_results/xgboost')
  lgb.to_csv(out/'lightgbm_metrics.csv',index=False,encoding='utf-8-sig'); xgb.to_csv(out/'xgboost_metrics.csv',index=False,encoding='utf-8-sig')
  gpu_audit={'completed_models':len(xgb)}
  if not xgb.empty:
    for col in ['pre_gpu_util','training_avg_gpu_util','estimated_xgb_added_gpu_util','training_max_gpu_util','scheduled_cooldown_seconds','launch_wait_seconds']:
      if col in xgb.columns:
        gpu_audit[f'{col}_mean']=float(pd.to_numeric(xgb[col],errors='coerce').mean())
        gpu_audit[f'{col}_max']=float(pd.to_numeric(xgb[col],errors='coerce').max())
    if 'pubg_detected' in xgb.columns: gpu_audit['pubg_detected_model_ratio']=float(xgb['pubg_detected'].astype(bool).mean())
  atomic_json(gpu_audit,out/'GPU_THROTTLE_AUDIT.json')
  recent=cfg['selection']['recent_folds']
  lp,lf,ls=paired_summary(lgb,'C0_P1_EXACT_DEDUP',sorted(lgb.profile.unique()) if not lgb.empty else [],recent); lp.to_csv(out/'lightgbm_paired_vs_c0.csv',index=False); lf.to_csv(out/'lightgbm_fold_effects.csv',index=False); ls.to_csv(out/'lightgbm_profile_summary.csv',index=False)
  xp,xf,xs=paired_summary(xgb,'P0_FULL_439',sorted(xgb.profile.unique()) if not xgb.empty else [],recent); xp.to_csv(out/'xgboost_paired_vs_p0.csv',index=False); xf.to_csv(out/'xgboost_fold_effects.csv',index=False); xs.to_csv(out/'xgboost_profile_summary.csv',index=False)
  decomp={
    'remove_high_missing2_from_C0':direct_compare(lgb,'C1_P1_MINUS_HIGH_MISSING','C0_P1_EXACT_DEDUP',recent),
    'remove_harmful3_from_C0':direct_compare(lgb,'C2_P1_MINUS_HARMFUL3','C0_P1_EXACT_DEDUP',recent),
    'remove_all5_from_C0':direct_compare(lgb,'C3_P2_ALL5_REMOVED','C0_P1_EXACT_DEDUP',recent),
    'C1_vs_P2_restore_harmful3_group':direct_compare(lgb,'C1_P1_MINUS_HIGH_MISSING','C3_P2_ALL5_REMOVED',recent),
    'C4_vs_P2_restore_harmful3_group_alias':direct_compare(lgb,'C4_P2_RESTORE_HARMFUL3','C3_P2_ALL5_REMOVED',recent),
    'C2_vs_P2_restore_high_missing2':direct_compare(lgb,'C2_P1_MINUS_HARMFUL3','C3_P2_ALL5_REMOVED',recent),
    'restore_balance_slope_vs_P2':direct_compare(lgb,'C5_P2_RESTORE_BALANCE_SLOPE','C3_P2_ALL5_REMOVED',recent),
    'restore_volume_sum5_vs_P2':direct_compare(lgb,'C6_P2_RESTORE_VOLUME_SUM5','C3_P2_ALL5_REMOVED',recent),
    'restore_cocrash_vs_P2':direct_compare(lgb,'C7_P2_RESTORE_COCRASH_FREQ','C3_P2_ALL5_REMOVED',recent),
  }
  hm=decomp['remove_high_missing2_from_C0']; hm_after=decomp['C2_vs_P2_restore_high_missing2']; h3=decomp['remove_harmful3_from_C0']; h3_after=decomp['C1_vs_P2_restore_harmful3_group']
  # Difference-in-differences: positive means the joint removal has extra benefit beyond additive main effects.
  interaction=None
  if hm and h3 and decomp['remove_all5_from_C0']:
    interaction=float(decomp['remove_all5_from_C0']['mean_delta_raw_pr_auc']-hm['mean_delta_raw_pr_auc']-h3['mean_delta_raw_pr_auc'])
  decomp['interaction_mean_delta']=interaction
  p7p2_lgb=direct_compare(lgb,'P7_CORR095_PLUS_CONDITIONAL','C3_P2_ALL5_REMOVED',recent)
  p7p2_xgb=direct_compare(xgb,'P7_CORR095_PLUS_CONDITIONAL','P2_DEDUP_CLEAN',recent)
  core_seeds=set(cfg['xgboost']['core_seeds']); core_complete=False
  if not xgb.empty:
    q=xgb[xgb.seed.isin(core_seeds)].groupby(['profile','outer_fold']).seed.nunique()
    core_complete=(len(q)==3*8 and int(q.min())>=len(core_seeds))
  lock={'status':'NO_LOCK_REVIEW_REQUIRED','locked_profile':None,'reason':[],'lightgbm_p7_vs_p2':p7p2_lgb,'xgboost_p7_vs_p2':p7p2_xgb,'xgb_core_complete':core_complete,'sealed_evaluation_allowed':False}
  if core_complete and p7p2_lgb and p7p2_xgb:
    l=float(p7p2_lgb['mean_delta_raw_pr_auc']); x=float(p7p2_xgb['mean_delta_raw_pr_auc']); margin=float(cfg['selection']['p7_xgb_noninferiority_margin']); material=float(cfg['selection']['material_xgb_advantage'])
    if l>=0 and x>=margin and float(p7p2_lgb.get('positive_fold_ratio',0))>=0.625 and float(p7p2_lgb.get('recent_positive_ratio',0))>=0.75:
      lock.update({'status':'LOCKED','locked_profile':'P7_CORR095_PLUS_CONDITIONAL','reason':['LightGBM 안정성 우위','XGBoost 비열등 한계 통과'],'sealed_evaluation_allowed':True})
    elif l<0 and x<=-material:
      lock.update({'status':'LOCKED','locked_profile':'P2_DEDUP_CLEAN','reason':['LightGBM 평균 P2 우위','XGBoost P2 실질 우위'],'sealed_evaluation_allowed':True})
    else: lock['reason']=['P2/P7 모델 간 증거가 엇갈리거나 차이가 작음']
  else: lock['reason']=['XGBoost core 3-seed 8-fold 완주가 필요함']
  # Exact user rule: C1/C4 above P2 invalidates deletion rationale for the three harmful candidates.
  c1=decomp['C1_vs_P2_restore_harmful3_group']; c4=decomp['C4_vs_P2_restore_harmful3_group_alias']; deletion_invalid=bool((c1 and c1.get('mean_delta_raw_pr_auc',-1)>0) or (c4 and c4.get('mean_delta_raw_pr_auc',-1)>0))
  cause={'decomposition':decomp,'harmful_three_deletion_rationale_invalidated':deletion_invalid,'decision':'DO_NOT_DELETE_HARMFUL3_AS_A_GROUP' if deletion_invalid else 'GROUP_DELETION_NOT_DISPROVEN','note':'C1 and C4 are identical feature sets and share trained results.'}
  atomic_json(cause,out/'P2_CAUSAL_DECOMPOSITION.json'); atomic_json(lock,out/'LOCKED_PROFILE.json')
  summary={'lgb_unique_completed':lgb_unique_complete,'lgb_report_rows_with_aliases':len(lgb),'xgb_completed':len(xgb),'xgb_core_complete':core_complete,'p2_cause':cause,'profile_lock':lock,'protected_harmful_ten':profile_manifest['protected_harmful_ten'],'drop_label_policy':'feature_master_decision.csv DROP labels ignored','gpu_throttle_audit':gpu_audit}
  atomic_json(summary,out/'FINAL_EXPERIMENT_SUMMARY.json')
  return summary
