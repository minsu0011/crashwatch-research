from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from subprocess import run
import sys

from surge_v13_data import TARGET_COLUMN, MOVE_COLUMN
from surge_v14_models import DOWN_COLUMN, crossfit_rank_features, deduplicate_ranked_features, policy_metrics, oracle_topk
from run_surge_competingrisk_hardfp_v14 import crossfit_second_level, apply_second_level_future


def make_data(seed=17):
    rng=np.random.default_rng(seed); rows=[]; sid=0
    for fold in range(8):
        for i in range(120):
            ticker=f"{(i%8)+1:06d}"; latent=rng.normal()+0.45*np.sin(i/7)+0.10*fold
            y=int(latent+rng.normal(0,.8)>1.05)
            move=int(abs(latent)+rng.normal(0,.4)>0.75 or y)
            down=int((latent+rng.normal(0,.7))<-0.8)
            pdirect=np.clip(.10+.60*y+rng.normal(0,.16),.01,.99)
            phazard=np.clip(.12+.52*y+rng.normal(0,.18),.01,.99)
            pdown=np.clip(.12+.58*down+rng.normal(0,.16),.01,.99)
            v13=np.clip(.10+.42*y+rng.normal(0,.20),.01,.99)
            pdir=np.clip(.38+.25*y+rng.normal(0,.16),.01,.99)
            pmove=np.clip(.15+.55*move+rng.normal(0,.15),.01,.99)
            base=np.clip(.25+.35*y+rng.normal(0,.24),.01,.99)
            rows.append({
                'source_row_id':sid,'fold_id':fold,'ticker':ticker,'date':pd.Timestamp('2020-01-01')+pd.Timedelta(days=sid),
                'market':'KOSPI' if i%2 else 'KOSDAQ','bucket':f'B{i%3}',TARGET_COLUMN:y,MOVE_COLUMN:move,DOWN_COLUMN:down,
                'p_direct':pdirect,'p_hazard_up':phazard,'p_down':pdown,'v13_stage_probability':v13,
                'v13_p_move':pmove,'v13_p_up_given_move':pdir,'v10_base_past_rank':base,
                'dir_a':latent+rng.normal(0,.35),'dir_b':.8*latent+rng.normal(0,.45),
                'v11_lag0_peer_signed_resid_mean':.3*latent+rng.normal(0,.8),
            }); sid+=1
    df=pd.DataFrame(rows); df['one_minus_p_down']=1-df.p_down
    df['expert_seed_score']=np.max(df[['p_direct','p_hazard_up','v13_stage_probability']].to_numpy(),axis=1)
    return df


def main():
    df=make_data(); disc=df[df.fold_id.isin([0,1,2])].copy()
    ranking=crossfit_rank_features(disc,['dir_a','dir_b'],target_column=TARGET_COLUMN,folds=[0,1,2],minimum_total_rows=150,minimum_holdout_rows=40,minimum_coverage=.8,minimum_fold_coverage=.8,minimum_worst_auc=.48)
    selected,dec=deduplicate_ranked_features(disc,ranking,max_features=2,corr_threshold=.99)
    hfp=[*selected,'v11_lag0_peer_signed_resid_mean','p_direct','p_hazard_up','v13_stage_probability','v13_p_move','v13_p_up_given_move','one_minus_p_down','v10_base_past_rank']
    d2,detail,hmodel,stack,mix=crossfit_second_level(disc,hardfp_features=hfp,hardfp_quantile=.72,seed=33,policy_mixes=[(1,0,0),(.65,.25,.1)],minimum_alerts=20,target_precision=.60)
    parts=[d2]
    for fold in range(3,8):
        cur=df[df.fold_id.eq(fold)].copy()
        # validation feature argument only needs merge keys/raw second-level columns
        q=apply_second_level_future(cur,cur,hardfp_features=hfp,hardfp_model=hmodel,stacker=stack,policy_mix=mix,reference_discovery=d2)
        parts.append(q)
    final=pd.concat(parts,ignore_index=True); threshold=float(detail['policy']['threshold'])
    metrics=policy_metrics(final,threshold); oracle=oracle_topk(final,minimum_alerts=20)
    root=Path(tempfile.mkdtemp(prefix='v14_synth_'))
    def j(name,obj): (root/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf8')
    j('RUN_STATUS.json',{'status':'SUCCESS'})
    j('DATA_AUDIT_V14.json',{'rows_full_target_valid':len(final),'tickers':final.ticker.nunique(),'raw_features':3,'stable_direct_pool':2,'stable_direction_pool':2,'stable_down_pool':2,'v13_v11_self_primary':False,'v11_lag0_hardfp_only':True})
    j('V14_CHAMPION_CONFIG.json',{'stacker_weights':detail['stacker_weights'],'v11_self_primary':False})
    j('FINAL_RECOMMENDATION_V14.json',{'policy_target':{'minimum_alerts':30,'precision':.70},'production_decision':'NO_ALERT'})
    ranking.to_csv(root/'V14_DIRECT_FEATURE_RANKING_CROSSFIT.csv',index=False); ranking.to_csv(root/'V14_DIRECTION_FEATURE_RANKING_CROSSFIT.csv',index=False); ranking.to_csv(root/'V14_DOWN_FEATURE_RANKING_CROSSFIT.csv',index=False)
    dec.to_csv(root/'V14_DIRECT_DEDUP_DECISIONS.csv',index=False); dec.to_csv(root/'V14_DIRECTION_DEDUP_DECISIONS.csv',index=False); dec.to_csv(root/'V14_DOWN_DEDUP_DECISIONS.csv',index=False)
    pd.DataFrame([{'config_id':'SYNTH'}]).to_csv(root/'V14_CONFIG_SCREENING.csv',index=False); pd.DataFrame([{'config_id':'SYNTH'}]).to_csv(root/'V14_DEVELOPMENT_CONFIG_VALIDATION.csv',index=False)
    final.to_csv(root/'v14_oof_predictions.csv.gz',index=False,compression='gzip'); metrics.to_csv(root/'v14_frozen_policy_by_fold.csv',index=False); oracle.to_csv(root/'v14_oracle_topk_by_fold.csv',index=False)
    pd.DataFrame([{'fold_id':f,'v14_pr_auc':.8} for f in range(8)]).to_csv(root/'v14_expert_metrics_by_fold.csv',index=False)
    verifier=Path(__file__).with_name('verify_surge_competingrisk_hardfp_v14.py')
    res=run([sys.executable,str(verifier),'--output',str(root),'--allow-synthetic'],capture_output=True,text=True)
    print(res.stdout)
    if res.returncode: print(res.stderr); raise SystemExit(res.returncode)
    print('SYNTHETIC_E2E_V14_PASS',root)

if __name__=='__main__': main()
