from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from cwregime.target import build_3d5_target_from_ret1
from cwregime.regimes import build_market_regimes, REGIMES
from cwregime.windows import build_two_bank_windows
from cwregime.engine import _lgb_bundle_worker

rng=np.random.default_rng(17)
dates=pd.bdate_range('2018-01-02',periods=1200)
market=np.zeros(len(dates),dtype=float)
# Eight repeating market states with deterministic shocks/rebounds.
for i in range(len(dates)):
    phase=(i//40)%8
    noise=rng.normal(0,0.003 if phase%2==0 else 0.010)
    drift=[-0.004,0.005,0.002,0.003,0.000,-0.001,-0.003,-0.002][phase]
    if phase==0 and i%8==0: drift=-0.018
    if phase==1 and i%7==0: drift=0.015
    market[i]=drift+noise
rows=[]
for di,d in enumerate(dates):
    for t in range(12):
        rows.append((int(d.value),f'{t:06d}',market[di]+rng.normal(0,0.014)))
arr=np.asarray(rows,dtype=object); dn=arr[:,0].astype(np.int64); tick=arr[:,1].astype(str); ret=arr[:,2].astype(float)
target=build_3d5_target_from_ret1(dn,tick,ret,horizon_days=3,drop_threshold=-0.05)
assert target.valid.sum()>0 and target.label[target.valid].sum()>0
reg=build_market_regimes(dn,np.repeat(market,12),rebound_5d=0.03,rebound_drawdown_60=-0.03)
assert set(reg.calendar.regime.unique()).issubset(set(REGIMES))
# Window construction may be data-dependent; verify with permissive minimum for synthetic data.
search,confirm,audit=build_two_bank_windows(reg.calendar,{'min_train_days':350,'purge_days':3,'window_days':10,'min_target_regime_days':1,'search_chronology_fraction':0.62})
assert len(search)==8 and len(confirm)==8

with tempfile.TemporaryDirectory() as td:
    td=Path(td); n=len(dn); X=rng.normal(size=(n,16)).astype(np.float32); X[:,0]=ret.astype(np.float32)
    np.save(td/'X.npy',X); np.save(td/'y.npy',target.label); np.save(td/'valid.npy',target.valid); np.save(td/'dates.npy',dn); np.save(td/'hit.npy',target.first_hit_day)
    w=search[0]
    plan={
        'matrix_path':str(td/'X.npy'),'target_path':str(td/'y.npy'),'valid_path':str(td/'valid.npy'),'dates_path':str(td/'dates.npy'),'first_hit_path':str(td/'hit.npy'),
        'train_cutoff_ns':w.train_cutoff_ns,'val_start_ns':w.start_ns,'val_end_ns':w.end_ns,'threads':2,'seeds':[17],
        'model_config':{'rounds':15,'learning_rate':0.05,'num_leaves':15,'min_data_in_leaf':20,'feature_fraction':0.9,'bagging_fraction':0.9,'bagging_freq':1,'lambda_l1':0.1,'lambda_l2':1.0,'max_bin':63,'class_weight_multiplier':1.0},
        'json_path':str(td/'result.json'),'npz_path':str(td/'pred.npz'),'bundle_id':'smoke','stage':'smoke','profile':'SYNTH','config_id':'LGB','window_id':w.window_id,
    }
    result=_lgb_bundle_worker(plan); assert result['status']=='complete', result
    # Close the NPZ handle before TemporaryDirectory cleanup; Windows otherwise
    # keeps the file locked and turns a successful smoke test into WinError 32.
    with np.load(td/'pred.npz') as payload:
        assert payload['predictions'].shape[0]==1 and payload['predictions'].shape[1]>0
print(json.dumps({'status':'SMOKE_PASS','target_positive_rate':target.audit['positive_rate'],'regimes':reg.audit['counts_by_regime'],'search_windows':len(search),'confirm_windows':len(confirm)},ensure_ascii=False,indent=2))
