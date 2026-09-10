from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json,read_json,hash_strings
from .common import file_sha256

def run_sealed_once(package_root:Path,project_root:Path,selection_output:Path,sealed_dataset:Path,confirm_token:str):
  if confirm_token!='I_UNDERSTAND_SEALED_IS_ONE_TIME': raise RuntimeError('명시적 확인 토큰이 필요합니다.')
  results=selection_output/'results'; lock=read_json(results/'LOCKED_PROFILE.json',{})
  if lock.get('status')!='LOCKED' or not lock.get('sealed_evaluation_allowed'): raise RuntimeError('P2/P7 프로필이 잠기지 않았습니다.')
  consumed=results/'SEALED_EVALUATION_CONSUMED.json'; started_marker=results/'SEALED_EVALUATION_STARTED.json'
  if consumed.exists() or started_marker.exists(): raise RuntimeError('sealed 평가는 이미 시작 또는 소비되었습니다. 재실행 금지.')
  profile_name=lock['locked_profile']; manifest=read_json(selection_output/'experiment_profile_manifest.json',{}); features=manifest['profiles'][profile_name]['features']
  # Load development cache context through the previous selection record.
  compat=read_json(selection_output/'dataset_compatibility.json',{}); cache=Path(compat['cache_root'])
  X=np.load(cache/'X_all_valid.npy',mmap_mode='r'); y=np.load(cache/'target.npy',mmap_mode='r'); dates=np.load(cache/'dates_ns.npy',mmap_mode='r'); all_features=json.load(open(cache/'feature_names.json',encoding='utf-8')); idx=[all_features.index(f) for f in features]
  import pyarrow.parquet as pq
  schema=pq.ParquetFile(sealed_dataset).schema_arrow.names; target='label_abs_crash_20'; date_col=next((x for x in ['date','Date','datetime','trade_date'] if x in schema),None)
  if target not in schema or date_col is None: raise RuntimeError('sealed 데이터의 target/date 열을 찾지 못했습니다.')
  missing=[f for f in features if f not in schema]
  if missing: raise RuntimeError(f'sealed 피처 누락 {len(missing)}개: {missing[:20]}')
  table=pq.read_table(sealed_dataset,columns=[date_col,target]+features).to_pandas(); sd=pd.to_datetime(table[date_col],errors='raise'); dev_max=pd.Timestamp(int(np.max(dates)))
  if sd.min()<=dev_max: raise RuntimeError(f'sealed 기간이 개발 데이터 이후가 아닙니다. sealed_min={sd.min()} dev_max={dev_max}')
  atomic_json({'started_epoch':time.time(),'profile':profile_name,'profile_hash':hash_strings(features),'sealed_path':str(sealed_dataset),'sealed_sha256':file_sha256(sealed_dataset),'dev_max_date':str(dev_max),'sealed_min_date':str(sd.min())},started_marker)
  import lightgbm as lgb
  xtr=np.ascontiguousarray(X[:,idx],dtype=np.float32); ytr=np.asarray(y,dtype=np.uint8); xva=table[features].to_numpy(dtype=np.float32,copy=True); yva=table[target].to_numpy(dtype=np.uint8); dns=sd.astype('int64').to_numpy()
  best={int(k):int(v) for k,v in read_json(package_root/'seed_results/effective_best_iterations.json',{}).items()}; rounds=int(np.median(list(best.values()))); seeds=[17,43,101,211,503]; preds=[]
  pos=max(1,int((ytr==1).sum()));neg=max(1,int((ytr==0).sum()))
  for seed in seeds:
    params={'objective':'binary','metric':'None','learning_rate':0.03,'num_leaves':127,'min_data_in_leaf':55,'feature_fraction':0.85,'bagging_fraction':0.90,'bagging_freq':1,'lambda_l1':0.25,'lambda_l2':1.8,'max_bin':255,'scale_pos_weight':neg/pos,'num_threads':8,'deterministic':True,'force_col_wise':True,'verbosity':-1,'seed':seed,'feature_fraction_seed':seed,'bagging_seed':seed}
    model=lgb.train(params,lgb.Dataset(xtr,label=ytr),num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)]); preds.append(model.predict(xva))
  pred=np.mean(np.vstack(preds),axis=0); metrics=compute_metrics(yva,pred,dns)
  payload={'status':'CONSUMED','profile':profile_name,'feature_count':len(features),'rounds':rounds,'seeds':seeds,'sealed_rows':len(yva),'metrics':metrics,'completed_epoch':time.time(),'warning':'Do not tune or rerun based on this result.'}
  atomic_json(payload,consumed); return payload
