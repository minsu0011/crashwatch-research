from __future__ import annotations
import concurrent.futures as cf, gc, multiprocessing as mp, os, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, canonical_hash, hash_strings, read_json
from .common import lower_child_priority
VERSION='cw_next2h_lgb_decomposition_v1'

def _params(cfg,y,threads,seed):
    pos=max(1,int((y==1).sum())); neg=max(1,int((y==0).sum()))
    return {'objective':'binary','metric':'None','learning_rate':float(cfg['learning_rate']),'num_leaves':int(cfg['num_leaves']),'max_depth':int(cfg['max_depth']),'min_data_in_leaf':int(cfg['min_data_in_leaf']),'feature_fraction':float(cfg['feature_fraction']),'bagging_fraction':float(cfg['bagging_fraction']),'bagging_freq':int(cfg['bagging_freq']),'lambda_l1':float(cfg['lambda_l1']),'lambda_l2':float(cfg['lambda_l2']),'max_bin':int(cfg['max_bin']),'min_gain_to_split':float(cfg['min_gain_to_split']),'scale_pos_weight':float(neg/pos),'num_threads':int(threads),'deterministic':True,'force_col_wise':True,'feature_pre_filter':False,'verbosity':-1,'seed':int(seed),'feature_fraction_seed':int(seed),'bagging_seed':int(seed),'drop_seed':int(seed)}

def _worker(plan:dict[str,Any])->dict[str,Any]:
    lower_child_priority(); os.environ['OMP_NUM_THREADS']=str(plan['threads']); os.environ['MKL_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    import lightgbm as lgb
    root=Path(plan['cache_root']); X=np.load(root/'X_all_valid.npy',mmap_mode='r'); y=np.load(root/'target.npy',mmap_mode='r'); dates=np.load(root/'dates_ns.npy',mmap_mode='r')
    index={f:i for i,f in enumerate(plan['feature_names'])}; out=Path(plan['task_dir']); out.mkdir(parents=True,exist_ok=True)
    completed=cached=failed=deadline_skipped=0; fold=plan['fold']; tr=slice(fold['train_start'],fold['train_stop']); va=slice(fold['validation_start'],fold['validation_stop'])
    ytr=np.asarray(y[tr],dtype=np.uint8); yva=np.asarray(y[va],dtype=np.uint8); dva=np.asarray(dates[va],dtype=np.int64)
    # Seed-first order preserves paired profile bundles if the deadline is reached.
    for seed in plan['seeds']:
      for profile,features in plan['profiles'].items():
        if time.time()>=plan['soft_deadline']: deadline_skipped+=1; continue
        idx=np.asarray([index[f] for f in features],dtype=np.int32)
        identity={'version':VERSION,'dataset_signature':plan['dataset_signature'],'fold':fold,'profile':profile,'profile_hash':hash_strings(features),'seed':int(seed),'best_iteration':int(plan['best_iteration']),'config':plan['config']}
        tid=canonical_hash(identity); path=out/f'{tid}.json'; old=read_json(path,{})
        if old.get('status')=='completed' and old.get('identity')==identity: cached+=1; continue
        started=time.perf_counter()
        try:
          xtr=np.ascontiguousarray(X[tr][:,idx],dtype=np.float32); xva=np.ascontiguousarray(X[va][:,idx],dtype=np.float32)
          ds=lgb.Dataset(xtr,label=ytr,free_raw_data=True,params={'max_bin':int(plan['config']['max_bin']),'feature_pre_filter':False})
          model=lgb.train(_params(plan['config'],ytr,plan['threads'],seed),ds,num_boost_round=int(plan['best_iteration']),callbacks=[lgb.log_evaluation(0)])
          pred=np.asarray(model.predict(xva),dtype=np.float32)
          result={'status':'completed','identity':identity,'task_id':tid,'backend':'lightgbm_cpu','test_type':'p2_causal_decomposition','profile':profile,'feature_count':len(features),'seed':int(seed),'outer_fold':int(fold['fold_id']),'elapsed_seconds':time.perf_counter()-started,**compute_metrics(yva,pred,dva)}
          atomic_json(result,path); completed+=1
          del xtr,xva,ds,model,pred
        except Exception as exc:
          failed+=1; atomic_json({'status':'failed','identity':identity,'task_id':tid,'profile':profile,'seed':seed,'outer_fold':fold['fold_id'],'error':repr(exc),'traceback':traceback.format_exc()},path)
        gc.collect()
    return {'fold':fold['fold_id'],'completed':completed,'cached':cached,'failed':failed,'deadline_skipped':deadline_skipped}

def run(prepared,folds,profiles,seeds,best_iterations,cfg,task_dir,workers,threads,soft_deadline,dataset_signature):
    plans=[{'cache_root':str(prepared.root),'feature_names':prepared.feature_names,'fold':f.to_dict(),'profiles':profiles,'seeds':seeds,'best_iteration':best_iterations[f.fold_id],'config':cfg,'task_dir':str(task_dir),'threads':threads,'soft_deadline':soft_deadline,'dataset_signature':dataset_signature} for f in sorted(folds,key=lambda x:[7,6,5,4,3,2,1,0].index(x.fold_id))]
    ctx=mp.get_context('spawn'); results=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
      fut=[ex.submit(_worker,p) for p in plans]
      for f in cf.as_completed(fut): results.append(f.result())
    return {'completed':sum(x['completed'] for x in results),'cached':sum(x['cached'] for x in results),'failed':sum(x['failed'] for x in results),'deadline_skipped':sum(x['deadline_skipped'] for x in results),'folds':results}
