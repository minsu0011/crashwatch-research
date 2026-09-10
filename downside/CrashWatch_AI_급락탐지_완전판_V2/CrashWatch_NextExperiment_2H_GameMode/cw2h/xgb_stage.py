from __future__ import annotations
import gc, os, threading, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import psutil
from cw7h.gpu_worker import detect_cuda
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, canonical_hash, hash_strings, read_json
from .common import lower_child_priority, pubg_running
VERSION='cw_next2h_xgb_profiles_v1'

class NvmlSampler:
  def __init__(self,interval:float=0.25): self.interval=interval; self.values=[]; self.memory=[]; self.stop_event=threading.Event(); self.thread=None; self.handle=None
  def start(self):
    try:
      import pynvml; pynvml.nvmlInit(); self.pynvml=pynvml; self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
      self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
    except Exception: self.handle=None
  def _loop(self):
    while not self.stop_event.wait(self.interval):
      try:
        u=self.pynvml.nvmlDeviceGetUtilizationRates(self.handle); m=self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.values.append(float(u.gpu)); self.memory.append(float(m.used/m.total*100))
      except Exception: pass
  def stop(self):
    self.stop_event.set()
    if self.thread: self.thread.join(timeout=2)
    return (float(np.mean(self.values)) if self.values else 0.0,float(np.max(self.values)) if self.values else 0.0,float(np.max(self.memory)) if self.memory else 0.0)

def snapshot():
  try:
    import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); u=pynvml.nvmlDeviceGetUtilizationRates(h); m=pynvml.nvmlDeviceGetMemoryInfo(h); return float(u.gpu),float(m.used/m.total*100)
  except Exception: return 0.0,0.0

def launch_guard(cfg,deadline):
  waited=0.0
  while time.time()<deadline and waited<float(cfg['launch_guard_max_wait_seconds']):
    util,mem=snapshot(); busy=pubg_running()
    # During an actual match, do not launch when the game already occupies the GPU heavily.
    util_limit=float(cfg['launch_guard_gpu_util_percent']) if busy else 90.0
    if util<=util_limit and mem<=float(cfg['launch_guard_memory_percent']): return waited,util,mem,busy
    t=float(cfg['launch_guard_poll_seconds']); time.sleep(t); waited+=t
  util,mem=snapshot(); return waited,util,mem,pubg_running()

def _train(xgb,xtr,ytr,xva,cfg,seed):
  mb=int(cfg['max_bin']); dtr=xgb.QuantileDMatrix(xtr,label=ytr,max_bin=mb,nthread=1); dva=xgb.QuantileDMatrix(xva,ref=dtr,max_bin=mb,nthread=1)
  pos=max(1,int((ytr==1).sum())); neg=max(1,int((ytr==0).sum()))
  p={'objective':'binary:logistic','eval_metric':'aucpr','device':'cuda','tree_method':'hist','learning_rate':float(cfg['learning_rate']),'max_depth':int(cfg['max_depth']),'max_leaves':int(cfg['max_leaves']),'grow_policy':str(cfg['grow_policy']),'min_child_weight':float(cfg['min_child_weight']),'subsample':float(cfg['subsample']),'colsample_bytree':float(cfg['colsample_bytree']),'reg_alpha':float(cfg['reg_alpha']),'reg_lambda':float(cfg['reg_lambda']),'max_bin':mb,'scale_pos_weight':float(neg/pos),'seed':int(seed),'nthread':1,'verbosity':0}
  model=xgb.train(p,dtr,num_boost_round=int(cfg['rounds']),verbose_eval=False); pred=np.asarray(model.predict(dva),dtype=np.float32); del model,dtr,dva; return pred

def run(prepared,folds,profiles,cfg,task_dir,soft_deadline,dataset_signature,summary_path):
  lower_child_priority(); os.environ['OMP_NUM_THREADS']='1'; os.environ['CUDA_DEVICE_MAX_CONNECTIONS']='1'
  gpu=detect_cuda(); task_dir.mkdir(parents=True,exist_ok=True)
  if not gpu.get('available'):
    r={'status':'skipped','gpu':gpu}; atomic_json(r,summary_path); return r
  import xgboost as xgb
  X=np.load(prepared.root/'X_all_valid.npy',mmap_mode='r'); y=np.load(prepared.root/'target.npy',mmap_mode='r'); dates=np.load(prepared.root/'dates_ns.npy',mmap_mode='r'); index={f:i for i,f in enumerate(prepared.feature_names)}
  core=[int(x) for x in cfg['core_seeds']]; bonus=[int(x) for x in cfg['bonus_seeds']]
  completed=cached=failed=deadline_skipped=0; wall=[]; cooldown_total=0.0; started=time.time()
  def execute_seed(seed:int,required:bool):
    nonlocal completed,cached,failed,deadline_skipped,cooldown_total
    for fold in sorted(folds,key=lambda f:f.fold_id,reverse=True):
      # Run P0/P2/P7 as one paired bundle.
      for profile,features in profiles.items():
        if time.time()>=soft_deadline: deadline_skipped+=1; return False
        idx=np.asarray([index[f] for f in features],dtype=np.int32); fd=fold.to_dict()
        ident={'version':VERSION,'dataset_signature':dataset_signature,'fold':fd,'profile':profile,'profile_hash':hash_strings(features),'seed':seed,'config':{k:v for k,v in cfg.items() if k not in ['core_seeds','bonus_seeds']}}
        tid=canonical_hash(ident); path=task_dir/f'{tid}.json'; old=read_json(path,{})
        if old.get('status')=='completed' and old.get('identity')==ident: cached+=1; continue
        total_wait=0.0
        while True:
          waited,pre_util,pre_mem,busy=launch_guard(cfg,min(soft_deadline,time.time()+float(cfg['launch_guard_max_wait_seconds'])))
          total_wait+=waited
          util_limit=float(cfg['launch_guard_gpu_util_percent']) if busy else 90.0
          allowed=(pre_util<=util_limit and pre_mem<=float(cfg['launch_guard_memory_percent']))
          if allowed: break
          if time.time()>=soft_deadline: deadline_skipped+=1; return False
        waited=total_wait
        tr=slice(fold.train_start,fold.train_stop); va=slice(fold.validation_start,fold.validation_stop)
        xtr=np.ascontiguousarray(X[tr][:,idx],dtype=np.float32); ytr=np.asarray(y[tr],dtype=np.uint8); xva=np.ascontiguousarray(X[va][:,idx],dtype=np.float32); yva=np.asarray(y[va],dtype=np.uint8); dva=np.asarray(dates[va],dtype=np.int64)
        sampler=NvmlSampler(float(cfg['sample_interval_seconds'])); sampler.start(); t0=time.perf_counter()
        try:
          pred=_train(xgb,xtr,ytr,xva,cfg,seed); train_s=time.perf_counter()-t0; avg,maxu,maxm=sampler.stop()
          target=max(1.0,float(cfg['target_average_gpu_percent'])); estimated_added_util=max(0.0,avg-pre_util); cooldown=train_s*max(0.0,estimated_added_util/target-1.0)
          minimum=float(cfg['pubg_minimum_cooldown_seconds'] if busy or pubg_running() else cfg['minimum_cooldown_seconds'])
          cooldown=min(float(cfg['maximum_cooldown_seconds']),max(minimum,cooldown)); cooldown=min(cooldown,max(0.0,soft_deadline-time.time()))
          result={'status':'completed','identity':ident,'task_id':tid,'backend':'xgboost_cuda','test_type':'p0_p2_p7_profile_confirmatory','profile':profile,'feature_count':len(features),'seed':seed,'outer_fold':fold.fold_id,'required_core_seed':required,'train_seconds':train_s,'launch_wait_seconds':waited,'pre_gpu_util':pre_util,'pre_gpu_memory_percent':pre_mem,'pubg_detected':busy,'training_avg_gpu_util':avg,'estimated_xgb_added_gpu_util':estimated_added_util,'training_max_gpu_util':maxu,'training_max_memory_percent':maxm,'scheduled_cooldown_seconds':cooldown,**compute_metrics(yva,pred,dva)}
          atomic_json(result,path); completed+=1; wall.append(train_s+cooldown); del pred
          if cooldown>0: time.sleep(cooldown); cooldown_total+=cooldown
        except Exception as exc:
          try: sampler.stop()
          except Exception: pass
          failed+=1; atomic_json({'status':'failed','identity':ident,'task_id':tid,'profile':profile,'seed':seed,'outer_fold':fold.fold_id,'error':repr(exc),'traceback':traceback.format_exc()},path)
        del xtr,xva,ytr,yva,dva; gc.collect()
    return True
  for seed in core:
    if not execute_seed(seed,True): break
  # Bonus seeds are attempted only when measured ETA fits the remaining protected window.
  for seed in bonus:
    remaining=soft_deadline-time.time(); median=float(np.median(wall[-12:])) if wall else 45.0; estimate=median*len(folds)*len(profiles)*1.20
    if remaining<max(float(cfg['bonus_seed_min_remaining_seconds']),estimate): break
    if not execute_seed(seed,False): break
  result={'status':'completed' if failed==0 else 'partial','gpu':gpu,'completed':completed,'cached':cached,'failed':failed,'deadline_skipped':deadline_skipped,'cooldown_total_seconds':cooldown_total,'elapsed_seconds':time.time()-started,'core_seed_count':len(core),'bonus_seed_count':len(bonus)}
  atomic_json(result,summary_path); return result
