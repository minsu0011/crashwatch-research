from __future__ import annotations
import json, logging, multiprocessing as mp, os, signal, threading, time
from pathlib import Path
from typing import Any
from cw7h.utils import atomic_json
from .common import setup_logging,set_game_process_mode,acquire_lock,release_lock,resolve_output,load_context
from .profiles import build_profiles
from .lgb_stage import run as run_lgb
from .xgb_stage import run as run_xgb
from .aggregate import aggregate
LOGGER=logging.getLogger(__name__)

def _terminate_tree(proc):
  if proc is None or not proc.pid: return
  try:
    import psutil
    root=psutil.Process(proc.pid)
    children=root.children(recursive=True)
    for child in reversed(children):
      try: child.terminate()
      except Exception: pass
    _,alive=psutil.wait_procs(children,timeout=5)
    for child in alive:
      try: child.kill()
      except Exception: pass
  except Exception: pass
  try:
    if proc.is_alive(): proc.terminate()
    proc.join(timeout=8)
    if proc.is_alive(): proc.kill(); proc.join(timeout=3)
  except Exception: pass

def _lgb_entry(q,kwargs):
  try:q.put({'ok':True,'result':run_lgb(**kwargs)})
  except Exception as exc:
    import traceback;q.put({'ok':False,'error':repr(exc),'traceback':traceback.format_exc()})

def _xgb_entry(q,kwargs):
  try:q.put({'ok':True,'result':run_xgb(**kwargs)})
  except Exception as exc:
    import traceback;q.put({'ok':False,'error':repr(exc),'traceback':traceback.format_exc()})

def run_pipeline(package_root:Path,project_root:Path,cfg:dict,dataset_override:str|None,output_override:str|None):
  started=time.time(); hard=started+float(cfg['runtime']['hard_limit_seconds']); soft=started+float(cfg['runtime']['model_soft_stop_seconds']); force=started+float(cfg['runtime']['forced_stop_seconds'])
  output=resolve_output(project_root,output_override,cfg['output_dir']); setup_logging(output/'next_experiment_2h.log'); lock=output/'pipeline.lock.json'; acquire_lock(lock)
  procs=[]
  try:
    hardware=set_game_process_mode(int(cfg['cpu']['affinity_threads'])); atomic_json(hardware,output/'hardware_game_mode.json')
    prepared,refs,folds,legacy,best=load_context(package_root,project_root,dataset_override,output)
    named,lgb_profiles,xgb_profiles,aliases,manifest=build_profiles(package_root,prepared.feature_names,output)
    task_lgb=output/'task_results/lightgbm'; task_xgb=output/'task_results/xgboost'
    ctx=mp.get_context('spawn'); lq=ctx.Queue(); xq=ctx.Queue()
    lkwargs={'prepared':prepared,'folds':folds,'profiles':lgb_profiles,'seeds':[int(x) for x in cfg['selection']['lightgbm_seeds']],'best_iterations':best,'cfg':cfg['lightgbm'],'task_dir':task_lgb,'workers':int(cfg['cpu']['lightgbm_workers']),'threads':int(cfg['cpu']['threads_per_lightgbm_worker']),'soft_deadline':soft,'dataset_signature':prepared.signature}
    xkwargs={'prepared':prepared,'folds':folds,'profiles':xgb_profiles,'cfg':cfg['xgboost'],'task_dir':task_xgb,'soft_deadline':soft,'dataset_signature':prepared.signature,'summary_path':output/'xgboost_stage_summary.json'}
    lp=ctx.Process(target=_lgb_entry,args=(lq,lkwargs),name='cw2h-lightgbm',daemon=False); procs.append(lp); lp.start()
    xp=None
    if bool(cfg['xgboost'].get('enabled',True)):
      xp=ctx.Process(target=_xgb_entry,args=(xq,xkwargs),name='cw2h-xgboost-lowgpu',daemon=False); procs.append(xp); xp.start()
    # Resource trace and strict watchdog.
    import psutil,csv
    trace=output/'resource_usage.csv'; f=trace.open('w',newline='',encoding='utf-8-sig'); w=csv.DictWriter(f,fieldnames=['epoch','elapsed','cpu_percent','available_ram_gb','lgb_alive','xgb_alive']); w.writeheader()
    while any(p.is_alive() for p in procs) and time.time()<force:
      w.writerow({'epoch':time.time(),'elapsed':time.time()-started,'cpu_percent':psutil.cpu_percent(interval=None),'available_ram_gb':psutil.virtual_memory().available/1024**3,'lgb_alive':lp.is_alive(),'xgb_alive':bool(xp and xp.is_alive())}); f.flush(); time.sleep(float(cfg['runtime']['resource_poll_seconds']))
    f.close()
    forced=[]
    for p in procs:
      if p.is_alive(): forced.append(p.name); _terminate_tree(p)
    lmsg={'ok':False,'error':'no result'}; xmsg={'ok':False,'error':'disabled'}
    try:lmsg=lq.get_nowait()
    except Exception:pass
    if xp is not None:
      try:xmsg=xq.get_nowait()
      except Exception:pass
    result=aggregate(output,aliases,cfg,manifest)
    complete=bool(result.get('lgb_unique_completed',0)>=320 and result.get('xgb_core_complete',False))
    final={'status':'completed' if complete else 'partial_budget_or_gpu_guard','elapsed_seconds':time.time()-started,'hard_limit_seconds':cfg['runtime']['hard_limit_seconds'],'hard_limit_respected':time.time()<=hard,'forced_processes':forced,'lightgbm_stage':lmsg,'xgboost_stage':xmsg,'summary':result,'output_dir':str(output)}
    atomic_json(final,output/'FINAL_RUN_STATUS.json'); return final
  finally:
    for p in procs:
      if p.is_alive(): _terminate_tree(p)
    release_lock(lock)
