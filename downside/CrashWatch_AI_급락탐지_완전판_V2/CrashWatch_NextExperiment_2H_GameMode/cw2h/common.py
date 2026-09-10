from __future__ import annotations
import json, logging, os, time, math, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import psutil
from cw7h.data import discover_dataset, load_references, prepare_data, PreparedData, References
from cw7h.folds import build_fold_slices, FoldSlice
from cw7h.utils import atomic_json, read_json, canonical_hash

LOGGER = logging.getLogger(__name__)
PUBG_NAMES = {"tslgame.exe", "tslgame_be.exe", "execpubg.exe", "beservice.exe"}

def setup_logging(path: Path) -> None:
    import sys
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")
    root=logging.getLogger(); root.handlers.clear(); root.setLevel(logging.INFO)
    a=logging.StreamHandler(sys.stdout); a.setFormatter(fmt); root.addHandler(a)
    b=logging.FileHandler(path,encoding="utf-8"); b.setFormatter(fmt); root.addHandler(b)

def set_game_process_mode(total_threads: int) -> dict[str,Any]:
    p=psutil.Process(os.getpid()); logical=psutil.cpu_count(logical=True) or 1
    count=max(1,min(int(total_threads),logical)); info={"logical_cpus":logical,"requested":count}
    try:
        affinity=list(range(max(0,logical-count),logical)); p.cpu_affinity(affinity); info["affinity"]=p.cpu_affinity()
    except Exception as exc: info["affinity_error"]=repr(exc)
    try:
        if os.name=="nt": p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); info["priority"]="below_normal"
    except Exception as exc: info["priority_error"]=repr(exc)
    return info

def lower_child_priority() -> None:
    try:
        if os.name=="nt": psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception: pass

def pubg_running() -> bool:
    for p in psutil.process_iter(["name"]):
        try:
            if str(p.info.get("name") or "").lower() in PUBG_NAMES: return True
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return False

def acquire_lock(path:Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=read_json(path,{})
        pid=int(old.get("pid",-1))
        if pid>0 and psutil.pid_exists(pid): raise RuntimeError(f"이미 실행 중입니다. PID={pid}")
    atomic_json({"pid":os.getpid(),"started_epoch":time.time()},path)

def release_lock(path:Path) -> None:
    try:
        old=read_json(path,{})
        if int(old.get("pid",-1))==os.getpid(): path.unlink(missing_ok=True)
    except Exception: pass

def resolve_output(project_root:Path, explicit:str|None, default_rel:str)->Path:
    p=Path(explicit).expanduser() if explicit else project_root/default_rel
    if not p.is_absolute(): p=project_root/p
    p=p.resolve(); p.mkdir(parents=True,exist_ok=True); return p

def load_context(package_root:Path, project_root:Path, dataset_override:str|None, output_dir:Path):
    seed=package_root/'seed_results'
    resolved=read_json(seed/'resolved_run_config.json',{})
    if 'config' not in resolved: raise FileNotFoundError('seed_results/resolved_run_config.json 누락')
    legacy=resolved['config']; summary=read_json(seed/'run_summary.json',{})
    refs=load_references(package_root/'reference',strict_counts=True)
    dataset=discover_dataset(project_root,str(legacy.get('dataset_path','AUTO')),dataset_override,list(legacy.get('sealed_path_tokens',['sealed','holdout'])))
    cache_dir=Path(legacy.get('cache_dir','crashwatch_ai_data/shared_cache/all_feature_ablation_7h_v1'))
    if not cache_dir.is_absolute(): cache_dir=project_root/cache_dir
    prepared=prepare_data(dataset,cache_dir,legacy,refs,force=False)
    dates=np.load(prepared.dates_path,mmap_mode='r')
    folds=build_fold_slices(dates,refs.folds,int(legacy['min_train_days']),int(legacy['validation_days']),int(legacy['purge_days']),output_dir/'fold_manifest.json')
    if len(folds)!=8 or any(not f.eligible for f in folds): raise RuntimeError('기존 8개 outer fold 재구성 실패')
    compatible=(int(prepared.manifest.get('rows',-1))==91919 and int(prepared.manifest.get('features',-1))==439 and str(prepared.manifest.get('date_min',''))[:10]=='2018-01-02' and str(prepared.manifest.get('date_max',''))[:10]=='2026-06-22')
    old_sig=str(summary.get('dataset_signature',''))
    if prepared.signature!=old_sig and not compatible: raise RuntimeError(f'데이터 불일치 current={prepared.signature} old={old_sig}')
    best={int(k):int(v) for k,v in read_json(seed/'effective_best_iterations.json',{}).items()}
    atomic_json({"dataset_path":str(dataset),"cache_root":str(prepared.root),"current_signature":prepared.signature,"legacy_signature":old_sig,"compatible":compatible},output_dir/'dataset_compatibility.json')
    return prepared,refs,folds,legacy,best

def file_sha256(path:Path, chunk:int=1<<20)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def seconds_left(deadline:float)->float: return max(0.0,deadline-time.time())
