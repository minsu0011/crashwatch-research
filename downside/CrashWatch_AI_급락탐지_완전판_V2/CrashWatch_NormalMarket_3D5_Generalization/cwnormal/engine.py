from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os, time, traceback
from pathlib import Path
from typing import Any

import numpy as np

from cw7h.utils import atomic_json, canonical_hash, read_json
from cwfull.common import set_worker_mode


def _checkpoint_paths(output_dir: Path, stage: str, model: str, bundle_id: str):
    root = Path(output_dir)/"checkpoints"/stage/model; root.mkdir(parents=True, exist_ok=True)
    return root/f"{bundle_id}.json", root/f"{bundle_id}.npz"


def make_plan(output_dir: Path, *, stage: str, model: str, candidate: dict[str,Any], fold: dict[str,Any], seeds: list[int], paths: dict[str,str], params: dict[str,Any], threads: int, sector_filter: str|None=None, exclude_ticker: str|None=None):
    ident={"schema":"normal_market_bundle_v2","runtime_hash":paths["runtime_hash"],"stage":stage,"model":model,"candidate_spec":candidate,"profile":candidate["profile"],"fold":fold,"seeds":seeds,"model_config":params,"sector_filter":sector_filter,"exclude_ticker":exclude_ticker}
    bid=canonical_hash(ident, 28); jp,npz=_checkpoint_paths(output_dir,stage,model,bid)
    return {**ident,"bundle_id":bid,"json_path":str(jp),"npz_path":str(npz),"candidate":candidate["id"],"fold_id":int(fold["fold_id"]),"threads":threads,"sector_filter":sector_filter,"exclude_ticker":exclude_ticker,**paths}


def _weights_and_indices(plan):
    X=np.load(plan["matrix_path"],mmap_mode="r"); y=np.load(plan["target_path"],mmap_mode="r"); valid=np.load(plan["valid_path"],mmap_mode="r")
    dates=np.load(plan["dates_path"],mmap_mode="r"); normal=np.load(plan["normal_path"],mmap_mode="r"); tickers=np.load(plan["tickers_path"],mmap_mode="r"); sectors=np.load(plan["sectors_path"],mmap_mode="r")
    fold=plan["fold"]
    if all(key in fold for key in ("train_stop","validation_start","validation_stop")):
        train_end=int(fold["train_stop"]); vs=int(fold["validation_start"]); ve=int(fold["validation_stop"])
    else:
        train_end=int(np.searchsorted(dates,int(fold["train_end_ns"]),side="right")); vs=int(np.searchsorted(dates,int(fold["validation_start_ns"]),side="left")); ve=int(np.searchsorted(dates,int(fold["validation_end_ns"]),side="right"))
    tr=np.flatnonzero(np.asarray(valid[:train_end],bool)); va=np.flatnonzero(np.asarray(valid[vs:ve],bool)&np.asarray(normal[vs:ve],bool))+vs
    spec=plan["candidate_spec"]; policy=spec["train_policy"]
    if plan.get("sector_filter"):
        sf=str(plan["sector_filter"]); tr=tr[np.asarray(sectors[tr]).astype(str)==sf]; va=va[np.asarray(sectors[va]).astype(str)==sf]
    if plan.get("exclude_ticker"):
        ex=str(plan["exclude_ticker"]); tr=tr[np.asarray(tickers[tr]).astype(str)!=ex]
        # validation stays the target ticker for LOTO diagnostic
        va=va[np.asarray(tickers[va]).astype(str)==ex]
    w=np.ones(len(tr),dtype=np.float32)
    if policy in {"normal_only","normal_recency","normal_rolling"}:
        keep=np.asarray(normal[tr],bool); tr=tr[keep]; w=np.ones(len(tr),np.float32)
    elif policy=="normal_soft":
        w=np.where(np.asarray(normal[tr],bool),1.0,float(spec.get("abnormal_weight",0.15))).astype(np.float32)
    if policy=="normal_rolling":
        n=int(spec.get("train_window_days",1000)); unique=np.unique(dates[tr]); cutoff=unique[max(0,len(unique)-n)] if len(unique) else -1; keep=dates[tr]>=cutoff; tr=tr[keep]; w=w[keep]
    if policy=="normal_recency":
        half=float(spec.get("half_life_days",504)); ud=np.unique(dates[tr]); rank={int(d):i for i,d in enumerate(ud)}; maxr=max(rank.values()) if rank else 0
        age=np.asarray([maxr-rank[int(d)] for d in dates[tr]],float); w*=np.exp(-np.log(2)*age/half).astype(np.float32)
    # Base candidate selection keeps the strict guard. Sector experts and
    # leave-one-ticker-out runs are report/overlay diagnostics: after the
    # NORMAL gate a valid sector can have only a handful of rows in one fold.
    # Requiring 20 there aborts the whole experiment even though the rows are
    # legitimate and the metrics layer already handles sparse/single-class
    # scopes as NaN.
    stage = str(plan.get("stage"))
    if stage in {"sector", "loto"} or plan.get("exclude_ticker"):
        min_val = 1
    elif stage == "recent_anchor":
        min_val = 8
    else:
        min_val = 20
    if len(tr)<500 or len(va)<min_val: raise RuntimeError(f"insufficient rows train={len(tr)} val={len(va)} policy={policy} min_val={min_val}")
    return X,y,dates,tickers,sectors,tr,va,w


def _run(plan, model):
    jp=Path(plan["json_path"]); npz=Path(plan["npz_path"]); cached=read_json(jp,{})
    if cached.get("status")=="complete" and cached.get("bundle_id")==plan["bundle_id"] and npz.exists() and npz.stat().st_size>0: return cached
    started=time.time()
    try:
        set_worker_mode(int(plan["threads"]),"high")
        X,y,dates,tickers,sectors,tr,va,w=_weights_and_indices(plan)
        xt=np.asarray(X[tr],np.float32); yt=np.asarray(y[tr],np.uint8); xv=np.asarray(X[va],np.float32); yv=np.asarray(y[va],np.uint8)
        pos=max(1,int((yt==1).sum())); neg=max(1,int((yt==0).sum())); cfg=dict(plan["model_config"]); rounds=int(cfg.pop("rounds")); mult=float(cfg.pop("class_weight_multiplier",1.0)); preds=[]
        if model=="lightgbm":
            import lightgbm as lgb
            base={"objective":"binary","metric":"None","verbosity":-1,"deterministic":True,"force_col_wise":True,"feature_pre_filter":False,"num_threads":int(plan["threads"]),"scale_pos_weight":(neg/pos)*mult}; base.update(cfg)
            ds=lgb.Dataset(xt,label=yt,weight=w,free_raw_data=False,params={"max_bin":int(base.get("max_bin",255)),"feature_pre_filter":False})
            for s in plan["seeds"]:
                p=dict(base); p.update({"seed":int(s),"feature_fraction_seed":int(s),"bagging_seed":int(s)})
                m=lgb.train(p,ds,num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)]); preds.append(np.asarray(m.predict(xv),np.float32))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"; import xgboost as xgb
            max_bin=int(cfg.get("max_bin",256)); dt=xgb.QuantileDMatrix(xt,label=yt,weight=w,max_bin=max_bin,nthread=int(plan["threads"])); dv=xgb.QuantileDMatrix(xv,ref=dt,max_bin=max_bin,nthread=int(plan["threads"]))
            for s in plan["seeds"]:
                p={"objective":"binary:logistic","eval_metric":"aucpr","device":"cuda","tree_method":"hist","scale_pos_weight":(neg/pos)*mult,"seed":int(s),"verbosity":0,"nthread":int(plan["threads"])}; p.update(cfg)
                m=xgb.train(p,dt,num_boost_round=rounds,verbose_eval=False); preds.append(np.asarray(m.predict(dv),np.float32))
        temp=npz.with_name(f".{npz.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temp,val_idx=va,y=yv,predictions=np.vstack(preds),date_ns=np.asarray(dates[va],np.int64),ticker=np.asarray(tickers[va]).astype(str),sector=np.asarray(sectors[va]).astype(str))
        os.replace(temp,npz)
        res={"status":"complete","bundle_id":plan["bundle_id"],"stage":plan["stage"],"model":model,"candidate":plan["candidate"],"profile":plan["profile"],"fold_id":plan["fold_id"],"sector_filter":plan.get("sector_filter"),"exclude_ticker":plan.get("exclude_ticker"),"train_rows":len(tr),"validation_rows":len(va),"elapsed_seconds":time.time()-started,"npz_path":str(npz)}; atomic_json(res,jp); return res
    except Exception as e:
        res={"status":"failed","bundle_id":plan.get("bundle_id"),"error":repr(e),"traceback":traceback.format_exc()}; atomic_json(res,jp); return res


def _lgb(p): return _run(p,"lightgbm")
def _xgb(p): return _run(p,"xgboost")

def _gpu_entry(plans,workers,path):
    ctx=mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex: out=list(ex.map(_xgb,plans))
    atomic_json(out,Path(path))

def run_parallel(lgb_plans,xgb_plans,*,lgb_workers:int,xgb_workers:int,scratch_dir:Path):
    scratch_dir.mkdir(parents=True,exist_ok=True); ctx=mp.get_context("spawn"); gp=None; result_path=scratch_dir/f"gpu_{canonical_hash([p['bundle_id'] for p in xgb_plans],16)}.json"
    if xgb_plans:
        gp=ctx.Process(target=_gpu_entry,args=(xgb_plans,xgb_workers,str(result_path))); gp.start()
    with cf.ProcessPoolExecutor(max_workers=lgb_workers,mp_context=ctx) as ex: lr=list(ex.map(_lgb,lgb_plans)) if lgb_plans else []
    if gp:
        gp.join();
        if gp.exitcode!=0: raise RuntimeError(f"GPU pool exit={gp.exitcode}")
        xr=read_json(result_path,[])
    else: xr=[]
    bad=[r for r in lr+xr if r.get("status")!="complete"]
    oom={r.get("bundle_id") for r in bad if "out of memory" in (str(r.get("error"))+str(r.get("traceback"))).lower()}
    if oom:
        retry=[p for p in xgb_plans if p["bundle_id"] in oom]
        retry_path=scratch_dir/f"gpu_retry_{canonical_hash(sorted(oom),16)}.json"
        _gpu_entry(retry,1,str(retry_path)); retried=read_json(retry_path,[])
        by_id={r.get("bundle_id"):r for r in xr}; by_id.update({r.get("bundle_id"):r for r in retried}); xr=list(by_id.values())
        bad=[r for r in lr+xr if r.get("status")!="complete"]
    if bad: raise RuntimeError(f"failed bundles: {bad[:3]}")
    return lr,xr
