from __future__ import annotations

import concurrent.futures as cf
import gc
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.gpu_worker import detect_cuda
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, bh_adjust, bootstrap_mean_ci, canonical_hash, exact_sign_flip_p, hash_strings, read_json

from .common import write_eta

LOGGER = logging.getLogger(__name__)
VERSION = "cw_xgb_profile_confirmatory_v1"


def _train(xgb: Any, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, cfg: dict[str, Any], seed: int) -> np.ndarray:
    max_bin = int(cfg.get("max_bin", 256))
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=max_bin, nthread=1)
    dvalid = xgb.QuantileDMatrix(x_valid, ref=dtrain, max_bin=max_bin, nthread=1)
    pos=max(1,int((y_train==1).sum())); neg=max(1,int((y_train==0).sum()))
    params={
        "objective":"binary:logistic","eval_metric":"aucpr","device":"cuda","tree_method":"hist",
        "learning_rate":float(cfg.get("learning_rate",0.035)),"max_depth":int(cfg.get("max_depth",0)),
        "max_leaves":int(cfg.get("max_leaves",96)),"grow_policy":str(cfg.get("grow_policy","lossguide")),
        "min_child_weight":float(cfg.get("min_child_weight",7.0)),"subsample":float(cfg.get("subsample",0.90)),
        "colsample_bytree":float(cfg.get("colsample_bytree",0.85)),"reg_alpha":float(cfg.get("reg_alpha",0.2)),
        "reg_lambda":float(cfg.get("reg_lambda",1.8)),"max_bin":max_bin,"scale_pos_weight":float(neg/pos),
        "seed":int(seed),"nthread":1,"verbosity":0,
    }
    model=xgb.train(params,dtrain,num_boost_round=int(cfg.get("rounds",260)),verbose_eval=False)
    pred=np.asarray(model.predict(dvalid),dtype=np.float32)
    del model,dtrain,dvalid
    return pred


def _worker(plan: dict[str,Any]) -> dict[str,Any]:
    import xgboost as xgb
    root=Path(plan["cache_root"]); X=np.load(root/"X_all_valid.npy",mmap_mode="r"); y=np.load(root/"target.npy",mmap_mode="r"); dates=np.load(root/"dates_ns.npy",mmap_mode="r")
    out=Path(plan["output_dir"]); out.mkdir(parents=True,exist_ok=True); completed=cached=failed=0
    index={f:i for i,f in enumerate(plan["feature_names"])}
    for fold in plan["folds"]:
        tr=slice(fold["train_start"],fold["train_stop"]); va=slice(fold["validation_start"],fold["validation_stop"])
        y_train=np.asarray(y[tr],dtype=np.uint8); y_valid=np.asarray(y[va],dtype=np.uint8); valid_dates=np.asarray(dates[va],dtype=np.int64)
        for profile,features in plan["profiles"].items():
            idx=np.asarray([index[f] for f in features],dtype=np.int32)
            x_train=np.ascontiguousarray(X[tr][:,idx],dtype=np.float32); x_valid=np.ascontiguousarray(X[va][:,idx],dtype=np.float32)
            for seed in plan["seeds"]:
                identity={"version":VERSION,"signature":plan["signature"],"fold":fold,"profile":profile,"profile_hash":hash_strings(features),"seed":seed,"config":plan["config"]}
                tid=canonical_hash(identity); path=out/f"{tid}.json"; old=read_json(path,{})
                if old.get("status")=="completed" and old.get("identity")==identity: cached+=1; continue
                start=time.perf_counter()
                try:
                    pred=_train(xgb,x_train,y_train,x_valid,plan["config"],seed)
                    atomic_json({"status":"completed","identity":identity,"task_id":tid,"backend":"xgboost_cuda","test_type":"profile_benchmark","profile":profile,"feature_count":len(features),"seed":seed,"outer_fold":fold["fold_id"],"elapsed_seconds":time.perf_counter()-start,**compute_metrics(y_valid,pred,valid_dates)},path)
                    completed+=1; del pred
                except Exception as exc:
                    failed+=1; atomic_json({"status":"failed","identity":identity,"task_id":tid,"profile":profile,"seed":seed,"outer_fold":fold["fold_id"],"error":repr(exc),"traceback":traceback.format_exc()},path)
            del x_train,x_valid; gc.collect()
    return {"completed":completed,"cached":cached,"failed":failed}


def run_xgb_profile_confirmatory(prepared:Any,folds:list[Any],previous_output:Path,profiles:dict[str,list[str]],config:dict[str,Any],*,workers:int,seeds:list[int],profile_names:list[str]) -> dict[str,Any]:
    gpu=detect_cuda(); out=previous_output/"confirmatory_v2"; task_dir=out/"task_results"/"xgboost_profiles"; task_dir.mkdir(parents=True,exist_ok=True)
    if not gpu.get("available"):
        result={"status":"skipped","gpu":gpu}; atomic_json(result,out/"xgb_profile_summary.json"); return result
    selected={k:profiles[k] for k in profile_names if k in profiles}
    signature=canonical_hash({"version":VERSION,"profiles":{k:hash_strings(v) for k,v in selected.items()},"seeds":seeds,"config":config})
    assignments=[[] for _ in range(max(1,workers))]
    for i,fold in enumerate(sorted(folds,key=lambda f:f.fold_id,reverse=True)): assignments[i%len(assignments)].append(f.to_dict())
    plans=[{"cache_root":str(prepared.root),"output_dir":str(task_dir),"feature_names":prepared.feature_names,"folds":a,"profiles":selected,"seeds":seeds,"config":config,"signature":signature} for a in assignments if a]
    started=time.time(); results=[]; ctx=mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=len(plans),mp_context=ctx) as ex:
        futures=[ex.submit(_worker,p) for p in plans]
        for i,f in enumerate(cf.as_completed(futures),1):
            results.append(f.result()); write_eta(out,"xgb_profile_confirmatory",i,len(plans),time.time()-started)
    rows=[]
    for p in task_dir.glob("*.json"):
        d=read_json(p,{})
        if d.get("status")=="completed": rows.append(d)
    metrics=pd.DataFrame(rows); metrics.to_csv(out/"xgb_profile_metrics.csv",index=False)
    if not metrics.empty:
        base=metrics[metrics.profile.eq("P0_FULL_439")][["outer_fold","seed","raw_pr_auc"]].rename(columns={"raw_pr_auc":"base_pr"})
        paired=metrics.merge(base,on=["outer_fold","seed"],how="inner"); paired["delta_raw_pr_auc"]=paired.raw_pr_auc-paired.base_pr; paired.to_csv(out/"xgb_profile_paired.csv",index=False)
        fold=paired.groupby(["profile","outer_fold"],as_index=False).delta_raw_pr_auc.mean()
        summary=[]
        for profile,g in fold.groupby("profile"):
            vals=g.delta_raw_pr_auc.to_numpy(float); lo,hi=bootstrap_mean_ci(vals); recent=g[g.outer_fold.isin([4,5,6,7])].delta_raw_pr_auc.to_numpy(float)
            summary.append({"profile":profile,"fold_count":len(vals),"mean_delta":vals.mean(),"median_delta":np.median(vals),"ci95_low":lo,"ci95_high":hi,"positive_fold_ratio":np.mean(vals>0),"worst_fold":vals.min(),"recent_mean_delta":recent.mean() if len(recent) else np.nan,"exact_sign_flip_p":exact_sign_flip_p(vals)})
        s=pd.DataFrame(summary); s["bh_q"]=bh_adjust(s.exact_sign_flip_p.to_numpy(float)); s.to_csv(out/"xgb_profile_summary.csv",index=False)
    result={"status":"completed" if sum(r["failed"] for r in results)==0 else "partial","gpu":gpu,"workers":workers,"completed":sum(r["completed"] for r in results),"cached":sum(r["cached"] for r in results),"failed":sum(r["failed"] for r in results),"elapsed_seconds":time.time()-started}
    atomic_json(result,out/"xgb_profile_run_summary.json"); return result
