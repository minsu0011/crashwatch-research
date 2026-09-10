from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import file_sha256, set_worker_mode
from .regimes import REGIMES, build_market_regimes
from .scoring import horizon_positive_recall, metrics_by_regime
from .target import build_3d5_target_from_ret1

DATE_CANDIDATES = ["date", "trade_date", "trading_date", "datetime", "dt", "일자", "날짜"]
TICKER_CANDIDATES = ["ticker", "stock_code", "code", "symbol", "종목코드", "단축코드"]


def _detect(names: list[str], candidates: list[str], kind: str) -> str:
    lower = {str(n).lower(): str(n) for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"{kind} column not found")


def _load_lock(selection_output: Path):
    lock = read_json(selection_output / "LOCKED_3D5_RECIPE.json", {})
    feat = read_json(selection_output / "LOCKED_3D5_FEATURE_MANIFEST.json", {})
    meta = read_json(selection_output / "RUNTIME_MANIFEST.json", {})
    if lock.get("status") != "LOCKED_FOR_FINAL_SEALED":
        raise RuntimeError(f"development recipe not locked: {lock}")
    if feat.get("profile") != lock.get("locked_profile"):
        raise RuntimeError("feature manifest and recipe mismatch")
    return lock, feat, meta


def _combined_regime_calendar(selection_output: Path, sealed_frame: pd.DataFrame, date_col: str, market_feature: str, config: dict[str, Any]) -> pd.DataFrame:
    _, _, meta = _load_lock(selection_output)
    dates_dev = np.load(meta["dates_path"], mmap_mode="r")
    X = np.load(meta["all_x_path"], mmap_mode="r")
    feature_names = list(meta["feature_names"]); idx = feature_names.index(market_feature)
    dev_market = np.asarray(X[:, idx], dtype=float)
    sealed_dates = pd.to_datetime(sealed_frame[date_col]).astype("int64").to_numpy(dtype=np.int64)
    sealed_market = pd.to_numeric(sealed_frame[market_feature], errors="coerce").to_numpy(dtype=float)
    dates = np.concatenate([np.asarray(dates_dev, dtype=np.int64), sealed_dates])
    market = np.concatenate([dev_market, sealed_market])
    rcfg = config["regime"]
    result = build_market_regimes(
        dates, market,
        high_vol_percentile=float(rcfg["high_vol_percentile"]), bull_20d=float(rcfg["bull_20d"]), bear_20d=float(rcfg["bear_20d"]),
        crash_5d=float(rcfg["crash_5d"]), crash_20d=float(rcfg["crash_20d"]), rebound_5d=float(rcfg["rebound_5d"]),
        rebound_drawdown_60=float(rcfg["rebound_drawdown_60"]), vol_rank_window=int(rcfg["vol_rank_window"]),
    )
    sealed_min = int(sealed_dates.min()); sealed_max = int(sealed_dates.max())
    return result.calendar[(result.calendar.date_ns >= sealed_min) & (result.calendar.date_ns <= sealed_max)].copy()


def preflight(selection_output: Path, sealed_dataset: Path, config: dict[str, Any]) -> dict[str, Any]:
    selection_output = selection_output.resolve(); sealed_dataset = sealed_dataset.resolve()
    lock, feat, meta = _load_lock(selection_output)
    if not sealed_dataset.exists() or sealed_dataset.suffix.lower() != ".parquet":
        raise FileNotFoundError(sealed_dataset)
    import pyarrow.parquet as pq
    names = pq.ParquetFile(sealed_dataset).schema_arrow.names
    date_col = _detect(names, DATE_CANDIDATES, "date"); ticker_col = _detect(names, TICKER_CANDIDATES, "ticker")
    ret_feature = str(meta["return_feature"]); market_feature = str(meta["market_proxy_feature"])
    required = [date_col, ticker_col, ret_feature, market_feature] + list(feat["features"])
    missing = [c for c in required if c not in names]
    if missing:
        raise KeyError(f"final sealed required columns missing ({len(missing)}): {missing[:40]}")
    small = pd.read_parquet(sealed_dataset, columns=[date_col, market_feature], engine="pyarrow")
    small[date_col] = pd.to_datetime(small[date_col], errors="raise")
    dev_max = pd.Timestamp(meta["dataset_date_max"])
    sealed_min = small[date_col].min(); sealed_max = small[date_col].max()
    if not sealed_min > dev_max:
        raise ValueError(f"FINAL_SEALED must start after development max: dev={dev_max.date()}, final={sealed_min.date()}")
    cal = _combined_regime_calendar(selection_output, small, date_col, market_feature, config)
    counts = cal["regime"].value_counts().to_dict()
    min_dates = int(config["final_sealed"]["min_dates_per_regime"])
    missing_regimes = [r for r in REGIMES if int(counts.get(r, 0)) < min_dates]
    final_dir = selection_output / "final_sealed"; final_dir.mkdir(parents=True, exist_ok=True)
    cal.to_csv(final_dir / "FINAL_SEALED_REGIME_CALENDAR_PREFLIGHT.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "READY" if not missing_regimes else "NOT_READY_REGIME_COVERAGE",
        "selection_output": str(selection_output), "locked_profile": lock["locked_profile"],
        "sealed_dataset": str(sealed_dataset), "sealed_sha256": file_sha256(sealed_dataset),
        "date_column": date_col, "ticker_column": ticker_col, "return_feature": ret_feature, "market_proxy_feature": market_feature,
        "sealed_date_min": sealed_min.strftime("%Y-%m-%d"), "sealed_date_max": sealed_max.strftime("%Y-%m-%d"),
        "development_date_max": dev_max.strftime("%Y-%m-%d"), "unique_dates": int(small[date_col].nunique()),
        "regime_date_counts": {r: int(counts.get(r, 0)) for r in REGIMES}, "min_dates_per_regime": min_dates,
        "missing_or_undercovered_regimes": missing_regimes,
        "target_labels_read": False, "predictions_computed": False,
        "note": "preflight reads dates and trailing market proxy only; it does not construct the 3d/5% target",
    }
    atomic_json(payload, final_dir / "FINAL_SEALED_PREFLIGHT.json")
    return payload


def _lgb_final_worker(plan: dict[str, Any]) -> dict[str, Any]:
    set_worker_mode(int(plan["threads"]), "above_normal")
    import lightgbm as lgb
    Xtr=np.load(plan["x_train"],mmap_mode="r"); ytr=np.load(plan["y_train"],mmap_mode="r"); Xte=np.load(plan["x_test"],mmap_mode="r")
    pos=max(1,int(np.sum(ytr==1))); neg=max(1,int(np.sum(ytr==0)))
    cfg=dict(plan["params"]); rounds=int(cfg.pop("rounds")); mult=float(cfg.pop("class_weight_multiplier",1.0))
    params={"objective":"binary","metric":"None","verbosity":-1,"deterministic":True,"force_col_wise":True,"feature_pre_filter":False,"num_threads":int(plan["threads"]),"scale_pos_weight":neg/pos*mult,"seed":int(plan["seed"]),"feature_fraction_seed":int(plan["seed"]),"bagging_seed":int(plan["seed"])}; params.update(cfg)
    ds=lgb.Dataset(np.asarray(Xtr),label=np.asarray(ytr),free_raw_data=True,params={"max_bin":int(params.get("max_bin",255)),"feature_pre_filter":False})
    model=lgb.train(params,ds,num_boost_round=rounds,callbacks=[lgb.log_evaluation(0)])
    pred=np.asarray(model.predict(np.asarray(Xte)),dtype=np.float32); np.save(plan["out"],pred)
    return {"seed":int(plan["seed"]),"out":plan["out"]}


def _xgb_final_group(plan: dict[str, Any], queue: mp.Queue) -> None:
    try:
        set_worker_mode(int(plan["threads"]), "above_normal"); os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import xgboost as xgb
        Xtr=np.load(plan["x_train"],mmap_mode="r"); ytr=np.load(plan["y_train"],mmap_mode="r"); Xte=np.load(plan["x_test"],mmap_mode="r")
        cfg=dict(plan["params"]); rounds=int(cfg.pop("rounds")); mult=float(cfg.pop("class_weight_multiplier",1.0)); max_bin=int(cfg.get("max_bin",256))
        dtr=xgb.QuantileDMatrix(np.asarray(Xtr),label=np.asarray(ytr),max_bin=max_bin,nthread=int(plan["threads"])); dte=xgb.QuantileDMatrix(np.asarray(Xte),ref=dtr,max_bin=max_bin,nthread=int(plan["threads"]))
        pos=max(1,int(np.sum(ytr==1))); neg=max(1,int(np.sum(ytr==0))); results=[]
        for seed in plan["seeds"]:
            params={"objective":"binary:logistic","eval_metric":"aucpr","device":"cuda","tree_method":"hist","scale_pos_weight":neg/pos*mult,"seed":int(seed),"verbosity":0,"nthread":int(plan["threads"])}; params.update(cfg)
            model=xgb.train(params,dtr,num_boost_round=rounds,verbose_eval=False); pred=np.asarray(model.predict(dte),dtype=np.float32)
            out=str(Path(plan["out_dir"])/f"xgb_seed_{seed}.npy"); np.save(out,pred); results.append({"seed":int(seed),"out":out})
        queue.put({"ok":True,"results":results})
    except Exception as exc:
        queue.put({"ok":False,"error":repr(exc),"traceback":traceback.format_exc()})


def run_once(selection_output: Path, sealed_dataset: Path, config: dict[str, Any], confirmation: str) -> dict[str, Any]:
    selection_output=selection_output.resolve(); sealed_dataset=sealed_dataset.resolve(); final_dir=selection_output/"final_sealed"
    pre=read_json(final_dir/"FINAL_SEALED_PREFLIGHT.json",{})
    if pre.get("status")!="READY": raise RuntimeError(f"FINAL_SEALED preflight not READY: {pre.get('status')}")
    if confirmation != str(config["final_sealed"]["confirmation_phrase"]): raise ValueError("final sealed confirmation phrase mismatch")
    if file_sha256(sealed_dataset)!=pre.get("sealed_sha256"): raise RuntimeError("final sealed file changed after preflight")
    started=final_dir/"FINAL_SEALED_STARTED.json"; consumed=final_dir/"FINAL_SEALED_CONSUMED.json"
    if started.exists() or consumed.exists(): raise RuntimeError("FINAL_SEALED already started/consumed; reuse forbidden")
    lock, feat, meta=_load_lock(selection_output)
    atomic_json({"status":"STARTED_IRREVERSIBLY","epoch":time.time(),"sealed_sha256":pre["sealed_sha256"],"locked_profile":lock["locked_profile"]},started)

    cols=list(dict.fromkeys([pre["date_column"],pre["ticker_column"],meta["return_feature"],meta["market_proxy_feature"]]+list(feat["features"])))
    frame=pd.read_parquet(sealed_dataset,columns=cols,engine="pyarrow"); frame[pre["date_column"]]=pd.to_datetime(frame[pre["date_column"]],errors="raise")
    frame=frame.sort_values([pre["date_column"],pre["ticker_column"]],kind="mergesort").reset_index(drop=True)
    dates=frame[pre["date_column"]].astype("int64").to_numpy(dtype=np.int64); tickers=frame[pre["ticker_column"]].astype(str).to_numpy()
    target=build_3d5_target_from_ret1(dates,tickers,pd.to_numeric(frame[meta["return_feature"]],errors="coerce").to_numpy(dtype=float),horizon_days=3,drop_threshold=-0.05)
    valid=target.valid; eval_frame=frame.loc[valid].reset_index(drop=True); y=target.label[valid]; hit=target.first_hit_day[valid]; eval_dates=dates[valid]; eval_tickers=tickers[valid]
    cal=pd.read_csv(final_dir/"FINAL_SEALED_REGIME_CALENDAR_PREFLIGHT.csv"); cal["date_ns"]=pd.to_numeric(cal["date_ns"],errors="raise").astype(np.int64); date_to_regime=dict(zip(cal.date_ns,cal.regime))
    regimes=np.asarray([date_to_regime.get(int(d),"UNKNOWN") for d in eval_dates],dtype=object)
    if any(r not in set(REGIMES) for r in regimes): raise RuntimeError("unclassified final sealed regime dates")

    # Development training matrix/target are already frozen from development stage.
    profile_matrix=np.load(meta["matrix_paths"][lock["locked_profile"]],mmap_mode="r"); ydev=np.load(meta["target_path"],mmap_mode="r"); vdev=np.load(meta["valid_path"],mmap_mode="r")
    train_idx=np.flatnonzero(np.asarray(vdev,dtype=bool)); Xtrain=np.asarray(profile_matrix[train_idx],dtype=np.float32); ytrain=np.asarray(ydev[train_idx],dtype=np.uint8)
    Xtest=eval_frame[list(feat["features"])].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float32)
    runtime=final_dir/"runtime_cache"; runtime.mkdir(parents=True,exist_ok=True); xtr=runtime/"x_train.npy"; ytr=runtime/"y_train.npy"; xte=runtime/"x_test.npy"
    np.save(xtr,Xtrain); np.save(ytr,ytrain); np.save(xte,Xtest); del Xtrain,ytrain,Xtest
    seeds=list(map(int,config["selection"]["confirm_seeds"])); weight=float(lock["xgb_weight"])
    context=mp.get_context("spawn"); q=context.Queue(); xgb_proc=None
    xgb_results=[]; lgb_results=[]
    if weight>0:
        assignments=[seeds[::2],seeds[1::2]]; procs=[]
        for i,ss in enumerate(assignments):
            if not ss: continue
            plan={"x_train":str(xtr),"y_train":str(ytr),"x_test":str(xte),"out_dir":str(final_dir),"seeds":ss,"params":lock["xgboost_params"],"threads":int(config["cpu"]["threads_per_xgboost_worker"])}
            p=context.Process(target=_xgb_final_group,args=(plan,q),name=f"final-xgb-{i}"); p.start(); procs.append(p)
        xgb_proc=procs
    if weight<1:
        plans=[{"x_train":str(xtr),"y_train":str(ytr),"x_test":str(xte),"out":str(final_dir/f"lgb_seed_{s}.npy"),"seed":s,"params":lock["lightgbm_params"],"threads":4} for s in seeds]
        with cf.ProcessPoolExecutor(max_workers=5,mp_context=context) as ex: lgb_results=list(ex.map(_lgb_final_worker,plans))
    if xgb_proc:
        for p in xgb_proc: p.join()
        messages=[q.get(timeout=20) for _ in xgb_proc]
        if any(not m.get("ok") for m in messages): raise RuntimeError(f"final XGB failed: {messages}")
        xgb_results=[r for m in messages for r in m["results"]]
    lgb_mean=None if not lgb_results else np.vstack([np.load(r["out"]) for r in sorted(lgb_results,key=lambda x:x["seed"])]).mean(axis=0)
    xgb_mean=None if not xgb_results else np.vstack([np.load(r["out"]) for r in sorted(xgb_results,key=lambda x:x["seed"])]).mean(axis=0)
    if weight<=0: final_pred=lgb_mean
    elif weight>=1: final_pred=xgb_mean
    else: final_pred=(1-weight)*lgb_mean+weight*xgb_mean
    pred=pd.DataFrame({"date":pd.to_datetime(eval_dates).strftime("%Y-%m-%d"),"ticker":eval_tickers,"target":y,"first_hit_day":hit,"regime":regimes,"prediction":final_pred})
    score,regime_metrics=metrics_by_regime(pred); hit_metrics=horizon_positive_recall(pred)
    pred.to_parquet(final_dir/"FINAL_SEALED_PREDICTIONS.parquet",index=False); regime_metrics.to_csv(final_dir/"FINAL_SEALED_REGIME_METRICS.csv",index=False,encoding="utf-8-sig"); hit_metrics.to_csv(final_dir/"FINAL_SEALED_HIT_DAY_RECALL.csv",index=False,encoding="utf-8-sig")
    result={"status":"CONSUMED_COMPLETED","completed_epoch":time.time(),"locked_profile":lock["locked_profile"],"feature_count":len(feat["features"]),"xgb_weight":weight,"target_audit":target.audit,"valid_eval_rows":int(valid.sum()),"selection_score_reporting_only":score["selection_score"],"overall_metrics":score["overall"],"regime_metrics_file":"FINAL_SEALED_REGIME_METRICS.csv","no_tuning_performed":True,"sealed_reuse_forbidden":True}
    atomic_json(result,final_dir/"FINAL_SEALED_RESULT.json"); atomic_json(result,consumed)
    return result
