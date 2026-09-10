from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_surge_magnitude_direction_v13 import prepare_runtime, RuntimeData
from surge_v13_data import TARGET_COLUMN, MOVE_COLUMN, V11_LAG0_PRIMARY_DIRECTION_COLUMNS
from surge_v13_models import StageSpec, fit_binary_model, predict_binary_model, make_sample_weights, safe_pr_auc, safe_roc_auc, gpu_preflight
from surge_v14_models import (
    SCHEMA, DOWN_COLUMN, FIRST_UP_DAY_COLUMN,
    add_competing_risk_labels, crossfit_rank_features, deduplicate_ranked_features,
    magnitude_balancing_weights, fit_nonnegative_stacker, predict_nonnegative_stacker,
    build_ranked_policy_score, hardfp_candidate_mask, select_threshold, policy_metrics, oracle_topk,
)

CACHE_REVISION = "v14_20260816_r02"


def log(msg: str) -> None:
    print(f"[V14] {msg}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


@dataclass(frozen=True)
class V14Config:
    config_id: str
    direct_k: int
    hazard_k: int
    down_k: int
    hardfp_k: int
    hardfp_quantile: float
    direct_depth: int = 5
    auxiliary_depth: int = 4
    recency_half_life: float = 756.0
    include_v11_lag0_hardfp: bool = True

    def key(self) -> str:
        return hashlib.sha1(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def predefined_configs(fast: bool = False) -> list[V14Config]:
    if fast:
        return [V14Config("V14_C1_DIRECT64_HFP24", 64, 48, 48, 24, 0.72)]
    return [
        V14Config("V14_C1_DIRECT64_HFP24", 64, 48, 48, 24, 0.72),
        V14Config("V14_C2_DIRECT96_HFP32", 96, 64, 64, 32, 0.75),
        V14Config("V14_C3_DIRECT128_HFP48", 128, 96, 96, 48, 0.78),
        V14Config("V14_C4_DIRECT160_HFP48", 160, 96, 96, 48, 0.72),
    ]


def load_v13_oof(v13_output: Path) -> pd.DataFrame:
    candidates = [v13_output / "v13_oof_predictions.csv.gz", v13_output / "v13_oof_predictions.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"V14 requires full V13 OOF predictions. Not found under {v13_output}. "
            "Use the actual V13 output directory, not the compact compact result archive."
        )
    df = pd.read_csv(path, dtype={"ticker": str})
    required = {
        "source_row_id", "fold_id", "ticker", TARGET_COLUMN, "p_move", "p_up_given_move",
        "stage_probability", "policy_score", "base_past_rank",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"V13 OOF missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    rename = {
        "p_move": "v13_p_move", "p_up_given_move": "v13_p_up_given_move",
        "stage_probability": "v13_stage_probability", "policy_score": "v13_policy_score",
        "base_past_rank": "v10_base_past_rank", "base_score_raw": "v10_base_score_raw",
    }
    df = df.rename(columns=rename)
    keep = [
        "source_row_id", "fold_id", "ticker", "v13_p_move", "v13_p_up_given_move",
        "v13_stage_probability", "v13_policy_score", "v10_base_past_rank",
    ]
    if "v10_base_score_raw" in df:
        keep.append("v10_base_score_raw")
    return df[keep].copy()


def build_validation_frame(runtime: RuntimeData, folds: Sequence[int], v13_oof: pd.DataFrame) -> pd.DataFrame:
    parts=[]
    for fold in folds:
        idx=np.asarray(runtime.fold_index[int(fold)]["validation"],dtype=np.int64)
        part=runtime.frame.iloc[idx].copy()
        part["row_index"] = idx
        part["fold_id"] = int(fold)
        lagcols=[c for c in V11_LAG0_PRIMARY_DIRECTION_COLUMNS if c in runtime.lag0_features.columns]
        if lagcols:
            lag=runtime.lag0_features.iloc[idx][lagcols].reset_index(drop=True)
            for c in lagcols: part[c]=lag[c].to_numpy()
        parts.append(part)
    out=pd.concat(parts,ignore_index=True)
    join=v13_oof.copy()
    out=out.merge(join,on=["source_row_id","fold_id","ticker"],how="left",validate="one_to_one")
    miss=out["v13_stage_probability"].isna().sum()
    if miss:
        raise RuntimeError(f"V13 OOF join missing rows: {miss}/{len(out)}")
    return out


def model_spec(depth: int, *, fast: bool, estimators: int = 1000, lr: float = 0.035) -> StageSpec:
    return StageSpec(
        backend="xgb", feature_k=0, n_estimators=(180 if fast else int(estimators)),
        learning_rate=float(lr), max_depth=int(depth), min_child_weight=5.0,
        subsample=0.82, colsample_bytree=0.72, reg_alpha=0.6, reg_lambda=4.5,
    )


def _fit_predict_head(
    train: pd.DataFrame, valid: pd.DataFrame, features: Sequence[str], target: str,
    *, spec: StageSpec, seed: int, cpu_threads: int, use_gpu: bool,
    recency_half_life: float, custom_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    if custom_weight is None:
        weights=make_sample_weights(
            train,pd.to_numeric(train[target],errors="raise").to_numpy(np.int8),
            half_life_days=float(recency_half_life),
        )
    else:
        weights=np.asarray(custom_weight,dtype=float)
    bundle=fit_binary_model(
        train,list(features),target,spec,sample_weight=weights,seed=int(seed),
        cpu_threads=int(cpu_threads),use_gpu=bool(use_gpu),ticker_onehot=True,market_bucket_onehot=True,
    )
    return predict_binary_model(bundle,valid), bool(bundle.gpu_used)


def fit_predict_experts_fold(
    runtime: RuntimeData, config: V14Config, fold: int, seed: int,
    *, direct_features: Sequence[str], hazard_features: Sequence[str], down_features: Sequence[str],
    v13_oof: pd.DataFrame, cpu_threads: int, use_gpu: bool, fast: bool,
) -> tuple[pd.DataFrame, dict[str,Any]]:
    train_idx=np.asarray(runtime.fold_index[int(fold)]["train"],dtype=np.int64)
    valid_idx=np.asarray(runtime.fold_index[int(fold)]["validation"],dtype=np.int64)
    train=runtime.frame.iloc[train_idx].copy().reset_index(drop=True)
    valid=runtime.frame.iloc[valid_idx].copy().reset_index(drop=True)
    valid["row_index"] = valid_idx
    valid["fold_id"] = int(fold)
    t0=time.monotonic()
    direct_spec=model_spec(config.direct_depth,fast=fast,estimators=1200,lr=0.035)
    aux_spec=model_spec(config.auxiliary_depth,fast=fast,estimators=850,lr=0.04)

    p_direct,gpu_direct=_fit_predict_head(
        train,valid,direct_features,TARGET_COLUMN,spec=direct_spec,seed=seed+11,cpu_threads=cpu_threads,
        use_gpu=use_gpu,recency_half_life=config.recency_half_life,
    )

    # Discrete-time upward crossing hazards. Official target remains the direct D3 head.
    hazard_preds=[]; hazard_gpu=[]
    first=pd.to_numeric(train[FIRST_UP_DAY_COLUMN],errors="coerce").to_numpy(int)
    for day in (1,2,3):
        risk=(first<=0) | (first>=day)
        htrain=train.loc[risk].copy()
        col=f"__hazard_up_d{day}"
        htrain[col]=(pd.to_numeric(htrain[FIRST_UP_DAY_COLUMN],errors="coerce").to_numpy(int)==day).astype(np.int8)
        if htrain[col].nunique()<2:
            hazard_preds.append(np.full(len(valid),float(htrain[col].mean() if len(htrain) else 0.0)))
            hazard_gpu.append(False); continue
        pred,gpu=_fit_predict_head(
            htrain,valid,hazard_features,col,spec=aux_spec,seed=seed+100+day,cpu_threads=cpu_threads,
            use_gpu=use_gpu,recency_half_life=config.recency_half_life,
        )
        hazard_preds.append(pred); hazard_gpu.append(gpu)
    h1,h2,h3=hazard_preds
    p_hazard=1.0-(1.0-h1)*(1.0-h2)*(1.0-h3)

    down_valid=train[DOWN_COLUMN].isin([0,1])
    dtrain=train.loc[down_valid].copy()
    p_down,gpu_down=_fit_predict_head(
        dtrain,valid,down_features,DOWN_COLUMN,spec=aux_spec,seed=seed+211,cpu_threads=cpu_threads,
        use_gpu=use_gpu,recency_half_life=config.recency_half_life,
    )

    meta=["source_row_id","date","ticker","market","bucket",TARGET_COLUMN,MOVE_COLUMN,DOWN_COLUMN,FIRST_UP_DAY_COLUMN]
    pred=valid[meta].copy()
    pred["row_index"]=valid_idx; pred["fold_id"]=int(fold); pred["seed"]=int(seed)
    pred["p_direct"]=p_direct; pred["p_hazard_up"]=np.clip(p_hazard,1e-8,1-1e-8); pred["p_down"]=p_down
    pred["p_hazard_d1"]=h1; pred["p_hazard_d2"]=h2; pred["p_hazard_d3"]=h3
    fixed=v13_oof.loc[v13_oof["fold_id"].eq(int(fold))].copy()
    pred=pred.merge(fixed,on=["source_row_id","fold_id","ticker"],how="left",validate="one_to_one")
    if pred["v13_stage_probability"].isna().any():
        raise RuntimeError(f"V13 OOF missing fixed expert rows in fold {fold}")
    diagnostic={
        "config_id":config.config_id,"fold_id":int(fold),"seed":int(seed),"train_rows":len(train),"valid_rows":len(valid),
        "direct_features":len(direct_features),"hazard_features":len(hazard_features),"down_features":len(down_features),
        "gpu_direct":gpu_direct,"gpu_hazard_all":bool(all(hazard_gpu)),"gpu_down":gpu_down,
        "seconds":float(time.monotonic()-t0),
    }
    return pred,diagnostic


def _feature_list_hash(features: Sequence[str]) -> str:
    payload="\n".join(map(str,features)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def fit_predict_experts_fold_cached(
    runtime: RuntimeData, config: V14Config, fold: int, seed: int,
    *, direct_features: Sequence[str], hazard_features: Sequence[str], down_features: Sequence[str],
    v13_oof: pd.DataFrame, cpu_threads: int, use_gpu: bool, fast: bool,
    cache_dir: Path, resume: bool, v13_fingerprint: str,
) -> tuple[pd.DataFrame, dict[str,Any]]:
    """Persistent cache for the expensive five-head GPU fold fit.

    The key includes data fingerprint, config, fold/seed, code revision and exact
    feature lists, so stale V13/V14 experiments cannot silently be reused.
    """
    cache_dir.mkdir(parents=True,exist_ok=True)
    payload={
        "revision":CACHE_REVISION,"runtime":str(runtime.cache_fingerprint),"v13_oof":str(v13_fingerprint),
        "config":config.key(),"fold":int(fold),"seed":int(seed),"fast":bool(fast),
        "direct":_feature_list_hash(direct_features),"hazard":_feature_list_hash(hazard_features),
        "down":_feature_list_hash(down_features),
    }
    key=hashlib.sha1(json.dumps(payload,sort_keys=True).encode("utf-8")).hexdigest()[:24]
    pred_path=cache_dir/f"expert_{key}.pkl"; diag_path=cache_dir/f"expert_{key}.json"
    if resume and pred_path.exists() and diag_path.exists():
        try:
            pred=pd.read_pickle(pred_path)
            diag=json.loads(diag_path.read_text(encoding="utf-8"))
            required={"source_row_id","fold_id","ticker","p_direct","p_hazard_up","p_down","v13_stage_probability"}
            if required.issubset(pred.columns) and len(pred)==len(runtime.fold_index[int(fold)]["validation"]):
                diag=dict(diag); diag["cache_hit"]=True
                return pred,diag
        except Exception:
            pass
    pred,diag=fit_predict_experts_fold(
        runtime,config,fold,seed,direct_features=direct_features,hazard_features=hazard_features,
        down_features=down_features,v13_oof=v13_oof,cpu_threads=cpu_threads,use_gpu=use_gpu,fast=fast,
    )
    tmp=pred_path.with_suffix(".pkl.tmp"); pred.to_pickle(tmp); os.replace(tmp,pred_path)
    tmpj=diag_path.with_suffix(".json.tmp"); tmpj.write_text(json.dumps({**diag,"cache_key":payload},ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmpj,diag_path)
    diag=dict(diag); diag["cache_hit"]=False
    return pred,diag


def ensemble_experts(predictions: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not predictions: raise ValueError("No expert predictions")
    keys=["source_row_id","date","ticker","market","bucket",TARGET_COLUMN,MOVE_COLUMN,DOWN_COLUMN,FIRST_UP_DAY_COLUMN,"row_index","fold_id"]
    first=predictions[0].sort_values(["fold_id","source_row_id"],kind="mergesort").reset_index(drop=True)
    out=first[keys].copy()
    avg_cols=["p_direct","p_hazard_up","p_down","p_hazard_d1","p_hazard_d2","p_hazard_d3"]
    for c in avg_cols:
        mats=[]
        for p in predictions:
            q=p.sort_values(["fold_id","source_row_id"],kind="mergesort").reset_index(drop=True)
            if not np.array_equal(q["source_row_id"].to_numpy(),first["source_row_id"].to_numpy()):
                raise ValueError("Expert ensemble row mismatch")
            mats.append(pd.to_numeric(q[c],errors="coerce").to_numpy(float))
        out[c]=np.mean(np.vstack(mats),axis=0)
    fixed=["v13_p_move","v13_p_up_given_move","v13_stage_probability","v13_policy_score","v10_base_past_rank"]
    if "v10_base_score_raw" in first: fixed.append("v10_base_score_raw")
    for c in fixed: out[c]=first[c].to_numpy()
    return out


def concatenate_expert_folds(predictions: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Join disjoint fold predictions without treating folds as ensemble members."""
    if not predictions:
        raise ValueError("No fold predictions")
    out=pd.concat(list(predictions),ignore_index=True)
    required={"source_row_id","fold_id"}
    if not required.issubset(out.columns):
        raise ValueError(f"Fold predictions missing keys: {sorted(required-set(out.columns))}")
    duplicates=int(out.duplicated(["source_row_id","fold_id"]).sum())
    if duplicates:
        raise ValueError(f"Duplicate expert fold rows: {duplicates}")
    return out.sort_values(["fold_id","source_row_id"],kind="mergesort").reset_index(drop=True)


def merge_second_level_features(pred: pd.DataFrame, validation_frame: pd.DataFrame, hardfp_raw: Sequence[str], lag0_cols: Sequence[str]) -> pd.DataFrame:
    # Only merge raw/regime columns not already carried by the expert prediction frame.
    # This prevents accidental _x/_y expert-score columns in synthetic/custom workflows.
    addcols=[c for c in [*list(hardfp_raw),*list(lag0_cols)] if c in validation_frame.columns and c not in pred.columns]
    cols=["source_row_id","fold_id",*list(dict.fromkeys(addcols))]
    extra=validation_frame[cols].copy()
    out=pred.merge(extra,on=["source_row_id","fold_id"],how="left",validate="one_to_one")
    out["one_minus_p_down"]=1.0-pd.to_numeric(out["p_down"],errors="coerce")
    out["expert_seed_score"]=np.nanmax(np.column_stack([
        pd.to_numeric(out["p_direct"],errors="coerce"),
        pd.to_numeric(out["p_hazard_up"],errors="coerce"),
        pd.to_numeric(out["v13_stage_probability"],errors="coerce"),
    ]),axis=1)
    return out


def _hardfp_spec(fast: bool) -> StageSpec:
    return StageSpec(backend="logit",feature_k=0,n_estimators=1,learning_rate=0,max_depth=0)


def fit_hardfp_model(train: pd.DataFrame, features: Sequence[str], quantile: float, seed: int):
    mask=hardfp_candidate_mask(train,"expert_seed_score",quantile)
    subset=train.loc[mask].copy()
    if len(subset)<100 or subset[TARGET_COLUMN].nunique()<2:
        subset=train.copy()
    base=make_sample_weights(subset,pd.to_numeric(subset[TARGET_COLUMN],errors="raise").to_numpy(np.int8),half_life_days=756)
    rank=subset.groupby(["ticker","fold_id"],sort=False)["expert_seed_score"].rank(pct=True,method="average").to_numpy(float)
    weights=base*(1.0+3.0*np.power(np.nan_to_num(rank,nan=0.5),4))
    return fit_binary_model(
        subset,list(features),TARGET_COLUMN,_hardfp_spec(False),sample_weight=weights,seed=int(seed),cpu_threads=8,use_gpu=False,
        ticker_onehot=True,market_bucket_onehot=True,
    ),len(subset)


def crossfit_second_level(
    discovery: pd.DataFrame, *, hardfp_features: Sequence[str], hardfp_quantile: float, seed: int,
    policy_mixes: Sequence[tuple[float,float,float]], minimum_alerts: int, target_precision: float,
) -> tuple[pd.DataFrame, dict[str,Any], Any, Any, tuple[float,float,float]]:
    folds=sorted(discovery["fold_id"].astype(int).unique().tolist())
    parts=[]
    for f in folds:
        tr=discovery.loc[~discovery["fold_id"].eq(f)].copy(); va=discovery.loc[discovery["fold_id"].eq(f)].copy()
        hardfp_model,_=fit_hardfp_model(tr,hardfp_features,hardfp_quantile,seed+f*101)
        va["p_hardfp"]=predict_binary_model(hardfp_model,va)
        parts.append(va)
    cross=pd.concat(parts,ignore_index=True).sort_values(["fold_id","source_row_id"],kind="mergesort").reset_index(drop=True)

    meta_cols=["p_direct","p_hazard_up","v13_stage_probability","v13_p_move","v13_p_up_given_move","p_hardfp","one_minus_p_down","v10_base_past_rank"]
    meta_parts=[]
    for f in folds:
        tr=cross.loc[~cross["fold_id"].eq(f)].copy(); va=cross.loc[cross["fold_id"].eq(f)].copy()
        seed_rank=tr.groupby(["ticker","fold_id"],sort=False)["expert_seed_score"].rank(pct=True,method="average").to_numpy(float)
        y=pd.to_numeric(tr[TARGET_COLUMN],errors="raise").to_numpy(np.int8)
        pos=max(int((y==1).sum()),1); neg=max(int((y==0).sum()),1)
        classw=np.where(y==1,len(y)/(2*pos),len(y)/(2*neg))
        weights=classw*(1.0+2.5*np.power(np.nan_to_num(seed_rank,nan=.5),4))
        stack=fit_nonnegative_stacker(tr,columns=meta_cols,target_column=TARGET_COLUMN,sample_weight=weights,l2=.25)
        va["meta_probability"]=predict_nonnegative_stacker(stack,va)
        meta_parts.append(va)
    cross=pd.concat(meta_parts,ignore_index=True).sort_values(["fold_id","source_row_id"],kind="mergesort").reset_index(drop=True)

    best=None
    for mix in policy_mixes:
        rw,hw,dw=mix
        scored=[]
        for f in folds:
            va=cross.loc[cross["fold_id"].eq(f)].copy()
            ref=cross.loc[~cross["fold_id"].eq(f),["ticker","meta_probability"]].copy()
            score,hist,dr=build_ranked_policy_score(ref,va,probability_column="meta_probability",raw_weight=rw,historical_weight=hw,date_weight=dw)
            va["policy_score"]=score; va["meta_historical_rank"]=hist; va["meta_date_rank"]=dr
            scored.append(va)
        candidate=pd.concat(scored,ignore_index=True)
        policy=select_threshold(candidate,folds=folds,score_column="policy_score",minimum_alerts=minimum_alerts,target_precision=target_precision)
        key=(int(policy["safe"]),policy["minimum_precision"],policy["minimum_wilson"],policy["mean_precision"],safe_pr_auc(candidate[TARGET_COLUMN],candidate["policy_score"]))
        if best is None or key>best[0]: best=(key,candidate,policy,mix)
    assert best is not None
    cross,policy,mix=best[1],best[2],best[3]

    final_hardfp,ncand=fit_hardfp_model(cross,hardfp_features,hardfp_quantile,seed+999)
    seed_rank=cross.groupby(["ticker","fold_id"],sort=False)["expert_seed_score"].rank(pct=True,method="average").to_numpy(float)
    y=pd.to_numeric(cross[TARGET_COLUMN],errors="raise").to_numpy(np.int8)
    pos=max(int((y==1).sum()),1); neg=max(int((y==0).sum()),1)
    classw=np.where(y==1,len(y)/(2*pos),len(y)/(2*neg))
    weights=classw*(1.0+2.5*np.power(np.nan_to_num(seed_rank,nan=.5),4))
    final_stack=fit_nonnegative_stacker(cross,columns=meta_cols,target_column=TARGET_COLUMN,sample_weight=weights,l2=.25)
    detail={"policy":policy,"policy_mix":mix,"hardfp_candidate_rows":ncand,"meta_columns":meta_cols,
            "stacker_weights":{c:float(w) for c,w in zip(final_stack.columns,final_stack.weights)},"stacker_intercept":float(final_stack.intercept)}
    return cross,detail,final_hardfp,final_stack,mix


def apply_second_level_future(
    predictions: pd.DataFrame, validation_features: pd.DataFrame, *, hardfp_features: Sequence[str],
    hardfp_model: Any, stacker: Any, policy_mix: tuple[float,float,float], reference_discovery: pd.DataFrame,
) -> pd.DataFrame:
    cur=merge_second_level_features(predictions,validation_features,hardfp_features,[c for c in V11_LAG0_PRIMARY_DIRECTION_COLUMNS if c in validation_features.columns])
    cur["p_hardfp"]=predict_binary_model(hardfp_model,cur)
    cur["meta_probability"]=predict_nonnegative_stacker(stacker,cur)
    rw,hw,dw=policy_mix
    score,hist,dr=build_ranked_policy_score(reference_discovery[["ticker","meta_probability"]],cur,probability_column="meta_probability",raw_weight=rw,historical_weight=hw,date_weight=dw)
    cur["policy_score"]=score; cur["meta_historical_rank"]=hist; cur["meta_date_rank"]=dr
    return cur


def config_summary(config: V14Config, disc: pd.DataFrame, policy: dict[str,Any]) -> dict[str,Any]:
    return {
        "config_id":config.config_id,"config_key":config.key(),"safe_discovery":bool(policy["safe"]),
        "threshold":float(policy["threshold"]),"minimum_discovery_precision":float(policy["minimum_precision"]),
        "mean_discovery_precision":float(policy["mean_precision"]),"minimum_discovery_wilson":float(policy["minimum_wilson"]),
        "discovery_pr_auc":safe_pr_auc(disc[TARGET_COLUMN],disc["policy_score"]),
        "direct_pr_auc":safe_pr_auc(disc[TARGET_COLUMN],disc["p_direct"]),
        "hazard_pr_auc":safe_pr_auc(disc[TARGET_COLUMN],disc["p_hazard_up"]),
        "v13_stage_pr_auc":safe_pr_auc(disc[TARGET_COLUMN],disc["v13_stage_probability"]),
        "hardfp_pr_auc":safe_pr_auc(disc[TARGET_COLUMN],disc["p_hardfp"]),
        "down_pr_auc":safe_pr_auc(disc[DOWN_COLUMN],disc["p_down"]),
    }


def eval_at_threshold(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return policy_metrics(pred,threshold,score_column="policy_score")


def build_posthoc_diagnostics(
    predictions: pd.DataFrame, *, minimum_alerts: int,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Research-only diagnostics that never alter the frozen champion or threshold."""
    score_columns=[c for c in [
        "policy_score","meta_probability","p_direct","p_hazard_up",
        "v13_stage_probability","p_hardfp","one_minus_p_down","v10_base_past_rank",
    ] if c in predictions.columns]
    tail_rows=[]; correlation_rows=[]; slice_rows=[]
    for fold,g in predictions.groupby("fold_id",sort=True):
        y=pd.to_numeric(g[TARGET_COLUMN],errors="coerce").to_numpy(float)
        for column in score_columns:
            score=pd.to_numeric(g[column],errors="coerce").to_numpy(float)
            finite=np.isfinite(y)&np.isfinite(score)&np.isin(y,[0,1])
            yy=y[finite].astype(int); ss=score[finite]
            if not len(yy):
                continue
            order=np.argsort(-ss,kind="mergesort"); ordered_y=yy[order]
            cumulative=np.cumsum(ordered_y==1); ks=np.arange(1,len(ordered_y)+1)
            precision=cumulative/ks
            start_k=min(max(int(minimum_alerts),1),len(ordered_y))
            eligible=precision[start_k-1:]
            oracle_offset=int(np.argmax(eligible)); oracle_k=start_k+oracle_offset
            row={
                "fold_id":int(fold),"expert":column,"rows":int(len(yy)),
                "pr_auc":safe_pr_auc(yy,ss),"roc_auc":safe_roc_auc(yy,ss),
                "top30_precision":float(precision[min(29,len(precision)-1)]),
                "oracle_k_ge_minimum_alerts":int(oracle_k),
                "oracle_precision_ge_minimum_alerts":float(precision[oracle_k-1]),
            }
            for fraction in (.01,.05,.10):
                k=min(max(int(np.ceil(len(yy)*fraction)),1),len(yy))
                row[f"top_{int(fraction*100)}pct_precision"]=float(precision[k-1])
                row[f"top_{int(fraction*100)}pct_rows"]=int(k)
            tail_rows.append(row)

        corr=g[score_columns].apply(pd.to_numeric,errors="coerce").corr(method="spearman")
        for i,left in enumerate(score_columns):
            for right in score_columns[i+1:]:
                correlation_rows.append({
                    "fold_id":int(fold),"left_expert":left,"right_expert":right,
                    "spearman":float(corr.loc[left,right]),
                })

        if "expert_seed_score" in g.columns:
            seed_score=pd.to_numeric(g["expert_seed_score"],errors="coerce")
            for quantile in (.70,.80,.90,.95):
                cutoff=float(seed_score.quantile(quantile)); selected=g.loc[seed_score.ge(cutoff)].copy()
                selected_y=pd.to_numeric(selected[TARGET_COLUMN],errors="coerce")
                positives=int(selected_y.eq(1).sum()); count=int(len(selected))
                slice_rows.append({
                    "fold_id":int(fold),"candidate_quantile":quantile,"score_cutoff":cutoff,
                    "rows":count,"positives":positives,"precision":float(positives/max(count,1)),
                    "mean_p_hardfp_positive":float(pd.to_numeric(selected.loc[selected_y.eq(1),"p_hardfp"],errors="coerce").mean()),
                    "mean_p_hardfp_negative":float(pd.to_numeric(selected.loc[selected_y.eq(0),"p_hardfp"],errors="coerce").mean()),
                    "mean_p_down_positive":float(pd.to_numeric(selected.loc[selected_y.eq(1),"p_down"],errors="coerce").mean()),
                    "mean_p_down_negative":float(pd.to_numeric(selected.loc[selected_y.eq(0),"p_down"],errors="coerce").mean()),
                })
    return pd.DataFrame(tail_rows),pd.DataFrame(correlation_rows),pd.DataFrame(slice_rows)


def main() -> None:
    parser=argparse.ArgumentParser(description="CrashWatch Surge V14: direct surge + crossing hazard + downside veto + hard-FP crossfit stack")
    # Same runtime contract as V13.
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--v10-output", required=True)
    parser.add_argument("--v10-2-output", required=True)
    parser.add_argument("--v11-output", default=None)
    parser.add_argument("--v11-1-output", default=None)
    parser.add_argument("--v13-output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-sidecar", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--feature-profile-manifest", required=True)
    parser.add_argument("--feature-profile", default="P0_ALL_VALID")
    parser.add_argument("--output", default="outputs/surge_competingrisk_hardfp_v14")
    parser.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-target-valid-rows",type=int,default=91775)
    parser.add_argument("--expected-ticker-count",type=int,default=48)
    parser.add_argument("--require-full-base-oof",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--require-gpu",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--cpu-threads",type=int,default=24)
    parser.add_argument("--fold-workers",type=int,default=1)
    parser.add_argument("--target-hours",type=float,default=7.0)
    parser.add_argument("--fast-mode",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--discovery-folds",default="0,1,2")
    parser.add_argument("--development-folds",default="3,4")
    parser.add_argument("--confirmation-folds",default="5,6")
    parser.add_argument("--recent-folds",default="7")
    parser.add_argument("--minimum-alerts",type=int,default=30)
    parser.add_argument("--target-precision",type=float,default=.70)
    parser.add_argument("--surge-threshold",type=float,default=.05)
    parser.add_argument("--max-target-mismatch-rate",type=float,default=.01)
    parser.add_argument("--allow-target-mismatch",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--robust-seeds",type=int,default=5)
    parser.add_argument("--max-stability-seeds",type=int,default=61)
    parser.add_argument("--seed",type=int,default=14014)
    args=parser.parse_args()
    if bool(args.fast_mode):
        args.robust_seeds=min(int(args.robust_seeds),2)
        args.max_stability_seeds=min(int(args.max_stability_seeds),5)

    output=Path(args.output).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"RUN_STATUS.json",{"status":"RUNNING","started_utc":utc_now(),"schema":SCHEMA})
    start=time.monotonic(); budget_seconds=max(float(args.target_hours)*3600,180); deadline=start+budget_seconds
    finalize_reserve=min(900.0,max(120.0,budget_seconds*0.03))
    stability_deadline=max(start+budget_seconds*0.50,deadline-finalize_reserve)
    try:
        preflight=gpu_preflight(int(args.seed)); atomic_json(output/"GPU_PREFLIGHT_V14.json",preflight)
        if args.require_gpu and not preflight.get("available",False): raise RuntimeError(f"GPU preflight failed: {preflight}")

        scratch=output/"_runtime_v13_compat"; scratch.mkdir(exist_ok=True)
        runtime,target_audit=prepare_runtime(args,scratch)
        runtime.frame=add_competing_risk_labels(runtime.frame,threshold=float(args.surge_threshold))
        cache_dir=output/"_cache_v14_experts"
        v13_oof=load_v13_oof(Path(args.v13_output).expanduser().resolve())
        v13_hash_cols=["source_row_id","fold_id","v13_stage_probability","v13_p_move","v13_p_up_given_move"]
        v13_fingerprint=hashlib.sha1(pd.util.hash_pandas_object(v13_oof[v13_hash_cols],index=False).to_numpy(np.uint64).tobytes()).hexdigest()[:16]
        all_folds=parse_ints(args.discovery_folds)+parse_ints(args.development_folds)+parse_ints(args.confirmation_folds)+parse_ints(args.recent_folds)
        valframe=build_validation_frame(runtime,all_folds,v13_oof)
        discovery_folds=parse_ints(args.discovery_folds); development_folds=parse_ints(args.development_folds)
        confirmation_folds=parse_ints(args.confirmation_folds); recent_folds=parse_ints(args.recent_folds)
        discovery=valframe.loc[valframe["fold_id"].isin(discovery_folds)].copy()

        def fit_fold_group(
            config: V14Config, folds_: Sequence[int], seed_: int, *,
            direct_features: Sequence[str], hazard_features: Sequence[str], down_features: Sequence[str],
        ) -> list[tuple[pd.DataFrame,dict[str,Any]]]:
            """Fit independent folds concurrently while keeping total CPU threads bounded."""
            ordered_folds=[int(f) for f in folds_]
            workers=max(1,min(int(args.fold_workers),len(ordered_folds)))
            threads_per_worker=max(1,int(args.cpu_threads)//workers)
            def run_one(fold_: int) -> tuple[pd.DataFrame,dict[str,Any]]:
                return fit_predict_experts_fold_cached(
                    runtime,config,fold_,seed_,direct_features=direct_features,
                    hazard_features=hazard_features,down_features=down_features,
                    v13_oof=v13_oof,cpu_threads=threads_per_worker,use_gpu=True,
                    fast=bool(args.fast_mode),cache_dir=cache_dir,resume=bool(args.resume),
                    v13_fingerprint=v13_fingerprint,
                )
            if workers==1:
                return [run_one(fold_) for fold_ in ordered_folds]
            completed={}
            with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="v14-fold") as executor:
                futures={executor.submit(run_one,fold_):fold_ for fold_ in ordered_folds}
                for future in as_completed(futures):
                    completed[futures[future]]=future.result()
            return [completed[fold_] for fold_ in ordered_folds]

        # V13 bottleneck fix: feature selection must generalize across discovery folds and have broad support.
        log("Cross-fit ranking direct-surge features")
        direct_rank=crossfit_rank_features(
            discovery,runtime.candidate_features,target_column=TARGET_COLUMN,folds=discovery_folds,
            allowed_families=None,minimum_total_rows=800,minimum_holdout_rows=120,minimum_coverage=.55,minimum_fold_coverage=.45,minimum_worst_auc=.485,
        )
        direct_pool,direct_dedup=deduplicate_ranked_features(discovery,direct_rank,max_features=220,corr_threshold=.965,max_per_stem=2)
        log("Cross-fit ranking direction features on large-move rows")
        large=pd.to_numeric(discovery[MOVE_COLUMN],errors="coerce").eq(1).to_numpy()
        direction_rank=crossfit_rank_features(
            discovery,runtime.candidate_features,target_column=TARGET_COLUMN,folds=discovery_folds,mask=large,
            allowed_families={"direction"},minimum_total_rows=500,minimum_holdout_rows=75,minimum_coverage=.50,minimum_fold_coverage=.40,minimum_worst_auc=.48,
        )
        direction_pool,direction_dedup=deduplicate_ranked_features(discovery,direction_rank,max_features=120,corr_threshold=.96,max_per_stem=2)
        log("Cross-fit ranking downside features")
        down_rank=crossfit_rank_features(
            discovery,runtime.candidate_features,target_column=DOWN_COLUMN,folds=discovery_folds,
            allowed_families={"magnitude","direction","mixed","unknown"},minimum_total_rows=800,minimum_holdout_rows=120,
            minimum_coverage=.55,minimum_fold_coverage=.45,minimum_worst_auc=.485,
        )
        down_pool,down_dedup=deduplicate_ranked_features(discovery,down_rank,max_features=160,corr_threshold=.965,max_per_stem=2)
        if len(direct_pool)<32 or len(direction_pool)<12 or len(down_pool)<24:
            raise RuntimeError(f"Too few stable features after V14 gates: direct={len(direct_pool)}, direction={len(direction_pool)}, down={len(down_pool)}")
        atomic_csv(output/"V14_DIRECT_FEATURE_RANKING_CROSSFIT.csv",direct_rank)
        atomic_csv(output/"V14_DIRECTION_FEATURE_RANKING_CROSSFIT.csv",direction_rank)
        atomic_csv(output/"V14_DOWN_FEATURE_RANKING_CROSSFIT.csv",down_rank)
        atomic_csv(output/"V14_DIRECT_DEDUP_DECISIONS.csv",direct_dedup)
        atomic_csv(output/"V14_DIRECTION_DEDUP_DECISIONS.csv",direction_dedup)
        atomic_csv(output/"V14_DOWN_DEDUP_DECISIONS.csv",down_dedup)

        audit={
            "schema":SCHEMA,"rows_full_target_valid":len(runtime.frame),"tickers":int(runtime.frame.ticker.nunique()),"raw_features":len(runtime.raw_features),
            "candidate_features":len(runtime.candidate_features),"stable_direct_pool":len(direct_pool),"stable_direction_pool":len(direction_pool),"stable_down_pool":len(down_pool),
            "v13_oof_rows":len(v13_oof),"v13_v11_self_primary":False,"v13_v11_directed_primary":False,"v11_lag0_hardfp_only":True,
            "v13_runtime_hours_observed":1.100752271833335,"v14_time_budget_hours":float(args.target_hours),
            "target_audit":target_audit,
        }
        atomic_json(output/"DATA_AUDIT_V14.json",audit)

        configs=predefined_configs(bool(args.fast_mode))
        lag0_cols=[c for c in V11_LAG0_PRIMARY_DIRECTION_COLUMNS if c in valframe.columns]
        policy_mixes=[(1.0,0.0,0.0),(0.65,0.25,0.10),(0.45,0.40,0.15)]
        screening=[]; screen_cache={}
        for ci,config in enumerate(configs):
            log(f"Screen {config.config_id}")
            dfeat=direct_pool[:min(config.direct_k,len(direct_pool))]
            hfeat=direct_pool[:min(config.hazard_k,len(direct_pool))]
            downfeat=down_pool[:min(config.down_k,len(down_pool))]
            hardraw=direction_pool[:min(config.hardfp_k,len(direction_pool))]
            hfpcols=[*hardraw]
            if config.include_v11_lag0_hardfp: hfpcols.extend(lag0_cols)
            hfpcols.extend(["p_direct","p_hazard_up","v13_stage_probability","v13_p_move","v13_p_up_given_move","one_minus_p_down","v10_base_past_rank"])
            preds=[]; diags=[]
            seed=int(args.seed)+ci*1000
            for p,d in fit_fold_group(
                config,discovery_folds,seed,direct_features=dfeat,hazard_features=hfeat,down_features=downfeat,
            ):
                preds.append(p);diags.append(d)
            ens=concatenate_expert_folds(preds)
            discfeatures=valframe.loc[valframe.fold_id.isin(discovery_folds)].copy()
            merged=merge_second_level_features(ens,discfeatures,hardraw,lag0_cols)
            disc2,detail,hmodel,stack,mix=crossfit_second_level(
                merged,hardfp_features=hfpcols,hardfp_quantile=config.hardfp_quantile,seed=seed+500,
                policy_mixes=policy_mixes,minimum_alerts=args.minimum_alerts,target_precision=args.target_precision,
            )
            row=config_summary(config,disc2,detail["policy"]); row["policy_mix"]=str(mix)
            screening.append(row); screen_cache[config.config_id]=(disc2,detail)
        screen_df=pd.DataFrame(screening).sort_values(["safe_discovery","minimum_discovery_precision","discovery_pr_auc"],ascending=[False,False,False],kind="mergesort")
        atomic_csv(output/"V14_CONFIG_SCREENING.csv",screen_df)
        shortlist=screen_df.head(min(3,len(screen_df)))["config_id"].astype(str).tolist()
        log(f"Robust shortlist: {shortlist}")

        robust_seed_values=[int(args.seed)+i*101 for i in range(max(1,int(args.robust_seeds)))]
        development_rows=[]; robust_cache={}
        for config in [c for c in configs if c.config_id in shortlist]:
            dfeat=direct_pool[:min(config.direct_k,len(direct_pool))]; hfeat=direct_pool[:min(config.hazard_k,len(direct_pool))]
            downfeat=down_pool[:min(config.down_k,len(down_pool))]; hardraw=direction_pool[:min(config.hardfp_k,len(direction_pool))]
            hfpcols=[*hardraw];
            if config.include_v11_lag0_hardfp: hfpcols.extend(lag0_cols)
            hfpcols.extend(["p_direct","p_hazard_up","v13_stage_probability","v13_p_move","v13_p_up_given_move","one_minus_p_down","v10_base_past_rank"])
            seed_preds=[]; diagnostics=[]
            for seed in robust_seed_values:
                parts=[]
                for p,d in fit_fold_group(
                    config,discovery_folds+development_folds,seed,
                    direct_features=dfeat,hazard_features=hfeat,down_features=downfeat,
                ):
                    parts.append(p); diagnostics.append(d)
                seed_preds.append(pd.concat(parts,ignore_index=True))
            ens=ensemble_experts(seed_preds)
            discover_pred=ens.loc[ens.fold_id.isin(discovery_folds)].copy(); dev_pred=ens.loc[ens.fold_id.isin(development_folds)].copy()
            discover_feat=valframe.loc[valframe.fold_id.isin(discovery_folds)].copy(); dev_feat=valframe.loc[valframe.fold_id.isin(development_folds)].copy()
            merged_discovery=merge_second_level_features(discover_pred,discover_feat,hardraw,lag0_cols)
            disc2,detail,hmodel,stack,mix=crossfit_second_level(
                merged_discovery,hardfp_features=hfpcols,hardfp_quantile=config.hardfp_quantile,seed=int(args.seed)+7000,
                policy_mixes=policy_mixes,minimum_alerts=args.minimum_alerts,target_precision=args.target_precision,
            )
            dev2=apply_second_level_future(dev_pred,dev_feat,hardfp_features=hfpcols,hardfp_model=hmodel,stacker=stack,policy_mix=mix,reference_discovery=disc2)
            pm=policy_metrics(dev2,float(detail["policy"]["threshold"])); minp=float(pm.precision.min()); meanp=float(pm.precision.mean())
            row={
                "config_id":config.config_id,"threshold":detail["policy"]["threshold"],"discovery_safe":detail["policy"]["safe"],
                "minimum_development_precision":minp,"mean_development_precision":meanp,"minimum_development_alerts":int(pm.alerts.min()),
                "mean_development_pr_auc":float(pm.pr_auc.mean()),"development_safe":bool((pm.alerts>=args.minimum_alerts).all() and (pm.precision>=args.target_precision).all()),
                "policy_mix":str(mix),"stacker_weights":json.dumps(detail["stacker_weights"],sort_keys=True),
            }
            development_rows.append(row); robust_cache[config.config_id]=(config,disc2,detail,hmodel,stack,mix,hfpcols,hardraw,dfeat,hfeat,downfeat,ens,pm)
        devdf=pd.DataFrame(development_rows).sort_values(["development_safe","minimum_development_precision","mean_development_pr_auc"],ascending=[False,False,False],kind="mergesort")
        atomic_csv(output/"V14_DEVELOPMENT_CONFIG_VALIDATION.csv",devdf)
        champion_id=str(devdf.iloc[0]["config_id"]); champion_data=robust_cache[champion_id]
        config,disc2,detail,hmodel,stack,mix,hfpcols,hardraw,dfeat,hfeat,downfeat,_,_=champion_data
        threshold=float(detail["policy"]["threshold"])
        atomic_json(output/"V14_CHAMPION_CONFIG.json",{
            "schema":SCHEMA,"champion":dataclasses.asdict(config),"threshold":threshold,"policy_mix":mix,
            "primary_seeds":robust_seed_values,"direct_features":dfeat,"hazard_features":hfeat,"down_features":downfeat,
            "hardfp_raw_features":hardraw,"hardfp_all_features":hfpcols,"stacker_weights":detail["stacker_weights"],
            "v11_self_primary":False,"v11_lag0_role":"hardfp_only","v11_directed_primary":False,
        })

        # Final primary prediction: exact same champion and robust seeds, now folds 0-7.
        seed_preds=[]; final_diags=[]
        for seed in robust_seed_values:
            parts=[]
            for p,d in fit_fold_group(
                config,all_folds,seed,direct_features=dfeat,hazard_features=hfeat,down_features=downfeat,
            ):
                parts.append(p); final_diags.append(d)
            seed_preds.append(pd.concat(parts,ignore_index=True))
        primary_ens=ensemble_experts(seed_preds)
        final_parts=[]
        for f in all_folds:
            part=primary_ens.loc[primary_ens.fold_id.eq(f)].copy(); feat=valframe.loc[valframe.fold_id.eq(f)].copy()
            if f in discovery_folds:
                q=disc2.loc[disc2.fold_id.eq(f)].copy()
            else:
                # The discovery calibration reference is frozen exactly once.  Do not let
                # fold-3 score distributions change fold-4 policy after champion selection.
                q=apply_second_level_future(part,feat,hardfp_features=hfpcols,hardfp_model=hmodel,stacker=stack,policy_mix=mix,reference_discovery=disc2)
            final_parts.append(q)
        final=pd.concat(final_parts,ignore_index=True).sort_values(["fold_id","date","ticker"],kind="mergesort").reset_index(drop=True)
        atomic_csv(output/"v14_oof_predictions.csv.gz",final,compression="gzip")
        metrics=policy_metrics(final,threshold); oracle=oracle_topk(final,minimum_alerts=args.minimum_alerts)
        atomic_csv(output/"v14_frozen_policy_by_fold.csv",metrics); atomic_csv(output/"v14_oracle_topk_by_fold.csv",oracle)
        atomic_csv(output/"V14_FINAL_FOLD_DIAGNOSTICS.csv",pd.DataFrame(final_diags))

        expert_rows=[]
        for f,g in final.groupby("fold_id",sort=True):
            y=pd.to_numeric(g[TARGET_COLUMN],errors="coerce")
            expert_rows.append({
                "fold_id":int(f),"rows":len(g),"base_rate":float(y.mean()),
                "v14_pr_auc":safe_pr_auc(y,g.policy_score),"v14_roc_auc":safe_roc_auc(y,g.policy_score),
                "direct_pr_auc":safe_pr_auc(y,g.p_direct),"hazard_pr_auc":safe_pr_auc(y,g.p_hazard_up),
                "v13_stage_pr_auc":safe_pr_auc(y,g.v13_stage_probability),"hardfp_pr_auc":safe_pr_auc(y,g.p_hardfp),
                "down_pr_auc":safe_pr_auc(g[DOWN_COLUMN],g.p_down),
            })
        atomic_csv(output/"v14_expert_metrics_by_fold.csv",pd.DataFrame(expert_rows))

        # These diagnostics are deliberately post-selection and cannot change the frozen model.
        tail_diag,pair_diag,hardfp_diag=build_posthoc_diagnostics(final,minimum_alerts=int(args.minimum_alerts))
        atomic_csv(output/"v14_expert_tail_diagnostics.csv",tail_diag)
        atomic_csv(output/"v14_expert_pairwise_spearman.csv",pair_diag)
        atomic_csv(output/"v14_hardfp_slice_diagnostics.csv",hardfp_diag)

        # Use remaining budget for genuine ensemble-size stability, never for champion/threshold selection.
        stability=[]; seed_count=len(robust_seed_values); maxseeds=max(seed_count,int(args.max_stability_seeds))
        diagnostic_seed_preds=list(seed_preds)
        stability_seed_seconds=[]; evaluated_seed_counts=set(); stability_stop_reason="MAX_STABILITY_SEEDS"

        def record_stability(count: int) -> None:
            ens=ensemble_experts(diagnostic_seed_preds)
            diagparts=[]
            for f in all_folds:
                feat=valframe.loc[valframe.fold_id.eq(f)].copy(); part=ens.loc[ens.fold_id.eq(f)].copy()
                q=apply_second_level_future(part,feat,hardfp_features=hfpcols,hardfp_model=hmodel,stacker=stack,policy_mix=mix,reference_discovery=disc2)
                diagparts.append(q)
            dg=pd.concat(diagparts,ignore_index=True); pm=policy_metrics(dg,threshold)
            for _,rr in pm.iterrows(): stability.append({"ensemble_seed_count":count,**rr.to_dict()})
            evaluated_seed_counts.add(int(count))

        record_stability(seed_count)
        stability_checkpoints={7,9,11,15,19,25,31,41,51,61,81,101,121,151,181,maxseeds}
        while seed_count<maxseeds:
            now=time.monotonic()
            if not bool(args.fast_mode) and now>=stability_deadline:
                stability_stop_reason="STABILITY_DEADLINE"; break
            if stability_seed_seconds and not bool(args.fast_mode):
                estimate=max(float(np.median(stability_seed_seconds[-3:]))*1.20,30.0)
                if stability_deadline-now<estimate:
                    stability_stop_reason="INSUFFICIENT_TIME_FOR_NEXT_SEED"; break
            seed_started=time.monotonic()
            seed=int(args.seed)+seed_count*101; parts=[]
            for p,_ in fit_fold_group(
                config,all_folds,seed,direct_features=dfeat,hazard_features=hfeat,down_features=downfeat,
            ):
                parts.append(p)
            diagnostic_seed_preds.append(pd.concat(parts,ignore_index=True)); seed_count+=1
            elapsed_seed=float(time.monotonic()-seed_started); stability_seed_seconds.append(elapsed_seed)
            log(f"Stability seed {seed_count}/{maxseeds} complete in {elapsed_seed:.1f}s")
            if seed_count in stability_checkpoints:
                record_stability(seed_count)
        if seed_count not in evaluated_seed_counts:
            record_stability(seed_count)
        atomic_csv(output/"v14_ensemble_size_stability.csv",pd.DataFrame(stability))
        atomic_json(output/"EXPERIMENT_BUDGET_V14.json",{
            "schema":SCHEMA,"target_hours":float(args.target_hours),"budget_seconds":budget_seconds,
            "finalize_reserve_seconds":finalize_reserve,"diagnostic_seeds_completed":seed_count,
            "max_stability_seeds":maxseeds,"stability_stop_reason":stability_stop_reason,
            "mean_new_stability_seed_seconds":float(np.mean(stability_seed_seconds)) if stability_seed_seconds else None,
            "elapsed_hours_at_stability_end":float((time.monotonic()-start)/3600),
            "diagnostic_only":True,"champion_or_threshold_changed":False,
        })

        roles={"discovery":discovery_folds,"development":development_folds,"confirmation":confirmation_folds,"recent":recent_folds}
        safety={}
        for role,folds_ in roles.items():
            m=metrics.loc[metrics.fold_id.isin(folds_)]
            safety[role]=bool(len(m)==len(folds_) and (m.alerts>=args.minimum_alerts).all() and (m.precision>=args.target_precision).all())
        recommendation={
            "schema":SCHEMA,"status":"SUCCESS_PENDING_VERIFIER","champion_config_id":champion_id,"frozen_threshold":threshold,
            "policy_target":{"minimum_alerts":args.minimum_alerts,"precision":args.target_precision},"role_safety":safety,
            "production_decision":"NO_ALERT","runtime_hours":float((time.monotonic()-start)/3600),
            "diagnostic_seeds_completed":seed_count,"stability_stop_reason":stability_stop_reason,
            "primary_bottleneck_hypothesis":"direction/top-tail discrimination and sparse feature-selection instability, not threshold tuning",
            "v14_changes":[
                "cross-fold held-out feature ranking with coverage gates","correlation/stem deduplication","direct surge expert",
                "D1-D3 crossing-hazard expert","downside competing-risk veto","cross-fitted hard-FP discriminator",
                "nonnegative discovery-crossfit stacker; V13 product retained as one expert only","V11 self/direct edges excluded from primary; lag0 hard-FP only",
            ],
            "additional_posthoc_diagnostics":[
                "expert upper-tail and oracle-at-minimum-alerts comparison",
                "foldwise expert Spearman correlation",
                "hard-FP/downside separation inside high-score candidate slices",
                "budget-bounded extended ensemble-size stability",
            ],
            "generated_utc":utc_now(),
        }
        atomic_json(output/"FINAL_RECOMMENDATION_V14.json",recommendation)
        atomic_json(output/"RUN_STATUS.json",{"status":"SUCCESS","completed_utc":utc_now(),"runtime_hours":recommendation["runtime_hours"]})

        verifier=HERE/"verify_surge_competingrisk_hardfp_v14.py"
        res=subprocess.run([sys.executable,str(verifier),"--output",str(output)],capture_output=True,text=True)
        (output/"VERIFIER_STDOUT_V14.txt").write_text(res.stdout+"\n"+res.stderr,encoding="utf-8")
        if res.returncode!=0: raise RuntimeError(f"V14 verifier failed; see {output/'VERIFIER_STDOUT_V14.txt'}")
        recommendation["status"]="SUCCESS_VERIFIED"; atomic_json(output/"FINAL_RECOMMENDATION_V14.json",recommendation)
        log(f"Complete: champion={champion_id}, runtime={recommendation['runtime_hours']:.2f}h")
    except Exception as e:
        atomic_json(output/"RUN_STATUS.json",{"status":"FAILED","error":repr(e),"traceback":traceback.format_exc(),"failed_utc":utc_now()})
        raise


if __name__ == "__main__":
    main()
