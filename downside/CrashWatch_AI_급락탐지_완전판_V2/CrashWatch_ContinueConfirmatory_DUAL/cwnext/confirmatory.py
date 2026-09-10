from __future__ import annotations

import concurrent.futures as cf
import gc
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.aggregate import METRICS
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, bh_adjust, bootstrap_mean_ci, canonical_hash, ensure_thread_env, exact_sign_flip_p, hash_strings, read_json

from .common import write_eta

LOGGER = logging.getLogger(__name__)
PROFILE_VERSION = "cw_confirmatory_profiles_v2"
TARGET_VERSION = "cw_confirmatory_targeted_v2"
SEEDS_DEFAULT = [17, 43, 101, 211, 503]


def _lgb_params(config: dict[str, Any], y_train: np.ndarray, threads: int, seed: int) -> dict[str, Any]:
    pos = max(1, int(np.sum(y_train == 1))); neg = max(1, int(np.sum(y_train == 0)))
    return {
        "objective": "binary", "metric": "None",
        "learning_rate": float(config.get("learning_rate", 0.03)),
        "num_leaves": int(config.get("num_leaves", 127)),
        "max_depth": int(config.get("max_depth", -1)),
        "min_data_in_leaf": int(config.get("min_data_in_leaf", 55)),
        "feature_fraction": float(config.get("feature_fraction", 0.85)),
        "feature_fraction_bynode": 1.0,
        "bagging_fraction": float(config.get("bagging_fraction", 0.90)),
        "bagging_freq": int(config.get("bagging_freq", 1)),
        "lambda_l1": float(config.get("lambda_l1", 0.25)),
        "lambda_l2": float(config.get("lambda_l2", 1.8)),
        "max_bin": int(config.get("max_bin", 255)),
        "min_gain_to_split": float(config.get("min_gain_to_split", 1e-12)),
        "scale_pos_weight": float(neg / pos),
        "num_threads": int(threads), "deterministic": True, "force_col_wise": True,
        "feature_pre_filter": False, "verbosity": -1,
        "seed": int(seed), "feature_fraction_seed": int(seed), "bagging_seed": int(seed),
        "drop_seed": int(seed),
    }


def build_profiles(previous_output: Path, seed_dir: Path, feature_names: list[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    feature_set = set(feature_names)
    duplicate_path = previous_output / "exact_duplicate_drop_candidates.tsv"
    if not duplicate_path.exists(): duplicate_path = seed_dir / "exact_duplicate_drop_candidates.tsv"
    dup = pd.read_csv(duplicate_path, sep="\t")
    dup_drop = [x for x in dup["drop_feature"].astype(str) if x in feature_set]

    corr = pd.read_csv(previous_output / "correlation" / "feature_correlation_summary.csv")
    loo = pd.read_csv(previous_output / "feature_ablation_summary.csv")
    cond = pd.read_csv(previous_output / "conditional_ablation_summary.csv")
    cluster = pd.read_csv(previous_output / "cluster_ablation_summary.csv") if (previous_output / "cluster_ablation_summary.csv").exists() else pd.DataFrame()
    gpu_loo = loo[loo["backend"].eq("xgboost_cuda")].copy() if "backend" in loo else pd.DataFrame()
    cpu_loo = loo[loo["backend"].eq("lightgbm_cpu")].copy() if "backend" in loo else loo.copy()
    cpu_cond = cond[cond["backend"].eq("lightgbm_cpu")].copy() if "backend" in cond else cond.copy()
    master = corr.copy()
    if not cpu_loo.empty:
        x = cpu_loo[["feature", "mean_pr_auc_utility", "positive_fold_ratio", "worst_fold_utility", "recent_mean_pr_auc_utility"]].copy()
        x.columns = ["feature", "loo_mean", "loo_pos", "loo_worst", "loo_recent"]
        master = master.merge(x, on="feature", how="left")
    if not cpu_cond.empty:
        x = cpu_cond.sort_values("mean_pr_auc_utility", ascending=False).drop_duplicates("feature")[["feature", "mean_pr_auc_utility", "positive_fold_ratio", "worst_fold_utility", "recent_mean_pr_auc_utility"]]
        x.columns = ["feature", "cond_mean", "cond_pos", "cond_worst", "cond_recent"]
        master = master.merge(x, on="feature", how="left")
    if not gpu_loo.empty:
        x = gpu_loo[["feature", "mean_pr_auc_utility"]].copy(); x.columns = ["feature", "gpu_mean"]
        master = master.merge(x, on="feature", how="left")
    if not cluster.empty:
        c = cluster[cluster.get("backend", pd.Series(dtype=str)).eq("lightgbm_cpu")].copy()
        if not c.empty:
            c = c[["cluster_id", "mean_pr_auc_utility"]].rename(columns={"mean_pr_auc_utility": "cluster_mean"})
            master = master.merge(c, left_on="primary_cluster_id", right_on="cluster_id", how="left")

    high_missing = master.loc[
        master["missing_ratio"].ge(0.80)
        & master.get("loo_mean", pd.Series(np.nan, index=master.index)).abs().fillna(0).le(0.001)
        & master.get("cond_mean", pd.Series(np.nan, index=master.index)).abs().fillna(0).le(0.001),
        "feature",
    ].astype(str).tolist()
    harmful = master.loc[
        master.get("loo_mean", pd.Series(np.nan, index=master.index)).le(-0.005)
        & master.get("loo_recent", pd.Series(np.nan, index=master.index)).le(0.0)
        & master.get("cond_mean", pd.Series(np.nan, index=master.index)).fillna(0).le(0.003)
        & master.get("gpu_mean", pd.Series(np.nan, index=master.index)).fillna(0).le(0.001)
        & master.get("cluster_mean", pd.Series(np.nan, index=master.index)).fillna(0).le(0.003),
        "feature",
    ].astype(str).tolist()
    # Preserve the three pre-registered strongest harmful candidates even when the optional GPU fold set was incomplete.
    for name in ["t_finshort_balance_slope_20", "t_finshort_volume_sum_5", "t_taildep_cocrash_freq_120"]:
        if name in feature_set and name not in harmful: harmful.append(name)

    assignments = pd.read_csv(previous_output / "correlation" / "all_cluster_assignments.csv")
    def reps(threshold: float) -> list[str]:
        part = assignments[np.isclose(assignments["threshold"].astype(float), threshold)]
        return [x for x in part.loc[part["is_representative"].astype(bool), "feature"].astype(str) if x in feature_set]

    p0 = list(feature_names)
    p1 = [f for f in feature_names if f not in set(dup_drop)]
    prereg_harmful = [f for f in [
        "t_finshort_balance_slope_20", "t_finshort_volume_sum_5", "t_taildep_cocrash_freq_120"
    ] if f in feature_set]
    p2_drop = set(dup_drop) | set(high_missing) | set(prereg_harmful)
    p2 = [f for f in feature_names if f not in p2_drop]
    p3, p4, p5 = reps(0.98), reps(0.95), reps(0.92)

    conditional_candidates = master.loc[
        master.get("cond_mean", pd.Series(np.nan, index=master.index)).gt(0)
        & master.get("cond_recent", pd.Series(np.nan, index=master.index)).fillna(0).gt(0),
        ["feature", "cond_mean", "cond_pos", "cond_worst"],
    ].sort_values(["cond_mean", "cond_pos"], ascending=False)["feature"].astype(str).tolist()
    conditional_nonrep = [f for f in conditional_candidates if f not in set(p4)][:20]
    p7 = list(dict.fromkeys(p4 + conditional_nonrep))

    # Auto profile: dedup + all dynamically consistent harmful/high-missing removals.
    p6_drop = set(dup_drop) | set(high_missing) | set(harmful)
    p6 = [f for f in feature_names if f not in p6_drop]
    profiles = {
        "P0_FULL_439": p0,
        "P1_EXACT_DEDUP": p1,
        "P2_DEDUP_CLEAN": p2,
        "P3_CORR098": p3,
        "P4_CORR095": p4,
        "P5_CORR092": p5,
        "P6_AUTO_CONSERVATIVE": p6,
        "P7_CORR095_PLUS_CONDITIONAL": p7,
    }
    manifest = {
        "profiles": {k: {"count": len(v), "feature_hash": hash_strings(v), "features": v} for k, v in profiles.items()},
        "exact_duplicate_drop_count": len(dup_drop), "exact_duplicate_drop": dup_drop,
        "high_missing_drop": high_missing, "preregistered_harmful_drop": prereg_harmful,
        "stable_harmful_drop": harmful,
        "conditional_nonrep_added": conditional_nonrep,
    }
    return profiles, manifest


def _profile_worker(plan: dict[str, Any]) -> dict[str, Any]:
    ensure_thread_env(int(plan["threads"]))
    import lightgbm as lgb
    X = np.load(Path(plan["cache_root"]) / "X_all_valid.npy", mmap_mode="r")
    y = np.load(Path(plan["cache_root"]) / "target.npy", mmap_mode="r")
    dates = np.load(Path(plan["cache_root"]) / "dates_ns.npy", mmap_mode="r")
    fold = plan["fold"]
    tr = slice(int(fold["train_start"]), int(fold["train_stop"])); va = slice(int(fold["validation_start"]), int(fold["validation_stop"]))
    idx = np.asarray(plan["indices"], dtype=np.int32)
    x_train = np.ascontiguousarray(X[tr][:, idx], dtype=np.float32)
    y_train = np.asarray(y[tr], dtype=np.uint8)
    x_valid = np.ascontiguousarray(X[va][:, idx], dtype=np.float32)
    y_valid = np.asarray(y[va], dtype=np.uint8); valid_dates = np.asarray(dates[va], dtype=np.int64)
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False, params={"max_bin": int(plan["model_config"].get("max_bin", 255)), "feature_pre_filter": False})
    train_set.construct()
    out_dir = Path(plan["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    completed = cached = failed = 0; times: list[float] = []
    for seed in plan["seeds"]:
        identity = {
            "version": PROFILE_VERSION, "dataset_signature": plan["dataset_signature"],
            "confirm_signature": plan["confirm_signature"], "fold": fold,
            "profile": plan["profile"], "profile_hash": plan["profile_hash"],
            "seed": int(seed), "best_iteration": int(plan["best_iteration"]),
            "model_config": plan["model_config"],
        }
        task_id = canonical_hash(identity)
        path = out_dir / f"{task_id}.json"
        old = read_json(path, {})
        if old.get("status") == "completed" and old.get("identity") == identity:
            cached += 1; continue
        started = time.perf_counter()
        try:
            model = lgb.train(_lgb_params(plan["model_config"], y_train, int(plan["threads"]), int(seed)), train_set, num_boost_round=int(plan["best_iteration"]), callbacks=[lgb.log_evaluation(0)])
            pred = np.asarray(model.predict(x_valid), dtype=np.float32)
            elapsed = time.perf_counter() - started
            result = {
                "status": "completed", "identity": identity, "task_id": task_id,
                "backend": "lightgbm_cpu", "test_type": "profile_benchmark",
                "profile": plan["profile"], "profile_hash": plan["profile_hash"],
                "feature_count": int(len(idx)), "seed": int(seed), "outer_fold": int(fold["fold_id"]),
                "best_iteration": int(plan["best_iteration"]), "elapsed_seconds": elapsed,
                **compute_metrics(y_valid, pred, valid_dates),
            }
            atomic_json(result, path); completed += 1; times.append(elapsed)
            del model, pred
        except Exception as exc:
            failed += 1
            atomic_json({"status": "failed", "identity": identity, "task_id": task_id, "profile": plan["profile"], "seed": seed, "outer_fold": fold["fold_id"], "error": repr(exc), "traceback": traceback.format_exc()}, path)
    del train_set, x_train, y_train, x_valid, y_valid, valid_dates, X, y, dates
    gc.collect()
    return {"completed": completed, "cached": cached, "failed": failed, "mean_seconds": float(np.mean(times)) if times else None}


def _run_plans(plans: list[dict[str, Any]], worker_fn: Any, workers: int, output_dir: Path, stage: str) -> dict[str, Any]:
    started = time.time(); results: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as executor:
        futures = [executor.submit(worker_fn, p) for p in plans]
        for i, future in enumerate(cf.as_completed(futures), start=1):
            results.append(future.result())
            write_eta(output_dir, stage, i, len(plans), time.time() - started)
            LOGGER.info("%s blocks=%s/%s trained=%s cached=%s failed=%s", stage, i, len(plans), sum(r.get("completed",0) for r in results), sum(r.get("cached",0) for r in results), sum(r.get("failed",0) for r in results))
    summary = {"stage": stage, "blocks": len(plans), "completed": sum(r.get("completed",0) for r in results), "cached": sum(r.get("cached",0) for r in results), "failed": sum(r.get("failed",0) for r in results), "elapsed_seconds": time.time()-started}
    atomic_json(summary, output_dir / f"{stage}_run_summary.json")
    return summary


def _read_json_dir(path: Path) -> pd.DataFrame:
    rows=[]
    for p in path.glob("*.json"):
        d=read_json(p,{})
        if d.get("status")=="completed": rows.append(d)
    return pd.DataFrame(rows)


def _effect_summary(fold_effects: pd.DataFrame, key: str, value_col: str = "fold_mean_delta") -> pd.DataFrame:
    rows=[]
    for item, g in fold_effects.groupby(key, sort=False):
        vals=g[value_col].to_numpy(float); vals=vals[np.isfinite(vals)]
        low,high=bootstrap_mean_ci(vals) if len(vals) else (np.nan,np.nan)
        recent=g[g.outer_fold.isin([4,5,6,7])][value_col].to_numpy(float); recent=recent[np.isfinite(recent)]
        rows.append({key:item,"fold_count":len(vals),"mean_delta":np.mean(vals) if len(vals) else np.nan,"median_delta":np.median(vals) if len(vals) else np.nan,"ci95_low":low,"ci95_high":high,"positive_fold_ratio":np.mean(vals>0) if len(vals) else np.nan,"worst_fold":np.min(vals) if len(vals) else np.nan,"recent_mean_delta":np.mean(recent) if len(recent) else np.nan,"recent_positive_fold_ratio":np.mean(recent>0) if len(recent) else np.nan,"exact_sign_flip_p":exact_sign_flip_p(vals)})
    out=pd.DataFrame(rows)
    if not out.empty: out["bh_q"]=bh_adjust(out["exact_sign_flip_p"].to_numpy(float))
    return out


def run_profile_benchmark(prepared: Any, folds: list[Any], previous_output: Path, seed_dir: Path, legacy_dataset_signature: str, best_iterations: dict[int,int], model_config: dict[str,Any], *, workers:int, threads:int, seeds:list[int]) -> tuple[dict[str,list[str]], str, dict[str,Any]]:
    out=previous_output/"confirmatory_v2"; task_dir=out/"task_results"/"lightgbm_profiles"; task_dir.mkdir(parents=True,exist_ok=True)
    profiles,manifest=build_profiles(previous_output,seed_dir,prepared.feature_names)
    confirm_signature=canonical_hash({"version":PROFILE_VERSION,"dataset_signature":legacy_dataset_signature,"profiles":{k:hash_strings(v) for k,v in profiles.items()},"seeds":seeds,"model_config":model_config,"best_iterations":best_iterations})
    manifest["confirm_signature"]=confirm_signature; atomic_json(manifest,out/"profile_manifest.json")
    index={f:i for i,f in enumerate(prepared.feature_names)}
    plans=[]
    for fold in folds:
        for name,features in profiles.items():
            plans.append({"cache_root":str(prepared.root),"output_dir":str(task_dir),"dataset_signature":legacy_dataset_signature,"confirm_signature":confirm_signature,"fold":fold.to_dict(),"profile":name,"profile_hash":hash_strings(features),"indices":[index[f] for f in features],"seeds":seeds,"best_iteration":best_iterations[fold.fold_id],"model_config":model_config,"threads":threads})
    run_summary=_run_plans(plans,_profile_worker,workers,out,"profile_benchmark")
    metrics=_read_json_dir(task_dir); metrics.to_csv(out/"profile_metrics.csv",index=False)
    base=metrics[metrics.profile.eq("P0_FULL_439")][["outer_fold","seed",*METRICS]].rename(columns={m:f"base_{m}" for m in METRICS})
    paired=metrics.merge(base,on=["outer_fold","seed"],how="inner",validate="many_to_one")
    for m in METRICS:
        if m in paired and f"base_{m}" in paired:
            paired[f"delta_{m}"]=paired[m]-paired[f"base_{m}"] if m not in {"raw_brier","raw_logloss"} else paired[f"base_{m}"]-paired[m]
    paired.to_csv(out/"profile_paired_deltas.csv",index=False)
    fold_effects=paired.groupby(["profile","outer_fold"],as_index=False)["delta_raw_pr_auc"].mean().rename(columns={"delta_raw_pr_auc":"fold_mean_delta"})
    fold_effects.to_csv(out/"profile_fold_effects.csv",index=False)
    summary=_effect_summary(fold_effects,"profile"); summary.to_csv(out/"profile_summary.csv",index=False)
    candidates=summary[summary.profile.ne("P0_FULL_439")].copy()
    eligible=candidates[(candidates.mean_delta>0)&(candidates.recent_mean_delta>=-0.002)&(candidates.positive_fold_ratio>=0.625)&(candidates.worst_fold>=-0.02)]
    winner = str(eligible.sort_values(["mean_delta", "recent_mean_delta"], ascending=False).iloc[0].profile) if not eligible.empty else "P0_FULL_439"
    decision={"winner":winner,"run_summary":run_summary,"profile_counts":{k:len(v) for k,v in profiles.items()}}
    atomic_json(decision,out/"profile_decision.json")
    return profiles,winner,decision


def _candidate_worker(plan: dict[str,Any]) -> dict[str,Any]:
    ensure_thread_env(int(plan["threads"])); import lightgbm as lgb
    X=np.load(Path(plan["cache_root"])/"X_all_valid.npy",mmap_mode="r"); y=np.load(Path(plan["cache_root"])/"target.npy",mmap_mode="r"); dates=np.load(Path(plan["cache_root"])/"dates_ns.npy",mmap_mode="r")
    fold=plan["fold"]; tr=slice(fold["train_start"],fold["train_stop"]); va=slice(fold["validation_start"],fold["validation_stop"])
    x_train=np.asarray(X[tr],dtype=np.float32); y_train=np.asarray(y[tr],dtype=np.uint8); x_valid=np.asarray(X[va],dtype=np.float32); y_valid=np.asarray(y[va],dtype=np.uint8); valid_dates=np.asarray(dates[va],dtype=np.int64)
    train_set=lgb.Dataset(x_train,label=y_train,feature_name=plan["feature_names"],free_raw_data=False,params={"max_bin":int(plan["model_config"].get("max_bin",255)),"feature_pre_filter":False}); train_set.construct()
    out=Path(plan["output_dir"]); out.mkdir(parents=True,exist_ok=True); completed=cached=failed=0
    for cond in plan["conditions"]:
        contrib=np.zeros(len(plan["feature_names"]),dtype=float); contrib[np.asarray(cond["enabled_indices"],dtype=np.int32)]=1.0
        for seed in plan["seeds"]:
            identity={"version":TARGET_VERSION,"confirm_signature":plan["confirm_signature"],"fold":fold,"condition_id":cond["condition_id"],"enabled_hash":hash_strings(map(str,cond["enabled_indices"])),"seed":seed,"best_iteration":plan["best_iteration"],"model_config":plan["model_config"]}
            tid=canonical_hash(identity); path=out/f"{tid}.json"; old=read_json(path,{})
            if old.get("status")=="completed" and old.get("identity")==identity: cached+=1; continue
            started=time.perf_counter()
            try:
                params=_lgb_params(plan["model_config"],y_train,plan["threads"],seed); params["feature_contri"]=contrib.tolist()
                model=lgb.train(params,train_set,num_boost_round=plan["best_iteration"],callbacks=[lgb.log_evaluation(0)]); pred=np.asarray(model.predict(x_valid),dtype=np.float32)
                result={"status":"completed","identity":identity,"task_id":tid,"backend":"lightgbm_cpu","test_type":"targeted_confirmatory","family":cond["family"],"condition_id":cond["condition_id"],"feature":cond.get("feature",""),"direction":cond["direction"],"seed":seed,"outer_fold":fold["fold_id"],"enabled_feature_count":len(cond["enabled_indices"]),"elapsed_seconds":time.perf_counter()-started,**compute_metrics(y_valid,pred,valid_dates)}
                atomic_json(result,path); completed+=1; del model,pred
            except Exception as exc:
                failed+=1; atomic_json({"status":"failed","identity":identity,"task_id":tid,"condition_id":cond["condition_id"],"seed":seed,"outer_fold":fold["fold_id"],"error":repr(exc),"traceback":traceback.format_exc()},path)
    del train_set,x_train,y_train,x_valid,y_valid,valid_dates,X,y,dates; gc.collect()
    return {"completed":completed,"cached":cached,"failed":failed}


def run_targeted_confirmatory(prepared:Any,folds:list[Any],previous_output:Path,profiles:dict[str,list[str]],winner:str,legacy_dataset_signature:str,best_iterations:dict[int,int],model_config:dict[str,Any],*,workers:int,threads:int,seeds:list[int]) -> dict[str,Any]:
    out=previous_output/"confirmatory_v2"; task_dir=out/"task_results"/"lightgbm_targeted"; task_dir.mkdir(parents=True,exist_ok=True)
    loo=pd.read_csv(previous_output/"feature_ablation_summary.csv"); cond=pd.read_csv(previous_output/"conditional_ablation_summary.csv")
    cpu=loo[loo.backend.eq("lightgbm_cpu")].copy(); gpu=loo[loo.backend.eq("xgboost_cuda")][["feature","mean_pr_auc_utility"]].rename(columns={"mean_pr_auc_utility":"gpu_mean"})
    c=cond[cond.backend.eq("lightgbm_cpu")].sort_values("mean_pr_auc_utility",ascending=False).drop_duplicates("feature")
    direct=cpu[(cpu.mean_pr_auc_utility>=0.0025)&(cpu.recent_mean_pr_auc_utility>=0.0025)&(cpu.positive_fold_ratio>=0.625)].merge(gpu,on="feature",how="left")
    direct=direct[(direct.gpu_mean.isna())|(direct.gpu_mean>=-0.002)].sort_values("mean_pr_auc_utility",ascending=False).feature.astype(str).tolist()[:15]
    add=c[(c.test_type.eq("conditional_add_nonrep"))&(c.mean_pr_auc_utility>0)&(c.recent_mean_pr_auc_utility.fillna(0)>0)].sort_values(["mean_pr_auc_utility","positive_fold_ratio"],ascending=False).feature.astype(str).tolist()
    add=[f for f in add if f not in set(profiles["P4_CORR095"])][:15]
    merged=cpu.merge(c[["feature","mean_pr_auc_utility"]].rename(columns={"mean_pr_auc_utility":"cond_mean"}),on="feature",how="left").merge(gpu,on="feature",how="left")
    harmful=merged[(merged.mean_pr_auc_utility<=-0.005)&(merged.recent_mean_pr_auc_utility<=0)&(merged.cond_mean.fillna(0)<=0.003)&(merged.gpu_mean.fillna(0)<=0.001)].sort_values("mean_pr_auc_utility").feature.astype(str).tolist()[:10]
    idx={f:i for i,f in enumerate(prepared.feature_names)}
    families=[]
    def add_family(name:str,base_features:list[str],features:list[str],op:str):
        base=set(base_features); base_idx=sorted(idx[f] for f in base)
        families.append({"family":name,"condition_id":f"BASE::{name}","feature":"","direction":"baseline","enabled_indices":base_idx})
        for f in features:
            if f not in idx: continue
            enabled=set(base)
            if op=="drop": enabled.discard(f)
            else: enabled.add(f)
            families.append({"family":name,"condition_id":f"{op.upper()}::{name}::{f}","feature":f,"direction":op,"enabled_indices":sorted(idx[x] for x in enabled)})
    add_family("direct_keep",profiles[winner],[f for f in direct if f in set(profiles[winner])],"drop")
    add_family("conditional_add",profiles["P4_CORR095"],add,"add")
    add_family("harmful_drop",profiles["P1_EXACT_DEDUP"],[f for f in harmful if f in set(profiles["P1_EXACT_DEDUP"])],"drop")
    confirm_signature=canonical_hash({"version":TARGET_VERSION,"dataset":legacy_dataset_signature,"winner":winner,"conditions":[(x["condition_id"],hash_strings(map(str,x["enabled_indices"]))) for x in families],"seeds":seeds})
    atomic_json({"confirm_signature":confirm_signature,"winner":winner,"direct_candidates":direct,"conditional_add_candidates":add,"harmful_candidates":harmful,"conditions":[{k:v for k,v in x.items() if k!="enabled_indices"}|{"enabled_count":len(x["enabled_indices"])} for x in families]},out/"targeted_manifest.json")
    plans=[{"cache_root":str(prepared.root),"output_dir":str(task_dir),"feature_names":prepared.feature_names,"fold":fold.to_dict(),"conditions":families,"seeds":seeds,"best_iteration":best_iterations[fold.fold_id],"model_config":model_config,"threads":threads,"confirm_signature":confirm_signature} for fold in folds]
    run_summary=_run_plans(plans,_candidate_worker,workers,out,"targeted_confirmatory")
    metrics=_read_json_dir(task_dir); metrics.to_csv(out/"targeted_metrics.csv",index=False)
    bases=metrics[metrics.direction.eq("baseline")][["family","outer_fold","seed","raw_pr_auc"]].rename(columns={"raw_pr_auc":"baseline_raw_pr_auc"})
    paired=metrics[~metrics.direction.eq("baseline")].merge(bases,on=["family","outer_fold","seed"],how="inner",validate="many_to_one")
    paired["utility"] = np.where(paired.family.eq("direct_keep"), paired.baseline_raw_pr_auc-paired.raw_pr_auc, paired.raw_pr_auc-paired.baseline_raw_pr_auc)
    paired.to_csv(out/"targeted_paired.csv",index=False)
    fold_effects=paired.groupby(["family","feature","outer_fold"],as_index=False).utility.mean().rename(columns={"utility":"fold_mean_delta"})
    fold_effects["candidate_key"] = fold_effects["family"].astype(str) + "::" + fold_effects["feature"].astype(str)
    fold_effects.to_csv(out/"targeted_fold_effects.csv",index=False)
    summary=_effect_summary(fold_effects,"candidate_key")
    key_map=fold_effects[["candidate_key","feature","family"]].drop_duplicates("candidate_key")
    summary=summary.merge(key_map,on="candidate_key",how="left")
    summary.to_csv(out/"targeted_summary.csv",index=False)
    final={"run_summary":run_summary,"winner":winner,"candidate_counts":{"direct":len(direct),"conditional_add":len(add),"harmful":len(harmful)}}; atomic_json(final,out/"targeted_final.json"); return final
