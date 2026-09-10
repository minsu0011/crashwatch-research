from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock, set_full_load_mode, setup_logging
from cwgeneralization.core import build_project_inventory, leakage_audit, prepare_runtime
from cwgeneralization.engine import make_plan, run_parallel
from cwgeneralization.report import confirmation_dependencies, final_analysis, selection_analysis


def _monitor(stop: threading.Event, output: Path, seconds: float) -> None:
    rows: list[dict[str, Any]] = []
    path = output / "resource_usage.csv"
    while not stop.wait(seconds):
        vm = psutil.virtual_memory()
        rows.append({
            "epoch": time.time(), "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": (vm.total - vm.available) / 1024**3, "ram_available_gb": vm.available / 1024**3,
            **nvml_snapshot(),
        })
        if len(rows) >= 12:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
            rows.clear()
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _plans(
    output: Path, stage: str, candidates: list[dict[str, Any]], folds: list[dict[str, Any]],
    seeds: list[int], meta: dict[str, Any], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lgb=[]; xgb=[]
    for candidate in candidates:
        for fold in folds:
            lgb.append(make_plan(output, stage, candidate, fold, "lightgbm", seeds, meta, config))
            xgb.append(make_plan(output, stage, candidate, fold, "xgboost", seeds, meta, config))
    return lgb, xgb


def _run_stage(
    output: Path, stage: str, lgb_plans: list[dict[str, Any]], xgb_plans: list[dict[str, Any]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status = {
        "status": "RUNNING", "stage": stage, "started_epoch": time.time(),
        "lightgbm_bundles": len(lgb_plans), "xgboost_bundles": len(xgb_plans),
        "fits_per_bundle": len(lgb_plans[0]["seeds"]) if lgb_plans else len(xgb_plans[0]["seeds"]) if xgb_plans else 0,
    }
    atomic_json(status, output / f"{stage.upper()}_STATUS.json")
    lgb, xgb = run_parallel(
        lgb_plans, xgb_plans,
        lgb_workers=int(config["resources"]["lightgbm_workers"]),
        xgb_workers=int(config["resources"]["xgboost_workers"]), output=output,
    )
    status.update({"status": "COMPLETE", "completed_epoch": time.time(),
                   "lightgbm_complete": len(lgb), "xgboost_complete": len(xgb)})
    atomic_json(status, output / f"{stage.upper()}_STATUS.json")
    return lgb, xgb


def _diagnostic_plans(
    output: Path, meta: dict[str, Any], config: dict[str, Any], folds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates=np.load(meta["dates_path"],mmap_mode="r"); valid=np.load(meta["valid_path"],mmap_mode="r").astype(bool)
    regimes=np.load(meta["regime_code_path"],mmap_mode="r"); sectors=np.load(meta["sector_code_path"],mmap_mode="r")
    diagnostic=[]
    base={"profile":"P2","weighting":"none","train_window_days":0,"extras":[],"eligible":False}
    for fold in [row for row in folds if row["role"] != "unused"]:
        validation = valid & (dates>=int(fold["validation_start_ns"])) & (dates<=int(fold["validation_end_ns"]))
        for index,name in enumerate(meta["regimes"]):
            if int(np.sum(validation & (regimes==index))) >= 40:
                candidate={**base,"id":f"LORO_{name}"}
                diagnostic.append(make_plan(output,"diagnostics",candidate,fold,"xgboost",[17],meta,config,
                                            exclude_train_regime=index,validation_regime=index))
        for index,name in enumerate(meta["sectors"]):
            if int(np.sum(validation & (sectors==index))) >= 40:
                candidate={**base,"id":f"LOSO_{name}"}
                diagnostic.append(make_plan(output,"diagnostics",candidate,fold,"xgboost",[17],meta,config,
                                            exclude_train_sector=index,validation_sector=index))
    permutation=[]
    fold=max((fold for fold in folds if fold["role"]=="selection"),key=lambda row:int(row["fold_id"]))
    for number in range(1,int(config["robustness"]["permutation_reps"])+1):
        candidate={**base,"id":f"BLOCK_PERMUTATION_{number:02d}"}
        permutation.append(make_plan(output,"permutation",candidate,fold,"xgboost",[17],meta,config,block_permutation=number))
    return diagnostic,permutation


def _print_verdict(verdict: dict[str, Any]) -> None:
    print("========================================")
    print("CRASHWATCH GENERALIZATION FINAL VERDICT")
    print("=======================================")
    print(f"CURRENT CHAMPION:\n{verdict['current_champion']}\n")
    print(f"BEST CHALLENGER:\n{verdict['best_challenger']}\n")
    print(f"PRIMARY GOAL:\n{verdict['primary_goal']}\n")
    print(f"WEAK TICKER IMPROVEMENT:\n{verdict['weak_ticker_improvement']}\n")
    print(f"MEAN PR LIFT:\n{verdict['mean_pr_lift']}\n")
    print(f"MEAN ROC-AUC:\n{verdict['mean_roc_auc']}\n")
    print(f"WORST REGIME ROC:\n{verdict['worst_regime_roc']}\n")
    print(f"RECENT PERIOD ROC:\n{verdict['recent_period_roc']}\n")
    print(f"OVERFIT RISK:\n{verdict['overfit_risk']}\n")
    print(f"GENERALIZATION:\n{verdict['generalization']}\n")
    print(f"REPLACE CURRENT CHAMPION:\n{verdict['replace_current_champion']}\n")
    print(f"READY FOR NEW FINAL SEALED:\n{verdict['ready_for_new_final_sealed']}")
    print("========")


def run(args: argparse.Namespace) -> dict[str, Any]:
    package=Path(__file__).resolve().parent; project=package.parent
    config=read_json(package/"config_generalization.json",{})
    if args.smoke:
        config=copy.deepcopy(config)
        config["candidates"]=[row for row in config["candidates"] if row["id"]=="CHAMPION_CURRENT"]
        config["derived_candidates"]=[]; config["seeds"]=[17]
        config["evaluation"]["selection_folds"]=[0]; config["evaluation"]["confirmation_folds"]=[6]
        for profile in config["models"].values():
            profile["lightgbm"]["rounds"]=20; profile["xgboost"]["rounds"]=20
        config["robustness"]["permutation_reps"]=1; config["robustness"]["bootstrap_reps"]=20
    elif os.environ.get("CRASHWATCH_RESOURCE_MODE", "").strip().lower() == "full":
        # Resource-only override: keep config_generalization.json unchanged so
        # model/data checkpoint identities remain reusable across mode switches.
        config = copy.deepcopy(config)
        config["resources"].update({
            "total_threads": 32,
            "affinity_cpus": list(range(32)),
            "lightgbm_workers": 5,
            "lightgbm_threads_per_worker": 4,
            "xgboost_workers": 4,
            "xgboost_threads_per_worker": 4,
            "xgboost_gpu_duty_cycle": 1.0,
            "priority": "high",
        })
    output=(project/("crashwatch_ai_data/long_horizon_3d4_smoke_v1" if args.smoke else config["output_dir"])).resolve()
    output.mkdir(parents=True,exist_ok=True)
    setup_logging(output/"experiment.log")
    lock=output/"experiment.lock.json"; acquire_lock(lock)
    stop=threading.Event(); monitor=threading.Thread(target=_monitor,args=(stop,output,float(config["resources"]["resource_poll_seconds"])),daemon=True)
    started=time.time()
    try:
        hardware=set_full_load_mode(config["resources"]["total_threads"],config["resources"]["priority"])
        affinity = config["resources"].get("affinity_cpus")
        if affinity:
            psutil.Process(os.getpid()).cpu_affinity([int(cpu) for cpu in affinity])
            hardware["affinity"] = psutil.Process(os.getpid()).cpu_affinity()
        hardware["gpu"]=nvml_snapshot()
        hardware["mode"]="game_mode" if config["resources"]["priority"] == "below_normal" else "full_load_7950x3d_96gb_rtx5080"
        atomic_json(hardware,output/"HARDWARE.json"); monitor.start()
        atomic_json({"status":"PREFLIGHT","started_epoch":started,"pid":os.getpid(),"smoke":args.smoke},output/"RUN_STATUS.json")
        build_project_inventory(package,project,config,output)
        audit=leakage_audit(package,project,config,output)
        if audit["status"]!="PASS_NO_HIGH_RISK": raise RuntimeError(f"leakage audit blocked: {audit}")
        meta=prepare_runtime(package,project,config,output)
        folds=meta["folds"]
        selection_folds=[row for row in folds if row["role"]=="selection"]
        confirmation_folds=[row for row in folds if row["role"]=="confirmation"]
        candidates=list(config["candidates"])
        selection_lgb_plans,selection_xgb_plans=_plans(output,"selection",candidates,selection_folds,config["seeds"],meta,config)
        selection_lgb,selection_xgb=_run_stage(output,"selection",selection_lgb_plans,selection_xgb_plans,config)
        selection=selection_analysis(package,output,config,meta,candidates,selection_lgb,selection_xgb)
        deps=confirmation_dependencies(selection["result"]["finalists"],config)
        confirmation_candidates=[row for row in candidates if row["id"] in deps]
        confirmation_lgb_plans,confirmation_xgb_plans=_plans(output,"confirmation",confirmation_candidates,confirmation_folds,config["seeds"],meta,config)
        confirmation_lgb,confirmation_xgb=_run_stage(output,"confirmation",confirmation_lgb_plans,confirmation_xgb_plans,config)
        diagnostic_plans,permutation_plans=_diagnostic_plans(output,meta,config,folds)
        _,diagnostic_results=_run_stage(output,"diagnostics",[],diagnostic_plans,config)
        _,permutation_results=_run_stage(output,"permutation",[],permutation_plans,config)
        final=final_analysis(package,project,output,config,meta,selection,candidates,selection_lgb,selection_xgb,
                             confirmation_candidates,confirmation_lgb,confirmation_xgb,diagnostic_results,permutation_results)
        status={"status":"COMPLETE","elapsed_seconds":time.time()-started,"output":str(output),"verdict":final["verdict"],
                "sealed_values_read":False,"new_final_sealed_opened":False}
        atomic_json(status,output/"RUN_STATUS.json"); _print_verdict(final["verdict"]); return status
    except Exception as exc:
        failed={"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.time()-started}
        atomic_json(failed,output/"RUN_STATUS.json"); raise
    finally:
        stop.set()
        if monitor.is_alive(): monitor.join(timeout=10)
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="CrashWatch long-horizon 3-day/-4% development-only generalization")
    parser.add_argument("--smoke",action="store_true")
    return parser.parse_args()


if __name__=="__main__":
    print(json.dumps(run(parse_args()),indent=2,ensure_ascii=False))
