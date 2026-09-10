from __future__ import annotations

import json, logging, math, os, time, threading, traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from cw7h.utils import atomic_json, canonical_hash
from cwfull.common import acquire_lock, load_context, nvml_snapshot, release_lock, set_full_load_mode, setup_logging
from cwregime.target import build_3d5_target_from_ret1
from cwregime.regimes import build_market_regimes
from .gate import build_normal_gate, NORMAL_REGIMES
from .engine import make_plan, run_parallel
from .scoring import safe_metrics, ticker_metrics, sector_metrics, score_candidate, block_bootstrap_delta, permutation_sanity

LOG=logging.getLogger(__name__)


def _atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp,index=False,encoding="utf-8-sig"); os.replace(temp,path)


def _resource_monitor(stop: threading.Event, output: Path, seconds: float):
    path=output/"resource_usage.csv"; rows=[]
    while not stop.wait(seconds):
        vm=psutil.virtual_memory(); rows.append({"epoch":time.time(),"cpu_percent":psutil.cpu_percent(interval=None),"ram_used_gb":(vm.total-vm.available)/1024**3,"ram_available_gb":vm.available/1024**3,**nvml_snapshot()})
        if len(rows)>=12:
            pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False); rows.clear()
    if rows: pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False)


def _profile_features(package_root:Path):
    m=json.load((package_root/"seed_results"/"profile_manifest.json").open(encoding="utf-8")); profiles={n:list(v["features"]) for n,v in m["profiles"].items()}
    req={"P2_DEDUP_CLEAN":371,"P7_CORR095_PLUS_CONDITIONAL":341}
    for k,n in req.items():
        if len(profiles.get(k,[]))!=n: raise RuntimeError(f"profile {k} expected={n} actual={len(profiles.get(k,[]))}")
    return {k:profiles[k] for k in req}


def _write_matrix(source:Path,names:list[str],wanted:list[str],out:Path):
    if out.exists() and out.stat().st_size>0:return
    ix={n:i for i,n in enumerate(names)}; miss=[x for x in wanted if x not in ix]
    if miss: raise KeyError(f"missing profile features: {miss[:20]}")
    cols=np.asarray([ix[x] for x in wanted],np.int32); X=np.load(source,mmap_mode="r"); mm=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=(X.shape[0],len(cols)))
    for s in range(0,X.shape[0],8192): mm[s:min(len(X),s+8192)]=np.asarray(X[s:min(len(X),s+8192)][:,cols],np.float32)
    mm.flush(); del mm


def _prepare(package_root, project_root, output_dir, dataset_override, cfg):
    prepared,refs,folds,legacy,best=load_context(package_root,project_root,dataset_override,output_dir)
    cache=output_dir/"runtime_cache"; cache.mkdir(parents=True,exist_ok=True); names=list(prepared.feature_names); ix={x:i for i,x in enumerate(names)}; X=np.load(prepared.x_path,mmap_mode="r")
    retf=cfg["target"]["return_feature"]; marketf=next((f for f in cfg["regime"]["market_proxy_candidates"] if f in ix),None)
    if retf not in ix or not marketf: raise KeyError(f"return/market feature unavailable ret={retf} market={marketf}")
    dates=np.load(prepared.dates_path,mmap_mode="r"); tickers=np.load(prepared.tickers_path,mmap_mode="r")
    target=build_3d5_target_from_ret1(dates,tickers,np.asarray(X[:,ix[retf]],float),horizon_days=int(cfg["target"]["horizon_trading_days"]),drop_threshold=float(cfg["target"]["drop_threshold"]))
    tp=cache/"target_3d5.npy"; vp=cache/"target_valid.npy"; np.save(tp,target.label); np.save(vp,target.valid); atomic_json(target.audit,output_dir/"TARGET_3D5_AUDIT.json")
    rcfg=cfg["regime"]; rr=build_market_regimes(dates,np.asarray(X[:,ix[marketf]],float),high_vol_percentile=rcfg["high_vol_percentile"],bull_20d=rcfg["bull_20d"],bear_20d=rcfg["bear_20d"],crash_5d=rcfg["crash_5d"],crash_20d=rcfg["crash_20d"],rebound_5d=rcfg["rebound_5d"],rebound_drawdown_60=rcfg["rebound_drawdown_60"],vol_rank_window=rcfg["vol_rank_window"])
    gate=build_normal_gate(dates,rr.calendar)
    gate_calendar=rr.calendar.assign(normal_market=rr.calendar["regime"].isin(NORMAL_REGIMES))
    _atomic_csv(gate_calendar,output_dir/"normal_market_calendar.csv")
    atomic_json(gate.audit,output_dir/"NORMAL_GATE_AUDIT.json")
    cov_rows=[]
    unique_dates=gate_calendar.sort_values("date_ns").reset_index(drop=True)
    for n in [63,126,252,504]:
        part=unique_dates.tail(n)
        cov_rows.append({"window_days":n,"available_dates":len(part),"normal_dates":int(part["normal_market"].sum()),"normal_coverage":float(part["normal_market"].mean()) if len(part) else np.nan})
    _atomic_csv(pd.DataFrame(cov_rows),output_dir/"normal_gate_recent_coverage.csv")
    yearly=(gate_calendar.assign(year=pd.to_datetime(gate_calendar["date_ns"]).dt.year)
            .groupby("year")["normal_market"]
            .agg(dates="count", normal_dates="sum", normal_coverage="mean")
            .reset_index())
    _atomic_csv(yearly,output_dir/"normal_gate_yearly_coverage.csv")
    npth=cache/"normal_mask.npy"; np.save(npth,gate.row_normal)
    profiles=_profile_features(package_root); matrices={}
    for n,fl in profiles.items():
        p=cache/f"X_{n}.npy"; _write_matrix(prepared.x_path,names,fl,p); matrices[n]=str(p)
    runtime_identity={"schema":"normal_market_runtime_v2","dataset_signature":prepared.signature,"target":cfg["target"],"regime":cfg["regime"],"profiles":profiles,"folds":[f.to_dict() for f in folds],"market_feature":marketf,"target_feature":retf}
    meta={"dataset":str(prepared.dataset_path),"date_min":prepared.manifest["date_min"],"date_max":prepared.manifest["date_max"],"features":names,"profiles":profiles,"matrix_paths":matrices,"target_path":str(tp),"valid_path":str(vp),"normal_path":str(npth),"dates_path":str(prepared.dates_path),"tickers_path":str(prepared.tickers_path),"sectors_path":str(prepared.buckets_path),"market_feature":marketf,"target_feature":retf,"dataset_signature":prepared.signature,"runtime_hash":canonical_hash(runtime_identity,32)}
    atomic_json(meta,output_dir/"RUNTIME_MANIFEST.json")
    return meta,[f.to_dict() for f in folds],rr.calendar


def _paths(meta,profile): return {"runtime_hash":meta["runtime_hash"],"matrix_path":meta["matrix_paths"][profile],"target_path":meta["target_path"],"valid_path":meta["valid_path"],"normal_path":meta["normal_path"],"dates_path":meta["dates_path"],"tickers_path":meta["tickers_path"],"sectors_path":meta["sectors_path"]}


def _collect(results, candidate):
    parts=[]
    for r in results:
        if r.get("candidate")!=candidate or r.get("status")!="complete":continue
        z=np.load(r["npz_path"]); idx=z["val_idx"].astype(np.int64); pred=z["predictions"].mean(axis=0)
        parts.append(pd.DataFrame({"row_id":idx,"date_ns":z["date_ns"].astype(np.int64),"ticker":z["ticker"].astype(str),"sector":z["sector"].astype(str),"target":z["y"].astype(np.uint8),"prediction":pred,"fold_id":int(r["fold_id"])}))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def _blend(lgb,xgb,w):
    keys=["row_id","fold_id"]
    m=lgb.merge(xgb[keys+["prediction"]],on=keys,suffixes=("_lgb","_xgb"),how="inner"); m["prediction"]=(1-w)*m["prediction_lgb"]+w*m["prediction_xgb"]
    return m.drop(columns=["prediction_lgb","prediction_xgb"])


def _fold_scope(frame, ids): return frame[frame["fold_id"].isin(ids)].copy()


def _choose_blend(lgb,xgb,weights,selection_folds):
    rows=[]
    for w in weights:
        f=_fold_scope(_blend(lgb,xgb,float(w)),selection_folds); s=score_candidate(f,recent_fold_id=max(selection_folds)); rows.append({"xgb_weight":float(w),"selection_score":s["selection_score"],"roc":s["overall"].get("raw_roc_auc"),"pr_lift":s["overall"].get("raw_pr_auc_lift")})
    tab=pd.DataFrame(rows).sort_values(["selection_score","roc"],ascending=False); return float(tab.iloc[0]["xgb_weight"]),tab


def _candidate_summary(frame,candidate,scope,folds,recent_fold):
    f=_fold_scope(frame,folds); s=score_candidate(f,recent_fold_id=recent_fold); o=s["overall"]
    return {"candidate":candidate,"scope":scope,"rows":o.get("rows"),"dates":int(f["date_ns"].nunique()),"positives":o.get("positives"),"positive_rate":o.get("positive_rate"),"pr_auc":o.get("raw_pr_auc"),"pr_auc_lift":o.get("raw_pr_auc_lift"),"roc_auc":o.get("raw_roc_auc"),"brier":o.get("raw_brier"),"logloss":o.get("raw_logloss"),"top3_precision":o.get("top_3pct_precision"),"top3_recall":o.get("top_3pct_recall"),"mean_fold_roc":s["mean_fold_roc"],"median_fold_roc":s["median_fold_roc"],"worst_fold_roc":s["worst_fold_roc"],"recent_fold_roc":s["recent_fold_roc"],"selection_score":s["selection_score"]}


def _custom_recent_folds(meta,cfg):
    dates=np.load(meta["dates_path"],mmap_mode="r"); ud=np.unique(dates); out=[]; purge=3
    for n in cfg["recent_audit_windows"]:
        n=int(n)
        if len(ud)<=n+purge+10: continue
        start_i=len(ud)-n; train_i=max(0,start_i-purge-1)
        out.append({"fold_id":10000+n,"train_end_ns":int(ud[train_i]),"validation_start_ns":int(ud[start_i]),"validation_end_ns":int(ud[-1]),"role":"recent_anchor","window_days":n})
    return out


def _run_impl(package_root:Path,project_root:Path,config_path:Path,dataset_override=None,output_override=None):
    cfg=json.load(config_path.open(encoding="utf-8")); out=Path(output_override) if output_override else project_root/cfg["output_dir"]
    if not out.is_absolute(): out=project_root/out
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True); setup_logging(out/"experiment.log")
    hw=set_full_load_mode(cfg["runtime"]["total_threads"],cfg["runtime"].get("priority","high")); hw["gpu"]=nvml_snapshot(); hw["mode"]="full_load_7950x3d_96gb_rtx5080"; atomic_json(hw,out/"HARDWARE.json")
    meta,folds,calendar=_prepare(package_root,project_root,out,dataset_override,cfg); atomic_json(cfg,out/"RESOLVED_CONFIG.json")
    seeds=cfg["seeds"]; lgbp=[]; xgbp=[]
    for cand in cfg["candidates"]:
        for fold in folds:
            paths=_paths(meta,cand["profile"]); lgbp.append(make_plan(out,stage="base",model="lightgbm",candidate=cand,fold=fold,seeds=seeds,paths=paths,params=cfg["models"]["lightgbm"],threads=cfg["runtime"]["threads_per_lgb"])); xgbp.append(make_plan(out,stage="base",model="xgboost",candidate=cand,fold=fold,seeds=seeds,paths=paths,params=cfg["models"]["xgboost"],threads=cfg["runtime"]["threads_per_xgb"]))
    LOG.info("base bundles: LGB=%d XGB=%d",len(lgbp),len(xgbp)); atomic_json({"stage":"BASE","status":"RUNNING","lgb_bundles":len(lgbp),"xgb_bundles":len(xgbp),"started_epoch":time.time()},out/"CURRENT_STAGE.json"); lr,xr=run_parallel(lgbp,xgbp,lgb_workers=cfg["runtime"]["lgb_workers"],xgb_workers=cfg["runtime"]["xgb_workers"],scratch_dir=out/"scratch"); atomic_json({"stage":"BASE","status":"COMPLETE","lgb_bundles":len(lr),"xgb_bundles":len(xr),"completed_epoch":time.time()},out/"BASE_STATUS.json")
    sel=cfg["fold_roles"]["selection"]; con=cfg["fold_roles"]["confirmation"]; aud=cfg["fold_roles"]["recent_audit"]
    frames={}; summaries=[]; blend_rows=[]
    for cand in cfg["candidates"]:
        cid=cand["id"]; lf=_collect(lr,cid); xf=_collect(xr,cid); w,bt=_choose_blend(lf,xf,cfg["blend_xgb_weights"],sel); bt.insert(0,"candidate",cid); blend_rows.append(bt); frame=_blend(lf,xf,w); frames[cid]=frame
        summaries += [_candidate_summary(frame,cid,"SELECTION",sel,max(sel)),_candidate_summary(frame,cid,"CONFIRMATION",con,max(con)),_candidate_summary(frame,cid,"RECENT_AUDIT",aud,max(aud))]
    _atomic_csv(pd.concat(blend_rows,ignore_index=True),out/"blend_selection.csv"); comp=pd.DataFrame(summaries); _atomic_csv(comp,out/"candidate_comparison.csv")
    # Winner is chosen on selection only, then has to survive confirmation/recent audit against fixed baseline.
    sel_tab=comp[comp.scope.eq("SELECTION")].sort_values("selection_score",ascending=False); proposed=str(sel_tab.iloc[0].candidate); baseline=cfg["baseline_candidate"]
    def row(cid,scope): return comp[(comp.candidate==cid)&(comp.scope==scope)].iloc[0]
    bcon,ccon=row(baseline,"CONFIRMATION"),row(proposed,"CONFIRMATION"); baud,caud=row(baseline,"RECENT_AUDIT"),row(proposed,"RECENT_AUDIT")
    confirm_pass=(float(ccon.roc_auc)>=float(bcon.roc_auc)+cfg["decision"]["min_confirm_roc_gain"] or float(ccon.roc_auc)>=cfg["decision"]["absolute_confirm_roc_goal"]) and float(ccon.pr_auc_lift)>=float(bcon.pr_auc_lift)-cfg["decision"]["max_confirm_pr_lift_loss"] and float(caud.roc_auc)>=float(baud.roc_auc)-cfg["decision"]["max_recent_roc_loss"]
    base_winner=proposed if confirm_pass else baseline
    # Block bootstrap on confirmation.
    boot=block_bootstrap_delta(_fold_scope(frames[baseline],con),_fold_scope(frames[base_winner],con),n_boot=cfg["robustness"]["bootstrap_replicates"],block_dates=cfg["robustness"]["bootstrap_block_dates"]); atomic_json(boot,out/"confirmation_bootstrap.json")

    # Sector expert overlay on the base winner. One shared blend weight for semiconductor+battery; weak ticker IDs never select the weight.
    bc=next(c for c in cfg["candidates"] if c["id"]==base_winner); slr=[]; sxr=[]
    for sector in cfg["sector_reinforcement"]["sectors"]:
        for fold in folds:
            paths=_paths(meta,bc["profile"]); slr.append(make_plan(out,stage="sector",model="lightgbm",candidate=bc,fold=fold,seeds=seeds,paths=paths,params=cfg["models"]["lightgbm"],threads=cfg["runtime"]["threads_per_lgb"],sector_filter=sector)); sxr.append(make_plan(out,stage="sector",model="xgboost",candidate=bc,fold=fold,seeds=seeds,paths=paths,params=cfg["models"]["xgboost"],threads=cfg["runtime"]["threads_per_xgb"],sector_filter=sector))
    atomic_json({"stage":"SECTOR","status":"RUNNING","lgb_bundles":len(slr),"xgb_bundles":len(sxr),"started_epoch":time.time()},out/"CURRENT_STAGE.json"); slres,sxres=run_parallel(slr,sxr,lgb_workers=max(2,cfg["runtime"]["lgb_workers"]//2),xgb_workers=cfg["runtime"]["xgb_workers"],scratch_dir=out/"scratch_sector"); atomic_json({"stage":"SECTOR","status":"COMPLETE","completed_epoch":time.time()},out/"SECTOR_STATUS.json")
    # Expert model ensemble uses same LGB/XGB weight selected for the candidate.
    selected_weight=float(pd.concat(blend_rows).query("candidate == @base_winner").sort_values("selection_score",ascending=False).iloc[0].xgb_weight)
    expert_parts=[]
    for sector in cfg["sector_reinforcement"]["sectors"]:
        l=_collect([r for r in slres if r.get("sector_filter")==sector],base_winner); x=_collect([r for r in sxres if r.get("sector_filter")==sector],base_winner); e=_blend(l,x,selected_weight); e["sector_filter"]=sector; expert_parts.append(e)
    experts=pd.concat(expert_parts,ignore_index=True) if expert_parts else pd.DataFrame()
    sector_weight_rows=[]; overlay_frames={}
    global_frame=frames[base_winner]
    for w in cfg["sector_reinforcement"]["blend_weights"]:
        f=global_frame.copy(); emap=experts.set_index("row_id")["prediction"] if not experts.empty else pd.Series(dtype=float); mask=f["row_id"].isin(emap.index); f.loc[mask,"prediction"]=(1-float(w))*f.loc[mask,"prediction"].to_numpy()+float(w)*f.loc[mask,"row_id"].map(emap).to_numpy(); overlay_frames[float(w)]=f
        s=score_candidate(_fold_scope(f,sel),recent_fold_id=max(sel)); sector_weight_rows.append({"sector_weight":float(w),"selection_score":s["selection_score"],"roc":s["overall"].get("raw_roc_auc"),"pr_lift":s["overall"].get("raw_pr_auc_lift")})
    swtab=pd.DataFrame(sector_weight_rows).sort_values("selection_score",ascending=False); _atomic_csv(swtab,out/"sector_blend_selection.csv"); sw=float(swtab.iloc[0].sector_weight); overlay=overlay_frames[sw]
    base_con=_candidate_summary(global_frame,base_winner,"CONFIRMATION",con,max(con)); ov_con=_candidate_summary(overlay,base_winner+"_SECTOR", "CONFIRMATION",con,max(con)); base_aud=_candidate_summary(global_frame,base_winner,"RECENT_AUDIT",aud,max(aud)); ov_aud=_candidate_summary(overlay,base_winner+"_SECTOR","RECENT_AUDIT",aud,max(aud))
    base_sector_confirm=sector_metrics(_fold_scope(global_frame,con)).set_index("sector")
    overlay_sector_confirm=sector_metrics(_fold_scope(overlay,con)).set_index("sector")
    sector_checks=[]
    for sec in cfg["sector_reinforcement"]["sectors"]:
        b=float(base_sector_confirm.loc[sec,"roc_auc"]) if sec in base_sector_confirm.index else np.nan
        o=float(overlay_sector_confirm.loc[sec,"roc_auc"]) if sec in overlay_sector_confirm.index else np.nan
        sector_checks.append({"sector":sec,"base_roc":b,"overlay_roc":o,"delta_roc":o-b if np.isfinite(b) and np.isfinite(o) else np.nan})
    sector_check_df=pd.DataFrame(sector_checks); _atomic_csv(sector_check_df,out/"sector_overlay_confirmation_checks.csv")
    sector_non_degrade=bool((sector_check_df["delta_roc"].dropna() >= -cfg["sector_reinforcement"]["max_each_sector_roc_loss"]).all())
    sector_pass=float(ov_con["roc_auc"])>=float(base_con["roc_auc"])+cfg["sector_reinforcement"]["min_confirm_roc_gain"] and float(ov_con["pr_auc_lift"])>=float(base_con["pr_auc_lift"])-cfg["sector_reinforcement"]["max_pr_lift_loss"] and float(ov_aud["roc_auc"])>=float(base_aud["roc_auc"])-cfg["sector_reinforcement"]["max_recent_roc_loss"] and sector_non_degrade
    final_frame=overlay if sector_pass else global_frame; final_id=base_winner+("_SECTOR_EXPERT" if sector_pass else "")

    # Weak tickers are REPORT-ONLY constraints, never selection targets.
    tm_parts=[]; sm_parts=[]
    for scope_name,scope_folds in [("SELECTION",sel),("CONFIRMATION",con),("RECENT_AUDIT",aud)]:
        t=ticker_metrics(_fold_scope(final_frame,scope_folds)); t.insert(0,"scope",scope_name); tm_parts.append(t)
        smm=sector_metrics(_fold_scope(final_frame,scope_folds)); smm.insert(0,"scope",scope_name); sm_parts.append(smm)
    tm=pd.concat(tm_parts,ignore_index=True); sm=pd.concat(sm_parts,ignore_index=True)
    _atomic_csv(tm,out/"ticker_metrics_normal.csv"); _atomic_csv(sm,out/"sector_metrics_normal.csv")
    weak=tm[(tm.scope=="CONFIRMATION") & tm.ticker.isin([str(x) for x in cfg["diagnostics"]["repeated_weak_tickers"]])].copy(); _atomic_csv(weak,out/"repeated_weak_ticker_report.csv")

    # Leave-one-ticker-out sector expert diagnostics for the two repeated weak names.
    loto=[]
    for ticker,sector in cfg["diagnostics"]["weak_ticker_sector_map"].items():
        plans=[]
        for fold in folds:
            plans.append(make_plan(out,stage="loto",model="lightgbm",candidate=bc,fold=fold,seeds=cfg["diagnostics"]["loto_seeds"],paths=_paths(meta,bc["profile"]),params=cfg["models"]["lightgbm"],threads=cfg["runtime"]["threads_per_lgb"],sector_filter=sector,exclude_ticker=ticker))
        rr,_=run_parallel(plans,[],lgb_workers=min(cfg["runtime"]["lgb_workers"],4),xgb_workers=1,scratch_dir=out/"scratch_loto"); ef=_collect(rr,base_winner); m=safe_metrics(ef); loto.append({"ticker":ticker,"sector":sector,"rows":m.get("rows"),"positives":m.get("positives"),"roc_auc":m.get("raw_roc_auc"),"pr_auc_lift":m.get("raw_pr_auc_lift")})
    _atomic_csv(pd.DataFrame(loto),out/"weak_ticker_leave_one_out_sector_diagnostic.csv")

    # Anchored recent NORMAL-market audit: diagnostic only, never model selection.
    rec_folds=_custom_recent_folds(meta,cfg["diagnostics"]); rlp=[]; rxp=[]
    for f in rec_folds:
        rlp.append(make_plan(out,stage="recent_anchor",model="lightgbm",candidate=bc,fold=f,seeds=seeds,paths=_paths(meta,bc["profile"]),params=cfg["models"]["lightgbm"],threads=cfg["runtime"]["threads_per_lgb"])); rxp.append(make_plan(out,stage="recent_anchor",model="xgboost",candidate=bc,fold=f,seeds=seeds,paths=_paths(meta,bc["profile"]),params=cfg["models"]["xgboost"],threads=cfg["runtime"]["threads_per_xgb"]))
    rlr,rxr=run_parallel(rlp,rxp,lgb_workers=max(2,cfg["runtime"]["lgb_workers"]//2),xgb_workers=cfg["runtime"]["xgb_workers"],scratch_dir=out/"scratch_recent"); rf=_blend(_collect(rlr,base_winner),_collect(rxr,base_winner),selected_weight) if rlr and rxr else pd.DataFrame(); recent_rows=[]
    if not rf.empty:
        for fid,part in rf.groupby("fold_id"):
            m=safe_metrics(part); recent_rows.append({"window_days":int(fid)-10000,"normal_dates":part.date_ns.nunique(),"rows":len(part),"positives":m.get("positives"),"roc_auc":m.get("raw_roc_auc"),"pr_auc_lift":m.get("raw_pr_auc_lift"),"top3_precision":m.get("top_3pct_precision")})
    _atomic_csv(pd.DataFrame(recent_rows).sort_values("window_days"),out/"recent_normal_market_audit.csv")

    sanity=permutation_sanity(_fold_scope(final_frame,con),n_perm=cfg["robustness"]["permutations"],block_dates=cfg["robustness"]["permutation_block_dates"]); atomic_json(sanity,out/"permutation_sanity.json")
    final_boot=block_bootstrap_delta(_fold_scope(frames[baseline],con),_fold_scope(final_frame,con),n_boot=cfg["robustness"]["bootstrap_replicates"],block_dates=cfg["robustness"]["bootstrap_block_dates"]); atomic_json(final_boot,out/"final_vs_baseline_bootstrap.json")
    final_m=safe_metrics(_fold_scope(final_frame,con)); audit_m=safe_metrics(_fold_scope(final_frame,aud)); normal_gate=json.load((out/"NORMAL_GATE_AUDIT.json").open(encoding="utf-8"))
    # Winner's-curse table: selection-to-confirmation decay for every candidate.
    wc=[]
    for cid in comp.candidate.unique():
        a=row(cid,"SELECTION"); b=row(cid,"CONFIRMATION"); wc.append({"candidate":cid,"selection_roc":a.roc_auc,"confirmation_roc":b.roc_auc,"roc_decay":float(b.roc_auc)-float(a.roc_auc),"selection_pr_lift":a.pr_auc_lift,"confirmation_pr_lift":b.pr_auc_lift,"pr_lift_decay":float(b.pr_auc_lift)-float(a.pr_auc_lift)})
    _atomic_csv(pd.DataFrame(wc),out/"winner_curse_normal_market.csv")
    ready=bool(np.isfinite(final_m.get("raw_roc_auc",np.nan)) and final_m.get("raw_roc_auc",0)>=cfg["decision"]["minimum_final_normal_roc"] and final_m.get("raw_pr_auc_lift",0)>=cfg["decision"]["minimum_final_pr_lift"] and sanity.get("status")=="PASS" and (not np.isfinite(final_boot.get("roc_ci_low",np.nan)) or final_boot.get("roc_ci_low",-1)>=-cfg["decision"]["max_bootstrap_roc_downside"]))
    recommendation={"target":"label_abs_crash_3d_5pct","market_policy":"NORMAL_CORE_V1","active_regimes":list(NORMAL_REGIMES),"abnormal_behavior":"ABSTAIN_ABNORMAL_MARKET","baseline_candidate":baseline,"selection_proposed":proposed,"base_confirm_pass":bool(confirm_pass),"base_winner":base_winner,"sector_overlay_weight":sw,"sector_overlay_pass":bool(sector_pass),"sector_non_degrade":sector_non_degrade,"final_candidate":final_id,"confirmation_metrics":{"roc_auc":final_m.get("raw_roc_auc"),"pr_auc_lift":final_m.get("raw_pr_auc_lift"),"top3_precision":final_m.get("top_3pct_precision")},"recent_audit_metrics":{"roc_auc":audit_m.get("raw_roc_auc"),"pr_auc_lift":audit_m.get("raw_pr_auc_lift")},"normal_market_date_coverage":normal_gate.get("date_coverage"),"weak_tickers_report_only":cfg["diagnostics"]["repeated_weak_tickers"],"permutation_sanity":sanity,"final_vs_baseline_bootstrap":final_boot,"ready_for_new_future_normal_market_holdout":ready}
    atomic_json(recommendation,out/"FINAL_RECOMMENDATION.json")
    freeze={"schema":"crashwatch_normal_market_3d5_v1","target":{"horizon_days":3,"drop_threshold":-0.05},"market_gate":{"active_regimes":list(NORMAL_REGIMES),"else":"ABSTAIN_ABNORMAL_MARKET"},"candidate":bc,"final_candidate":final_id,"sector_expert":{"enabled":bool(sector_pass),"sectors":cfg["sector_reinforcement"]["sectors"],"weight":sw},"model_params":cfg["models"],"seeds":seeds,"profile_features":meta["profiles"][bc["profile"]],"dataset_signature":meta["dataset_signature"],"selection_data_policy":"normal-market metrics only; weak tickers report-only; abnormal regimes 0 selection weight"}; atomic_json(freeze,out/"MODEL_FREEZE_MANIFEST.json")
    print("\n========================================\nCRASHWATCH NORMAL-MARKET 3D/-5% VERDICT\n========================================")
    print("BASELINE:",baseline); print("FINAL:",final_id); print("NORMAL CONFIRM ROC:",recommendation["confirmation_metrics"]["roc_auc"]); print("NORMAL CONFIRM PR LIFT:",recommendation["confirmation_metrics"]["pr_auc_lift"]); print("RECENT NORMAL ROC:",recommendation["recent_audit_metrics"]["roc_auc"]); print("NORMAL DATE COVERAGE:",recommendation["normal_market_date_coverage"]); print("ABNORMAL MARKET: ABSTAIN"); print("READY FOR NEW HOLDOUT:","YES" if recommendation["ready_for_new_future_normal_market_holdout"] else "NO"); print("========================================")
    return recommendation


def run(package_root:Path,project_root:Path,config_path:Path,dataset_override=None,output_override=None):
    cfg=json.load(config_path.open(encoding="utf-8")); out=Path(output_override) if output_override else project_root/cfg["output_dir"]
    if not out.is_absolute(): out=project_root/out
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True); setup_logging(out/"experiment.log")
    lock=out/"experiment.lock.json"; acquire_lock(lock); stop=threading.Event(); monitor=threading.Thread(target=_resource_monitor,args=(stop,out,float(cfg["runtime"].get("resource_poll_seconds",5))),daemon=True); started=time.time(); monitor.start()
    atomic_json({"status":"RUNNING","stage":"PREFLIGHT","pid":os.getpid(),"started_epoch":started},out/"RUN_STATUS.json")
    try:
        result=_run_impl(package_root,project_root,config_path,dataset_override,output_override)
        atomic_json({"status":"COMPLETE","pid":os.getpid(),"started_epoch":started,"completed_epoch":time.time(),"elapsed_seconds":time.time()-started,"output":str(out),"final_candidate":result.get("final_candidate"),"ready_for_new_future_normal_market_holdout":result.get("ready_for_new_future_normal_market_holdout")},out/"RUN_STATUS.json")
        return result
    except Exception as exc:
        atomic_json({"status":"FAILED","pid":os.getpid(),"started_epoch":started,"failed_epoch":time.time(),"elapsed_seconds":time.time()-started,"error":repr(exc),"traceback":traceback.format_exc()},out/"RUN_STATUS.json")
        raise
    finally:
        stop.set(); monitor.join(timeout=15); release_lock(lock)
