from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from cw7h.utils import atomic_json, canonical_hash, read_json
from cwfull.common import acquire_lock, load_context, nvml_snapshot, release_lock, set_full_load_mode
from .engine import make_bundle_plan, run_parallel_bundles
from .regimes import REGIMES, build_market_regimes
from .scoring import horizon_positive_recall, metrics_by_regime
from .target import build_3d5_target_from_ret1
from .windows import build_two_bank_windows


def _profile_features(package_root: Path) -> dict[str, list[str]]:
    manifest = json.load((package_root / "seed_results" / "profile_manifest.json").open(encoding="utf-8"))
    profiles = {name: list(payload["features"]) for name, payload in manifest["profiles"].items()}
    required = {"P2_DEDUP_CLEAN": 371, "P7_CORR095_PLUS_CONDITIONAL": 341}
    for name, count in required.items():
        if len(profiles.get(name, [])) != count:
            raise RuntimeError(f"{name} expected={count} actual={len(profiles.get(name, []))}")
    return {name: profiles[name] for name in required}


def _write_profile_matrix(source_path: Path, feature_names: list[str], wanted: list[str], output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in index]
    if missing:
        raise KeyError(f"profile features missing: {missing[:20]}")
    cols = np.asarray([index[name] for name in wanted], dtype=np.int32)
    X = np.load(source_path, mmap_mode="r")
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(X.shape[0], len(cols)))
    chunk = 8192
    for start in range(0, X.shape[0], chunk):
        stop = min(X.shape[0], start + chunk)
        out[start:stop] = np.asarray(X[start:stop][:, cols], dtype=np.float32)
    out.flush(); del out


def _monitor(stop_event: threading.Event, path: Path, seconds: float) -> None:
    rows = []
    while not stop_event.is_set():
        vm = psutil.virtual_memory()
        gpu = nvml_snapshot()
        rows.append({
            "epoch": time.time(), "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": (vm.total - vm.available) / 1024**3, "ram_available_gb": vm.available / 1024**3,
            **gpu,
        })
        if len(rows) >= 12:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
            rows.clear()
        stop_event.wait(seconds)
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _prepare_runtime_cache(package_root: Path, project_root: Path, output_dir: Path, dataset_override: str | None, config: dict[str, Any]):
    prepared, refs, old_folds, legacy, old_best = load_context(package_root, project_root, dataset_override, output_dir)
    cache = output_dir / "runtime_cache"; cache.mkdir(parents=True, exist_ok=True)
    feature_names = list(prepared.feature_names)
    index = {f: i for i, f in enumerate(feature_names)}
    ret_feature = str(config["target"].get("return_feature", "t_price_ret_1"))
    if ret_feature not in index:
        raise KeyError(f"target return feature missing: {ret_feature}")
    market_candidates = list(config["regime"].get("market_proxy_candidates", []))
    market_feature = next((f for f in market_candidates if f in index), None)
    if not market_feature:
        raise KeyError(f"market proxy feature not found: {market_candidates}")

    X = np.load(prepared.x_path, mmap_mode="r")
    dates = np.load(prepared.dates_path, mmap_mode="r")
    tickers = np.load(prepared.tickers_path, mmap_mode="r")
    ret1 = np.asarray(X[:, index[ret_feature]], dtype=np.float64)
    target = build_3d5_target_from_ret1(
        dates, tickers, ret1,
        horizon_days=int(config["target"]["horizon_trading_days"]),
        drop_threshold=float(config["target"]["drop_threshold"]),
    )
    target_path = cache / "target_3d5.npy"; valid_path = cache / "target_valid.npy"; first_hit_path = cache / "first_hit_day.npy"
    np.save(target_path, target.label); np.save(valid_path, target.valid); np.save(first_hit_path, target.first_hit_day)
    atomic_json(target.audit, output_dir / "TARGET_3D5_AUDIT.json")

    regime_cfg = config["regime"]
    regime_result = build_market_regimes(
        dates, np.asarray(X[:, index[market_feature]], dtype=np.float64),
        high_vol_percentile=float(regime_cfg["high_vol_percentile"]),
        bull_20d=float(regime_cfg["bull_20d"]), bear_20d=float(regime_cfg["bear_20d"]),
        crash_5d=float(regime_cfg["crash_5d"]), crash_20d=float(regime_cfg["crash_20d"]),
        rebound_5d=float(regime_cfg["rebound_5d"]), rebound_drawdown_60=float(regime_cfg["rebound_drawdown_60"]),
        vol_rank_window=int(regime_cfg["vol_rank_window"]),
    )
    regime_result.calendar.to_csv(output_dir / "regime_calendar.csv", index=False, encoding="utf-8-sig")
    regime_audit = dict(regime_result.audit); regime_audit["market_proxy_feature"] = market_feature
    atomic_json(regime_audit, output_dir / "REGIME_DEFINITION.json")

    search_windows, confirm_windows, window_audit = build_two_bank_windows(regime_result.calendar, config["banks"])
    pd.DataFrame([w.to_dict() for w in search_windows]).to_csv(output_dir / "regime_windows_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([w.to_dict() for w in confirm_windows]).to_csv(output_dir / "regime_windows_confirm.csv", index=False, encoding="utf-8-sig")
    atomic_json(window_audit, output_dir / "REGIME_BANK_AUDIT.json")

    profiles = _profile_features(package_root)
    matrix_paths = {}
    for name, features in profiles.items():
        path = cache / f"X_{name}.npy"
        _write_profile_matrix(prepared.x_path, feature_names, features, path)
        matrix_paths[name] = str(path)

    meta = {
        "dataset_path": str(prepared.dataset_path), "dataset_date_max": prepared.manifest["date_max"],
        "dataset_signature": prepared.signature, "market_proxy_feature": market_feature, "return_feature": ret_feature,
        "feature_names": feature_names, "profiles": profiles,
        "matrix_paths": matrix_paths, "target_path": str(target_path), "valid_path": str(valid_path),
        "first_hit_path": str(first_hit_path), "dates_path": str(prepared.dates_path), "tickers_path": str(prepared.tickers_path),
        "all_x_path": str(prepared.x_path), "legacy_cache_root": str(prepared.root),
    }
    atomic_json(meta, output_dir / "RUNTIME_MANIFEST.json")
    return meta, search_windows, confirm_windows, regime_result.calendar


def _bundle_paths_for_profile(meta: dict[str, Any], profile: str) -> dict[str, str]:
    return {
        "matrix_path": meta["matrix_paths"][profile], "target_path": meta["target_path"], "valid_path": meta["valid_path"],
        "dates_path": meta["dates_path"], "first_hit_path": meta["first_hit_path"],
    }


def _build_stage_plans(output_dir: Path, stage: str, windows, profiles, configs, seeds, meta, cpu_cfg):
    lgb_plans = []; xgb_plans = []
    for profile in profiles:
        paths = _bundle_paths_for_profile(meta, profile)
        for cfg in configs["lightgbm"]:
            for window in windows:
                lgb_plans.append(make_bundle_plan(
                    output_dir, stage, "lightgbm", profile, cfg["id"], cfg["params"], window.to_dict(), seeds, paths,
                    int(cpu_cfg["threads_per_lightgbm_worker"]),
                ))
        for cfg in configs["xgboost"]:
            for window in windows:
                xgb_plans.append(make_bundle_plan(
                    output_dir, stage, "xgboost", profile, cfg["id"], cfg["params"], window.to_dict(), seeds, paths,
                    int(cpu_cfg["threads_per_xgboost_worker"]),
                ))
    return lgb_plans, xgb_plans


def _date_regime_map(calendar: pd.DataFrame) -> dict[int, str]:
    return {int(r.date_ns): str(r.regime) for r in calendar.itertuples(index=False)}


def _collect_predictions(results: list[dict[str, Any]], profile: str, config_id: str, meta: dict[str, Any], calendar: pd.DataFrame) -> pd.DataFrame:
    dates = np.load(meta["dates_path"], mmap_mode="r")
    tickers = np.load(meta["tickers_path"], mmap_mode="r")
    date_to_regime = _date_regime_map(calendar)
    parts = []
    for result in results:
        if result.get("profile") != profile or result.get("config_id") != config_id or result.get("status") != "complete":
            continue
        # Explicit close matters on Windows when many checkpoint NPZ files are read.
        with np.load(result["npz_path"]) as payload:
            idx = payload["val_idx"].astype(np.int64)
            pred = payload["predictions"].mean(axis=0)
            rows = pd.DataFrame({
                "row_id": idx, "date_ns": np.asarray(dates[idx], dtype=np.int64), "ticker": np.asarray(tickers[idx]).astype(str),
                "target": payload["y"].astype(np.uint8), "first_hit_day": payload["first_hit"].astype(np.int8),
                "prediction": pred.astype(float), "window_id": result["window_id"],
            })
        rows["date"] = pd.to_datetime(rows["date_ns"]).dt.strftime("%Y-%m-%d")
        rows["regime"] = rows["date_ns"].map(date_to_regime)
        parts.append(rows)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Window selection is non-overlapping by construction, but keep deterministic guard.
    if out["row_id"].duplicated().any():
        agg = out.groupby("row_id", as_index=False).agg({
            "date_ns": "first", "ticker": "first", "target": "first", "first_hit_day": "first",
            "prediction": "mean", "window_id": lambda x: "+".join(sorted(set(map(str, x)))), "date": "first", "regime": "first",
        })
        return agg
    return out


def _summarize_candidates(results, profiles, config_list, meta, calendar, out_prefix: Path, model: str) -> pd.DataFrame:
    summaries = []; regime_frames = []
    for profile in profiles:
        for cfg in config_list:
            pred = _collect_predictions(results, profile, cfg["id"], meta, calendar)
            if pred.empty:
                continue
            score, by_regime = metrics_by_regime(pred)
            summaries.append({
                "model": model, "profile": profile, "config_id": cfg["id"], "selection_score": score["selection_score"],
                **{f"overall_{k}": v for k, v in score["overall"].items()},
                "regime_pr_lift_geometric_mean": score.get("regime_pr_lift_geometric_mean"),
                "worst_regime_pr_lift": score.get("worst_regime_pr_lift"), "top3_precision_lift": score.get("top3_precision_lift"),
            })
            by_regime.insert(0, "model", model); by_regime.insert(1, "profile", profile); by_regime.insert(2, "config_id", cfg["id"])
            regime_frames.append(by_regime)
    summary = pd.DataFrame(summaries).sort_values("selection_score", ascending=False)
    summary.to_csv(str(out_prefix) + f"_{model}_summary.csv", index=False, encoding="utf-8-sig")
    if regime_frames:
        pd.concat(regime_frames, ignore_index=True).to_csv(str(out_prefix) + f"_{model}_regime_metrics.csv", index=False, encoding="utf-8-sig")
    return summary


def _top_configs(summary: pd.DataFrame, profiles: list[str], top_n: int) -> dict[str, list[str]]:
    return {
        profile: summary[summary["profile"] == profile].head(top_n)["config_id"].astype(str).tolist()
        for profile in profiles
    }


def _filter_configs(all_cfg: list[dict[str, Any]], ids_by_profile: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    by_id = {cfg["id"]: cfg for cfg in all_cfg}
    return {profile: [by_id[x] for x in ids] for profile, ids in ids_by_profile.items()}


def _build_confirm_plans(output_dir, windows, profiles, lgb_selected, xgb_selected, seeds, meta, cpu_cfg):
    lgb=[]; xgb=[]
    for profile in profiles:
        paths = _bundle_paths_for_profile(meta, profile)
        for cfg in lgb_selected[profile]:
            for w in windows:
                lgb.append(make_bundle_plan(output_dir, "confirm", "lightgbm", profile, cfg["id"], cfg["params"], w.to_dict(), seeds, paths, int(cpu_cfg["threads_per_lightgbm_worker"])))
        for cfg in xgb_selected[profile]:
            for w in windows:
                xgb.append(make_bundle_plan(output_dir, "confirm", "xgboost", profile, cfg["id"], cfg["params"], w.to_dict(), seeds, paths, int(cpu_cfg["threads_per_xgboost_worker"])))
    return lgb,xgb


def _blend_grid(confirm_lgb, confirm_xgb, lgb_summary, xgb_summary, profiles, meta, calendar, output_dir, weights):
    rows=[]; regime_rows=[]; horizon_rows=[]; best_recipes={}
    for profile in profiles:
        best_lgb = lgb_summary[lgb_summary.profile == profile].iloc[0]
        best_xgb = xgb_summary[xgb_summary.profile == profile].iloc[0]
        lgb_pred = _collect_predictions(confirm_lgb, profile, str(best_lgb.config_id), meta, calendar)
        xgb_pred = _collect_predictions(confirm_xgb, profile, str(best_xgb.config_id), meta, calendar)
        merged = lgb_pred.merge(xgb_pred[["row_id", "prediction"]], on="row_id", suffixes=("_lgb", "_xgb"), validate="one_to_one")
        for w in weights:
            pred = merged.copy()
            pred["prediction"] = (1.0 - float(w)) * pred["prediction_lgb"] + float(w) * pred["prediction_xgb"]
            score, by_regime = metrics_by_regime(pred)
            row = {
                "profile": profile, "lgb_config_id": str(best_lgb.config_id), "xgb_config_id": str(best_xgb.config_id),
                "xgb_weight": float(w), "selection_score": score["selection_score"],
                **{f"overall_{k}": v for k, v in score["overall"].items()},
                "regime_pr_lift_geometric_mean": score.get("regime_pr_lift_geometric_mean"),
                "worst_regime_pr_lift": score.get("worst_regime_pr_lift"), "top3_precision_lift": score.get("top3_precision_lift"),
            }
            rows.append(row)
            by_regime.insert(0,"profile",profile); by_regime.insert(1,"xgb_weight",float(w)); regime_rows.append(by_regime)
            h = horizon_positive_recall(pred); h.insert(0,"profile",profile); h.insert(1,"xgb_weight",float(w)); horizon_rows.append(h)
        profile_rows = [r for r in rows if r["profile"] == profile]
        best_recipes[profile] = max(profile_rows, key=lambda r: r["selection_score"])
    grid = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    grid.to_csv(output_dir / "confirm_blend_grid.csv", index=False, encoding="utf-8-sig")
    pd.concat(regime_rows, ignore_index=True).to_csv(output_dir / "confirm_blend_regime_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(horizon_rows, ignore_index=True).to_csv(output_dir / "confirm_blend_hit_day_recall.csv", index=False, encoding="utf-8-sig")
    return grid, best_recipes


def _lock_recipe(best: dict[str, dict[str, Any]], config: dict[str, Any], meta: dict[str, Any], output_dir: Path, package_root: Path):
    p2 = best["P2_DEDUP_CLEAN"]; p7 = best["P7_CORR095_PLUS_CONDITIONAL"]
    diff = float(p7["selection_score"] - p2["selection_score"])
    material = float(config["selection"]["material_score_advantage"])
    worst_margin = float(config["selection"]["worst_regime_tiebreak_margin"])
    if diff >= material:
        chosen, reason = p7, "P7 material regime-balanced score advantage"
    elif diff <= -material:
        chosen, reason = p2, "P2 material regime-balanced score advantage"
    else:
        worst_diff = float(p7["worst_regime_pr_lift"] - p2["worst_regime_pr_lift"])
        if worst_diff >= worst_margin:
            chosen, reason = p7, "P7 worst-regime robustness tiebreak"
        elif worst_diff <= -worst_margin:
            chosen, reason = p2, "P2 worst-regime robustness tiebreak"
        else:
            chosen, reason = p7, "near-tie: prior locked P7 + 30 fewer features"
    profile = str(chosen["profile"])
    features = _profile_features(package_root)[profile]
    lock = {
        "status": "LOCKED_FOR_FINAL_SEALED",
        "locked_profile": profile,
        "feature_count": len(features),
        "feature_hash": canonical_hash(features, 24),
        "lgb_config_id": chosen["lgb_config_id"], "xgb_config_id": chosen["xgb_config_id"],
        "xgb_weight": float(chosen["xgb_weight"]), "selection_score": float(chosen["selection_score"]),
        "overall_raw_pr_auc": float(chosen["overall_raw_pr_auc"]), "overall_pr_lift": float(chosen["overall_raw_pr_auc_lift"]),
        "worst_regime_pr_lift": float(chosen["worst_regime_pr_lift"]), "reason": reason,
        "p7_minus_p2_selection_score": diff,
        "target": {"horizon_trading_days": 3, "drop_threshold": -0.05, "name": "label_abs_crash_3d_5pct"},
        "final_sealed_policy": "one-time only; no tuning after results are revealed",
        "development_data_max": str(meta["dataset_date_max"])[:10],
    }
    atomic_json(lock, output_dir / "LOCKED_3D5_RECIPE.json")
    atomic_json({"profile": profile, "feature_count": len(features), "feature_hash": lock["feature_hash"], "features": features}, output_dir / "LOCKED_3D5_FEATURE_MANIFEST.json")
    pd.DataFrame({"feature": features}).to_csv(output_dir / "LOCKED_3D5_FEATURES.csv", index=False, encoding="utf-8-sig")
    return lock


def _config_lookup(config: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    return {item["id"]: item["params"] for item in config["search_space"][model]}


def run_development(package_root: Path, project_root: Path, config: dict[str, Any], dataset_override: str | None = None, output_override: str | None = None) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_override).expanduser().resolve() if output_override else (project_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "pipeline.lock.json"
    acquire_lock(lock_path)
    stop = threading.Event()
    monitor: threading.Thread | None = None
    try:
        hardware = set_full_load_mode(int(config["cpu"]["total_threads"]), "high")
        hardware["gpu"] = nvml_snapshot(); atomic_json(hardware, output_dir / "HARDWARE_FULL_LOAD.json")
        monitor = threading.Thread(target=_monitor, args=(stop, output_dir / "resource_usage.csv", float(config["runtime"]["resource_poll_seconds"])), daemon=True); monitor.start()
        meta, search_windows, confirm_windows, calendar = _prepare_runtime_cache(package_root, project_root, output_dir, dataset_override, config)
        profiles = ["P2_DEDUP_CLEAN", "P7_CORR095_PLUS_CONDITIONAL"]
        search_seeds = list(map(int, config["selection"]["search_seeds"])); confirm_seeds = list(map(int, config["selection"]["confirm_seeds"]))
        search_cfg = config["search_space"]
        lgb_plans,xgb_plans = _build_stage_plans(output_dir, "search", search_windows, profiles, search_cfg, search_seeds, meta, config["cpu"])
        search_lgb, search_xgb = run_parallel_bundles(lgb_plans,xgb_plans,lgb_workers=int(config["cpu"]["lightgbm_workers_with_gpu"]),xgb_workers=int(config["gpu"]["workers"]),scratch_dir=output_dir/"scratch")
        search_lgb_summary = _summarize_candidates(search_lgb, profiles, search_cfg["lightgbm"], meta, calendar, output_dir/"search", "lightgbm")
        search_xgb_summary = _summarize_candidates(search_xgb, profiles, search_cfg["xgboost"], meta, calendar, output_dir/"search", "xgboost")

        top_n = int(config["selection"]["confirm_top_configs_per_model_profile"])
        lgb_ids = _top_configs(search_lgb_summary, profiles, top_n); xgb_ids = _top_configs(search_xgb_summary, profiles, top_n)
        lgb_selected = _filter_configs(search_cfg["lightgbm"], lgb_ids); xgb_selected = _filter_configs(search_cfg["xgboost"], xgb_ids)
        atomic_json({"lightgbm": lgb_ids, "xgboost": xgb_ids}, output_dir / "SEARCH_TO_CONFIRM_SELECTION.json")

        confirm_lgb_plans,confirm_xgb_plans = _build_confirm_plans(output_dir, confirm_windows, profiles, lgb_selected, xgb_selected, confirm_seeds, meta, config["cpu"])
        confirm_lgb, confirm_xgb = run_parallel_bundles(confirm_lgb_plans,confirm_xgb_plans,lgb_workers=int(config["cpu"]["lightgbm_workers_with_gpu"]),xgb_workers=int(config["gpu"]["workers"]),scratch_dir=output_dir/"scratch")
        flat_lgb_cfg=[cfg for p in profiles for cfg in lgb_selected[p]]; flat_xgb_cfg=[cfg for p in profiles for cfg in xgb_selected[p]]
        # De-duplicate ids for summary loop; configs are globally unique.
        flat_lgb_cfg=list({c['id']:c for c in flat_lgb_cfg}.values()); flat_xgb_cfg=list({c['id']:c for c in flat_xgb_cfg}.values())
        confirm_lgb_summary = _summarize_candidates(confirm_lgb, profiles, flat_lgb_cfg, meta, calendar, output_dir/"confirm", "lightgbm")
        confirm_xgb_summary = _summarize_candidates(confirm_xgb, profiles, flat_xgb_cfg, meta, calendar, output_dir/"confirm", "xgboost")
        weights = [float(x) for x in config["selection"]["blend_xgb_weights"]]
        blend_grid,best = _blend_grid(confirm_lgb,confirm_xgb,confirm_lgb_summary,confirm_xgb_summary,profiles,meta,calendar,output_dir,weights)
        lock = _lock_recipe(best, config, meta, output_dir, package_root)

        # Store exact hyperparameters for the final one-time sealed evaluator.
        lgb_lookup = _config_lookup(config, "lightgbm"); xgb_lookup = _config_lookup(config, "xgboost")
        lock["lightgbm_params"] = lgb_lookup[lock["lgb_config_id"]]; lock["xgboost_params"] = xgb_lookup[lock["xgb_config_id"]]
        atomic_json(lock, output_dir / "LOCKED_3D5_RECIPE.json")
        completion_audit = {
            "expected_search_lgb_bundles": len(lgb_plans),
            "actual_search_lgb_bundles": len(search_lgb),
            "expected_search_xgb_bundles": len(xgb_plans),
            "actual_search_xgb_bundles": len(search_xgb),
            "expected_confirm_lgb_bundles": len(confirm_lgb_plans),
            "actual_confirm_lgb_bundles": len(confirm_lgb),
            "expected_confirm_xgb_bundles": len(confirm_xgb_plans),
            "actual_confirm_xgb_bundles": len(confirm_xgb),
            "expected_model_fits": (
                len(lgb_plans) * len(search_seeds) + len(xgb_plans) * len(search_seeds)
                + len(confirm_lgb_plans) * len(confirm_seeds) + len(confirm_xgb_plans) * len(confirm_seeds)
            ),
            "actual_model_fits": sum(len(r.get("seeds", [])) for r in search_lgb + search_xgb + confirm_lgb + confirm_xgb),
        }
        completion_audit["counts_ok"] = all(
            completion_audit[key] == completion_audit[key.replace("expected_", "actual_")]
            for key in list(completion_audit) if key.startswith("expected_") and key != "expected_model_fits"
        ) and completion_audit["expected_model_fits"] == completion_audit["actual_model_fits"]
        atomic_json(completion_audit, output_dir / "DEVELOPMENT_COMPLETION_AUDIT.json")
        if not completion_audit["counts_ok"]:
            raise RuntimeError(f"development completion audit failed: {completion_audit}")
        result = {
            "status": "completed", "elapsed_seconds": time.time()-started, "output_dir": str(output_dir),
            "search_lgb_bundles": len(search_lgb), "search_xgb_bundles": len(search_xgb),
            "confirm_lgb_bundles": len(confirm_lgb), "confirm_xgb_bundles": len(confirm_xgb),
            "models_search": sum(len(r.get('seeds',[])) for r in search_lgb+search_xgb),
            "models_confirm": sum(len(r.get('seeds',[])) for r in confirm_lgb+confirm_xgb),
            "locked_recipe": lock,
            "regime_search_complete": True, "regime_confirm_complete": True,
            "final_sealed_consumed": False,
            "completion_audit": completion_audit,
        }
        atomic_json(result, output_dir / "FINAL_DEVELOPMENT_STATUS.json")
        return result
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        release_lock(lock_path)
