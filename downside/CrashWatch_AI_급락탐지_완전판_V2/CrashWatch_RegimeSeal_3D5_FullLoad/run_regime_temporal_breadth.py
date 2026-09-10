from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, release_lock
from cwregime.engine import make_bundle_plan
from cwregime.scoring import horizon_positive_recall, metrics_by_regime
from run_regime_game_followup import (
    PROFILES,
    _fit_calibrators,
    _frames,
    _paths,
    _resource_monitor,
    _run_lgb,
    _run_throttled_xgb,
    _search_predictions,
    _selected_configs,
    _set_game_mode,
    _sha256,
)


FIXED_CANDIDATES = [
    ("LOCKED_P2_RAW_W09", "P2_DEDUP_CLEAN", "raw", 0.9),
    ("P2_LGB_RAW", "P2_DEDUP_CLEAN", "raw", 0.0),
    ("P2_PLATT_W02", "P2_DEDUP_CLEAN", "platt", 0.2),
    ("P7_PLATT_W04", "P7_CORR095_PLUS_CONDITIONAL", "platt", 0.4),
    ("P7_PLATT_W02", "P7_CORR095_PLUS_CONDITIONAL", "platt", 0.2),
    ("P7_LGB_RAW", "P7_CORR095_PLUS_CONDITIONAL", "raw", 0.0),
]


def _overlap(a: int, b: int, c: int, d: int) -> bool:
    return not (b <= c or a >= d)


def _select_temporal_windows(
    calendar: pd.DataFrame,
    excluded: pd.DataFrame,
    bank_cfg: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    cal = calendar.sort_values("date_ns").reset_index(drop=True)
    dates = cal["date_ns"].to_numpy(dtype=np.int64)
    regimes = cal["regime"].astype(str).to_numpy()
    width = int(bank_cfg["window_days"])
    purge = int(bank_cfg["purge_days"])
    min_train = int(bank_cfg["min_train_days"])
    excluded_intervals = [
        (
            int(np.searchsorted(dates, int(row.start_ns), side="left")),
            int(np.searchsorted(dates, int(row.end_ns), side="right")),
        )
        for row in excluded.itertuples(index=False)
    ]
    candidate_starts = []
    for start in range(min_train + purge, len(dates) - width + 1):
        end = start + width
        if any(_overlap(start, end, a, b) for a, b in excluded_intervals):
            continue
        candidate_starts.append(start)
    if len(candidate_starts) < count:
        raise RuntimeError(f"only {len(candidate_starts)} temporal starts are available for {count} windows")

    targets = np.linspace(candidate_starts[0], candidate_starts[-1], int(count))
    chosen: list[int] = []
    for target in targets:
        options = sorted(candidate_starts, key=lambda value: (abs(value - target), value))
        selected = next(
            (value for value in options if not any(_overlap(value, value + width, prior, prior + width) for prior in chosen)),
            None,
        )
        if selected is None:
            raise RuntimeError("unable to construct non-overlapping temporal breadth windows")
        chosen.append(int(selected))
    chosen.sort()
    if len(set(chosen)) != int(count):
        raise RuntimeError("temporal breadth window selection produced duplicates")

    windows = []
    for index, start in enumerate(chosen):
        end = start + width
        cutoff = start - purge - 1
        block = index // 8 + 1
        values, counts = np.unique(regimes[start:end], return_counts=True)
        dominant_at = int(np.argmax(counts))
        dominant = str(values[dominant_at])
        windows.append({
            "bank": "TEMPORAL_BREADTH",
            "temporal_block": f"B{block}",
            "regime": dominant,
            "dominant_regime": dominant,
            "window_id": f"BREADTH_B{block}_{index + 1:02d}",
            "start_date": pd.Timestamp(dates[start]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(dates[end - 1]).strftime("%Y-%m-%d"),
            "start_ns": int(dates[start]),
            "end_ns": int(dates[end - 1]),
            "target_regime_days": int(counts[dominant_at]),
            "total_days": int(width),
            "purity": float(counts[dominant_at] / width),
            "train_cutoff_ns": int(dates[cutoff]),
            "train_cutoff_date": pd.Timestamp(dates[cutoff]).strftime("%Y-%m-%d"),
            "train_dates": int(cutoff + 1),
        })
    return windows


def _config_map(config: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item["params"]) for item in config["search_space"][model]}


def _build_plans(
    output: Path,
    windows: list[dict[str, Any]],
    seed: int,
    meta: dict[str, Any],
    config: dict[str, Any],
    selected: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookups = {model: _config_map(config, model) for model in ("lightgbm", "xgboost")}
    plans: dict[str, list[dict[str, Any]]] = {"lightgbm": [], "xgboost": []}
    for profile in PROFILES:
        for window in windows:
            for model, threads in (("lightgbm", 4), ("xgboost", 2)):
                config_id = selected[model][profile]
                plan = make_bundle_plan(
                    output,
                    "temporal_breadth",
                    model,
                    profile,
                    config_id,
                    lookups[model][config_id],
                    window,
                    [int(seed)],
                    _paths(meta, profile),
                    threads,
                )
                plan["priority"] = "below_normal"
                plans[model].append(plan)
    return plans["lightgbm"], plans["xgboost"]


def _merge_predictions(
    lgb_results: list[dict[str, Any]],
    xgb_results: list[dict[str, Any]],
    meta: dict[str, Any],
    calendar: pd.DataFrame,
    windows: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    lgb = _frames(lgb_results, meta, calendar)
    xgb = _frames(xgb_results, meta, calendar)
    block_map = {str(item["window_id"]): str(item["temporal_block"]) for item in windows}
    result = {}
    for profile in PROFILES:
        left = lgb[lgb["profile"] == profile].copy()
        right = xgb[xgb["profile"] == profile][["row_id", "window_id", "prediction"]].copy()
        merged = left.merge(
            right,
            on=["row_id", "window_id"],
            how="inner",
            suffixes=("_lgb", "_xgb"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(merged) != len(right):
            raise RuntimeError(f"temporal prediction merge mismatch for {profile}: {len(left)}/{len(right)}/{len(merged)}")
        merged["temporal_block"] = merged["window_id"].map(block_map)
        if merged["temporal_block"].isna().any():
            raise RuntimeError(f"missing temporal block mapping for {profile}")
        result[profile] = merged
    return result


def _candidate_frame(
    merged: dict[str, pd.DataFrame],
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    profile: str,
    calibration: str,
    weight: float,
) -> pd.DataFrame:
    frame = merged[profile].copy()
    lgb = calibrators[(profile, "lightgbm")][calibration](frame["prediction_lgb"].to_numpy())
    xgb = calibrators[(profile, "xgboost")][calibration](frame["prediction_xgb"].to_numpy())
    frame["prediction"] = (1.0 - float(weight)) * lgb + float(weight) * xgb
    return frame


def _metric_row(frame: pd.DataFrame, **identity: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    score, regimes = metrics_by_regime(frame)
    row = {
        **identity,
        "selection_score": score["selection_score"],
        **{f"overall_{key}": value for key, value in score["overall"].items()},
        "regime_pr_lift_geometric_mean": score.get("regime_pr_lift_geometric_mean"),
        "worst_regime_pr_lift": score.get("worst_regime_pr_lift"),
        "top3_precision_lift": score.get("top3_precision_lift"),
    }
    for key, value in reversed(list(identity.items())):
        regimes.insert(0, key, value)
    return row, regimes


def _evaluate_grid(
    merged: dict[str, pd.DataFrame],
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    output: Path,
) -> pd.DataFrame:
    rows = []
    regime_parts = []
    for profile in PROFILES:
        for calibration in ("raw", "platt", "isotonic"):
            for weight in np.linspace(0.0, 1.0, 11):
                frame = _candidate_frame(merged, calibrators, profile, calibration, float(weight))
                for scope in ["POOLED", "B1", "B2", "B3", "B4"]:
                    part = frame if scope == "POOLED" else frame[frame["temporal_block"] == scope]
                    row, regimes = _metric_row(
                        part,
                        scope=scope,
                        profile=profile,
                        calibration=calibration,
                        xgb_weight=float(weight),
                    )
                    rows.append(row)
                    regime_parts.append(regimes)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "TEMPORAL_GRID_METRICS.csv", index=False, encoding="utf-8-sig")
    pd.concat(regime_parts, ignore_index=True).to_csv(output / "TEMPORAL_GRID_REGIME_METRICS.csv", index=False, encoding="utf-8-sig")
    best = metrics.sort_values("selection_score", ascending=False).groupby("scope", as_index=False).first()
    best.to_csv(output / "TEMPORAL_BEST_BY_SCOPE.csv", index=False, encoding="utf-8-sig")
    return metrics


def _evaluate_fixed(
    merged: dict[str, pd.DataFrame],
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    output: Path,
) -> dict[str, pd.DataFrame]:
    prepared = {}
    rows = []
    regimes_all = []
    hits_all = []
    for name, profile, calibration, weight in FIXED_CANDIDATES:
        frame = _candidate_frame(merged, calibrators, profile, calibration, weight)
        prepared[name] = frame
        for scope in ["POOLED", "B1", "B2", "B3", "B4"]:
            part = frame if scope == "POOLED" else frame[frame["temporal_block"] == scope]
            row, regimes = _metric_row(part, candidate=name, scope=scope)
            rows.append(row)
            regimes_all.append(regimes)
            hits = horizon_positive_recall(part)
            hits.insert(0, "candidate", name)
            hits.insert(1, "scope", scope)
            hits_all.append(hits)
    pd.DataFrame(rows).to_csv(output / "FIXED_CANDIDATE_METRICS.csv", index=False, encoding="utf-8-sig")
    pd.concat(regimes_all, ignore_index=True).to_csv(output / "FIXED_CANDIDATE_REGIME_METRICS.csv", index=False, encoding="utf-8-sig")
    pd.concat(hits_all, ignore_index=True).to_csv(output / "FIXED_CANDIDATE_HIT_DAY_RECALL.csv", index=False, encoding="utf-8-sig")
    return prepared


def _prequential(
    merged: dict[str, pd.DataFrame],
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        (profile, calibration, float(weight))
        for profile in PROFILES
        for calibration in ("raw", "platt", "isotonic")
        for weight in np.linspace(0.0, 1.0, 11)
    ]
    cache = {
        (profile, calibration, weight): _candidate_frame(merged, calibrators, profile, calibration, weight)
        for profile, calibration, weight in candidates
    }
    selections = []
    outer_rows = []
    outer_parts = []
    for outer_index in (2, 3, 4):
        train_blocks = [f"B{value}" for value in range(1, outer_index)]
        outer_block = f"B{outer_index}"
        scored = []
        for profile, calibration, weight in candidates:
            frame = cache[(profile, calibration, weight)]
            train = frame[frame["temporal_block"].isin(train_blocks)]
            score, _ = metrics_by_regime(train)
            scored.append((float(score["selection_score"]), profile, calibration, weight))
        scored.sort(key=lambda item: item[0], reverse=True)
        train_score, profile, calibration, weight = scored[0]
        outer = cache[(profile, calibration, weight)]
        outer = outer[outer["temporal_block"] == outer_block].copy()
        row, regimes = _metric_row(
            outer,
            outer_block=outer_block,
            profile=profile,
            calibration=calibration,
            xgb_weight=weight,
            train_selection_score=train_score,
        )
        selections.append({
            "outer_block": outer_block,
            "train_blocks": ",".join(train_blocks),
            "profile": profile,
            "calibration": calibration,
            "xgb_weight": weight,
            "train_selection_score": train_score,
        })
        outer_rows.append(row)
        regimes.to_csv(output / f"PREQUENTIAL_REGIMES_{outer_block}.csv", index=False, encoding="utf-8-sig")
        outer_parts.append(outer)
    selection_df = pd.DataFrame(selections)
    outer_df = pd.DataFrame(outer_rows)
    selection_df.to_csv(output / "PREQUENTIAL_SELECTIONS.csv", index=False, encoding="utf-8-sig")
    outer_df.to_csv(output / "PREQUENTIAL_OUTER_METRICS.csv", index=False, encoding="utf-8-sig")
    pooled = pd.concat(outer_parts, ignore_index=True)
    pooled_row, pooled_regimes = _metric_row(pooled, scope="B2_TO_B4_PREQUENTIAL")
    atomic_json(pooled_row, output / "PREQUENTIAL_POOLED_METRICS.json")
    pooled_regimes.to_csv(output / "PREQUENTIAL_POOLED_REGIME_METRICS.csv", index=False, encoding="utf-8-sig")
    return selection_df, pooled


def _bootstrap(prepared: dict[str, pd.DataFrame], output: Path, reps: int) -> pd.DataFrame:
    names = [item[0] for item in FIXED_CANDIDATES]
    dates = np.asarray(sorted(prepared[names[0]]["date"].unique()))
    groups = {name: {date: part for date, part in frame.groupby("date", sort=False)} for name, frame in prepared.items()}
    rng = np.random.default_rng(20260808)
    rows = []
    for rep in range(int(reps)):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        scores = {}
        for name in names:
            pieces = []
            for occurrence, date in enumerate(sampled):
                part = groups[name][date].copy()
                part["date"] = (pd.Timestamp("2000-01-01") + pd.Timedelta(days=occurrence)).strftime("%Y-%m-%d")
                pieces.append(part)
            sample = pd.concat(pieces, ignore_index=True)
            score, _ = metrics_by_regime(sample)
            scores[name] = float(score["selection_score"])
        rows.append({
            "rep": rep,
            **{f"score_{key}": value for key, value in scores.items()},
            "p7_platt04_minus_locked": scores["P7_PLATT_W04"] - scores["LOCKED_P2_RAW_W09"],
            "p7_lgb_minus_p2_lgb": scores["P7_LGB_RAW"] - scores["P2_LGB_RAW"],
            "p7_platt02_minus_locked": scores["P7_PLATT_W02"] - scores["LOCKED_P2_RAW_W09"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "TEMPORAL_DATE_BOOTSTRAP_REPLICATES.csv", index=False, encoding="utf-8-sig")
    summary = []
    for column in ("p7_platt04_minus_locked", "p7_lgb_minus_p2_lgb", "p7_platt02_minus_locked"):
        values = frame[column].to_numpy(dtype=float)
        summary.append({
            "comparison": column,
            "reps": len(values),
            "mean_delta": float(values.mean()),
            "positive_rate": float(np.mean(values > 0)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output / "TEMPORAL_DATE_BOOTSTRAP_SUMMARY.csv", index=False, encoding="utf-8-sig")
    return summary_df


def _export(package: Path, output: Path, status: dict[str, Any]) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_RegimeSeal_TemporalBreadth_RESULTS_{stamp}.zip"
    start = "\n".join([
        "# CrashWatch RegimeSeal Temporal Breadth",
        "",
        f"- status: {status.get('status')}",
        f"- model fits: {status.get('actual_model_fits')}/{status.get('expected_model_fits')}",
        "- purpose: temporal generalization, profile/blend/calibration stability",
        "- sealed data used: false",
        "",
        "Read BOTTLENECK_AND_DECISION_REPORT.json, PREQUENTIAL_SELECTIONS.csv,",
        "TEMPORAL_DATE_BOOTSTRAP_SUMMARY.csv, and FIXED_CANDIDATE_REGIME_METRICS.csv first.",
    ])
    files = [path for path in sorted(output.iterdir()) if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".log"}]
    code = [
        package / "NEXT_TEMPORAL_BREADTH_DESIGN_KO.md",
        package / "run_regime_temporal_breadth.py",
        package / "config_regime_3d5_full_load.json",
        package / "cwregime" / "scoring.py",
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start.encode("utf-8"))
        for path in files:
            if path.name not in {"STOP_TEMPORAL_BREADTH", "SUPERVISOR_FAILED.json"}:
                zipped.write(path, f"results/{path.name}")
        for path in code:
            if path.exists():
                zipped.write(path, f"code_context/{path.relative_to(package).as_posix()}")
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        names = set(zipped.namelist())
    required = {
        "00_START_HERE.md",
        "results/FINAL_TEMPORAL_BREADTH_STATUS.json",
        "results/BOTTLENECK_AND_DECISION_REPORT.json",
    }
    if bad is not None or not required.issubset(names):
        raise RuntimeError(f"temporal breadth archive invalid: bad={bad}, missing={sorted(required - names)}")
    digest = _sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256.txt")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {"archive": str(archive), "size_bytes": archive.stat().st_size, "sha256": digest, "entries": len(names), "verified": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    selection = Path(args.selection_output).expanduser().resolve()
    prior = Path(args.prior_followup).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "temporal_breadth.lock.json"
    acquire_lock(lock)
    monitor_stop = threading.Event()
    monitor: threading.Thread | None = None
    started = time.time()
    try:
        budget_path = output / "TEMPORAL_BUDGET.json"
        budget = read_json(budget_path, {})
        if not budget:
            budget = {"started_epoch": started, "deadline_epoch": started + float(args.budget_seconds), "budget_seconds": float(args.budget_seconds)}
            atomic_json(budget, budget_path)
        deadline = float(budget["deadline_epoch"])
        hardware = _set_game_mode(int(args.cpu_threads))
        hardware.update({"gpu_policy": "single XGBoost seed burst followed by cooldown", "gpu_cooldown_multiplier": float(args.gpu_sleep_multiplier)})
        atomic_json(hardware, output / "TEMPORAL_GAME_MODE_HARDWARE.json")
        monitor = threading.Thread(target=_resource_monitor, args=(monitor_stop, output / "temporal_resource_usage.csv"), daemon=True)
        monitor.start()

        if (selection / "final_sealed" / "FINAL_SEALED_CONSUMED.json").exists():
            raise RuntimeError("temporal breadth refused after FINAL_SEALED was consumed")
        prior_final = read_json(prior / "FINAL_GAME_FOLLOWUP_STATUS.json", {})
        if prior_final.get("status") != "completed" or int(prior_final.get("actual_model_fits", 0)) != 192:
            raise RuntimeError("completed game follow-up evidence is required")
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        meta = read_json(selection / "RUNTIME_MANIFEST.json", {})
        calendar = pd.read_csv(selection / "regime_calendar.csv")
        primary = pd.read_csv(selection / "regime_windows_confirm.csv")
        alternate = pd.read_csv(prior / "ALTERNATE_CONFIRM_WINDOWS.csv")
        excluded = pd.concat([primary, alternate], ignore_index=True)
        windows = _select_temporal_windows(calendar, excluded, config["banks"], int(args.window_count))
        window_df = pd.DataFrame(windows)
        window_df.to_csv(output / "TEMPORAL_BREADTH_WINDOWS.csv", index=False, encoding="utf-8-sig")
        selected = _selected_configs(selection)
        seed = int(args.seed)
        lgb_plans, xgb_plans = _build_plans(output, windows, seed, meta, config, selected)
        atomic_json({
            "status": "planned",
            "prior_evidence_reused": str(prior),
            "candidate_configs_narrowed_from_prior": selected,
            "window_count": len(windows),
            "blocks": window_df.groupby("temporal_block").size().to_dict(),
            "seed": seed,
            "lgb_bundles": len(lgb_plans),
            "xgb_bundles": len(xgb_plans),
            "expected_model_fits": len(lgb_plans) + len(xgb_plans),
            "sealed_data_used": False,
        }, output / "TEMPORAL_EXPERIMENT_PLAN.json")
        atomic_json({
            "status": "bottleneck_confirmed",
            "base_to_alt_instability": {
                "p2_base_robust_xgb_weight": 0.6,
                "p2_alt_robust_xgb_weight": 0.2,
                "p7_base_robust": "platt/xgb_weight_0.4",
                "p7_alt_robust": "platt/xgb_weight_0.2",
                "alt_p2_minus_p7_bootstrap_mean": -0.041749,
                "alt_p2_beats_p7_rate": 0.066667,
            },
            "regime_bottlenecks": {
                "base_bull_high_vol_roc_auc": 0.558320,
                "alt_rebound_roc_auc": 0.504293,
                "alt_crash_stress_roc_auc": 0.518812,
            },
            "next_question": "which profile/calibration/blend survives broad blocked time validation",
        }, output / "PRIOR_BOTTLENECK_REPORT.json")

        lgb_results = _run_lgb(lgb_plans)
        atomic_json({"status": "completed", "count": len(lgb_results)}, output / "TEMPORAL_LGB_STAGE_STATUS.json")
        xgb_results = _run_throttled_xgb(
            xgb_plans,
            deadline_epoch=deadline,
            sleep_multiplier=float(args.gpu_sleep_multiplier),
            stop_marker=output / "STOP_TEMPORAL_BREADTH",
            progress_path=output / "TEMPORAL_XGB_STAGE_STATUS.json",
        )

        calibrators = {}
        calibration_audit = {}
        for profile in PROFILES:
            for model in ("lightgbm", "xgboost"):
                search = _search_predictions(selection, model, profile, selected[model][profile], meta, calendar)
                funcs, audit = _fit_calibrators(search)
                calibrators[(profile, model)] = funcs
                calibration_audit[f"{profile}/{model}"] = audit
        atomic_json({"fit_scope": "SEARCH only", "models": calibration_audit}, output / "TEMPORAL_CALIBRATION_AUDIT.json")

        merged = _merge_predictions(lgb_results, xgb_results, meta, calendar, windows)
        _evaluate_grid(merged, calibrators, output)
        fixed = _evaluate_fixed(merged, calibrators, output)
        selections, prequential_frame = _prequential(merged, calibrators, output)
        bootstrap = _bootstrap(fixed, output, int(args.bootstrap_reps))

        fixed_metrics = pd.read_csv(output / "FIXED_CANDIDATE_METRICS.csv")
        pooled = fixed_metrics[fixed_metrics["scope"] == "POOLED"].set_index("candidate")
        regime_metrics = pd.read_csv(output / "FIXED_CANDIDATE_REGIME_METRICS.csv")
        p7_regimes = regime_metrics[(regime_metrics["candidate"] == "P7_PLATT_W04") & (regime_metrics["scope"] == "POOLED")]
        weak = p7_regimes[p7_regimes["raw_roc_auc"] <= 0.55][["regime", "raw_roc_auc", "raw_pr_auc_lift", "unique_dates"]].to_dict("records")
        report = {
            "status": "completed",
            "headline": "temporal breadth validation completed without sealed data",
            "locked_score": float(pooled.loc["LOCKED_P2_RAW_W09", "selection_score"]),
            "p7_platt04_score": float(pooled.loc["P7_PLATT_W04", "selection_score"]),
            "p7_lgb_score": float(pooled.loc["P7_LGB_RAW", "selection_score"]),
            "bootstrap": bootstrap.to_dict("records"),
            "prequential_choices": selections.to_dict("records"),
            "weak_regimes_roc_auc_le_055": weak,
            "decision_rules": {
                "promote_p7_if_bootstrap_positive_rate_ge": 0.8,
                "stable_if_same_profile_selected_in_at_least_outer_blocks": 2,
                "separate_feature_work_if_regime_roc_auc_le": 0.55,
            },
            "sealed_data_used": False,
            "original_locked_recipe_overwritten": False,
        }
        atomic_json(report, output / "BOTTLENECK_AND_DECISION_REPORT.json")
        expected = len(lgb_plans) + len(xgb_plans)
        actual = len(lgb_results) + len(xgb_results)
        final = {
            "status": "completed" if actual == expected else "incomplete",
            "started_epoch": float(budget["started_epoch"]),
            "completed_epoch": time.time(),
            "elapsed_seconds_this_attempt": time.time() - started,
            "budget_seconds": float(budget["budget_seconds"]),
            "expected_model_fits": expected,
            "actual_model_fits": actual,
            "window_count": len(windows),
            "seed": seed,
            "sealed_data_used": False,
            "original_locked_recipe_overwritten": False,
        }
        atomic_json(final, output / "FINAL_TEMPORAL_BREADTH_STATUS.json")
        if final["status"] != "completed":
            raise RuntimeError(f"temporal breadth completion audit failed: {final}")
        export = _export(package, output, final)
        final["result_export"] = export
        atomic_json(final, output / "FINAL_TEMPORAL_BREADTH_STATUS.json")
        return final
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        release_lock(lock)


def main() -> int:
    parser = argparse.ArgumentParser(description="CrashWatch temporal breadth game-mode validation")
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--prior-followup", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--gpu-sleep-multiplier", type=float, default=3.0)
    parser.add_argument("--budget-seconds", type=float, default=18000.0)
    parser.add_argument("--window-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    raise SystemExit(main())
