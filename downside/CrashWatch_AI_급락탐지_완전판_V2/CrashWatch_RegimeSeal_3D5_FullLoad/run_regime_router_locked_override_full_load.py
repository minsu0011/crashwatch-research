from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock, set_full_load_mode
from cwregime.expert_router import route_prediction_columns
from cwregime.gating import evaluate_prediction
from cwregime.router_followup import TrainOnlyCalibrator, fit_conservative_pair_route
from run_regime_router_followup_full_load import _bootstrap
from run_regime_submodel_gate import _load_oof


BASE = "P2_LOCKED"
CHALLENGER = "P7_LGB"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(temp_name, index=False, encoding="utf-8-sig")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def add_locked(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[BASE] = 0.1 * out["P2_LGB"].to_numpy(dtype=float) + 0.9 * out["P2_XGB"].to_numpy(dtype=float)
    return out


def route_grid() -> list[dict[str, Any]]:
    candidates = [{
        "route_id": "NO_P7_OVERRIDE",
        "disabled": True,
        "shrink_rows": 0,
        "p7_margin": 999.0,
        "min_dates": 8,
        "min_positives": 10,
    }]
    for shrink_rows in (0, 500, 2000):
        for p7_margin in (0.0, 0.02, 0.05, 0.10, 0.15):
            for min_dates in (5, 8):
                item = {
                    "disabled": False,
                    "shrink_rows": shrink_rows,
                    "p7_margin": p7_margin,
                    "min_dates": min_dates,
                    "min_positives": 10,
                }
                payload = json.dumps(item, sort_keys=True, separators=(",", ":"))
                item["route_id"] = hashlib.sha256(payload.encode()).hexdigest()[:14]
                candidates.append(item)
    return candidates


def fit_route(frame: pd.DataFrame, candidate: dict[str, Any]):
    if candidate["disabled"]:
        route = fit_conservative_pair_route(
            frame,
            shrink_rows=0,
            p7_margin=999.0,
            min_dates=8,
            min_positives=10,
            base_expert=BASE,
            challenger_expert=CHALLENGER,
        )
    else:
        route = fit_conservative_pair_route(
            frame,
            shrink_rows=int(candidate["shrink_rows"]),
            p7_margin=float(candidate["p7_margin"]),
            min_dates=int(candidate["min_dates"]),
            min_positives=int(candidate["min_positives"]),
            base_expert=BASE,
            challenger_expert=CHALLENGER,
        )
    return route


def metric(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    row, _ = evaluate_prediction(frame, name)
    return row


def search_routes(wide: pd.DataFrame, output: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    route_tables = []
    for outer_number in (2, 3):
        block = f"B{outer_number}"
        train = wide[wide.temporal_block.isin([f"B{i}" for i in range(1, outer_number)])]
        validation = wide[wide.temporal_block == block]
        baseline = validation.copy(); baseline["prediction"] = baseline[BASE]
        base_metric = metric(baseline, "LOCKED_P2")
        for candidate in route_grid():
            route = fit_route(train, candidate)
            routed = route_prediction_columns(validation, route.mapping, fallback_model=BASE)
            row = metric(routed, candidate["route_id"])
            row.update(candidate)
            row.update({
                "outer_block": block,
                "mapping": json.dumps(route.mapping, ensure_ascii=False, sort_keys=True),
                "p7_regime_count": int(sum(value == CHALLENGER for value in route.mapping.values())),
                "score_ratio_vs_locked": float(row["selection_score"] / base_metric["selection_score"]),
                "pr_auc_delta_vs_locked": float(row["overall_raw_pr_auc"] - base_metric["overall_raw_pr_auc"]),
                "brier_delta_vs_locked": float(row["overall_raw_brier"] - base_metric["overall_raw_brier"]),
            })
            rows.append(row)
            table = route.table.copy()
            table.insert(0, "route_id", candidate["route_id"])
            table.insert(1, "outer_block", block)
            route_tables.append(table)
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "LOCKED_OVERRIDE_ROUTE_METRICS.csv")
    atomic_csv(pd.concat(route_tables, ignore_index=True), output / "LOCKED_OVERRIDE_ROUTE_TABLES.csv")
    summaries = []
    for route_id, part in metrics.groupby("route_id", sort=False):
        first = part.iloc[0]
        scores = np.clip(part.selection_score.to_numpy(dtype=float), 1e-9, None)
        maps = [json.loads(value) for value in part.sort_values("outer_block").mapping]
        churn = sum(maps[0].get(regime) != maps[1].get(regime) for regime in maps[0])
        summaries.append({
            "route_id": route_id,
            "disabled": bool(first.disabled),
            "shrink_rows": int(first.shrink_rows),
            "p7_margin": float(first.p7_margin),
            "min_dates": int(first.min_dates),
            "min_positives": int(first.min_positives),
            "B2_score": float(part.loc[part.outer_block == "B2", "selection_score"].iloc[0]),
            "B3_score": float(part.loc[part.outer_block == "B3", "selection_score"].iloc[0]),
            "gmean_score": float(np.exp(np.mean(np.log(scores)))),
            "min_score_ratio_vs_locked": float(part.score_ratio_vs_locked.min()),
            "mean_score_ratio_vs_locked": float(part.score_ratio_vs_locked.mean()),
            "mean_pr_auc_delta_vs_locked": float(part.pr_auc_delta_vs_locked.mean()),
            "mean_brier_delta_vs_locked": float(part.brier_delta_vs_locked.mean()),
            "mean_p7_regime_count": float(part.p7_regime_count.mean()),
            "route_churn": int(churn),
        })
    summary = pd.DataFrame(summaries)
    summary["eligible"] = summary.min_score_ratio_vs_locked >= 1.0
    eligible = summary[summary.eligible].copy()
    eligible = eligible.sort_values(
        ["min_score_ratio_vs_locked", "gmean_score", "route_churn", "mean_p7_regime_count"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    winner = eligible.iloc[0] if not eligible.empty else summary.loc[summary.min_score_ratio_vs_locked.idxmax()]
    summary["selected"] = summary.route_id.eq(winner.route_id)
    summary = summary.sort_values(["selected", "eligible", "min_score_ratio_vs_locked"], ascending=[False, False, False])
    atomic_csv(summary, output / "LOCKED_OVERRIDE_ROUTE_SELECTION.csv")
    selected = winner.to_dict()
    selected.update({
        "selection_blocks": ["B2", "B3"],
        "B4_used_for_selection": False,
        "selection_rule": "require no B2/B3 composite-score loss versus locked P2, then maximize worst ratio",
    })
    atomic_json(selected, output / "LOCKED_OVERRIDE_SELECTION_BEFORE_B4.json")
    return selected, metrics


def calibration_grid() -> list[dict[str, Any]]:
    out = [{"calibration_id": "NONE", "method": "none", "c_value": 1.0, "blend": 0.0}]
    for c_value in (0.1, 1.0, 10.0):
        for blend in (0.25, 0.50, 0.75, 1.0):
            item = {"method": "platt_global", "c_value": c_value, "blend": blend}
            payload = json.dumps(item, sort_keys=True, separators=(",", ":"))
            item["calibration_id"] = hashlib.sha256(payload.encode()).hexdigest()[:14]
            out.append(item)
    return out


def apply_risk_calibration(
    train_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, TrainOnlyCalibrator]:
    calibrator = TrainOnlyCalibrator.fit(
        train_raw.prediction.to_numpy(dtype=float),
        train_raw.target.to_numpy(dtype=np.uint8),
        train_raw.regime,
        method=str(candidate["method"]),
        c_value=float(candidate["c_value"]),
    )
    calibrated = calibrator.predict(validation_raw.prediction.to_numpy(dtype=float), validation_raw.regime)
    blend = float(candidate["blend"])
    risk = (1.0 - blend) * validation_raw.prediction.to_numpy(dtype=float) + blend * calibrated
    return np.clip(risk, 1e-7, 1.0 - 1e-7), calibrator


def search_calibration(wide: pd.DataFrame, route_selected: dict[str, Any], output: Path) -> dict[str, Any]:
    route_candidate = {key: route_selected[key] for key in ("route_id", "disabled", "shrink_rows", "p7_margin", "min_dates", "min_positives")}
    rows = []
    for outer_number in (2, 3):
        block = f"B{outer_number}"
        train = wide[wide.temporal_block.isin([f"B{i}" for i in range(1, outer_number)])]
        validation = wide[wide.temporal_block == block]
        route = fit_route(train, route_candidate)
        raw_train = route_prediction_columns(train, route.mapping, fallback_model=BASE)
        raw_validation = route_prediction_columns(validation, route.mapping, fallback_model=BASE)
        baseline = validation.copy(); baseline["prediction"] = baseline[BASE]
        base_metric = metric(baseline, "LOCKED_P2")
        for candidate in calibration_grid():
            risk, _ = apply_risk_calibration(raw_train, raw_validation, candidate)
            evaluated = raw_validation.copy(); evaluated["prediction"] = risk
            row = metric(evaluated, candidate["calibration_id"])
            row.update(candidate)
            row.update({
                "outer_block": block,
                "brier_ratio_vs_locked": float(row["overall_raw_brier"] / base_metric["overall_raw_brier"]),
                "logloss_ratio_vs_locked": float(row["overall_raw_logloss"] / base_metric["overall_raw_logloss"]),
            })
            rows.append(row)
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "LOCKED_OVERRIDE_CALIBRATION_METRICS.csv")
    summaries = []
    for cid, part in metrics.groupby("calibration_id", sort=False):
        first = part.iloc[0]
        summaries.append({
            "calibration_id": cid,
            "method": first.method,
            "c_value": float(first.c_value),
            "blend": float(first.blend),
            "mean_brier": float(part.overall_raw_brier.mean()),
            "max_brier_ratio_vs_locked": float(part.brier_ratio_vs_locked.max()),
            "mean_logloss": float(part.overall_raw_logloss.mean()),
            "max_logloss_ratio_vs_locked": float(part.logloss_ratio_vs_locked.max()),
        })
    summary = pd.DataFrame(summaries)
    summary["eligible"] = (summary.max_brier_ratio_vs_locked <= 1.01) & (summary.max_logloss_ratio_vs_locked <= 1.01)
    eligible = summary[summary.eligible].copy()
    if eligible.empty:
        eligible = summary.copy()
    eligible = eligible.sort_values(
        ["mean_brier", "max_brier_ratio_vs_locked", "mean_logloss", "blend"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    winner = eligible.iloc[0].to_dict()
    summary["selected"] = summary.calibration_id.eq(winner["calibration_id"])
    summary = summary.sort_values(["selected", "eligible", "mean_brier"], ascending=[False, False, True])
    atomic_csv(summary, output / "LOCKED_OVERRIDE_CALIBRATION_SELECTION.csv")
    winner.update({
        "selection_blocks": ["B2", "B3"],
        "B4_used_for_selection": False,
        "policy": "ranking_score stays raw; calibrated risk probability is a separate output",
    })
    atomic_json(winner, output / "LOCKED_OVERRIDE_CALIBRATION_BEFORE_B4.json")
    return winner


def b4_and_freeze(
    wide: pd.DataFrame,
    route_selected: dict[str, Any],
    calibration_selected: dict[str, Any],
    output: Path,
    bootstrap_reps: int,
    bootstrap_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    route_candidate = {key: route_selected[key] for key in ("route_id", "disabled", "shrink_rows", "p7_margin", "min_dates", "min_positives")}
    calibration_candidate = {key: calibration_selected[key] for key in ("calibration_id", "method", "c_value", "blend")}
    train = wide[wide.temporal_block.isin(["B1", "B2", "B3"])]
    b4 = wide[wide.temporal_block == "B4"]
    route = fit_route(train, route_candidate)
    train_raw = route_prediction_columns(train, route.mapping, fallback_model=BASE)
    b4_raw = route_prediction_columns(b4, route.mapping, fallback_model=BASE)
    risk, calibrator = apply_risk_calibration(train_raw, b4_raw, calibration_candidate)
    followup = b4_raw.copy(); followup["prediction"] = risk
    locked = b4.copy(); locked["prediction"] = locked[BASE]
    follow_metric = metric(followup, "LOCKED_P7_OVERRIDE_DUAL_HEAD")
    locked_metric = metric(locked, "LOCKED_P2")
    b4_metrics = pd.DataFrame([follow_metric, locked_metric])
    atomic_csv(b4_metrics, output / "LOCKED_OVERRIDE_B4_DIAGNOSTIC_METRICS.csv")
    result = {
        "status": "COMPLETE_REUSED_B4_DIAGNOSTIC",
        "fresh_sealed_claim": False,
        "route_mapping": route.mapping,
        "p7_regime_count": int(sum(value == CHALLENGER for value in route.mapping.values())),
        "calibration": {**calibrator.to_artifact(), "blend": float(calibration_candidate["blend"])},
        "selection_score_delta": float(follow_metric["selection_score"] - locked_metric["selection_score"]),
        "pr_auc_delta": float(follow_metric["overall_raw_pr_auc"] - locked_metric["overall_raw_pr_auc"]),
        "roc_auc_delta": float(follow_metric["overall_raw_roc_auc"] - locked_metric["overall_raw_roc_auc"]),
        "brier_improvement": float(locked_metric["overall_raw_brier"] - follow_metric["overall_raw_brier"]),
        "logloss_improvement": float(locked_metric["overall_raw_logloss"] - follow_metric["overall_raw_logloss"]),
    }
    atomic_json(result, output / "LOCKED_OVERRIDE_B4_DIAGNOSTIC.json")
    boot_frame = b4[["date", "ticker", "target", "regime"]].copy()
    boot_frame["followup"] = risk
    boot_frame["locked"] = locked["prediction"].to_numpy(dtype=float)
    bootstrap = _bootstrap(boot_frame, output, reps=bootstrap_reps, workers=bootstrap_workers)

    final_route = fit_route(wide, route_candidate)
    all_raw = route_prediction_columns(wide, final_route.mapping, fallback_model=BASE)
    final_cal = TrainOnlyCalibrator.fit(
        all_raw.prediction.to_numpy(dtype=float), all_raw.target.to_numpy(dtype=np.uint8), all_raw.regime,
        method=str(calibration_candidate["method"]), c_value=float(calibration_candidate["c_value"]),
    )
    policy = {
        "schema": "crashwatch_locked_p2_p7_override_dual_head_v1",
        "base_expert": {"name": BASE, "formula": "0.1*P2_LGB + 0.9*P2_XGB"},
        "challenger_expert": CHALLENGER,
        "regime_mapping": final_route.mapping,
        "route_candidate": route_candidate,
        "risk_calibration": {**final_cal.to_artifact(), "blend": float(calibration_candidate["blend"])},
        "ranking_output": "raw routed score",
        "risk_output": "blend of raw routed score and train-only global Platt probability",
        "selection_source": "B2/B3 prequential only",
        "B4_fresh_sealed_claim": False,
        "external_sealed_used": False,
        "frozen": True,
    }
    payload = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    policy["policy_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    atomic_json(policy, output / "LOCKED_OVERRIDE_FROZEN_POLICY.json")
    atomic_csv(final_route.table, output / "LOCKED_OVERRIDE_FINAL_ROUTE_TABLE.csv")
    return result, {"bootstrap": bootstrap, "policy": policy}


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection = Path(args.selection_output).expanduser().resolve()
    temporal = Path(args.temporal_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "locked_override.lock.json"
    acquire_lock(lock)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        hardware["gpu"] = nvml_snapshot()
        atomic_json(hardware, output / "HARDWARE_FULL_LOAD.json")
        wide, _ = _load_oof(selection, temporal)
        wide = add_locked(wide)
        route_selected, _ = search_routes(wide, output)
        calibration_selected = search_calibration(wide, route_selected, output)
        b4, final = b4_and_freeze(
            wide, route_selected, calibration_selected, output,
            int(args.bootstrap_reps), int(args.bootstrap_workers),
        )
        result = {
            "status": "LOCKED_P2_P7_OVERRIDE_EXPERIMENT_COMPLETE",
            "elapsed_seconds": float(time.time() - started),
            "route_candidates": len(route_grid()),
            "calibration_candidates": len(calibration_grid()),
            "selected_route": route_selected,
            "selected_calibration": calibration_selected,
            "B4_diagnostic": b4,
            "bootstrap_reps": final["bootstrap"]["reps"],
            "frozen_policy_hash": final["policy"]["policy_hash"],
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
        }
        atomic_json(result, output / "RUN_STATUS.json")
        return result
    except Exception as exc:
        failed = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started}
        atomic_json(failed, output / "RUN_STATUS.json")
        raise
    finally:
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Locked P2 base with conservative P7 regime override")
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_locked_p7_override_v1"))
    parser.add_argument("--total-threads", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-workers", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
