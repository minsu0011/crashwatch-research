from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock, set_full_load_mode, set_worker_mode
from cwregime.expert_router import route_prediction_columns
from cwregime.gating import apply_gate, evaluate_prediction, fit_gate
from cwregime.router_followup import (
    candidate_id,
    fit_and_apply_candidate,
    fit_conservative_pair_route,
    TrainOnlyCalibrator,
)
from run_regime_submodel_gate import _load_oof


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


def candidate_grid() -> list[dict[str, Any]]:
    calibrations = [
        ("none", 1.0),
        ("platt_global", 0.1),
        ("platt_global", 1.0),
        ("platt_global", 10.0),
        ("platt_regime", 0.1),
        ("platt_regime", 1.0),
        ("platt_regime", 10.0),
    ]
    candidates = []
    for shrink_rows in (0, 500, 2000):
        for p7_margin in (0.0, 0.02, 0.05, 0.10):
            for method, c_value in calibrations:
                item = {
                    "shrink_rows": shrink_rows,
                    "p7_margin": p7_margin,
                    "min_dates": 8,
                    "min_positives": 10,
                    "calibration_method": method,
                    "calibration_c": c_value,
                }
                item["candidate_id"] = candidate_id(item)
                candidates.append(item)
    return candidates


def _metric_row(frame: pd.DataFrame, candidate: str, block: str) -> tuple[dict[str, Any], pd.DataFrame]:
    row, regimes = evaluate_prediction(frame, candidate)
    row["outer_block"] = block
    regimes.insert(0, "outer_block", block)
    return row, regimes


def _prequential_search(wide: pd.DataFrame, output: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    regime_parts: list[pd.DataFrame] = []
    route_rows: list[pd.DataFrame] = []
    locked_by_block: dict[str, dict[str, float]] = {}
    candidates = candidate_grid()
    for outer_number in (2, 3):
        block = f"B{outer_number}"
        train_blocks = [f"B{number}" for number in range(1, outer_number)]
        train = wide[wide["temporal_block"].isin(train_blocks)]
        validation = wide[wide["temporal_block"] == block]
        locked = validation.copy()
        locked["prediction"] = 0.1 * locked["P2_LGB"] + 0.9 * locked["P2_XGB"]
        locked_row, _ = _metric_row(locked, "LOCKED_P2", block)
        locked_by_block[block] = {
            "selection_score": float(locked_row["selection_score"]),
            "brier": float(locked_row["overall_raw_brier"]),
            "logloss": float(locked_row["overall_raw_logloss"]),
        }
        for candidate in candidates:
            evaluated, route, _ = fit_and_apply_candidate(train, validation, **{
                key: candidate[key]
                for key in (
                    "shrink_rows", "p7_margin", "calibration_method", "calibration_c",
                    "min_dates", "min_positives",
                )
            })
            row, regimes = _metric_row(evaluated, candidate["candidate_id"], block)
            row.update(candidate)
            row["train_blocks"] = ",".join(train_blocks)
            row["mapping"] = json.dumps(route.mapping, sort_keys=True, ensure_ascii=False)
            row["p7_regime_count"] = int(sum(value == "P7_LGB" for value in route.mapping.values()))
            row["score_ratio_vs_locked"] = float(row["selection_score"] / locked_row["selection_score"])
            row["brier_ratio_vs_locked"] = float(row["overall_raw_brier"] / locked_row["overall_raw_brier"])
            rows.append(row)
            regimes.insert(0, "candidate_id", candidate["candidate_id"])
            regime_parts.append(regimes)
            table = route.table.copy()
            table.insert(0, "candidate_id", candidate["candidate_id"])
            table.insert(1, "outer_block", block)
            route_rows.append(table)

    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "FOLLOWUP_CANDIDATE_METRICS.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "FOLLOWUP_CANDIDATE_BY_REGIME.csv")
    atomic_csv(pd.concat(route_rows, ignore_index=True), output / "FOLLOWUP_CANDIDATE_ROUTE_TABLES.csv")

    summaries: list[dict[str, Any]] = []
    for cid, part in metrics.groupby("candidate_id", sort=False):
        first = part.iloc[0]
        scores = np.clip(part["selection_score"].to_numpy(dtype=float), 1e-9, None)
        mappings = [json.loads(value) for value in part.sort_values("outer_block")["mapping"]]
        churn = sum(mappings[0].get(regime) != mappings[1].get(regime) for regime in mappings[0])
        summaries.append({
            "candidate_id": cid,
            "shrink_rows": int(first["shrink_rows"]),
            "p7_margin": float(first["p7_margin"]),
            "min_dates": int(first["min_dates"]),
            "min_positives": int(first["min_positives"]),
            "calibration_method": str(first["calibration_method"]),
            "calibration_c": float(first["calibration_c"]),
            "B2_selection_score": float(part.loc[part.outer_block == "B2", "selection_score"].iloc[0]),
            "B3_selection_score": float(part.loc[part.outer_block == "B3", "selection_score"].iloc[0]),
            "geometric_mean_selection_score": float(np.exp(np.mean(np.log(scores)))),
            "min_score_ratio_vs_locked": float(part["score_ratio_vs_locked"].min()),
            "mean_score_ratio_vs_locked": float(part["score_ratio_vs_locked"].mean()),
            "mean_brier": float(part["overall_raw_brier"].mean()),
            "max_brier_ratio_vs_locked": float(part["brier_ratio_vs_locked"].max()),
            "mean_logloss": float(part["overall_raw_logloss"].mean()),
            "mean_p7_regime_count": float(part["p7_regime_count"].mean()),
            "route_churn_B2_to_B3": int(churn),
        })
    summary = pd.DataFrame(summaries)
    summary["passes_discrimination_guard"] = summary["min_score_ratio_vs_locked"] >= 0.98
    summary["passes_calibration_guard"] = summary["max_brier_ratio_vs_locked"] <= 1.0
    summary["eligible"] = summary["passes_discrimination_guard"] & summary["passes_calibration_guard"]
    eligible = summary[summary["eligible"]].copy()
    selection_fallback = False
    if eligible.empty:
        selection_fallback = True
        eligible = summary[summary["passes_discrimination_guard"]].copy()
    if eligible.empty:
        selection_fallback = True
        eligible = summary.copy()
    eligible = eligible.sort_values(
        [
            "min_score_ratio_vs_locked", "geometric_mean_selection_score", "mean_brier",
            "route_churn_B2_to_B3", "mean_p7_regime_count",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    winner = eligible.iloc[0].to_dict()
    summary["selected"] = summary["candidate_id"].eq(winner["candidate_id"])
    summary = summary.sort_values(
        ["selected", "eligible", "min_score_ratio_vs_locked", "mean_brier"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    atomic_csv(summary, output / "FOLLOWUP_CANDIDATE_SELECTION.csv")
    selected = {
        key: winner[key]
        for key in (
            "candidate_id", "shrink_rows", "p7_margin", "min_dates", "min_positives",
            "calibration_method", "calibration_c", "B2_selection_score", "B3_selection_score",
            "geometric_mean_selection_score", "min_score_ratio_vs_locked", "mean_brier",
            "max_brier_ratio_vs_locked", "route_churn_B2_to_B3",
        )
    }
    selected.update({
        "selection_blocks": ["B2", "B3"],
        "B4_used_for_selection": False,
        "selection_rule": (
            "require every B2/B3 score >=98% of locked P2 and every Brier <= locked P2; "
            "maximize worst score ratio, then geometric score, then minimize Brier/churn/complexity"
        ),
        "guard_fallback_used": selection_fallback,
        "locked_by_block": locked_by_block,
    })
    atomic_json(selected, output / "FOLLOWUP_SELECTION_BEFORE_B4.json")
    return selected, metrics


def _candidate_kwargs(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "shrink_rows": int(selected["shrink_rows"]),
        "p7_margin": float(selected["p7_margin"]),
        "calibration_method": str(selected["calibration_method"]),
        "calibration_c": float(selected["calibration_c"]),
        "min_dates": int(selected["min_dates"]),
        "min_positives": int(selected["min_positives"]),
    }


def _b4_diagnostic(
    wide: pd.DataFrame,
    selected: dict[str, Any],
    gate_selection: dict[str, Any],
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = wide[wide["temporal_block"].isin(["B1", "B2", "B3"])]
    b4 = wide[wide["temporal_block"] == "B4"].copy()
    followup, route, calibrator = fit_and_apply_candidate(train, b4, **_candidate_kwargs(selected))
    raw_followup = route_prediction_columns(b4, route.mapping)
    locked = b4.copy(); locked["prediction"] = 0.1 * locked["P2_LGB"] + 0.9 * locked["P2_XGB"]
    p2 = b4.copy(); p2["prediction"] = p2["P2_XGB"]
    p7 = b4.copy(); p7["prediction"] = p7["P7_LGB"]
    current_gate = fit_gate(
        train,
        method=str(gate_selection["method"]),
        shrink_rows=int(gate_selection["shrink_rows"]),
        n_clusters=int(gate_selection["n_clusters"]),
    )
    current = apply_gate(b4, current_gate)
    prepared = {
        "FOLLOWUP_CALIBRATED": followup,
        "FOLLOWUP_RAW": raw_followup,
        "CURRENT_FOUR_EXPERT_GATE": current,
        "LOCKED_P2": locked,
        "P2_XGB": p2,
        "P7_LGB": p7,
    }
    rows: list[dict[str, Any]] = []
    regime_parts: list[pd.DataFrame] = []
    for name, frame in prepared.items():
        row, regimes = evaluate_prediction(frame, name)
        row["validation_status"] = "B4_REUSED_DIAGNOSTIC_NOT_FRESH_SEALED"
        rows.append(row)
        regimes.insert(0, "validation_status", "B4_REUSED_DIAGNOSTIC_NOT_FRESH_SEALED")
        regime_parts.append(regimes)
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "FOLLOWUP_B4_DIAGNOSTIC_METRICS.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "FOLLOWUP_B4_DIAGNOSTIC_BY_REGIME.csv")
    atomic_csv(route.table, output / "FOLLOWUP_B4_TRAINED_ROUTE.csv")
    followup_row = metrics[metrics["candidate"] == "FOLLOWUP_CALIBRATED"].iloc[0]
    locked_row = metrics[metrics["candidate"] == "LOCKED_P2"].iloc[0]
    result = {
        "status": "COMPLETE_REUSED_DIAGNOSTIC",
        "fresh_sealed_claim": False,
        "route_mapping": route.mapping,
        "calibration": calibrator.to_artifact(),
        "followup_selection_score": float(followup_row["selection_score"]),
        "locked_selection_score": float(locked_row["selection_score"]),
        "selection_score_delta": float(followup_row["selection_score"] - locked_row["selection_score"]),
        "followup_pr_auc": float(followup_row["overall_raw_pr_auc"]),
        "locked_pr_auc": float(locked_row["overall_raw_pr_auc"]),
        "pr_auc_delta": float(followup_row["overall_raw_pr_auc"] - locked_row["overall_raw_pr_auc"]),
        "followup_brier": float(followup_row["overall_raw_brier"]),
        "locked_brier": float(locked_row["overall_raw_brier"]),
        "brier_improvement": float(locked_row["overall_raw_brier"] - followup_row["overall_raw_brier"]),
        "followup_logloss": float(followup_row["overall_raw_logloss"]),
        "locked_logloss": float(locked_row["overall_raw_logloss"]),
        "logloss_improvement": float(locked_row["overall_raw_logloss"] - followup_row["overall_raw_logloss"]),
    }
    atomic_json(result, output / "FOLLOWUP_B4_DIAGNOSTIC.json")
    bootstrap_frame = b4[["date", "ticker", "target", "regime"]].copy()
    bootstrap_frame["followup"] = followup["prediction"].to_numpy(dtype=float)
    bootstrap_frame["locked"] = locked["prediction"].to_numpy(dtype=float)
    return bootstrap_frame, result


def _bootstrap_scores(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    local = frame[["date", "ticker", "target", "regime", prediction]].rename(columns={prediction: "prediction"})
    row, _ = evaluate_prediction(local, "bootstrap")
    return {
        "selection_score": float(row["selection_score"]),
        "pr_auc": float(row["overall_raw_pr_auc"]),
        "roc_auc": float(row["overall_raw_roc_auc"]),
        "brier": float(row["overall_raw_brier"]),
        "logloss": float(row["overall_raw_logloss"]),
        "top3_precision_lift": float(row["top3_precision_lift"]),
    }


def _bootstrap_batch(args: tuple[pd.DataFrame, int, int, int]) -> list[dict[str, Any]]:
    frame, reps, seed, offset = args
    set_worker_mode(1, "above_normal")
    dates = np.asarray(sorted(frame["date"].unique()))
    groups = {date: np.asarray(index, dtype=np.int64) for date, index in frame.groupby("date").groups.items()}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for rep in range(reps):
        sampled = rng.choice(dates, len(dates), replace=True)
        indices = np.concatenate([groups[date] for date in sampled])
        local = frame.loc[indices].copy()
        local["date"] = np.concatenate([
            np.repeat(np.datetime64("2000-01-01") + np.timedelta64(i, "D"), len(groups[date]))
            for i, date in enumerate(sampled)
        ])
        followup = _bootstrap_scores(local, "followup")
        locked = _bootstrap_scores(local, "locked")
        row: dict[str, Any] = {"rep": offset + rep}
        for metric in followup:
            row[f"followup_{metric}"] = followup[metric]
            if metric in {"brier", "logloss"}:
                row[f"improvement_{metric}"] = locked[metric] - followup[metric]
            else:
                row[f"improvement_{metric}"] = followup[metric] - locked[metric]
        rows.append(row)
    return rows


def _bootstrap(
    frame: pd.DataFrame,
    output: Path,
    *,
    reps: int,
    workers: int,
    prefix: str = "FOLLOWUP_B4",
) -> dict[str, Any]:
    workers = max(1, min(int(workers), int(reps)))
    counts = [reps // workers + (1 if i < reps % workers else 0) for i in range(workers)]
    offsets = np.cumsum([0, *counts[:-1]]).tolist()
    tasks = [(frame, count, 20260809 + i * 1009, offsets[i]) for i, count in enumerate(counts) if count]
    context = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for part in executor.map(_bootstrap_batch, tasks):
            rows.extend(part)
    replicates = pd.DataFrame(rows).sort_values("rep")
    atomic_csv(replicates, output / f"{prefix}_BOOTSTRAP_REPLICATES.csv")
    summary_rows = []
    for column in [name for name in replicates if name.startswith("improvement_")]:
        values = replicates[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        summary_rows.append({
            "metric": column,
            "reps": int(len(values)),
            "mean_improvement": float(values.mean()),
            "median_improvement": float(np.median(values)),
            "positive_rate": float(np.mean(values > 0)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        })
    summary = pd.DataFrame(summary_rows)
    atomic_csv(summary, output / f"{prefix}_BOOTSTRAP_SUMMARY.csv")
    result = {
        "status": "COMPLETE",
        "reps": int(reps),
        "workers": workers,
        "sample_unit": "trading_date",
        "fresh_sealed_claim": False,
        "summary": summary.to_dict("records"),
    }
    atomic_json(result, output / f"{prefix}_BOOTSTRAP.json")
    return result


def _freeze_final_policy(wide: pd.DataFrame, selected: dict[str, Any], output: Path) -> dict[str, Any]:
    route = fit_conservative_pair_route(
        wide,
        shrink_rows=int(selected["shrink_rows"]),
        p7_margin=float(selected["p7_margin"]),
        min_dates=int(selected["min_dates"]),
        min_positives=int(selected["min_positives"]),
    )
    routed = route_prediction_columns(wide, route.mapping)
    calibrator = TrainOnlyCalibrator.fit(
        routed["prediction"].to_numpy(dtype=float),
        routed["target"].to_numpy(dtype=np.uint8),
        routed["regime"],
        method=str(selected["calibration_method"]),
        c_value=float(selected["calibration_c"]),
    )
    artifact = {
        "schema": "crashwatch_regime_router_followup_policy_v1",
        "selection_source": "B2/B3 prequential only; B4 excluded from candidate selection",
        "candidate": _candidate_kwargs(selected),
        "regime_mapping": route.mapping,
        "fallback_model": route.fallback_model,
        "calibration": calibrator.to_artifact(),
        "external_sealed_used": False,
        "frozen": True,
    }
    payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    artifact["policy_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    atomic_csv(route.table, output / "FOLLOWUP_FINAL_ROUTE_TABLE.csv")
    atomic_json(artifact, output / "FOLLOWUP_FROZEN_POLICY.json")
    return artifact


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    selection_output = Path(args.selection_output).expanduser().resolve()
    temporal_output = Path(args.temporal_output).expanduser().resolve()
    gate_output = Path(args.gate_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "followup_router.lock.json"
    acquire_lock(lock)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        hardware["gpu"] = nvml_snapshot()
        hardware["mode"] = "full_load_7950x3d_96gb_rtx5080_cache_analysis_cpu_bootstrap"
        atomic_json(hardware, output / "HARDWARE_FULL_LOAD.json")
        wide, _ = _load_oof(selection_output, temporal_output)
        selected, _ = _prequential_search(wide, output)
        gate_selection = read_json(gate_output / "GATE_SELECTION_BEFORE_B4.json", {})
        bootstrap_frame, b4 = _b4_diagnostic(wide, selected, gate_selection, output)
        bootstrap = _bootstrap(
            bootstrap_frame,
            output,
            reps=int(args.bootstrap_reps),
            workers=int(args.bootstrap_workers),
        )
        policy = _freeze_final_policy(wide, selected, output)
        result = {
            "status": "FOLLOWUP_ROUTER_EXPERIMENT_COMPLETE",
            "elapsed_seconds": float(time.time() - started),
            "candidates": len(candidate_grid()),
            "preselection_blocks": ["B2", "B3"],
            "selected": selected,
            "B4_diagnostic": b4,
            "bootstrap_reps": bootstrap["reps"],
            "frozen_policy_hash": policy["policy_hash"],
            "final_mapping": policy["regime_mapping"],
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
        }
        atomic_json(result, output / "RUN_STATUS.json")
        return result
    except Exception as exc:
        failed = {
            "status": "FAILED",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.time() - started),
            "external_sealed_target_consumed": False,
        }
        atomic_json(failed, output / "RUN_STATUS.json")
        raise
    finally:
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Full-load conservative pair routing and train-only calibration experiment")
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--gate-output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_expert_router_followup_v1"))
    parser.add_argument("--total-threads", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-workers", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
