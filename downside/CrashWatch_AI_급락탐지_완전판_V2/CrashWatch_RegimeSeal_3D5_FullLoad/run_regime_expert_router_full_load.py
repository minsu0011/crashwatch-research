from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock, set_full_load_mode
from cwregime.expert_router import (
    DEFAULT_ROUTE,
    REGIME_KO_V2,
    RegimeExpertRouter,
    build_router_artifact,
    route_prediction_columns,
)
from cwregime.gating import apply_gate, evaluate_prediction, fit_gate
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


def _comparison_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    routed: pd.DataFrame,
    validation_status: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    candidates = {
        "REGIME_EXPERT_ROUTER": routed["prediction"].to_numpy(dtype=float),
        "P2_XGB_SINGLE_EXPERT": frame["P2_XGB"].to_numpy(dtype=float),
        "P7_LGB_SINGLE_EXPERT": frame["P7_LGB"].to_numpy(dtype=float),
        "LOCKED_P2_10LGB_90XGB": (
            0.1 * frame["P2_LGB"].to_numpy(dtype=float)
            + 0.9 * frame["P2_XGB"].to_numpy(dtype=float)
        ),
    }
    rows: list[dict[str, Any]] = []
    regimes: list[pd.DataFrame] = []
    for candidate, prediction in candidates.items():
        evaluated = frame.copy()
        evaluated["prediction"] = prediction
        row, by_regime = evaluate_prediction(evaluated, candidate)
        row.update({
            "scope": scope,
            "validation_status": validation_status,
        })
        by_regime.insert(0, "scope", scope)
        by_regime.insert(1, "validation_status", validation_status)
        rows.append(row)
        regimes.append(by_regime)
    return rows, regimes


def _validate_router_oof(
    wide: pd.DataFrame,
    selected: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    config = {
        "method": str(selected["method"]),
        "shrink_rows": int(selected["shrink_rows"]),
        "n_clusters": int(selected["n_clusters"]),
    }
    rows: list[dict[str, Any]] = []
    regime_parts: list[pd.DataFrame] = []
    route_rows: list[dict[str, Any]] = []
    prequential_predictions: list[pd.DataFrame] = []
    for outer_number in (2, 3, 4):
        train_blocks = [f"B{number}" for number in range(1, outer_number)]
        block = f"B{outer_number}"
        train = wide[wide["temporal_block"].isin(train_blocks)]
        validation = wide[wide["temporal_block"] == block].copy()
        gate = fit_gate(train, **config)
        routed = apply_gate(validation, gate)
        routed["validation_block"] = block
        routed["gate_train_blocks"] = ",".join(train_blocks)
        prequential_predictions.append(routed)
        local_rows, local_regimes = _comparison_rows(
            validation,
            scope=block,
            routed=routed,
            validation_status="HONEST_PREQUENTIAL_EARLIER_BLOCKS_ONLY",
        )
        rows.extend(local_rows)
        regime_parts.extend(local_regimes)
        for regime, expert in gate.mapping.items():
            route_rows.append({
                "validation_block": block,
                "gate_train_blocks": ",".join(train_blocks),
                "regime": regime,
                "regime_ko": REGIME_KO_V2.get(regime, regime),
                "selected_submodel": expert,
                "fallback_model": gate.fallback_model,
            })

    # This deployment map is learned from all four development blocks.  It is
    # useful as a replay/integration audit, but is never reported as independent.
    frozen = route_prediction_columns(wide, DEFAULT_ROUTE)
    local_rows, local_regimes = _comparison_rows(
        wide,
        scope="ALL_DEVELOPMENT",
        routed=frozen,
        validation_status="POST_HOC_DEPLOYMENT_REPLAY_NOT_INDEPENDENT",
    )
    rows.extend(local_rows)
    regime_parts.extend(local_regimes)

    metrics = pd.DataFrame(rows)
    by_regime = pd.concat(regime_parts, ignore_index=True)
    routes = pd.DataFrame(route_rows)
    atomic_csv(metrics, output / "ROUTER_VALIDATION_METRICS.csv")
    atomic_csv(by_regime, output / "ROUTER_VALIDATION_BY_REGIME.csv")
    atomic_csv(routes, output / "PREQUENTIAL_ROUTE_MAPS.csv")

    prequential = pd.concat(prequential_predictions, ignore_index=True)
    prequential_summary = {
        "rows": int(len(prequential)),
        "dates": int(prequential["date"].nunique()),
        "blocks": sorted(prequential["validation_block"].unique().tolist()),
        "selected_expert_rows": prequential["selected_submodel"].value_counts().to_dict(),
        "sealed_rows": 0,
        "external_sealed_target_read": False,
    }
    b4 = metrics[(metrics["scope"] == "B4")].copy()
    b4_router = b4[b4["candidate"] == "REGIME_EXPERT_ROUTER"].iloc[0]
    b4_locked = b4[b4["candidate"] == "LOCKED_P2_10LGB_90XGB"].iloc[0]
    summary = {
        **prequential_summary,
        "B4_router_selection_score": float(b4_router["selection_score"]),
        "B4_locked_P2_selection_score": float(b4_locked["selection_score"]),
        "B4_router_minus_locked_selection_score": float(
            b4_router["selection_score"] - b4_locked["selection_score"]
        ),
        "B4_router_pr_auc": float(b4_router["overall_raw_pr_auc"]),
        "B4_locked_P2_pr_auc": float(b4_locked["overall_raw_pr_auc"]),
        "B4_router_minus_locked_pr_auc": float(
            b4_router["overall_raw_pr_auc"] - b4_locked["overall_raw_pr_auc"]
        ),
        "interpretation": (
            "B4 is the already-consumed development pseudo-seal. "
            "No route was changed after observing B4."
        ),
    }
    atomic_json(summary, output / "ROUTER_VALIDATION_SUMMARY.json")
    return summary


def _route_table(gate: dict[str, Any], output: Path) -> pd.DataFrame:
    table = pd.read_csv(output.parent / "regime_3d5_submodel_gate_v1" / "FINAL_REGIME_GATE_TABLE.csv")
    keep = [name for name in ("regime", "rows", "positives", "dates", "support_status") if name in table.columns]
    table = table[keep].copy()
    table["regime_ko"] = table["regime"].map(REGIME_KO_V2)
    table["selected_expert"] = table["regime"].map(DEFAULT_ROUTE)
    table["profile"] = table["selected_expert"].str.split("_", n=1).str[0]
    table["algorithm"] = table["selected_expert"].str.split("_", n=1).str[1]
    table["gate_hash"] = gate["artifact_hash"]
    table["selection_is_frozen"] = True
    table["external_sealed_used_for_selection"] = False
    atomic_csv(table, output / "REGIME_TO_EXPERT_ROUTE.csv")
    return table


def _full_inference_smoke(
    router: RegimeExpertRouter,
    meta: dict[str, Any],
    regime_calendar_path: Path,
    output: Path,
) -> dict[str, Any]:
    required = router.required_features()
    columns = ["date", "ticker", *required]
    started = time.time()
    frame = pd.read_parquet(meta["dataset_path"], columns=columns)
    calendar = pd.read_csv(regime_calendar_path, usecols=["date", "regime"])
    development_max = pd.Timestamp(str(meta["dataset_date_max"])[:10])
    if pd.to_datetime(frame["date"]).max() > development_max:
        raise RuntimeError("inference smoke input extends beyond development; refusing to touch future interval")
    predictions = router.predict_frame(frame, regime_calendar=calendar)
    elapsed = time.time() - started
    if len(predictions) != len(frame):
        raise RuntimeError("full inference smoke row count mismatch")
    sample = pd.concat(
        [part.head(2) for _, part in predictions.groupby("regime", sort=True)],
        ignore_index=True,
    )
    atomic_csv(sample, output / "DEVELOPMENT_INFERENCE_SMOKE_SAMPLE.csv")
    summary = {
        "status": "PASS",
        "scope": "targetless development features only; not a performance estimate",
        "rows": int(len(predictions)),
        "dates": int(predictions["date"].nunique()),
        "tickers": int(predictions["ticker"].nunique()),
        "date_min": str(predictions["date"].min().date()),
        "date_max": str(predictions["date"].max().date()),
        "required_union_features": int(len(required)),
        "prediction_min": float(predictions["prediction"].min()),
        "prediction_max": float(predictions["prediction"].max()),
        "prediction_mean": float(predictions["prediction"].mean()),
        "mean_seed_uncertainty": float(predictions["seed_uncertainty"].mean()),
        "alerts": int(predictions["alert_top_3pct"].sum()),
        "minimum_one_alert_per_date": bool(
            predictions.groupby("date")["alert_top_3pct"].sum().ge(1).all()
        ),
        "elapsed_seconds": float(elapsed),
        "rows_per_second": float(len(predictions) / max(elapsed, 1e-9)),
        "router_diagnostics": router.last_diagnostics,
        "target_columns_read": False,
        "external_future_rows_read": False,
        "external_sealed_predictions_made": False,
    }
    atomic_json(summary, output / "FULL_INFERENCE_SMOKE.json")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    project = package.parent
    gate_output = Path(args.gate_output).expanduser().resolve()
    selection_output = Path(args.selection_output).expanduser().resolve()
    temporal_output = Path(args.temporal_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "regime_expert_router.lock.json"
    acquire_lock(lock)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        hardware["gpu"] = nvml_snapshot()
        hardware["mode"] = "full_load_7950x3d_96gb_rtx5080"
        atomic_json(hardware, output / "HARDWARE_FULL_LOAD.json")

        gate = read_json(gate_output / "FROZEN_REGIME_GATE.json", {})
        if not gate.get("frozen"):
            raise RuntimeError("frozen gate artifact is missing or not frozen")
        manifest = read_json(gate_output / "FINAL_MODEL_MANIFEST.json", {})
        if manifest.get("status") != "complete" or int(manifest.get("completed_models", 0)) != 20:
            raise RuntimeError("final submodel manifest is incomplete")
        route_table = _route_table(gate, output)

        artifact = build_router_artifact(
            gate,
            gate_output / "models",
            route_mapping=DEFAULT_ROUTE,
            artifact_root=output,
        )
        artifact_path = output / "REGIME_EXPERT_ROUTER_V2.json"
        atomic_json(artifact, artifact_path)

        wide, meta = _load_oof(selection_output, temporal_output)
        selected = read_json(gate_output / "GATE_SELECTION_BEFORE_B4.json", {})
        validation = _validate_router_oof(wide, selected, output)

        router = RegimeExpertRouter(
            artifact_path,
            total_threads=int(args.total_threads),
            use_gpu=not args.cpu_only,
            verify_hashes=True,
        )
        smoke = _full_inference_smoke(
            router,
            meta,
            selection_output / "regime_calendar.csv",
            output,
        )
        result = {
            "status": "REGIME_EXPERT_ROUTER_V2_COMPLETE",
            "elapsed_seconds": float(time.time() - started),
            "router_artifact": str(artifact_path),
            "router_hash": artifact["router_hash"],
            "gate_hash": artifact["gate_hash"],
            "route_count": int(len(route_table)),
            "experts_used": sorted(set(DEFAULT_ROUTE.values())),
            "seed_models_used": sum(len(value["models"]) for value in artifact["experts"].values()),
            "validation": validation,
            "full_inference_smoke": smoke,
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
            "production_recommendation": (
                "The router is operational and frozen. Keep it as a challenger until a longer "
                "external sealed interval covers all eight regimes."
            ),
        }
        review = {
            "status": result["status"],
            "what_was_built": (
                "A date-level mixture-of-experts router. It selects P2_XGB or P7_LGB by the "
                "frozen market regime and averages five seeds inside the selected expert."
            ),
            "production_route": DEFAULT_ROUTE,
            "router_hash": artifact["router_hash"],
            "models": {
                "experts": result["experts_used"],
                "seed_models": result["seed_models_used"],
            },
            "honest_B4_pseudo_seal": {
                "router_selection_score": validation["B4_router_selection_score"],
                "locked_P2_selection_score": validation["B4_locked_P2_selection_score"],
                "router_minus_locked_selection_score": validation["B4_router_minus_locked_selection_score"],
                "router_pr_auc": validation["B4_router_pr_auc"],
                "locked_P2_pr_auc": validation["B4_locked_P2_pr_auc"],
                "router_minus_locked_pr_auc": validation["B4_router_minus_locked_pr_auc"],
            },
            "scientific_verdict": (
                "Operational PASS; discrimination is promising, but superiority over locked P2 "
                "is not confirmed because B4 composite score is lower and calibration is weaker."
            ),
            "full_inference_test": {
                "status": smoke["status"],
                "rows": smoke["rows"],
                "rows_per_second": smoke["rows_per_second"],
                "gpu_used_for_P2_XGB": next(
                    item["gpu_used"]
                    for item in smoke["router_diagnostics"]["experts"]
                    if item["expert"] == "P2_XGB"
                ),
                "minimum_one_alert_per_date": smoke["minimum_one_alert_per_date"],
            },
            "bugs_fixed": [
                "LightGBM native Windows loader could not open Korean model paths; models now load from an in-memory string.",
                "Concurrent XGBoost CUDA and LightGBM OpenMP entry could terminate Python natively; expert runtimes are entered sequentially while each expert still uses full internal parallelism.",
            ],
            "sealed_policy": {
                "external_target_consumed": False,
                "external_predictions_made": False,
                "reason": "the available 20-date future interval still covers only five of eight regimes",
            },
        }
        atomic_json(review, output / "ROUTER_FINAL_REVIEW.json")
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
    parser = argparse.ArgumentParser(description="Build and validate the P2/P7 regime expert router")
    parser.add_argument(
        "--gate-output",
        default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"),
    )
    parser.add_argument(
        "--selection-output",
        default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"),
    )
    parser.add_argument(
        "--temporal-output",
        default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"),
    )
    parser.add_argument(
        "--output",
        default=str(project / "crashwatch_ai_data/regime_expert_router_v2"),
    )
    parser.add_argument("--total-threads", type=int, default=32)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
