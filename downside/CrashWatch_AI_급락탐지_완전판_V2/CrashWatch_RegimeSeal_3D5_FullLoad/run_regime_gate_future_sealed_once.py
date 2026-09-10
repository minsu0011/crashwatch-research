from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import file_sha256, set_full_load_mode
from cwregime.gating import SUBMODELS, apply_gate, canonical_hash, evaluate_prediction
from cwregime.regimes import REGIMES
from cwregime.sealed_ticker import (
    add_daily_alert_flag,
    apply_platt_blend,
    evaluate_locked_p2_by_ticker,
)
from cwregime.target import build_3d5_target_from_ret1


CONFIRMATION_PHRASE = "I_UNDERSTAND_FINAL_SEALED_IS_ONE_TIME"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def _validate_models(output: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if manifest.get("status") != "complete" or int(manifest.get("completed_models", 0)) != 20:
        errors.append("final model manifest is not complete 20/20")
    for item in manifest.get("models", []):
        path = Path(str(item.get("model_path", "")))
        if not path.exists():
            errors.append(f"missing model: {path}")
        elif item.get("sha256") != file_sha256(path):
            errors.append(f"model hash mismatch: {path}")
    return not errors, errors


def preflight(output: Path) -> dict[str, Any]:
    gate = read_json(output / "FROZEN_REGIME_GATE.json", {})
    models = read_json(output / "FINAL_MODEL_MANIFEST.json", {})
    future = read_json(output / "FUTURE_SEALED_FEATURE_MANIFEST.json", {})
    parity = read_json(output / "FUTURE_FEATURE_PARITY_AUDIT.json", {})
    calendar_path = output / "FUTURE_SEALED_REGIME_CALENDAR_NO_TARGET.csv"
    feature_path = output / "FUTURE_SEALED_FEATURES_NO_TARGET.parquet"
    goal_path = output / "FUTURE_SEALED_TICKER_GOAL_POLICY.json"
    errors: list[str] = []
    if not gate.get("frozen"):
        errors.append("gate is not frozen")
    if future.get("gate_hash") != gate.get("artifact_hash"):
        errors.append("future feature gate hash mismatch")
    if future.get("target_read") or future.get("sealed_predictions_made"):
        errors.append("future preparation claims target/prediction consumption")
    if parity.get("status") != "PASS" or int(parity.get("matching_features", 0)) != 378:
        errors.append("future feature parity is not 378/378 PASS")
    if not feature_path.exists() or future.get("sha256") != file_sha256(feature_path):
        errors.append("future feature file missing or hash mismatch")
    models_ok, model_errors = _validate_models(output, models)
    errors.extend(model_errors)
    if models.get("gate_hash") != gate.get("artifact_hash"):
        errors.append("model manifest gate hash mismatch")
    if not calendar_path.exists():
        errors.append("future regime calendar missing")
    goal = read_json(goal_path, {})
    goal_hash = goal.get("policy_hash")
    goal_core = dict(goal)
    goal_core.pop("policy_hash", None)
    if not goal.get("frozen"):
        errors.append("per-ticker performance goal is not frozen")
    elif canonical_hash(goal_core, 32) != goal_hash:
        errors.append("per-ticker performance goal hash mismatch")
    elif len(goal.get("ticker_identity", {})) != 48:
        errors.append("per-ticker performance goal does not cover 48 tickers")

    counts = {regime: 0 for regime in REGIMES}
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path)
        observed = calendar["regime"].value_counts().to_dict()
        counts.update({str(key): int(value) for key, value in observed.items()})
    full_coverage = all(counts[regime] >= 8 for regime in REGIMES)
    partial_coverage = sum(value >= 1 for value in counts.values()) >= 3 and max(counts.values()) >= 8
    if errors:
        status = "NOT_READY"
    elif full_coverage:
        status = "READY_FULL_REGIME_COVERAGE"
    elif partial_coverage:
        status = "PARTIAL_READY_FULL_REGIME_BLOCKED"
    else:
        status = "NOT_READY_INSUFFICIENT_FUTURE_COVERAGE"
    result = {
        "status": status,
        "checked_epoch": time.time(),
        "gate_hash": gate.get("artifact_hash"),
        "feature_sha256": future.get("sha256"),
        "models_ok": models_ok,
        "model_count": models.get("completed_models", 0),
        "feature_parity": parity.get("status"),
        "future_rows": future.get("rows"),
        "future_dates": future.get("dates"),
        "future_tickers": future.get("tickers"),
        "regime_counts": counts,
        "regimes_with_at_least_8_dates": [name for name, value in counts.items() if value >= 8],
        "full_regime_coverage": full_coverage,
        "partial_evaluation_possible": bool(not errors and partial_coverage),
        "errors": errors,
        "confirmation_phrase_required": CONFIRMATION_PHRASE,
        "ticker_goal_policy_hash": goal_hash,
        "ticker_goal_frozen_before_sealed": bool(goal.get("frozen")),
        "partial_run_requires_allow_partial": True,
        "started_marker_exists": (output / "FUTURE_FINAL_SEALED_STARTED.json").exists(),
        "consumed_marker_exists": (output / "FUTURE_FINAL_SEALED_CONSUMED.json").exists(),
        "target_read_during_preflight": False,
        "predictions_made_during_preflight": False,
        "scientific_limit": (
            "20 dates cover only part of the 8-regime space; a partial result cannot prove all weak/strong regimes"
        ),
    }
    atomic_json(result, output / "FUTURE_FINAL_SEALED_PREFLIGHT.json")
    return result


def _load_lgb_model(path: Path):
    import lightgbm as lgb

    ascii_dir = Path(tempfile.gettempdir()) / "cwregime_lgb_inference"
    ascii_dir.mkdir(parents=True, exist_ok=True)
    ascii_path = ascii_dir / f"{file_sha256(path)[:20]}.txt"
    if not ascii_path.exists() or ascii_path.stat().st_size != path.stat().st_size:
        shutil.copyfile(path, ascii_path)
    return lgb.Booster(model_file=str(ascii_path))


def _predict_submodels(
    frame: pd.DataFrame,
    gate: dict[str, Any],
    manifest: dict[str, Any],
    threads: int,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SUBMODELS}
    for item in manifest["models"]:
        grouped[str(item["submodel"])].append(item)
    predictions: dict[str, np.ndarray] = {}
    for submodel in SUBMODELS:
        profile = submodel.split("_", 1)[0]
        features = list(gate["profile_features"][profile])
        matrix = frame[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        seed_predictions = []
        for item in sorted(grouped[submodel], key=lambda value: int(value["seed"])):
            path = Path(item["model_path"])
            if item["algorithm"] == "lightgbm":
                model = _load_lgb_model(path)
                seed_predictions.append(np.asarray(model.predict(matrix, num_threads=threads), dtype=float))
            else:
                import xgboost as xgb

                model = xgb.Booster()
                try:
                    model.load_model(path)
                except Exception:
                    ascii_path = Path(tempfile.gettempdir()) / "cwregime_xgb_inference" / f"{file_sha256(path)[:20]}.ubj"
                    ascii_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, ascii_path)
                    model.load_model(ascii_path)
                dmatrix = xgb.DMatrix(matrix, nthread=threads)
                seed_predictions.append(np.asarray(model.predict(dmatrix), dtype=float))
        if len(seed_predictions) != 5:
            raise RuntimeError(f"expected five seed models for {submodel}, got {len(seed_predictions)}")
        predictions[submodel] = np.mean(seed_predictions, axis=0)
    return predictions


def _export(output: Path, consumed: dict[str, Any]) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_P2_P7_FutureSealed_RESULTS_{stamp}.zip"
    names = [
        "FUTURE_FINAL_SEALED_PREFLIGHT.json", "FUTURE_FINAL_SEALED_STARTED.json",
        "FUTURE_FINAL_SEALED_CONSUMED.json", "FUTURE_SEALED_METRICS.csv",
        "FUTURE_SEALED_REGIME_METRICS.csv", "FUTURE_SEALED_RESULT.json",
        "FROZEN_REGIME_GATE.json", "FINAL_MODEL_MANIFEST.json",
        "FUTURE_SEALED_FEATURE_MANIFEST.json", "FUTURE_FEATURE_PARITY_AUDIT.json",
        "FUTURE_SEALED_TICKER_GOAL_POLICY.json", "DEVELOPMENT_LOCKED_P2_GOAL_SUMMARY.json",
        "DEVELOPMENT_LOCKED_P2_BASELINE_BY_TICKER.csv",
        "FUTURE_SEALED_LOCKED_P2_BY_TICKER.csv", "FUTURE_SEALED_LOCKED_P2_TARGET_MET.csv",
        "FUTURE_SEALED_LOCKED_P2_TARGET_NOT_MET.csv", "FUTURE_SEALED_LOCKED_P2_NOT_EVALUABLE.csv",
        "FUTURE_SEALED_LOCKED_P2_OVERALL.json",
    ]
    start = "\n".join([
        "# CrashWatch P2/P7 Future Sealed Result",
        "",
        f"- scope: {consumed.get('scope')}",
        f"- valid rows: {consumed.get('valid_rows')}",
        f"- valid dates: {consumed.get('valid_dates')}",
        f"- gate minus locked score: {consumed.get('gate_minus_locked_selection_score')}",
        "- warning: partial regime coverage must not be interpreted as proof across all 8 regimes",
    ])
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start.encode("utf-8"))
        for name in names:
            path = output / name
            if path.exists():
                zipped.write(path, f"results/{name}")
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        entries = len(zipped.namelist())
    if bad:
        raise RuntimeError(f"sealed export corrupt: {bad}")
    return {
        "archive": str(archive), "size_bytes": archive.stat().st_size,
        "sha256": file_sha256(archive), "entries": entries, "verified": True,
    }


def consume_once(output: Path, confirmation: str, allow_partial: bool, threads: int) -> dict[str, Any]:
    check = preflight(output)
    if check["status"] not in {"READY_FULL_REGIME_COVERAGE", "PARTIAL_READY_FULL_REGIME_BLOCKED"}:
        raise RuntimeError(f"sealed preflight is not ready: {check}")
    if check["status"] == "PARTIAL_READY_FULL_REGIME_BLOCKED" and not allow_partial:
        raise RuntimeError("only partial regime coverage is available; --allow-partial-sealed is required")
    if confirmation != CONFIRMATION_PHRASE:
        raise RuntimeError("exact one-time final sealed confirmation phrase is required")
    started_path = output / "FUTURE_FINAL_SEALED_STARTED.json"
    consumed_path = output / "FUTURE_FINAL_SEALED_CONSUMED.json"
    if started_path.exists() or consumed_path.exists():
        raise RuntimeError("future final sealed was already started or consumed")
    gate = read_json(output / "FROZEN_REGIME_GATE.json", {})
    manifest = read_json(output / "FINAL_MODEL_MANIFEST.json", {})
    feature_manifest = read_json(output / "FUTURE_SEALED_FEATURE_MANIFEST.json", {})
    goal_policy = read_json(output / "FUTURE_SEALED_TICKER_GOAL_POLICY.json", {})
    started = {
        "status": "STARTED_IRREVERSIBLE", "started_epoch": time.time(),
        "gate_hash": gate["artifact_hash"], "feature_sha256": feature_manifest["sha256"],
        "partial_coverage": check["status"] != "READY_FULL_REGIME_COVERAGE",
        "ticker_goal_policy_hash": goal_policy["policy_hash"],
    }
    atomic_json(started, started_path)

    frame = pd.read_parquet(output / "FUTURE_SEALED_FEATURES_NO_TARGET.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    target = build_3d5_target_from_ret1(
        frame["date"].astype("int64").to_numpy(), frame["ticker"].to_numpy(),
        pd.to_numeric(frame["t_price_ret_1"], errors="coerce").to_numpy(),
        horizon_days=3, drop_threshold=-0.05,
    )
    model_predictions = _predict_submodels(frame, gate, manifest, threads)
    for name, values in model_predictions.items():
        frame[name] = values
    frame["locked_p2_raw"] = 0.1 * frame["P2_LGB"].to_numpy(dtype=float) + 0.9 * frame["P2_XGB"].to_numpy(dtype=float)
    frame["locked_p2_risk_probability"] = apply_platt_blend(
        frame["locked_p2_raw"].to_numpy(dtype=float), goal_policy["risk_calibration"]
    )
    calendar = pd.read_csv(output / "FUTURE_SEALED_REGIME_CALENDAR_NO_TARGET.csv")
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.tz_localize(None)
    frame = frame.merge(calendar[["date", "regime"]], on="date", how="left", validate="many_to_one")
    frame["target"] = target.label
    frame["first_hit_day"] = target.first_hit_day
    frame["worst_forward_return"] = target.worst_forward_return
    frame["target_valid"] = target.valid
    valid = frame[frame["target_valid"]].copy()
    valid = add_daily_alert_flag(valid, "locked_p2_raw")
    gated = apply_gate(valid, gate["regime_mapping"])
    locked = valid.copy()
    locked["prediction"] = locked["locked_p2_raw"]
    candidates: list[tuple[str, pd.DataFrame, str]] = [
        ("FROZEN_REGIME_GATE", gated, "prediction"),
        ("LOCKED_P2_LGB10_XGB90", locked, "prediction"),
    ] + [(name, valid, name) for name in SUBMODELS]
    rows = []
    regimes = []
    for name, candidate, column in candidates:
        row, by_regime = evaluate_prediction(candidate, name, column)
        rows.append(row); regimes.append(by_regime)
    metrics = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    atomic_csv(metrics, output / "FUTURE_SEALED_METRICS.csv")
    atomic_csv(pd.concat(regimes, ignore_index=True), output / "FUTURE_SEALED_REGIME_METRICS.csv")
    ticker_metrics, ticker_summary = evaluate_locked_p2_by_ticker(valid, goal_policy)
    atomic_csv(ticker_metrics, output / "FUTURE_SEALED_LOCKED_P2_BY_TICKER.csv")
    atomic_csv(
        ticker_metrics[ticker_metrics["primary_goal_status"] == "MET"],
        output / "FUTURE_SEALED_LOCKED_P2_TARGET_MET.csv",
    )
    atomic_csv(
        ticker_metrics[ticker_metrics["primary_goal_status"] == "NOT_MET"],
        output / "FUTURE_SEALED_LOCKED_P2_TARGET_NOT_MET.csv",
    )
    atomic_csv(
        ticker_metrics[ticker_metrics["primary_goal_status"].str.startswith("NOT_EVALUABLE")],
        output / "FUTURE_SEALED_LOCKED_P2_NOT_EVALUABLE.csv",
    )
    atomic_json(ticker_summary, output / "FUTURE_SEALED_LOCKED_P2_OVERALL.json")
    gated.to_parquet(output / "FUTURE_SEALED_PREDICTIONS.parquet", index=False, compression="zstd")
    gate_row = metrics[metrics.candidate == "FROZEN_REGIME_GATE"].iloc[0]
    locked_row = metrics[metrics.candidate == "LOCKED_P2_LGB10_XGB90"].iloc[0]
    result = {
        "status": "CONSUMED_ONCE", "scope": "partial future sealed" if allow_partial else "full future sealed",
        "gate_hash": gate["artifact_hash"], "feature_sha256": feature_manifest["sha256"],
        "valid_rows": int(len(valid)), "valid_dates": int(valid["date"].nunique()),
        "tickers": int(valid["ticker"].nunique()), "positives": int(valid["target"].sum()),
        "positive_rate": float(valid["target"].mean()), "regimes_present": sorted(valid["regime"].dropna().unique()),
        "gate_selection_score": float(gate_row["selection_score"]),
        "locked_selection_score": float(locked_row["selection_score"]),
        "gate_minus_locked_selection_score": float(gate_row["selection_score"] - locked_row["selection_score"]),
        "gate_pr_auc": float(gate_row["overall_raw_pr_auc"]),
        "locked_pr_auc": float(locked_row["overall_raw_pr_auc"]),
        "gate_minus_locked_pr_auc": float(gate_row["overall_raw_pr_auc"] - locked_row["overall_raw_pr_auc"]),
        "target_audit": target.audit,
        "ticker_goal_policy_hash": goal_policy["policy_hash"],
        "ticker_goal_status_counts": ticker_summary["ticker_status_counts"],
        "locked_p2_calibrated_overall": ticker_summary["overall"],
        "scientific_limit": "partial coverage does not validate absent or low-support regimes",
    }
    atomic_json(result, output / "FUTURE_SEALED_RESULT.json")
    consumed = {**result, "consumed_epoch": time.time()}
    atomic_json(consumed, consumed_path)
    consumed["result_export"] = _export(output, consumed)
    atomic_json(consumed, consumed_path)
    return consumed


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Preflight or consume the frozen P2/P7 future sealed interval once")
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    parser.add_argument("--confirm", default="")
    parser.add_argument("--allow-partial-sealed", action="store_true")
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    set_full_load_mode(args.threads, "high")
    if args.confirm:
        result = consume_once(output, args.confirm, args.allow_partial_sealed, args.threads)
    else:
        result = preflight(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
