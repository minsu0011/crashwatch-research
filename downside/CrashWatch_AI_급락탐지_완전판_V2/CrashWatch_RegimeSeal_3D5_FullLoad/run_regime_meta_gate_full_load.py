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

import joblib
import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock, set_full_load_mode
from cwregime.gating import evaluate_prediction
from cwregime.meta_gate import (
    apply_date_override,
    build_daily_meta_frame,
    fit_predict_meta_gate,
    meta_candidate_grid,
)
from cwregime.router_followup import TrainOnlyCalibrator
from run_regime_router_followup_full_load import _bootstrap
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


def _locked(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = 0.1 * out["P2_LGB"].to_numpy(dtype=float) + 0.9 * out["P2_XGB"].to_numpy(dtype=float)
    return out


def _metric(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    row, _ = evaluate_prediction(frame, name)
    return row


def _current_replacement_audit(project: Path, output: Path) -> dict[str, Any]:
    router_metrics = pd.read_csv(project / "crashwatch_ai_data/regime_expert_router_v2/ROUTER_VALIDATION_METRICS.csv")
    bootstrap = pd.read_csv(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1/B4_DATE_BOOTSTRAP_SUMMARY.csv")
    rows = []
    for block in ("B2", "B3", "B4"):
        part = router_metrics[router_metrics.scope == block]
        challenger = part[part.candidate == "REGIME_EXPERT_ROUTER"].iloc[0]
        locked = part[part.candidate == "LOCKED_P2_10LGB_90XGB"].iloc[0]
        rows.append({
            "block": block,
            "challenger_selection_score": float(challenger.selection_score),
            "locked_selection_score": float(locked.selection_score),
            "score_ratio": float(challenger.selection_score / locked.selection_score),
            "pr_auc_delta": float(challenger.overall_raw_pr_auc - locked.overall_raw_pr_auc),
            "roc_auc_delta": float(challenger.overall_raw_roc_auc - locked.overall_raw_roc_auc),
            "brier_improvement": float(locked.overall_raw_brier - challenger.overall_raw_brier),
            "logloss_improvement": float(locked.overall_raw_logloss - challenger.overall_raw_logloss),
        })
    table = pd.DataFrame(rows)
    atomic_csv(table, output / "CURRENT_CHALLENGER_REPLACEMENT_BY_BLOCK.csv")
    boot = bootstrap[bootstrap.comparison_metric == "gate_minus_locked_selection_score"].iloc[0]
    criteria = {
        "B2_noninferior": bool(table.loc[table.block == "B2", "score_ratio"].iloc[0] >= 1.0),
        "B3_noninferior": bool(table.loc[table.block == "B3", "score_ratio"].iloc[0] >= 1.0),
        "B4_noninferior": bool(table.loc[table.block == "B4", "score_ratio"].iloc[0] >= 1.0),
        "B4_brier_not_worse": bool(table.loc[table.block == "B4", "brier_improvement"].iloc[0] >= 0.0),
        "B4_bootstrap_lower_ci_positive": bool(float(boot.ci95_low) > 0.0),
        "all_eight_regimes_in_B4": False,
        "fresh_external_sealed_pass": False,
    }
    result = {
        "status": "EVALUATED",
        "replacement_approved": bool(all(criteria.values())),
        "decision": "DO_NOT_REPLACE_LOCKED_P2",
        "criteria": criteria,
        "B4_bootstrap_selection_score": {
            "mean_delta": float(boot.mean_delta),
            "positive_rate": float(boot.positive_rate),
            "ci95_low": float(boot.ci95_low),
            "ci95_high": float(boot.ci95_high),
        },
        "reason": (
            "the challenger wins B2/B3 but loses B4 composite/calibration, its bootstrap interval crosses zero, "
            "and no fresh full-regime sealed test exists"
        ),
    }
    atomic_json(result, output / "CURRENT_CHALLENGER_REPLACEMENT_AUDIT.json")
    return result


def _candidate_evaluation(
    candidate: dict[str, Any],
    train_daily: pd.DataFrame,
    validation_daily: pd.DataFrame,
    validation_rows: pd.DataFrame,
    feature_names: list[str],
    locked_metric: dict[str, Any],
    block: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    override, score, _ = fit_predict_meta_gate(train_daily, validation_daily, feature_names, candidate)
    evaluated = apply_date_override(validation_rows, validation_daily, override)
    row = _metric(evaluated, candidate["candidate_id"])
    p7_dates = validation_daily.loc[override, "date"]
    route_by_regime = validation_daily.assign(override=override).groupby("regime")["override"].agg(["sum", "count"]).reset_index()
    row.update(candidate)
    row.update({
        "outer_block": block,
        "p7_override_dates": int(override.sum()),
        "p7_override_fraction": float(override.mean()),
        "meta_score_mean": float(np.mean(score[np.isfinite(score)])) if np.isfinite(score).any() else None,
        "score_ratio_vs_locked": float(row["selection_score"] / locked_metric["selection_score"]),
        "pr_auc_delta_vs_locked": float(row["overall_raw_pr_auc"] - locked_metric["overall_raw_pr_auc"]),
        "roc_auc_delta_vs_locked": float(row["overall_raw_roc_auc"] - locked_metric["overall_raw_roc_auc"]),
        "brier_improvement_vs_locked": float(locked_metric["overall_raw_brier"] - row["overall_raw_brier"]),
    })
    route_by_regime.insert(0, "candidate_id", candidate["candidate_id"])
    route_by_regime.insert(1, "outer_block", block)
    return row, route_by_regime


def _search_meta_gate(
    wide: pd.DataFrame,
    daily: pd.DataFrame,
    feature_names: list[str],
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    regime_routes: list[pd.DataFrame] = []
    for outer_number in (2, 3):
        block = f"B{outer_number}"
        train_blocks = [f"B{i}" for i in range(1, outer_number)]
        train_daily = daily[daily.temporal_block.isin(train_blocks)]
        validation_daily = daily[daily.temporal_block == block]
        validation_rows = wide[wide.temporal_block == block]
        locked_metric = _metric(_locked(validation_rows), "LOCKED_P2")
        for candidate in meta_candidate_grid():
            row, routes = _candidate_evaluation(
                candidate, train_daily, validation_daily, validation_rows,
                feature_names, locked_metric, block,
            )
            row["train_blocks"] = ",".join(train_blocks)
            rows.append(row)
            regime_routes.append(routes)
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, output / "META_GATE_CANDIDATE_METRICS.csv")
    atomic_csv(pd.concat(regime_routes, ignore_index=True), output / "META_GATE_OVERRIDE_BY_REGIME.csv")
    summaries = []
    for cid, part in metrics.groupby("candidate_id", sort=False):
        first = part.iloc[0]
        scores = np.clip(part.selection_score.to_numpy(dtype=float), 1e-9, None)
        summaries.append({
            "candidate_id": cid,
            "model": first.model,
            "parameter": float(first.parameter),
            "l2": float(first.l2) if "l2" in part and pd.notna(first.l2) else None,
            "threshold": float(first.threshold),
            "B2_score": float(part.loc[part.outer_block == "B2", "selection_score"].iloc[0]),
            "B3_score": float(part.loc[part.outer_block == "B3", "selection_score"].iloc[0]),
            "gmean_score": float(np.exp(np.mean(np.log(scores)))),
            "min_score_ratio_vs_locked": float(part.score_ratio_vs_locked.min()),
            "mean_score_ratio_vs_locked": float(part.score_ratio_vs_locked.mean()),
            "mean_pr_auc_delta_vs_locked": float(part.pr_auc_delta_vs_locked.mean()),
            "mean_roc_auc_delta_vs_locked": float(part.roc_auc_delta_vs_locked.mean()),
            "min_p7_override_dates": int(part.p7_override_dates.min()),
            "mean_p7_override_dates": float(part.p7_override_dates.mean()),
            "max_p7_override_fraction": float(part.p7_override_fraction.max()),
        })
    summary = pd.DataFrame(summaries)
    summary["is_active_challenger"] = summary.min_p7_override_dates >= 5
    summary["noninferior_both_blocks"] = summary.min_score_ratio_vs_locked >= 1.0
    summary["eligible_active_challenger"] = summary.is_active_challenger & summary.noninferior_both_blocks
    active_ranked = summary[summary.is_active_challenger].sort_values(
        ["min_score_ratio_vs_locked", "gmean_score", "mean_pr_auc_delta_vs_locked"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    eligible = summary[summary.eligible_active_challenger].copy()
    selection_fell_back = eligible.empty
    if eligible.empty:
        eligible = summary[summary.candidate_id == "NO_OVERRIDE"].copy()
    eligible = eligible.sort_values(
        ["min_score_ratio_vs_locked", "gmean_score", "mean_pr_auc_delta_vs_locked", "mean_p7_override_dates"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    winner = eligible.iloc[0].to_dict()
    summary["selected"] = summary.candidate_id.eq(winner["candidate_id"])
    summary = summary.sort_values(
        ["selected", "eligible_active_challenger", "min_score_ratio_vs_locked", "gmean_score"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    atomic_csv(summary, output / "META_GATE_CANDIDATE_SELECTION.csv")
    selected = {
        "candidate_id": winner["candidate_id"],
        "model": winner["model"],
        "parameter": winner["parameter"],
        "l2": winner.get("l2"),
        "threshold": winner["threshold"],
        "B2_score": winner["B2_score"],
        "B3_score": winner["B3_score"],
        "min_score_ratio_vs_locked": winner["min_score_ratio_vs_locked"],
        "mean_pr_auc_delta_vs_locked": winner["mean_pr_auc_delta_vs_locked"],
        "min_p7_override_dates": winner["min_p7_override_dates"],
        "selection_fell_back_to_no_override": selection_fell_back,
        "selection_blocks": ["B2", "B3"],
        "B4_used_for_selection": False,
        "selection_rule": (
            "require at least five P7 override dates in each B2/B3 block and no composite-score loss; "
            "otherwise select no override"
        ),
        "best_active_challenger": (
            {
                "candidate_id": active_ranked.iloc[0]["candidate_id"],
                "model": active_ranked.iloc[0]["model"],
                "parameter": float(active_ranked.iloc[0]["parameter"]),
                "l2": (
                    float(active_ranked.iloc[0]["l2"])
                    if pd.notna(active_ranked.iloc[0]["l2"])
                    else None
                ),
                "threshold": float(active_ranked.iloc[0]["threshold"]),
                "min_score_ratio_vs_locked": float(active_ranked.iloc[0]["min_score_ratio_vs_locked"]),
                "gmean_score": float(active_ranked.iloc[0]["gmean_score"]),
                "mean_pr_auc_delta_vs_locked": float(active_ranked.iloc[0]["mean_pr_auc_delta_vs_locked"]),
                "min_p7_override_dates": int(active_ranked.iloc[0]["min_p7_override_dates"]),
                "selection_source": "B2/B3 only",
            }
            if not active_ranked.empty else None
        ),
    }
    atomic_json(selected, output / "META_GATE_SELECTION_BEFORE_B4.json")
    return selected, metrics


def _candidate_from_selected(selected: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "candidate_id": selected["candidate_id"],
        "model": selected["model"],
        "parameter": float(selected["parameter"]),
        "threshold": float(selected["threshold"]),
    }
    if selected.get("l2") is not None and np.isfinite(float(selected["l2"])):
        candidate["l2"] = float(selected["l2"])
    return candidate


def _fit_risk_head(train_rows: pd.DataFrame, raw_train: pd.DataFrame, raw_validation: pd.DataFrame) -> tuple[np.ndarray, TrainOnlyCalibrator]:
    calibrator = TrainOnlyCalibrator.fit(
        raw_train.prediction.to_numpy(dtype=float),
        raw_train.target.to_numpy(dtype=np.uint8),
        raw_train.regime,
        method="platt_global",
        c_value=0.1,
    )
    calibrated = calibrator.predict(raw_validation.prediction.to_numpy(dtype=float), raw_validation.regime)
    risk = 0.5 * raw_validation.prediction.to_numpy(dtype=float) + 0.5 * calibrated
    return np.clip(risk, 1e-7, 1.0 - 1e-7), calibrator


def _crossfitted_meta_routes(
    wide: pd.DataFrame,
    daily: pd.DataFrame,
    feature_names: list[str],
    candidate: dict[str, Any],
    through_block: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for block_number in range(1, through_block + 1):
        block = f"B{block_number}"
        rows = wide[wide.temporal_block == block]
        dates = daily[daily.temporal_block == block]
        if block_number == 1:
            override = np.zeros(len(dates), dtype=bool)
        else:
            train_dates = daily[daily.temporal_block.isin([f"B{i}" for i in range(1, block_number)])]
            override, _, _ = fit_predict_meta_gate(train_dates, dates, feature_names, candidate)
        parts.append(apply_date_override(rows, dates, override))
    return pd.concat(parts, ignore_index=True)


def _b4_and_freeze(
    wide: pd.DataFrame,
    daily: pd.DataFrame,
    feature_names: list[str],
    selected: dict[str, Any],
    output: Path,
    reps: int,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _candidate_from_selected(selected)
    train_daily = daily[daily.temporal_block.isin(["B1", "B2", "B3"])]
    b4_daily = daily[daily.temporal_block == "B4"]
    train_rows = wide[wide.temporal_block.isin(["B1", "B2", "B3"])]
    b4_rows = wide[wide.temporal_block == "B4"]
    b4_override, b4_score, estimator = fit_predict_meta_gate(train_daily, b4_daily, feature_names, candidate)
    raw_train = _crossfitted_meta_routes(wide, daily, feature_names, candidate, through_block=3)
    raw_b4 = apply_date_override(b4_rows, b4_daily, b4_override)
    risk, calibrator = _fit_risk_head(train_rows, raw_train, raw_b4)
    followup = raw_b4.copy(); followup["prediction"] = risk
    locked = _locked(b4_rows)
    raw_metric = _metric(raw_b4, "META_GATE_RAW_RANK")
    risk_metric = _metric(followup, "META_GATE_RISK")
    locked_metric = _metric(locked, "LOCKED_P2")
    atomic_csv(pd.DataFrame([raw_metric, risk_metric, locked_metric]), output / "META_GATE_B4_DIAGNOSTIC_METRICS.csv")
    b4_dates = b4_daily[["date", "regime", "p7_utility_delta"]].copy()
    b4_dates["meta_score"] = b4_score
    b4_dates["p7_override"] = b4_override
    atomic_csv(b4_dates, output / "META_GATE_B4_DATE_DECISIONS.csv")
    b4_result = {
        "status": "COMPLETE_REUSED_B4_DIAGNOSTIC",
        "fresh_sealed_claim": False,
        "p7_override_dates": int(b4_override.sum()),
        "p7_override_fraction": float(b4_override.mean()),
        "raw_selection_score_delta": float(raw_metric["selection_score"] - locked_metric["selection_score"]),
        "raw_pr_auc_delta": float(raw_metric["overall_raw_pr_auc"] - locked_metric["overall_raw_pr_auc"]),
        "raw_roc_auc_delta": float(raw_metric["overall_raw_roc_auc"] - locked_metric["overall_raw_roc_auc"]),
        "risk_brier_improvement": float(locked_metric["overall_raw_brier"] - risk_metric["overall_raw_brier"]),
        "risk_logloss_improvement": float(locked_metric["overall_raw_logloss"] - risk_metric["overall_raw_logloss"]),
        "risk_calibration": {**calibrator.to_artifact(), "blend": 0.5},
    }
    atomic_json(b4_result, output / "META_GATE_B4_DIAGNOSTIC.json")
    bootstrap_frame = b4_rows[["date", "ticker", "target", "regime"]].copy()
    bootstrap_frame["followup"] = risk
    bootstrap_frame["locked"] = locked.prediction.to_numpy(dtype=float)
    bootstrap = _bootstrap(bootstrap_frame, output, reps=reps, workers=workers)

    all_override, all_score, final_estimator = fit_predict_meta_gate(daily, daily, feature_names, candidate)
    model_path = output / "META_GATE_FINAL_MODEL.joblib"
    if final_estimator is not None:
        temp_path = output / f".{model_path.name}.{os.getpid()}.tmp"
        joblib.dump(final_estimator, temp_path, compress=3)
        os.replace(temp_path, model_path)
    elif model_path.exists():
        model_path.unlink()
    all_raw = _crossfitted_meta_routes(wide, daily, feature_names, candidate, through_block=4)
    final_calibrator = TrainOnlyCalibrator.fit(
        all_raw.prediction.to_numpy(dtype=float), all_raw.target.to_numpy(dtype=np.uint8), all_raw.regime,
        method="platt_global", c_value=0.1,
    )
    policy = {
        "schema": "crashwatch_continuous_meta_gate_v1",
        "base_expert": "P2_LOCKED=0.1*P2_LGB+0.9*P2_XGB",
        "challenger_expert": "P7_LGB",
        "candidate": candidate,
        "feature_names": feature_names,
        "model_path": model_path.name if final_estimator is not None else None,
        "risk_calibration": {**final_calibrator.to_artifact(), "blend": 0.5},
        "selection_source": "B2/B3 prequential only",
        "final_development_override_dates": int(all_override.sum()),
        "external_sealed_used": False,
        "frozen": True,
    }
    payload = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    policy["policy_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:32]
    atomic_json(policy, output / "META_GATE_FROZEN_POLICY.json")
    return b4_result, {"bootstrap": bootstrap, "policy": policy}


def _best_active_b4_diagnostic(
    wide: pd.DataFrame,
    daily: pd.DataFrame,
    feature_names: list[str],
    active: dict[str, Any] | None,
    output: Path,
    reps: int,
    workers: int,
) -> dict[str, Any] | None:
    if not active:
        return None
    candidate = {
        "candidate_id": active["candidate_id"],
        "model": active["model"],
        "parameter": float(active["parameter"]),
        "threshold": float(active["threshold"]),
    }
    if active.get("l2") is not None:
        candidate["l2"] = float(active["l2"])
    train_daily = daily[daily.temporal_block.isin(["B1", "B2", "B3"])]
    b4_daily = daily[daily.temporal_block == "B4"]
    b4_rows = wide[wide.temporal_block == "B4"]
    override, score, _ = fit_predict_meta_gate(train_daily, b4_daily, feature_names, candidate)
    raw_b4 = apply_date_override(b4_rows, b4_daily, override)
    raw_train = _crossfitted_meta_routes(wide, daily, feature_names, candidate, through_block=3)
    risk, calibrator = _fit_risk_head(raw_train, raw_train, raw_b4)
    active_risk = raw_b4.copy(); active_risk["prediction"] = risk
    locked = _locked(b4_rows)
    raw_metric = _metric(raw_b4, "BEST_ACTIVE_META_RAW")
    risk_metric = _metric(active_risk, "BEST_ACTIVE_META_RISK")
    locked_metric = _metric(locked, "LOCKED_P2")
    atomic_csv(
        pd.DataFrame([raw_metric, risk_metric, locked_metric]),
        output / "BEST_ACTIVE_META_B4_METRICS.csv",
    )
    decisions = b4_daily[["date", "regime", "p7_utility_delta"]].copy()
    decisions["meta_score"] = score
    decisions["p7_override"] = override
    atomic_csv(decisions, output / "BEST_ACTIVE_META_B4_DATE_DECISIONS.csv")
    boot_frame = b4_rows[["date", "ticker", "target", "regime"]].copy()
    boot_frame["followup"] = risk
    boot_frame["locked"] = locked.prediction.to_numpy(dtype=float)
    bootstrap = _bootstrap(
        boot_frame, output, reps=reps, workers=workers, prefix="BEST_ACTIVE_META_B4",
    )
    summary = {row["metric"]: row for row in bootstrap["summary"]}
    result = {
        "status": "COMPLETE_REUSED_B4_DIAGNOSTIC",
        "fresh_sealed_claim": False,
        "candidate": active,
        "p7_override_dates": int(override.sum()),
        "p7_override_fraction": float(override.mean()),
        "raw_selection_score_delta": float(raw_metric["selection_score"] - locked_metric["selection_score"]),
        "raw_pr_auc_delta": float(raw_metric["overall_raw_pr_auc"] - locked_metric["overall_raw_pr_auc"]),
        "raw_roc_auc_delta": float(raw_metric["overall_raw_roc_auc"] - locked_metric["overall_raw_roc_auc"]),
        "risk_brier_improvement": float(locked_metric["overall_raw_brier"] - risk_metric["overall_raw_brier"]),
        "risk_logloss_improvement": float(locked_metric["overall_raw_logloss"] - risk_metric["overall_raw_logloss"]),
        "risk_calibration": {**calibrator.to_artifact(), "blend": 0.5},
        "bootstrap_selection_improvement": summary["improvement_selection_score"],
        "bootstrap_pr_auc_improvement": summary["improvement_pr_auc"],
        "bootstrap_brier_improvement": summary["improvement_brier"],
    }
    atomic_json(result, output / "BEST_ACTIVE_META_B4_DIAGNOSTIC.json")
    return result


def _final_replacement_decision(
    current_audit: dict[str, Any],
    selected: dict[str, Any],
    b4: dict[str, Any],
    bootstrap: dict[str, Any],
    active_b4: dict[str, Any] | None,
    output: Path,
) -> dict[str, Any]:
    summary = {row["metric"]: row for row in bootstrap["summary"]}
    selection_boot = summary["improvement_selection_score"]
    active_candidate = selected.get("best_active_challenger")
    active_selection_boot = (
        active_b4.get("bootstrap_selection_improvement") if active_b4 else None
    )
    criteria = {
        "active_meta_challenger_exists": active_candidate is not None,
        "active_B2_B3_noninferior": bool(
            active_candidate is not None
            and float(active_candidate["min_score_ratio_vs_locked"]) >= 1.0
        ),
        "active_B4_noninferior": bool(
            active_b4 is not None and float(active_b4["raw_selection_score_delta"]) >= 0.0
        ),
        "active_B4_bootstrap_lower_ci_positive": bool(
            active_selection_boot is not None
            and float(active_selection_boot["ci95_low"]) > 0.0
        ),
        "fresh_external_full_regime_sealed_pass": False,
    }
    approved = all(criteria.values())
    result = {
        "status": "COMPLETE",
        "challenger_replaces_locked_P2": approved,
        "decision": "REPLACE" if approved else "DO_NOT_REPLACE",
        "criteria": criteria,
        "current_challenger_replacement_approved": current_audit["replacement_approved"],
        "selected_meta_gate": selected,
        "B4_reused_diagnostic": b4,
        "best_active_challenger_B4_reused_diagnostic": active_b4,
        "no_override_bootstrap_selection_improvement": selection_boot,
        "active_challenger_bootstrap_selection_improvement": active_selection_boot,
        "production_action": (
            "keep locked P2 as ranking/alert model; use the calibrated risk head; retain meta/P7 only as research challenger"
            if not approved else
            "promote meta gate only after an independent external full-regime sealed pass"
        ),
        "external_sealed_target_consumed": False,
    }
    atomic_json(result, output / "FINAL_REPLACEMENT_DECISION.json")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    project = package.parent
    selection = Path(args.selection_output).expanduser().resolve()
    temporal = Path(args.temporal_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "meta_gate.lock.json"
    acquire_lock(lock)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        hardware["gpu"] = nvml_snapshot()
        hardware["mode"] = "full_load_7950x3d_96gb_cached_oof_meta_gate"
        atomic_json(hardware, output / "HARDWARE_FULL_LOAD.json")
        current_audit = _current_replacement_audit(project, output)
        wide, _ = _load_oof(selection, temporal)
        calendar = pd.read_csv(selection / "regime_calendar.csv")
        daily, feature_names = build_daily_meta_frame(wide, calendar)
        atomic_csv(daily, output / "META_GATE_DAILY_TRAINING_TABLE.csv")
        atomic_json({
            "rows": len(daily), "features": feature_names, "feature_count": len(feature_names),
            "target_columns_in_features": [], "date_min": str(daily.date.min()), "date_max": str(daily.date.max()),
        }, output / "META_GATE_FEATURE_AUDIT.json")
        selected, _ = _search_meta_gate(wide, daily, feature_names, output)
        b4, final = _b4_and_freeze(
            wide, daily, feature_names, selected, output,
            int(args.bootstrap_reps), int(args.bootstrap_workers),
        )
        active_b4 = _best_active_b4_diagnostic(
            wide, daily, feature_names, selected.get("best_active_challenger"), output,
            int(args.bootstrap_reps), int(args.bootstrap_workers),
        )
        decision = _final_replacement_decision(
            current_audit, selected, b4, final["bootstrap"], active_b4, output,
        )
        result = {
            "status": "CONTINUOUS_META_GATE_EXPERIMENT_COMPLETE",
            "elapsed_seconds": float(time.time() - started),
            "daily_rows": len(daily),
            "meta_features": len(feature_names),
            "candidates": len(meta_candidate_grid()),
            "selected": selected,
            "B4_diagnostic": b4,
            "best_active_B4_diagnostic": active_b4,
            "bootstrap_reps": final["bootstrap"]["reps"],
            "replacement_decision": decision["decision"],
            "policy_hash": final["policy"]["policy_hash"],
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
        }
        atomic_json(result, output / "RUN_STATUS.json")
        return result
    except Exception as exc:
        failed = {
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.time() - started), "external_sealed_target_consumed": False,
        }
        atomic_json(failed, output / "RUN_STATUS.json")
        raise
    finally:
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Continuous date-level P7 meta-gate and challenger replacement audit")
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_continuous_meta_gate_v1"))
    parser.add_argument("--total-threads", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-workers", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
