from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwfull.common import file_sha256
from cwregime.gating import canonical_hash
from cwregime.sealed_ticker import add_daily_alert_flag, apply_platt_blend, scope_metrics
from run_regime_submodel_gate import _load_oof


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def _ticker_identity(project: Path, tickers: list[str]) -> dict[str, dict[str, str]]:
    source = project / "CrashWatch_AI_V4_FocusedNested_DualMode/crashwatch_ai_data/development/training_dataset_finance11h.parquet"
    frame = pd.read_parquet(source, columns=["ticker", "name", "market"])
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates("ticker", keep="last").set_index("ticker")
    missing = sorted(set(tickers) - set(frame.index))
    if missing:
        raise RuntimeError(f"missing ticker identities: {missing}")
    return {
        ticker: {"name": str(frame.at[ticker, "name"]), "market": str(frame.at[ticker, "market"])}
        for ticker in tickers
    }


def freeze(output: Path, selection: Path, temporal: Path, project: Path) -> dict[str, Any]:
    started = output / "FUTURE_FINAL_SEALED_STARTED.json"
    consumed = output / "FUTURE_FINAL_SEALED_CONSUMED.json"
    if started.exists() or consumed.exists():
        raise RuntimeError("cannot create or change goal policy after final sealed starts")

    wide, _ = _load_oof(selection, temporal)
    wide["ticker"] = wide["ticker"].astype(str).str.zfill(6)
    wide["locked_p2_raw"] = 0.1 * wide["P2_LGB"].to_numpy(dtype=float) + 0.9 * wide["P2_XGB"].to_numpy(dtype=float)
    frozen = read_json(project / "crashwatch_ai_data/regime_locked_p7_override_v1/LOCKED_OVERRIDE_FROZEN_POLICY.json", {})
    calibration = frozen["risk_calibration"]
    wide["locked_p2_risk_probability"] = apply_platt_blend(wide["locked_p2_raw"].to_numpy(), calibration)
    wide = add_daily_alert_flag(wide, "locked_p2_raw")
    tickers = sorted(wide["ticker"].unique().tolist())
    identity = _ticker_identity(project, tickers)
    prevalence = {ticker: float(part["target"].mean()) for ticker, part in wide.groupby("ticker", sort=True)}

    core: dict[str, Any] = {
        "schema": "crashwatch_partial_future_sealed_ticker_goal_v1",
        "frozen": True,
        "model": {
            "name": "LOCKED_P2_LGB10_XGB90",
            "ranking_formula": "0.1*P2_LGB + 0.9*P2_XGB",
            "risk_probability": "0.5*raw + 0.5*frozen_global_platt(raw)",
            "challenger_selected": False,
        },
        "target": {"name": "label_abs_crash_3d_5pct", "horizon_trading_days": 3, "drop_threshold": -0.05},
        "primary_ranking_goal": {
            "pr_auc_lift_min": 1.0,
            "roc_auc_min": 0.5,
            "all_conditions_required": True,
            "interpretation": "outperform random/base-rate ranking on both PR and ROC axes",
        },
        "minimum_evaluation_support": {"min_rows": 12, "min_positives": 2, "min_negatives": 2},
        "secondary_diagnostics_not_in_primary_pass_fail": [
            "Brier score", "log loss", "Brier skill versus fixed development prevalence",
            "daily universe-wide Top 3% alert precision/recall/lift",
        ],
        "development_source": {
            "rows": int(len(wide)), "dates": int(wide["date"].nunique()),
            "date_min": str(wide["date"].min()), "date_max": str(wide["date"].max()),
            "blocks": sorted(wide["temporal_block"].unique().tolist()),
            "external_sealed_rows_read": 0,
        },
        "development_global_prevalence": float(wide["target"].mean()),
        "development_prevalence_by_ticker": prevalence,
        "ticker_identity": identity,
        "risk_calibration": calibration,
        "source_hashes": {
            "frozen_risk_policy": file_sha256(project / "crashwatch_ai_data/regime_locked_p7_override_v1/LOCKED_OVERRIDE_FROZEN_POLICY.json"),
            "frozen_regime_gate": file_sha256(output / "FROZEN_REGIME_GATE.json"),
            "final_model_manifest": file_sha256(output / "FINAL_MODEL_MANIFEST.json"),
        },
        "scientific_limits": [
            "The available future sealed interval has only 20 dates and about 17 label-valid dates.",
            "The interval lacks full coverage of all eight market regimes.",
            "NOT_EVALUABLE is distinct from NOT_MET.",
            "No threshold or model may be changed after sealed outcomes are opened.",
        ],
    }
    core["policy_hash"] = canonical_hash(core, 32)
    policy_path = output / "FUTURE_SEALED_TICKER_GOAL_POLICY.json"
    existing = read_json(policy_path, {})
    if existing and existing.get("policy_hash") != core["policy_hash"]:
        raise RuntimeError("a different frozen ticker goal policy already exists")

    rows = []
    for ticker, part in wide.groupby("ticker", sort=True):
        metrics = scope_metrics(
            part,
            raw_column="locked_p2_raw",
            risk_column="locked_p2_risk_probability",
            alert_column="is_top3_alert",
            benchmark_probability=prevalence[ticker],
        )
        rows.append({"ticker": ticker, **identity[ticker], **metrics})
    baseline = pd.DataFrame(rows)
    baseline["development_primary_goal_met"] = (
        baseline["pr_auc_lift"].ge(core["primary_ranking_goal"]["pr_auc_lift_min"])
        & baseline["roc_auc"].ge(core["primary_ranking_goal"]["roc_auc_min"])
    )
    atomic_csv(baseline, output / "DEVELOPMENT_LOCKED_P2_BASELINE_BY_TICKER.csv")
    atomic_json(core, policy_path)
    summary = {
        "status": "FROZEN_BEFORE_FINAL_SEALED",
        "policy_hash": core["policy_hash"],
        "development_tickers": int(len(baseline)),
        "development_goal_met": int(baseline["development_primary_goal_met"].sum()),
        "development_goal_not_met": int((~baseline["development_primary_goal_met"]).sum()),
        "sealed_target_read": False,
        "sealed_predictions_made": False,
    }
    atomic_json(summary, output / "DEVELOPMENT_LOCKED_P2_GOAL_SUMMARY.json")
    return summary


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Freeze per-ticker sealed success criteria using development OOF only")
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    package = Path(__file__).resolve().parent
    result = freeze(
        Path(args.output).expanduser().resolve(),
        Path(args.selection_output).expanduser().resolve(),
        Path(args.temporal_output).expanduser().resolve(),
        package.parent,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
