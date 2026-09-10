from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, read_json
from cwregime.gating import apply_gate, evaluate_prediction, fit_gate
from run_regime_submodel_gate import _load_oof, atomic_csv


def _scores(frame: pd.DataFrame) -> dict[str, float]:
    row, _ = evaluate_prediction(frame, "bootstrap")
    return {
        "selection_score": float(row["selection_score"]),
        "pr_auc": float(row["overall_raw_pr_auc"]),
        "roc_auc": float(row["overall_raw_roc_auc"]),
        "brier": float(row["overall_raw_brier"]),
        "top3_precision_lift": float(row["top3_precision_lift"]),
        "worst_regime_pr_lift": float(row["worst_regime_pr_lift"]),
    }


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    selection = Path(args.selection_output).expanduser().resolve()
    temporal = Path(args.temporal_output).expanduser().resolve()
    selected = read_json(output / "GATE_SELECTION_BEFORE_B4.json", {})
    wide, _ = _load_oof(selection, temporal)
    gate = fit_gate(
        wide[wide.temporal_block.isin(["B1", "B2", "B3"])],
        method=selected["method"], shrink_rows=int(selected["shrink_rows"]),
        n_clusters=int(selected["n_clusters"]),
    )
    b4 = wide[wide.temporal_block == "B4"].copy()
    gated = apply_gate(b4, gate)
    locked = b4.copy(); locked["prediction"] = 0.1 * locked.P2_LGB + 0.9 * locked.P2_XGB
    p2_lgb = b4.copy(); p2_lgb["prediction"] = p2_lgb.P2_LGB
    prepared = {"gate": gated, "locked": locked, "p2_lgb": p2_lgb}
    dates = np.asarray(sorted(b4.date.unique()))
    groups = {name: {date: part for date, part in frame.groupby("date", sort=False)} for name, frame in prepared.items()}
    rng = np.random.default_rng(20260808)
    rows = []
    for rep in range(int(args.reps)):
        sampled = rng.choice(dates, len(dates), replace=True)
        scores = {}
        for name in prepared:
            parts = []
            for occurrence, date in enumerate(sampled):
                part = groups[name][date].copy()
                part["date"] = (pd.Timestamp("2000-01-01") + pd.Timedelta(days=occurrence)).strftime("%Y-%m-%d")
                parts.append(part)
            scores[name] = _scores(pd.concat(parts, ignore_index=True))
        row = {"rep": rep}
        for metric in scores["gate"]:
            row[f"gate_{metric}"] = scores["gate"][metric]
            row[f"gate_minus_locked_{metric}"] = scores["gate"][metric] - scores["locked"][metric]
            row[f"gate_minus_p2_lgb_{metric}"] = scores["gate"][metric] - scores["p2_lgb"][metric]
        rows.append(row)
    replicates = pd.DataFrame(rows)
    atomic_csv(replicates, output / "B4_DATE_BOOTSTRAP_REPLICATES.csv")
    summary_rows = []
    for column in [name for name in replicates.columns if name.startswith("gate_minus_")]:
        values = replicates[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        summary_rows.append({
            "comparison_metric": column, "reps": len(values),
            "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
            "positive_rate": float(np.mean(values > 0)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        })
    summary = pd.DataFrame(summary_rows)
    atomic_csv(summary, output / "B4_DATE_BOOTSTRAP_SUMMARY.csv")
    result = {
        "status": "complete", "reps": int(args.reps), "sample_unit": "trading date",
        "selection_or_tuning_after_B4": False,
        "gate_mapping": gate.mapping,
        "key_results": summary[summary.comparison_metric.isin([
            "gate_minus_locked_selection_score", "gate_minus_locked_pr_auc",
            "gate_minus_locked_roc_auc", "gate_minus_p2_lgb_selection_score",
        ])].to_dict("records"),
    }
    atomic_json(result, output / "B4_DATE_BOOTSTRAP_STATUS.json")
    return result


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    parser.add_argument("--selection-output", default=str(project / "crashwatch_ai_data/regime_3d5_full_load_v1"))
    parser.add_argument("--temporal-output", default=str(project / "crashwatch_ai_data/regime_3d5_temporal_breadth_v1"))
    parser.add_argument("--reps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))

