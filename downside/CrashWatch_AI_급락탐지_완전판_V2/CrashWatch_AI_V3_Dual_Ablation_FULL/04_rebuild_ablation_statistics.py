from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_ablation.config import get_paths, load_baskets
from dual_ablation.experiment.runner import _write_mode_outputs
from dual_ablation.experiment.statistics import paired_delta_table, summarize_deltas
from dual_ablation.io_utils import atomic_csv, atomic_json


def rebuild(project: Path | None = None, bootstrap_samples: int = 4000) -> dict:
    paths = get_paths(project)
    metrics_path = paths.result_dual / "all_metrics_by_fold_seed_scope.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    # Preserve leading zeroes in Korean ticker identifiers when rebuilding
    # statistics from CSV rather than from the in-memory experiment frame.
    metrics = pd.read_csv(
        metrics_path,
        low_memory=False,
        dtype={"scope_value": "string", "target_ticker": "string"},
    )
    deltas = paired_delta_table(metrics)
    summary = summarize_deltas(deltas, bootstrap_samples=bootstrap_samples)

    atomic_csv(deltas, paths.result_dual / "paired_ablation_deltas.csv")
    atomic_csv(summary, paths.result_dual / "ablation_statistical_summary.csv")
    _write_mode_outputs(metrics, deltas, summary, paths)

    baskets = load_baskets(paths)
    bucket_source = summary.loc[
        summary["ablation_mode"].eq("bucket_mask")
        & summary["pair_scope_type"].eq("bucket")
        & summary["target_group"].notna()
        & summary["scope_value"].astype(str).eq(summary["target_bucket"].astype(str))
    ]
    if not bucket_source.empty:
        matrix = bucket_source.pivot_table(
            index="scope_value", columns="target_group", values="mean_delta", aggfunc="mean",
        ).reset_index().rename(columns={"scope_value": "bucket"})
        if "bucket_name" in baskets:
            names = baskets[["bucket", "bucket_name"]].drop_duplicates("bucket")
            matrix = matrix.merge(names, on="bucket", how="left")
        atomic_csv(matrix, paths.result_dual / "bucket_group_sensitivity_matrix.csv")

    ticker_source = summary.loc[
        summary["ablation_mode"].eq("ticker_mask")
        & summary["pair_scope_type"].eq("ticker")
        & summary["target_group"].notna()
        & summary["scope_value"].astype(str).eq(summary["target_ticker"].astype(str))
    ]
    if not ticker_source.empty:
        matrix = ticker_source.pivot_table(
            index="scope_value", columns="target_group", values="mean_delta", aggfunc="mean",
        ).reset_index().rename(columns={"scope_value": "ticker"})
        ticker_meta = baskets[["ticker", "name", "bucket"]].drop_duplicates("ticker")
        matrix = matrix.merge(ticker_meta, on="ticker", how="left")
        atomic_csv(matrix, paths.result_dual / "ticker_group_sensitivity_matrix.csv")

    run_summary_path = paths.result_dual / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary.update({
        "statistics_rebuilt_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_pvalue_correction": "add_one_monte_carlo",
    })
    atomic_json(run_summary, run_summary_path)
    return {
        "metric_rows": len(metrics),
        "paired_rows": len(deltas),
        "summary_rows": len(summary),
        "bootstrap_samples": bootstrap_samples,
        "result_dir": str(paths.result_dual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ablation statistics from cached metric rows.")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.project, args.bootstrap_samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
