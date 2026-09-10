from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from surge_leadlag_corrective_v11_1 import run_pipeline
from verify_surge_leadlag_v11_1 import check_output


def build_fixture(root: Path) -> SimpleNamespace:
    rng = np.random.default_rng(20260815)
    tickers = [f"{i:06d}" for i in range(1, 7)]
    dates = pd.bdate_range("2024-01-02", periods=440)
    returns = rng.normal(0, 0.012, size=(len(dates), len(tickers)))
    returns[2:, 1] = 0.82 * returns[:-2, 0] + rng.normal(0, 0.004, size=len(dates) - 2)
    returns[1:, 3] = -0.45 * returns[:-1, 2] + rng.normal(0, 0.010, size=len(dates) - 1)

    dataset_rows = []
    sid = 0
    for di, date in enumerate(dates):
        for ti, ticker in enumerate(tickers):
            dataset_rows.append({
                "date": date,
                "ticker": ticker,
                "name": f"T{ti}",
                "market": "KOSPI" if ti < 4 else "KOSDAQ",
                "bucket": f"B{ti // 2}",
                "industry_name": "UNKNOWN",
                "market_cap": float(1e12 - ti * 1e10),
                "t_price_ret_1": float(returns[di, ti]),
                "return_pct": float(returns[di, ti] * 100),
                "_sid": sid,
            })
            sid += 1
    dataset = pd.DataFrame(dataset_rows)
    dataset_path = root / "dataset.csv"
    dataset.drop(columns=["_sid"]).to_csv(dataset_path, index=False)

    target_rows = []
    for source_row_id, row in dataset.iterrows():
        di = dates.get_loc(row["date"])
        ti = tickers.index(row["ticker"])
        if di + 3 < len(dates):
            forward = returns[di + 1 : di + 4, ti]
            cumulative = np.cumprod(1.0 + forward) - 1.0
            best = float(np.max(cumulative))
            label = int(best >= 0.025)
            valid = True
        else:
            best = np.nan
            label = 0
            valid = False
        target_rows.append({
            "source_row_id": int(source_row_id),
            "date": row["date"],
            "ticker": row["ticker"],
            "label_abs_surge_3d_5pct": label,
            "target_valid": valid,
            "best_forward_return_3d": best,
        })
    target_path = root / "target.csv"
    pd.DataFrame(target_rows).to_csv(target_path, index=False)

    fold_rows = []
    for fold_id in range(8):
        val_start_i = 120 + fold_id * 38
        val_end_i = val_start_i + 29
        train_end_i = val_start_i - 8
        fold_rows.append({
            "fold_id": fold_id,
            "fold_role": "ignored",
            "train_start": str(dates[0].date()),
            "train_end": str(dates[train_end_i].date()),
            "purge_start": str(dates[train_end_i + 1].date()),
            "purge_end": str(dates[val_start_i - 1].date()),
            "validation_start": str(dates[val_start_i].date()),
            "validation_end": str(dates[val_end_i].date()),
        })
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")

    v10 = root / "v10"
    v10.mkdir()
    base_rows = []
    target_df = pd.DataFrame(target_rows)
    for fold in fold_rows:
        fid = int(fold["fold_id"])
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        subset = target_df.loc[(target_df["date"] >= start) & (target_df["date"] <= end) & target_df["target_valid"]]
        for row in subset.itertuples(index=False):
            score = float(np.clip(0.20 + 0.45 * row.label_abs_surge_3d_5pct + rng.normal(0, 0.12), 0.001, 0.999))
            if row.ticker == "000006" and fid == 3:
                score = np.nan
            base_rows.append({
                "source_row_id": row.source_row_id,
                "ticker": row.ticker,
                "fold_id": fid,
                "base_score_raw": score,
            })
    pd.DataFrame(base_rows).to_csv(v10 / "ticker_base_oof_predictions_v10_2.csv", index=False)

    reference_path = root / "reference.csv"
    pd.DataFrame([{
        "leader": "000001",
        "follower": "000002",
        "directed_lag": 2,
        "discovery_best_lag_correlation": 0.8,
    }]).to_csv(reference_path, index=False)

    return SimpleNamespace(
        output=root / "out",
        folds=folds_path,
        dataset=dataset_path,
        target_sidecar=target_path,
        tickers="",
        screening_permutations=99,
        adaptive_permutations=399,
        adaptive_trigger_p=0.05,
        block_size=5,
        minimum_pair_observations=15,
        threads=2,
        seed=20260815,
        maxstat_alpha=0.20,
        minimum_forward_abs_corr=0.03,
        v11_reference_strong=reference_path,
        rolling_windows=[20, 40],
        rolling_step=20,
        rolling_residual_fit_days=80,
        rolling_residual_min_fit=40,
        v10_2_output=v10,
        probe_edge_sets="v11_strong_repaired,maxstat_corrected",
        minimum_target_event_n=3,
        target_precision=0.60,
        minimum_portfolio_alerts=3,
        minimum_probe_training_rows=50,
        minimum_validation_base_rows=10,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v11_1_e2e_") as tmp:
        root = Path(tmp)
        args = build_fixture(root)
        run_pipeline(args)
        status = json.loads((args.output / "RUN_STATUS.json").read_text(encoding="utf-8"))
        assert status["status"] == "SUCCESS", status
        required = [
            "maxstat_pair_results_v11_1.csv",
            "v11_strong_edge_recheck_v11_1.csv",
            "frozen_lag_rolling_v11_1.csv",
            "target_aligned_lift_v11_1.csv",
            "matched_probe_metrics_v11_1.csv",
            "matched_probe_portfolio_v11_1.csv",
            "LEADLAG_CORRECTIVE_RECOMMENDATION_V11_1.json",
        ]
        for name in required:
            path = args.output / name
            assert path.exists(), name
        manifest = json.loads((args.output / "edge_sets" / "V11_STRONG_REPAIRED" / "TARGET_ALIGNED_FEATURE_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["future_offsets_used"] is False
        assert all(edge["all_offsets_nonpositive"] for edge in manifest["edges"])
        rolling = pd.read_csv(args.output / "frozen_lag_rolling_v11_1.csv")
        if not rolling.empty:
            assert not rolling["lag_reoptimized"].astype(bool).any()
        verification = check_output(args.output)
        assert verification["status"] == "PASS", verification
        print(f"SYNTHETIC_E2E_PASS verifier_checks={verification['checks']}")


if __name__ == "__main__":
    main()
