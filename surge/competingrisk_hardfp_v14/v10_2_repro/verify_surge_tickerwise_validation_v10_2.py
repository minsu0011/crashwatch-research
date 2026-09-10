from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = [
    "RUN_STATUS.json",
    "FINAL_RECOMMENDATION_V10_2.json",
    "INDUSTRY_AUDIT_V10_2.json",
    "ticker_probe_eligibility_v10_2.csv",
    "ticker_map_by_fold_v10_2.csv",
    "ticker_map_summary_v10_2.csv",
    "ticker_hierarchical_effect_map_v10_2.csv",
    "ticker_precision_separator_map_v10_2.csv",
    "ticker_precision_separator_source_representatives_v10_2.csv",
    "ticker_residual_similarity_matrix_v10_2.csv",
    "probe_ticker_ranking_v10_2.csv",
    "PROBE_PROFILES_V10_2.json",
]


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/surge_tickerwise_correlation_map_v10_2")
    p.add_argument("--allow-failed", action="store_true")
    p.add_argument("--allow-non439", action="store_true")
    args = p.parse_args()
    root = Path(args.output).expanduser().resolve()
    failures: list[str] = []
    for name in REQUIRED:
        check((root / name).exists(), f"missing {name}", failures)
    if failures:
        raise SystemExit("\n".join(failures))
    status = json.loads((root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    check(args.allow_failed or status.get("status") == "SUCCESS", f"RUN_STATUS={status.get('status')}", failures)
    final = json.loads((root / "FINAL_RECOMMENDATION_V10_2.json").read_text(encoding="utf-8"))
    check(args.allow_non439 or int(final.get("features", 0)) == 439, "feature count is not 439", failures)
    check(int(final.get("recent_recomputed_ticker_folds", 0)) > 0, "no recent ticker fold was recomputed", failures)
    industry = json.loads((root / "INDUSTRY_AUDIT_V10_2.json").read_text(encoding="utf-8"))
    maps = pd.read_csv(root / "ticker_map_summary_v10_2.csv")
    if not bool(industry.get("valid", False)) and "transform" in maps.columns:
        check(~maps["transform"].astype(str).str.contains("date_industry_rank", case=False, na=False).any(), "invalid industry transform survived", failures)
    elig = pd.read_csv(root / "ticker_probe_eligibility_v10_2.csv")
    recent = elig.loc[elig["role"].astype(str).eq("recent_audit")]
    check(len(recent) > 0, "no recent eligibility rows", failures)
    check(int(recent["eligible_model_v10_2"].sum()) > 0, "adaptive recent eligibility produced 0 eligible", failures)
    precision = pd.read_csv(root / "ticker_precision_separator_map_v10_2.csv")
    if not precision.empty:
        cand = precision.loc[precision["precision_separator_selection_candidate"].astype(bool)]
        if not cand.empty:
            check(cand["specific_q_ticker_axis"].notna().all(), "selection candidates missing FDR q", failures)
            conf = cand.loc[cand["confirmation_status"].eq("SUPPORTED")]
            if not conf.empty:
                check((pd.to_numeric(conf["confirmation_direction_consistency"], errors="coerce") >= 1.0 - 1e-12).all(), "confirmation-supported candidate has direction reversal", failures)
            tested = cand.loc[~cand["recent_status"].eq("UNTESTED")]
            check(len(tested) == int(final.get("recent_tested_selection_candidates", -1)), "recent tested count mismatch", failures)
    robust = pd.read_csv(root / "ticker_hierarchical_effect_map_v10_2.csv")
    check("specific_z_conservative" in robust.columns, "conservative z missing", failures)
    check("specific_q_ticker_axis" in robust.columns, "ticker-axis FDR missing", failures)
    sim = pd.read_csv(root / "ticker_residual_similarity_matrix_v10_2.csv", index_col=0)
    if len(sim) >= 2:
        values = sim.to_numpy(float)
        off = values[~np.eye(len(values), dtype=bool)]
        finite = off[np.isfinite(off)]
        check(len(finite) > 0, "residual similarity has no finite off-diagonal", failures)
    if final.get("probe_run"):
        for name in ["ticker_probe_metrics_v10_2.csv", "ticker_probe_champions_v10_2.csv", "ticker_probe_portfolio_by_fold_v10_2.csv"]:
            check((root / name).exists(), f"probe output missing {name}", failures)
        if (root / "ticker_probe_champions_v10_2.csv").exists():
            champions = pd.read_csv(root / "ticker_probe_champions_v10_2.csv")
            check(champions["ticker"].nunique() <= len(final.get("probe_tickers", [])), "champions exceed selected probe tickers", failures)
    result = {"checks": len(REQUIRED) + 10, "failures": failures, "status": "PASS" if not failures else "FAIL"}
    (root / "VERIFICATION_V10_2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
