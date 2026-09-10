from __future__ import annotations

import argparse
import io
import json
import math

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calendar", required=True)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    calendar = pd.read_csv(args.calendar, encoding="utf-8-sig")
    calendar["date"] = pd.to_datetime(calendar["date_ns"].astype("int64"))
    daily = (
        predictions.groupby("date", sort=True)
        .agg(
            scored=("status", lambda values: (values == "SCORED_NORMAL_MARKET").any()),
            missing=("status", lambda values: (values == "ABSTAIN_MARKET_PROXY_MISSING").any()),
            history=("history_dates", "max"),
            scored_rows=("status", lambda values: int((values == "SCORED_NORMAL_MARKET").sum())),
            alerts=("alert_top3pct", "sum"),
        )
        .reset_index()
    )
    merged = daily.merge(calendar[["date", "normal_market"]], on="date", how="left", validate="one_to_one")
    eligible = (merged["history"] >= 252) & (~merged["missing"])
    gate_mismatch = int((merged.loc[eligible, "scored"] != merged.loc[eligible, "normal_market"].astype(bool)).sum())
    active = merged["scored_rows"] > 0
    expected_alerts = merged.loc[active, "scored_rows"].map(lambda count: max(1, math.ceil(int(count) * 0.03)))
    alert_mismatch = int((merged.loc[active, "alerts"].astype(int).to_numpy() != expected_alerts.astype(int).to_numpy()).sum())
    score_mask = predictions["status"].eq("SCORED_NORMAL_MARKET")
    score_visibility_mismatch = int((predictions["risk_score"].notna() != score_mask).sum())
    csv_fixture = pd.read_csv(
        io.StringIO("date,code,value\n2026-01-01,005930,1\n"),
        dtype={name: "string" for name in ("ticker", "stock_code", "code", "symbol")},
    )
    result = {
        "gate_mismatch_dates": gate_mismatch,
        "alert_mismatch_dates": alert_mismatch,
        "score_visibility_mismatch_rows": score_visibility_mismatch,
        "active_dates": int(active.sum()),
        "csv_ticker_preserved": str(csv_fixture.loc[0, "code"]),
    }
    if any(result[key] for key in ("gate_mismatch_dates", "alert_mismatch_dates", "score_visibility_mismatch_rows")):
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
