from __future__ import annotations

import numpy as np
import pandas as pd

from cwregime.meta_gate import apply_date_override, build_daily_meta_frame, fit_predict_meta_gate


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(17)
    rows = []
    calendar = []
    for day in range(80):
        date = pd.Timestamp("2025-01-01") + pd.offsets.BDay(day)
        favorable = day % 3 == 0
        calendar.append({
            "date": date, "ret1": 0.01 if favorable else -0.002,
            "ret_5": 0.04 if favorable else -0.01, "ret_20": 0.05 if favorable else -0.02,
            "ret_60": 0.02, "vol_20": 0.2, "vol_rank_252": 0.5,
            "drawdown_60": -0.05 if favorable else -0.01,
        })
        for ticker in range(30):
            y = int(rng.random() < 0.15)
            p2 = np.clip(0.08 + (0.15 if not favorable else 0.02) * y + rng.normal(0, 0.05), 0.001, 0.999)
            p7 = np.clip(0.08 + (0.22 if favorable else 0.01) * y + rng.normal(0, 0.05), 0.001, 0.999)
            rows.append({
                "date": date, "ticker": f"{ticker:06d}", "target": y,
                "P2_LGB": p2, "P2_XGB": p2, "P7_LGB": p7,
                "regime": "REBOUND" if favorable else "SIDEWAYS_LOW_VOL",
                "temporal_block": "B1" if day < 60 else "B2",
            })
    return pd.DataFrame(rows), pd.DataFrame(calendar)


def main() -> None:
    rows, calendar = fixture()
    daily, features = build_daily_meta_frame(rows, calendar)
    forbidden = {"positives", "positive_rate", "p7_win", "p7_utility_delta"}
    assert not (forbidden & set(features))
    train = daily[daily.temporal_block == "B1"]
    validation = daily[daily.temporal_block == "B2"]
    candidate = {"model": "ridge", "parameter": 1.0, "threshold": 0.0}
    override, score, model = fit_predict_meta_gate(train, validation, features, candidate)
    assert model is not None and len(override) == len(validation) and np.isfinite(score).all()
    applied = apply_date_override(rows[rows.temporal_block == "B2"], validation, override)
    assert len(applied) == len(rows[rows.temporal_block == "B2"])
    assert set(applied.selected_submodel) <= {"P2_LOCKED", "P7_LGB"}
    print("regime meta gate tests: PASS")


if __name__ == "__main__":
    main()
