from __future__ import annotations

import numpy as np
import pandas as pd

from cwregime.expert_router import (
    DEFAULT_ROUTE,
    add_daily_alerts,
    attach_regime_calendar,
    route_prediction_columns,
)


def main() -> None:
    dates = pd.to_datetime(["2026-01-02"] * 48 + ["2026-01-05"] * 9)
    frame = pd.DataFrame({
        "date": dates,
        "ticker": [f"{value:06d}" for value in range(len(dates))],
        "P2_XGB": np.linspace(0.01, 0.99, len(dates)),
        "P7_LGB": np.linspace(0.99, 0.01, len(dates)),
    })
    calendar = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "regime": ["CRASH_STRESS", "REBOUND"],
    })
    attached = attach_regime_calendar(frame, calendar)
    routed = route_prediction_columns(attached, DEFAULT_ROUTE)
    assert set(routed.loc[routed.date == pd.Timestamp("2026-01-02"), "selected_submodel"]) == {"P2_XGB"}
    assert set(routed.loc[routed.date == pd.Timestamp("2026-01-05"), "selected_submodel"]) == {"P7_LGB"}
    assert np.allclose(
        routed.loc[routed.selected_submodel == "P2_XGB", "prediction"],
        routed.loc[routed.selected_submodel == "P2_XGB", "P2_XGB"],
    )
    alerted = add_daily_alerts(routed)
    alerts = alerted.groupby("date")["alert_top_3pct"].sum().to_dict()
    assert alerts[pd.Timestamp("2026-01-02")] == 2
    assert alerts[pd.Timestamp("2026-01-05")] == 1

    unknown = attached.copy()
    unknown.loc[unknown.date == pd.Timestamp("2026-01-05"), "regime"] = "UNKNOWN"
    fallback = route_prediction_columns(unknown, DEFAULT_ROUTE)
    assert set(fallback.loc[fallback.regime == "UNKNOWN", "selected_submodel"]) == {"P2_XGB"}
    assert fallback.loc[fallback.regime == "UNKNOWN", "route_was_fallback"].all()

    invalid = attached.copy()
    invalid.loc[invalid.index[0], "regime"] = "REBOUND"
    try:
        attach_regime_calendar(invalid, None)
    except ValueError as exc:
        assert "exactly one regime" in str(exc)
    else:
        raise AssertionError("mixed regimes on one market date were not rejected")
    print("regime expert router tests: PASS")


if __name__ == "__main__":
    main()
