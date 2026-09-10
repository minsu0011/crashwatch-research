from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REGIMES = [
    "CRASH_STRESS",
    "REBOUND",
    "BULL_LOW_VOL",
    "BULL_HIGH_VOL",
    "SIDEWAYS_LOW_VOL",
    "SIDEWAYS_HIGH_VOL",
    "BEAR_LOW_VOL",
    "BEAR_HIGH_VOL",
]


@dataclass(frozen=True)
class RegimeResult:
    calendar: pd.DataFrame
    audit: dict[str, Any]


def _rolling_percentile_rank(values: pd.Series, window: int) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        hist = arr[start:i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < max(20, min(window, 60)):
            continue
        current = arr[i]
        if np.isfinite(current):
            out[i] = float(np.mean(hist <= current))
    return pd.Series(out, index=values.index)


def build_market_regimes(
    dates_ns: np.ndarray,
    market_ret1_rows: np.ndarray,
    *,
    high_vol_percentile: float = 0.70,
    bull_20d: float = 0.03,
    bear_20d: float = -0.03,
    crash_5d: float = -0.05,
    crash_20d: float = -0.08,
    rebound_5d: float = 0.04,
    rebound_drawdown_60: float = -0.04,
    vol_rank_window: int = 252,
) -> RegimeResult:
    """Classify each trading date using only contemporaneous/trailing market information."""
    frame = pd.DataFrame({"date_ns": np.asarray(dates_ns, dtype=np.int64), "ret1": np.asarray(market_ret1_rows, dtype=float)})
    daily = frame.groupby("date_ns", sort=True)["ret1"].median().reset_index()
    daily["date"] = pd.to_datetime(daily["date_ns"])
    r = daily["ret1"].astype(float)
    gross = 1.0 + r.fillna(0.0)
    daily["ret_5"] = gross.rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    daily["ret_20"] = gross.rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    daily["ret_60"] = gross.rolling(60, min_periods=60).apply(np.prod, raw=True) - 1.0
    daily["vol_20"] = r.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    daily["vol_rank_252"] = _rolling_percentile_rank(daily["vol_20"], int(vol_rank_window))
    index_level = gross.cumprod()
    rolling_high = index_level.rolling(60, min_periods=20).max()
    daily["drawdown_60"] = index_level / rolling_high - 1.0

    regimes: list[str] = []
    for row in daily.itertuples(index=False):
        r5 = float(row.ret_5) if np.isfinite(row.ret_5) else 0.0
        r20 = float(row.ret_20) if np.isfinite(row.ret_20) else 0.0
        dd60 = float(row.drawdown_60) if np.isfinite(row.drawdown_60) else 0.0
        vrank = float(row.vol_rank_252) if np.isfinite(row.vol_rank_252) else 0.5
        high_vol = vrank >= float(high_vol_percentile)
        if r5 <= float(crash_5d) or r20 <= float(crash_20d):
            regime = "CRASH_STRESS"
        elif r5 >= float(rebound_5d) and dd60 <= float(rebound_drawdown_60):
            regime = "REBOUND"
        elif r20 >= float(bull_20d):
            regime = "BULL_HIGH_VOL" if high_vol else "BULL_LOW_VOL"
        elif r20 <= float(bear_20d):
            regime = "BEAR_HIGH_VOL" if high_vol else "BEAR_LOW_VOL"
        else:
            regime = "SIDEWAYS_HIGH_VOL" if high_vol else "SIDEWAYS_LOW_VOL"
        regimes.append(regime)
    daily["regime"] = regimes

    counts = daily["regime"].value_counts().to_dict()
    audit = {
        "regime_names": REGIMES,
        "counts_by_regime": {name: int(counts.get(name, 0)) for name in REGIMES},
        "definition": {
            "market_proxy": "daily median of configured market return feature across tickers",
            "crash_stress": f"ret_5 <= {crash_5d} OR ret_20 <= {crash_20d}",
            "rebound": f"ret_5 >= {rebound_5d} AND drawdown_60 <= {rebound_drawdown_60}",
            "direction": f"bull if ret_20 >= {bull_20d}, bear if ret_20 <= {bear_20d}, else sideways",
            "high_vol": f"trailing vol percentile over {vol_rank_window} dates >= {high_vol_percentile}",
            "leakage_policy": "regime labels use current/past returns only; no future target information",
        },
    }
    return RegimeResult(calendar=daily, audit=audit)
