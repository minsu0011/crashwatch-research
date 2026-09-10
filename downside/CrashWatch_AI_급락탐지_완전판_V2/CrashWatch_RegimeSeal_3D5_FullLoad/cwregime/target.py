from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetResult:
    label: np.ndarray
    valid: np.ndarray
    first_hit_day: np.ndarray
    worst_forward_return: np.ndarray
    audit: dict[str, Any]


def build_3d5_target_from_ret1(
    dates_ns: np.ndarray,
    tickers: np.ndarray,
    ret1: np.ndarray,
    *,
    horizon_days: int = 3,
    drop_threshold: float = -0.05,
) -> TargetResult:
    """Build label: min cumulative close-to-close return over t+1..t+h <= threshold.

    `ret1[t]` is assumed to be close_t / close_{t-1} - 1.  Therefore the
    cumulative return from t to t+k is the product of (1 + ret1[t+j]), j=1..k.
    Rows without a complete future horizon are marked invalid rather than negative.
    """
    dates_ns = np.asarray(dates_ns, dtype=np.int64)
    tickers = np.asarray(tickers)
    ret1 = np.asarray(ret1, dtype=np.float64)
    n = len(ret1)
    if not (len(dates_ns) == len(tickers) == n):
        raise ValueError("dates/tickers/ret1 length mismatch")
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    label = np.zeros(n, dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    first_hit = np.zeros(n, dtype=np.int8)
    worst = np.full(n, np.nan, dtype=np.float32)

    # Work ticker-by-ticker. Dataset is usually date-major, so do not assume ticker rows are contiguous.
    frame = pd.DataFrame({"row": np.arange(n, dtype=np.int64), "date": dates_ns, "ticker": tickers, "ret": ret1})
    for _, part in frame.groupby("ticker", sort=False):
        part = part.sort_values("date", kind="mergesort")
        rows = part["row"].to_numpy(dtype=np.int64)
        r = part["ret"].to_numpy(dtype=np.float64)
        m = len(rows)
        if m <= horizon_days:
            continue
        for local_i in range(0, m - horizon_days):
            future = r[local_i + 1: local_i + 1 + horizon_days]
            if len(future) != horizon_days or not np.isfinite(future).all():
                continue
            cumulative = np.cumprod(1.0 + future) - 1.0
            global_i = rows[local_i]
            valid[global_i] = True
            worst_value = float(np.min(cumulative))
            worst[global_i] = worst_value
            hits = np.flatnonzero(cumulative <= float(drop_threshold))
            if hits.size:
                label[global_i] = 1
                first_hit[global_i] = np.int8(int(hits[0]) + 1)

    positives = int(label[valid].sum())
    total = int(valid.sum())
    hit_counts = {str(day): int(np.sum(first_hit[valid] == day)) for day in range(1, horizon_days + 1)}
    audit = {
        "target_name": f"label_abs_crash_{horizon_days}d_{abs(drop_threshold)*100:g}pct",
        "definition": f"min cumulative close return over next 1..{horizon_days} trading days <= {drop_threshold:.4f}",
        "horizon_trading_days": int(horizon_days),
        "drop_threshold": float(drop_threshold),
        "rows": int(n),
        "valid_rows": total,
        "invalid_incomplete_horizon_rows": int(n - total),
        "positives": positives,
        "positive_rate": float(positives / total) if total else float("nan"),
        "first_hit_day_counts": hit_counts,
    }
    return TargetResult(label=label, valid=valid, first_hit_day=first_hit, worst_forward_return=worst, audit=audit)
