from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NORMAL_REGIMES = ("BULL_LOW_VOL", "SIDEWAYS_LOW_VOL", "BEAR_LOW_VOL")
ABNORMAL_REGIMES = ("CRASH_STRESS", "REBOUND", "BULL_HIGH_VOL", "SIDEWAYS_HIGH_VOL", "BEAR_HIGH_VOL")


@dataclass(frozen=True)
class GateResult:
    row_normal: np.ndarray
    calendar: pd.DataFrame
    audit: dict[str, Any]


def build_normal_gate(dates_ns: np.ndarray, regime_calendar: pd.DataFrame) -> GateResult:
    """Build a causal selective-prediction gate from precomputed trailing-only regimes.

    The gate is deliberately *not tuned*. It is active only in low-volatility bull,
    sideways, or bear regimes. Shock/rebound/high-volatility dates are abstained.
    """
    calendar = regime_calendar.copy()
    calendar["normal_market"] = calendar["regime"].isin(NORMAL_REGIMES)
    date_to_normal = dict(zip(calendar["date_ns"].astype("int64"), calendar["normal_market"].astype(bool)))
    row_normal = np.fromiter((bool(date_to_normal.get(int(x), False)) for x in np.asarray(dates_ns, dtype=np.int64)), dtype=bool, count=len(dates_ns))

    date_counts = calendar["normal_market"].value_counts().to_dict()
    regime_counts = calendar.groupby(["regime", "normal_market"], observed=False).size().reset_index(name="dates")
    audit = {
        "policy": "NORMAL_CORE_V1",
        "active_regimes": list(NORMAL_REGIMES),
        "abstain_regimes": list(ABNORMAL_REGIMES),
        "date_coverage": float(calendar["normal_market"].mean()) if len(calendar) else float("nan"),
        "active_dates": int(date_counts.get(True, 0)),
        "abstain_dates": int(date_counts.get(False, 0)),
        "selection_policy": "fixed before model evaluation; abnormal-market metrics never enter model selection",
        "inference_behavior": "emit risk score on normal dates; emit ABSTAIN_ABNORMAL_MARKET otherwise",
        "regime_counts": regime_counts.to_dict(orient="records"),
    }
    return GateResult(row_normal=row_normal, calendar=calendar, audit=audit)
