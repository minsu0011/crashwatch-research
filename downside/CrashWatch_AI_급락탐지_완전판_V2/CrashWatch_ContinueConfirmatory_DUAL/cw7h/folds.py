from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_json


@dataclass(frozen=True)
class FoldSlice:
    fold_id: int
    train_start: int
    train_stop: int
    validation_start: int
    validation_stop: int
    train_date_min: str
    train_date_max: str
    validation_date_min: str
    validation_date_max: str
    train_dates: int
    validation_dates: int
    train_rows: int
    validation_rows: int
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_fold_slices(
    dates_ns: np.ndarray,
    definitions: list[dict[str, Any]],
    min_train_days: int,
    validation_days: int,
    purge_days: int,
    output_path: Path | None = None,
) -> list[FoldSlice]:
    dates_ns = np.asarray(dates_ns, dtype=np.int64)
    unique_dates = np.unique(dates_ns)
    rows: list[FoldSlice] = []
    for item in definitions:
        ts = pd.Timestamp(item["train_start"]).value
        te = pd.Timestamp(item["train_end"]).value
        vs = pd.Timestamp(item["validation_start"]).value
        ve = pd.Timestamp(item["validation_end"]).value
        train_start = int(np.searchsorted(dates_ns, ts, side="left"))
        train_stop = int(np.searchsorted(dates_ns, te, side="right"))
        valid_start = int(np.searchsorted(dates_ns, vs, side="left"))
        valid_stop = int(np.searchsorted(dates_ns, ve, side="right"))
        train_unique = unique_dates[(unique_dates >= ts) & (unique_dates <= te)]
        valid_unique = unique_dates[(unique_dates >= vs) & (unique_dates <= ve)]
        purge_unique = unique_dates[(unique_dates > te) & (unique_dates < vs)]
        reasons: list[str] = []
        if len(train_unique) < min_train_days:
            reasons.append(f"train_dates<{min_train_days}")
        if len(valid_unique) != validation_days:
            reasons.append(f"validation_dates={len(valid_unique)} expected={validation_days}")
        if len(purge_unique) < purge_days:
            reasons.append(f"purge_dates={len(purge_unique)} expected_at_least={purge_days}")
        if train_stop <= train_start:
            reasons.append("empty_train")
        if valid_stop <= valid_start:
            reasons.append("empty_validation")
        if train_stop > valid_start:
            reasons.append("train_validation_overlap")
        rows.append(FoldSlice(
            fold_id=int(item["fold_id"]),
            train_start=train_start,
            train_stop=train_stop,
            validation_start=valid_start,
            validation_stop=valid_stop,
            train_date_min=pd.Timestamp(train_unique.min()).strftime("%Y-%m-%d") if len(train_unique) else "",
            train_date_max=pd.Timestamp(train_unique.max()).strftime("%Y-%m-%d") if len(train_unique) else "",
            validation_date_min=pd.Timestamp(valid_unique.min()).strftime("%Y-%m-%d") if len(valid_unique) else "",
            validation_date_max=pd.Timestamp(valid_unique.max()).strftime("%Y-%m-%d") if len(valid_unique) else "",
            train_dates=int(len(train_unique)),
            validation_dates=int(len(valid_unique)),
            train_rows=int(max(0, train_stop - train_start)),
            validation_rows=int(max(0, valid_stop - valid_start)),
            eligible=not reasons,
            reason=";".join(reasons),
        ))
    if output_path is not None:
        atomic_json([r.to_dict() for r in rows], output_path)
    return rows


def rolling_inner_slices(
    dates_ns: np.ndarray,
    outer: FoldSlice,
    validation_days: int,
    purge_days: int,
    min_train_days: int,
    windows: int,
    step_days: int,
) -> list[tuple[slice, slice]]:
    train_dates = np.unique(dates_ns[outer.train_start:outer.train_stop])
    result: list[tuple[slice, slice]] = []
    for window in range(windows):
        offset = window * step_days
        val_end_pos = len(train_dates) - offset
        val_start_pos = val_end_pos - validation_days
        train_end_pos = val_start_pos - purge_days
        if train_end_pos < min_train_days or val_start_pos < 0:
            continue
        train_end_date = train_dates[train_end_pos - 1]
        val_start_date = train_dates[val_start_pos]
        val_end_date = train_dates[val_end_pos - 1]
        train_stop = int(np.searchsorted(dates_ns, train_end_date, side="right"))
        val_start = int(np.searchsorted(dates_ns, val_start_date, side="left"))
        val_stop = int(np.searchsorted(dates_ns, val_end_date, side="right"))
        train_stop = min(train_stop, outer.train_stop)
        if train_stop > outer.train_start and val_stop > val_start:
            result.append((slice(outer.train_start, train_stop), slice(val_start, val_stop)))
    return result
