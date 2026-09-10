from __future__ import annotations

from dataclasses import dataclass, asdict
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
    profile_dir: Path,
    fold_definitions: list[dict[str, Any]],
    min_train_days: int,
    expected_validation_days: int,
    purge_days: int,
    output_path: Path | None = None,
) -> list[FoldSlice]:
    dates_ns = np.load(profile_dir / "dates_ns.npy", mmap_mode="r")
    unique_dates = np.unique(dates_ns)
    slices: list[FoldSlice] = []
    for fold in fold_definitions:
        train_start_ns = pd.Timestamp(fold["train_start"]).value
        train_end_ns = pd.Timestamp(fold["train_end"]).value
        val_start_ns = pd.Timestamp(fold["validation_start"]).value
        val_end_ns = pd.Timestamp(fold["validation_end"]).value
        train_start = int(np.searchsorted(dates_ns, train_start_ns, side="left"))
        train_stop = int(np.searchsorted(dates_ns, train_end_ns, side="right"))
        val_start = int(np.searchsorted(dates_ns, val_start_ns, side="left"))
        val_stop = int(np.searchsorted(dates_ns, val_end_ns, side="right"))
        train_unique = unique_dates[(unique_dates >= train_start_ns) & (unique_dates <= train_end_ns)]
        val_unique = unique_dates[(unique_dates >= val_start_ns) & (unique_dates <= val_end_ns)]
        purge_unique = unique_dates[(unique_dates > train_end_ns) & (unique_dates < val_start_ns)]
        reasons: list[str] = []
        if len(train_unique) < min_train_days:
            reasons.append(f"train_dates<{min_train_days}")
        if len(val_unique) != expected_validation_days:
            reasons.append(f"validation_dates={len(val_unique)} expected={expected_validation_days}")
        if len(purge_unique) < purge_days:
            reasons.append(f"purge_dates={len(purge_unique)} expected_at_least={purge_days}")
        if train_stop <= train_start:
            reasons.append("empty_train")
        if val_stop <= val_start:
            reasons.append("empty_validation")
        if train_stop > val_start:
            reasons.append("train_validation_overlap")
        slices.append(FoldSlice(
            fold_id=int(fold["fold_id"]),
            train_start=train_start,
            train_stop=train_stop,
            validation_start=val_start,
            validation_stop=val_stop,
            train_date_min=pd.Timestamp(train_unique.min()).strftime("%Y-%m-%d") if len(train_unique) else "",
            train_date_max=pd.Timestamp(train_unique.max()).strftime("%Y-%m-%d") if len(train_unique) else "",
            validation_date_min=pd.Timestamp(val_unique.min()).strftime("%Y-%m-%d") if len(val_unique) else "",
            validation_date_max=pd.Timestamp(val_unique.max()).strftime("%Y-%m-%d") if len(val_unique) else "",
            train_dates=int(len(train_unique)),
            validation_dates=int(len(val_unique)),
            train_rows=int(max(0, train_stop - train_start)),
            validation_rows=int(max(0, val_stop - val_start)),
            eligible=not reasons,
            reason=";".join(reasons)
        ))
    if output_path is not None:
        atomic_json([item.to_dict() for item in slices], output_path)
    return slices


def inner_tuning_slices(dates_ns: np.ndarray, outer: FoldSlice, inner_validation_days: int, inner_purge_days: int, min_train_days: int) -> tuple[slice, slice] | None:
    train_dates = np.unique(dates_ns[outer.train_start:outer.train_stop])
    required = inner_validation_days + inner_purge_days + min_train_days
    if len(train_dates) < required:
        return None
    val_dates = train_dates[-inner_validation_days:]
    purge_start_position = len(train_dates) - inner_validation_days - inner_purge_days
    train_end_date = train_dates[purge_start_position - 1]
    val_start_date = val_dates[0]
    train_stop = int(np.searchsorted(dates_ns, train_end_date, side="right"))
    val_start = int(np.searchsorted(dates_ns, val_start_date, side="left"))
    val_stop = int(np.searchsorted(dates_ns, val_dates[-1], side="right"))
    train_start = outer.train_start
    train_stop = min(train_stop, outer.train_stop)
    if train_stop <= train_start or val_stop <= val_start:
        return None
    return slice(train_start, train_stop), slice(val_start, val_stop)
