from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: str
    train_end: str
    purge_start: str
    purge_end: str
    validation_start: str
    validation_end: str
    train_dates: int
    validation_dates: int


def make_walk_forward_folds(
    dates: pd.Series | pd.DatetimeIndex,
    n_folds: int = 4,
    validation_days: int = 75,
    purge_days: int = 20,
    min_train_days: int = 500,
    min_train_fraction: float = 0.45,
) -> list[dict]:
    unique = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    required_train = max(min_train_days, int(len(unique) * min_train_fraction))
    required = required_train + purge_days + validation_days
    if len(unique) < required:
        raise ValueError(f"거래일 부족: {len(unique)}일, 최소 {required}일 필요")
    last_validation_end = len(unique)
    first_validation_start = required_train + purge_days
    max_start = last_validation_end - validation_days
    starts = []
    if n_folds == 1:
        starts = [max_start]
    else:
        span = max_start - first_validation_start
        starts = [first_validation_start + round(span * i / (n_folds - 1)) for i in range(n_folds)]
    folds = []
    for fold_id, val_start in enumerate(starts):
        val_end = min(val_start + validation_days, len(unique))
        purge_start = val_start - purge_days
        train_end = purge_start
        train_dates = unique[:train_end]
        purge_dates = unique[purge_start:val_start]
        validation_dates_idx = unique[val_start:val_end]
        if len(train_dates) < required_train or len(validation_dates_idx) < validation_days:
            continue
        folds.append({
            "fold_id": fold_id,
            "train_dates_index": train_dates,
            "validation_dates_index": validation_dates_idx,
            "metadata": asdict(Fold(
                fold_id=fold_id,
                train_start=str(train_dates[0].date()), train_end=str(train_dates[-1].date()),
                purge_start=str(purge_dates[0].date()), purge_end=str(purge_dates[-1].date()),
                validation_start=str(validation_dates_idx[0].date()), validation_end=str(validation_dates_idx[-1].date()),
                train_dates=len(train_dates), validation_dates=len(validation_dates_idx),
            )),
        })
    if len(folds) != n_folds:
        raise ValueError(f"요청 fold={n_folds}, 생성 fold={len(folds)}")
    return folds
