from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .regimes import REGIMES


@dataclass(frozen=True)
class RegimeWindow:
    bank: str
    regime: str
    window_id: str
    start_date: str
    end_date: str
    start_ns: int
    end_ns: int
    target_regime_days: int
    total_days: int
    purity: float
    train_cutoff_ns: int
    train_cutoff_date: str
    train_dates: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_windows(
    cal: pd.DataFrame,
    regime: str,
    start_i: int,
    end_i: int,
    width: int,
    min_target_days: int,
) -> list[tuple[float, int, int, int]]:
    out: list[tuple[float, int, int, int]] = []
    labels = cal["regime"].to_numpy()
    upper = min(end_i, len(cal))
    for i in range(start_i, max(start_i, upper - width + 1)):
        j = i + width
        if j > upper:
            break
        count = int(np.sum(labels[i:j] == regime))
        if count < min_target_days:
            continue
        purity = count / width
        # Prefer purity first, then later windows inside each bank to reflect newer dynamics.
        score = purity + 0.000001 * i
        out.append((score, i, j, count))
    out.sort(reverse=True)
    return out


def _choose_non_overlapping(
    cal: pd.DataFrame,
    bank: str,
    regimes: list[str],
    start_i: int,
    end_i: int,
    width: int,
    min_target_days: int,
    purge_days: int,
    min_train_days: int,
) -> list[RegimeWindow]:
    occurrence = {
        regime: int(np.sum(cal.iloc[start_i:end_i]["regime"].to_numpy() == regime))
        for regime in regimes
    }
    # Rare regimes first; common regimes can be placed around them later.
    order = sorted(regimes, key=lambda r: (occurrence[r], r))
    occupied: list[tuple[int, int]] = []
    chosen: list[RegimeWindow] = []
    dates = cal["date_ns"].to_numpy(dtype=np.int64)
    for regime in order:
        candidates = _candidate_windows(cal, regime, start_i, end_i, width, min_target_days)
        selected = None
        for score, i, j, count in candidates:
            if i - purge_days < min_train_days:
                continue
            if any(not (j <= a or i >= b) for a, b in occupied):
                continue
            selected = (i, j, count)
            break
        if selected is None:
            # Controlled rescue: allow a narrower window, still no overlap.
            rescue_width = max(8, width // 2)
            rescue_min = 1
            candidates = _candidate_windows(cal, regime, start_i, end_i, rescue_width, rescue_min)
            for score, i, j, count in candidates:
                if i - purge_days < min_train_days:
                    continue
                if any(not (j <= a or i >= b) for a, b in occupied):
                    continue
                selected = (i, j, count)
                break
        if selected is None:
            raise RuntimeError(
                f"{bank}에서 {regime} 구간을 확보하지 못했습니다. "
                f"occurrence={occurrence[regime]}, range={start_i}:{end_i}"
            )
        i, j, count = selected
        occupied.append((i, j))
        total = j - i
        cutoff_i = i - purge_days - 1
        chosen.append(
            RegimeWindow(
                bank=bank,
                regime=regime,
                window_id=f"{bank}_{regime}",
                start_date=pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(dates[j - 1]).strftime("%Y-%m-%d"),
                start_ns=int(dates[i]),
                end_ns=int(dates[j - 1]),
                target_regime_days=int(count),
                total_days=int(total),
                purity=float(count / total),
                train_cutoff_ns=int(dates[cutoff_i]),
                train_cutoff_date=pd.Timestamp(dates[cutoff_i]).strftime("%Y-%m-%d"),
                train_dates=int(cutoff_i + 1),
            )
        )
    chosen.sort(key=lambda w: w.start_ns)
    return chosen


def build_two_bank_windows(calendar: pd.DataFrame, config: dict[str, Any]) -> tuple[list[RegimeWindow], list[RegimeWindow], dict[str, Any]]:
    cal = calendar.sort_values("date_ns").reset_index(drop=True)
    n = len(cal)
    min_train = int(config.get("min_train_days", 500))
    purge = int(config.get("purge_days", 3))
    width = int(config.get("window_days", 15))
    min_target = int(config.get("min_target_regime_days", 5))
    preferred_fraction = float(config.get("search_chronology_fraction", 0.62))
    earliest = max(min_train + purge + width * 2, int(n * 0.48))
    latest = min(n - width * 8, int(n * 0.78))
    if latest <= earliest:
        raise RuntimeError("regime bank split leaves insufficient history")
    labels = cal["regime"].to_numpy()
    best_split = None
    best_key = None
    step = max(1, (latest - earliest) // 80)
    for candidate in range(earliest, latest + 1, step):
        left_counts = [int(np.sum(labels[min_train + purge:candidate] == r)) for r in REGIMES]
        right_counts = [int(np.sum(labels[candidate:] == r)) for r in REGIMES]
        min_cover = min(left_counts + right_counts)
        total_cover = sum(min(left_counts[i], right_counts[i]) for i in range(len(REGIMES)))
        fraction_penalty = abs(candidate / n - preferred_fraction)
        key = (min_cover, total_cover, -fraction_penalty)
        if best_key is None or key > best_key:
            best_key = key; best_split = candidate
    split = int(best_split)

    search = _choose_non_overlapping(
        cal, "SEARCH", REGIMES, min_train + purge, split,
        width, min_target, purge, min_train,
    )
    confirm = _choose_non_overlapping(
        cal, "DEV_CONFIRM", REGIMES, split, n,
        width, min_target, purge, min_train,
    )
    if not max(w.end_ns for w in search) < min(w.start_ns for w in confirm):
        raise RuntimeError("SEARCH/DEV_CONFIRM chronology separation failed")
    audit = {
        "status": "complete",
        "all_regimes_present_search": sorted({w.regime for w in search}) == sorted(REGIMES),
        "all_regimes_present_confirm": sorted({w.regime for w in confirm}) == sorted(REGIMES),
        "search_windows": [w.to_dict() for w in search],
        "confirm_windows": [w.to_dict() for w in confirm],
        "split_date": pd.Timestamp(cal.iloc[split]["date_ns"]).strftime("%Y-%m-%d") if split < n else None,
        "policy": "SEARCH windows chronologically precede DEV_CONFIRM windows; FINAL_SEALED is separate and never read here",
    }
    return search, confirm, audit
