from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
_TEMP_SEQUENCE = itertools.count()


def _temporary_path(path: Path) -> Path:
    """Return a short writer-unique temporary path next to the destination.

    Repeating a long destination filename in the temporary name can cross the
    legacy 260-character Windows limit even when the final path itself is
    valid.  PID plus a process-local monotonic counter is unique for concurrent
    writers while keeping the full path shorter than the destination.
    """
    return path.parent / f".tmp-{os.getpid():x}-{next(_TEMP_SEQUENCE):x}"


def _atomic_replace(tmp: Path, path: Path) -> None:
    """Replace *path*, tolerating transient Windows scanner/viewer locks.

    Windows may briefly deny ``os.replace`` while Explorer, an editor, or an
    antivirus process has the destination open.  A transient lock must not
    terminate a multi-hour experiment.  If the lock remains after retries,
    retain the newest complete snapshot in a stable ``.pending`` sidecar; a
    later successful checkpoint will replace the canonical file and remove it.
    """
    pending = path.with_suffix(path.suffix + ".pending")
    delay = 0.05
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass
            return
        except PermissionError:
            if attempt == 7:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 0.8)

    try:
        os.replace(tmp, pending)
    except OSError:
        # Preserve the original exception semantics only when even the
        # recovery snapshot cannot be retained.
        os.replace(tmp, path)
        return
    LOGGER.warning(
        "Destination remained locked; retained latest atomic snapshot at %s",
        pending,
    )


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _atomic_replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    _atomic_replace(tmp, path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(path)
    df.to_parquet(tmp, index=False)
    _atomic_replace(tmp, path)


def normalize_date(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        if out.index.name or not isinstance(out.index, pd.RangeIndex):
            out = out.reset_index().rename(columns={out.reset_index().columns[0]: column})
        else:
            raise KeyError(f"날짜 열이 없습니다: {column}")
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.tz_localize(None)
    return out.loc[out[column].notna()].sort_values(column)


def normalize_ticker(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def finite_float32(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def feature_hash(columns: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(map(str, columns))))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def read_parquet_tree(root: Path, pattern: str = "*.parquet") -> pd.DataFrame:
    paths = sorted(root.rglob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True, sort=False)


def merge_newer(existing: pd.DataFrame, incoming: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return incoming.sort_values(keys).drop_duplicates(keys, keep="last")
    if incoming.empty:
        return existing.sort_values(keys).drop_duplicates(keys, keep="last")
    return (
        pd.concat([existing, incoming], ignore_index=True, sort=False)
        .sort_values(keys)
        .drop_duplicates(keys, keep="last")
    )
