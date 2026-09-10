from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 8) -> None:
    import time
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(2.0, 0.15 * (2 ** attempt)))
    if last_error is not None:
        raise last_error


def _atomic_tmp(path: Path) -> Path:
    import uuid
    return path.parent / f".__cw_{uuid.uuid4().hex[:10]}{path.suffix}.tmp"


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp(path)
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp(path)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp(path)
    try:
        df.to_parquet(tmp, index=False)
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
