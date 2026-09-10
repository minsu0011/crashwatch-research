from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import pandas as pd

from ..config import ProjectPaths, get_paths, load_baskets
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class SourceRecord:
    source: str
    dataset: str
    scope: str
    status: str
    rows: int = 0
    started_at: str = ""
    completed_at: str = ""
    elapsed_seconds: float = 0.0
    file: str = ""
    error: str = ""
    checksum_sha256: str = ""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds * random.uniform(0.85, 1.25))


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if retry_if is not None and not retry_if(exc):
                raise
            if attempt + 1 >= attempts:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            LOGGER.warning("재시도 %s/%s: %s", attempt + 1, attempts, exc)
            sleep_with_jitter(delay)
    assert last is not None
    raise last


def read_enabled_tickers(paths: ProjectPaths) -> pd.DataFrame:
    baskets = load_baskets(paths).copy()
    baskets["ticker"] = normalize_ticker(baskets["ticker"])
    baskets = baskets.drop_duplicates("ticker", keep="last")
    return baskets.sort_values(["bucket", "ticker"]).reset_index(drop=True)


def date_chunks(start: str, end: str, months: int = 12) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if months <= 0:
        raise ValueError("months는 양수여야 합니다.")
    left = pd.Timestamp(start).normalize()
    right = pd.Timestamp(end).normalize()
    if left > right:
        raise ValueError(f"잘못된 기간: {start} > {end}")
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = left
    while cursor <= right:
        next_cursor = cursor + pd.DateOffset(months=months)
        chunk_end = min(next_cursor - pd.Timedelta(days=1), right)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def date8(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def safe_numeric(frame: pd.DataFrame, exclude: Iterable[str] = ("date", "ticker", "bucket", "name")) -> pd.DataFrame:
    out = frame.copy()
    excluded = set(exclude)
    for column in out.columns:
        if column not in excluded:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def normalize_frame(frame: pd.DataFrame | None, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date"])
    out = frame.copy()
    if "date" not in out.columns:
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[out.index.notna()].reset_index()
        out = out.rename(columns={out.columns[0]: "date"})
    if column_map:
        out = out.rename(columns={c: column_map.get(str(c).strip(), str(c).strip()) for c in out.columns})
    out.columns = [str(c).strip() for c in out.columns]
    out = normalize_date(out)
    out = safe_numeric(out)
    return out.sort_values("date").drop_duplicates("date", keep="last")


def merge_on_date(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [normalize_date(frame) for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=["date"])
    out = valid[0]
    for frame in valid[1:]:
        duplicate = [c for c in frame.columns if c != "date" and c in out.columns]
        if duplicate:
            frame = frame.drop(columns=duplicate)
        out = out.merge(frame, on="date", how="outer", validate="one_to_one")
    return out.sort_values("date").drop_duplicates("date", keep="last")


def save_frame(frame: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> tuple[Path, str]:
    atomic_csv(frame, csv_path)
    if parquet_path is not None:
        atomic_parquet(frame, parquet_path)
    return csv_path, sha256_file(csv_path)


def append_manifest(records: list[SourceRecord], path: Path) -> None:
    new = pd.DataFrame([asdict(row) for row in records])
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    if not combined.empty:
        key_cols = [c for c in ["source", "dataset", "scope", "started_at"] if c in combined.columns]
        combined = combined.drop_duplicates(key_cols, keep="last")
    atomic_csv(combined, path)


def load_local_env(project: Path) -> dict[str, str]:
    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None
    candidates = [project / ".env.data.local", project / ".env.dual", project / ".env"]
    if load_dotenv is not None:
        for path in candidates:
            if path.exists():
                load_dotenv(path, override=False)
    return {key: value for key, value in os.environ.items() if value is not None}


def sanitize_secret_error(message: str) -> str:
    text = str(message)
    for key in ["KRX_ID", "KRX_PW", "OPENDART_API_KEY", "DART_API_KEY", "ECOS_API_KEY", "DATA_GO_KR_SERVICE_KEY"]:
        value = os.getenv(key, "")
        if value:
            text = text.replace(value, f"<{key}:redacted>")
    return text[:2000]


def parse_ticker(value: Any) -> str:
    match = re.search(r"(\d{4,6})", str(value))
    return match.group(1).zfill(6) if match else ""


def project_paths(project: Path | None) -> ProjectPaths:
    return get_paths(project)


def write_summary(obj: dict[str, Any], path: Path) -> None:
    atomic_json(obj, path)


def records_to_frame(records: Iterable[SourceRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])
