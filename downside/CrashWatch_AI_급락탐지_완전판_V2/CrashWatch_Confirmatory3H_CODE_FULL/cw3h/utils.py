from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=json_default)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = df.copy()
        for col in columns:
            if col not in frame.columns:
                frame[col] = pd.Series(dtype="object")
        frame = frame[columns]
    else:
        frame = df
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        frame.to_csv(handle, index=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def canonical_json_hash(data: Any, length: int = 24) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def hash_strings(values: Iterable[str], length: int = 24) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:length]


def hash_file(path: Path, strong: bool = False, length: int = 24) -> str:
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(stat.st_size).encode("ascii"))
    h.update(str(stat.st_mtime_ns).encode("ascii"))
    sample_size = 2 * 1024 * 1024
    with path.open("rb") as handle:
        if strong:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        else:
            h.update(handle.read(sample_size))
            if stat.st_size > sample_size * 2:
                handle.seek(max(0, stat.st_size // 2 - sample_size // 2))
                h.update(handle.read(sample_size))
                handle.seek(max(0, stat.st_size - sample_size))
                h.update(handle.read(sample_size))
    return h.hexdigest()[:length]


def normalize_ticker(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def ensure_thread_env(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def free_disk_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(path).free / (1024**3)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def monotonic_seconds() -> float:
    return time.monotonic()


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})
