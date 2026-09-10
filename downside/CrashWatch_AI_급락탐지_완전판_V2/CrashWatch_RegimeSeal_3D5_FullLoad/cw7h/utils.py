from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(value: Any, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def hash_strings(values: Iterable[str], length: int = 24) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:length]


def hash_file(path: Path, *, strong: bool = False, length: int = 24) -> str:
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(stat.st_size).encode("ascii"))
    h.update(str(stat.st_mtime_ns).encode("ascii"))
    if strong:
        with path.open("rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    else:
        with path.open("rb") as f:
            first = f.read(4 * 1024 * 1024)
            h.update(first)
            if stat.st_size > 8 * 1024 * 1024:
                f.seek(max(0, stat.st_size - 4 * 1024 * 1024))
                h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:length]


def atomic_json(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ticker(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def ensure_thread_env(threads: int) -> None:
    value = str(max(1, int(threads)))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["LIGHTGBM_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))


def free_disk_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(Path(path).resolve()).free / 1024**3


def utc_timestamp() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan")
    if n > 20:
        rng = np.random.default_rng(17)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(200_000, n))
        perm = np.abs((signs * values).mean(axis=1))
        obs = abs(values.mean())
        return float((np.sum(perm >= obs - 1e-15) + 1) / (len(perm) + 1))
    obs = abs(values.mean())
    count = 0
    total = 1 << n
    for mask in range(total):
        signed_sum = 0.0
        for i, value in enumerate(values):
            signed_sum += value if (mask >> i) & 1 else -value
        if abs(signed_sum / n) >= obs - 1e-15:
            count += 1
    return float(count / total)


def bootstrap_mean_ci(values: np.ndarray, *, iterations: int = 20_000, seed: int = 17) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    reverse = np.empty_like(order)
    reverse[order] = np.arange(m)
    out[valid] = adjusted[reverse]
    return out


def deadline_remaining_seconds(deadline_epoch: float) -> float:
    return max(0.0, float(deadline_epoch) - time.time())
