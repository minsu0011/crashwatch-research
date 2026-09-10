from __future__ import annotations

"""Numba kernels used by the 11-hour finance feature pipeline.

The kernels accept contiguous float64 arrays and return float32 arrays.  They
intentionally avoid pandas rolling objects in the hot path.  Compilation is
cached on disk, so resumed runs do not pay the full JIT cost again.
"""

import numpy as np

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap
    prange = range


@njit(cache=True, nogil=True)
def _finite(value: float) -> bool:
    return np.isfinite(value)


@njit(cache=True, nogil=True)
def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.size
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        if _finite(a[i]) and _finite(b[i]) and abs(b[i]) > 1e-20:
            out[i] = np.float32(a[i] / b[i])
        else:
            out[i] = np.nan
    return out


@njit(cache=True, nogil=True)
def diff_lag(values: np.ndarray, lag: int) -> np.ndarray:
    n = values.size
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(lag, n):
        if _finite(values[i]) and _finite(values[i - lag]):
            out[i] = np.float32(values[i] - values[i - lag])
    return out


@njit(cache=True, nogil=True)
def pct_change(values: np.ndarray, lag: int) -> np.ndarray:
    n = values.size
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(lag, n):
        old = values[i - lag]
        new = values[i]
        if _finite(new) and _finite(old) and abs(old) > 1e-20:
            out[i] = np.float32(new / old - 1.0)
    return out


@njit(cache=True, nogil=True)
def rolling_sum(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = values.size
    out = np.full(n, np.nan, dtype=np.float32)
    running = 0.0
    count = 0
    for i in range(n):
        value = values[i]
        if _finite(value):
            running += value
            count += 1
        if i >= window:
            old = values[i - window]
            if _finite(old):
                running -= old
                count -= 1
        if count >= min_periods:
            out[i] = np.float32(running)
    return out


@njit(cache=True, nogil=True)
def rolling_mean_std(values: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    mean = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        total = 0.0
        total2 = 0.0
        count = 0
        for j in range(start, i + 1):
            v = values[j]
            if _finite(v):
                total += v
                total2 += v * v
                count += 1
        if count >= min_periods:
            m = total / count
            variance = max(0.0, total2 / count - m * m)
            mean[i] = np.float32(m)
            std[i] = np.float32(np.sqrt(variance))
    return mean, std


@njit(cache=True, nogil=True)
def rolling_zscore(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    mean, std = rolling_mean_std(values, window, min_periods)
    n = values.size
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if _finite(values[i]) and _finite(mean[i]) and _finite(std[i]) and std[i] > 1e-12:
            out[i] = np.float32((values[i] - mean[i]) / std[i])
    return out


@njit(cache=True, nogil=True)
def rolling_slope(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = values.size
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        count = 0
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        x = 0.0
        for j in range(start, i + 1):
            y = values[j]
            if _finite(y):
                sx += x
                sy += y
                sxx += x * x
                sxy += x * y
                count += 1
            x += 1.0
        den = count * sxx - sx * sx
        if count >= min_periods and abs(den) > 1e-20:
            out[i] = np.float32((count * sxy - sx * sy) / den)
    return out


@njit(cache=True, nogil=True)
def rolling_corr(a: np.ndarray, b: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = a.size
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        count = 0
        sa = sb = saa = sbb = sab = 0.0
        for j in range(start, i + 1):
            av = a[j]
            bv = b[j]
            if _finite(av) and _finite(bv):
                count += 1
                sa += av
                sb += bv
                saa += av * av
                sbb += bv * bv
                sab += av * bv
        if count >= min_periods:
            va = saa - sa * sa / count
            vb = sbb - sb * sb / count
            den = np.sqrt(max(0.0, va) * max(0.0, vb))
            if den > 1e-20:
                out[i] = np.float32((sab - sa * sb / count) / den)
    return out


@njit(cache=True, nogil=True)
def signed_streak(values: np.ndarray) -> np.ndarray:
    n = values.size
    out = np.zeros(n, dtype=np.float32)
    current = 0.0
    last_sign = 0
    for i in range(n):
        v = values[i]
        if not _finite(v) or v == 0.0:
            current = 0.0
            last_sign = 0
        else:
            sign = 1 if v > 0 else -1
            if sign == last_sign:
                current += sign
            else:
                current = float(sign)
                last_sign = sign
        out[i] = np.float32(current)
    return out


@njit(cache=True, nogil=True, parallel=True)
def rowwise_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows, cols = values.shape
    out = np.full(rows, np.nan, dtype=np.float32)
    for i in prange(rows):
        total = 0.0
        weight_total = 0.0
        for j in range(cols):
            v = values[i, j]
            w = weights[i, j]
            if _finite(v) and _finite(w) and w > 0:
                total += v * w
                weight_total += w
        if weight_total > 0:
            out[i] = np.float32(total / weight_total)
    return out


def warmup() -> None:
    """Compile all kernels once before the timed experiment begins."""
    x = np.linspace(1.0, 100.0, 256, dtype=np.float64)
    y = np.linspace(2.0, 80.0, 256, dtype=np.float64)
    safe_divide(x, y)
    diff_lag(x, 5)
    pct_change(x, 5)
    rolling_sum(x, 20, 5)
    rolling_mean_std(x, 20, 5)
    rolling_zscore(x, 20, 5)
    rolling_slope(x, 20, 5)
    rolling_corr(x, y, 20, 5)
    signed_streak(x - 50.0)
    rowwise_weighted_mean(np.vstack([x, y]).T, np.ones((256, 2), dtype=np.float64))
