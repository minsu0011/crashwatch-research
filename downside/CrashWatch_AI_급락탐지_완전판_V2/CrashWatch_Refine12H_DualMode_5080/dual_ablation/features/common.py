from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    denominator = b.replace(0, np.nan) if isinstance(b, pd.Series) else (np.nan if b == 0 else b)
    return a / denominator


def rolling_z(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(5, window // 3)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return safe_div(series - mean, std)


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window, min_periods=max(10, window // 2)).cov(x)
    var = x.rolling(window, min_periods=max(10, window // 2)).var()
    return safe_div(cov, var)


def downside_semivol(ret: pd.Series, window: int) -> pd.Series:
    downside = ret.where(ret < 0, 0.0)
    return np.sqrt((downside.pow(2)).rolling(window, min_periods=max(5, window // 2)).mean())


def drawdown(close: pd.Series, window: int) -> pd.Series:
    return safe_div(close, close.rolling(window, min_periods=max(5, window // 3)).max()) - 1.0


def event_decay(days_since: pd.Series, half_life: float) -> pd.Series:
    values = np.exp(-np.log(2) * days_since.clip(lower=0) / half_life)
    return values.where(days_since.notna(), 0.0)


def first_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)
