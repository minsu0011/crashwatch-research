from __future__ import annotations

import logging

import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_parquet

LOGGER = logging.getLogger(__name__)


def import_derivatives_csv(paths: ProjectPaths, *, overwrite: bool = False) -> pd.DataFrame:
    """Import point-in-time KRX derivatives data supplied by the user.

    Official KRX web downloads are not scraped here because endpoint contracts are not
    public/stable. Put a CSV at external_inputs/derivatives_daily.csv. If available_from
    is absent, the feature builder shifts values by one trading observation.
    """
    output = paths.raw_dual / "derivatives_daily.parquet"
    if output.exists() and not overwrite:
        return pd.read_parquet(output)
    source = paths.project / "external_inputs" / "derivatives_daily.csv"
    if not source.exists():
        LOGGER.info("선물/옵션 외부 CSV가 없어 파생 피처를 건너뜁니다: %s", source)
        return pd.DataFrame()
    df = pd.read_csv(source)
    if "date" not in df.columns:
        raise KeyError("derivatives_daily.csv에 date 열이 필요합니다.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "available_from" in df.columns:
        df["available_from"] = pd.to_datetime(df["available_from"], errors="coerce")
        if (df["available_from"] < df["date"]).any():
            raise ValueError("available_from은 관측 date보다 빠를 수 없습니다.")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    atomic_parquet(df, output)
    return df
