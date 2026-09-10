from __future__ import annotations

import io
import logging

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ProjectPaths
from ..io_utils import atomic_parquet

LOGGER = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def _download(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def collect_fred_credit(paths: ProjectPaths, start: str, end: str, *, overwrite: bool = False) -> pd.DataFrame:
    path = paths.raw_dual / "fred_credit.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    cfg_path = paths.configs / "fred_credit_series.csv"
    if not cfg_path.exists():
        return pd.DataFrame()
    cfg = pd.read_csv(cfg_path)
    cfg = cfg.loc[pd.to_numeric(cfg.get("enabled", 1), errors="coerce").fillna(1).astype(int).eq(1)]
    frames = []
    for _, row in cfg.iterrows():
        alias, series_id = str(row["alias"]), str(row["series_id"])
        try:
            raw = _download(series_id)
            raw.columns = ["date", alias]
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw[alias] = pd.to_numeric(raw[alias].replace(".", pd.NA), errors="coerce")
            raw = raw.loc[raw["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            frames.append(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("FRED %s(%s) 실패: %s", alias, series_id, exc)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    atomic_parquet(out, path)
    return out
