from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_parquet, merge_newer

LOGGER = logging.getLogger(__name__)


def collect_global_assets(paths: ProjectPaths, assets: pd.DataFrame, start: str, end: str, overwrite: bool = False) -> pd.DataFrame:
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for _, row in assets.iterrows():
        alias, symbol = str(row["alias"]), str(row["symbol"])
        try:
            raw = yf.download(
                symbol, start=start,
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                LOGGER.warning("글로벌 자산 데이터 없음: %s", symbol)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index().rename(columns={"Date": "date", "Close": f"{alias}_close", "Adj Close": f"{alias}_adj_close", "Volume": f"{alias}_volume"})
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
            keep = [c for c in ["date", f"{alias}_close", f"{alias}_adj_close", f"{alias}_volume"] if c in raw.columns]
            frames.append(raw[keep].dropna(subset=["date"]).drop_duplicates("date", keep="last"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("글로벌 자산 %s 실패: %s", symbol, exc)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date")
    path = paths.raw_dual / "global_assets.parquet"
    if path.exists() and not overwrite:
        out = merge_newer(pd.read_parquet(path), out, ["date"])
    atomic_parquet(out, path)
    return out
