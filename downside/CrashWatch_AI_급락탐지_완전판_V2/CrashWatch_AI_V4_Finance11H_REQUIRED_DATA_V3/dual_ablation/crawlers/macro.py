from __future__ import annotations

import logging
import os
from urllib.parse import quote

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ProjectPaths
from ..io_utils import atomic_parquet, merge_newer

LOGGER = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
def _json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def collect_ecos(paths: ProjectPaths, start: str, end: str, overwrite: bool = False) -> pd.DataFrame:
    api_key = os.getenv("ECOS_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("ECOS_API_KEY가 없어 ECOS 수집을 건너뜁니다.")
        return pd.DataFrame()
    cfg = pd.read_csv(paths.configs / "ecos_series.csv")
    cfg = cfg.loc[pd.to_numeric(cfg["enabled"], errors="coerce").fillna(0).astype(int).eq(1)]
    frames: list[pd.DataFrame] = []
    for _, row in cfg.iterrows():
        alias = str(row["alias"])
        cycle = str(row["cycle"])
        start_key = pd.Timestamp(start).strftime("%Y%m%d" if cycle == "D" else "%Y%m")
        end_key = pd.Timestamp(end).strftime("%Y%m%d" if cycle == "D" else "%Y%m")
        url = "/".join([
            "https://ecos.bok.or.kr/api/StatisticSearch", api_key, "json", "kr", "1", "100000",
            quote(str(row["stat_code"])), cycle, start_key, end_key, quote(str(row["item_code1"])),
        ])
        try:
            payload = _json(url)
            data = payload.get("StatisticSearch", {}).get("row", [])
            frame = pd.DataFrame(data)
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["TIME"], format="%Y%m%d" if cycle == "D" else "%Y%m", errors="coerce")
            frame[alias] = pd.to_numeric(frame["DATA_VALUE"], errors="coerce")
            frames.append(frame[["date", alias]].dropna(subset=["date"]).drop_duplicates("date", keep="last"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ECOS %s 실패: %s", alias, exc)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date")
    path = paths.raw_dual / "ecos_macro.parquet"
    if path.exists() and not overwrite:
        out = merge_newer(pd.read_parquet(path), out, ["date"])
    atomic_parquet(out, path)
    return out


def collect_kosis(paths: ProjectPaths, overwrite: bool = False) -> pd.DataFrame:
    """configs/kosis_series.csv에 정확한 통계표 ID를 입력한 계열만 수집한다."""
    api_key = os.getenv("KOSIS_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("KOSIS_API_KEY가 없어 KOSIS 수집을 건너뜁니다.")
        return pd.DataFrame()
    cfg = pd.read_csv(paths.configs / "kosis_series.csv", dtype=str).fillna("")
    cfg = cfg.loc[pd.to_numeric(cfg["enabled"], errors="coerce").fillna(0).astype(int).eq(1)]
    frames: list[pd.DataFrame] = []
    for _, row in cfg.iterrows():
        params = {
            "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
            "orgId": row["org_id"], "tblId": row["tbl_id"], "itmId": row["itm_id"],
            "objL1": row["obj_l1"], "prdSe": row["prd_se"],
            "startPrdDe": row["start_prd_de"], "endPrdDe": row["end_prd_de"],
        }
        try:
            payload = _json("https://kosis.kr/openapi/Param/statisticsParameterData.do", params)
            frame = pd.DataFrame(payload if isinstance(payload, list) else [])
            if frame.empty:
                continue
            time_col = next((c for c in ["PRD_DE", "PRD_DE_NM"] if c in frame.columns), None)
            value_col = next((c for c in ["DT", "DTV"] if c in frame.columns), None)
            if not time_col or not value_col:
                continue
            frame["date"] = pd.to_datetime(frame[time_col].astype(str), errors="coerce")
            frame[str(row["alias"])] = pd.to_numeric(frame[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            frames.append(frame[["date", str(row["alias"])]].dropna(subset=["date"]).drop_duplicates("date", keep="last"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("KOSIS %s 실패: %s", row["alias"], exc)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date")
    path = paths.raw_dual / "kosis_macro.parquet"
    if path.exists() and not overwrite:
        out = merge_newer(pd.read_parquet(path), out, ["date"])
    atomic_parquet(out, path)
    return out
