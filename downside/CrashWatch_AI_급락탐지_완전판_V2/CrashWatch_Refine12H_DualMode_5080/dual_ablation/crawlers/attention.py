from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ProjectPaths
from ..io_utils import atomic_parquet, normalize_ticker

LOGGER = logging.getLogger(__name__)
URL = "https://openapi.naver.com/v1/datalab/search"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def _post(headers: dict[str, str], payload: dict) -> dict:
    response = requests.post(URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def collect_naver_attention(
    paths: ProjectPaths,
    baskets: pd.DataFrame,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """Collect Naver DataLab relative search ratios.

    DataLab normalizes ratios within each request interval. Consequently this dataset is
    exploratory and is marked revised_scale_flag=1. It must not be enabled in strict
    point-in-time production backtests without a daily rolling-window archive.
    """
    path = paths.raw_dual / "naver_attention.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    client_id = os.getenv("NAVER_CLIENT_ID", "").strip()
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        LOGGER.warning("NAVER_CLIENT_ID/SECRET가 없어 검색 관심도 수집을 건너뜁니다.")
        return pd.DataFrame()

    config_path = paths.configs / "naver_attention_keywords.csv"
    if config_path.exists():
        cfg = pd.read_csv(config_path, dtype={"ticker": str})
        cfg["ticker"] = normalize_ticker(cfg["ticker"])
        cfg = cfg.loc[pd.to_numeric(cfg.get("enabled", 1), errors="coerce").fillna(1).astype(int).eq(1)]
    else:
        cfg = baskets[["ticker", "name", "bucket"]].copy()
        cfg["keywords"] = cfg["name"]
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }
    rows: list[dict] = []
    records = cfg.to_dict("records")
    for offset in range(0, len(records), 5):
        batch = records[offset: offset + 5]
        groups = []
        mapping = {}
        for item in batch:
            ticker = str(item["ticker"]).zfill(6)
            keywords = [x.strip() for x in str(item.get("keywords", item.get("name", ticker))).split("|") if x.strip()]
            group_name = ticker
            groups.append({"groupName": group_name, "keywords": keywords[:20]})
            mapping[group_name] = item
        payload = {
            "startDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "endDate": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "timeUnit": "date",
            "keywordGroups": groups,
        }
        result = _post(headers, payload)
        for series in result.get("results", []):
            item = mapping.get(series.get("title"), {})
            for point in series.get("data", []):
                rows.append({
                    "date": point.get("period"),
                    "ticker": item.get("ticker", series.get("title")),
                    "name": item.get("name", ""),
                    "bucket": item.get("bucket", ""),
                    "ratio": point.get("ratio"),
                    "revised_scale_flag": 1,
                })
        time.sleep(sleep_seconds)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = normalize_ticker(out["ticker"])
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    atomic_parquet(out, path)
    return out
