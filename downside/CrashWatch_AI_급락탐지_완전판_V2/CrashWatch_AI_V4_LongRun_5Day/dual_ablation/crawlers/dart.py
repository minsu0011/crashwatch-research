from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_parquet

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://opendart.fss.or.kr/api"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
def _get(url: str, params: dict | None = None) -> requests.Response:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response


def collect_corp_codes(paths: ProjectPaths, api_key: str, overwrite: bool = False) -> pd.DataFrame:
    path = paths.raw_dual / "dart_corp_codes.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    content = _get(f"{BASE_URL}/corpCode.xml", {"crtfc_key": api_key}).content
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml = zf.read(zf.namelist()[0])
    root = ElementTree.fromstring(xml)
    rows = []
    for item in root.findall("list"):
        rows.append({child.tag: child.text for child in item})
    out = pd.DataFrame(rows)
    if "stock_code" in out.columns:
        out["stock_code"] = out["stock_code"].fillna("").astype(str).str.zfill(6)
    atomic_parquet(out, path)
    return out


def classify_report(report_name: str) -> tuple[str, int]:
    text = str(report_name)
    rules = [
        ("capital_change", 5, ["유상증자", "무상증자", "감자", "전환사채", "신주인수권"]),
        ("contract_order", 4, ["단일판매", "공급계약", "수주", "계약체결"]),
        ("governance", 4, ["최대주주", "경영권", "합병", "분할", "영업양수도"]),
        ("buyback_dividend", 3, ["자기주식", "현금배당", "주식배당", "소각"]),
        ("risk_legal", 5, ["소송", "횡령", "배임", "부도", "회생", "상장폐지"]),
        ("earnings", 2, ["영업실적", "매출액", "손익구조", "잠정실적"]),
        ("clinical_license", 4, ["임상", "품목허가", "기술이전", "특허"]),
        ("facility_investment", 3, ["신규시설", "시설투자", "증설"]),
    ]
    for category, severity, keywords in rules:
        if any(word in text for word in keywords):
            return category, severity
    return "other", 1


def collect_disclosures(paths: ProjectPaths, baskets: pd.DataFrame, start: str, end: str, *, overwrite: bool = False, sleep_seconds: float = 0.2) -> pd.DataFrame:
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("DART_API_KEY가 없어 DART 수집을 건너뜁니다.")
        return pd.DataFrame()
    corp_codes = collect_corp_codes(paths, api_key, overwrite=overwrite)
    mapping = baskets.merge(corp_codes[["corp_code", "corp_name", "stock_code"]], left_on="ticker", right_on="stock_code", how="left")
    rows: list[dict] = []
    for _, row in mapping.dropna(subset=["corp_code"]).iterrows():
        ticker = str(row["ticker"]).zfill(6)
        page = 1
        while True:
            payload = _get(f"{BASE_URL}/list.json", {
                "crtfc_key": api_key, "corp_code": row["corp_code"],
                "bgn_de": pd.Timestamp(start).strftime("%Y%m%d"), "end_de": pd.Timestamp(end).strftime("%Y%m%d"),
                "page_no": page, "page_count": 100,
            }).json()
            status = payload.get("status")
            if status == "013":
                break
            if status != "000":
                LOGGER.warning("DART %s 상태=%s 메시지=%s", ticker, status, payload.get("message"))
                break
            items = payload.get("list", [])
            for item in items:
                category, severity = classify_report(item.get("report_nm", ""))
                item = dict(item)
                item.update({"ticker": ticker, "bucket": row.get("bucket", ""), "event_category": category, "event_severity": severity})
                rows.append(item)
            total_page = int(payload.get("total_page", page))
            if page >= total_page:
                break
            page += 1
            time.sleep(sleep_seconds)
        time.sleep(sleep_seconds)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rcept_dt"] = pd.to_datetime(out["rcept_dt"], errors="coerce")
    out["date"] = out["rcept_dt"]
    out = out.sort_values(["date", "ticker", "rcept_no"]).drop_duplicates(["rcept_no"], keep="last")
    atomic_parquet(out, paths.raw_dual / "dart_disclosures.parquet")
    atomic_csv(mapping[[c for c in ["bucket", "ticker", "name", "corp_code", "corp_name"] if c in mapping.columns]], paths.raw_dual / "dart_corp_mapping.csv")
    return out
