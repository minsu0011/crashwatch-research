from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_parquet, normalize_ticker
from .dart import BASE_URL, _get, collect_corp_codes

LOGGER = logging.getLogger(__name__)
REPORT_CODES = {"11013": 45, "11012": 45, "11014": 45, "11011": 90}


def _number(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "N/A", "nan", "None"}:
        return np.nan
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return pd.to_numeric(text, errors="coerce")


def collect_ownership_events(
    paths: ProjectPaths,
    baskets: pd.DataFrame,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    path = paths.raw_dual / "dart_ownership_events.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("DART_API_KEY가 없어 지분 공시 수집을 건너뜁니다.")
        return pd.DataFrame()
    codes = collect_corp_codes(paths, api_key, overwrite=False)
    mapping = baskets.merge(codes[["corp_code", "stock_code"]], left_on="ticker", right_on="stock_code", how="left")
    rows: list[dict] = []
    for _, item in mapping.dropna(subset=["corp_code"]).iterrows():
        ticker, corp_code = str(item["ticker"]).zfill(6), str(item["corp_code"])
        for endpoint, event_type, change_fields in [
            ("majorstock.json", "major", ["stkqy_irds", "stkrt_irds"]),
            ("elestock.json", "insider", ["sp_stock_lmp_irds_cnt", "sp_stock_lmp_cnt"]),
        ]:
            try:
                payload = _get(f"{BASE_URL}/{endpoint}", {"crtfc_key": api_key, "corp_code": corp_code}).json()
                if payload.get("status") not in {"000", "013"}:
                    LOGGER.warning("DART %s %s 상태=%s", ticker, endpoint, payload.get("status"))
                    continue
                for row in payload.get("list", []):
                    change = next((_number(row.get(field)) for field in change_fields if pd.notna(_number(row.get(field)))), np.nan)
                    rows.append({
                        "date": row.get("rcept_dt"),
                        "ticker": ticker,
                        "event_type": event_type,
                        "change_shares": change,
                        "reporter": row.get("repror") or row.get("repror_nm"),
                        "receipt_no": row.get("rcept_no"),
                    })
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("DART ownership %s %s 실패: %s", ticker, endpoint, exc)
            time.sleep(sleep_seconds)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # OpenDART는 과거 YYYYMMDD와 현재 YYYY-MM-DD 응답이 혼재할 수 있다.
    out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce")
    out["ticker"] = normalize_ticker(out["ticker"])
    out = out.dropna(subset=["date"]).sort_values(["ticker", "date", "receipt_no"]).drop_duplicates(["event_type", "receipt_no"], keep="last")
    atomic_parquet(out, path)
    return out


def _account_key(name: str) -> str | None:
    text = str(name).replace(" ", "")
    rules = [
        ("revenue", ["매출액", "영업수익", "수익(매출액)"]),
        ("operating_income", ["영업이익", "영업이익(손실)"]),
        ("net_income", ["당기순이익", "당기순이익(손실)", "분기순이익"]),
        ("assets", ["자산총계"]),
        ("liabilities", ["부채총계"]),
        ("equity", ["자본총계"]),
        ("current_assets", ["유동자산"]),
        ("current_liabilities", ["유동부채"]),
        ("cash", ["현금및현금성자산"]),
        ("inventory", ["재고자산"]),
        ("receivables", ["매출채권", "매출채권및기타채권"]),
        ("operating_cash_flow", ["영업활동현금흐름", "영업활동으로인한현금흐름"]),
        ("interest_expense", ["이자비용"]),
    ]
    for key, names in rules:
        if any(candidate in text for candidate in names):
            return key
    return None


def collect_financial_quality(
    paths: ProjectPaths,
    baskets: pd.DataFrame,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    path = paths.raw_dual / "dart_financial_quality.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    api_key = os.getenv("DART_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("DART_API_KEY가 없어 재무 품질 수집을 건너뜁니다.")
        return pd.DataFrame()
    codes = collect_corp_codes(paths, api_key, overwrite=False)
    mapping = baskets.merge(codes[["corp_code", "stock_code"]], left_on="ticker", right_on="stock_code", how="left")
    corp_codes = mapping.dropna(subset=["corp_code"])["corp_code"].astype(str).tolist()
    ticker_map = dict(zip(mapping["corp_code"].astype(str), mapping["ticker"].astype(str).str.zfill(6)))
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    raw_rows: list[dict] = []
    for year in years:
        for report_code, lag_days in REPORT_CODES.items():
            for offset in range(0, len(corp_codes), 100):
                batch = corp_codes[offset: offset + 100]
                try:
                    payload = _get(f"{BASE_URL}/fnlttMultiAcnt.json", {
                        "crtfc_key": api_key,
                        "corp_code": ",".join(batch),
                        "bsns_year": str(year),
                        "reprt_code": report_code,
                    }).json()
                    if payload.get("status") not in {"000", "013"}:
                        LOGGER.warning("DART financial %s %s 상태=%s", year, report_code, payload.get("status"))
                        continue
                    for row in payload.get("list", []):
                        key = _account_key(row.get("account_nm", ""))
                        if key is None:
                            continue
                        amount = _number(row.get("thstrm_add_amount"))
                        if not np.isfinite(amount):
                            amount = _number(row.get("thstrm_amount"))
                        period_end = pd.to_datetime(row.get("thstrm_dt"), errors="coerce")
                        if pd.isna(period_end):
                            month_day = {"11013": "03-31", "11012": "06-30", "11014": "09-30", "11011": "12-31"}[report_code]
                            period_end = pd.Timestamp(f"{year}-{month_day}")
                        raw_rows.append({
                            "corp_code": str(row.get("corp_code", "")),
                            "ticker": ticker_map.get(str(row.get("corp_code", "")), str(row.get("stock_code", "")).zfill(6)),
                            "report_code": report_code,
                            "period_end": period_end,
                            "available_from": period_end + pd.Timedelta(days=lag_days),
                            "account": key,
                            "amount": amount,
                            "fs_div": row.get("fs_div"),
                        })
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("DART financial %s/%s 실패: %s", year, report_code, exc)
                time.sleep(sleep_seconds)
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return raw
    raw = raw.sort_values(["ticker", "available_from", "account", "fs_div"])
    raw["_priority"] = raw["fs_div"].eq("CFS").astype(int)
    raw = raw.sort_values("_priority").drop_duplicates(["ticker", "available_from", "account"], keep="last")
    wide = raw.pivot_table(index=["ticker", "available_from", "period_end", "report_code"], columns="account", values="amount", aggfunc="last").reset_index()
    for col in ["assets", "liabilities", "equity", "current_assets", "current_liabilities", "cash", "inventory", "receivables", "revenue", "operating_income", "net_income", "operating_cash_flow", "interest_expense"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.sort_values(["ticker", "available_from"])
    wide["t_quality_roa"] = wide["net_income"] / wide["assets"].replace(0, np.nan)
    wide["t_quality_operating_margin"] = wide["operating_income"] / wide["revenue"].replace(0, np.nan)
    wide["t_quality_leverage"] = wide["liabilities"] / wide["assets"].replace(0, np.nan)
    wide["t_quality_current_ratio"] = wide["current_assets"] / wide["current_liabilities"].replace(0, np.nan)
    wide["t_quality_cash_ratio"] = wide["cash"] / wide["current_liabilities"].replace(0, np.nan)
    wide["t_quality_cfo_to_assets"] = wide["operating_cash_flow"] / wide["assets"].replace(0, np.nan)
    wide["t_quality_accruals"] = (wide["net_income"] - wide["operating_cash_flow"]) / wide["assets"].replace(0, np.nan)
    wide["t_quality_interest_coverage"] = wide["operating_income"] / wide["interest_expense"].replace(0, np.nan)
    for base in ["revenue", "inventory", "receivables", "liabilities", "operating_cash_flow"]:
        wide[f"t_quality_{base}_growth"] = wide.groupby("ticker")[base].pct_change(fill_method=None)
    keep = ["ticker", "available_from", "period_end", "report_code"] + [c for c in wide.columns if c.startswith("t_quality_")]
    out = wide[keep].replace([np.inf, -np.inf], np.nan)
    atomic_parquet(out, path)
    return out
