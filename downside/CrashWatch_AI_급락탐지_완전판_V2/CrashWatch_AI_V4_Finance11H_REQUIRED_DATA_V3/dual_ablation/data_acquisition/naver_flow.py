from __future__ import annotations

import logging
import time
from io import StringIO

import pandas as pd
import requests

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_ticker
from .common import SourceRecord, append_manifest, now_iso, read_enabled_tickers, retry_call, sanitize_secret_error, sha256_file, sleep_with_jitter

LOGGER = logging.getLogger(__name__)
NAVER_URL = "https://finance.naver.com/item/frgn.naver"


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(str(x).strip() for x in col if str(x) != "nan").strip("_") for col in out.columns]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_main_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), encoding="euc-kr")
    candidates: list[pd.DataFrame] = []
    for table in tables:
        frame = _flatten_columns(table)
        column_text = " ".join(frame.columns)
        if "날짜" in column_text and ("기관" in column_text or "외국인" in column_text):
            candidates.append(frame)
    if not candidates:
        raise RuntimeError("네이버 투자자별 매매동향 표를 찾지 못했습니다.")
    return max(candidates, key=len)


def _normalize_naver(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = _flatten_columns(frame)
    rename: dict[str, str] = {}
    for column in out.columns:
        text = str(column)
        if "날짜" in text:
            rename[column] = "date"
        elif "종가" in text:
            rename[column] = "close"
        elif "거래량" in text:
            rename[column] = "volume"
        elif "기관" in text and "순매매" in text:
            rename[column] = "institution_net_volume"
        elif "외국인" in text and "순매매" in text:
            rename[column] = "foreign_net_volume"
        elif "보유주수" in text or "보유수량" in text:
            rename[column] = "foreign_owned_shares"
        elif "보유율" in text:
            rename[column] = "foreign_ownership_rate"
    out = out.rename(columns=rename)
    if "date" not in out:
        return pd.DataFrame()
    keep = [c for c in ["date", "close", "volume", "institution_net_volume", "foreign_net_volume", "foreign_owned_shares", "foreign_ownership_rate"] if c in out]
    out = out[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[out["date"].notna()]
    for column in out.columns:
        if column != "date":
            out[column] = pd.to_numeric(out[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")
    if "foreign_ownership_rate" in out:
        values = out["foreign_ownership_rate"]
        if values.dropna().abs().quantile(0.95) > 1.5:
            out["foreign_ownership_rate"] = values / 100.0
    out["ticker"] = ticker
    out["source"] = "naver_finance_html_fallback"
    return out.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")


def collect_naver_flow_fallback(
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    max_pages: int = 250,
    sleep_seconds: float = 0.35,
) -> dict:
    baskets = read_enabled_tickers(paths)
    root = paths.raw_dual / "required_data_v3" / "naver_flow_fallback"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "naver_flow_manifest.csv"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/144 Safari/537.36",
        "Referer": "https://finance.naver.com/",
    })
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    records: list[SourceRecord] = []
    all_frames: list[pd.DataFrame] = []
    timer_all = time.perf_counter()

    for ticker in baskets["ticker"]:
        ticker_path = root / "ticker" / f"{ticker}.parquet"
        started = now_iso()
        timer = time.perf_counter()
        if ticker_path.exists() and not overwrite:
            frame = pd.read_parquet(ticker_path)
            status = "cached"
        else:
            pages: list[pd.DataFrame] = []
            try:
                for page in range(1, max_pages + 1):
                    def request_page() -> str:
                        response = session.get(NAVER_URL, params={"code": ticker, "page": page}, timeout=60)
                        response.raise_for_status()
                        response.encoding = "euc-kr"
                        return response.text

                    html = retry_call(request_page, attempts=4, base_delay=1.0)
                    page_frame = _normalize_naver(_find_main_table(html), ticker)
                    if page_frame.empty:
                        break
                    pages.append(page_frame)
                    oldest = page_frame["date"].min()
                    if oldest <= start_ts:
                        break
                    sleep_with_jitter(sleep_seconds)
                frame = pd.concat(pages, ignore_index=True, sort=False) if pages else pd.DataFrame()
                if not frame.empty:
                    frame = frame.loc[frame["date"].between(start_ts, end_ts)]
                    frame = frame.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
                    ticker_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_parquet(frame, ticker_path)
                    atomic_csv(frame, ticker_path.with_suffix(".csv"))
                status = "success" if len(frame) else "empty"
            except Exception as exc:  # noqa: BLE001
                record = SourceRecord("Naver Finance", "investor_foreign_fallback", ticker, "failed", 0, started, now_iso(), time.perf_counter() - timer, "", sanitize_secret_error(str(exc)), "")
                records.append(record)
                append_manifest([record], manifest_path)
                continue
        if not frame.empty:
            all_frames.append(frame)
        record = SourceRecord(
            "Naver Finance", "investor_foreign_fallback", ticker, status, len(frame), started, now_iso(), time.perf_counter() - timer,
            str(ticker_path.relative_to(paths.project)) if ticker_path.exists() else "", "", sha256_file(ticker_path) if ticker_path.exists() else "",
        )
        records.append(record)
        append_manifest([record], manifest_path)
        sleep_with_jitter(sleep_seconds)

    combined = pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    if not combined.empty:
        combined["ticker"] = normalize_ticker(combined["ticker"])
        combined = combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        atomic_csv(combined, root / "naver_investor_foreign_48_tickers.csv")
        atomic_parquet(combined, root / "naver_investor_foreign_48_tickers.parquet")
    coverage = []
    for ticker in baskets["ticker"]:
        block = combined.loc[combined.get("ticker", pd.Series(dtype=str)).eq(ticker)] if not combined.empty else pd.DataFrame()
        coverage.append({
            "ticker": ticker,
            "rows": len(block),
            "start_date": str(block["date"].min().date()) if len(block) else "",
            "end_date": str(block["date"].max().date()) if len(block) else "",
            "status": "ready" if len(block) >= 500 else "insufficient",
        })
    coverage_df = pd.DataFrame(coverage)
    atomic_csv(coverage_df, root / "naver_flow_coverage_48_tickers.csv")
    summary = {
        "completed_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - timer_all,
        "rows": len(combined),
        "ready_tickers": int(coverage_df["status"].eq("ready").sum()),
        "ticker_count": len(baskets),
        "usage": "Fallback only. KRX investor flow is preferred; Naver HTML structure can change.",
    }
    atomic_json(summary, root / "naver_flow_summary.json")
    return summary
