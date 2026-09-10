from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ..config import ProjectPaths, get_paths, load_baskets
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker

LOGGER = logging.getLogger(__name__)

COLUMN_MAP = {
    "시가": "open", "고가": "high", "저가": "low", "종가": "close",
    "거래량": "volume", "거래대금": "trading_value", "등락률": "return_pct",
    "시가총액": "market_cap", "상장주식수": "listed_shares",
    "PER": "per", "PBR": "pbr", "EPS": "eps", "BPS": "bps", "DIV": "dividend_yield", "DPS": "dps",
    "외국인합계": "foreign", "외국인": "foreign", "기관합계": "institution", "기관": "institution",
    "개인": "individual", "기타법인": "other_corp", "기타외국인": "other_foreign", "합계": "total",
    "보유수량": "foreign_owned_shares", "지분율": "foreign_ownership_rate", "한도수량": "foreign_limit_shares",
    "공매도": "short_volume", "공매도량": "short_volume", "매수": "total_buy_volume", "비중": "short_ratio",
    "공매도잔고": "short_balance_shares", "잔고": "short_balance_shares", "잔고수량": "short_balance_shares",
    "공매도금액": "short_value", "잔고금액": "short_balance_value",
}


@dataclass
class CrawlRecord:
    ticker: str
    bucket: str
    source: str
    chunk_start: str
    chunk_end: str
    status: str
    rows: int
    non_null_ratio: float
    attempt_count: int
    elapsed_seconds: float
    error: str = ""
    path: str = ""


def _date8(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _year_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first
    while cursor <= last:
        chunk_end = min(pd.Timestamp(year=cursor.year, month=12, day=31), last)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=20), reraise=True)
def _call(fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    return fn()


def _normalize(frame: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date"])
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].reset_index()
    out = out.rename(columns={out.columns[0]: "date", **COLUMN_MAP})
    out.columns = [str(c).strip() for c in out.columns]
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns if c != "date"})
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return normalize_date(out).drop_duplicates("date", keep="last").sort_values("date")


def _non_null_ratio(frame: pd.DataFrame) -> float:
    values = frame.drop(columns=["date", "ticker", "bucket", "name"], errors="ignore")
    if values.empty:
        return 0.0
    return float(values.notna().to_numpy().mean())


def _merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if not f.empty]
    if not valid:
        return pd.DataFrame()
    out = valid[0]
    for frame in valid[1:]:
        duplicate = [c for c in frame.columns if c != "date" and c in out.columns]
        out = out.merge(frame.drop(columns=duplicate), on="date", how="outer")
    return out.sort_values("date").drop_duplicates("date", keep="last")


def _ticker_jobs(ticker: str, start8: str, end8: str):
    from pykrx import stock

    return [
        ("short_status", "fs_status_", lambda: stock.get_shorting_status_by_date(start8, end8, ticker)),
        ("short_volume", "fs_volume_", lambda: stock.get_shorting_volume_by_date(start8, end8, ticker)),
        ("short_balance", "fs_balance_", lambda: stock.get_shorting_balance_by_date(start8, end8, ticker)),
        ("investor_value", "ff_value_", lambda: stock.get_market_trading_value_by_date(start8, end8, ticker)),
        ("investor_volume", "ff_volume_", lambda: stock.get_market_trading_volume_by_date(start8, end8, ticker)),
        ("foreign_ownership", "ff_foreign_", lambda: stock.get_exhaustion_rates_of_foreign_investment(start8, end8, ticker)),
        ("fundamental", "fv_", lambda: stock.get_market_fundamental(start8, end8, ticker)),
        ("market_cap", "fv_", lambda: stock.get_market_cap(start8, end8, ticker)),
    ]


def collect_finance_ticker_data(
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    baskets = load_baskets(paths)
    root = paths.raw_dual / "finance11h"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    combined_tickers: list[pd.DataFrame] = []

    for _, row in baskets.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        source_frames: dict[str, list[pd.DataFrame]] = {}
        for chunk_start, chunk_end in _year_chunks(start, end):
            start8, end8 = _date8(chunk_start), _date8(chunk_end)
            for source, prefix, fn in _ticker_jobs(ticker, start8, end8):
                path = root / "ticker_sources" / f"ticker={ticker}" / f"source={source}" / f"{chunk_start.year}.parquet"
                if path.exists() and not overwrite:
                    frame = pd.read_parquet(path)
                    source_frames.setdefault(source, []).append(frame)
                    records.append(asdict(CrawlRecord(
                        ticker=ticker, bucket=str(row.get("bucket", "")), source=source,
                        chunk_start=str(chunk_start.date()), chunk_end=str(chunk_end.date()), status="cached",
                        rows=len(frame), non_null_ratio=_non_null_ratio(frame), attempt_count=0,
                        elapsed_seconds=0.0, path=str(path),
                    )))
                    continue
                started = time.perf_counter()
                try:
                    raw = _call(fn)
                    frame = _normalize(raw, prefix)
                    elapsed = time.perf_counter() - started
                    status = "success" if not frame.empty and _non_null_ratio(frame) >= 0.50 else "partial"
                    if frame.empty:
                        status = "empty"
                    if not frame.empty:
                        frame["ticker"] = ticker
                        frame["bucket"] = str(row.get("bucket", ""))
                        frame["name"] = str(row.get("name", ""))
                        atomic_parquet(frame, path)
                        source_frames.setdefault(source, []).append(frame)
                    records.append(asdict(CrawlRecord(
                        ticker=ticker, bucket=str(row.get("bucket", "")), source=source,
                        chunk_start=str(chunk_start.date()), chunk_end=str(chunk_end.date()), status=status,
                        rows=len(frame), non_null_ratio=_non_null_ratio(frame), attempt_count=1,
                        elapsed_seconds=elapsed, path=str(path) if not frame.empty else "",
                    )))
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.perf_counter() - started
                    records.append(asdict(CrawlRecord(
                        ticker=ticker, bucket=str(row.get("bucket", "")), source=source,
                        chunk_start=str(chunk_start.date()), chunk_end=str(chunk_end.date()), status="failed",
                        rows=0, non_null_ratio=0.0, attempt_count=3, elapsed_seconds=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    )))
                time.sleep(sleep_seconds)

        stitched_sources: list[pd.DataFrame] = []
        for source, frames in source_frames.items():
            if not frames:
                continue
            stitched = pd.concat(frames, ignore_index=True, sort=False)
            stitched = normalize_date(stitched).sort_values("date").drop_duplicates("date", keep="last")
            stitched_sources.append(stitched)
        merged = _merge_sources(stitched_sources)
        if not merged.empty:
            merged["ticker"] = ticker
            merged["bucket"] = str(row.get("bucket", ""))
            merged["name"] = str(row.get("name", ""))
            combined_tickers.append(merged)

    manifest = pd.DataFrame(records)
    atomic_csv(manifest, root / "finance_crawl_manifest.csv")
    combined = pd.concat(combined_tickers, ignore_index=True, sort=False) if combined_tickers else pd.DataFrame()
    if not combined.empty:
        combined = normalize_date(combined)
        combined["ticker"] = normalize_ticker(combined["ticker"])
        combined = combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
        atomic_parquet(combined, root / "finance_ticker_timeseries.parquet")
    return manifest


def collect_finance_market_data(paths: ProjectPaths, start: str, end: str, *, overwrite: bool = False) -> pd.DataFrame:
    from pykrx import stock

    root = paths.raw_dual / "finance11h"
    output = root / "finance_market_timeseries.parquet"
    if output.exists() and not overwrite:
        return pd.read_parquet(output)
    frames: list[pd.DataFrame] = []
    records: list[dict] = []
    for market in ("KOSPI", "KOSDAQ"):
        start8, end8 = _date8(start), _date8(end)
        jobs = [
            (f"{market}_flow_value", f"uf_{market.lower()}_", lambda market=market: stock.get_market_trading_value_by_date(start8, end8, market, detail=True)),
            (f"{market}_short_investor_value", f"us_{market.lower()}_", lambda market=market: stock.get_shorting_investor_value_by_date(start8, end8, market)),
            (f"{market}_short_investor_volume", f"usv_{market.lower()}_", lambda market=market: stock.get_shorting_investor_volume_by_date(start8, end8, market)),
        ]
        for source, prefix, fn in jobs:
            started = time.perf_counter()
            try:
                frame = _normalize(_call(fn), prefix)
                status = "success" if not frame.empty else "empty"
                if not frame.empty:
                    frames.append(frame)
                records.append({"source": source, "status": status, "rows": len(frame), "elapsed_seconds": time.perf_counter() - started, "error": ""})
            except Exception as exc:  # noqa: BLE001
                records.append({"source": source, "status": "failed", "rows": 0, "elapsed_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(0.2)
    out = _merge_sources(frames)
    if not out.empty:
        atomic_parquet(out, output)
    atomic_csv(pd.DataFrame(records), root / "finance_market_manifest.csv")
    return out


def verify_mandatory_short_data(paths: ProjectPaths, minimum_ticker_coverage: float = 0.80) -> dict:
    root = paths.raw_dual / "finance11h"
    path = root / "finance_ticker_timeseries.parquet"
    baskets = load_baskets(paths)
    if not path.exists():
        raise RuntimeError("필수 공매도 시계열이 없습니다. 04A_금융시계열_강제수집.py를 먼저 실행하세요.")
    df = pd.read_parquet(path)
    df["ticker"] = normalize_ticker(df["ticker"])
    short_cols = [c for c in df.columns if c.startswith("fs_") and ("short" in c or "ratio" in c or "balance" in c)]
    if not short_cols:
        raise RuntimeError("공매도 컬럼이 하나도 생성되지 않았습니다.")
    ticker_ok = []
    rows = []
    for ticker in baskets["ticker"]:
        block = df.loc[df["ticker"].eq(ticker), short_cols]
        valid_ratio = float(block.notna().to_numpy().mean()) if not block.empty else 0.0
        ok = len(block) >= 100 and valid_ratio >= 0.10
        ticker_ok.append(ok)
        rows.append({"ticker": ticker, "rows": len(block), "short_non_null_ratio": valid_ratio, "status": "ready" if ok else "insufficient"})
    coverage = float(np.mean(ticker_ok)) if ticker_ok else 0.0
    atomic_csv(pd.DataFrame(rows), root / "mandatory_short_coverage.csv")
    summary = {"ticker_count": len(ticker_ok), "ready_tickers": int(sum(ticker_ok)), "coverage": coverage, "minimum_required": minimum_ticker_coverage, "short_columns": short_cols}
    atomic_json(summary, root / "mandatory_short_summary.json")
    if coverage < minimum_ticker_coverage:
        raise RuntimeError(f"공매도 데이터 커버리지 부족: {coverage:.1%} < {minimum_ticker_coverage:.1%}")
    return summary


def run_finance_crawl(project: Path | None, start: str, end: str, *, overwrite: bool = False, sleep_seconds: float = 0.15, strict: bool = True) -> dict:
    paths = get_paths(project)
    paths.raw_dual.mkdir(parents=True, exist_ok=True)
    manifest = collect_finance_ticker_data(paths, start, end, overwrite=overwrite, sleep_seconds=sleep_seconds)
    market = collect_finance_market_data(paths, start, end, overwrite=overwrite)
    verification = verify_mandatory_short_data(paths) if strict else {}
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "start": start, "end": end,
        "manifest_rows": len(manifest),
        "success_or_cached": int(manifest["status"].isin(["success", "cached"]).sum()) if not manifest.empty else 0,
        "failed": int(manifest["status"].eq("failed").sum()) if not manifest.empty else 0,
        "market_rows": len(market),
        "mandatory_short": verification,
    }
    atomic_json(summary, paths.raw_dual / "finance11h" / "finance_crawl_summary.json")
    return summary
