from __future__ import annotations

import io
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_ticker
from .common import SourceRecord, append_manifest, now_iso, read_enabled_tickers, retry_call, sanitize_secret_error, sha256_file, sleep_with_jitter

LOGGER = logging.getLogger(__name__)
DART_BASE = "https://opendart.fss.or.kr/api"
REPORT_CODES = {"annual": "11011", "q1": "11013", "half": "11012", "q3": "11014"}
REPORT_NAME_PATTERNS = {
    "annual": re.compile(r"사업보고서"),
    "q1": re.compile(r"분기보고서.*\(.*03"),
    "half": re.compile(r"반기보고서"),
    "q3": re.compile(r"분기보고서.*\(.*09"),
}


class DartDocumentUnavailableError(RuntimeError):
    """Permanent OpenDART 014 response; retrying cannot create the file."""


def _dart_key() -> str:
    return (os.getenv("OPENDART_API_KEY") or os.getenv("DART_API_KEY") or "").strip()


def _request_json(session: requests.Session, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(f"{DART_BASE}/{endpoint}", params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    status = str(payload.get("status", "000"))
    if status not in {"000", "013"}:
        raise RuntimeError(f"OpenDART {status}: {payload.get('message', '')}")
    return payload


def _corp_codes(session: requests.Session, key: str) -> pd.DataFrame:
    response = session.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        xml_name = next(name for name in names if name.lower().endswith(".xml"))
        xml_data = archive.read(xml_name)
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_data)
    rows: list[dict[str, str]] = []
    for item in root.findall("list"):
        rows.append({child.tag: (child.text or "").strip() for child in item})
    frame = pd.DataFrame(rows)
    if "stock_code" in frame:
        frame["stock_code"] = normalize_ticker(frame["stock_code"])
    return frame


def _list_reports(
    session: requests.Session,
    key: str,
    corp_code: str,
    begin: str,
    end: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": pd.Timestamp(begin).strftime("%Y%m%d"),
            "end_de": pd.Timestamp(end).strftime("%Y%m%d"),
            "pblntf_ty": "A",
            "page_no": page,
            "page_count": 100,
        }
        payload = retry_call(lambda: _request_json(session, "list.json", params), attempts=4, base_delay=1.0)
        rows = payload.get("list", []) or []
        all_rows.extend(rows)
        total_page = int(payload.get("total_page", 1) or 1)
        if page >= total_page:
            break
        page += 1
        sleep_with_jitter(sleep_seconds)
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        return frame
    frame["rcept_dt"] = pd.to_datetime(frame["rcept_dt"], format="%Y%m%d", errors="coerce")
    frame["stock_code"] = normalize_ticker(frame["stock_code"])
    frame["is_correction"] = frame["report_nm"].astype(str).str.contains("정정|첨부정정|기재정정", regex=True)
    frame["regular_report_type"] = frame["report_nm"].map(_report_type_from_name)
    frame["report_period_end"] = frame["report_nm"].astype(str).str.extract(r"\((\d{4}\.\d{2})\)", expand=False)
    return frame


def _report_type_from_name(name: Any) -> str:
    text = str(name)
    if "사업보고서" in text:
        return "annual"
    if "반기보고서" in text:
        return "half"
    if "분기보고서" in text:
        match = re.search(r"\((\d{4})\.(\d{2})\)", text)
        if match and match.group(2) == "03":
            return "q1"
        if match and match.group(2) == "09":
            return "q3"
        return "quarter"
    return ""




def _add_correction_chain(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    out = history.copy()
    normalized_name = (
        out["report_nm"].astype(str)
        .str.replace(r"^\s*\[?정정\]?\s*", "", regex=True)
        .str.replace(r"^\s*첨부정정\s*", "", regex=True)
        .str.replace(r"^\s*기재정정\s*", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out["correction_chain_key"] = (
        out.get("ticker", pd.Series("", index=out.index)).astype(str).str.zfill(6)
        + "|" + out["regular_report_type"].fillna("").astype(str)
        + "|" + out["report_period_end"].fillna("").astype(str)
        + "|" + normalized_name
    )
    out["correction_sequence"] = 0
    out["previous_receipt_no"] = ""
    out["supersedes_receipt_no"] = ""
    out["is_latest_correction"] = False
    for _, indexes in out.sort_values(["rcept_dt", "rcept_no"]).groupby("correction_chain_key", sort=False).groups.items():
        ordered = list(indexes)
        previous = ""
        for sequence, index in enumerate(ordered):
            receipt = str(out.at[index, "rcept_no"])
            out.at[index, "correction_sequence"] = sequence
            out.at[index, "previous_receipt_no"] = previous
            out.at[index, "supersedes_receipt_no"] = previous if sequence > 0 else ""
            previous = receipt
        if ordered:
            out.at[ordered[-1], "is_latest_correction"] = True
    return out

def _download_document(session: requests.Session, key: str, rcept_no: str, path: Path) -> None:
    response = session.get(
        f"{DART_BASE}/document.xml",
        params={"crtfc_key": key, "rcept_no": rcept_no},
        timeout=180,
    )
    response.raise_for_status()
    if response.content.lstrip().startswith(b"<"):
        text = response.content.decode("utf-8", errors="ignore")
        if "<status>014</status>" in text:
            raise DartDocumentUnavailableError(text[:500])
        if "<status>" in text and "000" not in text:
            raise RuntimeError(text[:500])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(response.content)
    tmp.replace(path)


def _financial_rows(
    session: requests.Session,
    key: str,
    corp_code: str,
    business_year: int,
    report_code: str,
) -> pd.DataFrame:
    params = {
        "crtfc_key": key,
        "corp_code": corp_code,
        "bsns_year": str(business_year),
        "reprt_code": report_code,
        "fs_div": "CFS",
    }
    payload = retry_call(lambda: _request_json(session, "fnlttSinglAcntAll.json", params), attempts=4, base_delay=1.0)
    rows = payload.get("list", []) or []
    if not rows:
        params["fs_div"] = "OFS"
        payload = retry_call(lambda: _request_json(session, "fnlttSinglAcntAll.json", params), attempts=4, base_delay=1.0)
        rows = payload.get("list", []) or []
    return pd.DataFrame(rows)


def _attach_availability(financial: pd.DataFrame, history: pd.DataFrame, report_type: str, year: int) -> pd.DataFrame:
    out = financial.copy()
    reports = history.loc[
        history["regular_report_type"].eq(report_type)
        & history["report_nm"].astype(str).str.contains(str(year), regex=False),
    ].sort_values("rcept_dt")
    if reports.empty:
        # report_nm often contains period year rather than business-year text after corrections; fallback by date.
        reports = history.loc[
            history["regular_report_type"].eq(report_type)
            & history["rcept_dt"].dt.year.isin([year, year + 1]),
        ].sort_values("rcept_dt")
    first_date = reports["rcept_dt"].min() if not reports.empty else pd.NaT
    latest_date = reports["rcept_dt"].max() if not reports.empty else pd.NaT
    latest_receipt = str(reports.iloc[-1]["rcept_no"]) if not reports.empty else ""
    out["report_type"] = report_type
    out["business_year"] = year
    out["first_filing_date"] = first_date
    # Values from fnlttSinglAcntAll are latest-at-download; using latest correction date is conservative and prevents lookahead.
    out["available_from"] = latest_date
    out["latest_receipt_no_at_download"] = latest_receipt
    out["point_in_time_quality"] = "latest_value_conservative_latest_correction_date"
    return out


def collect_dart_point_in_time(
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    download_documents: bool = True,
    sleep_seconds: float = 0.12,
) -> dict:
    key = _dart_key()
    if not key:
        raise RuntimeError(
            "OPENDART_API_KEY가 없습니다. 시점 보존 DART 자료에는 사용자 인증키가 필요합니다. "
            "무료 발급 후 .env.data.local에 입력하세요."
        )
    root = paths.raw_dual / "required_data_v3" / "dart_point_in_time"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dart_manifest.csv"
    session = requests.Session()
    session.headers.update({"User-Agent": "CrashWatch-AI-Research/3.0"})
    baskets = read_enabled_tickers(paths)
    records: list[SourceRecord] = []
    timer_all = time.perf_counter()

    corp_path = root / "corp_codes.parquet"
    if corp_path.exists() and not overwrite:
        corp = pd.read_parquet(corp_path)
    else:
        corp = retry_call(lambda: _corp_codes(session, key), attempts=4, base_delay=2.0)
        atomic_parquet(corp, corp_path)
        atomic_csv(corp, root / "corp_codes.csv")
    ticker_corp = baskets.merge(corp[["corp_code", "corp_name", "stock_code"]], left_on="ticker", right_on="stock_code", how="left")
    atomic_csv(ticker_corp, root / "ticker_corp_mapping.csv")

    history_frames: list[pd.DataFrame] = []
    for row in ticker_corp.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        corp_code = str(getattr(row, "corp_code", "") or "")
        if not corp_code or corp_code == "nan":
            records.append(SourceRecord("OpenDART", "report_history", ticker, "failed", 0, now_iso(), now_iso(), 0.0, "", "corp_code 매핑 실패", ""))
            continue
        path = root / "report_history" / f"{ticker}.parquet"
        started = now_iso()
        timer = time.perf_counter()
        try:
            if path.exists() and not overwrite:
                history = pd.read_parquet(path)
                status = "cached"
            else:
                history = _list_reports(session, key, corp_code, start, end, sleep_seconds)
                if not history.empty:
                    history["ticker"] = ticker
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_parquet(history, path)
                    atomic_csv(history, path.with_suffix(".csv"))
                status = "success" if len(history) else "empty"
            if not history.empty:
                history_frames.append(history)
            record = SourceRecord(
                "OpenDART", "report_history", ticker, status, len(history), started, now_iso(), time.perf_counter() - timer,
                str(path.relative_to(paths.project)) if path.exists() else "", "", sha256_file(path) if path.exists() else "",
            )
        except Exception as exc:  # noqa: BLE001
            record = SourceRecord("OpenDART", "report_history", ticker, "failed", 0, started, now_iso(), time.perf_counter() - timer, "", sanitize_secret_error(str(exc)), "")
        records.append(record)
        append_manifest([record], manifest_path)
        sleep_with_jitter(sleep_seconds)

    history_all = pd.concat(history_frames, ignore_index=True, sort=False) if history_frames else pd.DataFrame()
    if not history_all.empty:
        history_all = history_all.sort_values(["ticker", "rcept_dt", "rcept_no"]).drop_duplicates(["ticker", "rcept_no"], keep="last")
        history_all = _add_correction_chain(history_all)
        atomic_csv(history_all, root / "regular_report_history_48_tickers.csv")
        atomic_parquet(history_all, root / "regular_report_history_48_tickers.parquet")

    if download_documents and not history_all.empty:
        regular = history_all.loc[history_all["regular_report_type"].isin(REPORT_CODES)]
        for report in regular.itertuples(index=False):
            ticker = str(report.ticker).zfill(6)
            rcept_no = str(report.rcept_no)
            path = root / "original_documents" / f"ticker={ticker}" / f"{rcept_no}.zip"
            started = now_iso()
            timer = time.perf_counter()
            if path.exists() and not overwrite:
                record = SourceRecord("OpenDART", "original_document", f"{ticker}:{rcept_no}", "cached", 1, started, now_iso(), time.perf_counter() - timer, str(path.relative_to(paths.project)), "", sha256_file(path))
            else:
                try:
                    retry_call(
                        lambda: _download_document(session, key, rcept_no, path),
                        attempts=4,
                        base_delay=2.0,
                        retry_if=lambda exc: not isinstance(exc, DartDocumentUnavailableError),
                    )
                    record = SourceRecord("OpenDART", "original_document", f"{ticker}:{rcept_no}", "success", 1, started, now_iso(), time.perf_counter() - timer, str(path.relative_to(paths.project)), "", sha256_file(path))
                except Exception as exc:  # noqa: BLE001
                    record = SourceRecord("OpenDART", "original_document", f"{ticker}:{rcept_no}", "failed", 0, started, now_iso(), time.perf_counter() - timer, "", sanitize_secret_error(str(exc)), "")
            records.append(record)
            append_manifest([record], manifest_path)
            sleep_with_jitter(sleep_seconds)

    financial_frames: list[pd.DataFrame] = []
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    for row in ticker_corp.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        corp_code = str(getattr(row, "corp_code", "") or "")
        if not corp_code or corp_code == "nan":
            continue
        ticker_history = history_all.loc[history_all["ticker"].eq(ticker)] if not history_all.empty else pd.DataFrame()
        for year in range(start_year, end_year + 1):
            for report_type, report_code in REPORT_CODES.items():
                # Short partition names avoid crossing Windows MAX_PATH when
                # atomic_parquet appends its temporary suffix.
                path = root / "financial_pit" / f"t={ticker}" / f"y={year}" / f"{report_type}.parquet"
                started = now_iso()
                timer = time.perf_counter()
                if path.exists() and not overwrite:
                    frame = pd.read_parquet(path)
                    status = "cached"
                else:
                    try:
                        frame = _financial_rows(session, key, corp_code, year, report_code)
                        if not frame.empty:
                            frame["ticker"] = ticker
                            frame["corp_code"] = corp_code
                            frame = _attach_availability(frame, ticker_history, report_type, year)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            atomic_parquet(frame, path)
                            atomic_csv(frame, path.with_suffix(".csv"))
                        status = "success" if len(frame) else "empty"
                    except Exception as exc:  # noqa: BLE001
                        record = SourceRecord("OpenDART", "financial_statement", f"{ticker}:{year}:{report_type}", "failed", 0, started, now_iso(), time.perf_counter() - timer, "", sanitize_secret_error(str(exc)), "")
                        records.append(record)
                        append_manifest([record], manifest_path)
                        sleep_with_jitter(sleep_seconds)
                        continue
                if not frame.empty:
                    financial_frames.append(frame)
                record = SourceRecord(
                    "OpenDART", "financial_statement", f"{ticker}:{year}:{report_type}", status, len(frame), started,
                    now_iso(), time.perf_counter() - timer, str(path.relative_to(paths.project)) if path.exists() else "", "", sha256_file(path) if path.exists() else "",
                )
                records.append(record)
                append_manifest([record], manifest_path)
                sleep_with_jitter(sleep_seconds)

    financial_all = pd.concat(financial_frames, ignore_index=True, sort=False) if financial_frames else pd.DataFrame()
    if not financial_all.empty:
        financial_all["available_from"] = pd.to_datetime(financial_all["available_from"], errors="coerce")
        atomic_csv(financial_all, root / "financial_statements_conservative_pit_48_tickers.csv")
        atomic_parquet(financial_all, root / "financial_statements_conservative_pit_48_tickers.parquet")

    coverage = []
    for ticker in baskets["ticker"]:
        hist = history_all.loc[history_all.get("ticker", pd.Series(dtype=str)).eq(ticker)] if not history_all.empty else pd.DataFrame()
        fin = financial_all.loc[financial_all.get("ticker", pd.Series(dtype=str)).eq(ticker)] if not financial_all.empty else pd.DataFrame()
        coverage.append({
            "ticker": ticker,
            "report_history_rows": len(hist),
            "correction_rows": int(hist.get("is_correction", pd.Series(dtype=bool)).fillna(False).sum()) if len(hist) else 0,
            "financial_rows": len(fin),
            "original_document_count": len(list((root / "original_documents" / f"ticker={ticker}").glob("*.zip"))),
            "status": "ready" if len(hist) >= 4 and len(fin) > 0 else "insufficient",
        })
    coverage_df = pd.DataFrame(coverage)
    atomic_csv(coverage_df, root / "dart_pit_coverage_48_tickers.csv")
    summary = {
        "completed_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - timer_all,
        "ticker_count": len(baskets),
        "ready_tickers": int(coverage_df["status"].eq("ready").sum()),
        "history_rows": len(history_all),
        "financial_rows": len(financial_all),
        "download_documents": download_documents,
        "point_in_time_note": "원문 ZIP은 접수번호별 실제 버전 보존. 구조화 재무값은 최신값을 최신 정정 접수일 이후에만 사용하도록 보수적으로 매핑.",
    }
    atomic_json(summary, root / "dart_pit_summary.json")
    return summary
