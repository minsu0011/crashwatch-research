from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_ticker
from .common import (
    SourceRecord,
    append_manifest,
    date_chunks,
    now_iso,
    read_enabled_tickers,
    retry_call,
    sanitize_secret_error,
    sha256_file,
    sleep_with_jitter,
)

LOGGER = logging.getLogger(__name__)
DATA_GO_BASES = [
    "https://apis.data.go.kr/1160100/service/GetCMStckLnbInfoService",
    "https://apis.data.go.kr/1160100/service/GetStckLendInfoService",
]
ENDPOINTS = ("getStckLnbDetail", "getStckLnbProgress", "getStckLnbInvpnDetail")
CORE_COLUMNS = [
    "lending_contract_shares",
    "lending_repayment_shares",
    "lending_balance_shares",
    "lending_balance_value",
]


def _service_key() -> str:
    value = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip().strip('"').strip("'")
    return unquote(value)


def _extract_items(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    response = payload.get("response", payload)
    header = response.get("header", {}) if isinstance(response, dict) else {}
    result_code = str(header.get("resultCode", "00"))
    if result_code not in {"00", "0", "NORMAL_SERVICE"}:
        raise RuntimeError(f"공공데이터 API 오류 {result_code}: {header.get('resultMsg', '')}")
    body = response.get("body", {}) if isinstance(response, dict) else {}
    total = int(body.get("totalCount", 0) or 0) if isinstance(body, dict) else 0
    items = body.get("items", {}) if isinstance(body, dict) else {}
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    return ([item for item in items if isinstance(item, dict)] if isinstance(items, list) else []), total


def _request_json(session: requests.Session, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for base in DATA_GO_BASES:
        try:
            response = session.get(f"{base}/{endpoint}", params=params, timeout=90)
            response.raise_for_status()
            text = response.text.lstrip()
            if text.startswith("<"):
                try:
                    import xmltodict
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"XML 응답 수신. xmltodict 설치 필요: {text[:300]}") from exc
                return xmltodict.parse(response.text)
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def _normalize_lending(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    date_candidates = ["basDt", "trdDd", "date", "기준일자", "거래일자"]
    ticker_candidates = ["stckItmsCd", "srtnCd", "ticker", "종목코드"]
    rename = {
        "basDt": "date",
        "trdDd": "date",
        "stckItmsCd": "ticker",
        "srtnCd": "ticker",
        "itmsNm": "name",
        "stckLnbCntrVol": "lending_contract_shares",
        "stckLnbCntrAmt": "lending_contract_value",
        "stckLnbRpayVol": "lending_repayment_shares",
        "stckLnbRpayAmt": "lending_repayment_value",
        "stckLnbBlncVol": "lending_balance_shares",
        "stckLnbBlncAmt": "lending_balance_value",
        "lnbCntrVol": "lending_contract_shares",
        "lnbCntrAmt": "lending_contract_value",
        "lnbRpayVol": "lending_repayment_shares",
        "lnbRpayAmt": "lending_repayment_value",
        "lnbBlncVol": "lending_balance_shares",
        "lnbBlncAmt": "lending_balance_value",
        "lendCntrQty": "lending_contract_shares",
        "lendCntrAmt": "lending_contract_value",
        "lendRpayQty": "lending_repayment_shares",
        "lendRpayAmt": "lending_repayment_value",
        "lendBlncQty": "lending_balance_shares",
        "lendBlncAmt": "lending_balance_value",
        "invstSe": "participant_type",
        "invpnSe": "participant_type",
        "invstNm": "participant_name",
        "invpnNm": "participant_name",
    }
    out = out.rename(columns=rename)
    if "date" not in out:
        found = next((c for c in date_candidates if c in out), None)
        if found:
            out = out.rename(columns={found: "date"})
    if "ticker" not in out:
        found = next((c for c in ticker_candidates if c in out), None)
        if found:
            out = out.rename(columns={found: "ticker"})
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"].astype(str), errors="coerce")
        out = out.loc[out["date"].notna()]
    if "ticker" in out:
        out["ticker"] = normalize_ticker(out["ticker"])
    text_columns = {
        "date", "ticker", "name", "requested_ticker", "source_endpoint",
        "participant_type", "participant_name",
    }
    for column in out.columns:
        if column not in text_columns:
            converted = pd.to_numeric(out[column], errors="coerce")
            if converted.notna().any() or out[column].isna().all():
                out[column] = converted
    keys = [c for c in ["ticker", "date", "source_endpoint", "participant_type", "participant_name"] if c in out]
    return out.sort_values(keys).drop_duplicates(keys, keep="last") if keys else out


def collect_stock_lending(
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    sleep_seconds: float = 0.12,
    chunk_months: int = 12,
) -> dict:
    key = _service_key()
    if not key:
        raise RuntimeError(
            "DATA_GO_KR_SERVICE_KEY가 없습니다. 공공데이터포털 주식대차 API는 무료·자동승인이지만 사용자 키가 필수입니다. "
            ".env.data.local에 키를 넣은 뒤 재개하세요."
        )
    baskets = read_enabled_tickers(paths)
    root = paths.raw_dual / "required_data_v3" / "stock_lending"
    source_root = root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "stock_lending_manifest.csv"
    session = requests.Session()
    session.headers.update({"User-Agent": "CrashWatch-AI-Research/3.0"})
    records: list[SourceRecord] = []
    timer_all = time.perf_counter()

    for ticker in baskets["ticker"]:
        for left, right in date_chunks(start, end, months=chunk_months):
            begin = left.strftime("%Y%m%d")
            finish = right.strftime("%Y%m%d")
            chunk = f"{begin}_{finish}"
            for endpoint in ENDPOINTS:
                path = source_root / f"ticker={ticker}" / f"endpoint={endpoint}" / f"{chunk}.parquet"
                started = now_iso()
                timer = time.perf_counter()
                if path.exists() and not overwrite:
                    frame = pd.read_parquet(path)
                    record = SourceRecord(
                        "data.go.kr/KSD", endpoint, f"{ticker}:{chunk}", "cached", len(frame), started,
                        now_iso(), time.perf_counter() - timer, str(path.relative_to(paths.project)), "", sha256_file(path),
                    )
                    records.append(record)
                    append_manifest([record], manifest_path)
                    continue
                combined: list[dict[str, Any]] = []
                try:
                    page = 1
                    while True:
                        params = {
                            "serviceKey": key,
                            "pageNo": page,
                            "numOfRows": 1000,
                            "resultType": "json",
                            "beginBasDt": begin,
                            "endBasDt": finish,
                            "stckItmsCd": ticker,
                        }
                        payload = retry_call(
                            lambda: _request_json(session, endpoint, params),
                            attempts=4,
                            base_delay=1.5,
                            max_delay=30.0,
                        )
                        items, total = _extract_items(payload)
                        for item in items:
                            item = dict(item)
                            item["requested_ticker"] = ticker
                            item["source_endpoint"] = endpoint
                            combined.append(item)
                        if not items or page * 1000 >= total:
                            break
                        page += 1
                        sleep_with_jitter(sleep_seconds)
                    frame = _normalize_lending(pd.DataFrame(combined))
                    if not frame.empty:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_parquet(frame, path)
                        status = "success"
                        checksum = sha256_file(path)
                    else:
                        status = "empty"
                        checksum = ""
                    record = SourceRecord(
                        "data.go.kr/KSD", endpoint, f"{ticker}:{chunk}", status, len(frame), started,
                        now_iso(), time.perf_counter() - timer, str(path.relative_to(paths.project)) if path.exists() else "", "", checksum,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = sanitize_secret_error(f"{type(exc).__name__}: {exc}")
                    record = SourceRecord(
                        "data.go.kr/KSD", endpoint, f"{ticker}:{chunk}", "failed", 0, started,
                        now_iso(), time.perf_counter() - timer, "", error, "",
                    )
                records.append(record)
                append_manifest([record], manifest_path)
                sleep_with_jitter(sleep_seconds)

    combined = build_stock_lending(paths)
    summary = summarize_stock_lending(paths, combined)
    summary.update({
        "elapsed_seconds": time.perf_counter() - timer_all,
        "completed_at": now_iso(),
        "chunk_months": chunk_months,
        "endpoints": list(ENDPOINTS),
    })
    atomic_json(summary, root / "stock_lending_summary.json")
    return summary


def _collapse_endpoint(block: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    index_cols = ["ticker", "date"]
    excluded = set(index_cols + ["source_endpoint", "requested_ticker", "participant_type", "participant_name"])
    data_cols = [c for c in block.columns if c not in excluded]
    if block.empty:
        return pd.DataFrame(columns=index_cols)
    block = block[index_cols + data_cols].groupby(index_cols, as_index=False).last()
    if endpoint != "getStckLnbDetail":
        block = block.rename(columns={c: f"progress_{c}" for c in data_cols})
    return block


def build_stock_lending(paths: ProjectPaths) -> pd.DataFrame:
    root = paths.raw_dual / "required_data_v3" / "stock_lending"
    source_root = root / "sources"
    paths_list = sorted(source_root.rglob("*.parquet")) if source_root.exists() else []
    if not paths_list:
        return pd.DataFrame()
    frames = [_normalize_lending(pd.read_parquet(path)) for path in paths_list]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keys = [c for c in ["ticker", "date", "source_endpoint", "participant_type", "participant_name"] if c in combined]
    combined = combined.sort_values(keys).drop_duplicates(keys, keep="last")

    participant = combined.loc[combined["source_endpoint"].eq("getStckLnbInvpnDetail")].copy()
    if not participant.empty:
        participant = participant.sort_values(keys).drop_duplicates(keys, keep="last")
        atomic_csv(participant, root / "stock_lending_participant_daily.csv")
        atomic_parquet(participant, root / "stock_lending_participant_daily.parquet")

    endpoint_blocks: list[pd.DataFrame] = []
    for endpoint in ["getStckLnbDetail", "getStckLnbProgress"]:
        block = combined.loc[combined["source_endpoint"].eq(endpoint)]
        collapsed = _collapse_endpoint(block, endpoint)
        if not collapsed.empty:
            endpoint_blocks.append(collapsed)
    if not endpoint_blocks:
        return pd.DataFrame()
    out = endpoint_blocks[0]
    for block in endpoint_blocks[1:]:
        out = out.merge(block, on=["ticker", "date"], how="outer", validate="one_to_one")
    for column in list(out.columns):
        if not column.startswith("progress_"):
            continue
        base = column.removeprefix("progress_")
        if base in out.columns:
            out[base] = out[base].combine_first(out[column])
        else:
            out[base] = out[column]
    out = out.drop(columns=[c for c in out if c.startswith("progress_")], errors="ignore")
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    atomic_csv(out, root / "stock_lending_48_tickers_daily.csv")
    atomic_parquet(out, root / "stock_lending_48_tickers_daily.parquet")

    # Finance feature pipeline integration; intentionally separate from actual short data.
    finance_root = paths.raw_dual / "finance11h"
    finance_root.mkdir(parents=True, exist_ok=True)
    finance = out.rename(columns={c: f"fl_{c}" for c in out.columns if c not in {"date", "ticker", "name"}})
    atomic_csv(finance, finance_root / "stock_lending_ticker_timeseries.csv")
    atomic_parquet(finance, finance_root / "stock_lending_ticker_timeseries.parquet")
    return out


def summarize_stock_lending(paths: ProjectPaths, combined: pd.DataFrame | None = None) -> dict:
    root = paths.raw_dual / "required_data_v3" / "stock_lending"
    if combined is None:
        path = root / "stock_lending_48_tickers_daily.parquet"
        combined = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not combined.empty:
        combined["ticker"] = normalize_ticker(combined["ticker"])
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    tickers = read_enabled_tickers(paths)["ticker"].tolist()
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        block = combined.loc[combined["ticker"].eq(ticker)] if not combined.empty else pd.DataFrame()
        if len(block):
            start_date = block["date"].min()
            end_date = block["date"].max()
            span_days = int((end_date - start_date).days) if pd.notna(start_date) and pd.notna(end_date) else 0
        else:
            start_date = end_date = pd.NaT
            span_days = 0
        ratios = {c: float(block[c].notna().mean()) if c in block and len(block) else 0.0 for c in CORE_COLUMNS}
        available_ratios = [ratio for column, ratio in ratios.items() if column in block.columns]
        minimum_ratio = min(available_ratios) if available_ratios else 0.0
        ready = len(block) >= 600 and span_days >= 900 and minimum_ratio >= 0.50
        rows.append({
            "ticker": ticker,
            "rows": len(block),
            "start_date": str(start_date.date()) if pd.notna(start_date) else "",
            "end_date": str(end_date.date()) if pd.notna(end_date) else "",
            "span_days": span_days,
            **{f"{k}_non_null_ratio": v for k, v in ratios.items()},
            "required_min_non_null_ratio": minimum_ratio,
            "status": "ready" if ready else "insufficient",
        })
    coverage = pd.DataFrame(rows)
    atomic_csv(coverage, root / "stock_lending_coverage_48_tickers.csv")
    ready_count = int(coverage["status"].eq("ready").sum()) if not coverage.empty else 0
    participant_path = root / "stock_lending_participant_daily.parquet"
    participant_rows = len(pd.read_parquet(participant_path)) if participant_path.exists() else 0
    return {
        "ticker_count": len(tickers),
        "ready_tickers": ready_count,
        "ticker_coverage": ready_count / len(tickers) if tickers else 0.0,
        "rows": len(combined),
        "participant_rows": participant_rows,
        "ready_rule": {"minimum_rows": 600, "minimum_span_days": 900, "minimum_non_null_ratio": 0.50},
        "actual_stock_lending": bool(len(combined)),
        "note": "주식대차는 실제 공매도와 별도 그룹이며 fs_short_*로 변환하지 않습니다.",
    }
