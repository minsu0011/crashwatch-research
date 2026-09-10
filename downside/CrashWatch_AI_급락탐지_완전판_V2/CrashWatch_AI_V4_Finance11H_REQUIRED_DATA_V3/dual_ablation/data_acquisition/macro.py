from __future__ import annotations

import logging
import os
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pandas.tseries.offsets import BDay

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet
from .common import SourceRecord, append_manifest, now_iso, retry_call, sanitize_secret_error, sha256_file

LOGGER = logging.getLogger(__name__)
FRED_SERIES = {
    "usdkrw_fred": "DEXKOUS",
    "vix": "VIXCLS",
    "us_high_yield_oas": "BAMLH0A0HYM2",
    "us_ig_oas": "BAMLC0A0CM",
    "nfci": "NFCI",
    "stl_fsi": "STLFSI4",
    "fed_funds": "DFF",
    "us_2y": "DGS2",
    "us_10y": "DGS10",
}
ECOS_SERIES = {
    "base_rate": ("722Y001", "0101000", "D"),
    "usdkrw_ecos": ("731Y001", "0000001", "D"),
    "corp_bond_aa_minus": ("817Y002", "010300000", "D"),
    "treasury_3y": ("817Y002", "010200000", "D"),
}


def _fred_url(series_id: str, start: str, end: str) -> str:
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={pd.Timestamp(start).strftime('%Y-%m-%d')}&coed={pd.Timestamp(end).strftime('%Y-%m-%d')}"
    )


def _attach_conservative_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve observation date and expose a conservative next-business-day key.

    FRED/ECOS series have heterogeneous publication times and occasional
    revisions. Finance11H therefore consumes them no earlier than the next
    business day. The original observation date is retained for audit.
    """
    out = frame.copy()
    out["observation_date"] = pd.to_datetime(out["date"], errors="coerce")
    out["available_from"] = out["observation_date"] + BDay(1)
    return out


def _download_fred(session: requests.Session, alias: str, series_id: str, start: str, end: str) -> pd.DataFrame:
    response = session.get(_fred_url(series_id, start, end), timeout=90)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or len(frame.columns) < 2:
        raise RuntimeError(f"FRED 빈 응답: {series_id}")
    frame = frame.rename(columns={frame.columns[0]: "date", frame.columns[1]: alias})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
    frame["source_series_id"] = series_id
    frame = frame.loc[frame["date"].notna()].sort_values("date").drop_duplicates("date", keep="last")
    return _attach_conservative_availability(frame)


def _ecos_period(value: str, cycle: str) -> str:
    ts = pd.Timestamp(value)
    cycle = cycle.upper()
    if cycle == "D":
        return ts.strftime("%Y%m%d")
    if cycle == "M":
        return ts.strftime("%Y%m")
    if cycle == "Q":
        return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"
    return ts.strftime("%Y")


def _download_ecos(
    session: requests.Session,
    key: str,
    alias: str,
    stat_code: str,
    item_code: str,
    cycle: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    url = "/".join([
        "https://ecos.bok.or.kr/api/StatisticSearch",
        key,
        "json",
        "kr",
        "1",
        "100000",
        stat_code,
        cycle,
        _ecos_period(start, cycle),
        _ecos_period(end, cycle),
        item_code,
    ])
    response = session.get(url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if "RESULT" in payload:
        result = payload["RESULT"]
        raise RuntimeError(f"ECOS {result.get('CODE')}: {result.get('MESSAGE')}")
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if isinstance(rows, dict):
        rows = [rows]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", alias, "observation_date", "available_from"])
    frame = frame.rename(columns={"TIME": "date", "DATA_VALUE": alias})
    if cycle.upper() == "D":
        frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        frame["date"] = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
    frame["source_stat_code"] = stat_code
    frame["source_item_code"] = item_code
    frame = frame.loc[frame["date"].notna()].sort_values("date").drop_duplicates("date", keep="last")
    return _attach_conservative_availability(frame)


def _safe_series_frame(frame: pd.DataFrame, alias: str) -> pd.DataFrame:
    if frame.empty or alias not in frame:
        return pd.DataFrame(columns=["date", alias])
    out = frame[["available_from", alias]].rename(columns={"available_from": "date"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.loc[out["date"].notna()].sort_values("date").drop_duplicates("date", keep="last")


def collect_macro_credit(paths: ProjectPaths, start: str, end: str, *, overwrite: bool = False) -> dict:
    root = paths.raw_dual / "required_data_v3" / "macro_credit"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "macro_manifest.csv"
    # Keep Requests' default user agent. FRED's public CSV endpoint resets
    # connections from the synthetic project UA and from the tested browser UA
    # on this network, while the default Requests client is served normally.
    session = requests.Session()
    records: list[SourceRecord] = []
    frames: list[pd.DataFrame] = []
    timer_all = time.perf_counter()

    for alias, series_id in FRED_SERIES.items():
        csv_path = root / "fred" / f"{alias}.csv"
        parquet_path = root / "fred" / f"{alias}.parquet"
        started = now_iso()
        timer = time.perf_counter()
        if parquet_path.exists() and not overwrite:
            frame = pd.read_parquet(parquet_path)
            status = "cached"
        else:
            try:
                frame = retry_call(lambda: _download_fred(session, alias, series_id, start, end), attempts=4, base_delay=1.0)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_csv(frame, csv_path)
                atomic_parquet(frame, parquet_path)
                status = "success" if len(frame) else "empty"
            except Exception as exc:  # noqa: BLE001
                error = sanitize_secret_error(f"{type(exc).__name__}: {exc}")
                record = SourceRecord("FRED", alias, series_id, "failed", 0, started, now_iso(), time.perf_counter() - timer, "", error, "")
                records.append(record)
                append_manifest([record], manifest_path)
                continue
        frames.append(_safe_series_frame(frame, alias))
        record = SourceRecord(
            "FRED", alias, series_id, status, len(frame), started, now_iso(), time.perf_counter() - timer,
            str(csv_path.relative_to(paths.project)) if csv_path.exists() else str(parquet_path.relative_to(paths.project)),
            "", sha256_file(csv_path) if csv_path.exists() else sha256_file(parquet_path),
        )
        records.append(record)
        append_manifest([record], manifest_path)

    ecos_key = os.getenv("ECOS_API_KEY", "").strip()
    for alias, (stat_code, item_code, cycle) in ECOS_SERIES.items():
        if not ecos_key:
            record = SourceRecord(
                "ECOS", alias, f"{stat_code}/{item_code}", "skipped_missing_key", 0,
                now_iso(), now_iso(), 0.0, "", "ECOS_API_KEY 없음", "",
            )
            records.append(record)
            append_manifest([record], manifest_path)
            continue
        csv_path = root / "ecos" / f"{alias}.csv"
        parquet_path = root / "ecos" / f"{alias}.parquet"
        started = now_iso()
        timer = time.perf_counter()
        if parquet_path.exists() and not overwrite:
            frame = pd.read_parquet(parquet_path)
            status = "cached"
        else:
            try:
                frame = retry_call(
                    lambda: _download_ecos(session, ecos_key, alias, stat_code, item_code, cycle, start, end),
                    attempts=4,
                    base_delay=1.0,
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_csv(frame, csv_path)
                atomic_parquet(frame, parquet_path)
                status = "success" if len(frame) else "empty"
            except Exception as exc:  # noqa: BLE001
                error = sanitize_secret_error(f"{type(exc).__name__}: {exc}")
                record = SourceRecord("ECOS", alias, f"{stat_code}/{item_code}", "failed", 0, started, now_iso(), time.perf_counter() - timer, "", error, "")
                records.append(record)
                append_manifest([record], manifest_path)
                continue
        frames.append(_safe_series_frame(frame, alias))
        record = SourceRecord(
            "ECOS", alias, f"{stat_code}/{item_code}", status, len(frame), started, now_iso(), time.perf_counter() - timer,
            str(csv_path.relative_to(paths.project)) if csv_path.exists() else str(parquet_path.relative_to(paths.project)),
            "", sha256_file(csv_path) if csv_path.exists() else sha256_file(parquet_path),
        )
        records.append(record)
        append_manifest([record], manifest_path)

    combined = pd.DataFrame(columns=["date"])
    for frame in frames:
        if frame.empty:
            continue
        if combined.empty or (list(combined.columns) == ["date"] and not len(combined)):
            combined = frame.copy()
        else:
            combined = combined.merge(frame, on="date", how="outer", validate="one_to_one")
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.loc[combined["date"].notna()].sort_values("date").drop_duplicates("date", keep="last")
        combined["available_from"] = combined["date"]
        if {"corp_bond_aa_minus", "treasury_3y"}.issubset(combined.columns):
            combined["korea_credit_spread_aa_minus_3y"] = combined["corp_bond_aa_minus"] - combined["treasury_3y"]
        if {"us_10y", "us_2y"}.issubset(combined.columns):
            combined["us_term_spread_10y_2y"] = combined["us_10y"] - combined["us_2y"]
        if {"us_high_yield_oas", "us_ig_oas"}.issubset(combined.columns):
            combined["us_hy_minus_ig_oas"] = combined["us_high_yield_oas"] - combined["us_ig_oas"]
        atomic_csv(combined, root / "macro_credit_daily.csv")
        atomic_parquet(combined, root / "macro_credit_daily.parquet")
        market_root = paths.raw_dual / "market"
        market_root.mkdir(parents=True, exist_ok=True)
        atomic_parquet(combined, market_root / "required_macro_credit_daily.parquet")
        finance_root = paths.raw_dual / "finance11h"
        finance_root.mkdir(parents=True, exist_ok=True)
        finance_path = finance_root / "finance_market_timeseries.parquet"
        finance_input = combined.drop(columns=["available_from", "observation_date"], errors="ignore")
        if finance_path.exists():
            existing = pd.read_parquet(finance_path)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            duplicate = [c for c in finance_input.columns if c != "date" and c in existing.columns]
            existing = existing.drop(columns=duplicate, errors="ignore")
            finance_market = existing.merge(finance_input, on="date", how="outer", validate="one_to_one")
        else:
            finance_market = finance_input.copy()
        finance_market = finance_market.sort_values("date").drop_duplicates("date", keep="last")
        atomic_csv(finance_market, finance_root / "finance_market_timeseries.csv")
        atomic_parquet(finance_market, finance_path)
    summary = {
        "completed_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - timer_all,
        "rows": len(combined),
        "columns": list(combined.columns),
        "fred_success": sum(r.source == "FRED" and r.status in {"success", "cached"} for r in records),
        "ecos_success": sum(r.source == "ECOS" and r.status in {"success", "cached"} for r in records),
        "ecos_key_present": bool(ecos_key),
        "availability_policy": "observation_date + 1 business day (conservative anti-leakage)",
    }
    atomic_json(summary, root / "macro_summary.json")
    return summary
