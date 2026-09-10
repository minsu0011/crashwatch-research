from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT / "output"
DATA_GO_BASE = "https://apis.data.go.kr/1160100/service/GetCMStckLnbInfoService"
ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
OPENDART_BULK_URL = "https://opendart.fss.or.kr/disclosureinfo/fnltt/dwld/main.do"


@dataclass
class ManifestRow:
    source: str
    dataset: str
    status: str
    rows: int = 0
    file: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_key(value: str) -> str:
    value = (value or "").strip().strip('"').strip("'")
    return unquote(value)


def read_tickers(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype={"ticker": str})
    if "enabled" in df.columns:
        df = df.loc[pd.to_numeric(df["enabled"], errors="coerce").fillna(0).astype(int).eq(1)]
    tickers = df["ticker"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    return sorted(tickers.dropna().drop_duplicates().tolist())


def chunk_years(start: str, end: str) -> Iterable[tuple[str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for year in range(start_ts.year, end_ts.year + 1):
        left = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        right = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        yield left.strftime("%Y%m%d"), right.strftime("%Y%m%d")


def extract_items(payload: Any) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(payload, dict):
        return [], 0
    response = payload.get("response", payload)
    body = response.get("body", {}) if isinstance(response, dict) else {}
    total = int(body.get("totalCount", 0) or 0) if isinstance(body, dict) else 0
    items = body.get("items", {}) if isinstance(body, dict) else {}
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        items = []
    return [x for x in items if isinstance(x, dict)], total


def request_json(session: requests.Session, url: str, params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower() and response.text.lstrip().startswith("<"):
                raise RuntimeError(f"JSON 대신 XML/오류 응답 수신: {response.text[:300]}")
            return response.json()
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(10, 1.5 ** attempt))
    raise RuntimeError(str(last))


def download_data_go_stock_lending(
    key: str,
    tickers: list[str],
    start: str,
    end: str,
    output: Path,
    sleep_seconds: float = 0.08,
) -> list[ManifestRow]:
    key = normalize_key(key)
    if not key:
        return [ManifestRow("data.go.kr", "stock_lending", "skipped_missing_key", error="DATA_GO_KR_SERVICE_KEY 없음")]

    root = output / "data_go_kr_stock_lending"
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "CrashWatch-AI-Research/1.0"})
    manifests: list[ManifestRow] = []
    endpoints = {
        "lending_detail": "getStckLnbDetail",
        "lending_progress": "getStckLnbProgress",
    }

    for dataset, endpoint in endpoints.items():
        dataset_started = now_iso()
        combined: list[dict[str, Any]] = []
        errors: list[str] = []
        for ticker in tickers:
            for begin_dt, end_dt in chunk_years(start, end):
                page = 1
                while True:
                    params = {
                        "serviceKey": key,
                        "pageNo": page,
                        "numOfRows": 1000,
                        "resultType": "json",
                        "beginBasDt": begin_dt,
                        "endBasDt": end_dt,
                        "stckItmsCd": ticker,
                    }
                    try:
                        payload = request_json(session, f"{DATA_GO_BASE}/{endpoint}", params)
                        items, total = extract_items(payload)
                        for item in items:
                            item["requested_ticker"] = ticker
                            item["requested_begin"] = begin_dt
                            item["requested_end"] = end_dt
                            item["source_endpoint"] = endpoint
                        combined.extend(items)
                        if not items or page * 1000 >= total:
                            break
                        page += 1
                        time.sleep(sleep_seconds)
                    except Exception as exc:
                        errors.append(f"{ticker}/{begin_dt}-{end_dt}: {exc}")
                        break
                time.sleep(sleep_seconds)

        df = pd.DataFrame(combined)
        csv_path = root / f"{dataset}_{start}_{end}.csv"
        parquet_path = root / f"{dataset}_{start}_{end}.parquet"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if len(df):
            df.to_parquet(parquet_path, index=False)
        error_path = root / f"{dataset}_errors.txt"
        error_path.write_text("\n".join(errors), encoding="utf-8")
        status = "success" if len(df) and not errors else "partial" if len(df) else "failed"
        manifests.append(
            ManifestRow(
                "data.go.kr",
                dataset,
                status,
                rows=len(df),
                file=str(csv_path.relative_to(output)),
                started_at=dataset_started,
                completed_at=now_iso(),
                error=f"errors={len(errors)}",
            )
        )
    return manifests


def ecos_url(key: str, stat_code: str, cycle: str, start: str, end: str, item_code1: str) -> str:
    # ECOS StatisticSearch URL: key/json/kr/startRow/endRow/statCode/cycle/start/end/item1
    return "/".join(
        [
            ECOS_BASE.rstrip("/"),
            key,
            "json",
            "kr",
            "1",
            "100000",
            stat_code,
            cycle,
            start,
            end,
            item_code1 or "?",
        ]
    )


def ecos_date(value: str, cycle: str) -> str:
    ts = pd.Timestamp(value)
    cycle = cycle.upper()
    if cycle == "D":
        return ts.strftime("%Y%m%d")
    if cycle == "M":
        return ts.strftime("%Y%m")
    if cycle == "Q":
        return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"
    if cycle == "A":
        return ts.strftime("%Y")
    return ts.strftime("%Y%m%d")


def download_ecos(key: str, config: Path, start: str, end: str, output: Path) -> list[ManifestRow]:
    key = normalize_key(key)
    if not key:
        return [ManifestRow("ECOS", "macro_series", "skipped_missing_key", error="ECOS_API_KEY 없음")]
    root = output / "ecos"
    root.mkdir(parents=True, exist_ok=True)
    cfg = pd.read_csv(config)
    cfg = cfg.loc[pd.to_numeric(cfg.get("enabled", 1), errors="coerce").fillna(1).astype(int).eq(1)]
    session = requests.Session()
    result_rows: list[dict[str, Any]] = []
    manifests: list[ManifestRow] = []
    for row in cfg.itertuples(index=False):
        alias = str(row.alias)
        started = now_iso()
        try:
            url = ecos_url(
                key,
                str(row.stat_code),
                str(row.cycle),
                ecos_date(start, str(row.cycle)),
                ecos_date(end, str(row.cycle)),
                str(row.item_code1),
            )
            payload = request_json(session, url, {})
            block = payload.get("StatisticSearch", {})
            rows = block.get("row", []) if isinstance(block, dict) else []
            if isinstance(rows, dict):
                rows = [rows]
            normalized = []
            for item in rows or []:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item["alias"] = alias
                item["group"] = getattr(row, "group", "")
                normalized.append(item)
                result_rows.append(item)
            path = root / f"{alias}.csv"
            pd.DataFrame(normalized).to_csv(path, index=False, encoding="utf-8-sig")
            manifests.append(ManifestRow("ECOS", alias, "success" if normalized else "empty", len(normalized), str(path.relative_to(output)), started, now_iso()))
        except Exception as exc:
            manifests.append(ManifestRow("ECOS", alias, "failed", 0, "", started, now_iso(), str(exc)))
    combined = pd.DataFrame(result_rows)
    combined.to_csv(root / "ecos_combined.csv", index=False, encoding="utf-8-sig")
    if len(combined):
        combined.to_parquet(root / "ecos_combined.parquet", index=False)
    return manifests


def wait_download(download_dir: Path, before: set[str], timeout: float = 120.0) -> str | None:
    deadline = time.monotonic() + timeout
    last_new: set[str] = set()
    stable_since: float | None = None
    while time.monotonic() < deadline:
        current = {p.name for p in download_dir.iterdir() if p.is_file()}
        partial = [x for x in current if x.endswith((".crdownload", ".tmp", ".part"))]
        new_files = {x for x in current - before if not x.endswith((".crdownload", ".tmp", ".part"))}
        if new_files and not partial:
            if new_files == last_new:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 1.5:
                    return sorted(new_files)[-1]
            else:
                last_new = new_files
                stable_since = time.monotonic()
        time.sleep(0.5)
    return None


def download_opendart_bulk(
    output: Path,
    years: set[int],
    reports: set[str],
    headless: bool = False,
) -> list[ManifestRow]:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except Exception as exc:
        return [ManifestRow("OpenDART", "bulk_financials", "failed", error=f"selenium import 실패: {exc}")]

    root = (output / "opendart_bulk").resolve()
    root.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(root),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--window-size=1600,1000")

    driver = webdriver.Chrome(options=options)
    manifests: list[ManifestRow] = []
    try:
        driver.get(OPENDART_BULK_URL)
        WebDriverWait(driver, 40).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
        time.sleep(2)
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        if not rows:
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
        statement_names = ["balance_sheet", "income_statement", "cash_flow", "changes_in_equity"]
        for row_index, row in enumerate(rows):
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 7:
                continue
            row_text = " ".join(c.text.strip() for c in cells)
            year_match = re.search(r"\b(20\d{2})\b", row_text)
            if not year_match:
                continue
            year = int(year_match.group(1))
            if year not in years:
                continue
            report_text = cells[2].text.strip() if len(cells) > 2 else row_text
            report_key = (
                "annual" if "사업" in report_text else
                "half" if "반기" in report_text else
                "q1" if "1분기" in report_text else
                "q3" if "3분기" in report_text else "other"
            )
            if report_key not in reports:
                continue

            for statement_offset, statement in enumerate(statement_names, start=3):
                if statement_offset >= len(cells):
                    continue
                links = cells[statement_offset].find_elements(By.TAG_NAME, "a")
                if not links:
                    continue
                started = now_iso()
                before = {p.name for p in root.iterdir() if p.is_file()}
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", links[0])
                    driver.execute_script("arguments[0].click();", links[0])
                    filename = wait_download(root, before)
                    status = "success" if filename else "timeout"
                    manifests.append(ManifestRow("OpenDART", f"{year}_{report_key}_{statement}", status, 0, filename or "", started, now_iso(), "" if filename else "다운로드 완료 파일 미감지"))
                except Exception as exc:
                    manifests.append(ManifestRow("OpenDART", f"{year}_{report_key}_{statement}", "failed", 0, "", started, now_iso(), str(exc)))
                time.sleep(0.5)
    finally:
        driver.quit()
    return manifests


def write_manifest(rows: list[ManifestRow], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(ManifestRow.__dataclass_fields__)
    with (output / "download_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary = {
        "created_at": now_iso(),
        "total": len(rows),
        "status_counts": pd.Series([x.status for x in rows]).value_counts().to_dict() if rows else {},
        "rows_downloaded": int(sum(x.rows for x in rows)),
    }
    (output / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = map(int, part.split("-", 1))
            years.update(range(left, right + 1))
        else:
            years.add(int(part))
    return years


def main() -> None:
    load_dotenv(PROJECT / ".env", override=False)
    parser = argparse.ArgumentParser(description="CrashWatch 우선순위 금융 원자료 수집기")
    parser.add_argument("--sources", default="opendart,data_go,ecos", help="opendart,data_go,ecos")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tickers", type=Path, default=PROJECT / "configs" / "sector_baskets.csv")
    parser.add_argument("--ecos-config", type=Path, default=PROJECT / "configs" / "ecos_series.csv")
    parser.add_argument("--dart-years", default="2018-2025")
    parser.add_argument("--dart-reports", default="annual,half,q1,q3")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sources = {x.strip().lower() for x in args.sources.split(",") if x.strip()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tickers = read_tickers(args.tickers)
    rows: list[ManifestRow] = []

    if "opendart" in sources:
        rows.extend(download_opendart_bulk(output, parse_years(args.dart_years), {x.strip() for x in args.dart_reports.split(",")}, args.headless))
    if "data_go" in sources:
        rows.extend(download_data_go_stock_lending(os.getenv("DATA_GO_KR_SERVICE_KEY", ""), tickers, args.start, args.end, output))
    if "ecos" in sources:
        rows.extend(download_ecos(os.getenv("ECOS_API_KEY", ""), args.ecos_config, args.start, args.end, output))

    write_manifest(rows, output)
    print(json.dumps({"output": str(output), "manifest": str(output / "download_manifest.csv"), "tasks": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
