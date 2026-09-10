#!/usr/bin/env python
"""KRX 전종목 공매도 순보유잔고를 시장·날짜 단위로 수집한다.

개별 종목의 1년 구간 조회는 한 요청이 오래 걸린다. 이 수집기는
MDCSTAT305의 빠른 전종목 일별 응답을 KOSPI/KOSDAQ/KONEX별로 조회하고
프로젝트 활성 종목만 연도별 CSV에 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from krx_short_all_10y import (
    KrxClient,
    KrxDownloadError,
    acquire_instance_lock,
    atomic_json,
    iter_weekdays,
    load_progress,
    load_target_tickers,
    parse_date,
    save_progress,
)


START_DATE = "2016-07-28"
# 순보유잔고는 보고의무 발생일 기준 T+2에 공개된다.
END_DATE = "2026-07-23"
OUTPUT_FOLDER_NAME = "krx_short_balance_daily_output"
DATASET = "dbms/MDC_OUT/STAT/srt/MDCSTAT30501_OUT"
DEFAULT_WORKERS = 4
DEFAULT_DELAY_SECONDS = 0.20
MARKETS = {
    "KOSPI": "1",
    "KOSDAQ": "2",
    "KONEX": "6",
}

OUTPUT_HEADER = [
    "일자",
    "시장구분",
    "종목코드",
    "종목명",
    "공매도잔고수량",
    "상장주식수",
    "공매도잔고금액",
    "시가총액",
    "공매도잔고비중",
]


@dataclass(frozen=True)
class BalanceResult:
    trade_date: date
    market: str
    rows: list[list[str]]


def collect_day(
    client: KrxClient,
    trade_date: date,
    market: str,
    targets: dict[str, tuple[str, str]],
) -> BalanceResult:
    raw = client._post(  # noqa: SLF001 - 공통 세션/재시도 로직 재사용
        {
            "bld": DATASET,
            "locale": "ko_KR",
            "searchType": "1",
            "mktTpCd": MARKETS[market],
            "trdDd": trade_date.strftime("%Y%m%d"),
            "share": "1",
            "money": "1",
        }
    )
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise KrxDownloadError(
            f"JSON이 아닌 잔고 응답: {raw[:120]!r}"
        ) from exc
    source_rows = payload.get("OutBlock_1")
    if not isinstance(source_rows, list):
        raise KrxDownloadError("OutBlock_1 배열이 없는 잔고 응답")

    rows: list[list[str]] = []
    for item in source_rows:
        ticker = str(item.get("ISU_CD", "")).strip().zfill(6)
        if ticker not in targets:
            continue
        configured_name, configured_market = targets[ticker]
        rows.append(
            [
                trade_date.isoformat(),
                configured_market or market,
                ticker,
                configured_name,
                str(item.get("BAL_QTY", "")).strip(),
                str(item.get("LIST_SHRS", "")).strip(),
                str(item.get("BAL_AMT", "")).strip(),
                str(item.get("MKTCAP", "")).strip(),
                str(item.get("BAL_RTO", "")).strip(),
            ]
        )
    if client.delay_seconds:
        time.sleep(client.delay_seconds + random.uniform(0.0, 0.08))
    return BalanceResult(trade_date, market, rows)


def append_year_csv(output_dir: Path, result: BalanceResult) -> int:
    if not result.rows:
        return 0
    path = output_dir / f"KRX_short_balance_{result.trade_date.year}.csv"
    is_new = not path.exists() or path.stat().st_size == 0
    encoding = "utf-8-sig" if is_new else "utf-8"
    with path.open("a", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(OUTPUT_HEADER)
        writer.writerows(result.rows)
    return len(result.rows)


def append_error(
    path: Path,
    trade_date: date,
    market: str,
    error: Exception,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().isoformat(timespec='seconds')}\t"
            f"{trade_date}\t{market}\t{type(error).__name__}: {error}\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KRX 공매도 순보유잔고 10년치 일별 수집"
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)
    if start > end:
        raise ValueError("시작일이 종료일보다 늦습니다.")
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers는 1~4만 허용합니다.")

    project = Path(__file__).resolve().parent
    targets = load_target_tickers(project)
    output_dir = (args.output_dir or project / OUTPUT_FOLDER_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_lock = acquire_instance_lock(output_dir)
    progress_path = output_dir / "progress.json"
    error_path = output_dir / "errors.log"
    completed = load_progress(progress_path)
    weekdays = list(iter_weekdays(start, end))
    jobs = [
        (trade_date, market)
        for trade_date in weekdays
        for market in MARKETS
        if f"{trade_date:%Y%m%d}|{market}" not in completed
    ]
    total = len(weekdays) * len(MARKETS)

    print("=" * 72, flush=True)
    print("KRX 개별종목 공매도 순보유잔고 일별 수집", flush=True)
    print(f"기간          : {start} ~ {end}", flush=True)
    print(f"대상          : 활성 {len(targets)}종목", flush=True)
    print(f"전체/남은 요청: {total:,}/{len(jobs):,}", flush=True)
    print(f"동시 작업     : {args.workers}개", flush=True)
    print(f"출력          : {output_dir}", flush=True)
    print("=" * 72, flush=True)

    thread_state = threading.local()

    def task(trade_date: date, market: str) -> BalanceResult:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = KrxClient(args.delay)
            thread_state.client = client
        return collect_day(client, trade_date, market, targets)

    started = time.time()
    saved_rows = 0
    errors = 0
    finished = total - len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map: dict[
            Future[BalanceResult],
            tuple[date, str],
        ] = {
            executor.submit(task, trade_date, market): (trade_date, market)
            for trade_date, market in jobs
        }
        for future in as_completed(future_map):
            trade_date, market = future_map[future]
            finished += 1
            prefix = f"[{finished:,}/{total:,}] {trade_date} {market}"
            try:
                result = future.result()
                count = append_year_csv(output_dir, result)
                saved_rows += count
                completed.add(f"{trade_date:%Y%m%d}|{market}")
                save_progress(progress_path, completed)
                status = f"{count:,}행 저장" if count else "휴장/대상 데이터 없음"
                print(f"{prefix} - {status}", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                append_error(error_path, trade_date, market, exc)
                print(f"{prefix} - 오류: {exc}", flush=True)

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "target_tickers": len(targets),
        "requests_total": total,
        "requests_completed": len(completed),
        "rows_saved_this_run": saved_rows,
        "errors_this_run": errors,
        "workers": args.workers,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("=" * 72, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    instance_lock.close()


if __name__ == "__main__":
    main()
