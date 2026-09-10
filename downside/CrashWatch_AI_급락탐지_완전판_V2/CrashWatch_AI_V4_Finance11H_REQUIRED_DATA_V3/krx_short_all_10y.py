#!/usr/bin/env python
"""KRX 개별종목 공매도 거래 데이터를 재개 가능한 방식으로 수집한다.

KRX의 예전 OTP/CSV 경로는 현재 ``LOGOUT``을 반환한다. 이 수집기는
MDCSTAT301 화면이 실제로 사용하는 JSON 경로를 이용하며, 프로젝트의
활성 종목만 연도별 CSV에 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import json
import msvcrt
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


START_DATE = "2016-07-28"
END_DATE = "2026-07-27"
OUTPUT_FOLDER_NAME = "krx_short_selling_output"
DEFAULT_WORKERS = 2
DEFAULT_DELAY_SECONDS = 0.35
MAX_RETRIES = 6

BASE_URL = "https://data.krx.co.kr"
MAIN_URL = BASE_URL + "/contents/MDC/MAIN/main/index.cmd"
LOADER_URL = BASE_URL + "/comm/srt/srtLoader/index.cmd?screenId=MDCSTAT301"
REFERER_URL = BASE_URL + "/contents/MDC/STAT/srt/MDCSTAT301.jsp"
JSON_DATA_URL = BASE_URL + "/comm/bldAttendant/getJsonData.cmd"
DATASET = "dbms/MDC_OUT/STAT/srt/MDCSTAT30101_OUT"

OUTPUT_HEADER = [
    "거래일자",
    "시장구분",
    "종목코드",
    "종목명",
    "증권구분",
    "수량_공매도거래량_전체",
    "수량_공매도거래량_업틱룰적용",
    "수량_공매도거래량_업틱룰예외",
    "수량_거래량",
    "수량_비중",
    "금액_공매도거래대금_전체",
    "금액_공매도거래대금_업틱룰적용",
    "금액_공매도거래대금_업틱룰예외",
    "금액_거래대금",
    "금액_비중",
]

KRX_KEYS = [
    "ISU_CD",
    "ISU_ABBRV",
    "SECUGRP_NM",
    "CVSRTSELL_TRDVOL",
    "UPTICKRULE_APPL_TRDVOL",
    "UPTICKRULE_EXCPT_TRDVOL",
    "ACC_TRDVOL",
    "TRDVOL_WT",
    "CVSRTSELL_TRDVAL",
    "UPTICKRULE_APPL_TRDVAL",
    "UPTICKRULE_EXCPT_TRDVAL",
    "ACC_TRDVAL",
    "TRDVAL_WT",
]


class KrxDownloadError(RuntimeError):
    """KRX 응답을 정상 데이터로 해석할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class DownloadResult:
    trade_date: date
    rows: list[list[str]]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_weekdays(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(item) for item in payload.get("done", [])}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"진행 파일을 읽을 수 없습니다: {path}: {exc}") from exc


def save_progress(path: Path, completed: set[str]) -> None:
    atomic_json(
        path,
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "done": sorted(completed),
        },
    )


def acquire_instance_lock(output_dir: Path):
    """같은 출력 폴더에 수집기가 두 번 실행되는 것을 차단한다."""
    lock_path = output_dir / ".collector.lock"
    handle = lock_path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"같은 출력 폴더를 사용하는 수집기가 이미 실행 중입니다: {output_dir}"
        ) from exc
    return handle


def load_target_tickers(project: Path) -> dict[str, tuple[str, str]]:
    basket_path = project / "configs" / "sector_baskets.csv"
    if not basket_path.exists():
        raise FileNotFoundError(f"대상 종목 설정이 없습니다: {basket_path}")

    targets: dict[str, tuple[str, str]] = {}
    with basket_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            enabled = str(row.get("enabled", "1")).strip().lower()
            if enabled in {"0", "false", "no", "n"}:
                continue
            ticker = str(row.get("ticker", "")).strip().zfill(6)
            if len(ticker) != 6 or not ticker.isdigit():
                continue
            targets[ticker] = (
                str(row.get("name", "")).strip(),
                str(row.get("market", "")).strip(),
            )
    if not targets:
        raise RuntimeError(f"활성 대상 종목이 0개입니다: {basket_path}")
    return targets


class KrxClient:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        cookie_jar = http.cookiejar.CookieJar()
        context = ssl.create_default_context()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar),
            urllib.request.HTTPSHandler(context=context),
        )
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
            "Referer": REFERER_URL,
            "Origin": BASE_URL,
            "X-Requested-With": "XMLHttpRequest",
        }
        self.initialize_session()

    def initialize_session(self) -> None:
        for url in (MAIN_URL, LOADER_URL):
            request = urllib.request.Request(
                url,
                headers={**self.headers, "Accept": "text/html,*/*"},
                method="GET",
            )
            with self.opener.open(request, timeout=60) as response:
                response.read()

    @staticmethod
    def _retry_wait(attempt: int) -> float:
        return min(45.0, (2 ** (attempt - 1)) + random.uniform(0.4, 1.2))

    def _post(self, form: dict[str, str]) -> bytes:
        encoded = urllib.parse.urlencode(form).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            request = urllib.request.Request(
                JSON_DATA_URL,
                data=encoded,
                headers={
                    **self.headers,
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                },
                method="POST",
            )
            try:
                with self.opener.open(request, timeout=90) as response:
                    raw = response.read()
                if raw.decode("utf-8", errors="replace").strip() == "LOGOUT":
                    raise KrxDownloadError("KRX 세션 만료(LOGOUT)")
                return raw
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                last_error = exc
                retryable = exc.code in {403, 429, 500, 502, 503, 504}
                if exc.code == 400 and body == "LOGOUT":
                    retryable = True
                if not retryable:
                    raise KrxDownloadError(
                        f"HTTP {exc.code}: {body[:160] or exc.reason}"
                    ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                KrxDownloadError,
            ) as exc:
                last_error = exc

            if attempt < MAX_RETRIES:
                wait = self._retry_wait(attempt)
                time.sleep(wait)
                try:
                    self.initialize_session()
                except Exception:
                    pass

        raise KrxDownloadError(f"{MAX_RETRIES}회 재시도 후 실패: {last_error}")

    def download(self, trade_date: date) -> list[dict[str, str]]:
        raw = self._post(
            {
                "bld": DATASET,
                "locale": "ko_KR",
                "searchType": "1",
                "mktId": "ALL",
                "secugrpId": "STMFRTSCIFDRFS",
                "inqCond": "STMFRTSCIFDRFS",
                "trdDd": trade_date.strftime("%Y%m%d"),
                "share": "1",
                "money": "1",
            }
        )
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            preview = raw.decode("utf-8", errors="replace")[:160]
            raise KrxDownloadError(f"JSON이 아닌 응답: {preview!r}") from exc

        rows = payload.get("OutBlock_1")
        if not isinstance(rows, list):
            raise KrxDownloadError("OutBlock_1 배열이 없는 KRX 응답")
        if self.delay_seconds:
            time.sleep(self.delay_seconds + random.uniform(0.0, 0.12))
        return rows


def normalize_rows(
    trade_date: date,
    source_rows: list[dict[str, str]],
    targets: dict[str, tuple[str, str]],
) -> list[list[str]]:
    normalized: list[list[str]] = []
    for item in source_rows:
        ticker = str(item.get("ISU_CD", "")).strip().zfill(6)
        if ticker not in targets:
            continue
        configured_name, market = targets[ticker]
        values = [str(item.get(key, "")).strip() for key in KRX_KEYS]
        values[0] = ticker
        values[1] = configured_name or values[1]
        values[2] = "주식"
        normalized.append([trade_date.isoformat(), market, *values])
    return normalized


def append_year_csv(output_dir: Path, result: DownloadResult) -> int:
    if not result.rows:
        return 0
    path = output_dir / f"KRX_short_trade_{result.trade_date.year}.csv"
    is_new = not path.exists() or path.stat().st_size == 0
    encoding = "utf-8-sig" if is_new else "utf-8"
    with path.open("a", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(OUTPUT_HEADER)
        writer.writerows(result.rows)
    return len(result.rows)


def append_error(path: Path, trade_date: date, error: Exception) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().isoformat(timespec='seconds')}\t"
            f"{trade_date.isoformat()}\t{type(error).__name__}: {error}\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KRX 공매도 거래 10년치 수집(새 JSON 경로, 중단 재개 지원)"
    )
    parser.add_argument("--start", default=START_DATE, help="YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="프로젝트 48종목 필터를 해제하고 KRX 전체 종목을 저장",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = parse_date(args.start)
    end = parse_date(args.end)
    if start > end:
        raise ValueError("시작일이 종료일보다 늦습니다.")
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers는 KRX 부하 방지를 위해 1~4만 허용합니다.")

    project = Path(__file__).resolve().parent
    output_dir = (args.output_dir or project / OUTPUT_FOLDER_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_lock = acquire_instance_lock(output_dir)
    progress_path = output_dir / "progress.json"
    error_path = output_dir / "errors.log"
    targets = load_target_tickers(project)
    if args.all_stocks:
        targets = {}

    completed = load_progress(progress_path)
    dates = list(iter_weekdays(start, end))
    pending = [day for day in dates if day.strftime("%Y%m%d") not in completed]

    print("=" * 72, flush=True)
    print("KRX 개별종목 공매도 거래 수집", flush=True)
    print(f"기간          : {start} ~ {end}", flush=True)
    print(f"평일 요청     : {len(dates):,}건", flush=True)
    print(f"기존 완료     : {len(dates) - len(pending):,}건", flush=True)
    print(f"이번 실행     : {len(pending):,}건", flush=True)
    print(f"대상          : {'전체 종목' if args.all_stocks else f'활성 {len(targets)}종목'}", flush=True)
    print(f"동시 작업     : {args.workers}개", flush=True)
    print(f"출력          : {output_dir}", flush=True)
    print("=" * 72, flush=True)

    thread_state = threading.local()

    def collect(day: date) -> DownloadResult:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = KrxClient(args.delay)
            thread_state.client = client
        source = client.download(day)
        if args.all_stocks:
            all_targets = {
                str(row.get("ISU_CD", "")).strip().zfill(6): (
                    str(row.get("ISU_ABBRV", "")).strip(),
                    "",
                )
                for row in source
            }
            rows = normalize_rows(day, source, all_targets)
        else:
            rows = normalize_rows(day, source, targets)
        return DownloadResult(day, rows)

    started = time.time()
    saved_rows = 0
    errors = 0
    finished = len(dates) - len(pending)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map: dict[Future[DownloadResult], date] = {
                executor.submit(collect, day): day for day in pending
            }
            for future in as_completed(future_map):
                day = future_map[future]
                finished += 1
                prefix = f"[{finished:,}/{len(dates):,}] {day}"
                try:
                    result = future.result()
                    count = append_year_csv(output_dir, result)
                    saved_rows += count
                    completed.add(day.strftime("%Y%m%d"))
                    save_progress(progress_path, completed)
                    status = f"{count:,}행 저장" if count else "휴장/대상 데이터 없음"
                    print(f"{prefix} - {status}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    append_error(error_path, day, exc)
                    print(f"{prefix} - 오류: {exc}", flush=True)
    except KeyboardInterrupt:
        print("\n사용자 중단: 완료된 날짜까지 progress.json에 저장했습니다.", flush=True)
        raise

    elapsed = (time.time() - started) / 60.0
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requests_total": len(dates),
        "requests_completed": len(completed),
        "rows_saved_this_run": saved_rows,
        "errors_this_run": errors,
        "target_tickers": "all" if args.all_stocks else len(targets),
        "workers": args.workers,
        "elapsed_minutes": round(elapsed, 2),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("=" * 72, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    instance_lock.close()


if __name__ == "__main__":
    main()
