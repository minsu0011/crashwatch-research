from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_ticker
from .common import now_iso, read_enabled_tickers
from .krx_actual import build_krx_combined, summarize_krx_actual

COLUMN_ALIASES = {
    # KRX Data Marketplace "개별종목 공매도 거래" download headers.
    "수량_공매도거래량_전체": "short_trade_volume",
    "수량_공매도거래량_업틱룰적용": "short_trade_volume_uptick",
    "수량_공매도거래량_업틱룰예외": "short_trade_volume_exempt",
    "수량_거래량": "total_trade_volume",
    "수량_비중": "short_volume_ratio",
    "금액_공매도거래대금_전체": "short_trade_value",
    "금액_공매도거래대금_업틱룰적용": "short_trade_value_uptick",
    "금액_공매도거래대금_업틱룰예외": "short_trade_value_exempt",
    "금액_거래대금": "market_trading_value",
    "금액_비중": "short_value_ratio",
    "일자": "date",
    "날짜": "date",
    "거래일": "date",
    "거래일자": "date",
    "기준일": "date",
    "기준일자": "date",
    "종목코드": "ticker",
    "단축코드": "ticker",
    "티커": "ticker",
    "종목명": "name",
    "공매도": "short_trade_volume",
    "공매도수량": "short_trade_volume",
    "공매도거래량": "short_trade_volume",
    "공매도거래수량": "short_trade_volume",
    "매수": "total_trade_volume",
    "전체거래량": "total_trade_volume",
    "총거래량": "total_trade_volume",
    "공매도금액": "short_trade_value",
    "공매도거래대금": "short_trade_value",
    "공매도거래금액": "short_trade_value",
    "전체거래대금": "market_trading_value",
    "총거래대금": "market_trading_value",
    "거래대금": "market_trading_value",
    "비중": "short_volume_ratio",
    "공매도비중": "short_volume_ratio",
    "공매도수량비중": "short_volume_ratio",
    "공매도거래비중": "short_volume_ratio",
    "공매도거래량비중": "short_volume_ratio",
    "공매도금액비중": "short_value_ratio",
    "공매도거래대금비중": "short_value_ratio",
    "거래량비중": "short_volume_ratio",
    "거래대금비중": "short_value_ratio",
    "공매도잔고": "short_balance_shares",
    "공매도잔고수량": "short_balance_shares",
    "순보유잔고수량": "short_balance_shares",
    "잔고": "short_balance_shares",
    "잔고수량": "short_balance_shares",
    "공매도잔고금액": "short_balance_value",
    "순보유잔고금액": "short_balance_value",
    "잔고금액": "short_balance_value",
    "상장주식수": "listed_shares",
    "시가총액": "market_cap",
    "잔고비중": "short_balance_ratio",
    "공매도잔고비중": "short_balance_ratio",
    "순보유잔고비율": "short_balance_ratio",
    "기관합계": "net_value_institution",
    "기관": "net_value_institution",
    "외국인합계": "net_value_foreign",
    "외국인": "net_value_foreign",
    "개인": "net_value_individual",
    "보유수량": "foreign_owned_shares",
    "외국인보유수량": "foreign_owned_shares",
    "외국인보유주식수": "foreign_owned_shares",
    "지분율": "foreign_ownership_rate",
    "외국인지분율": "foreign_ownership_rate",
    "외국인보유율": "foreign_ownership_rate",
}
RATIO_COLUMNS = ["short_volume_ratio", "short_value_ratio", "short_balance_ratio", "foreign_ownership_rate"]


def _compact(text: Any) -> str:
    return re.sub(r"[\s_()%％/\-]+", "", str(text).replace("\ufeff", "").strip())


def _canonical(text: Any) -> str:
    compact = _compact(text)
    for alias, canonical in COLUMN_ALIASES.items():
        if _compact(alias) == compact:
            return canonical
    return str(text).replace("\ufeff", "").strip()


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}:{exc}")
    raise RuntimeError("CSV 인코딩 실패: " + " | ".join(errors))


def _ticker_from_filename(path: Path) -> str:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", path.name)
    return match.group(1) if match else ""


def _date_from_filename(path: Path) -> pd.Timestamp:
    matches = re.findall(r"(?<!\d)((?:19|20)\d{6})(?!\d)", path.stem)
    if not matches:
        return pd.NaT
    return pd.to_datetime(matches[-1], format="%Y%m%d", errors="coerce")


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    compact = text.str.replace("-", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(text, errors="coerce"))


def _normalize_ratio_units(frame: pd.DataFrame, ratio_unit: Literal["percent", "fraction", "auto"]) -> pd.DataFrame:
    out = frame.copy()
    for column in RATIO_COLUMNS:
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if ratio_unit == "percent":
            values = values / 100.0
        elif ratio_unit == "auto":
            finite = values.dropna().abs()
            # Official KRX CSVs normally use percent points. Values above one
            # are unambiguous; small values are left unchanged in auto mode.
            if not finite.empty and finite.quantile(0.95) > 1.0:
                values = values / 100.0
        out[column] = values
    return out


def _normalize_file(
    path: Path,
    valid_tickers: set[str],
    ratio_unit: Literal["percent", "fraction", "auto"],
) -> pd.DataFrame:
    frame = _read_csv_flexible(path)
    frame.columns = [_canonical(column) for column in frame.columns]
    if "date" not in frame:
        inferred_date = _date_from_filename(path)
        if pd.isna(inferred_date):
            raise RuntimeError("date/일자 열이 없고 파일명에도 YYYYMMDD 날짜가 없음")
        frame["date"] = inferred_date
    frame["date"] = _parse_dates(frame["date"])
    frame = frame.loc[frame["date"].notna()].copy()

    if "ticker" in frame:
        frame["ticker"] = normalize_ticker(frame["ticker"])
    else:
        ticker = _ticker_from_filename(path)
        if not ticker:
            raise RuntimeError("종목코드 열과 파일명의 6자리 종목코드가 모두 없음")
        frame["ticker"] = ticker

    frame = frame.loc[frame["ticker"].isin(valid_tickers)].copy()
    if frame.empty:
        raise RuntimeError("48종목에 해당하는 행 없음")

    for column in frame.columns:
        if column not in {"date", "ticker", "name"}:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
                errors="coerce",
            )
    frame = _normalize_ratio_units(frame, ratio_unit)
    recognized = [
        column
        for column in frame.columns
        if column.startswith("short_")
        or column.startswith("net_value_")
        or column in {"market_cap", "listed_shares", "market_trading_value", "foreign_owned_shares", "foreign_ownership_rate"}
    ]
    if not recognized:
        raise RuntimeError("공매도/수급/외국인보유/시가총액 인식 열 없음")
    return frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


def import_licensed_krx_csv(
    paths: ProjectPaths,
    input_dir: Path,
    *,
    overwrite: bool = False,
    ratio_unit: Literal["percent", "fraction", "auto"] = "percent",
) -> dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if ratio_unit not in {"percent", "fraction", "auto"}:
        raise ValueError(f"ratio_unit 오류: {ratio_unit}")

    root = paths.raw_dual / "required_data_v3" / "krx_actual" / "sources_imported_csv"
    root.mkdir(parents=True, exist_ok=True)
    baskets = read_enabled_tickers(paths)
    valid_tickers = set(baskets["ticker"])
    results: list[dict[str, Any]] = []
    actual_source_root = paths.raw_dual / "required_data_v3" / "krx_actual" / "sources"

    for path in sorted(input_dir.rglob("*.csv")):
        try:
            normalized = _normalize_file(path, valid_tickers, ratio_unit)
            file_rows = 0
            imported_tickers: list[str] = []
            for ticker, block in normalized.groupby("ticker", sort=True):
                ticker = str(ticker).zfill(6)
                saved_path = root / f"ticker={ticker}" / f"{path.stem}.parquet"
                # Keep the partition filename short enough for the legacy
                # Windows MAX_PATH limit.  The project path is already long,
                # and appending the original KRX filename plus ".tmp" can
                # otherwise exceed 260 characters during atomic writes.
                file_date = _date_from_filename(path)
                short_token = (
                    file_date.strftime("%Y%m%d")
                    if pd.notna(file_date)
                    else hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
                )
                source_path = (
                    actual_source_root
                    / f"ticker={ticker}"
                    / "source=licensed_csv_import"
                    / f"d{short_token}.parquet"
                )
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.parent.mkdir(parents=True, exist_ok=True)

                saved = block.copy()
                if saved_path.exists() and not overwrite:
                    old = pd.read_parquet(saved_path)
                    saved = pd.concat([old, saved], ignore_index=True, sort=False)
                saved = saved.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
                atomic_parquet(saved, saved_path)

                source_saved = saved.copy()
                if source_path.exists() and not overwrite:
                    old_source = pd.read_parquet(source_path)
                    source_saved = pd.concat([old_source, source_saved], ignore_index=True, sort=False)
                source_saved = source_saved.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
                atomic_parquet(source_saved, source_path)
                file_rows += len(saved)
                imported_tickers.append(ticker)

            results.append({
                "input": str(path),
                "tickers": "|".join(imported_tickers),
                "ticker_count": len(imported_tickers),
                "status": "success",
                "rows": file_rows,
                "ratio_unit": ratio_unit,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "input": str(path),
                "tickers": "",
                "ticker_count": 0,
                "status": "failed",
                "rows": 0,
                "ratio_unit": ratio_unit,
                "error": f"{type(exc).__name__}: {exc}",
            })

    build_krx_combined(paths)
    summary = summarize_krx_actual(paths)
    summary.update({
        "created_at": now_iso(),
        "input_dir": str(input_dir),
        "files_total": len(results),
        "files_success": sum(row["status"] == "success" for row in results),
        "files_failed": sum(row["status"] == "failed" for row in results),
        "ratio_unit": ratio_unit,
    })
    output_root = paths.raw_dual / "required_data_v3" / "krx_actual"
    atomic_csv(pd.DataFrame(results), output_root / "licensed_csv_import_manifest.csv")
    atomic_json(summary, output_root / "licensed_csv_import_summary.json")
    return summary
