from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, normalize_ticker
from .common import now_iso, read_enabled_tickers
from .krx_actual import summarize_krx_actual
from .stock_lending import summarize_stock_lending

ACTUAL_SHORT_REQUIRED = [
    "short_trade_volume",
    "short_trade_value",
    "short_volume_ratio",
    "short_balance_shares",
    "short_balance_value",
    "market_cap",
]
INVESTOR_REQUIRED = [
    "net_value_foreign",
    "net_value_institution",
    "net_value_individual",
    "foreign_owned_shares",
    "foreign_ownership_rate",
]
MINIMUM_ROWS = 600
MINIMUM_SPAN_DAYS = 900
MINIMUM_NON_NULL_RATIO = 0.50


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def _coverage(block: pd.DataFrame, required: list[str]) -> tuple[dict[str, float], int]:
    if block.empty or "date" not in block:
        return ({column: 0.0 for column in required}, 0)
    start_date = pd.to_datetime(block["date"], errors="coerce").min()
    end_date = pd.to_datetime(block["date"], errors="coerce").max()
    span_days = int((end_date - start_date).days) if pd.notna(start_date) and pd.notna(end_date) else 0
    ratios = {
        column: float(block[column].notna().mean()) if column in block.columns and len(block) else 0.0
        for column in required
    }
    return ratios, span_days


def _ready(block: pd.DataFrame, required: list[str]) -> tuple[bool, dict[str, float], int]:
    ratios, span_days = _coverage(block, required)
    minimum_ratio = min(ratios.values(), default=0.0)
    ready = len(block) >= MINIMUM_ROWS and span_days >= MINIMUM_SPAN_DAYS and minimum_ratio >= MINIMUM_NON_NULL_RATIO
    return ready, ratios, span_days


def validate_required_data(paths: ProjectPaths, *, strict: bool = False) -> dict[str, Any]:
    root = paths.raw_dual / "required_data_v3"
    root.mkdir(parents=True, exist_ok=True)
    baskets = read_enabled_tickers(paths)
    tickers = baskets["ticker"].tolist()

    actual = _read_if_exists(root / "krx_actual" / "krx_actual_ticker_daily.parquet")
    if not actual.empty:
        actual["ticker"] = normalize_ticker(actual["ticker"])
        actual["date"] = pd.to_datetime(actual["date"], errors="coerce")
        actual = actual.loc[actual["date"].notna()]
    lending = _read_if_exists(root / "stock_lending" / "stock_lending_48_tickers_daily.parquet")
    if not lending.empty:
        lending["ticker"] = normalize_ticker(lending["ticker"])
        lending["date"] = pd.to_datetime(lending["date"], errors="coerce")
        lending = lending.loc[lending["date"].notna()]
    naver = _read_if_exists(root / "naver_flow_fallback" / "naver_investor_foreign_48_tickers.parquet")
    if not naver.empty:
        naver["ticker"] = normalize_ticker(naver["ticker"])
        if "date" in naver:
            naver["date"] = pd.to_datetime(naver["date"], errors="coerce")
    dart_history = _read_if_exists(root / "dart_point_in_time" / "regular_report_history_48_tickers.parquet")
    dart_financial = _read_if_exists(root / "dart_point_in_time" / "financial_statements_conservative_pit_48_tickers.parquet")
    macro = _read_if_exists(root / "macro_credit" / "macro_credit_daily.parquet")

    coverage_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        block = actual.loc[actual["ticker"].eq(ticker)] if not actual.empty else pd.DataFrame()
        lend_block = lending.loc[lending["ticker"].eq(ticker)] if not lending.empty else pd.DataFrame()
        naver_block = naver.loc[naver["ticker"].eq(ticker)] if not naver.empty else pd.DataFrame()
        hist_block = (
            dart_history.loc[dart_history.get("ticker", pd.Series(dtype=str)).astype(str).str.zfill(6).eq(ticker)]
            if not dart_history.empty else pd.DataFrame()
        )
        fin_block = (
            dart_financial.loc[dart_financial.get("ticker", pd.Series(dtype=str)).astype(str).str.zfill(6).eq(ticker)]
            if not dart_financial.empty else pd.DataFrame()
        )

        short_ready, short_ratios, short_span = _ready(block, ACTUAL_SHORT_REQUIRED)
        investor_ready, investor_ratios, investor_span = _ready(block, INVESTOR_REQUIRED)
        lending_ready, lending_ratios, lending_span = _ready(
            lend_block,
            ["lending_contract_shares", "lending_repayment_shares", "lending_balance_shares", "lending_balance_value"],
        )
        dart_available = (
            pd.to_datetime(fin_block.get("available_from", pd.Series(dtype="datetime64[ns]")), errors="coerce").notna().mean()
            if len(fin_block) else 0.0
        )
        dart_ready = len(hist_block) >= 4 and len(fin_block) > 0 and dart_available >= 0.95
        coverage_rows.append({
            "ticker": ticker,
            "actual_short_rows": len(block),
            "actual_short_start": str(block["date"].min().date()) if len(block) else "",
            "actual_short_end": str(block["date"].max().date()) if len(block) else "",
            "actual_short_span_days": short_span,
            **{f"actual_{key}_nonnull": value for key, value in short_ratios.items()},
            "actual_short_status": "ready" if short_ready else "insufficient",
            "investor_span_days": investor_span,
            **{f"investor_{key}_nonnull": value for key, value in investor_ratios.items()},
            "investor_flow_status": "ready" if investor_ready else "insufficient",
            "lending_rows": len(lend_block),
            "lending_span_days": lending_span,
            **{f"lending_{key}_nonnull": value for key, value in lending_ratios.items()},
            "lending_status": "ready" if lending_ready else "insufficient",
            "naver_fallback_rows": len(naver_block),
            "dart_history_rows": len(hist_block),
            "dart_financial_rows": len(fin_block),
            "dart_available_from_ratio": float(dart_available),
            "dart_status": "ready" if dart_ready else "insufficient",
        })
    coverage = pd.DataFrame(coverage_rows)
    atomic_csv(coverage, root / "required_data_coverage_48_tickers.csv")

    actual_short_ready = int(coverage["actual_short_status"].eq("ready").sum()) if len(coverage) else 0
    investor_ready = int(coverage["investor_flow_status"].eq("ready").sum()) if len(coverage) else 0
    lending_ready = int(coverage["lending_status"].eq("ready").sum()) if len(coverage) else 0
    dart_ready = int(coverage["dart_status"].eq("ready").sum()) if len(coverage) else 0

    macro_required = ["usdkrw_fred", "vix", "treasury_3y", "corp_bond_aa_minus"]
    macro_status = {
        column: bool(column in macro.columns and pd.to_numeric(macro[column], errors="coerce").notna().sum() >= 600)
        for column in macro_required
    }
    macro_pit_status = bool(
        len(macro)
        and "available_from" in macro.columns
        and pd.to_datetime(macro["available_from"], errors="coerce").notna().mean() >= 0.95
    )

    public_sample_path = paths.raw_dual / "stock_lending_public"
    public_sample_files = [str(path.relative_to(paths.project)) for path in public_sample_path.glob("*.csv")] if public_sample_path.exists() else []

    # Reuse source-level summaries as an additional consistency check.
    actual_summary = summarize_krx_actual(paths, actual)
    lending_summary = summarize_stock_lending(paths, lending)
    strict_short_coverage = actual_short_ready / len(tickers) if tickers else 0.0
    summary = {
        "created_at": now_iso(),
        "ticker_count": len(tickers),
        "ready_rule": {
            "minimum_rows": MINIMUM_ROWS,
            "minimum_span_days": MINIMUM_SPAN_DAYS,
            "minimum_non_null_ratio": MINIMUM_NON_NULL_RATIO,
        },
        "actual_short": {
            "ready_tickers": actual_short_ready,
            "coverage": strict_short_coverage,
            "strict_required_coverage": 0.80,
            "source": "KRX via authenticated pykrx session or imported licensed KRX CSV",
            "is_actual_short": bool(actual_summary.get("actual_short_data")),
            "source_summary": actual_summary,
        },
        "investor_flow_foreign_ownership": {
            "ready_tickers": investor_ready,
            "coverage": investor_ready / len(tickers) if tickers else 0.0,
            "naver_fallback_rows": len(naver),
            "naver_is_fallback_only": True,
        },
        "stock_lending": {
            "ready_tickers": lending_ready,
            "coverage": lending_ready / len(tickers) if tickers else 0.0,
            "is_separate_from_short_selling": True,
            "source_summary": lending_summary,
        },
        "dart_point_in_time": {
            "ready_tickers": dart_ready,
            "coverage": dart_ready / len(tickers) if tickers else 0.0,
            "original_documents_preserved": (root / "dart_point_in_time" / "original_documents").exists(),
            "structured_values_are_conservative_latest_correction": True,
        },
        "macro": {
            "rows": len(macro),
            "required_series": macro_status,
            "available_from_ready": macro_pit_status,
            "ready": all(macro_status.values()) and macro_pit_status,
        },
        "included_public_sample_files": public_sample_files,
        "finance11h_strict_ready": bool(
            strict_short_coverage >= 0.80
            and investor_ready / len(tickers) >= 0.80
            and lending_ready / len(tickers) >= 0.80
            and dart_ready / len(tickers) >= 0.80
            and all(macro_status.values())
            and macro_pit_status
        ) if tickers else False,
        "finance11h_block_reasons": [],
    }
    if strict_short_coverage < 0.80:
        summary["finance11h_block_reasons"].append(f"actual_short_coverage={strict_short_coverage:.1%}<80%")
    if investor_ready / len(tickers) < 0.80:
        summary["finance11h_block_reasons"].append(f"investor_flow_coverage={investor_ready / len(tickers):.1%}<80%")
    if lending_ready / len(tickers) < 0.80:
        summary["finance11h_block_reasons"].append(f"stock_lending_coverage={lending_ready / len(tickers):.1%}<80%")
    if dart_ready / len(tickers) < 0.80:
        summary["finance11h_block_reasons"].append(f"dart_pit_coverage={dart_ready / len(tickers):.1%}<80%")
    if not all(macro_status.values()) or not macro_pit_status:
        summary["finance11h_block_reasons"].append("macro_required_series_or_available_from_missing")

    atomic_json(summary, root / "required_data_summary.json")
    if strict and not summary["finance11h_strict_ready"]:
        raise RuntimeError(
            "Finance11H 본 실험 차단 유지: " + "; ".join(summary["finance11h_block_reasons"])
            + ". 로컬 인증 수집 또는 합법적인 KRX CSV import가 필요합니다."
        )
    return summary
