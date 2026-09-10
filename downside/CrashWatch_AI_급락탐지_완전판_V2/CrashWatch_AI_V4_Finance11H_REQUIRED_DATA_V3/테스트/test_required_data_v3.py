from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.config import get_paths
from dual_ablation.data_acquisition.krx_actual import _coalescing_merge, _normalize_percent_points
from dual_ablation.data_acquisition.macro import _attach_conservative_availability
from dual_ablation.data_acquisition.stock_lending import _normalize_lending
from dual_ablation.data_acquisition.validation import validate_required_data


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    pd.DataFrame([
        {"ticker": "005930", "name": "삼성전자", "bucket": "semiconductor", "enabled": 1},
    ]).to_csv(project / "configs" / "sector_baskets.csv", index=False, encoding="utf-8-sig")
    return project


def test_krx_percentage_points_are_unit_fractions() -> None:
    frame = pd.DataFrame({"short_volume_ratio": [0.06, 13.02, np.nan]})
    result = _normalize_percent_points(frame, ["short_volume_ratio"])
    assert np.isclose(result.loc[0, "short_volume_ratio"], 0.0006)
    assert np.isclose(result.loc[1, "short_volume_ratio"], 0.1302)


def test_krx_coalescing_merge_keeps_fallback_values() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    primary = pd.DataFrame({"date": dates, "value": [1.0, np.nan]})
    fallback = pd.DataFrame({"date": dates, "value": [9.0, 2.0], "other": [3.0, 4.0]})
    result = _coalescing_merge([primary, fallback])
    assert result["value"].tolist() == [1.0, 2.0]
    assert result["other"].tolist() == [3.0, 4.0]


def test_krx_combined_coalesces_complementary_licensed_partitions(
    tmp_path: Path,
) -> None:
    from dual_ablation.data_acquisition.krx_actual import build_krx_combined

    project = _project(tmp_path)
    paths = get_paths(project)
    source = (
        paths.raw_dual
        / "required_data_v3"
        / "krx_actual"
        / "sources"
        / "ticker=005930"
        / "source=licensed_csv_import"
    )
    source.mkdir(parents=True)
    date = pd.to_datetime(["2024-01-02"])
    pd.DataFrame({
        "date": date,
        "ticker": "005930",
        "short_trade_volume": [100.0],
        "short_trade_value": [1_000.0],
    }).to_parquet(source / "trade.parquet", index=False)
    pd.DataFrame({
        "date": date,
        "ticker": "005930",
        "short_balance_shares": [200.0],
        "short_balance_value": [2_000.0],
        "market_cap": [10_000.0],
    }).to_parquet(source / "balance.parquet", index=False)

    result = build_krx_combined(paths)
    assert len(result) == 1
    assert result.loc[0, "short_trade_volume"] == 100.0
    assert result.loc[0, "short_balance_shares"] == 200.0
    assert result.loc[0, "market_cap"] == 10_000.0


def test_stock_lending_normalization_preserves_participant() -> None:
    raw = pd.DataFrame({
        "basDt": ["20240102"],
        "stckItmsCd": ["5930"],
        "stckLnbCntrVol": ["100"],
        "stckLnbBlncAmt": ["500000"],
        "invpnNm": ["외국인"],
        "source_endpoint": ["getStckLnbInvpnDetail"],
    })
    result = _normalize_lending(raw)
    assert result.loc[0, "ticker"] == "005930"
    assert result.loc[0, "lending_contract_shares"] == 100
    assert result.loc[0, "lending_balance_value"] == 500000
    assert result.loc[0, "participant_name"] == "외국인"


def test_stock_lending_current_public_api_field_names() -> None:
    raw = pd.DataFrame({
        "basDt": ["20240102"],
        "stckItmsCd": ["005930"],
        "stckItmsNm": ["삼성전자"],
        "cclStckCnt": ["100"],
        "rdptStckCnt": ["90"],
        "balnStckCnt": ["1000"],
        "balnStckAmt": ["500000"],
        "invpnClsfNm": ["외국인"],
        "invpnClsfDtlNm": ["외국계"],
    })
    result = _normalize_lending(raw)
    assert result.loc[0, "name"] == "삼성전자"
    assert result.loc[0, "lending_contract_shares"] == 100
    assert result.loc[0, "lending_repayment_shares"] == 90
    assert result.loc[0, "lending_balance_shares"] == 1000
    assert result.loc[0, "lending_balance_value"] == 500000
    assert result.loc[0, "participant_type"] == "외국인"
    assert result.loc[0, "participant_name"] == "외국계"


def test_macro_availability_is_next_business_day() -> None:
    raw = pd.DataFrame({"date": pd.to_datetime(["2024-01-05"]), "vix": [20.0]})  # Friday
    result = _attach_conservative_availability(raw)
    assert result.loc[0, "observation_date"] == pd.Timestamp("2024-01-05")
    assert result.loc[0, "available_from"] == pd.Timestamp("2024-01-08")


def test_lending_never_counts_as_actual_short(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    paths = get_paths(project)
    root = paths.raw_dual / "required_data_v3"
    lending_root = root / "stock_lending"
    lending_root.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2020-01-01", periods=800)
    lending = pd.DataFrame({
        "date": dates,
        "ticker": "005930",
        "lending_contract_shares": 100.0,
        "lending_repayment_shares": 90.0,
        "lending_balance_shares": 1000.0,
        "lending_balance_value": 1_000_000.0,
    })
    import dual_ablation.data_acquisition.validation as validation_module

    def fake_read(path: Path) -> pd.DataFrame:
        if path.name == "stock_lending_48_tickers_daily.parquet":
            return lending.copy()
        return pd.DataFrame()

    monkeypatch.setattr(validation_module, "_read_if_exists", fake_read)
    result = validate_required_data(paths, strict=False)
    assert result["stock_lending"]["ready_tickers"] == 1
    assert result["actual_short"]["ready_tickers"] == 0
    assert result["actual_short"]["is_actual_short"] is False
    assert result["finance11h_strict_ready"] is False


def test_dart_correction_chain_tracks_superseded_receipts() -> None:
    from dual_ablation.data_acquisition.dart_pit import _add_correction_chain

    history = pd.DataFrame({
        "ticker": ["005930", "005930"],
        "rcept_dt": pd.to_datetime(["2024-03-10", "2024-03-20"]),
        "rcept_no": ["20240310000001", "20240320000001"],
        "report_nm": ["사업보고서 (2023.12)", "[정정]사업보고서 (2023.12)"],
        "regular_report_type": ["annual", "annual"],
        "report_period_end": ["2023.12", "2023.12"],
    })
    result = _add_correction_chain(history).sort_values("rcept_dt").reset_index(drop=True)
    assert result.loc[0, "correction_sequence"] == 0
    assert result.loc[1, "correction_sequence"] == 1
    assert result.loc[1, "supersedes_receipt_no"] == "20240310000001"
    assert bool(result.loc[1, "is_latest_correction"])


def test_required_runner_continues_after_krx_failure(tmp_path: Path, monkeypatch) -> None:
    import dual_ablation.data_acquisition.runner as runner_module

    project = _project(tmp_path)
    calls: list[str] = []

    def failed_krx(*args, **kwargs):
        calls.append("krx")
        raise RuntimeError("auth failed")

    def success_macro(*args, **kwargs):
        calls.append("macro")
        return {"rows": 1}

    def fake_validate(*args, **kwargs):
        return {"finance11h_strict_ready": False}

    monkeypatch.setattr(runner_module, "collect_krx_actual_data", failed_krx)
    monkeypatch.setattr(runner_module, "collect_macro_credit", success_macro)
    monkeypatch.setattr(runner_module, "validate_required_data", fake_validate)
    result = runner_module.run_required_data_download(
        project,
        "2024-01-01",
        "2024-01-31",
        sources=["krx", "macro"],
        strict=False,
    )
    assert calls == ["krx", "macro"]
    assert result["steps"]["krx"]["status"] == "failed"
    assert result["steps"]["macro"]["status"] == "completed"


def test_licensed_krx_csv_multi_ticker_and_percent_units(tmp_path: Path) -> None:
    from dual_ablation.data_acquisition.krx_csv_import import _normalize_file

    path = tmp_path / "krx.csv"
    pd.DataFrame({
        "일자": ["2024-01-02", "2024-01-02"],
        "종목코드": ["005930", "000660"],
        "공매도수량": [100, 200],
        "공매도거래대금": [1_000_000, 2_000_000],
        "공매도비중": [0.06, 2.5],
        "공매도잔고수량": [1000, 2000],
        "공매도잔고금액": [10_000_000, 20_000_000],
        "시가총액": [100_000_000, 200_000_000],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    result = _normalize_file(path, {"005930", "000660"}, "percent")
    assert set(result["ticker"]) == {"005930", "000660"}
    assert np.isclose(result.loc[result["ticker"].eq("005930"), "short_volume_ratio"].iloc[0], 0.0006)
    assert np.isclose(result.loc[result["ticker"].eq("000660"), "short_volume_ratio"].iloc[0], 0.025)


def test_krx_marketplace_snapshot_uses_filename_date_and_composite_headers(tmp_path: Path) -> None:
    from dual_ablation.data_acquisition.krx_csv_import import _normalize_file

    path = tmp_path / "data_5510_20260728.csv"
    pd.DataFrame({
        "종목코드": ["005930"],
        "종목명": ["삼성전자"],
        "수량_공매도거래량_전체": [123],
        "수량_거래량": [1000],
        "수량_비중": [12.3],
        "금액_공매도거래대금_전체": [12_300_000],
        "금액_거래대금": [100_000_000],
        "금액_비중": [12.3],
    }).to_csv(path, index=False, encoding="utf-8-sig")

    result = _normalize_file(path, {"005930"}, "percent")
    assert result.loc[0, "date"] == pd.Timestamp("2026-07-28")
    assert result.loc[0, "short_trade_volume"] == 123
    assert result.loc[0, "short_trade_value"] == 12_300_000
    assert np.isclose(result.loc[0, "short_volume_ratio"], 0.123)
    assert np.isclose(result.loc[0, "short_value_ratio"], 0.123)


def test_new_krx_trade_json_rows_keep_only_project_tickers() -> None:
    from datetime import date

    from krx_short_all_10y import normalize_rows

    source = [
        {
            "ISU_CD": "005930",
            "ISU_ABBRV": "깨진서버명",
            "SECUGRP_NM": "깨진서버구분",
            "CVSRTSELL_TRDVOL": "100",
            "UPTICKRULE_APPL_TRDVOL": "90",
            "UPTICKRULE_EXCPT_TRDVOL": "10",
            "ACC_TRDVOL": "1,000",
            "TRDVOL_WT": "10.00",
            "CVSRTSELL_TRDVAL": "7,000",
            "UPTICKRULE_APPL_TRDVAL": "6,000",
            "UPTICKRULE_EXCPT_TRDVAL": "1,000",
            "ACC_TRDVAL": "70,000",
            "TRDVAL_WT": "10.00",
        },
        {"ISU_CD": "999999"},
    ]
    result = normalize_rows(
        date(2026, 7, 23),
        source,
        {"005930": ("삼성전자", "KOSPI")},
    )
    assert len(result) == 1
    assert result[0][:5] == [
        "2026-07-23",
        "KOSPI",
        "005930",
        "삼성전자",
        "주식",
    ]


def test_new_krx_balance_json_rows_keep_required_columns() -> None:
    from datetime import date

    from krx_short_balance_all_10y import collect_day

    class FakeClient:
        delay_seconds = 0

        def _post(self, _form):
            return json.dumps({
                "OutBlock_1": [
                    {
                        "ISU_CD": "005930",
                        "BAL_QTY": "100",
                        "LIST_SHRS": "1,000",
                        "BAL_AMT": "7,000",
                        "MKTCAP": "70,000",
                        "BAL_RTO": "10.00",
                    },
                    {"ISU_CD": "999999"},
                ]
            }).encode()

    result = collect_day(
        FakeClient(),
        date(2026, 7, 23),
        "KOSPI",
        {"005930": ("삼성전자", "KOSPI")},
    )
    assert result.rows == [[
        "2026-07-23",
        "KOSPI",
        "005930",
        "삼성전자",
        "100",
        "1,000",
        "7,000",
        "70,000",
        "10.00",
    ]]


def test_krx_progress_atomic_replace_retries_windows_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import krx_short_all_10y as collector

    destination = tmp_path / "progress.json"
    real_replace = collector.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("temporary Windows lock")
        return real_replace(source, target)

    monkeypatch.setattr(collector.os, "replace", flaky_replace)
    collector.atomic_json(destination, {"done": ["20260723"]})
    assert attempts["count"] == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "done": ["20260723"]
    }


def test_finance_feature_block_keeps_short_and_lending_separate() -> None:
    from dual_ablation.finance11h.features import _build_ticker_block

    dates = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame({
        "date": dates,
        "ticker": "005930",
        "close": np.linspace(70_000, 75_000, len(dates)),
        "volume": 1_000_000.0,
        "trading_value": 70_000_000_000.0,
        "fv_market_cap": 400_000_000_000_000.0,
        "fv_listed_shares": 5_969_782_550.0,
        "fs_short_trade_volume": np.linspace(10_000, 20_000, len(dates)),
        "fs_short_trade_value": np.linspace(700_000_000, 1_500_000_000, len(dates)),
        "fs_short_balance_shares": np.linspace(3_000_000, 4_000_000, len(dates)),
        "fs_short_balance_value": np.linspace(210_000_000_000, 300_000_000_000, len(dates)),
        "fl_lending_contract_shares": np.linspace(20_000, 40_000, len(dates)),
        "fl_lending_repayment_shares": np.linspace(10_000, 30_000, len(dates)),
        "fl_lending_balance_shares": np.linspace(5_000_000, 7_000_000, len(dates)),
        "fl_lending_balance_value": np.linspace(350_000_000_000, 520_000_000_000, len(dates)),
    })
    result = _build_ticker_block(frame)
    assert "t_finshort_balance_to_cap" in result
    assert "t_lending_balance_to_cap" in result
    assert result["t_finshort_balance_to_cap"].notna().all()
    assert result["t_lending_balance_to_cap"].notna().all()
    assert not np.allclose(
        result["t_finshort_balance_to_cap"].to_numpy(),
        result["t_lending_balance_to_cap"].to_numpy(),
    )
