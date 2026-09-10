from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dual_ablation.audit import _status, build_basket_coverage_report, feature_quality_audit
from dual_ablation.features.ticker import _add_disclosures, _base_features
from dual_ablation.features.universe import _realestate_unsafe, _regimes


def test_bucket_with_fewer_than_five_tickers_is_not_ready() -> None:
    baskets = pd.DataFrame({"bucket": ["small"] * 4, "ticker": [f"{i:06d}" for i in range(4)]})
    rows = [
        {"date": "2025-01-02", "ticker": ticker, "label_abs_crash_20": i % 2}
        for i, ticker in enumerate(baskets["ticker"])
    ]
    report = build_basket_coverage_report(pd.DataFrame(rows), baskets)
    assert report.loc[0, "status"] == "insufficient_tickers"


def test_empty_crawl_result_is_never_success() -> None:
    status, message, ratio, unique = _status(pd.DataFrame(), "date", ["close"])
    assert status == "empty"
    assert message and ratio == 0 and unique == 0


def test_constant_feature_is_not_valid() -> None:
    dates = pd.date_range("2024-01-01", periods=220, freq="B")
    panel = pd.DataFrame({"date": dates, "x": 1.0})
    audit = feature_quality_audit(panel, {"ticker": {"t_price_trend": ["x"]}})
    assert audit.loc[0, "status"] == "constant"


def test_dart_event_becomes_available_after_disclosure_date(tmp_path) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    features = pd.DataFrame({"date": dates, "ticker": "005930"})
    events = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-03")], "ticker": ["005930"],
        "report_nm": ["영업실적 공시"],
    })
    events.to_parquet(tmp_path / "dart_disclosures.parquet", index=False)
    out = _add_disclosures(SimpleNamespace(raw_dual=tmp_path), features)
    assert out.loc[out["date"].eq(pd.Timestamp("2025-01-03")), "t_event_total_count_5"].item() == 0
    assert out.loc[out["date"].eq(pd.Timestamp("2025-01-06")), "t_event_total_count_5"].item() == 1


def test_unverified_realestate_is_unavailable_and_unsafe() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=220, freq="B"))
    frame = _realestate_unsafe(dates)
    cols = [c for c in frame if c != "date"]
    assert frame[cols].isna().all().all()
    audit = feature_quality_audit(
        frame,
        {"universe": {"u_realestate_korea": cols}},
        {"u_realestate_korea": "unsafe_timestamp"},
    )
    assert set(audit["status"]) == {"unsafe_timestamp"}


def test_regime_threshold_does_not_use_future_values() -> None:
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
    original = pd.DataFrame({
        "date": dates, "u_kospi_vol_20": np.linspace(0.1, 0.4, len(dates)),
        "u_vix_level": np.linspace(10, 30, len(dates)), "u_liquidity_stress_index": 0.0,
        "u_foreign_net_buy_ratio": 0.0, "u_us_10y_change_5": 0.0,
        "u_usdkrw_vol_20": 0.0, "u_kospi_drawdown_60": 0.0,
    })
    changed = original.copy()
    changed.loc[250:, "u_kospi_vol_20"] = 999.0
    left = _regimes(original.copy()).loc[:249, "u_regime_high_vol"]
    right = _regimes(changed.copy()).loc[:249, "u_regime_high_vol"]
    pd.testing.assert_series_equal(left, right)


def test_ticker_features_vary_by_ticker() -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    rows = []
    for ticker, closes in [("000001", [10, 11, 12, 13, 14]), ("000002", [10, 9, 8, 7, 6])]:
        for date, close in zip(dates, closes):
            rows.append({
                "date": date, "ticker": ticker, "name": ticker, "bucket": "x", "market": "KOSPI",
                "role": "x", "open": close, "high": close + 1, "low": close - 1,
                "close": close, "volume": 100,
            })
    out = _base_features(pd.DataFrame(rows))
    last = out.loc[out["date"].eq(dates[-1])].set_index("ticker")["t_price_ret_1"]
    assert last["000001"] > 0 and last["000002"] < 0

