#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""인터넷/API 키 없이 수집기의 표준화·GDELT 파싱·yfinance 다중열 처리를 검사한다."""
from __future__ import annotations
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

def load():
    spec = importlib.util.spec_from_file_location("collector_mock_module", BASE / "01_데이터_크롤링.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collector_mock_module"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main():
    m = load()
    idx = pd.date_range("2026-01-01", periods=3)
    raw = pd.DataFrame({"시가": [1,2,3], "종가": [2,3,4], "비중": [1.1,1.2,1.3]}, index=idx)
    out = m.normalize_indexed_frame(raw, "short")
    assert {"date", "short_open", "short_close", "short_ratio_pct"} <= set(out.columns)

    payload = {"timeline": [{"data": [{"date": "20260701T000000Z", "value": 12, "norm": 100}]}]}
    parsed = m._parse_gdelt_timeline(payload, "market", "query")
    assert len(parsed) == 1 and parsed[0]["news_volume"] == 12

    with tempfile.TemporaryDirectory() as td:
        old_raw, old_overwrite = m.RAW_DIR, m.OVERWRITE
        old_download = m.yf.download
        m.RAW_DIR = Path(td)
        m.OVERWRITE = True
        dates = pd.date_range("2026-01-01", periods=4)
        symbols = list(m.CROSS_ASSETS.values())
        cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], symbols])
        fake = pd.DataFrame(
            np.arange(len(dates) * len(cols), dtype=float).reshape(len(dates), len(cols)) + 1,
            index=dates,
            columns=cols,
        )
        m.yf.download = lambda **kwargs: fake
        m.collect_cross_assets()
        saved = pd.read_parquet(m.RAW_DIR / "cross_assets.parquet")
        assert saved["alias"].nunique() == len(m.CROSS_ASSETS)
        assert "close" in saved.columns
        m.yf.download = old_download
        m.RAW_DIR, m.OVERWRITE = old_raw, old_overwrite

    print("크롤링 모의테스트 통과")
    print("- KRX 한글열 표준화")
    print("- GDELT timeline JSON 파싱")
    print("- yfinance MultiIndex 24개 자산 분해")

if __name__ == "__main__":
    main()
