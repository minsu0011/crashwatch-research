#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main():
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    f = load("feature_test_module", "02_피쳐_생성_봉인.py")
    a = load("ablation_test_module", "03_이탈테스트_봉인인증.py")

    # 연속일수
    s = pd.Series([0.1, 0.2, -0.1, -0.2, -0.3, 0.1])
    assert f.positive_streak(s).tolist() == [1, 2, 0, 0, 0, 1]
    assert f.negative_streak(s).tolist() == [0, 0, 1, 2, 3, 0]

    # 미래 라벨이 현재를 포함하지 않는지
    x = pd.Series([100, 50, 100, 100, 100], dtype=float)
    fm = f.future_min(x, 2)
    assert fm.iloc[0] == 50
    assert fm.iloc[1] == 100

    # 뉴스 시점 정렬: 18시 이전 당일, 이후 다음 거래일
    dates = pd.bdate_range("2026-01-05", periods=5)
    assert f._news_available_date(pd.Timestamp("2026-01-05 17:00"), dates) == dates[0]
    assert f._news_available_date(pd.Timestamp("2026-01-05 19:00"), dates) == dates[1]

    # 종목 피쳐 스모크
    n = 380
    d = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + 0.001 * np.sin(np.arange(n) / 7))
    raw = pd.DataFrame({
        "date": d, "ticker": "005930", "name": "삼성전자", "market": "KOSPI",
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1_000_000 + np.arange(n),
        "trading_value": (1_000_000 + np.arange(n)) * close,
        "market_cap": 4e14, "listed_shares": 6e9,
        "value_foreign_net": np.where(np.arange(n)%3==0, 1e9, -5e8),
        "value_institution_net": np.where(np.arange(n)%4==0, 8e8, -3e8),
        "short_volume": 10000, "short_volume_ratio_pct": 1.0,
        "short_balance_value": 1e11, "foreign_ownership_pct": 50.0,
    })
    out = f.build_one_stock_features(raw)
    for col in ["up_streak_days", "ret_20", "vol_20", "smart_money_net", "short_balance_chg_20"]:
        assert col in out.columns

    # 피쳐 catalog에 라벨/미래가 들어가면 안 됨
    out["label_abs_crash_20"] = 0
    out["future_close_ret_20"] = 0
    catalog = f.build_feature_catalog(out)
    flat = {c for cols in catalog.values() for c in cols}
    assert not any(c.startswith(("label_", "future_")) for c in flat)


    # 뉴스 집계 스모크와 데이터 존재 플래그
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        old_raw = f.RAW_DIR
        f.RAW_DIR = temp / "raw"
        (f.RAW_DIR / "news").mkdir(parents=True)
        news = pd.DataFrame({
            "published_at_kst": [pd.Timestamp("2026-01-05 17:00", tz="Asia/Seoul")],
            "scope": ["company"], "ticker": ["005930"], "alias": ["company_005930"],
            "title": ["삼성전자 실적 우려"], "source_domain": ["example.com"],
            "negative_keyword_count": [1], "risk_keyword_count": [1],
        })
        news.to_parquet(f.RAW_DIR / "news" / "naver_news.parquet", index=False)
        base = pd.DataFrame({"date": dates, "ticker": "005930"})
        nf = f.add_news_features(base, dates)
        assert nf.loc[nf["date"].eq(dates[0]), "news_company_article_count"].iloc[0] == 1
        assert nf.loc[nf["date"].eq(dates[0]), "news_any_available"].iloc[0] == 1
        assert nf.loc[nf["date"].eq(dates[1]), "news_any_available"].iloc[0] == 0
        f.RAW_DIR = old_raw

    # breadth·시장 beta·라벨·4개 봉인 분리 스모크
    panel = out.copy()
    panel["industry_code"] = "SEMICON"
    panel["industry_name"] = "반도체"
    panel = f.add_breadth_features(panel)
    panel = f.add_major_company_features(panel)
    panel = f.add_market_beta_and_ranks(panel)
    panel["feature_ready"] = panel["history_days"].ge(f.MIN_HISTORY_DAYS).astype("int8")
    panel = f.add_labels(panel)
    assert "label_abs_crash_20" in panel and panel["label_abs_crash_20"].notna().any()
    cat2 = f.build_feature_catalog(panel)
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        old_dev, old_seal, old_meta = f.DEVELOPMENT_DIR, f.SEALED_DIR, f.META_DIR
        f.DEVELOPMENT_DIR, f.SEALED_DIR, f.META_DIR = temp/"development", temp/"sealed", temp/"meta"
        for directory in (f.DEVELOPMENT_DIR, f.SEALED_DIR, f.META_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        f.split_and_seal(panel, cat2)
        manifest = __import__("json").loads((f.SEALED_DIR/"seal_manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["seals"]) == 4
        assert all(int(x["trading_days"]) == 15 for x in manifest["seals"])
        dev = pd.read_parquet(f.DEVELOPMENT_DIR/"training_dataset.parquet")
        seal_dates = set()
        for item in manifest["seals"]:
            seal_dates.update(pd.read_parquet(Path(item["path"]))["date"].astype(str))
        assert not set(dev["date"].astype(str)) & seal_dates
        f.DEVELOPMENT_DIR, f.SEALED_DIR, f.META_DIR = old_dev, old_seal, old_meta

    # CPU 모델 스모크
    a.USE_GPU = False
    a.XGB_PARAMS["n_estimators"] = 5
    xx = pd.DataFrame({"x1": np.linspace(-1, 1, 100), "x2": np.sin(np.arange(100))})
    yy = (xx["x1"] > 0).astype("int8").to_numpy()
    bundle = a.fit_model(xx, yy, np.ones(len(yy), dtype="float32"))
    pp = bundle.predict_proba(xx)
    assert len(pp) == len(yy) and np.isfinite(pp).all()

    # 모델 저장 구조는 __main__ 래퍼가 아닌 dict여야 함: 코드 문자열 검사
    code = (BASE / "03_이탈테스트_봉인인증.py").read_text(encoding="utf-8")
    assert '"model": bundle.model' in code
    assert "결과압축_생성" in code

    print("="*72)
    print("자체 테스트 통과")
    print("- 연속 상승/하락")
    print("- 미래 라벨 현재시점 제외")
    print("- 뉴스 발표시각 KRX 정렬")
    print("- 종목 시계열 피쳐 생성")
    print("- 뉴스 집계·존재구간 플래그")
    print("- 시장 breadth·라벨·4개 봉인 분리")
    print("- 라벨/미래열 피쳐 누수 차단")
    print("- CPU 모델 학습·예측")
    print("- 최종 모델 저장 호환성")
    print("="*72)

if __name__ == "__main__":
    main()
