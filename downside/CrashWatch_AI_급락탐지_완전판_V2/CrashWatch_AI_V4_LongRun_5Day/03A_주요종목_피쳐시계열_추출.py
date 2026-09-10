#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""이미 생성된 전체 피처 데이터에서 대표 종목의 개발/봉인 시계열을 분리한다.

이탈테스트 입력은 development만 사용한다. sealed는 최종 동결 모델의 종목별
인증 결과를 확인하기 위한 별도 파일로만 내보내며, 피처 선택에는 사용하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
DEV_PATH = DATA_ROOT / "development" / "training_dataset.parquet"
SEALED_DIR = DATA_ROOT / "sealed"
UNIVERSE_PATH = BASE_DIR / "주요종목_18선.csv"
OUT_DIR = DATA_ROOT / "ablation_sentinel" / "timeseries"
TARGET = "label_abs_crash_20"


def summarize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for ticker, block in df.groupby("ticker", sort=False):
        target = pd.to_numeric(block.get(TARGET), errors="coerce")
        missing_rate = float(block[feature_cols].isna().mean().mean()) if feature_cols else np.nan
        rows.append(
            {
                "ticker": ticker,
                "name": block["name"].dropna().iloc[-1] if "name" in block and block["name"].notna().any() else "",
                "market": block["market"].dropna().iloc[-1] if "market" in block and block["market"].notna().any() else "",
                "rows": len(block),
                "start_date": block["date"].min(),
                "end_date": block["date"].max(),
                "positive_count": int(target.fillna(0).sum()),
                "positive_rate": float(target.mean()) if target.notna().any() else np.nan,
                "mean_feature_missing_rate": missing_rate,
            }
        )
    return pd.DataFrame(rows)


def export_split(df: pd.DataFrame, name: str, universe: pd.DataFrame) -> None:
    tickers = set(universe["ticker"])
    out = df[df["ticker"].astype(str).str.zfill(6).isin(tickers)].copy()
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.merge(
        universe[["ticker", "sector", "sensitivity_axis", "priority"]],
        on="ticker",
        how="left",
        suffixes=("", "_sentinel"),
    )
    out = out.sort_values(["date", "ticker"])
    out.to_parquet(OUT_DIR / f"{name}.parquet", index=False)

    id_cols = {
        "date", "ticker", "name", "market", "sector", "sensitivity_axis", "priority",
        TARGET, "open", "high", "low", "close", "volume", "trading_value", "market_cap",
    }
    selected_numeric = [
        c for c in out.select_dtypes(include=[np.number, "bool"]).columns
        if c in id_cols or c.startswith((
            "ret_", "vol_", "downside_vol", "drawdown", "rsi_", "amihud", "turnover_",
            "foreign", "institution", "smart_money", "short_", "market_", "peer_",
            "major_", "xa_", "macro_", "news_", "dart_", "fin_", "event_", "beta_",
            "contagion_", "inter_", "cs_rank_",
        ))
    ]
    compact_cols = [c for c in ("date", "ticker", "name", "market", "sector", "priority", TARGET) if c in out]
    compact_cols += [c for c in selected_numeric if c not in compact_cols][:120]
    out[compact_cols].to_csv(
        OUT_DIR / f"{name}_compact.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    excluded = {
        "date", "ticker", "name", "market", "industry_code", "industry_name", "sector",
        "sensitivity_axis", "priority", TARGET, "seal_id", "sealed_do_not_train_or_tune",
    }
    feature_cols = [c for c in out.select_dtypes(include=[np.number, "bool"]).columns if c not in excluded and not c.startswith("label_")]
    summarize(out, feature_cols).to_csv(
        OUT_DIR / f"{name}_quality.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"개발 데이터가 없습니다: {DEV_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    universe["ticker"] = universe["ticker"].str.zfill(6)

    dev = pd.read_parquet(DEV_PATH)
    export_split(dev, "development_sentinel_features", universe)

    manifest_path = SEALED_DIR / "seal_manifest.json"
    exported_seals = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for seal in manifest.get("seals", []):
            path = Path(seal["path"])
            if path.exists():
                seal_df = pd.read_parquet(path)
                seal_id = str(seal["seal_id"])
                export_split(seal_df, f"sealed_{seal_id}_sentinel_features", universe)
                exported_seals.append(seal_id)

    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "development_source": str(DEV_PATH),
        "sentinel_count": len(universe),
        "exported_seals": exported_seals,
        "policy": "development 파일만 이탈테스트에 사용; sealed 파일은 최종 인증 전용",
    }
    (OUT_DIR / "extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
