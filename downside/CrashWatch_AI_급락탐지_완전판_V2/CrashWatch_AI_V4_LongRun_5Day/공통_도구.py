#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""공정별 품질보고와 결과 보관용 소형 ZIP 생성."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
META_DIR = DATA_ROOT / "meta"
LOG_DIR = DATA_ROOT / "logs"
ZIP_DIR = DATA_ROOT / "결과압축"
for _d in (META_DIR, LOG_DIR, ZIP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def 원자적_JSON_저장(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def 파일_SHA256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:
            return 0


def _parquet_date_range(path: Path) -> tuple[str, str]:
    try:
        df = pd.read_parquet(path, columns=["date"])
        s = pd.to_datetime(df["date"], errors="coerce").dropna()
        if s.empty:
            return "", ""
        return str(s.min().date()), str(s.max().date())
    except Exception:
        return "", ""


def _directory_parquet_summary(directory: Path) -> dict[str, Any]:
    files = list(directory.rglob("*.parquet")) if directory.exists() else []
    rows = sum(_parquet_rows(p) for p in files)
    start, end = "", ""
    # 날짜 범위는 너무 많은 파일을 읽지 않도록 처음/중간/마지막 일부만 확인한다.
    sample = files[:3] + files[len(files)//2:len(files)//2+2] + files[-3:]
    starts, ends = [], []
    for p in dict.fromkeys(sample):
        a, b = _parquet_date_range(p)
        if a:
            starts.append(a)
        if b:
            ends.append(b)
    if starts:
        start = min(starts)
    if ends:
        end = max(ends)
    return {"files": len(files), "rows": rows, "start": start, "end": end}


def 수집_품질_보고서_생성() -> pd.DataFrame:
    raw = DATA_ROOT / "raw"
    records: list[dict[str, Any]] = []

    def add(source: str, path: Path, required: bool, key_name: str = "", detail: str = "") -> None:
        exists = path.exists()
        key_present = bool(os.getenv(key_name, "").strip()) if key_name else True
        if path.is_dir():
            info = _directory_parquet_summary(path)
            rows, files, start, end = info["rows"], info["files"], info["start"], info["end"]
        elif exists and path.suffix == ".parquet":
            rows, files = _parquet_rows(path), 1
            start, end = _parquet_date_range(path)
        else:
            rows, files, start, end = 0, 0, "", ""

        data_ok = (rows > 0 and files > 0) if (path.is_dir() or path.suffix == ".parquet") else exists
        if exists and data_ok:
            status = "성공"
        elif key_name and not key_present:
            status = "건너뜀_API키없음"
        elif required:
            status = "실패_필수"
        else:
            status = "실패_선택"

        records.append(
            {
                "데이터원": source,
                "상태": status,
                "파일수": files,
                "행수": rows,
                "시작일": start,
                "종료일": end,
                "필수여부": required,
                "API환경변수": key_name,
                "설명": detail,
                "경로": str(path),
            }
        )

    add("KRX 종목 OHLCV·수급·공매도", raw / "krx" / "stocks", True)
    add("KRX 지수·업종지수", raw / "krx" / "indices", False)
    add("해외증시·코인·원자재·부동산", raw / "cross_assets.parquet", True)
    add("FRED 금리·환율·신용", raw / "macro_fred.parquet", False, "FRED_API_KEY")
    add("ECOS 기준금리·주택", raw / "macro_ecos.parquet", False, "ECOS_API_KEY")
    add("FOMC·한국은행 일정", raw / "policy_events.parquet", True)
    add("DART 기업개황", raw / "dart" / "company_profiles.parquet", False, "OPENDART_API_KEY")
    add("DART 공시목록", raw / "dart" / "disclosures.parquet", False, "OPENDART_API_KEY")
    add("DART 재무제표", raw / "dart" / "financials_major_accounts.parquet", False, "OPENDART_API_KEY")
    add("DART 위험공시 원문", raw / "dart" / "documents", False, "OPENDART_API_KEY")
    add("NAVER 기업·시장 뉴스", raw / "news" / "naver_news.parquet", False, "NAVER_CLIENT_ID")
    add("GDELT 글로벌 뉴스량", raw / "news" / "gdelt_timeline.parquet", False)

    report = pd.DataFrame(records)

    # 교차자산/뉴스의 기대 항목 누락을 상세 설명한다.
    cross_path = raw / "cross_assets.parquet"
    if cross_path.exists():
        try:
            cross = pd.read_parquet(cross_path, columns=["alias"])
            expected = {
                "sp500", "nasdaq", "dow", "russell2000", "semiconductor_sox", "nikkei225",
                "hangseng", "shanghai", "shenzhen", "eurostoxx50", "dax", "ftse100",
                "india_nifty50", "btc", "eth", "gold", "wti", "copper", "dollar_index",
                "us_reit_vnq", "us_realestate_xlre", "us_realestate_iyr",
                "us_homebuilders_xhb", "us_homebuilders_itb",
            }
            actual = set(cross["alias"].dropna().astype(str).unique())
            missing = sorted(expected - actual)
            detail = f"성공별칭={len(actual)}/{len(expected)}, 누락={missing}"
            report.loc[report["데이터원"].eq("해외증시·코인·원자재·부동산"), "설명"] = detail
            if missing:
                report.loc[report["데이터원"].eq("해외증시·코인·원자재·부동산"), "상태"] = "부분성공"
        except Exception as exc:
            report.loc[report["데이터원"].eq("해외증시·코인·원자재·부동산"), "설명"] = repr(exc)

    naver_path = raw / "news" / "naver_news.parquet"
    if naver_path.exists():
        try:
            news = pd.read_parquet(naver_path, columns=["alias", "scope"])
            detail = f"쿼리={news['alias'].nunique()}, 기업기사={int(news['scope'].eq('company').sum())}, 시장기사={int(news['scope'].eq('market').sum())}"
            report.loc[report["데이터원"].eq("NAVER 기업·시장 뉴스"), "설명"] = detail
        except Exception as exc:
            report.loc[report["데이터원"].eq("NAVER 기업·시장 뉴스"), "설명"] = repr(exc)

    # KRX 세부 성공/실패 현황
    krx_manifest = META_DIR / "krx_collection_manifest.parquet"
    if krx_manifest.exists():
        try:
            m = pd.read_parquet(krx_manifest)
            status_counts = m["status"].astype(str).value_counts().to_dict()
            partial = int(m.get("errors", pd.Series("", index=m.index)).astype(str).str.len().gt(2).sum())
            report.loc[report["데이터원"].eq("KRX 종목 OHLCV·수급·공매도"), "설명"] = (
                f"종목 상태={status_counts}, 세부항목 일부실패기록={partial}종목"
            )
        except Exception as exc:
            report.loc[report["데이터원"].eq("KRX 종목 OHLCV·수급·공매도"), "설명"] = repr(exc)

    report_path = META_DIR / "수집_품질_보고서.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    원자적_JSON_저장(report.to_dict("records"), META_DIR / "수집_품질_보고서.json")

    print("\n" + "=" * 96)
    print("데이터 크롤링 품질 보고서")
    print("=" * 96)
    for r in report.to_dict("records"):
        print(
            f"[{r['상태']:<16}] {r['데이터원']:<28} "
            f"파일={r['파일수']:<5} 행={r['행수']:<12,} "
            f"기간={r['시작일'] or '-'}~{r['종료일'] or '-'}"
        )
        if r["설명"]:
            print(f"  └─ {r['설명']}")
    print(f"보고서: {report_path}")

    required_failed = report["필수여부"].eq(True) & report["상태"].str.startswith("실패")
    if required_failed.any():
        failed = report.loc[required_failed, "데이터원"].tolist()
        raise RuntimeError(f"필수 데이터 수집 실패: {failed}")
    return report


def 피쳐_품질_보고서_생성(df: pd.DataFrame, catalog: dict[str, list[str]]) -> pd.DataFrame:
    # 수백 개 피처 × 수백만 행 전체 재스캔은 과도하므로 최대 500,000행의
    # 날짜·종목 혼합 표본으로 결측/상수 상태를 진단한다.
    audit = df if len(df) <= 500_000 else df.sample(n=500_000, random_state=20260721)
    rows = []
    for group, features in catalog.items():
        for col in features:
            if col not in audit.columns:
                rows.append(
                    {"그룹": group, "피쳐": col, "존재": False, "비결측률": 0.0, "고유값수": 0, "상태": "누락"}
                )
                continue
            s = pd.to_numeric(audit[col], errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
            nonnull = float(s.notna().mean())
            nunique = int(s.nunique(dropna=True))
            if nonnull == 0:
                status = "실패_전부결측"
            elif nonnull < 0.05:
                status = "경고_거의결측"
            elif nunique <= 1:
                status = "경고_상수"
            else:
                status = "성공"
            rows.append(
                {
                    "그룹": group,
                    "피쳐": col,
                    "존재": True,
                    "비결측률": nonnull,
                    "결측률": 1 - nonnull,
                    "고유값수": nunique,
                    "상태": status,
                }
            )
    report = pd.DataFrame(rows)
    path = META_DIR / "피쳐_품질_보고서.csv"
    report.to_csv(path, index=False, encoding="utf-8-sig")

    group_report = (
        report.groupby("그룹", as_index=False)
        .agg(
            피쳐수=("피쳐", "size"),
            성공수=("상태", lambda s: int((s == "성공").sum())),
            실패경고수=("상태", lambda s: int((s != "성공").sum())),
            평균비결측률=("비결측률", "mean"),
        )
        .sort_values(["실패경고수", "평균비결측률"], ascending=[False, True])
    )
    group_path = META_DIR / "피쳐그룹_품질_요약.csv"
    group_report.to_csv(group_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("피쳐 생성 품질 보고서")
    print("=" * 96)
    for r in group_report.to_dict("records"):
        marker = "성공" if r["실패경고수"] == 0 else "부분성공"
        print(
            f"[{marker:<8}] {r['그룹']:<34} 피쳐={r['피쳐수']:<4} "
            f"정상={r['성공수']:<4} 경고/실패={r['실패경고수']:<4} "
            f"평균비결측={r['평균비결측률']:.1%}"
        )
    worst = report[report["상태"] != "성공"].sort_values("비결측률").head(30)
    if not worst.empty:
        print("\n잘 생성되지 않았거나 데이터가 부족한 피쳐 상위 30개:")
        for r in worst.to_dict("records"):
            print(f"  - {r['피쳐']} | {r['그룹']} | {r['상태']} | 비결측={r['비결측률']:.1%}")
    print(f"상세: {path}")
    return report


def _copy_limited(src: Path, dst: Path, max_bytes: int = 8_000_000) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size <= max_bytes:
        shutil.copy2(src, dst)
        return True
    # 큰 로그/텍스트는 마지막 부분만 보존
    if src.suffix.lower() in {".log", ".txt", ".md", ".csv", ".json"}:
        data = src.read_bytes()[-max_bytes:]
        dst.write_bytes(data)
        return True
    return False


def 결과압축_생성(stage: str) -> Path:
    """대용량 원천 parquet와 모델은 제외하고 결과 검토에 필요한 결과만 ZIP으로 만든다."""
    stage = str(stage)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = ZIP_DIR / f"_{stage}_{stamp}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    include: list[Path] = []
    if stage == "01_데이터크롤링":
        include += [
            META_DIR / "collection_run_manifest.json",
            META_DIR / "krx_collection_manifest.parquet",
            META_DIR / "selected_universe.parquet",
            META_DIR / "수집_품질_보고서.csv",
            META_DIR / "수집_품질_보고서.json",
            LOG_DIR / "01_데이터_크롤링.log",
        ]
    elif stage == "02_피쳐생성_봉인":
        include += [
            META_DIR / "feature_catalog.json",
            META_DIR / "피쳐_품질_보고서.csv",
            META_DIR / "피쳐그룹_품질_요약.csv",
            DATA_ROOT / "sealed" / "seal_manifest.json",
            LOG_DIR / "02_피쳐_생성_봉인.log",
        ]
        dev = DATA_ROOT / "development" / "training_dataset.parquet"
        if dev.exists():
            try:
                sample = pd.read_parquet(dev).head(300)
                sample.to_csv(work / "개발데이터_샘플300행.csv", index=False, encoding="utf-8-sig")
            except Exception:
                pass
    elif stage == "03_이탈테스트_봉인인증":
        ab = DATA_ROOT / "ablation"
        include += [
            ab / "baseline_cv.csv",
            ab / "group_ablation.csv",
            ab / "feature_ablation.csv",
            ab / "selected_groups_cv.csv",
            ab / "feature_quality.csv",
            ab / "baseline_feature_importance.csv",
            ab / "frozen_feature_set.json",
            ab / "seal_certification.csv",
            ab / "run_summary.json",
            DATA_ROOT / "sealed" / "seal_manifest.json",
            LOG_DIR / "03_이탈테스트_봉인인증.log",
        ]

    copied = []
    for src in include:
        if _copy_limited(src, work / src.name):
            copied.append(src.name)

    summary = {
        "stage": stage,
        "created_at": datetime.now().astimezone().isoformat(),
        "included_files": copied,
        "excluded": "대용량 원천 parquet, 전체 피처 parquet, 학습 모델",
        "data_root": str(DATA_ROOT),
    }
    원자적_JSON_저장(summary, work / "결과_압축_설명.json")

    zip_path = ZIP_DIR / f"{stage}_RESULTS_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in work.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(work))
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n결과 검토용 결과 압축 생성: {zip_path}")
    print(f"크기: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    return zip_path
