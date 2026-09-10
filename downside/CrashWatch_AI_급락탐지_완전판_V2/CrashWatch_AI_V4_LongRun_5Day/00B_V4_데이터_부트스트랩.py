#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def _find_v3_dataset(project: Path) -> Path:
    downloads = Path.home() / "Downloads"
    candidates = list(downloads.glob(
        "CrashWatch*/**/CrashWatch_AI_V3_Dual_Ablation_FULL/crashwatch_ai_data/development/training_dataset_dual.parquet"
    ))
    candidates = [p for p in candidates if project not in p.parents]
    if not candidates:
        raise FileNotFoundError("V3 training_dataset_dual.parquet를 Downloads 아래에서 찾지 못했습니다.")
    return max(candidates, key=lambda p: (p.stat().st_size, p.stat().st_mtime_ns))


def _copy_tree(source: Path, destination: Path, overwrite: bool) -> tuple[int, int]:
    copied = skipped = 0
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(item, target)
        copied += 1
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 원천데이터를 보존한 채 V4 데이터 영역 준비")
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    source_dataset = (args.source_dataset or _find_v3_dataset(project)).resolve()
    source_root = source_dataset.parent.parent
    data_root = (args.data_root or project / "crashwatch_ai_data").resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["CRASHWATCH_DATA_DIR"] = str(data_root)

    raw_source = source_root / "raw" / "dual_ablation"
    raw_target = data_root / "raw" / "dual_ablation"
    copied, skipped = _copy_tree(raw_source, raw_target, args.overwrite)

    development = data_root / "development"
    development.mkdir(parents=True, exist_ok=True)
    base_dataset = development / "training_dataset_v3_base.parquet"
    if args.overwrite or not base_dataset.exists():
        shutil.copy2(source_dataset, base_dataset)

    # V3 프로젝트는 바로 위 원본 V2 프로젝트의 봉인 피처 카탈로그를 사용했다.
    base_catalog_source = source_root.parent.parent / "crashwatch_ai_data" / "meta" / "feature_catalog.json"
    base_catalog_target = data_root / "meta" / "feature_catalog.json"
    if base_catalog_source.exists() and (args.overwrite or not base_catalog_target.exists()):
        base_catalog_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_catalog_source, base_catalog_target)

    legacy_results = data_root / "legacy_v3_results"
    legacy_results.mkdir(parents=True, exist_ok=True)
    result_source = source_root / "ablation_dual"
    copied_results = 0
    for item in result_source.glob("*"):
        if not item.is_file() or item.suffix.lower() not in {".csv", ".json", ".log"}:
            continue
        target = legacy_results / item.name
        if target.exists() and not args.overwrite:
            continue
        # 대형 fold 상세표는 원본 위치를 기록하고 복제하지 않는다.
        if item.stat().st_size > 20 * 1024 * 1024:
            continue
        shutil.copy2(item, target)
        copied_results += 1

    cache_root = source_root / "ablation_dual" / "prediction_cache"
    cache_files = list(cache_root.rglob("*.parquet")) if cache_root.exists() else []
    manifest = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "source_dataset": str(source_dataset),
        "source_data_root": str(source_root),
        "v4_data_root": str(data_root),
        "base_dataset": str(base_dataset),
        "base_feature_catalog": str(base_catalog_target) if base_catalog_target.exists() else None,
        "raw_files_copied": copied,
        "raw_files_skipped": skipped,
        "legacy_result_files_copied": copied_results,
        "legacy_cache_root": str(cache_root),
        "legacy_cache_files": len(cache_files),
        "legacy_cache_bytes": int(sum(p.stat().st_size for p in cache_files)),
        "cache_policy": (
            "V3 prediction은 V3와 동일한 dataset/fold/model/calibration의 재분석에만 사용하며 "
            "V4 신규 피처 모델 캐시로 가장하지 않는다."
        ),
    }
    output = data_root / "bootstrap_manifest.json"
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
