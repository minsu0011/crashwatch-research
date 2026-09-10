#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    project = Path(__file__).resolve().parent
    config = json.loads((project / "configs" / "longrun_5day.json").read_text(encoding="utf-8"))
    data_root = Path(os.getenv("CRASHWATCH_DATA_DIR", project / "crashwatch_ai_data")).resolve()
    configured = str(config.get("dataset", "auto"))
    if configured not in {"", "auto"}:
        dataset = Path(configured).expanduser()
        if not dataset.is_absolute():
            dataset = (project / dataset).resolve()
        if dataset.parent.name == "development":
            data_root = dataset.parent.parent
    elif not (data_root / "development").exists():
        candidates = list(project.parent.glob("*/crashwatch_ai_data/development/training_dataset_dual.parquet"))
        if candidates:
            data_root = max(candidates, key=lambda x: x.stat().st_mtime_ns).parent.parent
    state = data_root / "ablation_longrun" / "longrun_state.json"
    summary = data_root / "ablation_longrun" / "longrun_summary.json"
    if not state.exists():
        print(f"상태 파일 없음: {state}")
        return
    payload = json.loads(state.read_text(encoding="utf-8"))
    jobs = pd.DataFrame(payload.get("jobs", []))
    print("\n[메타데이터]")
    print(json.dumps(payload.get("metadata", {}), ensure_ascii=False, indent=2))
    if not jobs.empty:
        cols = [c for c in ["job_id", "stage", "status", "attempts", "duration_seconds", "error"] if c in jobs]
        print("\n[작업 상태]")
        print(jobs[cols].to_string(index=False))
        print("\n[상태 집계]")
        print(jobs["status"].value_counts(dropna=False).to_string())
    if summary.exists():
        print("\n[최종 요약]")
        print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
