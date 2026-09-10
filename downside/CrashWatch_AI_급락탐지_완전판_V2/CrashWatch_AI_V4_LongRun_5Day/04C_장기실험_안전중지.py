#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parent
    config_path = project / "configs" / "longrun_5day.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = str(config.get("dataset", "auto"))
    data_root = Path(os.getenv("CRASHWATCH_DATA_DIR", project / "crashwatch_ai_data")).resolve()
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
    stop_file = data_root / "ablation_longrun" / "STOP_LONGRUN.txt"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("stop requested\n", encoding="utf-8")
    print(f"안전 중지 요청 생성: {stop_file}")


if __name__ == "__main__":
    main()
