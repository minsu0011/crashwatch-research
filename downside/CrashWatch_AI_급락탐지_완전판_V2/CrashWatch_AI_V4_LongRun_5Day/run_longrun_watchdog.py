#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from dual_ablation.longrun.orchestrator import load_config, run_longrun


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V4 장기실험 자동복구 watchdog")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "longrun_5day.json")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config(args.config)
    runtime_seconds = float(config["runtime_hours"]) * 3600
    reset = args.reset
    while True:
        try:
            result = run_longrun(project, args.config, reset=reset)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            reset = False
            data_root = Path(config.get("data_root", project / "crashwatch_ai_data"))
            if not data_root.is_absolute():
                data_root = (project / data_root).resolve()
            root = data_root / "ablation_longrun"
            if (root / "STOP_LONGRUN.txt").exists():
                logging.getLogger("crashwatch.watchdog").warning("사용자 중지 파일을 확인해 watchdog을 종료합니다.")
                return
            state = root / "longrun_state.json"
            if state.exists():
                try:
                    metadata = json.loads(state.read_text(encoding="utf-8")).get("metadata", {})
                    started = float(metadata.get("started_at_epoch", time.time()))
                    if time.time() >= started + runtime_seconds:
                        logging.getLogger("crashwatch.watchdog").error("96시간 예산 종료 후 예외: %s", exc)
                        return
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            root.mkdir(parents=True, exist_ok=True)
            with (root / "watchdog_errors.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{pd_timestamp()} | {exc}\n{traceback.format_exc()}\n")
            logging.getLogger("crashwatch.watchdog").exception(
                "장기실험 예외. %s초 후 checkpoint에서 재시작합니다.", args.retry_seconds
            )
            time.sleep(max(5, args.retry_seconds))


def pd_timestamp() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


if __name__ == "__main__":
    main()
