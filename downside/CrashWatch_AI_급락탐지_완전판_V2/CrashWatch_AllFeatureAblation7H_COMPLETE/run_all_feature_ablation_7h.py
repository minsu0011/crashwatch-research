from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from cw7h.runner import run_experiment
from cw7h.utils import atomic_json, canonical_hash


def start_absolute_watchdog(output_dir: Path, minutes: float) -> threading.Event:
    """Terminate this process tree at the configured absolute wall-clock ceiling."""
    stop_event = threading.Event()
    deadline = time.time() + float(minutes) * 60.0

    def watchdog() -> None:
        remaining = max(0.0, deadline - time.time())
        if stop_event.wait(timeout=remaining):
            return
        try:
            atomic_json({
                "status": "absolute_watchdog_triggered",
                "deadline_epoch": deadline,
                "triggered_epoch": time.time(),
                "reason": "Process tree stopped at the configured absolute wall-clock ceiling",
            }, output_dir / "absolute_watchdog.json")
        except Exception:
            pass
        try:
            import psutil
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except Exception:
                    pass
            _, alive = psutil.wait_procs(children, timeout=12)
            for child in alive:
                try:
                    child.kill()
                except Exception:
                    pass
        finally:
            os._exit(124)

    threading.Thread(target=watchdog, name="cw7h-absolute-watchdog", daemon=True).start()
    return stop_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrashWatch 7시간 전 피처 상관·이탈 실험")
    parser.add_argument("--config", default="config.json", help="설정 JSON 경로")
    parser.add_argument("--dataset", default=None, help="development Parquet 경로")
    parser.add_argument("--project-root", default=None, help="CrashWatch 프로젝트 최상위 폴더")
    parser.add_argument("--force-cache", action="store_true", help="행렬 캐시 재생성")
    parser.add_argument("--force-correlation", action="store_true", help="상관 분석 재실행")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "all_feature_ablation_7h.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)


def main() -> int:
    args = parse_args()
    package_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = package_root / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["config_hash"] = canonical_hash(config)
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else package_root.parent.resolve()
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    setup_logging(output_dir)
    watchdog_minutes = float(config.get("runtime", {}).get("absolute_watchdog_minutes", 419.5))
    watchdog_stop = start_absolute_watchdog(output_dir, watchdog_minutes)
    logging.info("CrashWatch all-feature ablation 7H started")
    logging.info("package_root=%s", package_root)
    logging.info("project_root=%s", project_root)
    logging.info("config=%s", config_path)
    try:
        result = run_experiment(
            package_root,
            project_root,
            config,
            dataset_override=args.dataset,
            force_cache=args.force_cache,
            force_correlation=args.force_correlation,
        )
        logging.info("completed: status=%s elapsed=%.1fs", result.get("status"), result.get("elapsed_seconds", -1))
        return 0 if result.get("status") == "completed" else 2
    except Exception as exc:
        logging.exception("fatal experiment error")
        atomic_json({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "config_path": str(config_path),
            "project_root": str(project_root),
        }, output_dir / "fatal_error.json")
        return 1
    finally:
        watchdog_stop.set()


if __name__ == "__main__":
    if os.name == "nt":
        import multiprocessing as mp
        mp.freeze_support()
    raise SystemExit(main())
