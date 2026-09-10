#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="8시간 상관·시드·기법 보강 이탈테스트"
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cache-namespace", default="finance11h_v3")
    parser.add_argument("--output-name", default="ablation_finance8h")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _set_resources(threads: int, gpu: bool) -> None:
    value = str(threads)
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]:
        os.environ[key] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu else "-1"
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".numba_cache"),
    )


def _worker(args: argparse.Namespace) -> int:
    _set_resources(args.threads, not args.cpu)
    import logging

    import psutil

    from dual_ablation.finance8h import Finance8HRunner

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            process.cpu_affinity(list(range(psutil.cpu_count(logical=True) or args.threads)))
    except Exception:
        pass
    runner = Finance8HRunner(
        Path(__file__).resolve().parent,
        args.dataset,
        hours=args.hours,
        threads=args.threads,
        cache_namespace=args.cache_namespace,
        output_name=args.output_name,
        prefer_gpu=not args.cpu,
        overwrite_cache=args.overwrite_cache,
    )
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _terminate_tree(process: subprocess.Popen) -> None:
    try:
        import psutil

        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        _, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            child.kill()
        root.terminate()
        try:
            root.wait(timeout=10)
        except psutil.TimeoutExpired:
            root.kill()
    except Exception:
        process.kill()


def _supervise(args: argparse.Namespace) -> int:
    _set_resources(args.threads, not args.cpu)
    forwarded = [value for value in sys.argv[1:] if value != "--worker"]
    command = [sys.executable, str(Path(__file__).resolve()), *forwarded, "--worker"]
    started = time.monotonic()
    worker = subprocess.Popen(command)
    hard_seconds = max(60.0, args.hours * 3600)
    try:
        return int(worker.wait(timeout=hard_seconds))
    except subprocess.TimeoutExpired:
        _terminate_tree(worker)
        output = (
            Path(__file__).resolve().parent
            / "crashwatch_ai_data"
            / args.output_name
        )
        output.mkdir(parents=True, exist_ok=True)
        event = {
            "hard_deadline_enforced": True,
            "budget_hours": args.hours,
            "elapsed_seconds": time.monotonic() - started,
            "worker_pid": worker.pid,
            "created_at_epoch": time.time(),
        }
        temporary = output / "hard_deadline_event.json.tmp"
        temporary.write_text(
            json.dumps(event, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, output / "hard_deadline_event.json")
        return 124


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
