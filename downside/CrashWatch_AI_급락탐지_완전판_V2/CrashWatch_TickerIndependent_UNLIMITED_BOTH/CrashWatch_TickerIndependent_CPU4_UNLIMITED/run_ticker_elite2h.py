from __future__ import annotations

import argparse
import os
from pathlib import Path

# Seal CUDA before importing any module that can load XGBoost/CatBoost.
if os.environ.get("CRASHWATCH_EXECUTION_PROFILE", "cpu4").strip().lower() == "cpu4":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CRASHWATCH_BACKEND_MODE"] = "cpu_only"
    for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_thread_variable] = "4"

from dual_ablation.ticker_elite2h.runner import run_supervisor, run_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="CrashWatch per-ticker independent experiment (run until complete)")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--worker-role", choices=["model", "correlation"], default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    if args.worker_role:
        if args.result_dir is None:
            parser.error("worker requires --result-dir")
        return run_worker(project, args.result_dir.resolve(), args.worker_role, args.worker_index)
    run_supervisor(project, args.dataset.resolve() if args.dataset else None, args.result_dir.resolve() if args.result_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
