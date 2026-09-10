from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_command(command: Sequence[str]) -> None:
    print("\n$ " + " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "CrashWatch Surge V3 전체 파이프라인: pre-model gate를 검증/생성한 뒤 "
            "전체 피처 leave-one-out 이탈을 실행합니다."
        )
    )
    parser.add_argument("--package-root", type=Path, default=default_root)
    parser.add_argument("--correlation-dir", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument(
        "--mode",
        choices=["lightgbm", "xgboost_gpu", "both", "dry_run", "gate_only"],
        default="lightgbm",
    )
    parser.add_argument("--stages", default="baseline,profiles,feature_loo")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads-per-worker", type=int, default=3)
    parser.add_argument("--xgboost-threads", type=int, default=1)
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--fold-ids", default="")
    parser.add_argument("--limit-features", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.package_root.resolve()
    starter = Path(__file__).resolve().parent
    correlation_dir = (
        args.correlation_dir.resolve()
        if args.correlation_dir is not None
        else root / "outputs" / "surge_correlation_map_complete"
    )
    gate_output = (
        args.gate_output.resolve()
        if args.gate_output is not None
        else root / "outputs" / "surge_pre_model_gate_v3"
    )

    if not args.skip_gate:
        gate_command = [
            sys.executable,
            str(starter / "run_surge_pre_model_gate.py"),
            "--package-root",
            str(root),
            "--correlation-dir",
            str(correlation_dir),
            "--output",
            str(gate_output),
        ]
        if not args.no_resume:
            gate_command.append("--resume")
        run_command(gate_command)

    if args.mode == "gate_only":
        print(f"\nPre-model gate completed: {gate_output}", flush=True)
        return

    if args.mode == "lightgbm" or args.mode == "dry_run":
        backends = "lightgbm_cpu"
        default_name = "surge_all_feature_ablation_v3_lightgbm"
    elif args.mode == "xgboost_gpu":
        backends = "xgboost_gpu"
        default_name = "surge_all_feature_ablation_v3_xgboost_gpu"
    else:
        backends = "lightgbm_cpu,xgboost_gpu"
        default_name = "surge_all_feature_ablation_v3_both"

    model_output = (
        args.model_output.resolve()
        if args.model_output is not None
        else root / "outputs" / default_name
    )
    model_command = [
        sys.executable,
        str(starter / "run_surge_all_feature_ablation.py"),
        "--package-root",
        str(root),
        "--correlation-dir",
        str(correlation_dir),
        "--pre-model-dir",
        str(gate_output),
        "--output",
        str(model_output),
        "--backends",
        backends,
        "--stages",
        args.stages,
        "--workers",
        str(args.workers),
        "--threads-per-worker",
        str(args.threads_per_worker),
        "--xgboost-threads",
        str(args.xgboost_threads),
        "--seeds",
        args.seeds,
    ]
    if args.fold_ids:
        model_command.extend(["--fold-ids", args.fold_ids])
    if args.limit_features > 0:
        model_command.extend(["--limit-features", str(args.limit_features)])
    if not args.no_resume:
        model_command.append("--resume")
    if args.mode == "dry_run":
        model_command.append("--dry-run")
    run_command(model_command)
    print(f"\nCompleted: {model_output}", flush=True)


if __name__ == "__main__":
    main()
