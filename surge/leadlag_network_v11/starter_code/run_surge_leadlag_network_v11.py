from __future__ import annotations

import argparse
import os
from pathlib import Path

from surge_leadlag_network_v11 import run_pipeline


def resolve_defaults(args: argparse.Namespace) -> None:
    root = Path(args.package_root).expanduser().resolve()
    workspace = root.parent
    if args.dataset is None:
        args.dataset = workspace / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "training_dataset_finance11h.parquet"
    if args.target_sidecar is None:
        args.target_sidecar = workspace / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "surge_target_3d5.parquet"
    if args.folds is None:
        args.folds = workspace / "CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809" / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json"
    if args.v10_2_output is None:
        args.v10_2_output = workspace / "CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815" / "outputs" / "surge_tickerwise_correlation_map_v10_2"
    output = Path(args.output)
    args.output = output.resolve() if output.is_absolute() else (root / output).resolve()
    for name in ["dataset", "target_sidecar", "folds", "v10_2_output"]:
        value = Path(getattr(args, name)).expanduser().resolve()
        if not value.exists():
            raise FileNotFoundError(f"{name}: {value}")
        setattr(args, name, value)
    args.package_root = root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V11 inter-ticker lead-lag and dynamic co-movement map")
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--v10-2-output", type=Path)
    parser.add_argument("--output", default="outputs/surge_leadlag_network_v11")
    universe = parser.add_mutually_exclusive_group()
    universe.add_argument("--tickers", default="")
    universe.add_argument("--top-n", type=int, default=0)
    universe.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-pair-observations", type=int, default=30)
    parser.add_argument("--rolling-step", type=int, default=20)
    parser.add_argument("--maximum-incoming-features", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--minimum-portfolio-alerts", type=int, default=30)
    parser.add_argument("--run-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.tickers and args.top_n <= 0 and not args.all_tickers:
        parser.error("Choose one of --tickers, --top-n, or --all-tickers")
    if not 1 <= args.threads <= 8:
        parser.error("Game mode permits 1..8 CPU threads")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    resolve_defaults(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
