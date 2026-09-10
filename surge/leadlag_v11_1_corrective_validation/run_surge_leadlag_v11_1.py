from __future__ import annotations

import argparse
import os
from pathlib import Path

from surge_leadlag_corrective_v11_1 import run_pipeline


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_defaults(args: argparse.Namespace) -> None:
    root = Path(args.package_root).expanduser().resolve()
    workspace = root.parent

    if args.dataset is None:
        args.dataset = _first_existing([
            workspace / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "training_dataset_finance11h.parquet",
            root / "data" / "training_dataset_finance11h.parquet",
        ])
    if args.target_sidecar is None:
        args.target_sidecar = _first_existing([
            workspace / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "surge_target_3d5.parquet",
            root / "data" / "surge_target_3d5.parquet",
        ])
    if args.folds is None:
        args.folds = _first_existing([
            workspace / "CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809" / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json",
            root / "data" / "walk_forward_folds.json",
        ])
    if args.v10_2_output is None:
        args.v10_2_output = _first_existing([
            workspace / "CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815" / "outputs" / "surge_tickerwise_correlation_map_v10_2",
            workspace / "CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_2026081512" / "outputs" / "surge_tickerwise_correlation_map_v10_2",
            root / "v10_2_output",
        ])
    if args.v11_reference_strong is None:
        args.v11_reference_strong = root / "reference_v11" / "strong_directed_edges_7.csv"

    output = Path(args.output)
    args.output = output.resolve() if output.is_absolute() else (root / output).resolve()
    args.package_root = root

    for name in ["dataset", "target_sidecar", "folds", "v10_2_output", "v11_reference_strong"]:
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        setattr(args, name, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CrashWatch Surge V11.1 corrective validation: adaptive max-stat, frozen lag, target alignment, matched probe"
    )
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--v10-2-output", type=Path)
    parser.add_argument("--v11-reference-strong", type=Path)
    parser.add_argument("--output", default="outputs/surge_leadlag_v11_1_corrective_validation")
    parser.add_argument("--tickers", default="", help="Optional comma-separated subset. Empty means all target-valid tickers.")

    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument(
        "--maxstat-executor",
        choices=["thread", "process"],
        default="process",
        help="Process mode uses physical CPU cores for the max-stat permutation map.",
    )
    parser.add_argument("--minimum-pair-observations", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--screening-permutations", type=int, default=499)
    parser.add_argument("--adaptive-permutations", type=int, default=19999)
    parser.add_argument("--adaptive-trigger-p", type=float, default=0.01)
    parser.add_argument("--maxstat-alpha", type=float, default=0.10)
    parser.add_argument("--minimum-forward-abs-corr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260815)

    parser.add_argument("--rolling-windows", default="60,120,252")
    parser.add_argument("--rolling-step", type=int, default=20)
    parser.add_argument("--rolling-residual-fit-days", type=int, default=252)
    parser.add_argument("--rolling-residual-min-fit", type=int, default=120)

    parser.add_argument(
        "--probe-edge-sets",
        default="v11_strong_repaired,maxstat_corrected",
        help="Comma separated: v11_strong_repaired,maxstat_corrected,forward_effect_stable",
    )
    parser.add_argument("--minimum-target-event-n", type=int, default=5)
    parser.add_argument("--minimum-probe-training-rows", type=int, default=120)
    parser.add_argument("--minimum-validation-base-rows", type=int, default=30)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--minimum-portfolio-alerts", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    maximum_workers = os.cpu_count() or 32
    if not 1 <= args.threads <= maximum_workers:
        parser.error(f"--threads must be in 1..{maximum_workers}")
    if args.screening_permutations < 99:
        parser.error("--screening-permutations must be >=99 for a meaningful max-stat screen")
    if args.adaptive_permutations < args.screening_permutations:
        parser.error("--adaptive-permutations must be >= --screening-permutations")
    if not 0 < args.adaptive_trigger_p <= 1:
        parser.error("--adaptive-trigger-p must be in (0,1]")
    try:
        args.rolling_windows = [int(value) for value in str(args.rolling_windows).split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"Invalid --rolling-windows: {exc}")
    if not args.rolling_windows or any(value < 20 for value in args.rolling_windows):
        parser.error("--rolling-windows must contain integers >=20")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    resolve_defaults(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
