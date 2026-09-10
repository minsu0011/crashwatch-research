from __future__ import annotations

import json
import shutil
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.data import load_cached_ticker
from dual_ablation.ticker_elite2h.runner import load_plan


def copy_if_exists(source: Path, target: Path) -> int:
    if not source.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
    return 1


def validate_imported_folds(target: Path, plan: dict) -> tuple[int, int]:
    """Reject reused folds if the current split policy no longer matches them."""
    ticker_count = 0
    fold_count = 0
    for ticker_dir in sorted((target / "ticker_models").glob("*")):
        if not ticker_dir.is_dir():
            continue
        data = load_cached_ticker(target, ticker_dir.name, plan, mmap=True)
        expected = {int(fold.fold_id): fold for fold in data.folds}
        metric_paths = sorted(ticker_dir.glob("fold_*_metrics.json"))
        if len(metric_paths) != len(expected):
            raise RuntimeError(
                f"checkpoint fold count mismatch for {ticker_dir.name}: "
                f"imported={len(metric_paths)} current={len(expected)}"
            )
        for metric_path in metric_paths:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            fold_id = int(metric["fold"])
            fold = expected.get(fold_id)
            if fold is None:
                raise RuntimeError(f"unexpected imported fold {fold_id} for {ticker_dir.name}")
            checks = {
                "fit_date_min": fold.fit_date_min,
                "fit_date_max": fold.fit_date_max,
                "calibration_date_min": fold.calibration_date_min,
                "calibration_date_max": fold.calibration_date_max,
                "validation_date_min": fold.validation_date_min,
                "validation_date_max": fold.validation_date_max,
            }
            for field, expected_value in checks.items():
                if str(metric.get(field, "")) != str(expected_value):
                    raise RuntimeError(
                        f"checkpoint split mismatch for {ticker_dir.name} fold={fold_id} "
                        f"field={field}: imported={metric.get(field)} current={expected_value}"
                    )
            stem = f"fold_{fold_id}"
            required = [
                ticker_dir / f"{stem}_complete.json",
                ticker_dir / f"{stem}_predictions.parquet",
                ticker_dir / f"{stem}_recipe.csv",
            ]
            missing = [path.name for path in required if not path.exists()]
            if missing:
                raise RuntimeError(f"incomplete checkpoint for {ticker_dir.name} fold={fold_id}: {missing}")
            fold_count += 1
        ticker_count += 1
    return ticker_count, fold_count


def main() -> None:
    project = Path(__file__).resolve().parent
    paths = get_paths(project)
    plan = load_plan(project)
    source = paths.data_root / "ticker_independent_cpu4_unlimited"
    target = paths.data_root / str(plan["result_subdir"])
    if not source.exists():
        print(f"Previous CPU4 result not found: {source}")
        return
    summary_path = source / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Previous CPU4 run_summary.json not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("tasks", {}).get("tasks_unfinished", -1)) != 0 or int(summary.get("tasks", {}).get("tasks_failed", -1)) != 0:
        raise RuntimeError("Previous CPU4 run is not a clean completed run; refusing checkpoint import")
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied += copy_if_exists(source / "ticker_cache", target / "ticker_cache")
    for name in ["ticker_data_readiness.csv"]:
        copied += copy_if_exists(source / name, target / name)
    source_models = source / "ticker_models"
    if source_models.exists():
        patterns = [
            "fold_*_complete.json", "fold_*_metrics.json", "fold_*_predictions.parquet",
            "fold_*_recipe.csv", "fold_*_importance.csv", "fold_*_error_correlations.csv",
            "fold_*_policy_scout.csv", "fold_*_model_scout.csv", "fold_*_window_scout.csv",
        ]
        for ticker_dir in source_models.iterdir():
            if not ticker_dir.is_dir():
                continue
            out = target / "ticker_models" / ticker_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for pattern in patterns:
                for file in ticker_dir.glob(pattern):
                    copied += copy_if_exists(file, out / file.name)
    ticker_count, fold_count = validate_imported_folds(target, plan)
    print(f"Imported {copied} cache/checkpoint items from:")
    print(source)
    print(f"Validated {fold_count} compatible folds across {ticker_count} tickers.")
    print("Initial outer folds will be reused. New rolling audits, common-fold replay, final models, and fingerprints will be rebuilt.")
    print(f"Target: {target}")


if __name__ == "__main__":
    main()
