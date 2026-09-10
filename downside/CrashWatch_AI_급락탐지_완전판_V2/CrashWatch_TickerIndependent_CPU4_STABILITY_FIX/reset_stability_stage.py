from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result_dir = get_paths(project).data_root / str(plan["result_subdir"])
    model_root = result_dir / "ticker_models"
    reset_tickers = 0
    if model_root.exists():
        for ticker_dir in model_root.iterdir():
            if not ticker_dir.is_dir():
                continue
            for directory in [ticker_dir / "recipe_replay", ticker_dir / "development_model"]:
                shutil.rmtree(directory, ignore_errors=True)
            for name in [
                "recipe_replay_candidates.csv", "recipe_replay_fold_metrics.csv",
                "recipe_replay_summary.csv", "recipe_replay_common_folds.json",
                "recipe_meta_oof_fold_metrics.csv", "recipe_meta_oof_summary.json",
                "task_summary.json", "development_model_error.json",
            ]:
                (ticker_dir / name).unlink(missing_ok=True)
            reset_tickers += 1
    registry = result_dir / "task_registry.sqlite"
    reset_tasks = 0
    if registry.exists():
        with sqlite3.connect(registry) as connection:
            reset_tasks = int(connection.execute("SELECT COUNT(*) FROM tasks WHERE stage='elite_model'").fetchone()[0])
            connection.execute(
                "UPDATE tasks SET status='pending',claimed_by=NULL,claimed_worker=NULL,claimed_at=NULL,"
                "heartbeat_at=NULL,completed_at=NULL,result_path=NULL,error='stability stage reset',attempts=0 "
                "WHERE stage='elite_model'"
            )
            connection.commit()
    print(f"Preserved initial outer fold checkpoints for {reset_tickers} ticker folders.")
    print(f"Reset {reset_tasks} elite tasks. Run RUN_CPU4_STABILITY_FIX.bat to replay recipes and rebuild final models.")


if __name__ == "__main__":
    main()
