from __future__ import annotations

from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result_dir = get_paths(project).data_root / str(plan["result_subdir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "STOP_REQUESTED"
    path.write_text("manual safe stop requested", encoding="utf-8")
    print(f"Safe stop requested: {path}")
    print("The current fold is allowed to save; rerun the main RUN file to resume.")


if __name__ == "__main__":
    main()
