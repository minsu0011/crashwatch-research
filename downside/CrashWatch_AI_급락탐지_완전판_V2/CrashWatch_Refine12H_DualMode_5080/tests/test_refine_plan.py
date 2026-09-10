from __future__ import annotations

import json
from pathlib import Path


def test_refine_plan_has_required_families_and_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs" / "refine12h_plan.json").read_text(encoding="utf-8"))
    assert plan["tree_families"] == ["xgboost", "lightgbm", "catboost"]
    assert set(plan["deep_families"]) == {"tcn", "transformer"}
    assert plan["pubg_threads"] == 4
    assert plan["full_tree_workers"] == 1
    assert plan["full_threads_per_tree_worker"] == 16
    assert plan["calibration_max_roc_drop"] <= 0.005
