import json
from pathlib import Path


def test_plan_has_core_and_diagnostics():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs" / "base12h_plan.json").read_text(encoding="utf-8"))
    groups = {row["group"] for row in plan["global_groups"]}
    assert {"u_financial_shorting", "u_financial_market", "u_etf_pressure"} <= groups
    assert {"u_market_fx", "u_market_rates", "u_market_credit", "redundant_feature_cluster"} <= groups
    assert {
        "u_market_low_missing", "u_market_rates_conditions",
        "u_core_short_market_low_missing", "u_core_all_low_missing",
    } <= groups
    assert plan["outer_folds"] == 8
    assert len(plan["outer_seeds"]) >= 3
