from pathlib import Path

import pandas as pd

import dual_ablation.base12h.runner as runner


def test_compile_pairs_within_block_and_backend(tmp_path: Path, monkeypatch):
    rows = []
    for backend, fold, seed in [("cpu", 0, 17), ("cuda", 1, 17)]:
        block_id = f"{backend}_{fold}_{seed}"
        common = {
            "block_id": block_id, "dataset_signature": "sig", "backend": backend,
            "threads": 4 if backend == "cpu" else 8, "outer_fold": fold, "seed": seed,
            "scope_type": "all", "scope_value": "all_validation", "config": "c",
            "calibration_method": "none", "decision_threshold": 0.5,
            "positive_rate": 0.2, "accuracy": 0.7, "balanced_accuracy": 0.6,
            "roc_auc": 0.66, "raw_roc_auc": 0.67, "brier": 0.18, "logloss": 0.55,
            "top_3pct_precision": 0.4, "pr_auc_lift": 1.7, "brier_skill": 0.0,
            "logloss_skill": 0.0, "mean_prediction": 0.2,
        }
        rows.append({**common, "task_id": f"b{fold}", "experiment": "baseline", "stage": "baseline", "mode": "none", "group": None, "target_bucket": None, "pr_auc": 0.34, "raw_pr_auc": 0.35})
        rows.append({**common, "task_id": f"a{fold}", "experiment": "global__u_financial_shorting", "stage": "base12h_global", "mode": "global_drop", "group": "u_financial_shorting", "target_bucket": None, "pr_auc": 0.32, "raw_pr_auc": 0.33})
    metrics = pd.DataFrame(rows)
    monkeypatch.setattr(runner, "_collect_task_outputs", lambda _: (metrics, pd.DataFrame()))
    plan = {"success_targets": {"roc_auc": 0.7, "balanced_accuracy": 0.65, "pr_auc_lift": 1.5, "top_3pct_precision": 0.4}}
    output = runner._compile_summary(tmp_path, plan, "sig")
    assert output["paired_rows"] == 2
    paired = pd.read_csv(tmp_path / "paired_ablation_deltas.csv")
    assert paired["backend_pair_ok"].all()
    assert set(paired["backend"]) == {"cpu", "cuda"}
