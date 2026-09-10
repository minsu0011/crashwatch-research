from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import atomic_csv, atomic_json


def load_strength_map(project: Path) -> pd.DataFrame:
    path = project / "configs" / "ticker_strength_map.csv"
    frame = pd.read_csv(path, dtype={"ticker": str})
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    frame["training_enabled"] = frame["training_enabled"].astype(str).str.lower().isin(["true", "1", "yes"])
    frame["priority"] = pd.to_numeric(frame["priority"], errors="coerce").fillna(0).astype(int)
    return frame


def strength_lookup(project: Path) -> dict[str, dict[str, Any]]:
    frame = load_strength_map(project)
    return {str(row["ticker"]): row.to_dict() for _, row in frame.iterrows()}


def _filter_candidates(plan: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in plan["model_candidates"]]
    if not names:
        return candidates
    allowed = set(map(str, names))
    return [item for item in candidates if str(item.get("name")) in allowed]


def plan_for_tier(base_plan: dict[str, Any], tier: str) -> dict[str, Any]:
    plan = copy.deepcopy(base_plan)
    overrides = dict(plan.get("tier_overrides", {}).get(str(tier), {}))
    candidate_names = overrides.pop("model_candidate_names", None)
    plan.update(overrides)
    plan["model_candidates"] = _filter_candidates(base_plan, candidate_names)
    plan["active_search_tier"] = str(tier)
    if not plan["model_candidates"]:
        raise RuntimeError(f"no model candidates for tier={tier}")
    return plan


def write_locked_manifest(project: Path, result_dir: Path, data_root: Path, plan: dict[str, Any]) -> pd.DataFrame:
    strength = load_strength_map(project)
    locked = strength.loc[strength["search_tier"].eq("strong_lock")].copy()
    previous = data_root / str(plan.get("previous_result_subdir", "ticker_independent_cpu4_stability_fix"))
    locked["previous_result_dir"] = str(previous)
    locked["previous_model_dir"] = locked["ticker"].map(lambda t: str(previous / "ticker_models" / str(t) / "development_model"))
    locked["model_dir_exists"] = locked["ticker"].map(lambda t: (previous / "ticker_models" / str(t) / "development_model" / "model_card.json").exists())
    locked["action"] = "preserved_without_retraining"
    atomic_csv(locked, result_dir / "strong_locked_manifest.csv")
    atomic_csv(strength, result_dir / "ticker_strength_map.csv")
    return locked


def _baseline_meta_oof(previous: Path) -> pd.DataFrame:
    path = previous / "recipe_meta_oof_summary_all.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns and "source_path" in frame.columns:
        frame["ticker"] = frame["source_path"].astype(str).str.extract(r"ticker_models[\\/](\d{6})")
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    return frame


def build_improvement_report(project: Path, result_dir: Path, data_root: Path, plan: dict[str, Any]) -> pd.DataFrame:
    previous = data_root / str(plan.get("previous_result_subdir", "ticker_independent_cpu4_stability_fix"))
    baseline = _baseline_meta_oof(previous)
    current_path = result_dir / "recipe_meta_oof_summary_all.csv"
    current = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    if not current.empty and "ticker" not in current.columns and "source_path" in current.columns:
        current["ticker"] = current["source_path"].astype(str).str.extract(r"ticker_models[\\/](\d{6})")
    for frame in (baseline, current):
        if not frame.empty and "ticker" in frame.columns:
            frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    strength = load_strength_map(project)
    if baseline.empty or current.empty:
        output = strength.copy()
        output["comparison_status"] = "missing_baseline_or_current_meta_oof"
        atomic_csv(output, result_dir / "weak_search_improvement.csv")
        return output
    bcols = {
        "raw_pr_auc_mean": "baseline_pr_auc",
        "raw_roc_auc_mean": "baseline_roc_auc",
        "raw_pr_auc_std": "baseline_pr_std",
        "raw_pr_auc_min": "baseline_worst_pr",
        "balanced_accuracy_mean": "baseline_balanced_accuracy",
        "top_10pct_precision_mean": "baseline_top10_precision",
    }
    ccols = {key: value.replace("baseline_", "current_") for key, value in bcols.items()}
    b = baseline[["ticker", *[c for c in bcols if c in baseline.columns]]].rename(columns=bcols)
    c = current[["ticker", *[c for c in ccols if c in current.columns]]].rename(columns=ccols)
    out = strength.merge(b, on="ticker", how="left").merge(c, on="ticker", how="left")
    for metric in ["pr_auc", "roc_auc", "balanced_accuracy", "top10_precision"]:
        left, right = f"current_{metric}", f"baseline_{metric}"
        if left in out and right in out:
            out[f"delta_{metric}"] = pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")
    if {"current_pr_std", "baseline_pr_std"}.issubset(out.columns):
        out["delta_pr_std_lower_is_better"] = pd.to_numeric(out["current_pr_std"], errors="coerce") - pd.to_numeric(out["baseline_pr_std"], errors="coerce")
    if {"current_worst_pr", "baseline_worst_pr"}.issubset(out.columns):
        out["delta_worst_pr"] = pd.to_numeric(out["current_worst_pr"], errors="coerce") - pd.to_numeric(out["baseline_worst_pr"], errors="coerce")
    out["promotion_score"] = (
        0.9 * pd.to_numeric(out.get("delta_pr_auc"), errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(out.get("delta_roc_auc"), errors="coerce").fillna(0)
        + 0.9 * pd.to_numeric(out.get("delta_worst_pr"), errors="coerce").fillna(0)
        - 0.9 * pd.to_numeric(out.get("delta_pr_std_lower_is_better"), errors="coerce").fillna(0)
    )
    out["promoted_to_strong_candidate"] = (
        pd.to_numeric(out.get("current_pr_auc"), errors="coerce").ge(float(plan.get("strong_lock_pr_auc", 0.55)))
        & pd.to_numeric(out.get("current_roc_auc"), errors="coerce").ge(float(plan.get("strong_lock_roc_auc", 0.60)))
        & pd.to_numeric(out.get("current_worst_pr"), errors="coerce").ge(float(plan.get("strong_lock_worst_pr", 0.28)))
        & pd.to_numeric(out.get("current_pr_std"), errors="coerce").le(float(plan.get("strong_lock_pr_std", 0.20)))
    )
    out["comparison_status"] = np.where(out["current_pr_auc"].notna(), "compared", "not_run_or_incomplete")
    atomic_csv(out, result_dir / "weak_search_improvement.csv")
    promoted = out.loc[out["promoted_to_strong_candidate"].fillna(False)].copy()
    atomic_csv(promoted, result_dir / "new_strong_candidates.csv")
    lines = [
        "# CrashWatch 약한 종목 대규모 탐색 결과", "",
        f"- 비교 완료 종목: {int((out['comparison_status'] == 'compared').sum())}",
        f"- 새 강한 모델 후보: {len(promoted)}",
        f"- 강한 모델 잠금: {int((out['search_tier'] == 'strong_lock').sum())}",
        "", "## 새 강한 후보", "",
    ]
    if promoted.empty:
        lines.append("- 없음")
    else:
        for _, row in promoted.sort_values("promotion_score", ascending=False).iterrows():
            lines.append(
                f"- {row.get('name')} ({row.get('ticker')}): "
                f"PR={row.get('current_pr_auc'):.4f}, ROC={row.get('current_roc_auc'):.4f}, "
                f"worst PR={row.get('current_worst_pr'):.4f}, PR std={row.get('current_pr_std'):.4f}"
            )
    compared = out.loc[out["comparison_status"].eq("compared")].copy()
    if not compared.empty:
        lines.extend(["", "## 개선 폭 상위", ""] )
        for _, row in compared.sort_values("promotion_score", ascending=False).head(12).iterrows():
            lines.append(
                f"- {row.get('name')} ({row.get('ticker')}): "
                f"ΔPR={row.get('delta_pr_auc', float('nan')):.4f}, "
                f"ΔROC={row.get('delta_roc_auc', float('nan')):.4f}, "
                f"Δworst PR={row.get('delta_worst_pr', float('nan')):.4f}, "
                f"ΔPR std={row.get('delta_pr_std_lower_is_better', float('nan')):.4f}"
            )
    (result_dir / "WEAK_SEARCH_BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def copy_previous_diagnostics(data_root: Path, result_dir: Path, plan: dict[str, Any]) -> None:
    previous = data_root / str(plan.get("previous_result_subdir", "ticker_independent_cpu4_stability_fix"))
    destination = result_dir / "previous_stability_reference"
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "recipe_meta_oof_summary_all.csv", "ticker_elite_summary.csv", "ticker_elite_bottleneck_map.csv",
        "ticker_model_fingerprints.csv", "independent_vs_pooled_exact.csv", "RESULT_BRIEF.md",
    ]:
        source = previous / name
        if source.exists():
            shutil.copy2(source, destination / name)
    atomic_json({"previous_result_dir": str(previous), "copied_to": str(destination)}, destination / "reference_manifest.json")
