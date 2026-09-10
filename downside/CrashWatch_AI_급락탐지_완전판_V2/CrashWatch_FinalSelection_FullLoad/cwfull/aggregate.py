from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.utils import atomic_json, bh_adjust, bootstrap_mean_ci, exact_sign_flip_p, hash_strings, read_json
from .profiles import FACTOR_FEATURES, PROTECTED_HARMFUL_TEN, PRIORITY_FEATURES

METRICS = [
    "raw_pr_auc",
    "raw_roc_auc",
    "raw_pr_auc_lift",
    "top_3pct_precision",
    "top_3pct_recall",
    "raw_brier",
    "raw_logloss",
]


def _read_tasks(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for file in path.rglob("*.json"):
            payload = read_json(file, {})
            if payload.get("status") == "completed" and "raw_pr_auc" in payload:
                rows.append(payload)
    return pd.DataFrame(rows)


def _profile_hashes(section: dict[str, Any]) -> dict[str, str]:
    return {name: str(payload["feature_hash"]) for name, payload in section.items()}


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    seen = set()
    current = name
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def _expanded_lgb(raw: pd.DataFrame, manifest: dict[str, Any], config_name: str) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    raw = raw[raw["config_name"].eq(config_name)].copy()
    if raw.empty:
        return raw
    if config_name == "operational":
        mapping = _profile_hashes(manifest["core_operational"]) | _profile_hashes(manifest["targeted_operational"])
        aliases = dict(manifest.get("operational_aliases", {}))
    else:
        mapping = _profile_hashes(manifest["diagnostic_full_column"])
        aliases = dict(manifest.get("diagnostic_aliases", {}))
        aliases.update(manifest.get("named_aliases", {}))
        aliases["P2_DEDUP_CLEAN"] = manifest["named_aliases"]["P2_DEDUP_CLEAN"]
        aliases["P1_EXACT_DEDUP"] = manifest["named_aliases"]["P1_EXACT_DEDUP"]
    rows = []
    names = list(mapping) + list(aliases)
    for name in names:
        source = _resolve_alias(name, aliases)
        profile_hash = mapping.get(source)
        if profile_hash is None:
            continue
        part = raw[raw["profile_hash"].eq(profile_hash)].copy()
        if part.empty:
            continue
        part["profile"] = name
        part["source_profile"] = source
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fold_summary(values: pd.DataFrame, delta_column: str = "delta_raw_pr_auc", recent_folds: list[int] | None = None) -> dict[str, Any]:
    recent_folds = recent_folds or [4, 5, 6, 7]
    fold = values.groupby("outer_fold", as_index=False)[delta_column].mean()
    vals = fold[delta_column].to_numpy(float)
    recent = fold[fold["outer_fold"].isin(recent_folds)][delta_column].to_numpy(float)
    if len(vals) == 0:
        return {}
    low, high = bootstrap_mean_ci(vals)
    return {
        "fold_count": int(len(vals)),
        "mean_delta": float(np.mean(vals)),
        "median_delta": float(np.median(vals)),
        "ci95_low": low,
        "ci95_high": high,
        "positive_fold_ratio": float(np.mean(vals > 0)),
        "worst_fold": float(np.min(vals)),
        "recent_mean_delta": float(np.mean(recent)) if len(recent) else None,
        "recent_positive_ratio": float(np.mean(recent > 0)) if len(recent) else None,
        "exact_sign_flip_p": exact_sign_flip_p(vals),
    }


def direct_compare(metrics: pd.DataFrame, a: str, b: str, recent_folds: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    if metrics.empty:
        return {}, pd.DataFrame()
    columns = ["outer_fold", "seed"] + METRICS
    left = metrics[metrics["profile"].eq(a)][columns].rename(columns={metric: f"a_{metric}" for metric in METRICS})
    right = metrics[metrics["profile"].eq(b)][columns].rename(columns={metric: f"b_{metric}" for metric in METRICS})
    paired = left.merge(right, on=["outer_fold", "seed"], how="inner")
    if paired.empty:
        return {}, paired
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"a_{metric}"] - paired[f"b_{metric}"]
    summary = _fold_summary(paired, "delta_raw_pr_auc", recent_folds)
    fold = paired.groupby("outer_fold", as_index=False)[[f"delta_{metric}" for metric in METRICS]].mean()
    summary.update(
        {
            "a": a,
            "b": b,
            "paired_model_count": int(len(paired)),
            "mean_delta_raw_roc_auc": float(fold["delta_raw_roc_auc"].mean()),
            "mean_delta_raw_pr_auc_lift": float(fold["delta_raw_pr_auc_lift"].mean()),
            "mean_delta_top_3pct_precision": float(fold["delta_top_3pct_precision"].mean()),
            "mean_delta_top_3pct_recall": float(fold["delta_top_3pct_recall"].mean()),
            "mean_delta_raw_brier": float(fold["delta_raw_brier"].mean()),
            "mean_delta_raw_logloss": float(fold["delta_raw_logloss"].mean()),
        }
    )
    return summary, paired


def _absolute_profile_summary(metrics: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    part = metrics[metrics["profile"].isin(profiles)]
    rows = []
    for profile, group in part.groupby("profile"):
        fold = group.groupby("outer_fold", as_index=False)[METRICS].mean()
        row = {"profile": profile, "fold_count": int(fold["outer_fold"].nunique()), "model_count": int(len(group))}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(fold[metric].mean())
            row[f"median_{metric}"] = float(fold[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _factorial_effects(metrics: pd.DataFrame, manifest: dict[str, Any], config_name: str, recent_folds: list[int]):
    factor_bits = {name: tuple(int(x) for x in bits) for name, bits in manifest["factor_bits"].items()}
    bit_to_name = {bits: name for name, bits in factor_bits.items()}
    main_rows = []
    main_pair_rows = []
    for factor_index, feature in enumerate(FACTOR_FEATURES):
        pieces = []
        for bits0 in itertools.product([0, 1], repeat=len(FACTOR_FEATURES)):
            if bits0[factor_index] != 0:
                continue
            bits1 = list(bits0)
            bits1[factor_index] = 1
            keep_name = bit_to_name[tuple(bits0)]
            remove_name = bit_to_name[tuple(bits1)]
            summary, paired = direct_compare(metrics, remove_name, keep_name, recent_folds)
            if paired.empty:
                continue
            paired = paired.copy()
            paired["factor"] = feature
            paired["keep_profile"] = keep_name
            paired["remove_profile"] = remove_name
            pieces.append(paired)
        if not pieces:
            continue
        all_pairs = pd.concat(pieces, ignore_index=True)
        # Average all matched combinations and seeds inside each outer fold.
        fold = all_pairs.groupby("outer_fold", as_index=False)["delta_raw_pr_auc"].mean()
        summary = _fold_summary(fold, "delta_raw_pr_auc", recent_folds)
        summary.update({"config_name": config_name, "factor": feature, "pair_rows": int(len(all_pairs))})
        main_rows.append(summary)
        main_pair_rows.append(all_pairs)

    interaction_rows = []
    interaction_detail = []
    for i, j in itertools.combinations(range(len(FACTOR_FEATURES)), 2):
        records = []
        other_positions = [position for position in range(len(FACTOR_FEATURES)) if position not in (i, j)]
        for other_values in itertools.product([0, 1], repeat=len(other_positions)):
            base = [0] * len(FACTOR_FEATURES)
            for position, value in zip(other_positions, other_values):
                base[position] = value
            names = {}
            for bi, bj in itertools.product([0, 1], repeat=2):
                bits = list(base)
                bits[i] = bi
                bits[j] = bj
                names[(bi, bj)] = bit_to_name[tuple(bits)]
            key = ["outer_fold", "seed"]
            frames = []
            for state, name in names.items():
                frame = metrics[metrics["profile"].eq(name)][key + ["raw_pr_auc"]].rename(
                    columns={"raw_pr_auc": f"y{state[0]}{state[1]}"}
                )
                frames.append(frame)
            merged = frames[0]
            for frame in frames[1:]:
                merged = merged.merge(frame, on=key, how="inner")
            if merged.empty:
                continue
            merged["interaction"] = merged["y11"] - merged["y10"] - merged["y01"] + merged["y00"]
            merged["factor_a"] = FACTOR_FEATURES[i]
            merged["factor_b"] = FACTOR_FEATURES[j]
            records.append(merged)
        if not records:
            continue
        detail = pd.concat(records, ignore_index=True)
        fold = detail.groupby("outer_fold", as_index=False)["interaction"].mean()
        summary = _fold_summary(fold.rename(columns={"interaction": "delta_raw_pr_auc"}), "delta_raw_pr_auc", recent_folds)
        summary.update(
            {
                "config_name": config_name,
                "factor_a": FACTOR_FEATURES[i],
                "factor_b": FACTOR_FEATURES[j],
                "pair_rows": int(len(detail)),
            }
        )
        interaction_rows.append(summary)
        interaction_detail.append(detail)
    return (
        pd.DataFrame(main_rows),
        pd.concat(main_pair_rows, ignore_index=True) if main_pair_rows else pd.DataFrame(),
        pd.DataFrame(interaction_rows),
        pd.concat(interaction_detail, ignore_index=True) if interaction_detail else pd.DataFrame(),
    )


def _experiment_summary(metrics: pd.DataFrame, manifest: dict[str, Any], config_name: str, recent_folds: list[int]) -> pd.DataFrame:
    rows = []
    meta = manifest["experiment_meta"]
    baseline = {"P1": "P1_EXACT_DEDUP", "P2": "P2_DEDUP_CLEAN", "P4": "P4_CORR095", "P7": "P7_CORR095_PLUS_CONDITIONAL"}
    for profile, item in meta.items():
        if config_name == "diagnostic" and not (item["family"] == "harmful10" and item["context"] == "P1"):
            continue
        if profile not in set(metrics.get("profile", pd.Series(dtype=str))):
            continue
        summary, _ = direct_compare(metrics, profile, baseline[item["context"]], recent_folds)
        if not summary:
            continue
        rows.append({"config_name": config_name, "profile": profile, **item, **summary})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_q"] = bh_adjust(result["exact_sign_flip_p"].to_numpy(float))
    return result


def _harmful_final_policy(experiments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in PROTECTED_HARMFUL_TEN:
        group = experiments[(experiments["family"] == "harmful10") & (experiments["feature"] == feature)]
        if group.empty:
            continue
        def row_for(config: str, context: str):
            part = group[(group["config_name"] == config) & (group["context"] == context)]
            return part.iloc[0] if not part.empty else None
        p1_op = row_for("operational", "P1")
        p1_diag = row_for("diagnostic", "P1")
        p2 = row_for("operational", "P2")
        p7 = row_for("operational", "P7")
        direct_support = bool(
            p1_op is not None
            and p1_diag is not None
            and p1_op["operation"] == "drop"
            and float(p1_op["mean_delta"]) >= 0.003
            and float(p1_op["positive_fold_ratio"]) >= 0.75
            and float(p1_op["recent_positive_ratio"]) >= 0.75
            and float(p1_diag["mean_delta"]) >= 0.0
        )
        context_safe = True
        for context_row in [p2, p7]:
            if context_row is None:
                continue
            if context_row["operation"] == "drop" and float(context_row["mean_delta"]) < 0:
                context_safe = False
            if context_row["operation"] == "add" and float(context_row["mean_delta"]) > 0:
                context_safe = False
        policy = "REVIEW_DELETE_CANDIDATE" if direct_support and context_safe else "PROTECTED_DO_NOT_DELETE"
        rows.append(
            {
                "feature": feature,
                "final_policy": policy,
                "p1_operational_drop_delta": None if p1_op is None else float(p1_op["mean_delta"]),
                "p1_diagnostic_drop_delta": None if p1_diag is None else float(p1_diag["mean_delta"]),
                "p2_operation": None if p2 is None else str(p2["operation"]),
                "p2_delta": None if p2 is None else float(p2["mean_delta"]),
                "p7_operation": None if p7 is None else str(p7["operation"]),
                "p7_delta": None if p7 is None else float(p7["mean_delta"]),
                "old_drop_label_used": False,
            }
        )
    return pd.DataFrame(rows)


def _secondary_votes(summary: dict[str, Any]) -> int:
    if not summary:
        return 0
    votes = 0
    votes += int(float(summary.get("mean_delta_raw_roc_auc", 0)) > 0)
    votes += int(float(summary.get("mean_delta_top_3pct_precision", 0)) > 0)
    votes += int(float(summary.get("mean_delta_top_3pct_recall", 0)) > 0)
    votes += int(float(summary.get("mean_delta_raw_brier", 0)) < 0)
    votes += int(float(summary.get("mean_delta_raw_logloss", 0)) < 0)
    return votes


def _lock_profile(
    lgb: pd.DataFrame,
    xgb: pd.DataFrame,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    p2_causal: dict[str, Any],
    recent_folds: list[int],
) -> dict[str, Any]:
    lgb_p7_p2, _ = direct_compare(lgb, "P7_CORR095_PLUS_CONDITIONAL", "P2_DEDUP_CLEAN", recent_folds)
    xgb_p7_p2, _ = direct_compare(xgb, "P7_CORR095_PLUS_CONDITIONAL", "P2_DEDUP_CLEAN", recent_folds)
    lgb_p7_p0, _ = direct_compare(lgb, "P7_CORR095_PLUS_CONDITIONAL", "P0_FULL_439", recent_folds)
    xgb_p7_p0, _ = direct_compare(xgb, "P7_CORR095_PLUS_CONDITIONAL", "P0_FULL_439", recent_folds)
    expected_lgb = 3 * 8 * len(cfg["selection"]["seeds"])
    expected_xgb = 3 * 8 * len(cfg["selection"]["seeds"])
    lgb_complete = len(lgb[lgb["profile"].isin(["P0_FULL_439", "P2_DEDUP_CLEAN", "P7_CORR095_PLUS_CONDITIONAL"])]) >= expected_lgb
    xgb_complete = len(xgb[xgb["profile"].isin(["P0_FULL_439", "P2_DEDUP_CLEAN", "P7_CORR095_PLUS_CONDITIONAL"])]) >= expected_xgb
    result: dict[str, Any] = {
        "status": "NO_LOCK_INCOMPLETE",
        "locked_profile": None,
        "sealed_evaluation_allowed": False,
        "lgb_complete": lgb_complete,
        "xgb_complete": xgb_complete,
        "lgb_p7_vs_p2": lgb_p7_p2,
        "xgb_p7_vs_p2": xgb_p7_p2,
        "lgb_p7_vs_p0": lgb_p7_p0,
        "xgb_p7_vs_p0": xgb_p7_p0,
        "p2_deletion_rationale_invalidated": bool(p2_causal.get("harmful3_deletion_rationale_invalidated", False)),
        "decision_rule_version": "final_selection_lock_v1",
    }
    if not (lgb_complete and xgb_complete and lgb_p7_p2 and xgb_p7_p2):
        result["reason"] = ["P0/P2/P7의 LightGBM·XGBoost 8-fold×5-seed 결과가 모두 필요합니다."]
        return result

    lgb_delta = float(lgb_p7_p2["mean_delta"])
    xgb_delta = float(xgb_p7_p2["mean_delta"])
    weighted = 0.5 * lgb_delta + 0.5 * xgb_delta
    p7_votes = _secondary_votes(lgb_p7_p2) + _secondary_votes(xgb_p7_p2)
    p2_invalid = bool(p2_causal.get("harmful3_deletion_rationale_invalidated", False))
    p7_safe = bool(
        lgb_p7_p0 and xgb_p7_p0
        and float(lgb_p7_p0["mean_delta"]) >= float(cfg["lock"]["p7_vs_p0_safety_floor"])
        and float(xgb_p7_p0["mean_delta"]) >= float(cfg["lock"]["p7_vs_p0_safety_floor"])
    )
    reason = []
    locked = None
    status = "LOCKED_BY_PREREGISTERED_RULE"
    if p2_invalid and p7_safe:
        locked = "P7_CORR095_PLUS_CONDITIONAL"
        reason = ["C1/C4가 P2보다 높아 harmful3 삭제 근거가 무효화됨", "P7은 P0 대비 안전 하한 통과"]
    elif lgb_delta >= float(cfg["lock"]["material_advantage"]) and xgb_delta >= float(cfg["lock"]["material_advantage"]):
        locked = "P7_CORR095_PLUS_CONDITIONAL"
        reason = ["LightGBM과 XGBoost 모두 P7의 실질적 PR-AUC 우위"]
    elif lgb_delta <= -float(cfg["lock"]["material_advantage"]) and xgb_delta <= -float(cfg["lock"]["material_advantage"]):
        locked = "P2_DEDUP_CLEAN"
        reason = ["LightGBM과 XGBoost 모두 P2의 실질적 PR-AUC 우위"]
    else:
        p7_noninferior = (
            lgb_delta >= float(cfg["lock"]["p7_lgb_noninferiority_margin"])
            and xgb_delta >= float(cfg["lock"]["p7_xgb_noninferiority_margin"])
            and weighted >= float(cfg["lock"]["combined_noninferiority_margin"])
        )
        recent_support = (
            float(lgb_p7_p2.get("recent_positive_ratio") or 0) >= 0.75
            or float(xgb_p7_p2.get("recent_positive_ratio") or 0) >= 0.75
        )
        if p7_noninferior and p7_safe and (recent_support or p7_votes >= int(cfg["lock"]["minimum_secondary_votes"])):
            locked = "P7_CORR095_PLUS_CONDITIONAL"
            reason = ["P7 비열등 기준 통과", "최근 fold·보조지표·피처 수 30개 절감의 사전등록 tie-break 적용"]
            status = "LOCKED_BY_NONINFERIORITY_TIEBREAK"
        elif weighted < float(cfg["lock"]["p2_weighted_advantage_threshold"]):
            locked = "P2_DEDUP_CLEAN"
            reason = ["두 모델의 가중 PR-AUC에서 P2 우위"]
            status = "LOCKED_BY_WEIGHTED_TIEBREAK"
        elif p7_safe:
            locked = "P7_CORR095_PLUS_CONDITIONAL"
            reason = ["차이가 작아 사전등록된 안정성·간결성 tie-break로 P7 선택"]
            status = "LOCKED_BY_PARSIMONY_TIEBREAK"
        else:
            result["status"] = "NO_LOCK_SAFETY_FLOOR_FAILED"
            result["reason"] = ["P7이 P0 안전 하한을 통과하지 못했고 P2의 일관된 우위도 없음"]
            return result

    result.update(
        {
            "status": status,
            "locked_profile": locked,
            "sealed_evaluation_allowed": True,
            "reason": reason,
            "weighted_p7_minus_p2_pr_auc": weighted,
            "p7_secondary_votes_out_of_10": p7_votes,
            "locked_feature_count": int(manifest["base_profiles"][locked]["count"]),
            "locked_feature_hash": str(manifest["base_profiles"][locked]["feature_hash"]),
        }
    )
    return result


def aggregate(output_dir: Path, manifest: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    recent_folds = [int(value) for value in cfg["selection"]["recent_folds"]]

    lgb_raw = _read_tasks(output_dir / "task_results" / "lightgbm")
    xgb = _read_tasks(output_dir / "task_results" / "xgboost")
    lgb_operational = _expanded_lgb(lgb_raw, manifest, "operational")
    lgb_diagnostic = _expanded_lgb(lgb_raw, manifest, "diagnostic")
    lgb_raw.to_csv(results_dir / "lightgbm_raw_unique_metrics.csv", index=False, encoding="utf-8-sig")
    lgb_operational.to_csv(results_dir / "lightgbm_operational_metrics_expanded.csv", index=False, encoding="utf-8-sig")
    lgb_diagnostic.to_csv(results_dir / "lightgbm_diagnostic_metrics_expanded.csv", index=False, encoding="utf-8-sig")
    xgb.to_csv(results_dir / "xgboost_profile_metrics.csv", index=False, encoding="utf-8-sig")

    comparison_profiles = ["P0_FULL_439", "P2_DEDUP_CLEAN", "P7_CORR095_PLUS_CONDITIONAL"]
    lgb_abs = _absolute_profile_summary(lgb_operational, comparison_profiles)
    xgb_abs = _absolute_profile_summary(xgb, comparison_profiles)
    lgb_abs.to_csv(results_dir / "lightgbm_p0_p2_p7_absolute.csv", index=False, encoding="utf-8-sig")
    xgb_abs.to_csv(results_dir / "xgboost_p0_p2_p7_absolute.csv", index=False, encoding="utf-8-sig")

    profile_comparisons = {}
    comparison_rows = []
    for backend, frame in [("lightgbm", lgb_operational), ("xgboost", xgb)]:
        for a, b in [
            ("P2_DEDUP_CLEAN", "P0_FULL_439"),
            ("P7_CORR095_PLUS_CONDITIONAL", "P0_FULL_439"),
            ("P7_CORR095_PLUS_CONDITIONAL", "P2_DEDUP_CLEAN"),
        ]:
            summary, paired = direct_compare(frame, a, b, recent_folds)
            profile_comparisons[f"{backend}:{a}_vs_{b}"] = summary
            if not paired.empty:
                paired["backend"] = backend
                paired["comparison"] = f"{a}_vs_{b}"
                comparison_rows.append(paired)
    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(
            results_dir / "p0_p2_p7_paired_deltas.csv", index=False, encoding="utf-8-sig"
        )
    atomic_json(profile_comparisons, results_dir / "p0_p2_p7_comparison_summary.json")

    factorial_outputs = []
    for config_name, frame in [("operational", lgb_operational), ("diagnostic", lgb_diagnostic)]:
        main, main_detail, interactions, interaction_detail = _factorial_effects(frame, manifest, config_name, recent_folds)
        factorial_outputs.append(main)
        main_detail.to_csv(results_dir / f"factorial_main_detail_{config_name}.csv", index=False, encoding="utf-8-sig")
        interactions.to_csv(results_dir / f"factorial_interactions_{config_name}.csv", index=False, encoding="utf-8-sig")
        interaction_detail.to_csv(results_dir / f"factorial_interaction_detail_{config_name}.csv", index=False, encoding="utf-8-sig")
    factorial_main = pd.concat(factorial_outputs, ignore_index=True) if factorial_outputs else pd.DataFrame()
    if not factorial_main.empty:
        factorial_main["bh_q_within_all_main_tests"] = bh_adjust(factorial_main["exact_sign_flip_p"].to_numpy(float))
    factorial_main.to_csv(results_dir / "factorial_main_effects.csv", index=False, encoding="utf-8-sig")

    c1_vs_p2, _ = direct_compare(lgb_operational, "C1_P1_MINUS_HIGH_MISSING", "P2_DEDUP_CLEAN", recent_folds)
    c2_vs_p2, _ = direct_compare(lgb_operational, "C2_P1_MINUS_HARMFUL3", "P2_DEDUP_CLEAN", recent_folds)
    c0_vs_p2, _ = direct_compare(lgb_operational, "C0_P1_EXACT_DEDUP", "P2_DEDUP_CLEAN", recent_folds)
    c5_vs_p2, _ = direct_compare(lgb_operational, "C5_P2_RESTORE_BALANCE_SLOPE", "P2_DEDUP_CLEAN", recent_folds)
    c6_vs_p2, _ = direct_compare(lgb_operational, "C6_P2_RESTORE_VOLUME_SUM5", "P2_DEDUP_CLEAN", recent_folds)
    c7_vs_p2, _ = direct_compare(lgb_operational, "C7_P2_RESTORE_COCRASH_FREQ", "P2_DEDUP_CLEAN", recent_folds)
    p2_causal = {
        "C0_vs_P2": c0_vs_p2,
        "C1_or_C4_vs_P2_restore_harmful3_group": c1_vs_p2,
        "C2_vs_P2_restore_high_missing2": c2_vs_p2,
        "restore_balance_slope_vs_P2": c5_vs_p2,
        "restore_volume_sum5_vs_P2": c6_vs_p2,
        "restore_cocrash_vs_P2": c7_vs_p2,
        "harmful3_deletion_rationale_invalidated": bool(c1_vs_p2 and float(c1_vs_p2["mean_delta"]) > 0),
        "exact_user_rule": "C1/C4 mean PR-AUC > P2이면 harmful3 삭제 근거 무효",
        "C1_C4_same_feature_set": True,
    }
    atomic_json(p2_causal, results_dir / "P2_CAUSAL_DECOMPOSITION.json")

    operational_experiments = _experiment_summary(lgb_operational, manifest, "operational", recent_folds)
    diagnostic_experiments = _experiment_summary(lgb_diagnostic, manifest, "diagnostic", recent_folds)
    experiments = pd.concat([operational_experiments, diagnostic_experiments], ignore_index=True)
    experiments.to_csv(results_dir / "targeted_experiment_summary.csv", index=False, encoding="utf-8-sig")
    harmful_policy = _harmful_final_policy(experiments)
    harmful_policy.to_csv(results_dir / "harmful10_final_policy.csv", index=False, encoding="utf-8-sig")
    priority = experiments[experiments["family"].eq("priority")].copy()
    priority.to_csv(results_dir / "priority_feature_context_tests.csv", index=False, encoding="utf-8-sig")
    conditional = experiments[experiments["family"].eq("conditional")].copy()
    conditional.to_csv(results_dir / "p7_conditional_feature_tests.csv", index=False, encoding="utf-8-sig")

    lock = _lock_profile(lgb_operational, xgb, manifest, cfg, p2_causal, recent_folds)
    atomic_json(lock, results_dir / "LOCKED_PROFILE.json")
    if lock.get("locked_profile"):
        profile = str(lock["locked_profile"])
        features = list(manifest["base_profiles"][profile]["features"])
        atomic_json(
            {
                "profile": profile,
                "feature_count": len(features),
                "feature_hash": hash_strings(features),
                "features": features,
                "selection_result": lock,
            },
            results_dir / "LOCKED_FEATURE_MANIFEST.json",
        )
        pd.DataFrame({"feature": features}).to_csv(results_dir / "LOCKED_FEATURES.csv", index=False, encoding="utf-8-sig")
        (results_dir / "LOCKED_FEATURES.txt").write_text("\n".join(features) + "\n", encoding="utf-8")

    expected_xgb = 3 * 8 * len(cfg["selection"]["seeds"])
    summary = {
        "status": "completed",
        "lightgbm_unique_completed_models": int(len(lgb_raw)),
        "lightgbm_operational_expanded_rows": int(len(lgb_operational)),
        "lightgbm_diagnostic_expanded_rows": int(len(lgb_diagnostic)),
        "xgboost_completed_models": int(len(xgb)),
        "xgboost_expected_models": expected_xgb,
        "xgboost_complete": int(len(xgb)) >= expected_xgb,
        "p2_causal": p2_causal,
        "lock": lock,
        "harmful10_policy_counts": harmful_policy["final_policy"].value_counts().to_dict() if not harmful_policy.empty else {},
        "feature_master_decision_drop_labels_used": False,
        "sealed_data_used": False,
    }
    atomic_json(summary, results_dir / "FINAL_SELECTION_SUMMARY.json")

    report_lines = [
        "CrashWatch 최종 프로필 선택 실험 요약",
        "=" * 56,
        f"LightGBM 완료 모델: {len(lgb_raw):,}",
        f"XGBoost 완료 모델: {len(xgb):,}/{expected_xgb:,}",
        f"P2 harmful3 삭제 근거 무효화: {p2_causal['harmful3_deletion_rationale_invalidated']}",
        f"잠긴 프로필: {lock.get('locked_profile')}",
        f"잠금 상태: {lock.get('status')}",
        "",
        "주의: feature_master_decision.csv의 DROP 라벨은 사용하지 않았습니다.",
        "Sealed 데이터는 이 선택 실험에서 읽지 않았습니다.",
        "Sealed 평가는 별도 run_sealed_once.py로 단 한 번만 실행합니다.",
    ]
    (results_dir / "FINAL_RESULT_GUIDE_KO.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary
