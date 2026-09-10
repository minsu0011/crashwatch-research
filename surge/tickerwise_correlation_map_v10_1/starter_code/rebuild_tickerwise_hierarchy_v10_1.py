from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from surge_ticker_hierarchy_v10_1 import (
    SCHEMA_VERSION,
    RobustHierarchyConfig,
    aggregate_hierarchy_sensitivity,
    audit_hierarchy_levels,
    build_adaptive_probe_eligibility,
    build_fixed_signature_similarity,
    build_hierarchy_for_strength,
    build_precision_separator_map,
    build_profiles_v10_1,
    normalize_map_frames,
    _prepare_base_effect_frame,
    write_json,
)


_STRENGTH_WORKER_CONTEXT: tuple[pd.DataFrame, tuple[int, ...], dict[str, float], RobustHierarchyConfig] | None = None


def _initialize_strength_worker(
    base: pd.DataFrame,
    selection_folds: tuple[int, ...],
    weights: dict[str, float],
    config: RobustHierarchyConfig,
) -> None:
    global _STRENGTH_WORKER_CONTEXT
    _STRENGTH_WORKER_CONTEXT = (base, selection_folds, weights, config)


def _build_strength_worker(strength: float) -> pd.DataFrame:
    if _STRENGTH_WORKER_CONTEXT is None:
        raise RuntimeError("Strength worker was not initialized")
    base, selection_folds, weights, config = _STRENGTH_WORKER_CONTEXT
    return build_hierarchy_for_strength(base, selection_folds, float(strength), weights, config)


def log(message: str) -> None:
    print(f"[Tickerwise V10.1] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def find_v10_output(package_root: Path, explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"V10 output does not exist: {candidate}")
        return candidate
    candidates = [
        package_root / "outputs" / "surge_tickerwise_correlation_map_v10",
        package_root / "surge_tickerwise_correlation_map_v10",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate outputs/surge_tickerwise_correlation_map_v10. "
        "Pass --v10-output explicitly."
    )


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required V10 artifact missing: {path}")
    return path


def load_v10_maps(v10_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = pd.read_csv(require(v10_output / "ticker_target_summary.csv"), dtype={"ticker": str})
    eligibility = pd.read_csv(require(v10_output / "ticker_fold_eligibility.csv"), dtype={"ticker": str})
    summary_paths = [
        v10_output / "ticker_feature_map_summary.csv",
        v10_output / "ticker_extended_map_summary.csv",
    ]
    fold_paths = [
        v10_output / "ticker_feature_map_by_fold.csv",
        v10_output / "ticker_extended_map_by_fold.csv",
    ]
    summaries = [pd.read_csv(require(p), dtype={"ticker": str}) for p in summary_paths]
    folds = [pd.read_csv(require(p), dtype={"ticker": str}) for p in fold_paths]
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    by_fold = pd.concat(folds, ignore_index=True, sort=False)
    # Ensure a unique ticker-axis-node row. V10 raw and extended maps should not
    # overlap node IDs; if they do, prefer the row with stronger selection evidence.
    summary, by_fold = normalize_map_frames(summary, by_fold)
    if summary.duplicated(["ticker", "axis", "node_id"]).any():
        summary["_evidence"] = pd.to_numeric(summary.get("selection_evidence_score"), errors="coerce").fillna(0.0)
        summary = summary.sort_values(
            ["ticker", "axis", "node_id", "_evidence"], ascending=[True, True, True, False], kind="mergesort"
        ).drop_duplicates(["ticker", "axis", "node_id"], keep="first").drop(columns=["_evidence"])
        allowed = set(zip(summary["ticker"], summary["axis"], summary["node_id"]))
        by_fold = by_fold.loc[
            [(t, a, n) in allowed for t, a, n in zip(by_fold["ticker"], by_fold["axis"], by_fold["node_id"])]
        ].copy()
    return target, eligibility, summary, by_fold


def parse_strengths(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values or any(v <= 0 for v in values):
        raise ValueError("--prior-strength-grid must contain positive values")
    return values


def robust_classification(row: pd.Series, config: RobustHierarchyConfig) -> str:
    if not bool(row.get("ticker_specific_robust", False)):
        if abs(float(row.get("posterior_signed_effect_median", 0.0) or 0.0)) < config.weak_effect:
            return "WEAK_OR_NEUTRAL"
        return "PEER_SHARED_OR_LOW_HETEROGENEITY"
    if bool(row.get("direction_reversal_robust", False)):
        return "TICKER_DIRECTION_REVERSAL_ROBUST"
    posterior = float(row.get("posterior_signed_effect_median", 0.0) or 0.0)
    prior = float(row.get("peer_prior_signed_effect_median", 0.0) or 0.0)
    if abs(prior) < config.weak_effect and abs(posterior) >= config.strong_effect:
        return "TICKER_UNIQUE_DRIVER_ROBUST"
    if np.sign(posterior) == np.sign(prior):
        if abs(posterior) > abs(prior):
            return "TICKER_AMPLIFIED_DRIVER_ROBUST"
        return "TICKER_DAMPENED_DRIVER_ROBUST"
    return "TICKER_SPECIFIC_ROBUST"


def make_report(
    output: Path,
    hierarchy_audit: pd.DataFrame,
    robust_map: pd.DataFrame,
    precision: pd.DataFrame,
    eligibility: pd.DataFrame,
    similarity_edges: pd.DataFrame,
    signature_nodes: pd.DataFrame,
) -> None:
    industry_row = hierarchy_audit.loc[hierarchy_audit["level"].eq("industry")]
    industry_text = "UNKNOWN"
    if not industry_row.empty:
        r = industry_row.iloc[0]
        industry_text = f"valid={bool(r['valid'])}, reason={r['reason']}, effective_weight={float(r['effective_weight']):.3f}"
    specific = int(robust_map["ticker_specific_robust"].sum()) if not robust_map.empty else 0
    strong = int(robust_map["ticker_specific_strong"].sum()) if not robust_map.empty else 0
    unique_tickers = int(robust_map.loc[robust_map["ticker_specific_robust"], "ticker"].nunique()) if not robust_map.empty else 0
    psel = int(precision["precision_separator_selection_candidate"].sum()) if not precision.empty else 0
    pconf = int(precision["precision_separator_confirmed"].sum()) if not precision.empty else 0
    prec_tickers = int(precision.loc[precision["precision_separator_selection_candidate"], "ticker"].nunique()) if not precision.empty else 0
    recent_total = eligibility.loc[eligibility["role"].astype(str).eq("recent_audit")]
    recent_eligible = int(recent_total["v10_1_eligible_model"].sum()) if not recent_total.empty else 0
    recent_count = int(len(recent_total))
    lines = [
        "# CrashWatch Surge Tickerwise Map V10.1 correction report",
        "",
        "## 핵심 수정",
        "",
        "- V10의 hierarchy에서 이미 AUC shrinkage된 값을 다시 hierarchy shrinkage하던 이중 shrinkage를 제거했다.",
        "- industry/bucket/market metadata를 자동 감사하고 실제 peer hierarchy로 부적절한 level은 weight 0으로 만든다.",
        "- prior strength 20/40/80/120 등 여러 설정에서 효과가 살아남는지 sensitivity를 계산한다.",
        "- 절대 delta threshold 하나가 아니라 ticker-vs-peer heterogeneity z-score와 prior sensitivity로 종목 특이성을 판정한다.",
        "- AB 축에 대해 Selection-only precision-separator 후보와 Confirmation/Recent 진단을 분리했다.",
        "- 모든 ticker가 같은 raw signature node를 사용하도록 similarity map을 다시 계산한다.",
        "- fold 자체가 60행 미만인 recent에서 min_rows=60 때문에 0/48이 되던 eligibility를 adaptive threshold로 수정했다.",
        "",
        "## Metadata hierarchy audit",
        "",
        f"- industry: {industry_text}",
        "",
        "## Corrected ticker-specific map",
        "",
        f"- robust ticker-specific nodes: **{specific}**",
        f"- strong ticker-specific nodes: **{strong}**",
        f"- tickers with >=1 robust ticker-specific node: **{unique_tickers}**",
        "",
        "## Precision separator",
        "",
        f"- Selection-only AB precision-separator candidates: **{psel}**",
        f"- Confirmation-supported precision separators: **{pconf}**",
        f"- tickers with >=1 Selection precision separator: **{prec_tickers}**",
        "",
        "## Recent eligibility",
        "",
        f"- recent adaptive eligible ticker-folds: **{recent_eligible}/{recent_count}**",
        "",
        "## Similarity",
        "",
        f"- fixed signature nodes: **{len(signature_nodes)}**",
        f"- selected similarity edges: **{len(similarity_edges)}**",
        "",
        "## 해석 원칙",
        "",
        "Selection-only precision separator는 모델 선택 후보로 사용할 수 있다. Confirmation/Recent support는 진단용으로만 사용한다.",
        "종목 고유 신호 수가 늘었다는 이유만으로 신뢰하면 안 되며, prior-strength sensitivity와 heterogeneity z가 동시에 안정적인 노드만 우선한다.",
    ]
    (output / "V10_1_CORRECTION_REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild V10 ticker hierarchy and precision separators without retraining the expensive maps.")
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--v10-output", default=None)
    parser.add_argument("--output", default="outputs/surge_tickerwise_correlation_map_v10_1")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--prior-strength-grid", default="20,40,80,120")
    parser.add_argument("--strength-workers", type=int, default=1)
    parser.add_argument("--minimum-peer-tickers", type=int, default=2)
    parser.add_argument("--industry-weight", type=float, default=0.35)
    parser.add_argument("--bucket-weight", type=float, default=0.35)
    parser.add_argument("--market-weight", type=float, default=0.20)
    parser.add_argument("--global-weight", type=float, default=0.10)
    parser.add_argument("--minimum-reliability", type=float, default=0.08)
    parser.add_argument("--minimum-specific-z", type=float, default=1.25)
    parser.add_argument("--robust-specific-z", type=float, default=1.64)
    parser.add_argument("--precision-min-selection-auc", type=float, default=0.57)
    parser.add_argument("--precision-min-selection-min-auc", type=float, default=0.52)
    parser.add_argument("--precision-min-selection-folds", type=int, default=2)
    parser.add_argument("--precision-min-effective-n", type=float, default=12.0)
    parser.add_argument("--precision-min-matched-concordance", type=float, default=0.54)
    parser.add_argument("--similarity-feature-count", type=int, default=180)
    parser.add_argument("--similarity-min-node-coverage", type=float, default=0.75)
    parser.add_argument("--configured-min-validation-rows", type=int, default=60)
    parser.add_argument("--adaptive-recent-row-fraction", type=float, default=0.85)
    parser.add_argument("--minimum-recent-rows-floor", type=int, default=45)
    parser.add_argument("--minimum-validation-positive", type=int, default=3)
    parser.add_argument("--minimum-validation-negative", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    package_root = Path(args.package_root).expanduser().resolve()
    v10_output = find_v10_output(package_root, args.v10_output)
    output = Path(args.output)
    if not output.is_absolute():
        output = (package_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_folds = tuple(int(x.strip()) for x in str(args.selection_folds).split(",") if x.strip())
    config = RobustHierarchyConfig(
        prior_strength_grid=parse_strengths(args.prior_strength_grid),
        minimum_peer_tickers=int(args.minimum_peer_tickers),
        minimum_reliability=float(args.minimum_reliability),
        minimum_specific_z=float(args.minimum_specific_z),
        robust_specific_z=float(args.robust_specific_z),
        requested_industry_weight=float(args.industry_weight),
        requested_bucket_weight=float(args.bucket_weight),
        requested_market_weight=float(args.market_weight),
        requested_global_weight=float(args.global_weight),
        precision_min_selection_auc=float(args.precision_min_selection_auc),
        precision_min_selection_min_auc=float(args.precision_min_selection_min_auc),
        precision_min_selection_folds=int(args.precision_min_selection_folds),
        precision_min_effective_n=float(args.precision_min_effective_n),
        precision_min_matched_concordance=float(args.precision_min_matched_concordance),
        similarity_feature_count=int(args.similarity_feature_count),
        similarity_min_node_coverage=float(args.similarity_min_node_coverage),
    )
    log(f"Loading V10 output: {v10_output}")
    target, eligibility, summary, by_fold = load_v10_maps(v10_output)
    audit, weights = audit_hierarchy_levels(target, config)
    atomic_csv(output / "hierarchy_level_audit_v10_1.csv", audit)
    write_json(output / "HIERARCHY_LEVEL_AUDIT_V10_1.json", {
        "schema": SCHEMA_VERSION,
        "effective_weights": weights,
        "levels": audit.to_dict(orient="records"),
    })
    log("Effective hierarchy weights: " + ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))
    base = _prepare_base_effect_frame(summary, by_fold, target, selection_folds)
    strengths = tuple(float(value) for value in config.prior_strength_grid)
    strength_workers = max(1, min(int(args.strength_workers), len(strengths)))
    long_parts: list[pd.DataFrame] = []
    if strength_workers == 1:
        for strength in strengths:
            log(f"Hierarchy sensitivity prior_strength={strength:g}")
            long_parts.append(build_hierarchy_for_strength(base, selection_folds, strength, weights, config))
    else:
        log(f"Hierarchy sensitivity strengths={len(strengths)} workers={strength_workers}")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=strength_workers,
            initializer=_initialize_strength_worker,
            initargs=(base, selection_folds, dict(weights), config),
        ) as executor:
            # executor.map preserves the configured strength order, keeping
            # output and downstream sensitivity aggregation deterministic.
            for strength, current in zip(strengths, executor.map(_build_strength_worker, strengths)):
                log(f"Hierarchy sensitivity prior_strength={strength:g} complete")
                long_parts.append(current)
    sensitivity_long = pd.concat(long_parts, ignore_index=True, sort=False)
    atomic_csv(output / "ticker_hierarchy_sensitivity_long.csv", sensitivity_long)
    robust_map = aggregate_hierarchy_sensitivity(sensitivity_long, config)
    robust_map["ticker_effect_class_v10_1"] = robust_map.apply(lambda row: robust_classification(row, config), axis=1)
    atomic_csv(output / "ticker_hierarchical_effect_map_v10_1.csv", robust_map)
    precision = build_precision_separator_map(robust_map, config)
    atomic_csv(output / "ticker_precision_separator_map_v10_1.csv", precision)
    profiles, membership = build_profiles_v10_1(robust_map, precision)
    write_json(output / "TICKER_DRIVER_PROFILES_V10_1.json", profiles)
    atomic_csv(output / "ticker_driver_profile_membership_v10_1.csv", membership)
    edges, clusters, similarity, overlap, signature = build_fixed_signature_similarity(robust_map, config)
    atomic_csv(output / "ticker_similarity_edges_v10_1.csv", edges)
    atomic_csv(output / "ticker_map_clusters_v10_1.csv", clusters)
    similarity.to_csv(output / "ticker_similarity_matrix_v10_1.csv", encoding="utf-8-sig")
    overlap.to_csv(output / "ticker_similarity_overlap_v10_1.csv", encoding="utf-8-sig")
    atomic_csv(output / "ticker_similarity_signature_nodes_v10_1.csv", signature)
    eligibility_v101 = build_adaptive_probe_eligibility(
        eligibility,
        configured_min_rows=int(args.configured_min_validation_rows),
        recent_row_fraction=float(args.adaptive_recent_row_fraction),
        minimum_recent_rows_floor=int(args.minimum_recent_rows_floor),
        minimum_positive=int(args.minimum_validation_positive),
        minimum_negative=int(args.minimum_validation_negative),
    )
    atomic_csv(output / "ticker_probe_eligibility_v10_1.csv", eligibility_v101)
    taxonomy = robust_map.groupby(["axis", "ticker_effect_class_v10_1"], as_index=False).agg(
        nodes=("node_id", "size"), tickers=("ticker", "nunique"),
        mean_specificity_score=("ticker_specificity_score_v10_1", "mean"),
    )
    atomic_csv(output / "ticker_effect_taxonomy_summary_v10_1.csv", taxonomy)
    precision_summary = precision.groupby("ticker", as_index=False).agg(
        selection_candidates=("precision_separator_selection_candidate", "sum"),
        confirmed_candidates=("precision_separator_confirmed", "sum"),
        recent_confirmed_candidates=("precision_separator_recent_confirmed", "sum"),
        max_precision_separator_score=("precision_separator_score_v10_1", "max"),
    ) if not precision.empty else pd.DataFrame()
    atomic_csv(output / "ticker_precision_separator_summary_v10_1.csv", precision_summary)
    make_report(output, audit, robust_map, precision, eligibility_v101, edges, signature)
    final = {
        "schema": SCHEMA_VERSION,
        "status": "TICKER_HIERARCHY_CORRECTED_MAP_READY",
        "source_v10_output": str(v10_output),
        "source_v10_manifest_sha256": sha256_file(v10_output / "TICKER_SEPARATION_MAP_MANIFEST_V10.json") if (v10_output / "TICKER_SEPARATION_MAP_MANIFEST_V10.json").exists() else None,
        "prior_strength_grid": list(config.prior_strength_grid),
        "hierarchy_effective_weights": weights,
        "industry_hierarchy_valid": bool(audit.loc[audit["level"].eq("industry"), "valid"].iloc[0]) if not audit.loc[audit["level"].eq("industry")].empty else False,
        "robust_ticker_specific_nodes": int(robust_map["ticker_specific_robust"].sum()),
        "strong_ticker_specific_nodes": int(robust_map["ticker_specific_strong"].sum()),
        "tickers_with_robust_specific_nodes": int(robust_map.loc[robust_map["ticker_specific_robust"], "ticker"].nunique()),
        "selection_precision_separator_candidates": int(precision["precision_separator_selection_candidate"].sum()) if not precision.empty else 0,
        "confirmation_supported_precision_separators": int(precision["precision_separator_confirmed"].sum()) if not precision.empty else 0,
        "tickers_with_selection_precision_separators": int(precision.loc[precision["precision_separator_selection_candidate"], "ticker"].nunique()) if not precision.empty else 0,
        "fixed_similarity_nodes": int(len(signature)),
        "similarity_edges": int(len(edges)),
        "recent_adaptive_eligible": int(eligibility_v101.loc[eligibility_v101["role"].astype(str).eq("recent_audit"), "v10_1_eligible_model"].sum()),
        "selection_only_profiles_safe_for_model_selection": True,
        "confirmation_recent_are_diagnostic_only": True,
    }
    write_json(output / "FINAL_RECOMMENDATION_V10_1.json", final)
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            manifest_rows.append({"file": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_csv(output / "OUTPUT_INVENTORY_V10_1.csv", pd.DataFrame(manifest_rows))
    write_json(output / "RUN_STATUS.json", {"status": "SUCCESS", "schema": SCHEMA_VERSION})
    log("Completed corrected ticker hierarchy map.")
    log(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
