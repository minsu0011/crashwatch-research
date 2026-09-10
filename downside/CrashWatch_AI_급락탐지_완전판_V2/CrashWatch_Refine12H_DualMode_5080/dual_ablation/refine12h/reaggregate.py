from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..finance_nested.runner import _bh_qvalues, _exact_sign_p, _metric_bundle
from ..io_utils import atomic_csv, atomic_json
from .calibration import SafeCalibrationPolicy, apply_calibrator, select_safe_policy
from .data import RefineDataBundle


@lru_cache(maxsize=8)
def _source_file_index(source_root_text: str) -> dict[str, tuple[str, ...]]:
    """Index the source once; legacy records can contain mojibake absolute paths."""
    source_root = Path(source_root_text)
    output: dict[str, list[str]] = {}
    for path in source_root.rglob("*"):
        if path.is_file():
            output.setdefault(path.name, []).append(str(path))
    return {name: tuple(paths) for name, paths in output.items()}


def _suffix_match_score(candidate: Path, path_text: str) -> int:
    expected = [part.lower() for part in path_text.replace("\\", "/").split("/") if part]
    actual = [part.lower() for part in candidate.parts]
    score = 0
    for left, right in zip(reversed(expected), reversed(actual)):
        if left != right:
            break
        score += 1
    return score


def _resolve_existing(path_text: str | None, source_root: Path, filename_hint: str | None = None) -> Path | None:
    if path_text:
        path = Path(path_text)
        if path.exists():
            return path
        if not path.is_absolute():
            candidate = source_root / path
            if candidate.exists():
                return candidate
        # The Korean prefix in old absolute paths may be mojibake while the
        # ASCII subtree from ``ablation_base12h`` onward remains intact.
        normalized = path_text.replace("\\", "/")
        marker = f"/{source_root.name}/"
        marker_index = normalized.lower().rfind(marker.lower())
        if marker_index >= 0:
            relative_text = normalized[marker_index + len(marker) :]
            candidate = source_root.joinpath(*[part for part in relative_text.split("/") if part])
            if candidate.exists():
                return candidate
        filename_hint = filename_hint or path.name
    if filename_hint:
        indexed = _source_file_index(str(source_root.resolve()))
        matches = [Path(value) for value in indexed.get(filename_hint, ())]
        if matches:
            if path_text:
                return max(matches, key=lambda candidate: _suffix_match_score(candidate, path_text))
            return matches[0]
    return None


def _load_policy_index(source_root: Path) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    index: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    policy_roots = [source_root / "policies", source_root / "nested_policies"]
    for root in policy_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                experiment = payload.get("experiment", {})
                name = experiment.get("name") if isinstance(experiment, dict) else payload.get("experiment")
                if not name:
                    continue
                fold = int(payload.get("outer_fold", -1))
                backend = str(payload.get("backend", ""))
                config_payload = payload.get("config", {})
                config = str(config_payload.get("name", payload.get("config_name", ""))) if isinstance(config_payload, dict) else str(config_payload)
                payload["_policy_file"] = str(path)
                index[(str(name), fold, backend, config)] = payload
            except Exception:
                continue
    return index


def _join_regime(frame: pd.DataFrame, bundle: RefineDataBundle) -> np.ndarray | None:
    if not bundle.regime_features:
        return None
    lookup = getattr(bundle, "_refine_regime_lookup", None)
    if lookup is None:
        columns = ["date", "ticker", *bundle.regime_features]
        source = bundle.df[columns].copy()
        source["date"] = pd.to_datetime(source["date"]).dt.tz_localize(None)
        source["ticker"] = source["ticker"].astype(str).str.zfill(6)
        source = source.drop_duplicates(["date", "ticker"], keep="last")
        lookup = source.set_index(["date", "ticker"])[bundle.regime_features].sort_index()
        setattr(bundle, "_refine_regime_lookup", lookup)
    keys = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(frame["date"]).dt.tz_localize(None),
            frame["ticker"].astype(str).str.zfill(6),
        ],
        names=["date", "ticker"],
    )
    return lookup.reindex(keys).to_numpy(dtype=np.float64)


def _scope_frame(prediction: pd.DataFrame, scope_type: str, scope_value: str) -> pd.DataFrame:
    if scope_type in {"all", "all_validation"}:
        return prediction
    if scope_type in {"bucket", "target_bucket"}:
        return prediction.loc[prediction["bucket"].astype(str).eq(str(scope_value))]
    if scope_type in {"ticker", "target_ticker"}:
        return prediction.loc[prediction["ticker"].astype(str).str.zfill(6).eq(str(scope_value).zfill(6))]
    return prediction


def _policy_from_payload(payload: dict[str, Any]) -> SafeCalibrationPolicy:
    return SafeCalibrationPolicy(**payload)


def _find_calibration_prediction(
    record: dict[str, Any],
    metadata: pd.DataFrame,
    policy_index: dict[tuple[str, int, str, str], dict[str, Any]],
    source_root: Path,
) -> pd.DataFrame | None:
    experiment = str(record.get("experiment", metadata.iloc[0].get("experiment", "")))
    fold = int(record.get("outer_fold", metadata.iloc[0].get("outer_fold", -1)))
    backend = str(record.get("backend", metadata.iloc[0].get("backend", "")))
    config = str(record.get("config", metadata.iloc[0].get("config", "")))
    payload = policy_index.get((experiment, fold, backend, config))
    if payload is None:
        candidates = [value for key, value in policy_index.items() if key[:3] == (experiment, fold, backend)]
        payload = candidates[0] if candidates else None
    if payload is None:
        return None
    raw_path_text = payload.get("raw_prediction_path")
    raw_path = _resolve_existing(raw_path_text, source_root, Path(raw_path_text).name if raw_path_text else None)
    if raw_path is None:
        return None
    return pd.read_parquet(raw_path)


def _safe_policy_for_record(
    record: dict[str, Any],
    metadata: pd.DataFrame,
    policy_index: dict[tuple[str, int, str, str], dict[str, Any]],
    source_root: Path,
    bundle: RefineDataBundle,
    plan: dict[str, Any],
    baseline_policy: SafeCalibrationPolicy | None,
) -> tuple[SafeCalibrationPolicy, str]:
    calibration = _find_calibration_prediction(record, metadata, policy_index, source_root)
    if calibration is None or "raw_prediction" not in calibration.columns:
        fallback = SafeCalibrationPolicy(
            method="none", params={}, threshold=baseline_policy.threshold if baseline_policy else 0.5,
            selection_score=math.inf, validation_brier=math.nan, validation_logloss=math.nan,
            validation_raw_roc_auc=math.nan, validation_calibrated_roc_auc=math.nan,
            validation_raw_pr_auc=math.nan, validation_calibrated_pr_auc=math.nan,
            rank_correlation=1.0, validation_rows=0, validation_positives=0,
            fallback_reason="calibration raw prediction cache missing",
        )
        return fallback, "missing_calibration_cache"
    regime = _join_regime(calibration, bundle)
    policy = select_safe_policy(
        calibration["target"].to_numpy(dtype=np.int8),
        calibration["raw_prediction"].to_numpy(dtype=float),
        pd.to_datetime(calibration["date"]).to_numpy(),
        regime=regime,
        regime_names=bundle.regime_features,
        methods=tuple(plan["calibration_methods"]),
        forced_method=baseline_policy.method if baseline_policy else None,
        forced_threshold=baseline_policy.threshold if baseline_policy else None,
        max_roc_drop=float(plan["calibration_max_roc_drop"]),
        max_pr_drop=float(plan["calibration_max_pr_drop"]),
        min_rank_correlation=float(plan["calibration_min_rank_correlation"]),
    )
    return policy, "fitted"


def _paired_and_summary(metrics: pd.DataFrame, result_dir: Path, plan: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        empty = pd.DataFrame()
        atomic_csv(empty, result_dir / "safe_paired_ablation_deltas.csv")
        atomic_csv(empty, result_dir / "safe_ablation_summary.csv")
        return empty, empty
    key = ["outer_fold", "seed", "backend", "scope_type", "scope_value"]
    baseline = metrics.loc[metrics["experiment"].eq("baseline")].copy()
    experiments = metrics.loc[~metrics["experiment"].eq("baseline")].copy()
    base_cols = [
        "raw_pr_auc", "raw_roc_auc", "pr_auc", "roc_auc", "balanced_accuracy",
        "brier", "logloss", "top_3pct_precision", "brier_skill", "logloss_skill",
    ]
    rename = {column: f"baseline_{column}" for column in base_cols}
    compared = experiments.merge(baseline[key + base_cols].rename(columns=rename), on=key, how="inner", validate="many_to_one")
    for metric in ["raw_pr_auc", "raw_roc_auc", "pr_auc", "roc_auc", "balanced_accuracy", "top_3pct_precision"]:
        compared[f"{metric}_loss"] = compared[f"baseline_{metric}"] - compared[metric]
    compared["brier_increase"] = compared["brier"] - compared["baseline_brier"]
    compared["logloss_increase"] = compared["logloss"] - compared["baseline_logloss"]
    atomic_csv(compared, result_dir / "safe_paired_ablation_deltas.csv")

    group_keys = ["experiment", "stage", "mode", "group", "target_bucket", "scope_type", "scope_value"]
    metric_cols = [
        "raw_pr_auc_loss", "raw_roc_auc_loss", "pr_auc_loss", "roc_auc_loss",
        "balanced_accuracy_loss", "top_3pct_precision_loss", "brier_increase", "logloss_increase",
    ]
    rows: list[dict[str, Any]] = []
    for values, block in compared.groupby(group_keys, dropna=False, sort=False):
        row = dict(zip(group_keys, values if isinstance(values, tuple) else (values,)))
        fold = block.groupby("outer_fold", as_index=False)[metric_cols].mean(numeric_only=True)
        row["fold_count"] = int(fold["outer_fold"].nunique())
        row["seed_count"] = int(block["seed"].nunique())
        row["pair_count"] = int(len(block))
        for column in metric_cols:
            array = pd.to_numeric(fold[column], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(array)) if len(array) else math.nan
            row[f"{column}_median"] = float(np.median(array)) if len(array) else math.nan
            row[f"{column}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else math.nan
            row[f"{column}_positive_fold_ratio"] = float(np.mean(array > 0)) if len(array) else math.nan
            row[f"{column}_sign_p"] = _exact_sign_p(array)
            for backend in sorted(block["backend"].dropna().unique()):
                backend_values = block.loc[block["backend"].eq(backend)].groupby("outer_fold")[column].mean().dropna().to_numpy(dtype=float)
                row[f"{backend}_{column}_mean"] = float(np.mean(backend_values)) if len(backend_values) else math.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["test_family"] = np.where(summary["stage"].astype(str).str.contains("bucket"), "bucket", "global")
        for column in ["raw_pr_auc_loss_sign_p", "pr_auc_loss_sign_p"]:
            summary[column.replace("_p", "_q")] = summary.groupby("test_family", group_keys=False)[column].apply(_bh_qvalues)
    atomic_csv(summary, result_dir / "safe_ablation_summary.csv")

    gate_rows: list[dict[str, Any]] = []
    gating = plan["gating"]
    for _, row in summary.iterrows():
        is_bucket = str(row.get("scope_type")) == "target_bucket"
        threshold = float(gating["bucket_min_effect"] if is_bucket else gating["global_min_effect"])
        raw_effect = float(row.get("raw_pr_auc_loss_mean", math.nan))
        calibrated_effect = float(row.get("pr_auc_loss_mean", math.nan))
        raw_ratio = float(row.get("raw_pr_auc_loss_positive_fold_ratio", math.nan))
        calibrated_ratio = float(row.get("pr_auc_loss_positive_fold_ratio", math.nan))
        direction_agreement = np.isfinite(raw_effect) and np.isfinite(calibrated_effect) and raw_effect > 0 and calibrated_effect > 0
        backend_means = [
            float(row[column]) for column in row.index
            if column.endswith("_raw_pr_auc_loss_mean") and column not in {"raw_pr_auc_loss_mean"} and pd.notna(row[column])
        ]
        backend_agreement = all(value >= 0 for value in backend_means) if backend_means else True
        keep = (
            direction_agreement
            and raw_effect >= threshold
            and calibrated_effect >= threshold
            and raw_ratio >= float(gating["min_positive_fold_ratio"])
            and calibrated_ratio >= float(gating["min_positive_fold_ratio"])
            and (backend_agreement or not bool(gating["require_backend_direction_agreement"]))
        )
        gate_rows.append({
            "experiment": row.get("experiment"), "stage": row.get("stage"), "group": row.get("group"),
            "target_bucket": row.get("target_bucket"), "scope_type": row.get("scope_type"),
            "raw_pr_auc_loss_mean": raw_effect, "calibrated_pr_auc_loss_mean": calibrated_effect,
            "raw_positive_fold_ratio": raw_ratio, "calibrated_positive_fold_ratio": calibrated_ratio,
            "direction_agreement": direction_agreement, "backend_direction_agreement": backend_agreement,
            "effect_threshold": threshold, "keep": bool(keep),
            "decision": "keep" if keep else "drop_or_hold",
        })
    atomic_csv(pd.DataFrame(gate_rows), result_dir / "gating_decisions_safe.csv")
    return compared, summary


def reaggregate_base12h_raw(
    source_root: Path,
    result_dir: Path,
    bundle: RefineDataBundle,
    plan: dict[str, Any],
) -> dict[str, Any]:
    source_root = source_root.resolve()
    records_root = source_root / "task_records"
    if not records_root.exists():
        report = {"status": "missing_source_task_records", "source_root": str(source_root)}
        atomic_json(report, result_dir / "raw_reaggregation_status.json")
        return report
    policy_index = _load_policy_index(source_root)
    records: list[tuple[dict[str, Any], Path, pd.DataFrame]] = []
    for record_path in sorted(records_root.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            metrics_path = _resolve_existing(record.get("metrics_path"), source_root, Path(record.get("metrics_path", "")).name)
            prediction_path = _resolve_existing(record.get("prediction_path"), source_root, Path(record.get("prediction_path", "")).name)
            if metrics_path is None or prediction_path is None:
                continue
            metadata = pd.read_parquet(metrics_path)
            if metadata.empty:
                continue
            record["_prediction_path"] = str(prediction_path)
            records.append((record, record_path, metadata))
        except Exception:
            continue
    # Baseline first so every ablation can reuse method and threshold from the same fold/backend/config.
    records.sort(key=lambda item: (0 if item[0].get("experiment") == "baseline" else 1, int(item[0].get("outer_fold", -1)), int(item[0].get("seed", -1))))
    baseline_policies: dict[tuple[int, str, str], SafeCalibrationPolicy] = {}
    policy_cache: dict[tuple[Any, ...], tuple[SafeCalibrationPolicy, str]] = {}
    metric_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    missing_cache = 0
    for record, record_path, metadata in records:
        prediction = pd.read_parquet(Path(record["_prediction_path"]))
        if "raw_prediction" not in prediction.columns:
            continue
        fold = int(record.get("outer_fold", metadata.iloc[0].get("outer_fold", -1)))
        backend = str(record.get("backend", metadata.iloc[0].get("backend", "")))
        config = str(record.get("config", metadata.iloc[0].get("config", "")))
        key = (fold, backend, config)
        baseline_policy = None if record.get("experiment") == "baseline" else baseline_policies.get(key)
        baseline_constraint = (
            None
            if baseline_policy is None
            else (baseline_policy.method, round(float(baseline_policy.threshold), 12))
        )
        policy_key = (str(record.get("experiment", "")), fold, backend, config, baseline_constraint)
        cached_policy = policy_cache.get(policy_key)
        if cached_policy is None:
            cached_policy = _safe_policy_for_record(
                record, metadata, policy_index, source_root, bundle, plan, baseline_policy
            )
            policy_cache[policy_key] = cached_policy
        policy, status = cached_policy
        if status != "fitted":
            missing_cache += 1
        if record.get("experiment") == "baseline":
            baseline_policies[key] = policy
        regime = _join_regime(prediction, bundle)
        calibrated = apply_calibrator(policy.method, policy.params, prediction["raw_prediction"].to_numpy(dtype=float), regime)
        prediction = prediction.copy()
        prediction["safe_prediction"] = calibrated
        for _, meta in metadata.iterrows():
            scope_type = str(meta.get("scope_type", "all"))
            scope_value = str(meta.get("scope_value", "all_validation"))
            block = _scope_frame(prediction, scope_type, scope_value)
            if block.empty:
                continue
            values = _metric_bundle(
                block["target"].to_numpy(dtype=np.int8),
                block["raw_prediction"].to_numpy(dtype=float),
                block["safe_prediction"].to_numpy(dtype=float),
                pd.to_datetime(block["date"]).to_numpy(),
                policy.threshold,
            )
            row = meta.to_dict()
            row.update(values)
            row["calibration_method"] = policy.method
            row["calibration_fallback_reason"] = policy.fallback_reason
            row["safe_reaggregation"] = True
            metric_rows.append(row)
        policy_rows.append({
            "experiment": record.get("experiment"), "outer_fold": fold, "seed": record.get("seed"),
            "backend": backend, "config": config, "method": policy.method, "threshold": policy.threshold,
            "validation_raw_roc_auc": policy.validation_raw_roc_auc,
            "validation_calibrated_roc_auc": policy.validation_calibrated_roc_auc,
            "validation_raw_pr_auc": policy.validation_raw_pr_auc,
            "validation_calibrated_pr_auc": policy.validation_calibrated_pr_auc,
            "rank_correlation": policy.rank_correlation, "fallback_reason": policy.fallback_reason,
            "record_path": str(record_path),
        })
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(metrics, result_dir / "safe_reaggregated_metrics.csv")
    atomic_csv(pd.DataFrame(policy_rows), result_dir / "calibration_safety_audit.csv")
    compared, summary = _paired_and_summary(metrics, result_dir, plan)
    report = {
        "status": "completed" if len(metrics) else "no_metrics",
        "source_root": str(source_root), "records_found": len(records), "metrics_rows": len(metrics),
        "paired_rows": len(compared), "summary_rows": len(summary), "missing_calibration_cache_records": missing_cache,
        "baseline_policy_count": len(baseline_policies),
    }
    atomic_json(report, result_dir / "raw_reaggregation_status.json")
    return report
