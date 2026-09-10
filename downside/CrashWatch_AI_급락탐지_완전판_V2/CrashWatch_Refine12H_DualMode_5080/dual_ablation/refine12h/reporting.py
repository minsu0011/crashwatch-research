from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..finance_nested.runner import _metric_bundle
from ..io_utils import atomic_csv, atomic_json, atomic_parquet
from .calibration import apply_calibrator, select_safe_policy


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_parquet_outputs(root: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.rglob(pattern))
    frames = []
    for path in paths:
        try:
            frame = pd.read_parquet(path)
            frame["source_file"] = str(path)
            frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def compile_model_summary(result_dir: Path) -> dict[str, Any]:
    metrics = read_parquet_outputs(result_dir / "model_metrics", "*.parquet")
    atomic_csv(metrics, result_dir / "model_family_metrics.csv")
    if metrics.empty:
        atomic_csv(pd.DataFrame(), result_dir / "model_family_summary.csv")
        return {"metrics_rows": 0, "summary_rows": 0}
    numeric = [
        "raw_pr_auc", "raw_roc_auc", "pr_auc", "roc_auc", "balanced_accuracy", "accuracy",
        "brier", "logloss", "brier_skill", "logloss_skill", "pr_auc_lift", "raw_pr_auc_lift",
        "top_3pct_precision", "positive_rate", "mean_prediction", "rows",
    ]
    rows: list[dict[str, Any]] = []
    for values, block in metrics.groupby(["family", "feature_profile"], dropna=False, sort=False):
        family, profile = values
        fold = block.groupby("outer_fold", as_index=False)[numeric].mean(numeric_only=True)
        row: dict[str, Any] = {
            "family": family, "feature_profile": profile,
            "fold_count": int(fold["outer_fold"].nunique()), "seed_count": int(block["seed"].nunique()),
            "task_count": int(block["task_id"].nunique()),
        }
        for column in numeric:
            values_array = pd.to_numeric(fold[column], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values_array)) if len(values_array) else math.nan
            row[f"{column}_std"] = float(np.std(values_array, ddof=1)) if len(values_array) > 1 else math.nan
        row["backend_list"] = ",".join(sorted(set(block["actual_backend"].dropna().astype(str))))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["base_score"] = (
            summary["raw_pr_auc_mean"].fillna(0.0)
            + 0.50 * summary["raw_roc_auc_mean"].fillna(0.0)
            + 0.10 * summary["brier_skill_mean"].fillna(-1.0)
            + 0.05 * summary["logloss_skill_mean"].fillna(-1.0)
            - 0.25 * summary["raw_pr_auc_std"].fillna(0.0)
        )
        summary = summary.sort_values("base_score", ascending=False)
    atomic_csv(summary, result_dir / "model_family_summary.csv")

    paired_rows = []
    for (family, fold_id, seed), block in metrics.groupby(["family", "outer_fold", "seed"], sort=False):
        indexed = block.set_index("feature_profile")
        if not {"full_reduced", "common_period"}.issubset(indexed.index):
            continue
        full = indexed.loc["full_reduced"]
        common = indexed.loc["common_period"]
        paired_rows.append({
            "family": family, "outer_fold": fold_id, "seed": seed,
            "backend_full": full.get("actual_backend"), "backend_common": common.get("actual_backend"),
            "backend_pair_ok": full.get("actual_backend") == common.get("actual_backend"),
            "common_minus_full_raw_pr_auc": common.get("raw_pr_auc") - full.get("raw_pr_auc"),
            "common_minus_full_raw_roc_auc": common.get("raw_roc_auc") - full.get("raw_roc_auc"),
            "common_minus_full_brier_skill": common.get("brier_skill") - full.get("brier_skill"),
            "common_minus_full_logloss_skill": common.get("logloss_skill") - full.get("logloss_skill"),
        })
    paired = pd.DataFrame(paired_rows)
    atomic_csv(paired, result_dir / "common_vs_full_paired.csv")

    backend_audit = metrics.groupby(["family", "feature_profile", "actual_backend"], as_index=False).agg(
        tasks=("task_id", "nunique"), raw_pr_auc=("raw_pr_auc", "mean"), raw_roc_auc=("raw_roc_auc", "mean"),
        brier_skill=("brier_skill", "mean"), elapsed_seconds=("elapsed_seconds", "mean"),
    )
    atomic_csv(backend_audit, result_dir / "model_backend_audit.csv")
    return {"metrics_rows": len(metrics), "summary_rows": len(summary), "paired_rows": len(paired)}


def _optimize_weights(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    count = predictions.shape[1]
    initial = np.full(count, 1.0 / count, dtype=float)

    def objective(weights: np.ndarray) -> float:
        blended = np.clip(predictions @ weights, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(blended) + (1 - y) * np.log(1 - blended)))

    result = minimize(
        objective, initial, method="SLSQP", bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return initial
    weights = np.maximum(0.0, result.x)
    return weights / max(1e-12, weights.sum())


def compile_ensembles(result_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    outer_paths = sorted((result_dir / "predictions" / "outer").rglob("*.parquet"))
    calibration_paths = sorted((result_dir / "predictions" / "calibration").rglob("*.parquet"))
    outer_index: dict[tuple[str, int, int], list[Path]] = {}
    calibration_index: dict[tuple[str, int, int], list[Path]] = {}
    for path in outer_paths:
        try:
            frame = pd.read_parquet(path, columns=["feature_profile", "outer_fold", "seed"])
            if frame.empty:
                continue
            key = (str(frame.iloc[0]["feature_profile"]), int(frame.iloc[0]["outer_fold"]), int(frame.iloc[0]["seed"]))
            outer_index.setdefault(key, []).append(path)
        except Exception:
            continue
    for path in calibration_paths:
        try:
            frame = pd.read_parquet(path, columns=["feature_profile", "outer_fold", "seed"])
            if frame.empty:
                continue
            key = (str(frame.iloc[0]["feature_profile"]), int(frame.iloc[0]["outer_fold"]), int(frame.iloc[0]["seed"]))
            calibration_index.setdefault(key, []).append(path)
        except Exception:
            continue

    metric_rows = []
    weight_rows = []
    disagreement_rows = []
    for key, paths in sorted(outer_index.items()):
        if len(paths) < 2 or key not in calibration_index:
            continue
        profile, fold_id, seed = key
        cal_frames = [pd.read_parquet(path) for path in calibration_index[key]]
        out_frames = [pd.read_parquet(path) for path in paths]
        cal_merged = None
        out_merged = None
        families = []
        for frame in cal_frames:
            family = str(frame.iloc[0]["family"])
            column = f"p_{family}"
            block = frame[["date", "ticker", "target", "raw_prediction"]].rename(columns={"raw_prediction": column})
            cal_merged = block if cal_merged is None else cal_merged.merge(block, on=["date", "ticker", "target"], how="inner")
            families.append(family)
        families = list(dict.fromkeys(families))
        for frame in out_frames:
            family = str(frame.iloc[0]["family"])
            if family not in families:
                continue
            column = f"p_{family}"
            block = frame[["date", "ticker", "bucket", "target", "raw_prediction"]].rename(columns={"raw_prediction": column})
            out_merged = block if out_merged is None else out_merged.merge(block, on=["date", "ticker", "bucket", "target"], how="inner")
        if cal_merged is None or out_merged is None:
            continue
        common_families = [family for family in families if f"p_{family}" in cal_merged and f"p_{family}" in out_merged]
        if len(common_families) < 2:
            continue
        cal_matrix = cal_merged[[f"p_{family}" for family in common_families]].to_numpy(dtype=float)
        out_matrix = out_merged[[f"p_{family}" for family in common_families]].to_numpy(dtype=float)
        weights = _optimize_weights(cal_merged["target"].to_numpy(dtype=np.int8), cal_matrix)
        cal_raw = np.clip(cal_matrix @ weights, 1e-7, 1 - 1e-7)
        out_raw = np.clip(out_matrix @ weights, 1e-7, 1 - 1e-7)
        policy = select_safe_policy(
            cal_merged["target"].to_numpy(dtype=np.int8), cal_raw, pd.to_datetime(cal_merged["date"]).to_numpy(),
            methods=tuple(method for method in plan["calibration_methods"] if method != "regime_logistic"),
            max_roc_drop=float(plan["calibration_max_roc_drop"]), max_pr_drop=float(plan["calibration_max_pr_drop"]),
            min_rank_correlation=float(plan["calibration_min_rank_correlation"]),
        )
        calibrated = apply_calibrator(policy.method, policy.params, out_raw)
        metrics = _metric_bundle(
            out_merged["target"].to_numpy(dtype=np.int8), out_raw, calibrated,
            pd.to_datetime(out_merged["date"]).to_numpy(), policy.threshold,
        )
        task_id = f"ensemble__{profile}__fold{fold_id}__seed{seed}"
        metric_rows.append({
            "task_id": task_id, "family": "ensemble", "feature_profile": profile,
            "outer_fold": fold_id, "seed": seed, "actual_backend": "mixed_predictions",
            "calibration_method": policy.method, "component_count": len(common_families), **metrics,
        })
        for family, weight in zip(common_families, weights):
            weight_rows.append({"task_id": task_id, "family": family, "weight": float(weight)})
        correlation = pd.DataFrame(out_matrix, columns=common_families).corr()
        for left in common_families:
            for right in common_families:
                if left < right:
                    disagreement_rows.append({
                        "feature_profile": profile, "outer_fold": fold_id, "seed": seed,
                        "left": left, "right": right, "prediction_correlation": float(correlation.loc[left, right]),
                        "mean_absolute_difference": float(np.mean(np.abs(out_merged[f"p_{left}"] - out_merged[f"p_{right}"]))),
                    })
        prediction = out_merged[["date", "ticker", "bucket", "target"]].copy()
        prediction["raw_prediction"] = out_raw.astype(np.float32)
        prediction["prediction"] = calibrated.astype(np.float32)
        prediction["family"] = "ensemble"
        prediction["feature_profile"] = profile
        prediction["outer_fold"] = fold_id
        prediction["seed"] = seed
        atomic_parquet(prediction, result_dir / "predictions" / "ensemble" / f"{task_id}.parquet")
    ensemble_metrics = pd.DataFrame(metric_rows)
    atomic_csv(ensemble_metrics, result_dir / "ensemble_metrics.csv")
    atomic_csv(pd.DataFrame(weight_rows), result_dir / "ensemble_weights.csv")
    atomic_csv(pd.DataFrame(disagreement_rows), result_dir / "model_disagreement.csv")
    return {"ensemble_rows": len(ensemble_metrics), "weight_rows": len(weight_rows)}


def write_bottleneck_report(result_dir: Path) -> dict[str, Any]:
    model_summary = _read_csv_or_empty(result_dir / "model_family_summary.csv")
    calibration = _read_csv_or_empty(result_dir / "model_calibration_audit.csv")
    safe_gate = _read_csv_or_empty(result_dir / "gating_decisions_safe.csv")
    common = _read_csv_or_empty(result_dir / "common_vs_full_paired.csv")
    diagnostics: list[dict[str, Any]] = []
    if not calibration.empty:
        inversion = int((pd.to_numeric(calibration.get("calibrated_roc_auc"), errors="coerce") < pd.to_numeric(calibration.get("raw_roc_auc"), errors="coerce") - 0.01).sum())
        diagnostics.append({
            "bottleneck": "calibration_rank_inversion", "status": "resolved" if inversion == 0 else "remaining",
            "evidence": f"unsafe calibration rows={inversion}",
            "interpretation": "positive-slope constraints and rank guards should prevent Fold 2-style inversion",
            "next_action": "remaining rows use no-calibration fallback; inspect regime features if fallback rate is high",
        })
    if not model_summary.empty:
        best = model_summary.iloc[0]
        diagnostics.append({
            "bottleneck": "model_ranking_capacity", "status": "remaining" if float(best.get("raw_roc_auc_mean", 0)) < 0.70 else "improved",
            "evidence": f"best={best.get('family')}:{best.get('feature_profile')}, raw ROC={best.get('raw_roc_auc_mean', math.nan):.4f}, raw PR={best.get('raw_pr_auc_mean', math.nan):.4f}",
            "interpretation": "identical splits isolate whether tree family or temporal architecture adds ranking signal",
            "next_action": "use the best stable family as base; retain deep model only if it improves ensemble and disagreement is useful",
        })
    if not common.empty:
        diagnostics.append({
            "bottleneck": "missingness_period_bias", "status": "measured",
            "evidence": f"mean common-full raw PR delta={pd.to_numeric(common['common_minus_full_raw_pr_auc'], errors='coerce').mean():.5f}",
            "interpretation": "positive values support removing high-missing features and restricting to common availability",
            "next_action": "choose common-period only if gains persist across at least 6/8 folds and recent coverage is adequate",
        })
    if not safe_gate.empty:
        diagnostics.append({
            "bottleneck": "sector_gating_overfit", "status": "controlled",
            "evidence": f"kept gates={int(safe_gate['keep'].astype(bool).sum())}/{len(safe_gate)}",
            "interpretation": "raw and calibrated directions must agree before a gate is admitted",
            "next_action": "do not promote held gates until a later untouched time window confirms them",
        })
    frame = pd.DataFrame(diagnostics)
    atomic_csv(frame, result_dir / "bottleneck_report.csv")
    atomic_json({"diagnostics": diagnostics}, result_dir / "bottleneck_report.json")
    return {"diagnostic_rows": len(frame)}


def desktop_path() -> Path:
    candidates = [Path.home() / "Desktop", Path(os.environ.get("USERPROFILE", "")) / "Desktop"]
    for path in candidates:
        if str(path) and path.exists():
            return path
    return Path.cwd()


def create_result_package(result_dir: Path, destination: Path | None = None) -> Path:
    timestamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    destination = destination or desktop_path() / f"CrashWatch_Refine12H_RESULTS_{timestamp}.zip"
    include = [
        "POST_RUN_AUDIT.md", "execution_failure.log", "experiment_plan.json",
        "RESULT_BRIEF.md", "run_summary.json", "registry_status.json", "refine_data_summary.json",
        "raw_reaggregation_status.json", "calibration_safety_audit.csv", "safe_ablation_summary.csv",
        "safe_paired_ablation_deltas.csv", "safe_reaggregated_metrics.csv", "feature_quality_refine.csv",
        "gating_decisions_safe.csv", "redundant_features_maintained.csv", "high_missing_features_removed.csv",
        "common_period_coverage.csv", "model_family_summary.csv", "model_family_metrics.csv",
        "model_family_metrics_with_ensemble.csv",
        "common_vs_full_paired.csv", "model_backend_audit.csv", "model_calibration_audit.csv",
        "ensemble_metrics.csv", "ensemble_weights.csv", "model_disagreement.csv",
        "bottleneck_report.csv", "bottleneck_report.json", "resource_usage.csv",
        "base_candidate/model_card.json", "base_candidate/feature_list.csv", "base_candidate/calibration_policy.json",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in include:
            path = result_dir / relative
            if path.exists() and path.is_file():
                archive.write(path, arcname=relative.replace("\\", "/"))
    return destination


def write_result_brief(result_dir: Path, run_summary: dict[str, Any]) -> None:
    summary = _read_csv_or_empty(result_dir / "model_family_summary.csv")
    gates = _read_csv_or_empty(result_dir / "gating_decisions_safe.csv")
    best_line = "모델 결과 없음"
    if not summary.empty:
        best = summary.iloc[0]
        best_line = (
            f"{best.get('family')} / {best.get('feature_profile')} | raw ROC-AUC={best.get('raw_roc_auc_mean', math.nan):.4f}, "
            f"raw PR-AUC={best.get('raw_pr_auc_mean', math.nan):.4f}, Brier skill={best.get('brier_skill_mean', math.nan):.4f}"
        )
    kept = [] if gates.empty else gates.loc[gates["keep"].astype(bool), "experiment"].astype(str).tolist()
    text = f"""# CrashWatch Refine12H Result Brief

## 실행
- 프로필: {run_summary.get('profile')}
- 시간예산: {run_summary.get('hours')}시간
- 완료 작업: {run_summary.get('registry', {}).get('tasks_completed')}/{run_summary.get('registry', {}).get('tasks_total')}
- 데이터 signature: {run_summary.get('dataset_signature')}

## 최고 모델 후보
- {best_line}

## 핵심 수정
- Fold 2와 같은 음수 sigmoid slope를 금지했다.
- calibration 후 ROC/PR 또는 순위상관이 안전기준을 위반하면 `none`으로 자동 복귀한다.
- 기존 raw prediction으로 모든 이탈 결과를 재집계한다.
- 업종 gate는 raw와 calibrated 방향이 모두 양수일 때만 유지한다.
- 기존 고상관 중복 피처 제거 목록을 유지한다.
- full-reduced와 common-period를 동일 validation 날짜에서 비교한다.
- XGBoost, LightGBM, CatBoost, TCN, Transformer를 같은 outer split으로 평가한다.

## 유지 판정 gate
{json.dumps(kept, ensure_ascii=False)}

## 확인할 파일
1. `calibration_safety_audit.csv`
2. `safe_ablation_summary.csv`
3. `gating_decisions_safe.csv`
4. `model_family_summary.csv`
5. `common_vs_full_paired.csv`
6. `ensemble_metrics.csv`
7. `bottleneck_report.csv`
8. `base_candidate/model_card.json`
"""
    (result_dir / "RESULT_BRIEF.md").write_text(text, encoding="utf-8")
