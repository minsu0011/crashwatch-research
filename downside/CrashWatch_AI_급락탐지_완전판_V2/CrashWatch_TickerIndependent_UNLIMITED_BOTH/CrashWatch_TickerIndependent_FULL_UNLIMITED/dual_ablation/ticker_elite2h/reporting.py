from __future__ import annotations

import json
import math
import os
import sqlite3
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..io_utils import atomic_csv, atomic_json


def _read_jsons(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["source_path"] = str(path)
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={"ticker": str})
            frame["source_path"] = str(path)
            frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "date": ["date", "trade_date", "validation_date"],
        "ticker": ["ticker", "code", "symbol"],
        "target": ["target", "y_true", "label"],
        "raw_prediction": ["raw_prediction", "prediction", "probability", "score", "pred"],
        "seed": ["seed"],
        "fold": ["fold", "fold_id"],
    }
    rename: dict[str, str] = {}
    lower = {str(column).lower(): column for column in frame.columns}
    for canonical, names in aliases.items():
        for name in names:
            if name in lower:
                rename[lower[name]] = canonical
                break
    frame = frame.rename(columns=rename)
    required = {"date", "ticker", "target", "raw_prediction"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    output = frame[[column for column in ["date", "ticker", "target", "raw_prediction", "seed", "fold"] if column in frame.columns]].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["ticker"] = output["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    output["target"] = pd.to_numeric(output["target"], errors="coerce")
    output["raw_prediction"] = pd.to_numeric(output["raw_prediction"], errors="coerce")
    output = output.dropna(subset=["date", "ticker", "target", "raw_prediction"])
    return output


def _find_pooled_predictions(data_root: Path | None) -> tuple[str, pd.DataFrame]:
    if data_root is None:
        return "", pd.DataFrame()
    candidates = [
        ("lightgbm", data_root / "refine12h" / "predictions" / "outer" / "lightgbm" / "full_reduced"),
        ("xgboost", data_root / "refine12h" / "predictions" / "outer" / "xgboost" / "full_reduced"),
        ("focused_nested", data_root / "ablation_finance_nested_focus" / "prediction_cache"),
        ("base12h", data_root / "ablation_base12h" / "prediction_cache"),
    ]
    for family, root in candidates:
        if not root.exists():
            continue
        frames: list[pd.DataFrame] = []
        for path in list(root.rglob("*.parquet")) + list(root.rglob("*.csv")):
            try:
                frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
                frame = _normalize_prediction_frame(frame)
                if frame.empty:
                    continue
                if "seed" in frame.columns:
                    seed_values = pd.to_numeric(frame["seed"], errors="coerce")
                    if seed_values.notna().any():
                        preferred = 17 if (seed_values == 17).any() else int(seed_values.dropna().iloc[0])
                        frame = frame.loc[seed_values.eq(preferred)]
                frames.append(frame)
            except Exception:
                continue
        if frames:
            pooled = pd.concat(frames, ignore_index=True, sort=False)
            dedup = [column for column in ["date", "ticker", "fold"] if column in pooled.columns]
            pooled = pooled.drop_duplicates(dedup or ["date", "ticker"], keep="last")
            return family, pooled
    return "", pd.DataFrame()


def exact_pooled_comparison(result_dir: Path, data_root: Path | None) -> pd.DataFrame:
    independent_frames: list[pd.DataFrame] = []
    for ticker_dir in sorted((result_dir / "ticker_models").glob("*")):
        if not ticker_dir.is_dir():
            continue
        outer = ticker_dir / "outer_predictions.parquet"
        prediction_paths = [outer] if outer.exists() else sorted(ticker_dir.glob("fold_*_predictions.parquet"))
        for path in prediction_paths:
            try:
                frame = _normalize_prediction_frame(pd.read_parquet(path))
                if not frame.empty:
                    independent_frames.append(frame)
            except Exception:
                continue
    if not independent_frames:
        return pd.DataFrame()
    independent = pd.concat(independent_frames, ignore_index=True, sort=False)
    family, pooled = _find_pooled_predictions(data_root)
    if pooled.empty:
        return pd.DataFrame()
    merged = independent.merge(
        pooled[["date", "ticker", "target", "raw_prediction"]].rename(columns={"raw_prediction": "pooled_prediction"}),
        on=["date", "ticker", "target"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ticker, block in merged.groupby("ticker"):
        if len(block) < 40 or block["target"].nunique() < 2:
            continue
        y = block["target"].to_numpy(dtype=np.int8)
        independent_p = block["raw_prediction"].to_numpy(dtype=float)
        pooled_p = block["pooled_prediction"].to_numpy(dtype=float)
        prior = float(y.mean())
        independent_pr = float(average_precision_score(y, independent_p))
        pooled_pr = float(average_precision_score(y, pooled_p))
        independent_roc = float(roc_auc_score(y, independent_p))
        pooled_roc = float(roc_auc_score(y, pooled_p))
        rows.append({
            "ticker": ticker,
            "pooled_family": family,
            "paired_rows": int(len(block)),
            "paired_date_min": str(block["date"].min().date()),
            "paired_date_max": str(block["date"].max().date()),
            "paired_positive_rate": prior,
            "independent_raw_pr_auc": independent_pr,
            "pooled_raw_pr_auc": pooled_pr,
            "independent_minus_pooled_pr_auc": independent_pr - pooled_pr,
            "independent_raw_pr_lift": independent_pr / prior if prior > 0 else math.nan,
            "pooled_raw_pr_lift": pooled_pr / prior if prior > 0 else math.nan,
            "independent_raw_roc_auc": independent_roc,
            "pooled_raw_roc_auc": pooled_roc,
            "independent_minus_pooled_roc_auc": independent_roc - pooled_roc,
        })
    compare = pd.DataFrame(rows)
    if not compare.empty:
        atomic_csv(compare.sort_values("independent_minus_pooled_pr_auc", ascending=False), result_dir / "independent_vs_pooled_exact.csv")
        atomic_parquet_safe(merged, result_dir / "independent_vs_pooled_exact_rows.parquet")
    return compare


def atomic_parquet_safe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _feature_group(feature: str) -> str:
    tokens = str(feature).split("_")
    if len(tokens) >= 2:
        return "_".join(tokens[:2])
    return str(feature)


def build_fingerprints(result_dir: Path, summaries: pd.DataFrame, correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    importance_frames = _read_csvs(sorted((result_dir / "ticker_models").glob("*/feature_importance_by_fold.csv")))
    recipes = _read_csvs(sorted((result_dir / "ticker_models").glob("*/recipe_history.csv")))
    group_importance: dict[str, dict[str, float]] = defaultdict(dict)
    if not importance_frames.empty:
        importance_frames["feature_group"] = importance_frames["feature"].astype(str).map(_feature_group)
        grouped = importance_frames.groupby(["ticker", "feature_group"])["importance"].mean().reset_index()
        totals = grouped.groupby("ticker")["importance"].transform("sum").replace(0, 1.0)
        grouped["share"] = grouped["importance"] / totals
        for ticker, block in grouped.groupby("ticker"):
            group_importance[str(ticker)] = dict(zip(block["feature_group"], block["share"]))
    recipe_counts: dict[str, dict[str, float]] = defaultdict(dict)
    if not recipes.empty:
        for ticker, block in recipes.groupby("ticker"):
            total = max(1, len(block))
            values: dict[str, float] = {}
            for family, count in block["family"].value_counts().items():
                values[f"family::{family}"] = float(count / total)
            for policy, count in block["feature_policy"].value_counts().items():
                values[f"policy::{policy}"] = float(count / total)
            recipe_counts[str(ticker)] = values

    all_groups = sorted({group for mapping in group_importance.values() for group in mapping})
    all_recipes = sorted({name for mapping in recipe_counts.values() for name in mapping})
    correlation_map = correlations.set_index("ticker").to_dict("index") if not correlations.empty and "ticker" in correlations.columns else {}
    for _, summary in summaries.iterrows():
        ticker = str(summary["ticker"]).zfill(6)
        row: dict[str, Any] = {
            "ticker": ticker,
            "bucket": summary.get("bucket", ""),
            "raw_pr_auc": summary.get("raw_pr_auc", np.nan),
            "raw_pr_auc_std": summary.get("raw_pr_auc_std", np.nan),
            "raw_pr_lift": summary.get("raw_pr_lift", np.nan),
            "raw_roc_auc": summary.get("raw_roc_auc", np.nan),
            "balanced_accuracy": summary.get("balanced_accuracy", np.nan),
            "brier_skill": summary.get("brier_skill", np.nan),
            "fold_positive_rate_std": summary.get("fold_positive_rate_std", np.nan),
        }
        corr = correlation_map.get(ticker, {})
        for name in ["top40_mean_psi", "top40_mean_sign_agreement", "largest_cluster_ratio", "high_drift_feature_ratio"]:
            row[name] = corr.get(name, np.nan)
        for group in all_groups:
            row[f"importance::{group}"] = group_importance.get(ticker, {}).get(group, 0.0)
        for recipe in all_recipes:
            row[recipe] = recipe_counts.get(ticker, {}).get(recipe, 0.0)
        rows.append(row)
    fingerprints = pd.DataFrame(rows)
    if fingerprints.empty:
        return fingerprints, pd.DataFrame()
    numeric_columns = [column for column in fingerprints.columns if column not in {"ticker", "bucket"}]
    matrix = fingerprints[numeric_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix)
    components = min(8, max(2, min(len(fingerprints) - 1, scaled.shape[1])))
    if len(fingerprints) >= 3 and components >= 2:
        embedded = PCA(n_components=components, random_state=17).fit_transform(scaled)
        clusters = min(8, max(2, int(round(math.sqrt(len(fingerprints))))))
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(embedded)
        fingerprints["preliminary_cluster"] = labels.astype(int)
        for index in range(min(components, 4)):
            fingerprints[f"fingerprint_pc{index + 1}"] = embedded[:, index]
        normalized = embedded / np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)
        similarity_matrix = normalized @ normalized.T
        similarity = pd.DataFrame(similarity_matrix, index=fingerprints["ticker"], columns=fingerprints["ticker"])
    else:
        fingerprints["preliminary_cluster"] = 0
        similarity = pd.DataFrame()
    atomic_csv(fingerprints, result_dir / "ticker_model_fingerprints.csv")
    if not similarity.empty:
        atomic_csv(similarity.reset_index().rename(columns={"index": "ticker"}), result_dir / "ticker_model_similarity.csv")
    return fingerprints, similarity


def aggregate_results(result_dir: Path, plan: dict[str, Any], registry_status: dict[str, Any], data_root: Path | None = None) -> dict[str, Any]:
    readiness = pd.read_csv(result_dir / "ticker_data_readiness.csv", dtype={"ticker": str}) if (result_dir / "ticker_data_readiness.csv").exists() else pd.DataFrame()
    summaries = _read_jsons(sorted((result_dir / "ticker_models").glob("*/task_summary.json")))
    fold_metric_jsons = _read_jsons(sorted((result_dir / "ticker_models").glob("*/fold_*_metrics.json")))
    fold_metrics = fold_metric_jsons
    if fold_metrics.empty:
        fold_metrics = _read_csvs(sorted((result_dir / "ticker_models").glob("*/fold_metrics.csv")))
    correlations = _read_jsons(sorted((result_dir / "correlation_maps").glob("*/correlation_summary.json")))
    atomic_csv(summaries, result_dir / "ticker_elite_summary.csv")
    atomic_csv(fold_metrics, result_dir / "ticker_elite_fold_metrics.csv")
    atomic_csv(correlations, result_dir / "ticker_correlation_summary.csv")

    paired = exact_pooled_comparison(result_dir, data_root)
    completed_for_fingerprint = summaries.loc[summaries["status"].eq("completed")].copy() if (not summaries.empty and "status" in summaries.columns) else pd.DataFrame()
    fingerprints, _ = build_fingerprints(result_dir, completed_for_fingerprint, correlations)

    base = readiness.copy()
    if not summaries.empty:
        useful = [
            "ticker", "status", "fold_count", "raw_pr_auc", "raw_pr_auc_std", "raw_pr_auc_min",
            "raw_pr_lift", "raw_roc_auc", "balanced_accuracy", "brier_skill", "logloss_skill",
            "top_10pct_precision", "fold_positive_rate_std", "most_used_family", "most_used_policy", "elapsed_seconds",
            "final_model_status", "final_model_dir", "final_model_calibration_brier_skill",
            "final_model_calibration_logloss_skill", "final_selected_family",
            "final_selected_config_name", "final_selected_feature_policy",
            "final_selected_recipe_score",
        ]
        base = base.merge(summaries[[column for column in useful if column in summaries.columns]], on="ticker", how="left", suffixes=("", "_model"))
    if not correlations.empty:
        corr_columns = [
            "ticker", "top40_mean_psi", "top40_mean_sign_agreement", "largest_cluster_ratio",
            "high_drift_feature_ratio", "top20_mean_abs_target_corr",
        ]
        base = base.merge(correlations[[column for column in corr_columns if column in correlations.columns]], on="ticker", how="left")
    if not paired.empty:
        base = base.merge(paired, on="ticker", how="left")

    if not base.empty:
        base["fold_instability_score"] = pd.to_numeric(base.get("raw_pr_auc_std"), errors="coerce").fillna(1.0)
        base["worst_fold_score"] = 1.0 - pd.to_numeric(base.get("raw_pr_auc_min"), errors="coerce").fillna(0.0)
        base["calibration_score"] = -pd.to_numeric(base.get("brier_skill"), errors="coerce").fillna(-2.0)
        base["label_drift_score"] = pd.to_numeric(base.get("fold_positive_rate_std"), errors="coerce").fillna(1.0)
        base["feature_drift_score"] = pd.to_numeric(base.get("top40_mean_psi"), errors="coerce").fillna(0.0)
        base["model_weakness_score"] = 1.0 - pd.to_numeric(base.get("raw_roc_auc"), errors="coerce").fillna(0.5)
        base["paired_comparison_score"] = -pd.to_numeric(base.get("independent_minus_pooled_pr_auc"), errors="coerce").fillna(0.0)
        # Pooled comparison is diagnostic only and never gates storage or primary bottleneck ranking.
        score_columns = [
            "fold_instability_score", "worst_fold_score", "calibration_score", "label_drift_score",
            "feature_drift_score", "model_weakness_score",
        ]
        normalized = pd.DataFrame(index=base.index)
        for column in score_columns:
            values = pd.to_numeric(base[column], errors="coerce")
            lo, hi = values.quantile(0.05), values.quantile(0.95)
            normalized[column] = ((values - lo) / max(float(hi - lo), 1e-9)).clip(0, 1).fillna(0.5)
        base["overall_bottleneck_score"] = normalized.mean(axis=1)
        base["primary_bottleneck"] = normalized.idxmax(axis=1).str.replace("_score", "", regex=False)
        base = base.sort_values(["ready", "overall_bottleneck_score"], ascending=[False, False])
    atomic_csv(base, result_dir / "ticker_elite_bottleneck_map.csv")

    diagnostic_rows: list[dict[str, Any]] = []
    if not base.empty:
        metrics = ["raw_pr_auc", "raw_roc_auc", "raw_pr_lift", "balanced_accuracy"]
        diagnostics = ["positives", "fold_count", "raw_pr_auc_std", "fold_positive_rate_std", "top40_mean_psi", "top40_mean_sign_agreement", "largest_cluster_ratio"]
        for diagnostic in diagnostics:
            if diagnostic not in base.columns:
                continue
            for metric in metrics:
                if metric not in base.columns:
                    continue
                block = base[[diagnostic, metric]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(block) < 8:
                    continue
                result = spearmanr(block[diagnostic], block[metric], nan_policy="omit")
                diagnostic_rows.append({
                    "diagnostic": diagnostic,
                    "performance_metric": metric,
                    "spearman_correlation": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "ticker_count": int(len(block)),
                })
    diagnostics = pd.DataFrame(diagnostic_rows)
    atomic_csv(diagnostics, result_dir / "performance_bottleneck_correlations.csv")

    completed = summaries.loc[summaries.get("status", pd.Series(dtype=str)).eq("completed")] if not summaries.empty else pd.DataFrame()
    summary = {
        "schema_version": plan.get("schema_version"),
        "execution_profile": plan.get("execution_profile"),
        "backend_mode": plan.get("backend_mode"),
        "backend_policy": "all_cpu" if plan.get("execution_profile") == "cpu4" else "lightgbm_cpu_xgboost_catboost_cuda",
        "pooled_comparison_role": "diagnostic_only",
        "all_feasible_independent_models_are_saved": True,
        "tasks": registry_status,
        "ticker_count": int(len(readiness)),
        "ready_tickers": int(readiness["ready"].astype(str).str.lower().isin(["true", "1"]).sum()) if not readiness.empty else 0,
        "completed_ticker_models": int(len(completed)),
        "mean_raw_pr_auc": float(pd.to_numeric(completed.get("raw_pr_auc"), errors="coerce").mean()) if not completed.empty else None,
        "mean_raw_roc_auc": float(pd.to_numeric(completed.get("raw_roc_auc"), errors="coerce").mean()) if not completed.empty else None,
        "mean_raw_pr_lift": float(pd.to_numeric(completed.get("raw_pr_lift"), errors="coerce").mean()) if not completed.empty else None,
        "mean_fold_pr_auc_std": float(pd.to_numeric(completed.get("raw_pr_auc_std"), errors="coerce").mean()) if not completed.empty else None,
        "paired_pooled_tickers": int(len(paired)),
        "independent_paired_pr_wins": int((paired["independent_minus_pooled_pr_auc"] > 0).sum()) if not paired.empty else 0,
        "independent_paired_pr_and_roc_wins": int(((paired["independent_minus_pooled_pr_auc"] > 0) & (paired["independent_minus_pooled_roc_auc"] > 0)).sum()) if not paired.empty else 0,
        "fingerprint_tickers": int(len(fingerprints)),
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    atomic_json(summary, result_dir / "run_summary.json")
    write_result_brief(result_dir, summary, base, paired)
    return summary


def write_result_brief(result_dir: Path, summary: dict[str, Any], bottlenecks: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# CrashWatch 종목별 독립 특화 모델 결과 요약",
        "",
        f"- 실행 프로필: {summary.get('execution_profile')}",
        f"- backend 정책: {summary.get('backend_policy')}",
        f"- 완료 종목 모델: {summary.get('completed_ticker_models', 0)}",
        f"- 평균 raw PR-AUC: {summary.get('mean_raw_pr_auc')}",
        f"- 평균 raw ROC-AUC: {summary.get('mean_raw_roc_auc')}",
        f"- 평균 PR-AUC lift: {summary.get('mean_raw_pr_lift')}",
        f"- 평균 fold PR-AUC 표준편차: {summary.get('mean_fold_pr_auc_std')}",
        f"- 동일 날짜 pooled 비교 종목: {summary.get('paired_pooled_tickers', 0)}",
        f"- 동일 날짜 비교에서 독립 모델 PR 우세: {summary.get('independent_paired_pr_wins', 0)}",
        f"- 동일 날짜 비교에서 PR·ROC 모두 우세: {summary.get('independent_paired_pr_and_roc_wins', 0)}",
        "",
        "## 해석 주의",
        "",
        "- 모델은 종목별 데이터만 사용하며 다른 종목 행을 학습에 사용하지 않는다.",
        "- 커버리지는 목표가 아니며 최소 fold와 양성 사건 기준을 충족한 종목만 모델링한다.",
        "- pooled 비교는 동일 ticker/date/target 행이 정확히 겹치는 경우만 계산한다.",
        "- pooled 결과는 진단용이며 독립 모델 저장 여부나 recipe 선택에 사용하지 않는다.",
        "- 학습 가능한 모든 종목은 각각 독립된 recipe와 calibration으로 저장한다.",
        "- preliminary_cluster는 추후 클러스터링을 위한 지문 기반 탐색값이며 확정 군집이 아니다.",
    ]
    if not bottlenecks.empty:
        lines.extend(["", "## 주요 병목 종목", ""])
        for _, row in bottlenecks.head(12).iterrows():
            lines.append(f"- {row.get('ticker')}: {row.get('primary_bottleneck')} / score={row.get('overall_bottleneck_score')}")
    if not paired.empty:
        lines.extend(["", "## 동일 날짜 pooled 비교 상위", ""])
        for _, row in paired.sort_values("independent_minus_pooled_pr_auc", ascending=False).head(10).iterrows():
            lines.append(f"- {row['ticker']}: PR-AUC delta={row['independent_minus_pooled_pr_auc']:.6f}, rows={int(row['paired_rows'])}")
    (result_dir / "RESULT_BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def desktop_path() -> Path:
    candidates = [Path.home() / "Desktop", Path.home() / "바탕 화면", Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"]
    for path in candidates:
        if path.exists():
            return path
    return Path.home()


def pack_results(result_dir: Path, max_mb: int = 28) -> Path:
    registry_path = result_dir / "task_registry.sqlite"
    if registry_path.exists():
        try:
            with sqlite3.connect(registry_path) as connection:
                connection.execute("PRAGMA wal_checkpoint(FULL)")
        except Exception:
            pass
    mode = "CPU4" if "cpu4" in result_dir.name.lower() else "FULL"
    destination = desktop_path() / f"CrashWatch_TickerIndependent_{mode}_RESULTS_{pd.Timestamp.now():%Y%m%d_%H%M%S}.zip"
    core_names = [
        "RESULT_BRIEF.md", "run_summary.json", "ticker_data_readiness.csv", "ticker_elite_summary.csv",
        "ticker_elite_fold_metrics.csv", "ticker_elite_bottleneck_map.csv", "ticker_correlation_summary.csv",
        "independent_vs_pooled_exact.csv", "performance_bottleneck_correlations.csv",
        "ticker_model_fingerprints.csv", "ticker_model_similarity.csv", "runtime_worker_plan.json",
        "resource_usage.csv", "task_registry.sqlite",
    ]
    files: list[tuple[Path, str]] = []
    for name in core_names:
        path = result_dir / name
        if path.exists():
            files.append((path, name))
    for path in sorted((result_dir / "ticker_models").glob("*/task_summary.json")):
        files.append((path, str(path.relative_to(result_dir))))
    for path in sorted((result_dir / "ticker_models").glob("*/recipe_history.csv")):
        files.append((path, str(path.relative_to(result_dir))))
    for pattern in [
        "*/development_model/model_card.json",
        "*/development_model/calibration_policy.json",
        "*/development_model/recipe_selection_frequency.csv",
        "*/development_model/member_*_features.csv",
    ]:
        for path in sorted((result_dir / "ticker_models").glob(pattern)):
            files.append((path, str(path.relative_to(result_dir))))
    for path in sorted((result_dir / "ticker_models").glob("*/fold_*_metrics.json")):
        files.append((path, str(path.relative_to(result_dir))))
    for path in sorted((result_dir / "correlation_maps").glob("*/correlation_summary.json")):
        files.append((path, str(path.relative_to(result_dir))))
    for path in sorted((result_dir / "correlation_maps").glob("*/cluster_summary.csv")):
        files.append((path, str(path.relative_to(result_dir))))
    max_bytes = int(max_mb) * 1024 * 1024
    used = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path, arcname in files:
            size = path.stat().st_size
            if used + size > max_bytes and used > 0:
                continue
            archive.write(path, arcname)
            used += size
    return destination
