#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""종목별 피처 그룹 민감도를 측정하는 panel-model ablation test.

핵심 원칙
1. 모델은 development 전체 종목 패널로 학습한다.
2. 이탈테스트 중 sealed/S00~S03은 절대 읽지 않는다.
3. 각 fold에서 전체 피처 기준모델과 그룹 제거 모델을 동일한 날짜/행으로 비교한다.
4. 대표 종목 18개에 대해 PR-AUC, Brier, Top-3% 알림 성능의 변화를 별도 기록한다.
5. 그룹을 제거했을 때 성능이 나빠진 양을 sensitivity로 정의한다.

실행 예:
    python 03B_종목별_그룹_이탈테스트.py
    python 03B_종목별_그룹_이탈테스트.py --groups price_trend,investor_flow,news_sentiment_attention
    python 03B_종목별_그룹_이탈테스트.py --feature-top-n 10
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
DEV_PATH = DATA_ROOT / "development" / "training_dataset.parquet"
CATALOG_PATH = DATA_ROOT / "meta" / "feature_catalog.json"
UNIVERSE_PATH = BASE_DIR / "주요종목_18선.csv"
OUT_DIR = DATA_ROOT / "ablation_sentinel"
CACHE_DIR = OUT_DIR / "prediction_cache"
LOG_DIR = DATA_ROOT / "logs"
TARGET = "label_abs_crash_20"

RANDOM_SEED = 20260721
DEFAULT_VALID_DAYS = 75
DEFAULT_PURGE_DAYS = 20
DEFAULT_CV_FOLDS = 4
DEFAULT_MAX_TRAIN_ROWS = 1_500_000
DEFAULT_MAX_VALID_ROWS = 400_000
MISSING_LIMIT = 0.995

for directory in (OUT_DIR, CACHE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "03B_종목별_그룹_이탈테스트.log", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("ticker_ablation")


@dataclass
class ModelBundle:
    model: Any
    backend: str

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x))[:, 1]
        return np.clip(np.asarray(self.model.predict(x), dtype=float), 0, 1)

    def feature_importance(self, features: Sequence[str]) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_, dtype=float)
        else:
            values = np.zeros(len(features), dtype=float)
        if len(values) != len(features):
            values = np.resize(values, len(features))
        return pd.DataFrame({"feature": list(features), "importance": values})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="", help="쉼표로 구분. 비우면 전체 그룹")
    parser.add_argument("--feature-top-n", type=int, default=0, help="0이면 개별 피처 이탈 생략")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--valid-days", type=int, default=DEFAULT_VALID_DAYS)
    parser.add_argument("--purge-days", type=int, default=DEFAULT_PURGE_DAYS)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--max-valid-rows", type=int, default=DEFAULT_MAX_VALID_ROWS)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(features)).encode("utf-8")).hexdigest()


def finite_float32(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, features].copy().replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def select_valid_features(
    df: pd.DataFrame, catalog: dict[str, list[str]]
) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    requested = sorted({feature for features in catalog.values() for feature in features if feature in df.columns})
    records = []
    valid = []
    for col in requested:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_rate = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))
        keep = missing_rate <= MISSING_LIMIT and nunique > 1
        records.append(
            {"feature": col, "missing_rate": missing_rate, "nunique": nunique, "keep": keep}
        )
        if keep:
            valid.append(col)
    valid_set = set(valid)
    clean_catalog = {
        group: [feature for feature in features if feature in valid_set]
        for group, features in catalog.items()
    }
    clean_catalog = {group: features for group, features in clean_catalog.items() if features}
    return valid, clean_catalog, pd.DataFrame(records)


def make_walk_forward_folds(
    dates: pd.DatetimeIndex, cv_folds: int, valid_days: int, purge_days: int
) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    min_train_days = max(500, int(len(dates) * 0.45))
    latest_start = len(dates) - valid_days
    if latest_start <= min_train_days + purge_days:
        raise RuntimeError("워크포워드 fold를 만들 거래일이 부족합니다.")
    starts = np.linspace(min_train_days, latest_start, cv_folds, dtype=int)
    folds: list[dict[str, Any]] = []
    for val_start in dict.fromkeys(int(x) for x in starts):
        train_end = val_start - purge_days
        val_end = min(len(dates), val_start + valid_days)
        if train_end < 260 or val_end <= val_start:
            continue
        folds.append(
            {
                "fold": len(folds),
                "train_dates": dates[:train_end],
                "valid_dates": dates[val_start:val_end],
                "train_start": dates[0],
                "train_end": dates[train_end - 1],
                "valid_start": dates[val_start],
                "valid_end": dates[val_end - 1],
            }
        )
    if len(folds) < 2:
        raise RuntimeError("유효 fold가 2개 미만입니다.")
    return folds


def sample_training_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    if len(df) <= max_rows:
        return df, np.ones(len(df), dtype="float32")
    positive = df[df[TARGET].eq(1)]
    negative = df[df[TARGET].eq(0)]
    negative_n = max(1, max_rows - len(positive))
    negative_sample = negative.sample(n=min(negative_n, len(negative)), random_state=RANDOM_SEED)
    sampled = pd.concat([positive, negative_sample]).sort_values(["date", "ticker"])
    weights = np.ones(len(sampled), dtype="float32")
    if len(negative_sample):
        weights[sampled[TARGET].to_numpy() == 0] = len(negative) / len(negative_sample)
    return sampled, weights


def cap_validation_rows(
    df: pd.DataFrame, max_rows: int, sentinel_tickers: set[str]
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    sentinel = df[df["ticker"].isin(sentinel_tickers)]
    other = df[~df.index.isin(sentinel.index)]
    remaining = max(0, max_rows - len(sentinel))
    if remaining == 0:
        return sentinel.sort_values(["date", "ticker"])
    per_day = max(1, int(remaining / max(1, other["date"].nunique())))
    sampled_blocks = []
    for _, block in other.groupby("date", sort=False):
        if len(block) > per_day:
            block = block.sample(n=per_day, random_state=RANDOM_SEED)
        sampled_blocks.append(block)
    sampled_other = pd.concat(sampled_blocks) if sampled_blocks else other.iloc[0:0]
    return pd.concat([sentinel, sampled_other]).sort_values(["date", "ticker"])


def fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    use_gpu: bool,
    threads: int,
) -> ModelBundle:
    try:
        from xgboost import XGBClassifier

        params: dict[str, Any] = {
            "n_estimators": 700,
            "max_depth": 7,
            "learning_rate": 0.04,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
            "max_bin": 256,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": threads,
            "random_state": RANDOM_SEED,
        }
        if use_gpu:
            params["device"] = "cuda"
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return ModelBundle(model=model, backend="xgboost_cuda" if use_gpu else "xgboost_cpu")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("XGBoost 실패, HistGradientBoosting 사용: %s", exc)
        model = HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=350,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return ModelBundle(model=model, backend="sklearn_hist")


def add_daily_top3_alert(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["risk_rank"] = out.groupby("date")["pred"].rank(
        ascending=False, method="first"
    )
    daily_count = out.groupby("date")["ticker"].transform("size")
    daily_k = np.ceil(daily_count * 0.03).clip(lower=1)
    out["risk_rank_pct"] = out["risk_rank"] / daily_count
    out["alert_top3"] = out["risk_rank"].le(daily_k).astype("int8")
    return out


def safe_metrics(block: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(block[TARGET], errors="coerce").to_numpy(dtype=float)
    p = np.clip(pd.to_numeric(block["pred"], errors="coerce").to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    rows = len(y)
    positives = float(y.sum())
    result: dict[str, float] = {
        "rows": float(rows),
        "positives": positives,
        "positive_rate": float(y.mean()) if rows else np.nan,
        "mean_pred": float(p.mean()) if rows else np.nan,
    }
    if rows and np.unique(y).size > 1:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    else:
        result.update({"pr_auc": np.nan, "roc_auc": np.nan})
    if rows:
        result["brier"] = float(brier_score_loss(y, p))
        result["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    else:
        result.update({"brier": np.nan, "logloss": np.nan})

    alert = pd.to_numeric(block.loc[mask, "alert_top3"], errors="coerce").fillna(0).to_numpy(dtype=int)
    alert_count = int(alert.sum())
    true_alert = float(y[alert == 1].sum()) if alert_count else 0.0
    result["alert_count"] = float(alert_count)
    result["alert_precision_top3_daily"] = true_alert / alert_count if alert_count else np.nan
    result["alert_recall_top3_daily"] = true_alert / positives if positives > 0 else np.nan
    trading_days = max(1, int(block["date"].nunique()))
    result["alerts_per_250d"] = alert_count / trading_days * 250
    return result


def metric_rows(
    pred_df: pd.DataFrame,
    experiment: str,
    fold: int,
    sentinel_meta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    overall = safe_metrics(pred_df)
    rows.append({"experiment": experiment, "fold": fold, "scope": "all_validation", "ticker": "ALL", "name": "전체검증", **overall})

    sentinel_tickers = set(sentinel_meta["ticker"])
    sentinel = pred_df[pred_df["ticker"].isin(sentinel_tickers)]
    if not sentinel.empty:
        rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "scope": "sentinel_pool",
                "ticker": "SENTINEL",
                "name": "대표종목합계",
                **safe_metrics(sentinel),
            }
        )

    meta_map = sentinel_meta.set_index("ticker").to_dict("index")
    for ticker, block in sentinel.groupby("ticker", sort=False):
        meta = meta_map.get(ticker, {})
        rows.append(
            {
                "experiment": experiment,
                "fold": fold,
                "scope": "ticker",
                "ticker": ticker,
                "name": meta.get("name", ticker),
                "market": meta.get("market", ""),
                "sector": meta.get("sector", ""),
                "priority": meta.get("priority", ""),
                **safe_metrics(block),
            }
        )
    return pd.DataFrame(rows)


def cache_path(experiment: str, fold: int, features: Sequence[str]) -> Path:
    safe = experiment.replace("::", "__").replace("/", "_")
    return CACHE_DIR / f"{safe}_fold{fold}_{feature_hash(features)[:12]}.parquet"


def evaluate_experiment(
    df: pd.DataFrame,
    features: list[str],
    folds: Sequence[dict[str, Any]],
    experiment: str,
    sentinel_meta: pd.DataFrame,
    args: argparse.Namespace,
    collect_importance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_parts = []
    importance_parts = []
    sentinel_tickers = set(sentinel_meta["ticker"])

    for fold in folds:
        path = cache_path(experiment, fold["fold"], features)
        if path.exists() and not args.overwrite_cache:
            pred_df = pd.read_parquet(path)
            LOGGER.info("캐시 사용: %s fold=%s", experiment, fold["fold"])
        else:
            train = df[df["date"].isin(fold["train_dates"])].copy()
            valid = df[df["date"].isin(fold["valid_dates"])].copy()
            train, weights = sample_training_rows(train, args.max_train_rows)
            valid = cap_validation_rows(valid, args.max_valid_rows, sentinel_tickers)
            if train.empty or valid.empty:
                continue
            y_train = train[TARGET].astype("int8").to_numpy()
            if np.unique(y_train).size < 2:
                LOGGER.warning("학습 라벨 단일값: %s fold=%s", experiment, fold["fold"])
                continue

            started = time.time()
            x_train = finite_float32(train, features)
            x_valid = finite_float32(valid, features)
            bundle = fit_model(x_train, y_train, weights, not args.cpu, args.threads)
            pred = bundle.predict_proba(x_valid)
            pred_df = valid[[c for c in ("date", "ticker", "name", "market", "industry_name", TARGET) if c in valid.columns]].copy()
            pred_df["pred"] = pred
            pred_df["experiment"] = experiment
            pred_df["fold"] = fold["fold"]
            pred_df["backend"] = bundle.backend
            pred_df = add_daily_top3_alert(pred_df)
            pred_df.to_parquet(path, index=False)
            LOGGER.info(
                "%s fold=%s 학습완료 rows=%d features=%d %.1fs",
                experiment, fold["fold"], len(train), len(features), time.time() - started,
            )
            if collect_importance:
                importance = bundle.feature_importance(features)
                importance["experiment"] = experiment
                importance["fold"] = fold["fold"]
                importance_parts.append(importance)
            del train, valid, x_train, x_valid, bundle
            gc.collect()

        metric_parts.append(metric_rows(pred_df, experiment, fold["fold"], sentinel_meta))

    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    importance = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    return metrics, importance


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("experiment", "scope", "ticker", "name", "market", "sector", "priority") if c in metrics.columns]
    numeric = [
        c for c in (
            "rows", "positives", "positive_rate", "mean_pred", "pr_auc", "roc_auc", "brier", "logloss",
            "alert_count", "alert_precision_top3_daily", "alert_recall_top3_daily", "alerts_per_250d",
        )
        if c in metrics.columns
    ]
    agg = metrics.groupby(keys, dropna=False, as_index=False)[numeric].mean()
    fold_count = metrics.groupby(keys, dropna=False).size().rename("fold_count").reset_index()
    return agg.merge(fold_count, on=keys, how="left")


def compare_with_baseline(aggregated: pd.DataFrame) -> pd.DataFrame:
    baseline = aggregated[aggregated["experiment"].eq("baseline")].copy()
    baseline = baseline.drop(columns=["experiment"]).rename(
        columns={
            c: f"baseline_{c}"
            for c in aggregated.select_dtypes(include=[np.number]).columns
            if c != "fold_count"
        }
    )
    keys = [c for c in ("scope", "ticker", "name", "market", "sector", "priority") if c in aggregated.columns]
    compared = aggregated[~aggregated["experiment"].eq("baseline")].merge(baseline, on=keys, how="left")
    compared["group"] = compared["experiment"].str.replace("drop_group::", "", regex=False)

    # 양수일수록 해당 그룹이 도움이 되었고, 제거 시 성능이 악화되었다는 뜻이다.
    compared["pr_auc_loss_when_removed"] = compared["baseline_pr_auc"] - compared["pr_auc"]
    compared["roc_auc_loss_when_removed"] = compared["baseline_roc_auc"] - compared["roc_auc"]
    compared["brier_increase_when_removed"] = compared["brier"] - compared["baseline_brier"]
    compared["logloss_increase_when_removed"] = compared["logloss"] - compared["baseline_logloss"]
    compared["alert_recall_loss_when_removed"] = (
        compared["baseline_alert_recall_top3_daily"] - compared["alert_recall_top3_daily"]
    )
    compared["alert_precision_loss_when_removed"] = (
        compared["baseline_alert_precision_top3_daily"] - compared["alert_precision_top3_daily"]
    )
    return compared


def build_group_summary(compared: pd.DataFrame) -> pd.DataFrame:
    ticker = compared[compared["scope"].eq("ticker")].copy()
    overall = compared[compared["scope"].isin(["all_validation", "sentinel_pool"])].copy()

    ticker_summary = ticker.groupby("group", as_index=False).agg(
        ticker_count=("ticker", "nunique"),
        median_pr_auc_loss=("pr_auc_loss_when_removed", "median"),
        mean_pr_auc_loss=("pr_auc_loss_when_removed", "mean"),
        max_pr_auc_loss=("pr_auc_loss_when_removed", "max"),
        min_pr_auc_loss=("pr_auc_loss_when_removed", "min"),
        std_pr_auc_loss=("pr_auc_loss_when_removed", "std"),
        median_brier_increase=("brier_increase_when_removed", "median"),
        median_alert_recall_loss=("alert_recall_loss_when_removed", "median"),
    )
    q1 = ticker.groupby("group")["pr_auc_loss_when_removed"].quantile(0.25).rename("q1_pr_auc_loss")
    q3 = ticker.groupby("group")["pr_auc_loss_when_removed"].quantile(0.75).rename("q3_pr_auc_loss")
    ticker_summary = ticker_summary.merge(q1, on="group").merge(q3, on="group")
    ticker_summary["iqr_pr_auc_loss"] = ticker_summary["q3_pr_auc_loss"] - ticker_summary["q1_pr_auc_loss"]

    valid_ticker = ticker.dropna(subset=["pr_auc_loss_when_removed"]).copy()
    if valid_ticker.empty:
        worst = pd.DataFrame(columns=[
            "group", "most_sensitive_ticker", "most_sensitive_name", "most_sensitive_pr_auc_loss"
        ])
        best = pd.DataFrame(columns=[
            "group", "least_sensitive_ticker", "least_sensitive_name", "least_sensitive_pr_auc_loss"
        ])
    else:
        worst = (
            valid_ticker.sort_values("pr_auc_loss_when_removed", ascending=False)
            .drop_duplicates("group")
            [["group", "ticker", "name", "pr_auc_loss_when_removed"]]
            .rename(
                columns={
                    "ticker": "most_sensitive_ticker",
                    "name": "most_sensitive_name",
                    "pr_auc_loss_when_removed": "most_sensitive_pr_auc_loss",
                }
            )
        )
        best = (
            valid_ticker.sort_values("pr_auc_loss_when_removed", ascending=True)
            .drop_duplicates("group")
            [["group", "ticker", "name", "pr_auc_loss_when_removed"]]
            .rename(
                columns={
                    "ticker": "least_sensitive_ticker",
                    "name": "least_sensitive_name",
                    "pr_auc_loss_when_removed": "least_sensitive_pr_auc_loss",
                }
            )
        )
    summary = ticker_summary.merge(worst, on="group", how="left").merge(best, on="group", how="left")

    overall_pivot = overall.pivot_table(
        index="group",
        columns="scope",
        values=["pr_auc_loss_when_removed", "brier_increase_when_removed", "alert_recall_loss_when_removed"],
        aggfunc="first",
    )
    overall_pivot.columns = [f"{metric}__{scope}" for metric, scope in overall_pivot.columns]
    overall_pivot = overall_pivot.reset_index()
    summary = summary.merge(overall_pivot, on="group", how="left")

    # 단순 자동 판정. 최종 제거는 반복 seed/통계 검증 후 결정해야 한다.
    summary["heterogeneous"] = summary["iqr_pr_auc_loss"].fillna(0).gt(0.01)
    summary["recommendation"] = np.select(
        [
            summary["median_pr_auc_loss"].gt(0.002),
            summary["median_pr_auc_loss"].lt(-0.002),
            summary["heterogeneous"],
        ],
        ["KEEP_USEFUL", "DROP_CANDIDATE", "KEEP_OR_INTERACT_HETEROGENEOUS"],
        default="NEUTRAL_RETEST",
    )
    return summary.sort_values(["median_pr_auc_loss", "max_pr_auc_loss"], ascending=False)


def run_feature_ablation(
    df: pd.DataFrame,
    all_features: list[str],
    folds: Sequence[dict[str, Any]],
    baseline_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    sentinel_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.feature_top_n <= 0 or importance.empty:
        return pd.DataFrame()
    ranked = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    candidates = [f for f in ranked["feature"].head(args.feature_top_n) if f in all_features]
    last_fold = [folds[-1]]

    # 비교 공정성을 위해 baseline도 마지막 fold만 다시 집계한다. 예측 캐시는 재사용한다.
    base_last, _ = evaluate_experiment(
        df, all_features, last_fold, "baseline", sentinel_meta, args, collect_importance=False
    )
    records = []
    base_agg = aggregate_metrics(base_last)
    for feature in candidates:
        metrics, _ = evaluate_experiment(
            df,
            [f for f in all_features if f != feature],
            last_fold,
            f"drop_feature::{feature}",
            sentinel_meta,
            args,
            collect_importance=False,
        )
        agg = aggregate_metrics(metrics)
        compared = agg.merge(
            base_agg.drop(columns=["experiment"]),
            on=[c for c in ("scope", "ticker", "name", "market", "sector", "priority") if c in agg.columns],
            how="left",
            suffixes=("", "_baseline"),
        )
        compared["feature"] = feature
        compared["pr_auc_loss_when_removed"] = compared["pr_auc_baseline"] - compared["pr_auc"]
        compared["brier_increase_when_removed"] = compared["brier"] - compared["brier_baseline"]
        records.append(compared)
        if records:
            atomic_csv(pd.concat(records, ignore_index=True), OUT_DIR / "feature_ablation_by_ticker.csv")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def main() -> None:
    args = parse_args()
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"개발 데이터 없음: {DEV_PATH}")
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"피처 카탈로그 없음: {CATALOG_PATH}")
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"대표 종목 목록 없음: {UNIVERSE_PATH}")

    LOGGER.info("development 데이터 로드")
    df = pd.read_parquet(DEV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.dropna(subset=["date", TARGET]).sort_values(["date", "ticker"])
    if "sealed_do_not_train_or_tune" in df.columns and df["sealed_do_not_train_or_tune"].fillna(0).ne(0).any():
        raise RuntimeError("development에 sealed 행이 섞여 있습니다.")

    sentinel_meta = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    sentinel_meta["ticker"] = sentinel_meta["ticker"].str.zfill(6)
    present = set(df["ticker"])
    sentinel_meta["present_in_development"] = sentinel_meta["ticker"].isin(present)
    atomic_csv(sentinel_meta, OUT_DIR / "sentinel_universe_checked.csv")
    sentinel_meta = sentinel_meta[sentinel_meta["present_in_development"]].copy()
    if len(sentinel_meta) < 10:
        raise RuntimeError(f"development에 존재하는 대표 종목이 너무 적습니다: {len(sentinel_meta)}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = {str(k): [str(v) for v in values] for k, values in catalog.items()}
    all_features, clean_catalog, quality = select_valid_features(df, catalog)
    atomic_csv(quality, OUT_DIR / "feature_quality.csv")

    requested_groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    if requested_groups:
        unknown = sorted(set(requested_groups) - set(clean_catalog))
        if unknown:
            raise ValueError(f"존재하지 않는 그룹: {unknown}; 사용 가능: {sorted(clean_catalog)}")
        groups = requested_groups
    else:
        groups = list(clean_catalog)

    folds = make_walk_forward_folds(
        pd.DatetimeIndex(df["date"].unique()), args.cv_folds, args.valid_days, args.purge_days
    )
    write_json(
        [{k: v for k, v in fold.items() if k not in {"train_dates", "valid_dates"}} for fold in folds],
        OUT_DIR / "walk_forward_folds.json",
    )

    LOGGER.info("기준모델: features=%d groups=%d sentinel=%d", len(all_features), len(groups), len(sentinel_meta))
    baseline_metrics, importance = evaluate_experiment(
        df, all_features, folds, "baseline", sentinel_meta, args, collect_importance=True
    )
    if baseline_metrics.empty:
        raise RuntimeError("기준모델 평가 결과가 없습니다.")
    atomic_csv(baseline_metrics, OUT_DIR / "baseline_metrics_by_fold_scope.csv")
    if not importance.empty:
        importance_avg = importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        atomic_csv(importance_avg, OUT_DIR / "baseline_feature_importance.csv")
        joblib.dump(importance_avg, OUT_DIR / "baseline_feature_importance.joblib", compress=3)

    all_metric_parts = [baseline_metrics]
    for index, group in enumerate(groups, start=1):
        removed = set(clean_catalog[group])
        remaining = [feature for feature in all_features if feature not in removed]
        LOGGER.info("그룹 이탈 %d/%d: %s (%d개 제거)", index, len(groups), group, len(removed))
        metrics, _ = evaluate_experiment(
            df,
            remaining,
            folds,
            f"drop_group::{group}",
            sentinel_meta,
            args,
            collect_importance=False,
        )
        all_metric_parts.append(metrics)
        atomic_csv(pd.concat(all_metric_parts, ignore_index=True), OUT_DIR / "all_ablation_metrics_by_fold.csv")

    all_metrics = pd.concat(all_metric_parts, ignore_index=True)
    aggregated = aggregate_metrics(all_metrics)
    atomic_csv(aggregated, OUT_DIR / "all_ablation_metrics_mean.csv")

    compared = compare_with_baseline(aggregated)
    by_ticker = compared[compared["scope"].eq("ticker")].copy()
    overall = compared[compared["scope"].isin(["all_validation", "sentinel_pool"])].copy()
    atomic_csv(by_ticker, OUT_DIR / "group_ablation_by_ticker.csv")
    atomic_csv(overall, OUT_DIR / "group_ablation_overall.csv")

    matrix = by_ticker.pivot_table(
        index=["ticker", "name", "sector"],
        columns="group",
        values="pr_auc_loss_when_removed",
        aggfunc="first",
    ).reset_index()
    atomic_csv(matrix, OUT_DIR / "ticker_group_sensitivity_matrix.csv")

    summary = build_group_summary(compared)
    atomic_csv(summary, OUT_DIR / "group_ablation_recommendations.csv")

    feature_result = run_feature_ablation(
        df, all_features, folds, baseline_metrics, importance, sentinel_meta, args
    )
    if not feature_result.empty:
        atomic_csv(feature_result, OUT_DIR / "feature_ablation_by_ticker.csv")

    run_summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "target": TARGET,
        "development_source": str(DEV_PATH),
        "sealed_used": False,
        "feature_count": len(all_features),
        "feature_hash": feature_hash(all_features),
        "tested_groups": groups,
        "sentinel_tickers": sentinel_meta[["ticker", "name", "sector"]].to_dict("records"),
        "interpretation": {
            "pr_auc_loss_when_removed": "양수면 해당 그룹 제거 시 성능 악화, 즉 유용",
            "brier_increase_when_removed": "양수면 해당 그룹 제거 시 확률보정 악화",
            "heterogeneous": "종목별 PR-AUC 손실 IQR이 큰 그룹",
        },
        "warning": "이 결과만으로 피처를 제거하지 말고 반복 seed, 기간 안정성, 봉인 전 동결 절차를 추가 확인할 것",
    }
    write_json(run_summary, OUT_DIR / "run_summary.json")
    print(f"완료: {OUT_DIR}")
    print(f"핵심 결과: {OUT_DIR / 'group_ablation_recommendations.csv'}")
    print(f"종목×그룹 행렬: {OUT_DIR / 'ticker_group_sensitivity_matrix.csv'}")


if __name__ == "__main__":
    main()
