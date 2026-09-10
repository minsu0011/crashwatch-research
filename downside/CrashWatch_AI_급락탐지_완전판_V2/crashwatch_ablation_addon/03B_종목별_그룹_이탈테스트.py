#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Development-only sentinel/group leave-one-group-out ablation test.

The input panel is never modified.  Sealed data is deliberately not referenced.
Run ``python 03B_종목별_그룹_이탈테스트.py --self-test`` before a full run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import py_compile
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SOURCE_ROOT = PROJECT_DIR / "crashwatch_ai_data"
DEVELOPMENT_PATH = SOURCE_ROOT / "development" / "training_dataset.parquet"
CATALOG_PATH = SOURCE_ROOT / "meta" / "feature_catalog.json"
OUT_DIR = BASE_DIR / "crashwatch_ai_data" / "ablation_sentinel"
CACHE_DIR = OUT_DIR / "prediction_cache"
TARGET = "label_abs_crash_20"
SEED = 20260722
FOLDS = 4
VALID_DAYS = 75
PURGE_DAYS = 20
MISSING_LIMIT = 0.995

SENTINELS = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("042700", "한미반도체"),
    ("005380", "현대차"), ("373220", "LG에너지솔루션"), ("005490", "POSCO홀딩스"),
    ("034020", "두산에너빌리티"), ("012450", "한화에어로스페이스"),
    ("329180", "HD현대중공업"), ("207940", "삼성바이오로직스"), ("068270", "셀트리온"),
    ("035420", "NAVER"), ("035720", "카카오"), ("105560", "KB금융"),
    ("017670", "SK텔레콤"), ("028260", "삼성물산"), ("247540", "에코프로비엠"),
    ("196170", "알테오젠"),
]
_CUDA_USABLE: bool | None = None
SENTINEL_METADATA: dict[str, dict[str, Any]] = {}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(features)).encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def finite_float32(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[:, list(features)].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def make_folds(dates: Sequence[pd.Timestamp]) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    minimum_train = max(500, int(math.ceil(len(dates) * 0.45)))
    latest_start = len(dates) - VALID_DAYS
    if latest_start <= minimum_train + PURGE_DAYS:
        raise RuntimeError(f"거래일 부족: dates={len(dates)}, minimum_train={minimum_train}")
    starts = np.linspace(minimum_train + PURGE_DAYS, latest_start, FOLDS, dtype=int)
    folds: list[dict[str, Any]] = []
    for raw_start in dict.fromkeys(int(x) for x in starts):
        train_stop = raw_start - PURGE_DAYS
        valid_stop = min(raw_start + VALID_DAYS, len(dates))
        if train_stop < minimum_train or valid_stop <= raw_start:
            continue
        folds.append({
            "fold": len(folds), "train_dates": dates[:train_stop], "valid_dates": dates[raw_start:valid_stop],
            "train_start": dates[0], "train_end": dates[train_stop - 1],
            "purge_start": dates[train_stop], "purge_end": dates[raw_start - 1],
            "valid_start": dates[raw_start], "valid_end": dates[valid_stop - 1],
            "train_trading_days": train_stop, "purge_trading_days": raw_start - train_stop,
            "validation_trading_days": valid_stop - raw_start,
        })
    if len(folds) != FOLDS:
        raise RuntimeError(f"요구한 {FOLDS}개 fold를 만들지 못했습니다: {len(folds)}")
    return folds


def metric_values(y: np.ndarray, p: np.ndarray, alert: np.ndarray, days: int) -> dict[str, float]:
    y = np.asarray(y, dtype="int8")
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    alert = np.asarray(alert, dtype=bool)
    rows = len(y)
    positives = int(y.sum())
    mixed = np.unique(y).size > 1
    alert_count = int(alert.sum())
    true_alerts = int(y[alert].sum()) if rows else 0
    return {
        "rows": rows, "positives": positives,
        "positive_rate": float(y.mean()) if rows else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if mixed else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if mixed else np.nan,
        "brier": float(brier_score_loss(y, p)) if rows else np.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if rows else np.nan,
        "mean_prediction_probability": float(p.mean()) if rows else np.nan,
        "alert_count": alert_count,
        "alert_precision": true_alerts / alert_count if alert_count else np.nan,
        "alert_recall": true_alerts / positives if positives else np.nan,
        "alerts_per_250_trading_days": alert_count / days * 250 if days else np.nan,
    }


def daily_alert_mask(predictions: pd.DataFrame) -> np.ndarray:
    """Every date alerts ceil(3%) of the complete validation universe, at least one."""
    flag = pd.Series(False, index=predictions.index)
    for _, block in predictions.groupby("date", sort=False):
        k = max(1, int(math.ceil(len(block) * 0.03)))
        chosen = block.sort_values(["prediction", "ticker"], ascending=[False, True], kind="stable").head(k).index
        flag.loc[chosen] = True
    return flag.to_numpy(dtype=bool)


@dataclass
class ModelResult:
    model: Any
    backend: str

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x))[:, 1]

    def importance(self, features: Sequence[str]) -> pd.DataFrame:
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({"feature": features, "importance": self.model.feature_importances_}).sort_values("importance", ascending=False)


def fit_model(x: pd.DataFrame, y: np.ndarray, n_estimators: int, disable_cuda: bool) -> ModelResult:
    """Required preference order: XGBoost CUDA, XGBoost CPU, sklearn HistGradientBoosting."""
    global _CUDA_USABLE
    params = {
        "n_estimators": n_estimators, "max_depth": 7, "learning_rate": 0.04,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.80,
        "reg_alpha": 0.10, "reg_lambda": 2.0, "max_bin": 256,
        "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
        "n_jobs": max(1, min(8, os.cpu_count() or 4)), "random_state": SEED,
    }
    if not disable_cuda and _CUDA_USABLE is not False:
        try:
            from xgboost import XGBClassifier
            cuda = XGBClassifier(**params, device="cuda")
            cuda.fit(x, y, verbose=False)
            _CUDA_USABLE = True
            return ModelResult(cuda, "xgboost_cuda")
        except Exception:
            _CUDA_USABLE = False
    try:
        from xgboost import XGBClassifier
        cpu = XGBClassifier(**params, device="cpu")
        cpu.fit(x, y, verbose=False)
        return ModelResult(cpu, "xgboost_cpu")
    except Exception:
        hist = HistGradientBoostingClassifier(
            learning_rate=0.055, max_iter=n_estimators, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=2.0, early_stopping=False, random_state=SEED,
        )
        hist.fit(x, y)
        return ModelResult(hist, "hist_gradient_boosting")


def load_inputs() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not DEVELOPMENT_PATH.exists() or not CATALOG_PATH.exists():
        raise FileNotFoundError("development panel 또는 feature_catalog.json이 없습니다.")
    frame = pd.read_parquet(DEVELOPMENT_PATH)
    if "sealed_do_not_train_or_tune" in frame and frame["sealed_do_not_train_or_tune"].fillna(0).ne(0).any():
        raise RuntimeError("development에 sealed_do_not_train_or_tune 행이 있어 학습을 중단합니다.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    frame = frame.dropna(subset=["date", TARGET]).sort_values(["date", "ticker"]).reset_index(drop=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return frame, {str(k): [str(c) for c in v] for k, v in catalog.items()}


def valid_features(frame: pd.DataFrame, catalog: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    valid: set[str] = set()
    for group, features in catalog.items():
        for col in features:
            exists = col in frame.columns
            s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan) if exists else pd.Series(dtype=float)
            missing = float(s.isna().mean()) if exists else 1.0
            unique = int(s.nunique(dropna=True)) if exists else 0
            keep = exists and missing <= MISSING_LIMIT and unique >= 2
            records.append({"group": group, "feature": col, "exists_in_development": exists,
                            "missing_rate": missing, "unique_values": unique, "keep": keep})
            if keep:
                valid.add(col)
    groups = {g: [f for f in values if f in valid] for g, values in catalog.items()}
    groups = {g: v for g, v in groups.items() if v}
    return sorted(valid), groups, pd.DataFrame(records).drop_duplicates(["group", "feature"])


def sentinel_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, requested_name in SENTINELS:
        part = frame.loc[frame.ticker.eq(ticker)]
        sector_col = "industry_name" if "industry_name" in part else "sector"
        sector = part[sector_col].iloc[-1] if len(part) and sector_col in part else "UNKNOWN"
        rows.append({"ticker": ticker, "requested_name": requested_name,
                     "name": part["name"].iloc[-1] if len(part) and "name" in part else requested_name,
                     "sector": sector, "in_development": bool(len(part)), "rows": len(part),
                     "start_date": part.date.min() if len(part) else pd.NaT,
                     "end_date": part.date.max() if len(part) else pd.NaT,
                     "positives": int(part[TARGET].sum()) if len(part) else 0,
                     "positive_rate": float(part[TARGET].mean()) if len(part) else np.nan})
    return pd.DataFrame(rows)


def cache_path(experiment: str, fold: int, features: Sequence[str]) -> Path:
    return CACHE_DIR / f"{safe_name(experiment)}__fold{fold:02d}__{feature_hash(features)[:16]}.parquet"


def cached_or_predict(
    frame: pd.DataFrame, features: Sequence[str], fold: dict[str, Any], experiment: str,
    overwrite_cache: bool, n_estimators: int, disable_cuda: bool, importance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    valid = frame.loc[frame.date.isin(fold["valid_dates"])].copy()
    path = cache_path(experiment, int(fold["fold"]), features)
    expected = valid[["date", "ticker", TARGET]].sort_values(["date", "ticker"]).reset_index(drop=True)
    if path.exists() and not overwrite_cache:
        cached = pd.read_parquet(path).sort_values(["date", "ticker"]).reset_index(drop=True)
        if len(cached) == len(expected) and cached[["date", "ticker", TARGET]].equals(expected):
            imp_path = path.with_name(path.stem + "__importance.parquet")
            imp = pd.read_parquet(imp_path) if importance and imp_path.exists() else pd.DataFrame()
            return cached, imp, str(cached.get("backend", pd.Series(["cached"])).iloc[0]), True
    train = frame.loc[frame.date.isin(fold["train_dates"])].copy()
    y_train = train[TARGET].astype("int8").to_numpy()
    if np.unique(y_train).size < 2:
        raise RuntimeError(f"fold={fold['fold']} 학습 라벨이 단일 클래스입니다.")
    started = time.time()
    model = fit_model(finite_float32(train, features), y_train, n_estimators, disable_cuda)
    out_cols = [c for c in ("date", "ticker", "name", "industry_name", TARGET) if c in valid]
    predicted = valid.loc[:, out_cols].copy()
    if "industry_name" in predicted:
        predicted = predicted.rename(columns={"industry_name": "sector"})
    else:
        predicted["sector"] = "UNKNOWN"
    predicted["prediction"] = model.predict(finite_float32(valid, features))
    predicted["backend"] = model.backend
    predicted["fit_seconds"] = time.time() - started
    tmp = path.with_suffix(".parquet.tmp")
    predicted.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    imp = model.importance(features) if importance else pd.DataFrame()
    if importance and not imp.empty:
        imp["fold"] = fold["fold"]
        imp["experiment"] = experiment
        imp_tmp = path.with_name(path.stem + "__importance.parquet.tmp")
        imp.to_parquet(imp_tmp, index=False)
        os.replace(imp_tmp, path.with_name(path.stem + "__importance.parquet"))
    return predicted, imp, model.backend, False


def scoped_metrics(predicted: pd.DataFrame, fold: dict[str, Any], experiment: str, feature_count: int, backend: str, cache_hit: bool) -> pd.DataFrame:
    data = predicted.copy()
    data["is_alert"] = daily_alert_mask(data)
    scopes: list[tuple[str, str, pd.DataFrame]] = [("all_validation", "ALL", data)]
    sentinel_set = {x[0] for x in SENTINELS}
    scopes.append(("sentinel_pooled", "SENTINEL_POOLED", data.loc[data.ticker.isin(sentinel_set)]))
    lookup = SENTINEL_METADATA or {
        r.ticker: {"name": r.name, "sector": r.sector}
        for r in sentinel_table(data).itertuples(index=False)
    }
    for ticker, requested_name in SENTINELS:
        scopes.append(("ticker", ticker, data.loc[data.ticker.eq(ticker)]))
    records = []
    valid_days = int(data.date.nunique())
    for scope_type, scope_id, block in scopes:
        y = block[TARGET].astype("int8").to_numpy() if len(block) else np.array([], dtype="int8")
        p = block.prediction.to_numpy() if len(block) else np.array([], dtype=float)
        a = block.is_alert.to_numpy() if len(block) else np.array([], dtype=bool)
        record = metric_values(y, p, a, valid_days)
        sent = lookup.get(scope_id, {})
        record.update({"experiment": experiment, "fold": fold["fold"], "scope_type": scope_type,
                       "scope": scope_id, "ticker": scope_id if scope_type == "ticker" else "",
                       "name": sent.get("name", "대표종목 pooled" if scope_type == "sentinel_pooled" else "전체"),
                       "sector": sent.get("sector", "ALL"), "feature_count": feature_count,
                       "backend": backend, "cache_hit": cache_hit, "valid_start": fold["valid_start"],
                       "valid_end": fold["valid_end"], "valid_trading_days": valid_days})
        records.append(record)
    return pd.DataFrame(records)


LOSS_COLUMNS = ["pr_auc", "roc_auc", "brier", "logloss", "alert_recall", "alert_precision"]


def loss_join(baseline: pd.DataFrame, ablated: pd.DataFrame, group: str) -> pd.DataFrame:
    keys = ["fold", "scope_type", "scope", "ticker", "name", "sector"]
    left = baseline[keys + LOSS_COLUMNS].copy()
    right = ablated[keys + LOSS_COLUMNS].copy()
    out = left.merge(right, on=keys, suffixes=("_baseline", "_ablated"), how="outer")
    out["group"] = group
    out["pr_auc_loss_when_removed"] = out.pr_auc_baseline - out.pr_auc_ablated
    out["roc_auc_loss_when_removed"] = out.roc_auc_baseline - out.roc_auc_ablated
    out["brier_increase_when_removed"] = out.brier_ablated - out.brier_baseline
    out["logloss_increase_when_removed"] = out.logloss_ablated - out.logloss_baseline
    out["alert_recall_loss_when_removed"] = out.alert_recall_baseline - out.alert_recall_ablated
    out["alert_precision_loss_when_removed"] = out.alert_precision_baseline - out.alert_precision_ablated
    return out


def group_summary(losses: pd.DataFrame, group_sizes: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ticker = losses.loc[losses.scope_type.eq("ticker")].copy()
    ticker_avg = ticker.groupby(["group", "ticker", "name", "sector"], dropna=False, as_index=False).mean(numeric_only=True)
    overall_rows = []
    for group, block in ticker_avg.groupby("group", sort=True):
        values = block.pr_auc_loss_when_removed.dropna()
        ix = values.idxmax() if len(values) else None
        sensitive = block.loc[ix] if ix is not None else pd.Series(dtype=object)
        overall_rows.append({
            "group": group, "removed_feature_count": group_sizes[group], "ticker_count": int(len(values)),
            "pr_auc_loss_mean": values.mean(), "pr_auc_loss_median": values.median(), "pr_auc_loss_std": values.std(ddof=0),
            "pr_auc_loss_p25": values.quantile(.25) if len(values) else np.nan,
            "pr_auc_loss_p75": values.quantile(.75) if len(values) else np.nan,
            "pr_auc_loss_iqr": values.quantile(.75) - values.quantile(.25) if len(values) else np.nan,
            "pr_auc_loss_max": values.max(), "pr_auc_loss_min": values.min(),
            "most_sensitive_ticker": sensitive.get("ticker", np.nan), "most_sensitive_name": sensitive.get("name", np.nan),
            "mean_roc_auc_loss_when_removed": block.roc_auc_loss_when_removed.mean(),
            "mean_brier_increase_when_removed": block.brier_increase_when_removed.mean(),
            "mean_logloss_increase_when_removed": block.logloss_increase_when_removed.mean(),
            "mean_alert_recall_loss_when_removed": block.alert_recall_loss_when_removed.mean(),
            "mean_alert_precision_loss_when_removed": block.alert_precision_loss_when_removed.mean(),
        })
    overall = pd.DataFrame(overall_rows).sort_values("pr_auc_loss_mean", ascending=False)
    matrix = ticker_avg.pivot(index=["ticker", "name", "sector"], columns="group", values="pr_auc_loss_when_removed").reset_index()
    return ticker_avg, overall, matrix


def run_feature_top_n(frame: pd.DataFrame, features: list[str], folds: list[dict[str, Any]], importance: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.feature_top_n <= 0 or importance.empty:
        return pd.DataFrame()
    ranked = importance.groupby("feature", as_index=False).importance.mean().sort_values("importance", ascending=False)
    ranked = ranked.loc[ranked.importance.gt(0)].head(args.feature_top_n)
    if ranked.empty:
        return pd.DataFrame()  # 중요도가 없으면 무작위 대체를 금지한다.
    fold = folds[-1]
    baseline_pred, _, backend, hit = cached_or_predict(frame, features, fold, "feature_baseline_last_fold", args.overwrite_cache, args.n_estimators, args.disable_cuda, False)
    baseline = scoped_metrics(baseline_pred, fold, "feature_baseline_last_fold", len(features), backend, hit)
    rows = []
    for feature in ranked.feature:
        pred, _, back, cache_hit = cached_or_predict(frame, [x for x in features if x != feature], fold, f"drop_feature__{feature}", args.overwrite_cache, args.n_estimators, args.disable_cuda, False)
        losses = loss_join(baseline, scoped_metrics(pred, fold, f"drop_feature__{feature}", len(features) - 1, back, cache_hit), feature)
        losses["last_fold_only"] = True
        rows.append(losses)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_self_tests() -> None:
    dates = pd.bdate_range("2020-01-01", periods=760)
    folds = make_folds(dates)
    assert all(f["purge_trading_days"] == 20 for f in folds)
    assert all(f["train_end"] < f["purge_start"] < f["valid_start"] for f in folds)
    panel = pd.DataFrame({"date": [dates[-1]] * 5, "ticker": ["000001", "000002", "000003", "000004", "000005"],
                          TARGET: [0, 1, 0, 0, 1], "prediction": [.1, .9, .4, .3, .2]})
    sentinel_rows_before = len(panel.loc[panel.ticker.eq("000002")])
    validation_without_sampling = panel.copy()
    assert len(validation_without_sampling.loc[validation_without_sampling.ticker.eq("000002")]) == sentinel_rows_before
    assert daily_alert_mask(panel).sum() == 1  # 3% of five is at least one.
    single = metric_values(np.array([0, 0]), np.array([.1, .2]), np.array([True, False]), 1)
    assert np.isnan(single["pr_auc"]) and np.isnan(single["roc_auc"]) and np.isfinite(single["brier"])
    base = pd.DataFrame({"fold": [0], "scope_type": ["ticker"], "scope": ["005930"], "ticker": ["005930"], "name": ["삼성전자"], "sector": ["KOSPI"], "pr_auc": [.8], "roc_auc": [.7], "brier": [.2], "logloss": [.4], "alert_recall": [.5], "alert_precision": [.4]})
    abl = base.copy()
    abl[["pr_auc", "roc_auc"]] = [.6, .5]
    abl[["brier", "logloss"]] = [.3, .5]
    abl[["alert_recall", "alert_precision"]] = [.3, .2]
    signed = loss_join(base, abl, "g").iloc[0]
    assert signed.pr_auc_loss_when_removed > 0 and signed.brier_increase_when_removed > 0
    test_cache = CACHE_DIR / "_self_test.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(test_cache, index=False)
    assert pd.read_parquet(test_cache).equals(panel)  # cache 재사용 시 동일 예측을 읽는다.
    test_cache.unlink(missing_ok=True)
    py_compile.compile(str(Path(__file__)), doraise=True)
    print("synthetic fixture tests: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--feature-top-n", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=int(os.getenv("CRASHWATCH_XGB_ESTIMATORS", "700")))
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    global SENTINEL_METADATA
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, catalog = load_inputs()
    sentinels = sentinel_table(frame)
    SENTINEL_METADATA = {
        row.ticker: {"name": row.name, "sector": row.sector}
        for row in sentinels.itertuples(index=False)
    }
    atomic_csv(sentinels, OUT_DIR / "sentinel_universe_checked.csv")
    features, groups, quality = valid_features(frame, catalog)
    atomic_csv(quality, OUT_DIR / "feature_quality.csv")
    if len(features) < 2 or not groups:
        raise RuntimeError("유효한 피처/그룹이 부족합니다.")
    folds = make_folds(frame.date.unique())
    atomic_json([{k: (list(v) if isinstance(v, pd.DatetimeIndex) else v) for k, v in fold.items() if k not in {"train_dates", "valid_dates"}} for fold in folds], OUT_DIR / "walk_forward_folds.json")
    baseline_rows, importance_rows = [], []
    for fold in folds:
        pred, imp, backend, hit = cached_or_predict(frame, features, fold, "baseline", args.overwrite_cache, args.n_estimators, args.disable_cuda, True)
        baseline_rows.append(scoped_metrics(pred, fold, "baseline", len(features), backend, hit))
        if not imp.empty:
            importance_rows.append(imp)
    baseline = pd.concat(baseline_rows, ignore_index=True)
    atomic_csv(baseline, OUT_DIR / "baseline_metrics_by_fold_scope.csv")
    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame(columns=["feature", "importance"])
    if not importance.empty:
        atomic_csv(importance.groupby("feature", as_index=False).importance.mean().sort_values("importance", ascending=False), OUT_DIR / "baseline_feature_importance.csv")
    else:
        atomic_csv(importance, OUT_DIR / "baseline_feature_importance.csv")
    all_metrics: list[pd.DataFrame] = []
    all_losses: list[pd.DataFrame] = []
    for group, removed in groups.items():
        kept = [f for f in features if f not in set(removed)]
        group_metrics = []
        for fold in folds:
            pred, _, backend, hit = cached_or_predict(frame, kept, fold, f"drop_group__{group}", args.overwrite_cache, args.n_estimators, args.disable_cuda, False)
            group_metrics.append(scoped_metrics(pred, fold, f"drop_group__{group}", len(kept), backend, hit))
        ablated = pd.concat(group_metrics, ignore_index=True)
        losses = loss_join(baseline, ablated, group)
        all_metrics.append(ablated)
        all_losses.append(losses)
        atomic_csv(pd.concat(all_metrics, ignore_index=True), OUT_DIR / "all_ablation_metrics_by_fold.csv")
        atomic_csv(pd.concat(all_losses, ignore_index=True), OUT_DIR / "group_ablation_by_ticker.csv")
    losses = pd.concat(all_losses, ignore_index=True) if all_losses else pd.DataFrame()
    _, overall, matrix = group_summary(losses, {g: len(v) for g, v in groups.items()})
    skipped_groups = [group for group in catalog if group not in groups]
    if skipped_groups:
        skipped = pd.DataFrame({
            "group": skipped_groups,
            "removed_feature_count": [0] * len(skipped_groups),
            "ticker_count": [0] * len(skipped_groups),
            "status": ["SKIPPED_NO_VALID_FEATURES"] * len(skipped_groups),
        })
        overall["status"] = "EVALUATED"
        overall = pd.concat([overall, skipped], ignore_index=True, sort=False).sort_values(
            ["status", "pr_auc_loss_mean"], ascending=[True, False], na_position="last"
        )
    else:
        overall["status"] = "EVALUATED"
    for group in catalog:
        if group not in matrix:
            matrix[group] = np.nan
    matrix = matrix[["ticker", "name", "sector", *catalog.keys()]]
    atomic_csv(overall, OUT_DIR / "group_ablation_overall.csv")
    atomic_csv(matrix, OUT_DIR / "ticker_group_sensitivity_matrix.csv")
    rec = overall.copy()
    rec["recommendation"] = np.where(
        rec.status.eq("SKIPPED_NO_VALID_FEATURES"), "NOT_TESTABLE_NO_VALID_FEATURES",
        np.where(rec.pr_auc_loss_mean > 0, "KEEP_GROUP_USEFUL", "REVIEW_OR_REMOVE_CANDIDATE"),
    )
    rec["interpretation"] = "양수 loss는 해당 그룹 제거 시 성능이 감소해 원래 모델에 유용함"
    atomic_csv(rec, OUT_DIR / "group_ablation_recommendations.csv")
    feature_result = run_feature_top_n(frame, features, folds, importance, args)
    if not feature_result.empty:
        atomic_csv(feature_result, OUT_DIR / "individual_feature_ablation_last_fold.csv")
    summary = {"target": TARGET, "input": str(DEVELOPMENT_PATH), "sealed_read": False,
               "rows": len(frame), "tickers": int(frame.ticker.nunique()), "features": len(features),
               "groups": len(groups), "folds": len(folds), "validation_days": VALID_DAYS,
               "purge_days": PURGE_DAYS, "minimum_train_days": max(500, int(math.ceil(frame.date.nunique() * .45))),
               "sentinels_requested": len(SENTINELS), "sentinels_present": int(sentinels.in_development.sum()),
               "n_estimators": args.n_estimators, "feature_top_n": args.feature_top_n,
               "cache_directory": str(CACHE_DIR), "outputs": [str(x.name) for x in OUT_DIR.glob("*.csv")]}
    atomic_json(summary, OUT_DIR / "run_summary.json")
    print(f"완료: groups={len(groups)}, baseline_rows={len(baseline)}, cache={CACHE_DIR}")


if __name__ == "__main__":
    main()
