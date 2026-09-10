#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_이탈테스트_봉인인증.py
===================
개발 데이터만 사용해 워크포워드 기준모델과 피처 이탈테스트를 수행한다.
피처/그룹 선택이 끝나면 목록을 동결하고, 마지막에만 S00~S03 봉인구간을
한 번씩 평가한다. 봉인 결과는 피처 선택·튜닝에 절대 사용하지 않는다.

설치:
    pip install pandas numpy pyarrow scikit-learn xgboost joblib

실행:
    python 03_이탈테스트_봉인인증.py

권장 하드웨어 설정:
- RTX GPU 사용: USE_GPU = True
- Ryzen 5900X: N_THREADS = 22

출력:
    crashwatch_ai_data/ablation/baseline_cv.csv
    crashwatch_ai_data/ablation/group_ablation.csv
    crashwatch_ai_data/ablation/feature_ablation.csv
    crashwatch_ai_data/ablation/frozen_feature_set.json
    crashwatch_ai_data/ablation/seal_certification.csv
    crashwatch_ai_data/ablation/seal_predictions/*.parquet
    crashwatch_ai_data/ablation/final_model.joblib

중요:
- 이탈테스트는 development/training_dataset.parquet에서만 수행한다.
- sealed/S00~S03은 최종 동결 모델 인증에서만 읽는다.
- 스크립트는 봉인 점수를 이용해 피처를 다시 고르지 않는다.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from 공통_도구 import 결과압축_생성
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# 설정

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
DEVELOPMENT_PATH = DATA_ROOT / "development" / "training_dataset.parquet"
SEALED_DIR = DATA_ROOT / "sealed"
META_DIR = DATA_ROOT / "meta"
OUT_DIR = DATA_ROOT / "ablation"
PRED_DIR = OUT_DIR / "seal_predictions"
LOG_DIR = DATA_ROOT / "logs"
for d in (OUT_DIR, PRED_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "label_abs_crash_20"
USE_GPU = True
N_THREADS = 22
RANDOM_SEED = 20260721
CV_FOLDS = 4
VALID_DAYS = 75
PURGE_DAYS = 20
MAX_TRAIN_ROWS = 1_500_000
MAX_VALID_ROWS = 400_000
MISSING_LIMIT = 0.995

# 그룹 단위 leave-one-group-out 이탈테스트
RUN_GROUP_ABLATION = True
# 기준모델 중요도 상위 N개에 대해 마지막 개발 fold에서 개별 피처 이탈테스트
RUN_FEATURE_LEVEL_ABLATION = True
FEATURE_ABLATION_TOP_N = 30

# 그룹 제거 후 PR-AUC가 이만큼 이상 좋아지고 Top3% recall/precision이 악화되지 않으면 제거 후보
DROP_GROUP_MIN_PR_GAIN = 0.0010
DROP_GROUP_MAX_RECALL_LOSS = 0.0050
DROP_GROUP_MAX_PRECISION_LOSS = 0.0050

XGB_PARAMS = {
    "n_estimators": 700,
    "max_depth": 7,
    "learning_rate": 0.04,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.10,
    "reg_lambda": 2.0,
    "gamma": 0.0,
    "max_bin": 256,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": N_THREADS,
    "random_state": RANDOM_SEED,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "03_이탈테스트_봉인인증.log", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("ablation")


# 유틸리티

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(features)).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan")


def top_fraction_metrics(y: np.ndarray, p: np.ndarray, fraction: float) -> dict[str, float]:
    n = len(y)
    if n == 0:
        return {"precision": np.nan, "recall": np.nan, "lift": np.nan, "count": 0}
    k = max(1, int(math.ceil(n * fraction)))
    idx = np.argpartition(-p, k - 1)[:k]
    positives = float(y.sum())
    tp = float(y[idx].sum())
    precision = tp / k
    recall = tp / positives if positives > 0 else np.nan
    base_rate = positives / n
    lift = precision / base_rate if base_rate > 0 else np.nan
    return {"precision": precision, "recall": recall, "lift": lift, "count": k}


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    metrics = {
        "rows": float(len(y)),
        "positives": float(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size > 1 else np.nan,
        "roc_auc": safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }
    for pct in (0.01, 0.03, 0.05, 0.10):
        top = top_fraction_metrics(y, p, pct)
        suffix = int(pct * 100)
        metrics[f"precision_top{suffix}"] = top["precision"]
        metrics[f"recall_top{suffix}"] = top["recall"]
        metrics[f"lift_top{suffix}"] = top["lift"]
    return metrics


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def finite_float32(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("float32")
    return x



# 모델 래퍼

@dataclass
class ModelBundle:
    model: Any
    backend: str

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x))[:, 1]
        pred = np.asarray(self.model.predict(x), dtype=float)
        return np.clip(pred, 0, 1)

    def feature_importance(self, features: Sequence[str]) -> pd.DataFrame:
        values: np.ndarray
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_, dtype=float)
        else:
            values = np.zeros(len(features), dtype=float)
        if len(values) != len(features):
            values = np.resize(values, len(features))
        return pd.DataFrame({"feature": list(features), "importance": values}).sort_values(
            "importance", ascending=False
        )


def make_xgboost_model() -> Any:
    from xgboost import XGBClassifier  # type: ignore

    params = dict(XGB_PARAMS)
    if USE_GPU:
        # XGBoost 2.x 방식
        params["device"] = "cuda"
    return XGBClassifier(**params)


def make_hist_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=350,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def fit_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> ModelBundle:
    # 1차: 최신 XGBoost GPU/CPU
    try:
        model = make_xgboost_model()
        model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
        return ModelBundle(model=model, backend="xgboost_cuda" if USE_GPU else "xgboost_cpu")
    except Exception as first_error:  # noqa: BLE001
        LOGGER.warning("XGBoost 최신 GPU 방식 실패: %s", first_error)

    # 2차: 구버전 XGBoost gpu_hist
    if USE_GPU:
        try:
            from xgboost import XGBClassifier  # type: ignore

            params = dict(XGB_PARAMS)
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
            model = XGBClassifier(**params)
            model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
            return ModelBundle(model=model, backend="xgboost_gpu_hist")
        except Exception as second_error:  # noqa: BLE001
            LOGGER.warning("XGBoost gpu_hist 실패: %s", second_error)

    # 3차: CPU HistGradientBoosting
    LOGGER.warning("scikit-learn HistGradientBoosting으로 대체")
    model = make_hist_model()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return ModelBundle(model=model, backend="sklearn_hist")



# 데이터·피처 준비

def load_feature_catalog() -> dict[str, list[str]]:
    path = META_DIR / "feature_catalog.json"
    if not path.exists():
        raise FileNotFoundError("feature_catalog.json 없음. 02_피쳐_생성_봉인.py를 먼저 실행하세요.")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [str(x) for x in v] for k, v in catalog.items()}


def select_valid_features(df: pd.DataFrame, catalog: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    requested = sorted({c for values in catalog.values() for c in values})
    requested = [c for c in requested if c in df.columns]
    stats = []
    valid = []
    corr_base = df if len(df) <= 300_000 else df.sample(n=300_000, random_state=RANDOM_SEED)
    corr_target = pd.to_numeric(corr_base[TARGET], errors="coerce").rename("y")
    for col in requested:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        keep = missing <= MISSING_LIMIT and nunique > 1
        corr_x = pd.to_numeric(corr_base[col], errors="coerce").rename("x")
        pair = pd.concat([corr_x, corr_target], axis=1).dropna()
        abs_corr = float(abs(pair["x"].corr(pair["y"]))) if len(pair) >= 50 and pair["x"].nunique() > 1 else 0.0
        stats.append({"feature": col, "missing_rate": missing, "nunique": nunique, "abs_corr_target": abs_corr, "keep": keep})
        if keep:
            valid.append(col)
    valid_set = set(valid)
    clean_catalog = {
        group: [c for c in cols if c in valid_set]
        for group, cols in catalog.items()
    }
    clean_catalog = {g: cols for g, cols in clean_catalog.items() if cols}
    return valid, clean_catalog, pd.DataFrame(stats)


def make_walk_forward_folds(dates: pd.DatetimeIndex) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    min_train_days = max(500, int(len(dates) * 0.45))
    latest_start = len(dates) - VALID_DAYS
    if latest_start <= min_train_days + PURGE_DAYS:
        raise RuntimeError("워크포워드 fold를 만들 거래일이 부족합니다.")
    starts = np.linspace(min_train_days, latest_start, CV_FOLDS, dtype=int)
    folds = []
    used = set()
    for i, val_start in enumerate(starts):
        val_start = int(val_start)
        if val_start in used:
            continue
        used.add(val_start)
        train_end = val_start - PURGE_DAYS
        val_end = min(len(dates), val_start + VALID_DAYS)
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
        raise RuntimeError("유효한 fold가 2개 미만입니다.")
    return folds


def sample_training_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """양수는 모두 유지하고 음수를 샘플링하되 표본가중치로 원래 모집단을 복원한다."""
    if len(df) <= max_rows:
        return df, np.ones(len(df), dtype="float32")
    pos = df[df[TARGET].eq(1)]
    neg = df[df[TARGET].eq(0)]
    neg_n = max(1, max_rows - len(pos))
    neg_sample = neg.sample(n=min(neg_n, len(neg)), random_state=RANDOM_SEED)
    sampled = pd.concat([pos, neg_sample], ignore_index=False).sort_values("date")
    weights = np.ones(len(sampled), dtype="float32")
    if len(neg_sample) > 0:
        neg_weight = len(neg) / len(neg_sample)
        weights[sampled[TARGET].to_numpy() == 0] = neg_weight
    return sampled, weights


def cap_validation_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_VALID_ROWS:
        return df
    # 날짜 구조를 유지하면서 종목을 균일하게 샘플링
    per_day = max(1, int(MAX_VALID_ROWS / max(1, df["date"].nunique())))
    blocks = []
    for _, block in df.groupby("date", sort=False):
        if len(block) > per_day:
            block = block.sample(n=per_day, random_state=RANDOM_SEED)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else df.iloc[0:0].copy()



# CV 평가

def evaluate_feature_set(
    df: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[dict[str, Any]],
    experiment: str,
    collect_importance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    importance_rows = []
    for fold in folds:
        train = df[df["date"].isin(fold["train_dates"]) & df[TARGET].notna()].copy()
        valid = df[df["date"].isin(fold["valid_dates"]) & df[TARGET].notna()].copy()
        train, weights = sample_training_rows(train, MAX_TRAIN_ROWS)
        valid = cap_validation_rows(valid)
        if train.empty or valid.empty:
            continue

        x_train = finite_float32(train, features)
        y_train = train[TARGET].astype("int8").to_numpy()
        x_valid = finite_float32(valid, features)
        y_valid = valid[TARGET].astype("int8").to_numpy()
        if np.unique(y_train).size < 2:
            LOGGER.warning("%s fold=%s 학습 라벨이 한 종류라 건너뜀", experiment, fold["fold"])
            continue

        started = time.time()
        bundle = fit_model(x_train, y_train, weights)
        pred = bundle.predict_proba(x_valid)
        metrics = classification_metrics(y_valid, pred)
        metrics.update(
            {
                "experiment": experiment,
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "feature_count": len(features),
                "backend": bundle.backend,
                "fit_seconds": time.time() - started,
            }
        )
        rows.append(metrics)
        LOGGER.info(
            "%s fold=%s PR=%.5f RecallTop3=%.4f PrecisionTop3=%.4f backend=%s",
            experiment,
            fold["fold"],
            metrics["pr_auc"],
            metrics["recall_top3"],
            metrics["precision_top3"],
            bundle.backend,
        )
        if collect_importance:
            imp = bundle.feature_importance(features)
            imp["experiment"] = experiment
            imp["fold"] = fold["fold"]
            importance_rows.append(imp)
        del bundle, x_train, x_valid, train, valid
        gc.collect()
    result = pd.DataFrame(rows)
    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    return result, importance


def summarize_cv(rows: pd.DataFrame) -> dict[str, float]:
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "brier",
        "logloss",
        "precision_top1",
        "recall_top1",
        "lift_top1",
        "precision_top3",
        "recall_top3",
        "lift_top3",
        "precision_top5",
        "recall_top5",
        "lift_top5",
    ]
    return {f"mean_{c}": float(rows[c].mean()) for c in metric_cols if c in rows}



# 그룹 이탈테스트

def run_group_ablation(
    df: pd.DataFrame,
    all_features: list[str],
    catalog: dict[str, list[str]],
    folds: Sequence[dict[str, Any]],
    baseline_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    baseline_summary = summarize_cv(baseline_rows)
    results = []
    drop_candidates = []
    keep_groups = []

    for group, group_features in catalog.items():
        remaining = [f for f in all_features if f not in set(group_features)]
        if not remaining:
            continue
        rows, _ = evaluate_feature_set(
            df,
            remaining,
            folds,
            experiment=f"drop_group::{group}",
            collect_importance=False,
        )
        summary = summarize_cv(rows)
        record = {
            "group": group,
            "removed_feature_count": len(group_features),
            "remaining_feature_count": len(remaining),
            **summary,
        }
        record["delta_pr_auc_vs_all"] = record.get("mean_pr_auc", np.nan) - baseline_summary.get("mean_pr_auc", np.nan)
        record["delta_recall_top3_vs_all"] = record.get("mean_recall_top3", np.nan) - baseline_summary.get("mean_recall_top3", np.nan)
        record["delta_precision_top3_vs_all"] = record.get("mean_precision_top3", np.nan) - baseline_summary.get("mean_precision_top3", np.nan)
        record["delta_brier_vs_all"] = record.get("mean_brier", np.nan) - baseline_summary.get("mean_brier", np.nan)
        should_drop = (
            record["delta_pr_auc_vs_all"] >= DROP_GROUP_MIN_PR_GAIN
            and record["delta_recall_top3_vs_all"] >= -DROP_GROUP_MAX_RECALL_LOSS
            and record["delta_precision_top3_vs_all"] >= -DROP_GROUP_MAX_PRECISION_LOSS
        )
        record["decision"] = "DROP_CANDIDATE" if should_drop else "KEEP"
        results.append(record)
        if should_drop:
            drop_candidates.append(group)
        else:
            keep_groups.append(group)
        atomic_csv(pd.DataFrame(results), OUT_DIR / "group_ablation.csv")

    # 독립 leave-one-group-out 결과에서 제거 후보들을 한꺼번에 제거한 뒤 CV 재검증
    selected_features = [
        f
        for group, cols in catalog.items()
        if group not in drop_candidates
        for f in cols
    ]
    selected_features = sorted(set(selected_features))
    if not selected_features:
        selected_features = all_features
        drop_candidates = []
        keep_groups = list(catalog)
    return pd.DataFrame(results), selected_features, drop_candidates



# 개별 피처 이탈테스트

def run_feature_ablation(
    df: pd.DataFrame,
    selected_features: list[str],
    folds: Sequence[dict[str, Any]],
    importance: pd.DataFrame,
) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    avg_imp = importance.groupby("feature", as_index=False)["importance"].mean().sort_values(
        "importance", ascending=False
    )
    candidates = [f for f in avg_imp["feature"].head(FEATURE_ABLATION_TOP_N) if f in selected_features]
    if not candidates:
        return pd.DataFrame()
    # 계산량 제어를 위해 가장 최근 개발 fold 하나에서만 개별 피처 제거 효과를 측정한다.
    fold = [folds[-1]]
    base_rows, _ = evaluate_feature_set(
        df,
        selected_features,
        fold,
        experiment="selected_feature_baseline_last_fold",
        collect_importance=False,
    )
    base = summarize_cv(base_rows)
    records = []
    for feature in candidates:
        remaining = [f for f in selected_features if f != feature]
        rows, _ = evaluate_feature_set(
            df,
            remaining,
            fold,
            experiment=f"drop_feature::{feature}",
            collect_importance=False,
        )
        summary = summarize_cv(rows)
        records.append(
            {
                "feature": feature,
                "baseline_importance": float(
                    avg_imp.loc[avg_imp["feature"] == feature, "importance"].iloc[0]
                ),
                "delta_pr_auc_vs_selected": summary.get("mean_pr_auc", np.nan) - base.get("mean_pr_auc", np.nan),
                "delta_recall_top3_vs_selected": summary.get("mean_recall_top3", np.nan) - base.get("mean_recall_top3", np.nan),
                "delta_precision_top3_vs_selected": summary.get("mean_precision_top3", np.nan) - base.get("mean_precision_top3", np.nan),
                "mean_pr_auc": summary.get("mean_pr_auc", np.nan),
                "mean_recall_top3": summary.get("mean_recall_top3", np.nan),
                "mean_precision_top3": summary.get("mean_precision_top3", np.nan),
            }
        )
        atomic_csv(pd.DataFrame(records), OUT_DIR / "feature_ablation.csv")
    return pd.DataFrame(records).sort_values("delta_pr_auc_vs_selected", ascending=False)



# 최종 동결·봉인 인증

def train_final_model(df: pd.DataFrame, features: Sequence[str]) -> ModelBundle:
    train = df[df[TARGET].notna()].copy()
    train, weights = sample_training_rows(train, MAX_TRAIN_ROWS)
    x = finite_float32(train, features)
    y = train[TARGET].astype("int8").to_numpy()
    LOGGER.info("최종 모델 학습: rows=%d features=%d positive_rate=%.5f", len(train), len(features), y.mean())
    bundle = fit_model(x, y, weights)
    joblib.dump(
        {
            "model": bundle.model,
            "backend": bundle.backend,
            "features": list(features),
            "target": TARGET,
            "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        },
        OUT_DIR / "final_model.joblib",
        compress=3,
    )
    return bundle


def certify_seals(bundle: ModelBundle, features: Sequence[str], frozen_hash: str) -> pd.DataFrame:
    manifest_path = SEALED_DIR / "seal_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("seal_manifest.json 없음")
    seal_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for seal_info in seal_manifest.get("seals", []):
        seal_id = str(seal_info["seal_id"])
        path = Path(seal_info["path"])
        current_sha = sha256_file(path)
        if current_sha != seal_info.get("sha256"):
            raise RuntimeError(f"{seal_id} 봉인 해시 불일치")
        seal = pd.read_parquet(path)
        seal["date"] = pd.to_datetime(seal["date"], errors="coerce")
        valid = seal[seal[TARGET].notna()].copy()
        missing = [f for f in features if f not in valid.columns]
        if missing:
            raise RuntimeError(f"{seal_id} 피처 누락: {missing[:10]}")
        x = finite_float32(valid, features)
        y = valid[TARGET].astype("int8").to_numpy()
        pred = bundle.predict_proba(x)
        metrics = classification_metrics(y, pred)
        metrics.update(
            {
                "seal_id": seal_id,
                "start_date": valid["date"].min(),
                "end_date": valid["date"].max(),
                "feature_count": len(features),
                "feature_list_sha256": frozen_hash,
                "model_backend": bundle.backend,
                "seal_file_sha256": current_sha,
                "certification_status": "EVALUATED_ONCE_NO_RETUNING",
            }
        )
        records.append(metrics)
        pred_out = valid[[c for c in ("date", "ticker", "name", "market", TARGET) if c in valid.columns]].copy()
        pred_out["pred_crash_probability"] = pred
        pred_out["seal_id"] = seal_id
        pred_out["feature_list_sha256"] = frozen_hash
        pred_out.to_parquet(PRED_DIR / f"{seal_id}_predictions.parquet", index=False)
        LOGGER.info(
            "%s 인증 PR=%.5f RecallTop3=%.4f PrecisionTop3=%.4f positives=%d",
            seal_id,
            metrics["pr_auc"],
            metrics["recall_top3"],
            metrics["precision_top3"],
            int(metrics["positives"]),
        )
    result = pd.DataFrame(records)
    atomic_csv(result, OUT_DIR / "seal_certification.csv")
    return result



# 메인

def main() -> None:
    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError("개발 데이터 없음. 02_피쳐_생성_봉인.py를 먼저 실행하세요.")

    LOGGER.info("개발 데이터 로드")
    df = pd.read_parquet(DEVELOPMENT_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET]).sort_values(["date", "ticker"])
    if "sealed_do_not_train_or_tune" in df.columns and df["sealed_do_not_train_or_tune"].fillna(0).ne(0).any():
        raise RuntimeError("개발 파일에 봉인 데이터가 섞여 있습니다.")

    catalog = load_feature_catalog()
    all_features, clean_catalog, feature_stats = select_valid_features(df, catalog)
    atomic_csv(feature_stats, OUT_DIR / "feature_quality.csv")
    if len(all_features) < 10:
        raise RuntimeError(f"유효 피처가 너무 적습니다: {len(all_features)}")
    LOGGER.info("유효 피처 %d개, 그룹 %d개", len(all_features), len(clean_catalog))

    folds = make_walk_forward_folds(pd.DatetimeIndex(df["date"].unique()))
    write_json(
        [
            {
                k: v
                for k, v in fold.items()
                if k not in {"train_dates", "valid_dates"}
            }
            for fold in folds
        ],
        OUT_DIR / "walk_forward_folds.json",
    )

    LOGGER.info("전체 피처 기준모델 CV")
    baseline_rows, baseline_importance = evaluate_feature_set(
        df,
        all_features,
        folds,
        experiment="all_features_baseline",
        collect_importance=True,
    )
    atomic_csv(baseline_rows, OUT_DIR / "baseline_cv.csv")
    if not baseline_importance.empty:
        atomic_csv(baseline_importance, OUT_DIR / "baseline_feature_importance_by_fold.csv")
        avg_imp = baseline_importance.groupby("feature", as_index=False)["importance"].mean().sort_values(
            "importance", ascending=False
        )
        atomic_csv(avg_imp, OUT_DIR / "baseline_feature_importance.csv")

    if RUN_GROUP_ABLATION:
        LOGGER.info("그룹 이탈테스트 시작")
        group_result, selected_features, drop_groups = run_group_ablation(
            df, all_features, clean_catalog, folds, baseline_rows
        )
        atomic_csv(group_result, OUT_DIR / "group_ablation.csv")
    else:
        selected_features = all_features
        drop_groups = []
        group_result = pd.DataFrame()

    selected_groups = [g for g in clean_catalog if g not in drop_groups]
    selected_features = sorted(set(selected_features))

    LOGGER.info("선택 피처 조합 재검증: %d개", len(selected_features))
    selected_rows, selected_importance = evaluate_feature_set(
        df,
        selected_features,
        folds,
        experiment="selected_groups_cv",
        collect_importance=True,
    )
    atomic_csv(selected_rows, OUT_DIR / "selected_groups_cv.csv")
    if not selected_importance.empty:
        atomic_csv(selected_importance, OUT_DIR / "selected_feature_importance_by_fold.csv")

    feature_ablation = pd.DataFrame()
    if RUN_FEATURE_LEVEL_ABLATION:
        LOGGER.info("개별 피처 이탈테스트 시작")
        importance_source = selected_importance if not selected_importance.empty else baseline_importance
        if importance_source.empty or float(importance_source.get("importance", pd.Series(dtype=float)).sum()) <= 0:
            importance_source = (
                feature_stats[feature_stats["feature"].isin(selected_features)]
                [["feature", "abs_corr_target"]]
                .rename(columns={"abs_corr_target": "importance"})
            )
            importance_source["fold"] = -1
            importance_source["experiment"] = "fallback_abs_corr"
        feature_ablation = run_feature_ablation(
            df, selected_features, folds, importance_source
        )
        if not feature_ablation.empty:
            atomic_csv(feature_ablation, OUT_DIR / "feature_ablation.csv")

    frozen_hash = feature_hash(selected_features)
    frozen = {
        "target": TARGET,
        "created_at": pd.Timestamp.now(tz="Asia/Seoul"),
        "selection_source": "development walk-forward CV only",
        "seal_usage": "certification only; never used for selection or tuning",
        "all_feature_count": len(all_features),
        "selected_feature_count": len(selected_features),
        "selected_groups": selected_groups,
        "dropped_groups": drop_groups,
        "features": selected_features,
        "feature_list_sha256": frozen_hash,
        "model_parameters": XGB_PARAMS,
        "use_gpu": USE_GPU,
        "threads": N_THREADS,
        "baseline_cv_mean": summarize_cv(baseline_rows),
        "selected_cv_mean": summarize_cv(selected_rows),
    }
    write_json(frozen, OUT_DIR / "frozen_feature_set.json")

    # 여기서부터 피처 목록과 하이퍼파라미터는 변경하지 않는다.
    LOGGER.info("최종 모델 학습 및 봉인 인증")
    final_bundle = train_final_model(df, selected_features)
    seal_result = certify_seals(final_bundle, selected_features, frozen_hash)

    summary = {
        "target": TARGET,
        "model_backend": final_bundle.backend,
        "feature_list_sha256": frozen_hash,
        "selected_feature_count": len(selected_features),
        "dropped_groups": drop_groups,
        "development_baseline": summarize_cv(baseline_rows),
        "development_selected": summarize_cv(selected_rows),
        "seal_mean": {
            col: float(seal_result[col].mean())
            for col in (
                "pr_auc",
                "roc_auc",
                "brier",
                "precision_top3",
                "recall_top3",
                "lift_top3",
            )
            if col in seal_result
        },
        "seal_rows": seal_result.to_dict("records"),
        "warning": "봉인 결과를 보고 이 실행에서 피처나 모델을 재선택하지 마십시오.",
    }
    write_json(summary, OUT_DIR / "run_summary.json")

    print("\n=== 완료 ===")
    print(f"선택 피처: {len(selected_features):,}개")
    print(f"제거 후보 그룹: {drop_groups}")
    print(f"피처 해시: {frozen_hash}")
    print(f"봉인 인증: {OUT_DIR / 'seal_certification.csv'}")
    결과압축_생성("03_이탈테스트_봉인인증")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            결과압축_생성("03_이탈테스트_봉인인증")
        finally:
            raise
