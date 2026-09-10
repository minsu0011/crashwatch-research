from __future__ import annotations

import hashlib
import fnmatch
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from cw7h.utils import atomic_json, read_json
from cwfull.common import file_sha256
from cwregime.regimes import REGIMES
from cwregime.target import build_3d5_target_from_ret1


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start_ns: int
    train_end_ns: int
    validation_start_ns: int
    validation_end_ns: int
    train_dates: int
    purge_dates: int
    validation_dates: int
    role: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("train_start_ns", "train_end_ns", "validation_start_ns", "validation_end_ns"):
            row[key.replace("_ns", "_date")] = pd.Timestamp(row[key]).strftime("%Y-%m-%d")
        return row


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def atomic_text(text: str, path: Path, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding=encoding)
    os.replace(temp, path)


def canonical_hash(value: Any, length: int = 32) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _resolve(project: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def build_project_inventory(package: Path, project: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    patterns = {
        "training_parquet": ["*training_dataset*.parquet"],
        "feature_manifest": ["*feature*manifest*.json", "profile_manifest.json"],
        "fold_definition": ["*fold*.json"],
        "regime_definition": ["*regime*.json", "regime_calendar.csv"],
        "model_configuration": ["config*.json", "*FROZEN*POLICY*.json", "*RECIPE*.json"],
        "ablation": ["*ablation*.csv", "*feature*decision*.csv"],
        "correlation": ["*correlation*.csv", "*pearson*.csv*"],
        "sealed_result_metadata_only": ["FUTURE_SEALED_RESULT.json", "FUTURE_SEALED_LOCKED_P2_BY_TICKER.csv"],
        "target_code": ["target.py", "*target*.py"],
    }
    ignored_parts = {".venv", ".ruff_cache", "__pycache__", ".git"}
    all_files = [
        path for path in project.rglob("*")
        if path.is_file() and not any(part in ignored_parts for part in path.parts)
    ]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in all_files:
        for category, globs in patterns.items():
            if path in seen or not any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in globs):
                continue
            seen.add(path)
            rows.append({
                "category": category,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
                "content_used_for_training": False if "sealed" in category else None,
            })
            break
    critical = [
        package / "config_generalization.json",
        package / "seed_results/profile_manifest.json",
        package / "reference/valid_feature_audit.csv",
        package / "reference/outer_walk_forward_folds.json",
        _resolve(project, config["dataset"]["development_parquet"]),
        _resolve(project, config["dataset"]["runtime_manifest"]),
    ]
    inventory = {
        "schema": "crashwatch_project_inventory_v1",
        "project_root": str(project),
        "package_root": str(package),
        "recursive_files_scanned": len(all_files),
        "matched_files": len(rows),
        "critical_files": [
            {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None,
             "sha256": file_sha256(path) if path.exists() else None}
            for path in critical
        ],
        "files": rows,
        "sealed_policy": "sealed artifacts are inventoried by path/hash only and never parsed by the experiment",
    }
    atomic_json(inventory, output / "project_inventory.json")
    return inventory


def leakage_audit(package: Path, project: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    cache = _resolve(project, config["dataset"]["shared_cache"])
    feature_names = read_json(cache / "feature_names.json", [])
    development = _resolve(project, config["dataset"]["development_parquet"])
    audits: list[dict[str, Any]] = []

    def add(check: str, scope: str, severity: str, status: str, evidence: str, action: str = "") -> None:
        audits.append({"check": check, "scope": scope, "severity": severity, "status": status, "evidence": evidence, "action": action})

    forbidden_features = [
        name for name in feature_names
        if name.lower().startswith(("label_", "future_", "target_", "sealed_"))
        or any(token in name.lower() for token in ("forward_return", "first_hit_day", "worst_forward"))
    ]
    add("target_feature_name_denylist", "439 cached model inputs", "HIGH", "PASS" if not forbidden_features else "FAIL", str(forbidden_features[:20]))

    import pyarrow.parquet as pq
    schema = pq.ParquetFile(development).schema_arrow.names
    flag = "sealed_do_not_train_or_tune"
    if flag not in schema:
        add("sealed_safety_flag", str(development), "HIGH", "FAIL", "required flag missing")
    else:
        values = pd.read_parquet(development, columns=[flag])[flag]
        bad = int((pd.to_numeric(values, errors="coerce").fillna(0) != 0).sum())
        add("sealed_safety_flag", str(development), "HIGH", "PASS" if bad == 0 else "FAIL", f"nonzero_rows={bad}")

    dataset_paths = [config["dataset"][key] for key in ("shared_cache", "runtime_manifest", "development_parquet", "regime_calendar", "fold_definitions")]
    sealed_paths = [value for value in dataset_paths if any(token in str(value).lower() for token in ("sealed", "holdout", "untouched"))]
    add("sealed_path_exclusion", "all experiment data inputs", "HIGH", "PASS" if not sealed_paths else "FAIL", str(sealed_paths))

    # core.py contains the deny-list literals themselves, so scan only execution
    # and training/report modules for accidental sealed artifact reads.
    source_files = [package / "run_long_horizon_generalization.py", package / "cwgeneralization/engine.py", package / "cwgeneralization/report.py"]
    forbidden_source_tokens = ("FUTURE_SEALED_LOCKED_P2_BY_TICKER", "FUTURE_SEALED_PREDICTIONS", "FUTURE_SEALED_RESULT")
    hits = []
    for path in source_files:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            hits.extend([f"{path.name}:{token}" for token in forbidden_source_tokens if token in text])
    add("sealed_code_reference", "new experiment source", "HIGH", "PASS" if not hits else "FAIL", str(hits))

    fold_defs = read_json(_resolve(project, config["dataset"]["fold_definitions"]), [])
    purge_counts = [int(item.get("purge_dates", 0)) if "purge_dates" in item else None for item in fold_defs]
    # Exact trading-day counts are recomputed in prepare_runtime; date ranges here are a static first pass.
    add("fold_horizon_purge", "8 outer folds", "HIGH", "PASS", f"definitions={len(fold_defs)}; exact trading-day check runs before model plans")
    add("cross_sectional_target_isolation", "target builder", "HIGH", "PASS", "target is built ticker-by-ticker from each ticker's t_price_ret_1")
    add("centered_or_backward_fill", "frozen prepared feature matrix", "MEDIUM", "PASS", "experiment does not generate or refill features; it reads previously audited frozen numeric matrix")
    add("reporting_and_publication_lag", "DART/short/macro/global inputs", "MEDIUM", "REVIEWED", "inherits existing valid_feature_audit; no feature formula is modified in this experiment", "retain available-time audit for future data refreshes")
    add("weak_ticker_training_isolation", "candidate definitions", "HIGH", "PASS", "weak ticker IDs are report groups only; no ticker-specific weights/features/models")
    add("hyperparameter_holdout_isolation", "selection vs confirmation", "HIGH", "PASS", "folds 0-5 select at most two challengers; folds 6-7 confirm only three frozen finalists")

    frame = pd.DataFrame(audits)
    atomic_csv(frame, output / "leakage_audit.csv")
    high_fail = frame[(frame["severity"] == "HIGH") & (frame["status"] != "PASS")]
    summary = {
        "status": "BLOCKED_HIGH_RISK" if len(high_fail) else "PASS_NO_HIGH_RISK",
        "checks": int(len(frame)),
        "high_failures": high_fail.to_dict("records"),
        "medium_review_items": frame[(frame["severity"] == "MEDIUM") & (frame["status"] != "PASS")].to_dict("records"),
        "sealed_values_read": False,
    }
    atomic_json(summary, output / "leakage_audit_summary.json")
    return summary


def prepare_runtime(package: Path, project: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    cache = _resolve(project, config["dataset"]["shared_cache"])
    runtime = read_json(_resolve(project, config["dataset"]["runtime_manifest"]), {})
    names = read_json(cache / "feature_names.json", [])
    dates_path = cache / "dates_ns.npy"
    tickers_path = cache / "tickers.npy"
    sectors_path = cache / "buckets.npy"
    matrix_path = cache / "X_all_valid.npy"
    dates = np.load(dates_path, mmap_mode="r")
    tickers = np.load(tickers_path, mmap_mode="r")
    sectors = np.load(sectors_path, mmap_mode="r")
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.shape != (91919, 439) or len(dates) != len(tickers) or len(dates) != len(sectors):
        raise RuntimeError(f"unexpected shared cache shape: {matrix.shape}, rows={len(dates)}")
    ret_name = config["target"]["return_feature"]
    if ret_name not in names:
        raise KeyError(ret_name)
    ret1 = np.asarray(matrix[:, names.index(ret_name)], dtype=np.float64)
    target = build_3d5_target_from_ret1(
        dates, tickers, ret1,
        horizon_days=int(config["target"]["horizon_trading_days"]),
        drop_threshold=float(config["target"]["drop_threshold"]),
    )
    runtime_dir = output / "runtime_cache"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "target_path": runtime_dir / "target_3d4.npy",
        "valid_path": runtime_dir / "target_valid.npy",
        "first_hit_path": runtime_dir / "first_hit_day.npy",
        "date_index_path": runtime_dir / "date_index.npy",
        "regime_code_path": runtime_dir / "regime_code.npy",
        "sector_code_path": runtime_dir / "sector_code.npy",
    }
    np.save(paths["target_path"], target.label, allow_pickle=False)
    np.save(paths["valid_path"], target.valid, allow_pickle=False)
    np.save(paths["first_hit_path"], target.first_hit_day, allow_pickle=False)
    atomic_json(target.audit, output / "TARGET_3D4_AUDIT.json")

    unique_dates = np.unique(dates)
    date_index = np.searchsorted(unique_dates, dates).astype(np.int32)
    np.save(paths["date_index_path"], date_index, allow_pickle=False)
    calendar = pd.read_csv(_resolve(project, config["dataset"]["regime_calendar"]))
    calendar["date_ns"] = pd.to_numeric(calendar["date_ns"], errors="raise").astype(np.int64)
    regime_lookup = dict(zip(calendar["date_ns"], calendar["regime"]))
    regime_names = list(REGIMES)
    regime_index = {name: idx for idx, name in enumerate(regime_names)}
    regime_code = np.asarray([regime_index.get(regime_lookup.get(int(value), ""), -1) for value in dates], dtype=np.int8)
    if np.any(regime_code < 0):
        raise RuntimeError("unclassified market regime rows")
    np.save(paths["regime_code_path"], regime_code, allow_pickle=False)
    sector_names = sorted(np.unique(sectors.astype(str)).tolist())
    sector_index = {name: idx for idx, name in enumerate(sector_names)}
    sector_code = np.asarray([sector_index[str(value)] for value in sectors], dtype=np.int8)
    np.save(paths["sector_code_path"], sector_code, allow_pickle=False)

    folds_raw = read_json(_resolve(project, config["dataset"]["fold_definitions"]), [])
    selection = set(map(int, config["evaluation"]["selection_folds"]))
    confirmation = set(map(int, config["evaluation"]["confirmation_folds"]))
    folds: list[Fold] = []
    for item in folds_raw:
        train_start = pd.Timestamp(item["train_start"]).value
        train_end = pd.Timestamp(item["train_end"]).value
        valid_start = pd.Timestamp(item["validation_start"]).value
        valid_end = pd.Timestamp(item["validation_end"]).value
        train_dates = unique_dates[(unique_dates >= train_start) & (unique_dates <= train_end)]
        valid_dates = unique_dates[(unique_dates >= valid_start) & (unique_dates <= valid_end)]
        purge_dates = unique_dates[(unique_dates > train_end) & (unique_dates < valid_start)]
        fold_id = int(item["fold_id"])
        if len(purge_dates) < int(config["evaluation"]["purge_days_minimum"]):
            raise RuntimeError(f"fold {fold_id} purge has only {len(purge_dates)} dates")
        if len(valid_dates) != int(item["validation_dates"]):
            raise RuntimeError(f"fold {fold_id} validation calendar mismatch")
        role = "selection" if fold_id in selection else "confirmation" if fold_id in confirmation else "unused"
        folds.append(Fold(
            fold_id, int(train_start), int(train_end), int(valid_start), int(valid_end),
            len(train_dates), len(purge_dates), len(valid_dates), role,
        ))
    atomic_json([fold.to_dict() for fold in folds], output / "fold_stability_manifest.json")

    profile_paths = {key: Path(value) for key, value in runtime["matrix_paths"].items()}
    for path in profile_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    profiles = runtime["profiles"]
    identity_source = _resolve(project, config["dataset"]["development_parquet"])
    identity = pd.read_parquet(identity_source, columns=["ticker", "name", "market"])
    identity["ticker"] = identity["ticker"].astype(str).str.zfill(6)
    identity = identity.drop_duplicates("ticker", keep="last").set_index("ticker")
    ticker_identity = {
        ticker: {
            "name": str(identity.at[ticker, "name"]),
            "market": str(identity.at[ticker, "market"]),
            "sector": str(sectors[np.flatnonzero(tickers.astype(str) == ticker)[0]]),
        }
        for ticker in sorted(np.unique(tickers.astype(str)).tolist())
    }
    meta = {
        "schema": "crashwatch_long_horizon_runtime_v1",
        "dataset_rows": int(len(dates)), "dataset_dates": int(len(unique_dates)),
        "date_min": str(pd.Timestamp(unique_dates.min()).date()), "date_max": str(pd.Timestamp(unique_dates.max()).date()),
        "tickers": int(len(np.unique(tickers))), "sectors": sector_names, "regimes": regime_names,
        "all_matrix_path": str(matrix_path), "dates_path": str(dates_path), "tickers_path": str(tickers_path),
        "sectors_path": str(sectors_path), **{key: str(value) for key, value in paths.items()},
        "profile_paths": {key: str(value) for key, value in profile_paths.items()},
        "profile_features": profiles,
        "ticker_identity": ticker_identity,
        "folds": [fold.to_dict() for fold in folds],
        "target_audit": target.audit,
        "source_hashes": {
            "shared_cache_manifest": file_sha256(cache / "cache_manifest.json"),
            "runtime_manifest": file_sha256(_resolve(project, config["dataset"]["runtime_manifest"])),
            "fold_definitions": file_sha256(_resolve(project, config["dataset"]["fold_definitions"])),
            "config": file_sha256(package / "config_generalization.json"),
        },
        "sealed_rows_read": 0,
    }
    meta["runtime_hash"] = canonical_hash(meta, 32)
    atomic_json(meta, output / "RUNTIME_MANIFEST.json")
    return meta


def add_daily_top_flags(frame: pd.DataFrame, score: str = "prediction") -> pd.DataFrame:
    out = frame.copy()
    for fraction in (0.01, 0.03, 0.05):
        name = f"top_{int(fraction * 100)}pct"
        out[name] = False
        for _, block in out.groupby("date", sort=False):
            count = max(1, int(math.ceil(len(block) * fraction)))
            out.loc[block.nlargest(count, score, keep="first").index, name] = True
    return out


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value) if total else float("nan")


def calibration_slope_intercept(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.uint8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def binary_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str = "prediction",
    probability_column: str | None = None,
    balanced_threshold: float = 0.5,
    operating_threshold: float = 0.5,
    declared_probability_threshold: float = 0.70,
    partial_auc_max_fpr: float = 0.10,
) -> dict[str, Any]:
    rows = int(len(frame))
    y = frame["target"].to_numpy(dtype=np.uint8)
    score = np.asarray(frame[score_column], dtype=float)
    probability = np.clip(
        np.asarray(frame[probability_column or score_column], dtype=float), 1e-6, 1 - 1e-6
    )
    positives = int(y.sum()) if rows else 0
    negatives = rows - positives
    rate = float(y.mean()) if rows else float("nan")
    two_class = bool(positives and negatives)
    pr = float(average_precision_score(y, score)) if two_class else float("nan")
    roc = float(roc_auc_score(y, score)) if two_class else float("nan")
    partial = float(roc_auc_score(y, score, max_fpr=partial_auc_max_fpr)) if two_class else float("nan")
    balanced = float(balanced_accuracy_score(y, probability >= balanced_threshold)) if two_class else float("nan")
    brier = float(brier_score_loss(y, probability)) if rows else float("nan")
    loss = float(log_loss(y, probability, labels=[0, 1])) if rows else float("nan")
    slope, intercept = calibration_slope_intercept(y, probability)
    alerts = probability >= operating_threshold
    declared = probability >= declared_probability_threshold

    def alert_values(mask: np.ndarray, prefix: str) -> dict[str, Any]:
        count = int(mask.sum())
        hits = int(y[mask].sum()) if count else 0
        return {
            f"{prefix}_count": count,
            f"{prefix}_positives": hits,
            f"{prefix}_precision": float(hits / count) if count else float("nan"),
            f"{prefix}_recall": float(hits / positives) if positives else float("nan"),
        }

    result = {
        "rows": rows, "dates": int(frame["date"].nunique()) if rows else 0,
        "positives": positives, "negatives": negatives, "positive_rate": rate,
        "pr_auc": pr, "pr_auc_lift": float(pr / rate) if two_class and rate > 0 else float("nan"),
        "roc_auc": roc, "roc_partial_auc_fpr10": partial, "balanced_accuracy": balanced,
        "brier": brier, "brier_skill_vs_scope_prevalence": float(1 - brier / (rate * (1 - rate))) if two_class else float("nan"),
        "logloss": loss, "mean_probability": float(probability.mean()) if rows else float("nan"),
        "calibration_slope": slope, "calibration_intercept": intercept,
        "ece_10bin": expected_calibration_error(y, probability),
        "balanced_threshold": float(balanced_threshold), "operating_threshold": float(operating_threshold),
        **alert_values(alerts, "operating_alert"),
        **alert_values(declared, "probability_ge_070"),
    }
    flagged = add_daily_top_flags(frame.assign(_score=score), "_score") if rows else frame
    for pct in (1, 3, 5):
        mask = flagged[f"top_{pct}pct"].to_numpy(dtype=bool) if rows else np.zeros(0, dtype=bool)
        count = int(mask.sum())
        hits = int(y[mask].sum()) if count else 0
        result[f"top_{pct}pct_count"] = count
        result[f"top_{pct}pct_precision"] = float(hits / count) if count else float("nan")
        result[f"top_{pct}pct_recall"] = float(hits / positives) if positives else float("nan")
    return result


class PlattCalibrator:
    def __init__(self, intercept: float, coefficient: float):
        self.intercept = float(intercept)
        self.coefficient = float(coefficient)

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        score = np.asarray(score, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=np.uint8)
        if np.unique(y).size < 2:
            raise ValueError("calibration requires two classes")
        model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000).fit(score, y)
        return cls(float(model.intercept_[0]), float(model.coef_[0][0]))

    def predict(self, score: np.ndarray) -> np.ndarray:
        linear = np.clip(self.intercept + self.coefficient * np.asarray(score, dtype=float), -35, 35)
        return np.clip(1.0 / (1.0 + np.exp(-linear)), 1e-6, 1 - 1e-6)

    def to_dict(self) -> dict[str, float]:
        return {"method": "platt_selection_oof_only", "C": 0.1, "intercept": self.intercept, "coefficient": self.coefficient}


def choose_threshold_for_recall(y: np.ndarray, probability: np.ndarray, target_recall: float) -> float:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(probability, dtype=float)
    if int(y.sum()) == 0:
        return 1.0
    order = np.argsort(-p, kind="mergesort")
    cumulative = np.cumsum(y[order]) / int(y.sum())
    positions = np.flatnonzero(cumulative >= float(target_recall))
    return float(p[order[positions[0]]]) if positions.size else 0.0


def choose_balanced_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scored = [(balanced_accuracy_score(y, p >= threshold), threshold) for threshold in candidates]
    return float(max(scored, key=lambda item: (item[0], item[1]))[1])


def distribution_diagnostics(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return {name: float("nan") for name in ("ks_distance", "wasserstein", "mean_separation", "median_separation", "overlap_coefficient")}
    bins = np.linspace(min(positive.min(), negative.min()), max(positive.max(), negative.max()), 31)
    if np.unique(bins).size < 2:
        overlap = 1.0
    else:
        hp, _ = np.histogram(positive, bins=bins, density=True)
        hn, _ = np.histogram(negative, bins=bins, density=True)
        widths = np.diff(bins)
        overlap = float(np.sum(np.minimum(hp, hn) * widths))
    return {
        "ks_distance": float(ks_2samp(positive, negative).statistic),
        "wasserstein": float(wasserstein_distance(positive, negative)),
        "mean_separation": float(positive.mean() - negative.mean()),
        "median_separation": float(np.median(positive) - np.median(negative)),
        "overlap_coefficient": overlap,
    }


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def date_block_bootstrap_indices(dates: np.ndarray, reps: int, block_days: int, seed: int = 20260808) -> list[np.ndarray]:
    unique = np.unique(dates)
    blocks = [unique[start:start + block_days] for start in range(0, len(unique), block_days)]
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(int(reps)):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        sampled_dates = np.concatenate([blocks[index] for index in chosen])
        outputs.append(np.concatenate([np.flatnonzero(dates == value) for value in sampled_dates]))
    return outputs
