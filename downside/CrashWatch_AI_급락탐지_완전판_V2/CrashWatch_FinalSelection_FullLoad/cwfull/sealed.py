from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, read_json
from .common import file_sha256, set_worker_mode

DATE_CANDIDATES = ["date", "trade_date", "trading_date", "datetime", "dt", "일자", "날짜"]
TICKER_CANDIDATES = ["ticker", "stock_code", "code", "symbol", "종목코드", "단축코드"]
TARGET_COLUMN = "label_abs_crash_20"


def _detect(names: list[str], candidates: list[str], kind: str) -> str:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"sealed dataset에서 {kind} column을 찾지 못했습니다: {names[:30]}")


def _load_lock(selection_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = read_json(selection_output / "results" / "LOCKED_PROFILE.json", {})
    features = read_json(selection_output / "results" / "LOCKED_FEATURE_MANIFEST.json", {})
    if not lock.get("sealed_evaluation_allowed") or not lock.get("locked_profile"):
        raise RuntimeError(f"프로필이 잠기지 않았습니다: {lock}")
    if not features.get("features"):
        raise RuntimeError("LOCKED_FEATURE_MANIFEST.json이 없습니다.")
    if features.get("profile") != lock.get("locked_profile"):
        raise RuntimeError("LOCKED_PROFILE과 LOCKED_FEATURE_MANIFEST가 일치하지 않습니다.")
    return lock, features


def preflight(selection_output: Path, sealed_dataset: Path, package_root: Path) -> dict[str, Any]:
    selection_output = selection_output.resolve()
    sealed_dataset = sealed_dataset.resolve()
    lock, feature_manifest = _load_lock(selection_output)
    if not sealed_dataset.exists() or sealed_dataset.suffix.lower() != ".parquet":
        raise FileNotFoundError(sealed_dataset)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow가 필요합니다. INSTALL_REQUIREMENTS.bat를 실행하세요.") from exc
    names = pq.ParquetFile(sealed_dataset).schema_arrow.names
    date_column = _detect(names, DATE_CANDIDATES, "date")
    ticker_column = _detect(names, TICKER_CANDIDATES, "ticker")
    required = [TARGET_COLUMN, date_column, ticker_column] + list(feature_manifest["features"])
    missing = [column for column in required if column not in names]
    if missing:
        raise KeyError(f"sealed dataset 필수 컬럼 누락 ({len(missing)}): {missing[:40]}")
    dates = pd.read_parquet(sealed_dataset, columns=[date_column], engine="pyarrow")[date_column]
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        raise ValueError("sealed date column에 파싱 실패 값이 있습니다.")
    compatibility = read_json(selection_output / "dataset_compatibility.json", {})
    development_max = pd.Timestamp(compatibility.get("manifest", {}).get("date_max"))
    sealed_min = pd.Timestamp(dates.min())
    sealed_max = pd.Timestamp(dates.max())
    if not sealed_min > development_max:
        raise ValueError(f"sealed 시작일이 개발 데이터 이후가 아닙니다: dev_max={development_max}, sealed_min={sealed_min}")
    payload = {
        "status": "preflight_passed",
        "created_epoch": time.time(),
        "selection_output": str(selection_output),
        "locked_profile": lock["locked_profile"],
        "locked_feature_count": len(feature_manifest["features"]),
        "locked_feature_hash": feature_manifest["feature_hash"],
        "sealed_dataset": str(sealed_dataset),
        "sealed_sha256": file_sha256(sealed_dataset),
        "sealed_rows": int(len(dates)),
        "sealed_unique_dates": int(dates.nunique()),
        "sealed_date_min": sealed_min.strftime("%Y-%m-%d"),
        "sealed_date_max": sealed_max.strftime("%Y-%m-%d"),
        "development_date_max": development_max.strftime("%Y-%m-%d"),
        "date_column": date_column,
        "ticker_column": ticker_column,
        "target_column_present": TARGET_COLUMN in names,
        "target_distribution_read": False,
        "metrics_computed": False,
        "package_root": str(package_root),
    }
    sealed_dir = selection_output / "sealed_evaluation"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, sealed_dir / "SEALED_PREFLIGHT.json")
    return payload


def _lgb_seed_worker(plan: dict[str, Any]) -> dict[str, Any]:
    set_worker_mode(int(plan["threads"]), "normal")
    import lightgbm as lgb
    x_train = np.load(plan["x_train_path"], mmap_mode="r")
    y_train = np.load(plan["y_train_path"], mmap_mode="r")
    x_test = np.load(plan["x_test_path"], mmap_mode="r")
    positives = max(1, int(np.sum(y_train == 1)))
    negatives = max(1, int(np.sum(y_train == 0)))
    params = dict(plan["params"])
    params.update(
        {
            "objective": "binary", "metric": "None", "scale_pos_weight": negatives / positives,
            "num_threads": int(plan["threads"]), "deterministic": True, "force_col_wise": True,
            "feature_pre_filter": False, "verbosity": -1, "seed": int(plan["seed"]),
            "feature_fraction_seed": int(plan["seed"]), "bagging_seed": int(plan["seed"]),
            "drop_seed": int(plan["seed"]),
        }
    )
    dataset = lgb.Dataset(np.asarray(x_train), label=np.asarray(y_train), free_raw_data=True, params={"max_bin": int(params["max_bin"]), "feature_pre_filter": False})
    model = lgb.train(params, dataset, num_boost_round=int(plan["rounds"]), callbacks=[lgb.log_evaluation(0)])
    prediction = np.asarray(model.predict(np.asarray(x_test)), dtype=np.float32)
    path = Path(plan["prediction_path"])
    np.save(path, prediction)
    return {"seed": int(plan["seed"]), "prediction_path": str(path), "backend": "lightgbm_cpu"}


def _xgb_assignment_worker(plan: dict[str, Any], queue: mp.Queue) -> None:
    try:
        set_worker_mode(int(plan["params"]["nthread"]), "normal")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import xgboost as xgb
        x_train = np.load(plan["x_train_path"], mmap_mode="r")
        y_train = np.load(plan["y_train_path"], mmap_mode="r")
        x_test = np.load(plan["x_test_path"], mmap_mode="r")
        dtrain = xgb.QuantileDMatrix(np.asarray(x_train), label=np.asarray(y_train), max_bin=int(plan["params"]["max_bin"]), nthread=int(plan["params"]["nthread"]))
        dtest = xgb.QuantileDMatrix(np.asarray(x_test), ref=dtrain, max_bin=int(plan["params"]["max_bin"]), nthread=int(plan["params"]["nthread"]))
        positives = max(1, int(np.sum(y_train == 1)))
        negatives = max(1, int(np.sum(y_train == 0)))
        results = []
        for seed in plan["seeds"]:
            params = dict(plan["params"])
            params.update({"objective": "binary:logistic", "eval_metric": "aucpr", "device": "cuda", "tree_method": "hist", "scale_pos_weight": negatives / positives, "seed": int(seed), "verbosity": 0})
            model = xgb.train(params, dtrain, num_boost_round=int(plan["rounds"]), verbose_eval=False)
            prediction = np.asarray(model.predict(dtest), dtype=np.float32)
            path = Path(plan["output_dir"]) / f"xgb_seed_{seed}.npy"
            np.save(path, prediction)
            results.append({"seed": int(seed), "prediction_path": str(path), "backend": "xgboost_cuda"})
            del model, prediction
        queue.put({"ok": True, "results": results})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


def run_once(
    selection_output: Path,
    sealed_dataset: Path,
    package_root: Path,
    config: dict[str, Any],
    confirmation: str,
) -> dict[str, Any]:
    selection_output = selection_output.resolve()
    sealed_dataset = sealed_dataset.resolve()
    sealed_dir = selection_output / "sealed_evaluation"
    preflight_payload = read_json(sealed_dir / "SEALED_PREFLIGHT.json", {})
    if preflight_payload.get("status") != "preflight_passed":
        raise RuntimeError("먼저 RUN_SEALED_PREFLIGHT.bat를 실행하세요.")
    required_phrase = str(config["sealed"]["confirmation_phrase"])
    if confirmation != required_phrase:
        raise ValueError(f"확인 문구가 필요합니다: {required_phrase}")
    current_hash = file_sha256(sealed_dataset)
    if current_hash != preflight_payload.get("sealed_sha256"):
        raise RuntimeError("preflight 이후 sealed 파일이 변경되었습니다.")
    started_marker = sealed_dir / "SEALED_EVALUATION_STARTED.json"
    consumed_marker = sealed_dir / "SEALED_EVALUATION_CONSUMED.json"
    if started_marker.exists() or consumed_marker.exists():
        raise RuntimeError("sealed 평가는 이미 시작되었거나 소비되었습니다. 재실행할 수 없습니다.")
    lock, feature_manifest = _load_lock(selection_output)
    atomic_json(
        {
            "status": "started_irreversibly",
            "started_epoch": time.time(),
            "sealed_sha256": current_hash,
            "locked_profile": lock["locked_profile"],
            "warning": "이 마커가 생성된 뒤에는 기술적 실패가 발생해도 sealed를 다시 실행하지 않습니다.",
        },
        started_marker,
    )

    date_column = str(preflight_payload["date_column"])
    ticker_column = str(preflight_payload["ticker_column"])
    features = list(feature_manifest["features"])
    columns = [date_column, ticker_column, TARGET_COLUMN] + features
    sealed = pd.read_parquet(sealed_dataset, columns=columns, engine="pyarrow")
    sealed[date_column] = pd.to_datetime(sealed[date_column], errors="raise")
    sealed = sealed.sort_values([date_column, ticker_column], kind="mergesort").reset_index(drop=True)
    y_test = pd.to_numeric(sealed[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.uint8)
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("sealed target이 binary가 아닙니다.")
    x_test = sealed[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    dates_test = sealed[date_column].astype("int64").to_numpy(dtype=np.int64)

    compatibility = read_json(selection_output / "dataset_compatibility.json", {})
    cache_root = Path(compatibility["cache_root"])
    all_features = json.load((cache_root / "feature_names.json").open(encoding="utf-8"))
    index = {feature: position for position, feature in enumerate(all_features)}
    indices = np.asarray([index[feature] for feature in features], dtype=np.int32)
    X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
    y = np.load(cache_root / "target.npy", mmap_mode="r")
    dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
    unique_dates = np.unique(dates)
    purge_days = int(config["sealed"]["purge_trading_days"])
    if len(unique_dates) <= purge_days:
        raise RuntimeError("development dates가 purge보다 짧습니다.")
    train_cutoff = unique_dates[-(purge_days + 1)]
    train_stop = int(np.searchsorted(dates, train_cutoff, side="right"))
    x_train = np.ascontiguousarray(X[:train_stop][:, indices], dtype=np.float32)
    y_train = np.asarray(y[:train_stop], dtype=np.uint8)

    cache = sealed_dir / "runtime_cache"
    cache.mkdir(parents=True, exist_ok=True)
    x_train_path = cache / "x_train.npy"; y_train_path = cache / "y_train.npy"; x_test_path = cache / "x_test.npy"
    np.save(x_train_path, x_train); np.save(y_train_path, y_train); np.save(x_test_path, x_test)
    del x_train, y_train, x_test

    best = read_json(package_root / "seed_results" / "effective_best_iterations.json", {})
    lgb_rounds = int(round(float(np.median([int(value) for value in best.values()]))))
    seeds = [int(seed) for seed in config["selection"]["seeds"]]
    lgb_params = dict(config["lightgbm"])
    lgb_plans = [
        {
            "x_train_path": str(x_train_path), "y_train_path": str(y_train_path), "x_test_path": str(x_test_path),
            "prediction_path": str(sealed_dir / f"lgb_seed_{seed}.npy"), "seed": seed, "rounds": lgb_rounds,
            "threads": 6, "params": lgb_params,
        }
        for seed in seeds
    ]

    context = mp.get_context("spawn")
    xgb_queue: mp.Queue = context.Queue()
    xgb_assignments = [seeds[::2], seeds[1::2]]
    xgb_processes = []
    xgb_params = dict(config["xgboost"])
    for worker_index, worker_seeds in enumerate(xgb_assignments):
        plan = {
            "x_train_path": str(x_train_path), "y_train_path": str(y_train_path), "x_test_path": str(x_test_path),
            "output_dir": str(sealed_dir), "seeds": worker_seeds, "rounds": int(config["sealed"]["xgboost_rounds"]),
            "params": xgb_params,
        }
        process = context.Process(target=_xgb_assignment_worker, args=(plan, xgb_queue), name=f"sealed-xgb-{worker_index}")
        process.start(); xgb_processes.append(process)

    with cf.ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        lgb_results = list(executor.map(_lgb_seed_worker, lgb_plans))
    for process in xgb_processes:
        process.join()
    xgb_messages = [xgb_queue.get(timeout=10) for _ in xgb_processes]
    if any(not message.get("ok") for message in xgb_messages):
        raise RuntimeError(f"sealed XGBoost 실패: {xgb_messages}")
    xgb_results = [item for message in xgb_messages for item in message["results"]]

    lgb_predictions = np.vstack([np.load(item["prediction_path"]) for item in sorted(lgb_results, key=lambda x: x["seed"])])
    xgb_predictions = np.vstack([np.load(item["prediction_path"]) for item in sorted(xgb_results, key=lambda x: x["seed"])])
    lgb_mean = lgb_predictions.mean(axis=0)
    xgb_mean = xgb_predictions.mean(axis=0)
    equal_mean = 0.5 * lgb_mean + 0.5 * xgb_mean
    metrics = {
        "primary_lightgbm_5seed_mean": compute_metrics(y_test, lgb_mean, dates_test),
        "secondary_xgboost_5seed_mean": compute_metrics(y_test, xgb_mean, dates_test),
        "exploratory_equal_weight_lgb_xgb": compute_metrics(y_test, equal_mean, dates_test),
    }
    prediction_frame = pd.DataFrame(
        {
            "date": sealed[date_column].dt.strftime("%Y-%m-%d"),
            "ticker": sealed[ticker_column].astype(str),
            "target": y_test,
            "prediction_lightgbm": lgb_mean,
            "prediction_xgboost": xgb_mean,
            "prediction_equal_weight": equal_mean,
        }
    )
    prediction_frame.to_parquet(sealed_dir / "sealed_predictions.parquet", index=False)
    result = {
        "status": "consumed_completed",
        "completed_epoch": time.time(),
        "locked_profile": lock["locked_profile"],
        "feature_count": len(features),
        "feature_hash": feature_manifest["feature_hash"],
        "sealed_sha256": current_hash,
        "development_train_rows_after_purge": train_stop,
        "development_train_end": pd.Timestamp(train_cutoff).strftime("%Y-%m-%d"),
        "purge_trading_days": purge_days,
        "sealed_rows": int(len(sealed)),
        "sealed_date_min": sealed[date_column].min().strftime("%Y-%m-%d"),
        "sealed_date_max": sealed[date_column].max().strftime("%Y-%m-%d"),
        "lightgbm_rounds": lgb_rounds,
        "xgboost_rounds": int(config["sealed"]["xgboost_rounds"]),
        "primary_endpoint": "primary_lightgbm_5seed_mean.raw_pr_auc",
        "metrics": metrics,
        "no_tuning_performed": True,
        "sealed_reuse_forbidden": True,
    }
    atomic_json(result, sealed_dir / "SEALED_RESULT.json")
    atomic_json(result, consumed_marker)
    return result
