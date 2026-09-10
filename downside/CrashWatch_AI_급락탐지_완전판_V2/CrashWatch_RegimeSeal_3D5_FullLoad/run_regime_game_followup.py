from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, nvml_snapshot, release_lock
from cwregime.engine import _lgb_bundle_worker, _xgb_bundle_worker, make_bundle_plan
from cwregime.regimes import REGIMES
from cwregime.scoring import horizon_positive_recall, metrics_by_regime


PROFILES = ["P2_DEDUP_CLEAN", "P7_CORR095_PLUS_CONDITIONAL"]


def _set_game_mode(cpu_threads: int) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    available = list(process.cpu_affinity())
    selected = available[-max(1, min(int(cpu_threads), len(available))):]
    process.cpu_affinity(selected)
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for key, value in {
        "OMP_NUM_THREADS": "4",
        "LIGHTGBM_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }.items():
        os.environ[key] = value
    return {
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity": selected,
        "priority": "below_normal",
        "ram_gb": psutil.virtual_memory().total / 1024**3,
        "gpu": nvml_snapshot(),
    }


def _resource_monitor(stop: threading.Event, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    while not stop.wait(5.0):
        vm = psutil.virtual_memory()
        rows.append({
            "epoch": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_available_gb": vm.available / 1024**3,
            **nvml_snapshot(),
        })
        if len(rows) >= 12:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
            rows.clear()
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _overlap(a: int, b: int, c: int, d: int) -> bool:
    return not (b <= c or a >= d)


def _alternate_windows(calendar: pd.DataFrame, primary: pd.DataFrame, bank_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cal = calendar.sort_values("date_ns").reset_index(drop=True)
    dates = cal["date_ns"].to_numpy(dtype=np.int64)
    labels = cal["regime"].astype(str).to_numpy()
    split_ns = int(primary["start_ns"].min())
    start_bound = int(np.searchsorted(dates, split_ns, side="left"))
    end_bound = len(cal)
    purge = int(bank_cfg["purge_days"])
    min_train = int(bank_cfg["min_train_days"])
    width = int(bank_cfg["window_days"])
    min_target = int(bank_cfg["min_target_regime_days"])
    primary_intervals = [
        (
            int(np.searchsorted(dates, int(row.start_ns), side="left")),
            int(np.searchsorted(dates, int(row.end_ns), side="right")),
        )
        for row in primary.itertuples(index=False)
    ]
    occurrence = {regime: int(np.sum(labels[start_bound:end_bound] == regime)) for regime in REGIMES}
    order = sorted(REGIMES, key=lambda regime: (occurrence[regime], regime))
    occupied: list[tuple[int, int]] = []
    selected: list[dict[str, Any]] = []

    def candidates(regime: str, candidate_width: int, required: int, avoid_primary: bool):
        found = []
        for i in range(start_bound, end_bound - candidate_width + 1):
            j = i + candidate_width
            if i - purge - 1 < min_train:
                continue
            count = int(np.sum(labels[i:j] == regime))
            if count < required or any(_overlap(i, j, a, b) for a, b in occupied):
                continue
            if avoid_primary and any(_overlap(i, j, a, b) for a, b in primary_intervals):
                continue
            purity = count / candidate_width
            found.append((purity, i, j, count))
        found.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return found

    for regime in order:
        choice = None
        used_primary_overlap = False
        for candidate_width, required, avoid_primary in [
            (width, min_target, True),
            (max(8, width // 2), 1, True),
            (width, 1, False),
            (max(8, width // 2), 1, False),
        ]:
            options = candidates(regime, candidate_width, required, avoid_primary)
            if options:
                purity, i, j, count = options[0]
                choice = (i, j, count, purity)
                used_primary_overlap = not avoid_primary
                break
        if choice is None:
            raise RuntimeError(f"alternate window unavailable for {regime}; occurrence={occurrence[regime]}")
        i, j, count, purity = choice
        occupied.append((i, j))
        cutoff_i = i - purge - 1
        selected.append({
            "bank": "ALT_CONFIRM",
            "regime": regime,
            "window_id": f"ALT_CONFIRM_{regime}",
            "start_date": pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(dates[j - 1]).strftime("%Y-%m-%d"),
            "start_ns": int(dates[i]),
            "end_ns": int(dates[j - 1]),
            "target_regime_days": int(count),
            "total_days": int(j - i),
            "purity": float(purity),
            "train_cutoff_ns": int(dates[cutoff_i]),
            "train_cutoff_date": pd.Timestamp(dates[cutoff_i]).strftime("%Y-%m-%d"),
            "train_dates": int(cutoff_i + 1),
            "overlaps_primary_confirm": bool(used_primary_overlap),
        })
    selected.sort(key=lambda item: int(item["start_ns"]))
    return selected


def _config_map(config: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item["params"]) for item in config["search_space"][model]}


def _selected_configs(selection_output: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {"lightgbm": {}, "xgboost": {}}
    for model in result:
        table = pd.read_csv(selection_output / f"confirm_{model}_summary.csv")
        for profile in PROFILES:
            part = table[table["profile"] == profile].sort_values("selection_score", ascending=False)
            if part.empty:
                raise RuntimeError(f"no confirmed {model} config for {profile}")
            result[model][profile] = str(part.iloc[0]["config_id"])
    return result


def _paths(meta: dict[str, Any], profile: str) -> dict[str, str]:
    return {
        "matrix_path": str(meta["matrix_paths"][profile]),
        "target_path": str(meta["target_path"]),
        "valid_path": str(meta["valid_path"]),
        "dates_path": str(meta["dates_path"]),
        "first_hit_path": str(meta["first_hit_path"]),
    }


def _build_plans(
    output: Path,
    windows: list[dict[str, Any]],
    seeds: list[int],
    meta: dict[str, Any],
    config: dict[str, Any],
    selected: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookups = {model: _config_map(config, model) for model in ("lightgbm", "xgboost")}
    lgb: list[dict[str, Any]] = []
    xgb: list[dict[str, Any]] = []
    for profile in PROFILES:
        for window in windows:
            for seed in seeds:
                for model, destination, threads in (("lightgbm", lgb, 4), ("xgboost", xgb, 2)):
                    config_id = selected[model][profile]
                    plan = make_bundle_plan(
                        output,
                        "game_followup",
                        model,
                        profile,
                        config_id,
                        lookups[model][config_id],
                        window,
                        [int(seed)],
                        _paths(meta, profile),
                        threads,
                    )
                    plan["priority"] = "below_normal"
                    destination.append(plan)
    return lgb, xgb


def _run_lgb(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        results = list(executor.map(_lgb_bundle_worker, plans))
    failed = [item for item in results if item.get("status") != "complete"]
    if len(results) != len(plans) or failed:
        raise RuntimeError(f"LightGBM follow-up failed: count={len(results)}/{len(plans)}, failed={failed[:2]}")
    return results


def _is_cached(plan: dict[str, Any]) -> bool:
    cached = read_json(Path(plan["json_path"]), {})
    path = Path(plan["npz_path"])
    return bool(cached.get("status") == "complete" and path.exists() and path.stat().st_size > 0)


def _run_throttled_xgb(
    plans: list[dict[str, Any]],
    *,
    deadline_epoch: float,
    sleep_multiplier: float,
    stop_marker: Path,
    progress_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        if stop_marker.exists():
            raise RuntimeError("STOP_GAME_FOLLOWUP marker detected")
        if time.time() > deadline_epoch - 900.0:
            raise TimeoutError(f"analysis reserve reached after {len(results)}/{len(plans)} XGBoost bundles")
        cached_before = _is_cached(plan)
        active_started = time.time()
        result = _xgb_bundle_worker(plan)
        active_seconds = time.time() - active_started
        if result.get("status") != "complete":
            result = _xgb_bundle_worker(plan)
        if result.get("status") != "complete":
            raise RuntimeError(f"XGBoost follow-up failed: {result}")
        results.append(result)
        atomic_json({
            "status": "running",
            "completed": len(results),
            "expected": len(plans),
            "last_bundle": result.get("bundle_id"),
            "cached_last": cached_before,
            "deadline_epoch": deadline_epoch,
        }, progress_path)
        if not cached_before and index < len(plans):
            sleep_seconds = max(0.0, active_seconds * float(sleep_multiplier))
            sleep_until = min(time.time() + sleep_seconds, deadline_epoch - 900.0)
            while time.time() < sleep_until:
                if stop_marker.exists():
                    raise RuntimeError("STOP_GAME_FOLLOWUP marker detected during GPU cooldown")
                time.sleep(min(5.0, sleep_until - time.time()))
    atomic_json({"status": "completed", "completed": len(results), "expected": len(plans)}, progress_path)
    return results


def _frame_from_result(result: dict[str, Any], meta: dict[str, Any], date_to_regime: dict[int, str]) -> pd.DataFrame:
    dates = np.load(meta["dates_path"], mmap_mode="r")
    tickers = np.load(meta["tickers_path"], mmap_mode="r")
    with np.load(result["npz_path"]) as payload:
        idx = payload["val_idx"].astype(np.int64)
        predictions = payload["predictions"]
        if predictions.shape[0] != 1:
            raise RuntimeError(f"follow-up bundle must contain one seed: {result['bundle_id']}")
        frame = pd.DataFrame({
            "row_id": idx,
            "date_ns": np.asarray(dates[idx], dtype=np.int64),
            "ticker": np.asarray(tickers[idx]).astype(str),
            "target": payload["y"].astype(np.uint8),
            "first_hit_day": payload["first_hit"].astype(np.int8),
            "prediction": predictions[0].astype(float),
        })
    frame["date"] = pd.to_datetime(frame["date_ns"]).dt.strftime("%Y-%m-%d")
    frame["regime"] = frame["date_ns"].map(date_to_regime)
    frame["seed"] = int(result["seeds"][0])
    frame["profile"] = str(result["profile"])
    frame["window_id"] = str(result["window_id"])
    frame["bank"] = np.where(frame["window_id"].str.startswith("ALT_CONFIRM_"), "ALT_CONFIRM", "BASE_CONFIRM")
    return frame


def _frames(results: list[dict[str, Any]], meta: dict[str, Any], calendar: pd.DataFrame) -> pd.DataFrame:
    mapping = dict(zip(calendar["date_ns"].astype(np.int64), calendar["regime"].astype(str)))
    return pd.concat([_frame_from_result(item, meta, mapping) for item in results], ignore_index=True)


def _search_predictions(selection_output: Path, model: str, profile: str, config_id: str, meta: dict[str, Any], calendar: pd.DataFrame) -> pd.DataFrame:
    mapping = dict(zip(calendar["date_ns"].astype(np.int64), calendar["regime"].astype(str)))
    dates = np.load(meta["dates_path"], mmap_mode="r")
    parts = []
    for path in sorted((selection_output / "checkpoints" / "search" / model).glob("*.json")):
        result = read_json(path, {})
        if result.get("status") != "complete" or result.get("profile") != profile or result.get("config_id") != config_id:
            continue
        with np.load(result["npz_path"]) as payload:
            idx = payload["val_idx"].astype(np.int64)
            pred = payload["predictions"].mean(axis=0).astype(float)
            part = pd.DataFrame({
                "row_id": idx,
                "date_ns": np.asarray(dates[idx], dtype=np.int64),
                "target": payload["y"].astype(np.uint8),
                "prediction": pred,
            })
        part["date"] = pd.to_datetime(part["date_ns"]).dt.strftime("%Y-%m-%d")
        part["regime"] = part["date_ns"].map(mapping)
        parts.append(part)
    if not parts:
        raise RuntimeError(f"missing SEARCH predictions for {model}/{profile}/{config_id}")
    return pd.concat(parts, ignore_index=True).drop_duplicates("row_id")


def _fit_calibrators(frame: pd.DataFrame) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, Any]]:
    y = frame["target"].to_numpy(dtype=np.uint8)
    p = np.clip(frame["prediction"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    platt = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000, random_state=17)
    platt.fit(logits, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
    iso.fit(p, y)

    def raw(values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)

    def apply_platt(values: np.ndarray) -> np.ndarray:
        values = raw(values)
        values = np.log(values / (1.0 - values)).reshape(-1, 1)
        return platt.predict_proba(values)[:, 1]

    def apply_iso(values: np.ndarray) -> np.ndarray:
        return np.clip(iso.predict(raw(values)), 1e-6, 1.0 - 1e-6)

    audit = {
        "train_rows": int(len(y)),
        "train_positives": int(y.sum()),
        "train_positive_rate": float(y.mean()),
        "raw_mean_prediction": float(p.mean()),
        "platt_mean_prediction": float(apply_platt(p).mean()),
        "isotonic_mean_prediction": float(apply_iso(p).mean()),
        "platt_coef": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
    }
    return {"raw": raw, "platt": apply_platt, "isotonic": apply_iso}, audit


def _evaluate(
    lgb_frame: pd.DataFrame,
    xgb_frame: pd.DataFrame,
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = []
    regime_rows = []
    hit_rows = []
    ensembles: dict[tuple[str, str], pd.DataFrame] = {}
    for profile in PROFILES:
        for bank in ("BASE_CONFIRM", "ALT_CONFIRM"):
            seed_merged = []
            seeds = sorted(set(lgb_frame[(lgb_frame.profile == profile) & (lgb_frame.bank == bank)].seed))
            for seed in seeds:
                left = lgb_frame[(lgb_frame.profile == profile) & (lgb_frame.bank == bank) & (lgb_frame.seed == seed)]
                right = xgb_frame[(xgb_frame.profile == profile) & (xgb_frame.bank == bank) & (xgb_frame.seed == seed)]
                merged = left.merge(
                    right[["row_id", "prediction"]], on="row_id", suffixes=("_lgb", "_xgb"), validate="one_to_one"
                )
                if len(merged) != len(left) or len(merged) != len(right):
                    raise RuntimeError(f"prediction merge mismatch: {profile}/{bank}/seed={seed}")
                seed_merged.append(merged.assign(seed=int(seed)))
                for calibration in ("raw", "platt", "isotonic"):
                    lgb_p = calibrators[(profile, "lightgbm")][calibration](merged["prediction_lgb"].to_numpy())
                    xgb_p = calibrators[(profile, "xgboost")][calibration](merged["prediction_xgb"].to_numpy())
                    for weight in np.linspace(0.0, 1.0, 11):
                        evaluated = merged.copy()
                        evaluated["prediction"] = (1.0 - weight) * lgb_p + weight * xgb_p
                        score, by_regime = metrics_by_regime(evaluated)
                        rows.append({
                            "profile": profile,
                            "bank": bank,
                            "seed": int(seed),
                            "calibration": calibration,
                            "xgb_weight": float(weight),
                            "selection_score": score["selection_score"],
                            **{f"overall_{key}": value for key, value in score["overall"].items()},
                            "regime_pr_lift_geometric_mean": score.get("regime_pr_lift_geometric_mean"),
                            "worst_regime_pr_lift": score.get("worst_regime_pr_lift"),
                            "top3_precision_lift": score.get("top3_precision_lift"),
                        })
                        by_regime.insert(0, "profile", profile)
                        by_regime.insert(1, "bank", bank)
                        by_regime.insert(2, "seed", int(seed))
                        by_regime.insert(3, "calibration", calibration)
                        by_regime.insert(4, "xgb_weight", float(weight))
                        regime_rows.append(by_regime)
                        hit = horizon_positive_recall(evaluated)
                        hit.insert(0, "profile", profile)
                        hit.insert(1, "bank", bank)
                        hit.insert(2, "seed", int(seed))
                        hit.insert(3, "calibration", calibration)
                        hit.insert(4, "xgb_weight", float(weight))
                        hit_rows.append(hit)
            all_seeds = pd.concat(seed_merged, ignore_index=True)
            keys = ["row_id", "date_ns", "ticker", "target", "first_hit_day", "date", "regime", "window_id", "bank"]
            ensembles[(profile, bank)] = all_seeds.groupby(keys, as_index=False).agg(
                prediction_lgb=("prediction_lgb", "mean"),
                prediction_xgb=("prediction_xgb", "mean"),
            )
    metrics = pd.DataFrame(rows)
    regimes = pd.concat(regime_rows, ignore_index=True)
    hits = pd.concat(hit_rows, ignore_index=True)
    metrics.to_csv(output / "FOLLOWUP_METRICS_BY_SEED_BANK.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(output / "FOLLOWUP_REGIME_METRICS.csv", index=False, encoding="utf-8-sig")
    hits.to_csv(output / "FOLLOWUP_HIT_DAY_RECALL.csv", index=False, encoding="utf-8-sig")
    return metrics, regimes, hits, ensembles


def _aggregate_metrics(metrics: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["profile", "bank", "calibration", "xgb_weight"]
    numeric = [column for column in metrics.columns if column not in groups + ["seed"]]
    grouped = metrics.groupby(groups)[numeric].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = ["_".join([part for part in column if part]) if isinstance(column, tuple) else column for column in grouped.columns]
    grouped["robust_selection_score"] = grouped["selection_score_mean"] - 0.5 * grouped["selection_score_std"].fillna(0.0)
    grouped.to_csv(output / "FOLLOWUP_AGGREGATED_METRICS.csv", index=False, encoding="utf-8-sig")
    robust = grouped.sort_values("robust_selection_score", ascending=False).groupby(["profile", "bank"], as_index=False).first()
    robust.to_csv(output / "FOLLOWUP_ROBUST_SELECTIONS.csv", index=False, encoding="utf-8-sig")
    return grouped, robust


def _bootstrap(
    ensembles: dict[tuple[str, str], pd.DataFrame],
    calibrators: dict[tuple[str, str], dict[str, Callable[[np.ndarray], np.ndarray]]],
    robust: pd.DataFrame,
    output: Path,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260807)
    summaries = []
    paired = []
    for bank in ("BASE_CONFIRM", "ALT_CONFIRM"):
        robust_p2 = robust[(robust.profile == "P2_DEDUP_CLEAN") & (robust.bank == bank)].iloc[0]
        candidates = [
            ("P2_LOCKED_RAW_W09", "P2_DEDUP_CLEAN", "raw", 0.9),
            ("P2_RAW_W07", "P2_DEDUP_CLEAN", "raw", 0.7),
            ("P2_LGB_ONLY", "P2_DEDUP_CLEAN", "raw", 0.0),
            ("P2_XGB_ONLY", "P2_DEDUP_CLEAN", "raw", 1.0),
            ("P7_RAW_W09", "P7_CORR095_PLUS_CONDITIONAL", "raw", 0.9),
            ("P2_ROBUST", "P2_DEDUP_CLEAN", str(robust_p2.calibration), float(robust_p2.xgb_weight)),
        ]
        prepared: dict[str, pd.DataFrame] = {}
        for name, profile, calibration, weight in candidates:
            frame = ensembles[(profile, bank)].copy()
            lgb_p = calibrators[(profile, "lightgbm")][calibration](frame.prediction_lgb.to_numpy())
            xgb_p = calibrators[(profile, "xgboost")][calibration](frame.prediction_xgb.to_numpy())
            frame["prediction"] = (1.0 - weight) * lgb_p + weight * xgb_p
            prepared[name] = frame
        unique_dates = np.asarray(sorted(prepared[candidates[0][0]]["date"].unique()))
        scores = {name: [] for name, *_ in candidates}
        for rep in range(int(reps)):
            sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
            for name, *_ in candidates:
                source = prepared[name]
                pieces = []
                for occurrence, date in enumerate(sampled):
                    part = source[source["date"] == date].copy()
                    part["date"] = (pd.Timestamp("2000-01-01") + pd.Timedelta(days=occurrence)).strftime("%Y-%m-%d")
                    pieces.append(part)
                sample = pd.concat(pieces, ignore_index=True)
                score, _ = metrics_by_regime(sample)
                scores[name].append(float(score["selection_score"]))
            paired.append({
                "bank": bank,
                "rep": rep,
                "p2_minus_p7_locked_raw_w09": scores["P2_LOCKED_RAW_W09"][-1] - scores["P7_RAW_W09"][-1],
                "p2_robust_minus_locked": scores["P2_ROBUST"][-1] - scores["P2_LOCKED_RAW_W09"][-1],
            })
        for name, *_ in candidates:
            values = np.asarray(scores[name], dtype=float)
            summaries.append({
                "bank": bank,
                "candidate": name,
                "reps": int(len(values)),
                "mean_selection_score": float(np.mean(values)),
                "std_selection_score": float(np.std(values, ddof=1)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            })
    summary = pd.DataFrame(summaries)
    paired_df = pd.DataFrame(paired)
    summary.to_csv(output / "FOLLOWUP_DATE_BOOTSTRAP.csv", index=False, encoding="utf-8-sig")
    paired_df.groupby("bank").agg(
        p2_minus_p7_mean=("p2_minus_p7_locked_raw_w09", "mean"),
        p2_minus_p7_positive_rate=("p2_minus_p7_locked_raw_w09", lambda x: float(np.mean(np.asarray(x) > 0))),
        robust_minus_locked_mean=("p2_robust_minus_locked", "mean"),
        robust_minus_locked_positive_rate=("p2_robust_minus_locked", lambda x: float(np.mean(np.asarray(x) > 0))),
    ).reset_index().to_csv(output / "FOLLOWUP_BOOTSTRAP_PAIRED_DELTAS.csv", index=False, encoding="utf-8-sig")
    return summary, paired_df


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export(package_root: Path, output: Path, status: dict[str, Any]) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_RegimeSeal_GameFollowup_RESULTS_{stamp}.zip"
    files = [path for path in sorted(output.iterdir()) if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".log"}]
    code = [
        package_root / "run_regime_game_followup.py",
        package_root / "config_regime_3d5_full_load.json",
        package_root / "cwregime" / "engine.py",
        package_root / "cwregime" / "scoring.py",
    ]
    start_here = "\n".join([
        "# CrashWatch RegimeSeal game-mode follow-up",
        "",
        f"- status: {status.get('status')}",
        f"- new model fits: {status.get('actual_model_fits')}/{status.get('expected_model_fits')}",
        "- purpose: seed/window/calibration robustness audit; original locked recipe was not overwritten.",
        "- sealed data used: false",
        "",
        "Read FOLLOWUP_REVIEW.json, FOLLOWUP_ROBUST_SELECTIONS.csv,",
        "FOLLOWUP_BOOTSTRAP_PAIRED_DELTAS.csv, and FOLLOWUP_REGIME_METRICS.csv first.",
        "",
    ])
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start_here.encode("utf-8"))
        for path in files:
            if path.name not in {"SUPERVISOR_FAILED.json", "STOP_GAME_FOLLOWUP"}:
                zipped.write(path, f"results/{path.name}")
        for path in code:
            if path.exists():
                zipped.write(path, f"code_context/{path.relative_to(package_root).as_posix()}")
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        names = set(zipped.namelist())
    required = {"00_START_HERE.md", "results/FINAL_GAME_FOLLOWUP_STATUS.json", "results/FOLLOWUP_REVIEW.json"}
    if bad is not None or not required.issubset(names):
        raise RuntimeError(f"follow-up ZIP invalid: bad={bad}, missing={sorted(required - names)}")
    digest = _sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256.txt")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {"archive": str(archive), "size_bytes": archive.stat().st_size, "sha256": digest, "entries": len(names), "verified": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    selection_output = Path(args.selection_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "game_followup.lock.json"
    acquire_lock(lock_path)
    stop_event = threading.Event()
    monitor: threading.Thread | None = None
    started = time.time()
    try:
        budget_path = output / "GAME_BUDGET.json"
        budget = read_json(budget_path, {})
        if not budget:
            budget = {"started_epoch": started, "deadline_epoch": started + float(args.budget_seconds), "budget_seconds": float(args.budget_seconds)}
            atomic_json(budget, budget_path)
        deadline = float(budget["deadline_epoch"])
        hardware = _set_game_mode(int(args.cpu_threads))
        hardware.update({"gpu_policy": "one worker, one seed per burst, cooldown multiplier", "gpu_cooldown_multiplier": float(args.gpu_sleep_multiplier)})
        atomic_json(hardware, output / "GAME_MODE_HARDWARE.json")
        monitor = threading.Thread(target=_resource_monitor, args=(stop_event, output / "game_resource_usage.csv"), daemon=True)
        monitor.start()

        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        meta = read_json(selection_output / "RUNTIME_MANIFEST.json", {})
        if not meta or not Path(meta.get("target_path", "")).exists():
            raise RuntimeError("selection runtime manifest/cache is unavailable")
        if (selection_output / "final_sealed" / "FINAL_SEALED_CONSUMED.json").exists():
            raise RuntimeError("follow-up refused after FINAL_SEALED was consumed")
        calendar = pd.read_csv(selection_output / "regime_calendar.csv")
        primary = pd.read_csv(selection_output / "regime_windows_confirm.csv")
        alternate = _alternate_windows(calendar, primary, config["banks"])
        pd.DataFrame(alternate).to_csv(output / "ALTERNATE_CONFIRM_WINDOWS.csv", index=False, encoding="utf-8-sig")
        primary_records = primary.to_dict("records")
        windows = primary_records + alternate
        selected = _selected_configs(selection_output)
        seeds = [int(value) for value in str(args.seeds).split(",") if value.strip()]
        lgb_plans, xgb_plans = _build_plans(output, windows, seeds, meta, config, selected)
        atomic_json({
            "status": "planned",
            "profiles": PROFILES,
            "selected_configs": selected,
            "new_seeds": seeds,
            "primary_windows": len(primary_records),
            "alternate_windows": len(alternate),
            "lightgbm_bundles": len(lgb_plans),
            "xgboost_bundles": len(xgb_plans),
            "expected_model_fits": len(lgb_plans) + len(xgb_plans),
            "sealed_data_used": False,
            "locked_recipe_overwritten": False,
        }, output / "FOLLOWUP_PLAN.json")
        atomic_json({
            "status": "preflight_completed",
            "reviewed_source": str(selection_output),
            "original_result": {
                "locked_profile": "P2_DEDUP_CLEAN",
                "locked_lgb_config": "L04_159_REG",
                "locked_xgb_config": "X08_CW075",
                "locked_xgb_weight": 0.9,
                "selection_score": 2.023042196425134,
                "overall_pr_auc": 0.20968156994260145,
                "overall_pr_lift": 1.938284237821592,
                "worst_regime_pr_lift": 1.311022006300723,
                "mean_prediction": 0.267876,
                "positive_rate": 0.108179,
            },
            "bottlenecks": [
                "BULL_HIGH_VOL discrimination: ROC-AUC 0.567970 and PR lift 1.311022",
                "probability over-estimation: mean prediction 0.267876 versus positive rate 0.108179",
                "CRASH_STRESS uncertainty: only 8 dates and 19 positives in original confirm",
                "blend weight 0.7 versus 0.9 selection-score gap only 0.001704",
                "LightGBM search-to-confirm rank instability for P2",
            ],
            "followup_controls": {
                "new_seeds": seeds,
                "primary_windows": len(primary_records),
                "alternate_windows": len(alternate),
                "alternate_overlaps_primary": int(sum(bool(item["overlaps_primary_confirm"]) for item in alternate)),
                "calibration_fit_scope": "SEARCH only",
                "cpu_affinity": hardware.get("affinity"),
                "priority": hardware.get("priority"),
                "gpu_policy": hardware.get("gpu_policy"),
                "sealed_data_used": False,
            },
        }, output / "ORIGINAL_RESULT_REVIEW.json")

        lgb_results = _run_lgb(lgb_plans)
        atomic_json({"status": "completed", "results": lgb_results}, output / "LGB_STAGE_STATUS.json")
        xgb_results = _run_throttled_xgb(
            xgb_plans,
            deadline_epoch=deadline,
            sleep_multiplier=float(args.gpu_sleep_multiplier),
            stop_marker=output / "STOP_GAME_FOLLOWUP",
            progress_path=output / "XGB_STAGE_STATUS.json",
        )

        calibration_audit = {}
        calibrators = {}
        for profile in PROFILES:
            for model in ("lightgbm", "xgboost"):
                search = _search_predictions(selection_output, model, profile, selected[model][profile], meta, calendar)
                funcs, audit = _fit_calibrators(search)
                calibrators[(profile, model)] = funcs
                calibration_audit[f"{profile}/{model}"] = audit
        atomic_json({"fit_scope": "SEARCH only", "applied_scope": "BASE_CONFIRM and ALT_CONFIRM", "models": calibration_audit}, output / "TRAIN_ONLY_CALIBRATION_AUDIT.json")

        lgb_frame = _frames(lgb_results, meta, calendar)
        xgb_frame = _frames(xgb_results, meta, calendar)
        metrics, regimes, hits, ensembles = _evaluate(lgb_frame, xgb_frame, calibrators, output)
        aggregate, robust = _aggregate_metrics(metrics, output)
        bootstrap, paired = _bootstrap(ensembles, calibrators, robust, output, int(args.bootstrap_reps))

        bull = regimes[regimes["regime"] == "BULL_HIGH_VOL"].groupby(["profile", "bank", "calibration", "xgb_weight"], as_index=False).agg(
            raw_roc_auc_mean=("raw_roc_auc", "mean"),
            raw_pr_auc_lift_mean=("raw_pr_auc_lift", "mean"),
            top3_recall_mean=("top_3pct_recall", "mean"),
        )
        bull.to_csv(output / "BULL_HIGH_VOL_BOTTLENECK.csv", index=False, encoding="utf-8-sig")
        expected = len(lgb_plans) + len(xgb_plans)
        actual = len(lgb_results) + len(xgb_results)
        review = {
            "status": "completed" if actual == expected else "incomplete",
            "original_findings": {
                "locked_profile": "P2_DEDUP_CLEAN",
                "p7_minus_p2_selection_score": -0.05193288592574574,
                "locked_mean_prediction": 0.267876,
                "confirm_positive_rate": 0.108179,
                "bull_high_vol_roc_auc": 0.567970,
                "crash_stress_unique_dates": 8,
                "crash_stress_positives": 19,
                "blend_07_vs_09_score_gap": 0.001704,
            },
            "followup_scope": "new seeds on primary and alternate development windows; SEARCH-only calibration; date bootstrap",
            "expected_model_fits": expected,
            "actual_model_fits": actual,
            "robust_selection_file": "FOLLOWUP_ROBUST_SELECTIONS.csv",
            "bootstrap_file": "FOLLOWUP_BOOTSTRAP_PAIRED_DELTAS.csv",
            "bull_high_vol_file": "BULL_HIGH_VOL_BOTTLENECK.csv",
            "sealed_data_used": False,
            "original_locked_recipe_overwritten": False,
        }
        atomic_json(review, output / "FOLLOWUP_REVIEW.json")
        final = {
            "status": review["status"],
            "started_epoch": float(budget["started_epoch"]),
            "completed_epoch": time.time(),
            "elapsed_seconds_this_attempt": time.time() - started,
            "budget_seconds": float(budget["budget_seconds"]),
            "deadline_epoch": deadline,
            "expected_model_fits": expected,
            "actual_model_fits": actual,
            "lightgbm_bundles": len(lgb_results),
            "xgboost_bundles": len(xgb_results),
            "new_seeds": seeds,
            "primary_windows": len(primary_records),
            "alternate_windows": len(alternate),
            "sealed_data_used": False,
            "original_locked_recipe_overwritten": False,
        }
        atomic_json(final, output / "FINAL_GAME_FOLLOWUP_STATUS.json")
        if final["status"] != "completed":
            raise RuntimeError(f"follow-up completion audit failed: {final}")
        export = _export(package_root, output, final)
        final["result_export"] = export
        atomic_json(final, output / "FINAL_GAME_FOLLOWUP_STATUS.json")
        return final
    finally:
        stop_event.set()
        if monitor is not None:
            monitor.join(timeout=10)
        release_lock(lock_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="CrashWatch RegimeSeal game-mode robustness follow-up")
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--gpu-sleep-multiplier", type=float, default=3.0)
    parser.add_argument("--budget-seconds", type=float, default=18000.0)
    parser.add_argument("--seeds", default="307,701,997")
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    raise SystemExit(main())
