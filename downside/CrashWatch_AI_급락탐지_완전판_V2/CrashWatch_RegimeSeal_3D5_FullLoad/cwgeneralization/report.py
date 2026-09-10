from __future__ import annotations

import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from cw7h.utils import atomic_json, read_json
from cwfull.common import file_sha256
from .core import (
    PlattCalibrator,
    atomic_csv,
    atomic_text,
    benjamini_hochberg,
    binary_metrics,
    canonical_hash,
    choose_balanced_threshold,
    choose_threshold_for_recall,
    date_block_bootstrap_indices,
    distribution_diagnostics,
)
from .engine import load_checkpoint_predictions


def _metadata_arrays(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "dates": np.load(meta["dates_path"], mmap_mode="r"),
        "tickers": np.load(meta["tickers_path"], mmap_mode="r").astype(str),
        "regimes": np.load(meta["regime_code_path"], mmap_mode="r"),
        "sectors": np.load(meta["sector_code_path"], mmap_mode="r"),
    }


def _base_store(
    candidates: list[dict[str, Any]],
    fold_ids: list[int],
    lgb_results: list[dict[str, Any]],
    xgb_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    store: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for candidate in candidates:
        cid = candidate["id"]
        weight = float(config["models"][candidate["profile"]]["xgb_blend_weight"])
        store[cid] = {}
        for fold_id in fold_ids:
            lgb = load_checkpoint_predictions(lgb_results, cid, fold_id, "lightgbm")
            xgb = load_checkpoint_predictions(xgb_results, cid, fold_id, "xgboost")
            for key in ("val_idx", "y", "first_hit", "seeds"):
                if not np.array_equal(lgb[key], xgb[key]):
                    raise RuntimeError(f"LGB/XGB mismatch {cid}/fold{fold_id}/{key}")
            store[cid][fold_id] = {
                "val_idx": lgb["val_idx"], "y": lgb["y"], "first_hit": lgb["first_hit"],
                "seeds": lgb["seeds"],
                "predictions": ((1.0 - weight) * lgb["predictions"] + weight * xgb["predictions"]).astype(np.float32),
            }
    return store


def _add_derived(
    store: dict[str, dict[int, dict[str, np.ndarray]]],
    derived: list[dict[str, Any]],
    fold_ids: list[int],
) -> None:
    for candidate in derived:
        cid = candidate["id"]
        members = list(candidate["members"])
        weights = np.asarray(candidate["weights"], dtype=float)
        weights /= weights.sum()
        if not all(member in store for member in members):
            continue
        store[cid] = {}
        for fold_id in fold_ids:
            first = store[members[0]][fold_id]
            values = []
            for member in members:
                payload = store[member][fold_id]
                if not np.array_equal(payload["val_idx"], first["val_idx"]):
                    raise RuntimeError(f"derived candidate row mismatch {cid}/{fold_id}")
                values.append(payload["predictions"])
            prediction = np.tensordot(weights, np.stack(values), axes=(0, 0)).astype(np.float32)
            store[cid][fold_id] = {**{key: first[key] for key in ("val_idx", "y", "first_hit", "seeds")}, "predictions": prediction}


def _frame_for(
    candidate: str,
    store: dict[str, dict[int, dict[str, np.ndarray]]],
    fold_ids: list[int],
    meta: dict[str, Any],
    *,
    seed_index: int | None = None,
) -> pd.DataFrame:
    arrays = _metadata_arrays(meta)
    parts = []
    regime_names = list(meta["regimes"]); sector_names = list(meta["sectors"])
    for fold_id in fold_ids:
        payload = store[candidate][fold_id]
        idx = payload["val_idx"].astype(np.int64)
        prediction = payload["predictions"].mean(axis=0) if seed_index is None else payload["predictions"][seed_index]
        parts.append(pd.DataFrame({
            "row_id": idx, "date": pd.to_datetime(arrays["dates"][idx]), "ticker": arrays["tickers"][idx],
            "sector": [sector_names[int(value)] for value in arrays["sectors"][idx]],
            "regime": [regime_names[int(value)] for value in arrays["regimes"][idx]],
            "target": payload["y"].astype(np.uint8), "first_hit_day": payload["first_hit"].astype(np.int8),
            "prediction": prediction.astype(float), "fold_id": int(fold_id), "candidate": candidate,
        }))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def _ticker_table(frame: pd.DataFrame, config: dict[str, Any], candidate: str, scope: str) -> pd.DataFrame:
    minimum = config["evaluation"]
    weak = set(config["weak_groups"]["WEAK_EVALUABLE"])
    low = set(config["weak_groups"]["LOW_EVENT_SUPPORT"])
    rows = []
    for ticker, part in frame.groupby("ticker", sort=True):
        metrics = binary_metrics(part)
        evaluable = (
            metrics["rows"] >= int(minimum["min_ticker_rows"])
            and metrics["positives"] >= int(minimum["min_ticker_positives"])
            and metrics["negatives"] >= int(minimum["min_ticker_negatives"])
        )
        passed = bool(
            evaluable
            and metrics["pr_auc_lift"] >= float(config["goals"]["primary_pr_auc_lift_min"])
            and metrics["roc_auc"] >= float(config["goals"]["primary_roc_auc_min"])
        )
        status = "MET" if passed else "NOT_MET" if evaluable else "NOT_EVALUABLE"
        group = "WEAK_EVALUABLE" if ticker in weak else "LOW_EVENT_SUPPORT" if ticker in low else "STRONG"
        identity = config.get("ticker_identity", {}).get(ticker, {})
        rows.append({"scope": scope, "candidate": candidate, "ticker": ticker, "name": identity.get("name", ""),
                     "sector": part["sector"].iloc[0], "ticker_group": group, **metrics,
                     "primary_status": status})
    return pd.DataFrame(rows)


def _group_summary(ticker_table: pd.DataFrame, group: str) -> dict[str, float]:
    part = ticker_table[(ticker_table["ticker_group"] == group) & (ticker_table["primary_status"] != "NOT_EVALUABLE")]
    return {
        "ticker_count": int(len(part)),
        "mean_roc_auc": float(part["roc_auc"].mean()) if len(part) else float("nan"),
        "mean_pr_auc_lift": float(part["pr_auc_lift"].mean()) if len(part) else float("nan"),
        "pass_count": int((part["primary_status"] == "MET").sum()),
        "pass_ratio": float((part["primary_status"] == "MET").mean()) if len(part) else float("nan"),
    }


def _regime_table(frame: pd.DataFrame, candidate: str, scope: str, **metric_kwargs: Any) -> pd.DataFrame:
    rows = []
    for regime in sorted(frame["regime"].unique()):
        rows.append({"scope": scope, "candidate": candidate, "regime": regime,
                     **binary_metrics(frame[frame["regime"] == regime], **metric_kwargs)})
    return pd.DataFrame(rows)


def _sector_table(frame: pd.DataFrame, candidate: str, scope: str, **metric_kwargs: Any) -> pd.DataFrame:
    rows = []
    for sector in sorted(frame["sector"].unique()):
        rows.append({"scope": scope, "candidate": candidate, "sector": sector,
                     **binary_metrics(frame[frame["sector"] == sector], **metric_kwargs)})
    return pd.DataFrame(rows)


def _objective(overall: dict[str, Any], tickers: pd.DataFrame, regimes: pd.DataFrame) -> float:
    weak = _group_summary(tickers, "WEAK_EVALUABLE")
    evaluable = tickers[tickers["primary_status"] != "NOT_EVALUABLE"]
    pass_ratio = float((evaluable["primary_status"] == "MET").mean()) if len(evaluable) else 0.0
    worst_regime = float(regimes["roc_auc"].min()) if len(regimes) else 0.5
    return float(
        0.30 * overall["roc_auc"]
        + 0.20 * min(overall["pr_auc_lift"], 2.0) / 2.0
        + 0.20 * weak["mean_roc_auc"]
        + 0.15 * worst_regime
        + 0.15 * pass_ratio
    )


def selection_analysis(
    package: Path,
    output: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    lgb_results: list[dict[str, Any]],
    xgb_results: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_ids = list(map(int, config["evaluation"]["selection_folds"]))
    store = _base_store(candidates, fold_ids, lgb_results, xgb_results, config)
    _add_derived(store, config.get("derived_candidates", []), fold_ids)
    config["ticker_identity"] = meta["ticker_identity"]
    rows = []; fold_rows = []; ticker_parts = []; regime_parts = []; seed_rows = []
    all_candidates = [*candidates, *[row for row in config.get("derived_candidates", []) if row["id"] in store]]
    eligible_lookup = {row["id"]: bool(row.get("eligible", True)) for row in all_candidates}
    frames: dict[str, pd.DataFrame] = {}
    for candidate in all_candidates:
        cid = candidate["id"]
        frame = _frame_for(cid, store, fold_ids, meta)
        frames[cid] = frame
        overall = binary_metrics(frame)
        ticker = _ticker_table(frame, config, cid, "SELECTION")
        regime = _regime_table(frame, cid, "SELECTION")
        ticker_parts.append(ticker); regime_parts.append(regime)
        weak_summary = _group_summary(ticker, "WEAK_EVALUABLE")
        strong_summary = _group_summary(ticker, "STRONG")
        evaluable = ticker[ticker.primary_status != "NOT_EVALUABLE"]
        row = {
            "scope": "SELECTION", "candidate": cid, "eligible": eligible_lookup[cid], **overall,
            "evaluable_ticker_count": int(len(evaluable)),
            "primary_pass_count": int((evaluable.primary_status == "MET").sum()),
            "primary_pass_ratio": float((evaluable.primary_status == "MET").mean()),
            "mean_ticker_pr_auc_lift": float(evaluable.pr_auc_lift.mean()),
            "mean_ticker_roc_auc": float(evaluable.roc_auc.mean()),
            "median_ticker_roc_auc": float(evaluable.roc_auc.median()),
            "weak_mean_roc_auc": weak_summary["mean_roc_auc"], "weak_pass_count": weak_summary["pass_count"],
            "strong_mean_pr_auc_lift": strong_summary["mean_pr_auc_lift"],
            "worst_regime_roc_auc": float(regime.roc_auc.min()),
        }
        row["selection_objective"] = _objective(overall, ticker, regime)
        rows.append(row)
        for fold_id in fold_ids:
            part = frame[frame.fold_id == fold_id]
            fold_rows.append({"scope": "SELECTION", "candidate": cid, "fold_id": fold_id, **binary_metrics(part)})
        for seed_index, seed in enumerate(config["seeds"]):
            seed_frame = _frame_for(cid, store, fold_ids, meta, seed_index=seed_index)
            seed_rows.append({"scope": "SELECTION", "candidate": cid, "seed": int(seed), **binary_metrics(seed_frame)})
    comparison = pd.DataFrame(rows)
    champion = comparison[comparison.candidate == "CHAMPION_CURRENT"].iloc[0]
    comparison["delta_pr_auc_lift_vs_champion"] = comparison.pr_auc_lift - champion.pr_auc_lift
    comparison["delta_roc_auc_vs_champion"] = comparison.roc_auc - champion.roc_auc
    comparison["delta_weak_roc_vs_champion"] = comparison.weak_mean_roc_auc - champion.weak_mean_roc_auc
    comparison["pareto_dominated_by_champion"] = (
        comparison.pr_auc_lift.lt(champion.pr_auc_lift) & comparison.roc_auc.lt(champion.roc_auc)
    )
    fold_table = pd.DataFrame(fold_rows)
    pvalues = []
    champion_folds = fold_table[fold_table.candidate == "CHAMPION_CURRENT"].sort_values("fold_id")
    for cid in comparison.candidate:
        if cid == "CHAMPION_CURRENT":
            pvalues.append(float("nan")); continue
        other = fold_table[fold_table.candidate == cid].sort_values("fold_id")
        try:
            pvalues.append(float(wilcoxon(other.roc_auc.to_numpy(), champion_folds.roc_auc.to_numpy(), alternative="greater").pvalue))
        except Exception:
            pvalues.append(float("nan"))
    comparison["roc_improvement_pvalue"] = pvalues
    comparison["roc_improvement_bh_qvalue"] = benjamini_hochberg(pvalues)
    comparison["finalist_eligible"] = comparison.eligible & ~comparison.pareto_dominated_by_champion
    finalists = ["CHAMPION_CURRENT"]
    challengers = comparison[(comparison.candidate != "CHAMPION_CURRENT") & comparison.finalist_eligible].sort_values(
        ["selection_objective", "weak_mean_roc_auc", "roc_auc"], ascending=False
    )
    finalists.extend(challengers.head(2).candidate.tolist())
    comparison["selected_finalist"] = comparison.candidate.isin(finalists)
    comparison = comparison.sort_values(["selected_finalist", "selection_objective"], ascending=[False, False])
    atomic_csv(comparison, output / "candidate_comparison_selection.csv")
    atomic_csv(fold_table, output / "fold_stability_selection.csv")
    atomic_csv(pd.DataFrame(seed_rows), output / "seed_stability_selection.csv")
    atomic_csv(pd.concat(ticker_parts, ignore_index=True), output / "ticker_metrics_selection.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "regime_generalization_selection.csv")
    atomic_csv(comparison[["candidate", "pr_auc_lift", "roc_auc", "pareto_dominated_by_champion", "selected_finalist"]], output / "roc_pr_pareto_frontier.csv")
    result = {
        "status": "SELECTION_FROZEN_BEFORE_CONFIRMATION",
        "finalists": finalists,
        "selection_folds": fold_ids,
        "confirmation_folds_read": [],
        "selection_rows": int(len(next(iter(frames.values())))),
        "selection_hash": canonical_hash(comparison.fillna("NaN").to_dict("records"), 32),
    }
    atomic_json(result, output / "SELECTION_FINALISTS.json")
    return {"result": result, "store": store, "frames": frames, "comparison": comparison, "fold_table": fold_table}


def confirmation_dependencies(finalists: list[str], config: dict[str, Any]) -> list[str]:
    derived = {row["id"]: row for row in config.get("derived_candidates", [])}
    result = {"CHAMPION_CURRENT"}
    for cid in finalists:
        if cid in derived:
            result.update(derived[cid]["members"])
        else:
            result.add(cid)
    return sorted(result)


def _calibrated_frame(frame: pd.DataFrame, calibrator: PlattCalibrator) -> pd.DataFrame:
    out = frame.copy()
    out["risk_probability"] = calibrator.predict(out.prediction.to_numpy(dtype=float))
    return out


def _nanmean_columns(matrix: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    totals = np.where(finite, matrix, 0.0).sum(axis=0, dtype=np.float64)
    means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means


def _weak_reinforcement(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> pd.DataFrame:
    rows = []
    for ticker in config["weak_groups"]["WEAK_EVALUABLE"]:
        base = champion[champion.ticker == ticker]
        test = challenger[challenger.ticker == ticker]
        bm = binary_metrics(base, probability_column="risk_probability")
        tm = binary_metrics(test, probability_column="risk_probability")
        regimes = []
        for regime, part in base.groupby("regime"):
            metric = binary_metrics(part)
            regimes.append((metric["roc_auc"] if np.isfinite(metric["roc_auc"]) else 1.0, regime))
        worst_regime = min(regimes)[1] if regimes else "NOT_EVALUABLE"
        base_status = "MET" if bm["pr_auc_lift"] >= 1 and bm["roc_auc"] >= .5 else "NOT_MET"
        test_status = "MET" if tm["pr_auc_lift"] >= 1 and tm["roc_auc"] >= .5 else "NOT_MET"
        dates = test.date.to_numpy()
        deltas = []
        weak_bootstrap_reps = min(300, int(config["robustness"]["bootstrap_reps"]))
        for idx in date_block_bootstrap_indices(dates, weak_bootstrap_reps, 20, seed=20260808 + int(ticker)):
            try:
                b = binary_metrics(base.iloc[idx])
                t = binary_metrics(test.iloc[idx])
                deltas.append(t["roc_auc"] - b["roc_auc"])
            except Exception:
                continue
        rows.append({
            "ticker": ticker, "name": config["ticker_identity"][ticker]["name"],
            "champion_pr_lift": bm["pr_auc_lift"], "challenger_pr_lift": tm["pr_auc_lift"],
            "delta_pr_lift": tm["pr_auc_lift"] - bm["pr_auc_lift"],
            "champion_roc": bm["roc_auc"], "challenger_roc": tm["roc_auc"], "delta_roc": tm["roc_auc"] - bm["roc_auc"],
            "champion_status": base_status, "challenger_status": test_status, "regime_of_failure": worst_regime,
            "confidence_interval_low": float(np.nanquantile(deltas, .025)) if deltas else float("nan"),
            "confidence_interval_high": float(np.nanquantile(deltas, .975)) if deltas else float("nan"),
            "final_decision": "IMPROVED" if tm["roc_auc"] > bm["roc_auc"] else "NOT_IMPROVED",
        })
    table = pd.DataFrame(rows)
    atomic_csv(table, output / "weak_ticker_reinforcement.csv")
    return table


def _weak_diagnostics(
    champion: pd.DataFrame,
    config: dict[str, Any],
    meta: dict[str, Any],
    operating_threshold: float,
    output: Path,
) -> pd.DataFrame:
    p7_path = Path(meta["profile_paths"]["P7_CORR095_PLUS_CONDITIONAL"])
    p7_features = list(meta["profile_features"]["P7_CORR095_PLUS_CONDITIONAL"])
    matrix = np.load(p7_path, mmap_mode="r")
    all_idx = champion.row_id.to_numpy(dtype=np.int64)
    y_all = champion.target.to_numpy(dtype=np.uint8)
    x_all = np.asarray(matrix[all_idx], dtype=np.float32)
    global_delta = _nanmean_columns(x_all[y_all == 1]) - _nanmean_columns(x_all[y_all == 0])
    rows = []
    for ticker in config["weak_groups"]["WEAK_EVALUABLE"]:
        part = champion[champion.ticker == ticker].copy()
        y = part.target.to_numpy(dtype=np.uint8); score = part.prediction.to_numpy(dtype=float)
        probability = part.risk_probability.to_numpy(dtype=float)
        predicted = probability >= operating_threshold
        diag = distribution_diagnostics(score[y == 1], score[y == 0])
        idx = part.row_id.to_numpy(dtype=np.int64)
        x = np.asarray(matrix[idx], dtype=np.float32)
        local_delta = _nanmean_columns(x[y == 1]) - _nanmean_columns(x[y == 0]) if y.sum() and (y == 0).sum() else np.full(len(p7_features), np.nan)
        reversed_mask = np.isfinite(local_delta) & np.isfinite(global_delta) & (np.sign(local_delta) != np.sign(global_delta))
        divergence = np.abs(local_delta - global_delta)
        top = np.argsort(np.nan_to_num(divergence, nan=-1.0))[-10:][::-1]
        regime_rows = []
        for regime, block in part.groupby("regime"):
            metric = binary_metrics(block)
            if np.isfinite(metric["roc_auc"]): regime_rows.append((metric["roc_auc"], regime))
        rows.append({
            "ticker": ticker, "name": config["ticker_identity"][ticker]["name"], "rows": len(part),
            "positives": int(y.sum()), "true_positive": int(np.sum(predicted & (y == 1))),
            "false_positive": int(np.sum(predicted & (y == 0))), "true_negative": int(np.sum(~predicted & (y == 0))),
            "false_negative": int(np.sum(~predicted & (y == 1))), **diag,
            "worst_regime": min(regime_rows)[1] if regime_rows else "NOT_EVALUABLE",
            "reversed_feature_response_count": int(reversed_mask.sum()),
            "top_feature_response_differences": json.dumps([p7_features[index] for index in top], ensure_ascii=False),
            "core_tail_beta_120_reversed": bool(reversed_mask[p7_features.index("t_taildep_tail_beta_120")]) if "t_taildep_tail_beta_120" in p7_features else None,
            "core_down_corr_60_reversed": bool(reversed_mask[p7_features.index("t_taildep_down_corr_60")]) if "t_taildep_down_corr_60" in p7_features else None,
            "core_finshort_z20_reversed": bool(reversed_mask[p7_features.index("t_finshort_balance_z_20")]) if "t_finshort_balance_z_20" in p7_features else None,
        })
    table = pd.DataFrame(rows)
    atomic_csv(table, output / "weak_ticker_diagnostics.csv")
    return table


def _feature_stability(
    output: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    results_lgb: list[dict[str, Any]],
    results_xgb: list[dict[str, Any]],
) -> pd.DataFrame:
    features = list(meta["profile_features"]["P2_DEDUP_CLEAN"])
    weight = float(config["models"]["P2"]["xgb_blend_weight"])
    fold_importance: dict[int, np.ndarray] = {}
    for fold_id in [*config["evaluation"]["selection_folds"], *config["evaluation"]["confirmation_folds"]]:
        values = []
        for results, model, model_weight in ((results_lgb, "lightgbm", 1 - weight), (results_xgb, "xgboost", weight)):
            match = [row for row in results if row.get("candidate_id") == "CHAMPION_CURRENT" and int(row.get("fold_id", -1)) == fold_id]
            if len(match) != 1: continue
            with np.load(match[0]["npz_path"]) as payload:
                importance = payload["importance_gain"].mean(axis=0)[:len(features)].astype(float)
            importance = importance / importance.sum() if importance.sum() > 0 else importance
            values.append(model_weight * importance)
        if values:
            fold_importance[fold_id] = np.sum(values, axis=0)
    rows = []
    ids = sorted(fold_importance)
    for left_pos, left in enumerate(ids):
        for right in ids[left_pos + 1:]:
            a = fold_importance[left]; b = fold_importance[right]
            rows.append({
                "fold_left": left, "fold_right": right,
                "spearman_rank_correlation": float(spearmanr(a, b).statistic),
                "top20_overlap": len(set(np.argsort(a)[-20:]) & set(np.argsort(b)[-20:])),
                "top50_overlap": len(set(np.argsort(a)[-50:]) & set(np.argsort(b)[-50:])),
            })
    table = pd.DataFrame(rows)
    atomic_csv(table, output / "feature_stability.csv")
    if fold_importance:
        mean = np.mean(list(fold_importance.values()), axis=0)
        atomic_csv(pd.DataFrame({"feature": features, "mean_normalized_gain": mean}).sort_values("mean_normalized_gain", ascending=False), output / "feature_importance_champion.csv")
    return table


def final_analysis(
    package: Path,
    project: Path,
    output: Path,
    config: dict[str, Any],
    meta: dict[str, Any],
    selection: dict[str, Any],
    selection_candidates: list[dict[str, Any]],
    selection_lgb: list[dict[str, Any]],
    selection_xgb: list[dict[str, Any]],
    confirmation_candidates: list[dict[str, Any]],
    confirmation_lgb: list[dict[str, Any]],
    confirmation_xgb: list[dict[str, Any]],
    diagnostic_results: list[dict[str, Any]],
    permutation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    selection_ids = list(map(int, config["evaluation"]["selection_folds"]))
    confirmation_ids = list(map(int, config["evaluation"]["confirmation_folds"]))
    finalists = list(selection["result"]["finalists"])
    selection_store = selection["store"]
    confirmation_store = _base_store(confirmation_candidates, confirmation_ids, confirmation_lgb, confirmation_xgb, config)
    _add_derived(confirmation_store, config.get("derived_candidates", []), confirmation_ids)
    config["ticker_identity"] = meta["ticker_identity"]
    calibration_rows=[]; comparison_rows=[]; ticker_parts=[]; sector_parts=[]; regime_parts=[]; time_parts=[]; fold_rows=[]; seed_rows=[]
    calibrated: dict[str, dict[str, pd.DataFrame]] = {}
    policies: dict[str, Any] = {}
    for cid in finalists:
        selection_frame = _frame_for(cid, selection_store, selection_ids, meta)
        confirmation_frame = _frame_for(cid, confirmation_store, confirmation_ids, meta)
        calibrator = PlattCalibrator.fit(selection_frame.prediction, selection_frame.target)
        selection_frame = _calibrated_frame(selection_frame, calibrator)
        confirmation_frame = _calibrated_frame(confirmation_frame, calibrator)
        operating = choose_threshold_for_recall(selection_frame.target, selection_frame.risk_probability, config["goals"]["target_event_recall"])
        balanced = choose_balanced_threshold(selection_frame.target, selection_frame.risk_probability)
        policies[cid] = {"calibration": calibrator.to_dict(), "operating_threshold": operating, "balanced_threshold": balanced}
        calibrated[cid] = {"selection": selection_frame, "confirmation": confirmation_frame}
        kwargs = {"probability_column": "risk_probability", "balanced_threshold": balanced,
                  "operating_threshold": operating, "declared_probability_threshold": config["goals"]["declared_probability_threshold"]}
        for scope, frame in (("SELECTION", selection_frame), ("CONFIRMATION", confirmation_frame)):
            metrics = binary_metrics(frame, **kwargs)
            ticker = _ticker_table(frame, config, cid, scope)
            # Replace ticker calibration metrics with the frozen global risk probability diagnostics.
            ticker_parts.append(ticker)
            sector = _sector_table(frame, cid, scope, **kwargs); regime = _regime_table(frame, cid, scope, **kwargs)
            sector_parts.append(sector); regime_parts.append(regime)
            evaluable = ticker[ticker.primary_status != "NOT_EVALUABLE"]
            weak_summary = _group_summary(ticker, "WEAK_EVALUABLE"); strong_summary = _group_summary(ticker, "STRONG")
            comparison_rows.append({
                "scope": scope, "candidate": cid, **metrics,
                "evaluable_ticker_count": int(len(evaluable)), "primary_pass_count": int((evaluable.primary_status == "MET").sum()),
                "primary_pass_ratio": float((evaluable.primary_status == "MET").mean()),
                "mean_ticker_pr_auc_lift": float(evaluable.pr_auc_lift.mean()), "mean_ticker_roc_auc": float(evaluable.roc_auc.mean()),
                "median_ticker_roc_auc": float(evaluable.roc_auc.median()),
                "weak_mean_roc_auc": weak_summary["mean_roc_auc"], "weak_pass_count": weak_summary["pass_count"],
                "strong_mean_pr_auc_lift": strong_summary["mean_pr_auc_lift"], "worst_regime_roc_auc": float(regime.roc_auc.min()),
            })
            calibration_rows.append({"scope": scope, "candidate": cid, **calibrator.to_dict(),
                                     **{key: metrics[key] for key in ("brier", "brier_skill_vs_scope_prevalence", "logloss", "calibration_slope", "calibration_intercept", "ece_10bin", "probability_ge_070_count", "probability_ge_070_precision", "probability_ge_070_recall")}})
        for fold_id in confirmation_ids:
            part = confirmation_frame[confirmation_frame.fold_id == fold_id]
            fold_rows.append({"scope": "CONFIRMATION", "candidate": cid, "fold_id": fold_id, **binary_metrics(part, **kwargs)})
        for seed_index, seed in enumerate(config["seeds"]):
            seed_frame = _frame_for(cid, confirmation_store, confirmation_ids, meta, seed_index=seed_index)
            seed_frame = _calibrated_frame(seed_frame, calibrator)
            seed_rows.append({"scope": "CONFIRMATION", "candidate": cid, "seed": seed, **binary_metrics(seed_frame, **kwargs)})
        all_oof = pd.concat([selection_frame, confirmation_frame], ignore_index=True).sort_values("date")
        unique_dates = np.sort(all_oof.date.unique())
        for days in config["evaluation"]["time_windows_days"]:
            selected_dates = set(unique_dates[-int(days):])
            part = all_oof[all_oof.date.isin(selected_dates)]
            time_parts.append({"candidate": cid, "window_trading_days": int(days), "date_min": str(part.date.min().date()), "date_max": str(part.date.max().date()), **binary_metrics(part, **kwargs)})

    comparison = pd.DataFrame(comparison_rows)
    champion_confirm = comparison[(comparison.scope == "CONFIRMATION") & (comparison.candidate == "CHAMPION_CURRENT")].iloc[0]
    for column in ("pr_auc_lift", "roc_auc", "weak_mean_roc_auc", "strong_mean_pr_auc_lift", "worst_regime_roc_auc"):
        comparison[f"delta_{column}_vs_champion"] = comparison[column] - float(champion_confirm[column])
    atomic_csv(comparison, output / "candidate_comparison.csv")
    atomic_csv(pd.concat(ticker_parts, ignore_index=True), output / "ticker_metrics.csv")
    atomic_csv(pd.concat(sector_parts, ignore_index=True), output / "sector_metrics.csv")
    atomic_csv(pd.concat(regime_parts, ignore_index=True), output / "regime_generalization_report.csv")
    atomic_csv(pd.DataFrame(time_parts), output / "time_window_metrics.csv")
    seed_table = pd.concat([pd.read_csv(output / "seed_stability_selection.csv"), pd.DataFrame(seed_rows)], ignore_index=True)
    fold_table = pd.concat([pd.read_csv(output / "fold_stability_selection.csv"), pd.DataFrame(fold_rows)], ignore_index=True)
    atomic_csv(seed_table, output / "seed_stability.csv"); atomic_csv(fold_table, output / "fold_stability.csv")
    atomic_csv(pd.DataFrame(calibration_rows), output / "calibration_report.csv")

    challenger_ids = [cid for cid in finalists if cid != "CHAMPION_CURRENT"]
    best_challenger = max(challenger_ids, key=lambda cid: float(comparison[(comparison.scope == "CONFIRMATION") & (comparison.candidate == cid)].roc_auc)) if challenger_ids else "CHAMPION_CURRENT"
    reinforcement = _weak_reinforcement(calibrated["CHAMPION_CURRENT"]["confirmation"], calibrated[best_challenger]["confirmation"], config, output)
    weak_diagnostics = _weak_diagnostics(calibrated["CHAMPION_CURRENT"]["selection"], config, meta, policies["CHAMPION_CURRENT"]["operating_threshold"], output)

    overfit_rows=[]; winner_rows=[]
    for cid in finalists:
        selection_row = comparison[(comparison.scope == "SELECTION") & (comparison.candidate == cid)].iloc[0]
        confirm_row = comparison[(comparison.scope == "CONFIRMATION") & (comparison.candidate == cid)].iloc[0]
        winner_rows.append({"candidate": cid, "selection_pr_lift": selection_row.pr_auc_lift, "confirmation_pr_lift": confirm_row.pr_auc_lift,
                            "pr_lift_decay": confirm_row.pr_auc_lift - selection_row.pr_auc_lift,
                            "selection_roc": selection_row.roc_auc, "confirmation_roc": confirm_row.roc_auc,
                            "roc_decay": confirm_row.roc_auc - selection_row.roc_auc,
                            "risk": "WINNERS_CURSE_RISK" if confirm_row.roc_auc - selection_row.roc_auc < -.03 else "OK"})
        if cid != "CHAMPION_CURRENT":
            candidate_folds = fold_table[(fold_table.scope == "SELECTION") & (fold_table.candidate == cid)].sort_values("fold_id")
            champion_folds = fold_table[(fold_table.scope == "SELECTION") & (fold_table.candidate == "CHAMPION_CURRENT")].sort_values("fold_id")
            delta = candidate_folds.roc_auc.to_numpy() - champion_folds.roc_auc.to_numpy()
            total_abs = float(np.abs(delta).sum())
            concentration = float(np.max(np.abs(delta)) / total_abs) if total_abs else 0.0
            overfit_rows.append({"audit": "fold_concentration", "candidate": cid, "value": concentration,
                                 "status": "FOLD_CONCENTRATION_WARNING" if concentration > .5 else "PASS"})
            old = float(np.mean(delta[:3])); recent = float(np.mean(delta[3:]))
            status = "DIRECTION_CONSISTENT" if old * recent >= 0 else "RECENT_REGIME_DEPENDENT" if recent > 0 else "STALE_SIGNAL"
            overfit_rows.append({"audit": "recent_vs_old_consistency", "candidate": cid, "value": recent - old, "status": status})

    # Date-block bootstrap on the frozen best challenger versus champion.
    champion_frame = calibrated["CHAMPION_CURRENT"]["confirmation"].reset_index(drop=True)
    challenger_frame = calibrated[best_challenger]["confirmation"].reset_index(drop=True)
    boot_roc=[]; boot_pr=[]
    for indices in date_block_bootstrap_indices(champion_frame.date.to_numpy(), config["robustness"]["bootstrap_reps"], config["robustness"]["bootstrap_block_days"]):
        try:
            bm=binary_metrics(champion_frame.iloc[indices]); tm=binary_metrics(challenger_frame.iloc[indices])
            boot_roc.append(tm["roc_auc"]-bm["roc_auc"]); boot_pr.append(tm["pr_auc_lift"]-bm["pr_auc_lift"])
        except Exception: continue
    overfit_rows.append({"audit": "date_block_bootstrap_roc_delta", "candidate": best_challenger,
                         "value": float(np.nanmean(boot_roc)), "ci_low": float(np.nanquantile(boot_roc,.025)), "ci_high": float(np.nanquantile(boot_roc,.975)),
                         "status": "PASS" if np.nanquantile(boot_roc,.025) > -.02 else "UNCERTAIN"})
    overfit_rows.append({"audit": "date_block_bootstrap_pr_lift_delta", "candidate": best_challenger,
                         "value": float(np.nanmean(boot_pr)), "ci_low": float(np.nanquantile(boot_pr,.025)), "ci_high": float(np.nanquantile(boot_pr,.975)),
                         "status": "PASS" if np.nanquantile(boot_pr,.025) > -.03 else "UNCERTAIN"})

    # Reduced block-permutation sanity test and random-feature control.
    if permutation_results:
        permutation_roc=[]
        for result in permutation_results:
            with np.load(result["npz_path"]) as payload:
                idx=payload["val_idx"]; y=payload["y"]; p=payload["predictions"].mean(axis=0)
            frame=pd.DataFrame({"date":pd.to_datetime(np.load(meta["dates_path"],mmap_mode="r")[idx]),"target":y,"prediction":p})
            permutation_roc.append(binary_metrics(frame)["roc_auc"])
        actual_fold = selection["frames"]["CHAMPION_CURRENT"]
        actual_fold = actual_fold[actual_fold.fold_id == max(config["evaluation"]["selection_folds"])]
        actual_roc = binary_metrics(actual_fold)["roc_auc"]
        pvalue=(1+sum(value>=actual_roc for value in permutation_roc))/(1+len(permutation_roc))
        overfit_rows.append({"audit":"block_permutation_sanity","candidate":"CHAMPION_CURRENT","value":actual_roc,
                             "permutation_mean":float(np.mean(permutation_roc)),"permutation_pvalue":pvalue,
                             "status":"PASS" if pvalue<=.05 else "PERMUTATION_NOT_SEPARATED"})
    selection_comparison=selection["comparison"]
    random_row=selection_comparison[selection_comparison.candidate=="RANDOM_FEATURE_CONTROL"]
    if len(random_row):
        overfit_rows.append({"audit":"random_feature_control","candidate":"RANDOM_FEATURE_CONTROL",
                             "value":float(random_row.iloc[0].roc_auc-selection_comparison[selection_comparison.candidate=="CHAMPION_CURRENT"].iloc[0].roc_auc),
                             "status":"PASS" if float(random_row.iloc[0].roc_auc)<=float(selection_comparison[selection_comparison.candidate=="CHAMPION_CURRENT"].iloc[0].roc_auc)+.005 else "RANDOM_CONTROL_WARNING"})

    atomic_csv(pd.DataFrame(overfit_rows), output / "overfit_audit.csv")
    atomic_csv(pd.DataFrame(winner_rows), output / "winner_curse_analysis.csv")
    feature_stability = _feature_stability(output, config, meta, [*selection_lgb,*confirmation_lgb], [*selection_xgb,*confirmation_xgb])

    # Diagnostic LORO/sector results are already target-isolated development checks.
    generalization_rows=[]
    dates_all=np.load(meta["dates_path"],mmap_mode="r")
    for result in diagnostic_results:
        with np.load(result["npz_path"]) as payload:
            idx=payload["val_idx"]; y=payload["y"]; p=payload["predictions"].mean(axis=0)
        frame=pd.DataFrame({"date":pd.to_datetime(dates_all[idx]),"target":y,"prediction":p})
        generalization_rows.append({"diagnostic":result["candidate_id"],"fold_id":result["fold_id"],**binary_metrics(frame)})
    atomic_csv(pd.DataFrame(generalization_rows), output / "generalization_audit.csv")

    best_row = comparison[(comparison.scope=="CONFIRMATION") & (comparison.candidate==best_challenger)].iloc[0]
    improved_weak=int((reinforcement.delta_roc>0).sum())
    seed_part=seed_table[(seed_table.scope=="CONFIRMATION") & (seed_table.candidate==best_challenger)]
    seed_direction_ok=bool((seed_part.roc_auc > seed_table[(seed_table.scope=="CONFIRMATION") & (seed_table.candidate=="CHAMPION_CURRENT")].roc_auc.to_numpy()).sum()>=3) if best_challenger!="CHAMPION_CURRENT" else True
    replace_conditions={
        "pr_lift_not_worse":bool(best_row.pr_auc_lift>=champion_confirm.pr_auc_lift),
        "roc_improvement_or_065":bool(best_row.roc_auc>=champion_confirm.roc_auc+.01 or best_row.roc_auc>=.65),
        "weak_mean_roc_improved":bool(best_row.weak_mean_roc_auc>champion_confirm.weak_mean_roc_auc),
        "weak_tickers_improved_at_least_5":bool(improved_weak>=5),
        "strong_pr_lift_degradation_within_003":bool(best_row.strong_mean_pr_auc_lift>=champion_confirm.strong_mean_pr_auc_lift-.03),
        "worst_regime_not_worse_by_002":bool(best_row.worst_regime_roc_auc>=champion_confirm.worst_regime_roc_auc-.02),
        "seed_direction_consistent":seed_direction_ok,
        "bootstrap_no_serious_degradation":bool(np.nanquantile(boot_roc,.025)>-.02 and np.nanquantile(boot_pr,.025)>-.03),
        "selection_confirmation_no_collapse":bool([row for row in winner_rows if row["candidate"]==best_challenger][0]["risk"]=="OK"),
    }
    replace=bool(best_challenger!="CHAMPION_CURRENT" and all(replace_conditions.values()))
    recommended=best_challenger if replace else "CHAMPION_CURRENT"
    recommended_row=comparison[(comparison.scope=="CONFIRMATION") & (comparison.candidate==recommended)].iloc[0]
    operational_pass=bool(recommended_row.operating_alert_recall>=config["goals"]["target_event_recall"])
    probability_070_pass=bool(
        recommended_row.probability_ge_070_count>0
        and recommended_row.probability_ge_070_precision>=config["goals"]["declared_probability_precision_min"]
    )
    primary_goal=bool(
        recommended_row.primary_pass_ratio>=config["goals"]["evaluable_ticker_pass_ratio_min"]
        and recommended_row.mean_ticker_pr_auc_lift>=config["goals"]["overall_mean_pr_auc_lift_min"]
        and recommended_row.mean_ticker_roc_auc>=config["goals"]["overall_mean_roc_auc_min"]
        and recommended_row.median_ticker_roc_auc>=config["goals"]["median_ticker_roc_auc_min"]
        and recommended_row.worst_regime_roc_auc>=config["goals"]["worst_regime_roc_auc_min"]
    )
    overfit_statuses=pd.DataFrame(overfit_rows).status.astype(str).tolist()
    overfit_risk="HIGH" if any("WARNING" in value or "NOT_SEPARATED" in value for value in overfit_statuses) else "MEDIUM" if any(value in {"UNCERTAIN","RECENT_REGIME_DEPENDENT","STALE_SIGNAL"} for value in overfit_statuses) else "LOW"
    generalization="PASS" if primary_goal and overfit_risk=="LOW" else "CONDITIONAL" if primary_goal or operational_pass else "FAIL"

    source_files=[package/"run_long_horizon_generalization.py",package/"cwgeneralization/core.py",package/"cwgeneralization/engine.py",package/"cwgeneralization/report.py",package/"config_generalization.json"]
    freeze_core={
        "schema":"crashwatch_3d4_model_freeze_v1","recommended_model":recommended,
        "finalists":finalists,"target":config["target"],"seeds":config["seeds"],
        "folds":meta["folds"],"purge_policy":"existing conservative ~20 trading-day gap; minimum 3",
        "candidate":next(row for row in [*config["candidates"],*config.get("derived_candidates",[])] if row["id"]==recommended),
        "calibration":policies[recommended]["calibration"],"operating_threshold":policies[recommended]["operating_threshold"],
        "balanced_threshold":policies[recommended]["balanced_threshold"],"alert_rule":"daily top 1/3/5% plus frozen probability threshold",
        "feature_lists":{key:meta["profile_features"][value["profile"]] for key,value in config["models"].items()},
        "feature_hashes":{key:canonical_hash(meta["profile_features"][value["profile"]],24) for key,value in config["models"].items()},
        "source_hashes":{path.name:file_sha256(path) for path in source_files},
        "package_versions":{name:__import__(name).__version__ for name in ("numpy","pandas","sklearn","lightgbm","xgboost")},
        "sealed_data_used_for_selection":False,"new_final_sealed_opened":False,
    }
    freeze_core["manifest_hash"]=canonical_hash(freeze_core,32)
    atomic_json(freeze_core,output/"MODEL_FREEZE_MANIFEST.json")
    recommendation={
        "current_champion":"CHAMPION_CURRENT","recommended_model":recommended,"replace_champion":replace,
        "best_challenger":best_challenger,
        "primary_goal":{"pr_auc_lift_threshold":1.0,"roc_auc_threshold":.5,"pass":primary_goal},
        "operational_70_goal":{"event_recall_target":.70,"event_recall":recommended_row.operating_alert_recall,"pass":operational_pass,
                               "probability_threshold":.70,"observed_precision":recommended_row.probability_ge_070_precision,
                               "probability_alert_count":int(recommended_row.probability_ge_070_count),"probability_precision_pass":probability_070_pass},
        "overall_metrics":recommended_row.to_dict(),
        "weak_ticker_metrics":{"improved_count":improved_weak,"target_count":8,"mean_roc":recommended_row.weak_mean_roc_auc},
        "regime_metrics":{"worst_roc_auc":recommended_row.worst_regime_roc_auc},
        "replacement_conditions":replace_conditions,"generalization_status":generalization,
        "overfit_risk":overfit_risk,"calibration_status":"FROZEN_FROM_SELECTION_OOF_ONLY",
        "sealed_ready":bool(generalization in {"PASS","CONDITIONAL"} and overfit_risk!="HIGH"),
        "blocking_reasons":[key for key,value in replace_conditions.items() if not value],
        "notes":["Consumed sealed values were not read or used.","A genuinely new future interval is required for final sealed evaluation."],
    }
    atomic_json(recommendation,output/"FINAL_RECOMMENDATION.json")
    summary=f"""# CrashWatch Long-Horizon 3D/-4% Generalization

- Current champion: CHAMPION_CURRENT
- Best challenger: {best_challenger}
- Recommended model: {recommended}
- Replace champion: {'YES' if replace else 'NO'}
- Primary generalization goal: {'PASS' if primary_goal else 'FAIL'}
- 70% event-recall goal: {'PASS' if operational_pass else 'FAIL'} ({recommended_row.operating_alert_recall:.4f})
- Probability >= 0.70 observed precision: {recommended_row.probability_ge_070_precision}
- Mean ticker PR lift: {recommended_row.mean_ticker_pr_auc_lift:.4f}
- Mean ticker ROC-AUC: {recommended_row.mean_ticker_roc_auc:.4f}
- Worst regime ROC-AUC: {recommended_row.worst_regime_roc_auc:.4f}
- Weak tickers improved: {improved_weak}/8
- Overfit risk: {overfit_risk}
- Generalization: {generalization}

No consumed sealed label or metric was used for training, weighting, tuning, calibration, or model selection.
"""
    atomic_text(summary, output/"EXPERIMENT_SUMMARY.md")
    verdict={"current_champion":"CHAMPION_CURRENT","best_challenger":best_challenger,"primary_goal":"PASS" if primary_goal else "FAIL",
             "weak_ticker_improvement":f"{improved_weak} / 8","mean_pr_lift":float(recommended_row.mean_ticker_pr_auc_lift),
             "mean_roc_auc":float(recommended_row.mean_ticker_roc_auc),"worst_regime_roc":float(recommended_row.worst_regime_roc_auc),
             "recent_period_roc":float(recommended_row.roc_auc),"overfit_risk":overfit_risk,"generalization":generalization,
             "replace_current_champion":"YES" if replace else "NO","ready_for_new_final_sealed":"YES" if recommendation["sealed_ready"] else "NO"}
    atomic_json(verdict,output/"FINAL_VERDICT.json")
    return {"recommendation":recommendation,"verdict":verdict,"policies":policies}
