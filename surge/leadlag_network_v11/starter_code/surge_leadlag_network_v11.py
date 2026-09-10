from __future__ import annotations

import itertools
import json
import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from surge_leadlag_common_v11 import (
    SCHEMA_VERSION, FoldSpec, atomic_csv, atomic_json, atomic_text, best_precision_policy,
    bh_fdr, binary_metrics, binary_mutual_information, fisher_combine, granger_incremental,
    lag_align, load_folds, normalize_ticker, percentile_rank, quantile_mutual_information,
    role_for_fold, safe_auc, safe_corr, sha256_file, sign_metrics, stable_hash, tail_metrics,
    utc_now, weighted_mean,
)


def log(message: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] V11 {message}", flush=True)


def load_frame(dataset: Path, target_sidecar: Path) -> pd.DataFrame:
    columns = [
        "date", "ticker", "name", "market", "bucket", "industry_name", "market_cap",
        "t_price_ret_1", "return_pct",
    ]
    source = pd.read_parquet(dataset, columns=columns).reset_index(drop=True)
    source["source_row_id"] = np.arange(len(source), dtype=np.int64)
    source["ticker"] = source["ticker"].map(normalize_ticker)
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    primary = pd.to_numeric(source["t_price_ret_1"], errors="coerce")
    fallback = pd.to_numeric(source["return_pct"], errors="coerce") / 100.0
    source["return_1d"] = primary.where(primary.notna(), fallback)
    target = pd.read_parquet(target_sidecar)
    target["ticker"] = target["ticker"].map(normalize_ticker)
    target["date"] = pd.to_datetime(target["date"], errors="raise")
    merged = source.merge(
        target[["source_row_id", "date", "ticker", "label_abs_surge_3d_5pct", "target_valid", "best_forward_return_3d"]],
        on=["source_row_id", "date", "ticker"], how="inner", validate="one_to_one",
    )
    merged = merged.loc[merged["target_valid"].astype(bool) & merged["label_abs_surge_3d_5pct"].isin([0, 1])].copy()
    if merged.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Duplicate ticker/date rows")
    return merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def choose_universe(frame: pd.DataFrame, folds: Sequence[FoldSpec], *, tickers: str, top_n: int, all_tickers: bool) -> list[str]:
    available = sorted(frame["ticker"].astype(str).unique())
    if tickers:
        requested = [normalize_ticker(value) for value in tickers.split(",") if value.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Unknown tickers: {missing}")
        return list(dict.fromkeys(requested))
    if top_n > 0:
        cutoff = max(fold.train_end for fold in folds if fold.fold_id <= 2)
        history = frame.loc[frame["date"].le(cutoff) & frame["market_cap"].notna()].copy()
        latest = history.sort_values(["ticker", "date"], kind="mergesort").groupby("ticker", as_index=False).tail(1)
        selected = latest.sort_values(["market_cap", "ticker"], ascending=[False, True], kind="mergesort")["ticker"].head(top_n).tolist()
        if len(selected) < top_n:
            raise RuntimeError(f"Only {len(selected)} tickers have point-in-time market cap before {cutoff.date()}")
        return selected
    if not all_tickers:
        raise ValueError("Choose --tickers, --top-n, or --all-tickers")
    return available


def make_panel(frame: pd.DataFrame, universe: Sequence[str]) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = frame.loc[frame["ticker"].isin(universe)].copy()
    returns = selected.pivot(index="date", columns="ticker", values="return_1d").reindex(columns=universe).sort_index()
    target = selected.pivot(index="date", columns="ticker", values="label_abs_surge_3d_5pct").reindex(index=returns.index, columns=universe)
    source_ids = selected.pivot(index="date", columns="ticker", values="source_row_id").reindex(index=returns.index, columns=universe)
    return pd.DatetimeIndex(returns.index), returns, target, source_ids


def ticker_metadata(frame: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    rows = []
    for ticker in universe:
        part = frame.loc[frame["ticker"].eq(ticker)].sort_values("date")
        def latest(column: str) -> str:
            values = part[column].dropna().astype(str)
            return values.iloc[-1] if len(values) else "UNKNOWN"
        cap = pd.to_numeric(part["market_cap"], errors="coerce").dropna()
        rows.append({"ticker": ticker, "name": latest("name"), "market": latest("market"), "bucket": latest("bucket"),
                     "industry": latest("industry_name"), "latest_observed_market_cap": float(cap.iloc[-1]) if len(cap) else math.nan,
                     "rows": int(len(part)), "date_min": str(part["date"].min().date()), "date_max": str(part["date"].max().date())})
    return pd.DataFrame(rows)


def leave_one_out_factor(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    groups = np.asarray(groups, dtype=str)
    for group in sorted(set(groups)):
        positions = np.flatnonzero(groups == group)
        block = values[:, positions]
        valid = np.isfinite(block)
        total = np.nansum(block, axis=1)
        count = valid.sum(axis=1)
        for local, position in enumerate(positions):
            denominator = count - valid[:, local]
            numerator = total - np.where(valid[:, local], block[:, local], 0.0)
            output[:, position] = np.divide(numerator, denominator, out=np.full(len(values), np.nan), where=denominator > 0)
    return output


def fit_apply_residual(y: np.ndarray, factors: np.ndarray, train: np.ndarray, evaluate: np.ndarray) -> np.ndarray:
    y_train = y[train]
    x_train = factors[train]
    valid = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    if valid.sum() < max(60, x_train.shape[1] * 10):
        return np.full(len(evaluate), np.nan)
    design = np.column_stack([np.ones(valid.sum()), x_train[valid]])
    coefficients, *_ = np.linalg.lstsq(design, y_train[valid], rcond=None)
    x_eval = factors[evaluate]
    result = np.full(len(evaluate), np.nan)
    good = np.isfinite(y[evaluate]) & np.all(np.isfinite(x_eval), axis=1)
    result[good] = y[evaluate][good] - np.column_stack([np.ones(good.sum()), x_eval[good]]) @ coefficients
    return result


def build_fold_variants(dates: pd.DatetimeIndex, returns: pd.DataFrame, metadata: pd.DataFrame, folds: Sequence[FoldSpec]) -> dict[int, dict[str, Any]]:
    raw = returns.to_numpy(float)
    cross_mean = np.nanmean(raw, axis=1, keepdims=True)
    cross = raw - cross_mean
    markets = metadata.set_index("ticker").reindex(returns.columns)["market"].fillna("UNKNOWN").astype(str).to_numpy()
    buckets = metadata.set_index("ticker").reindex(returns.columns)["bucket"].fillna("UNKNOWN").astype(str).to_numpy()
    market_factor = leave_one_out_factor(raw, markets)
    bucket_factor = leave_one_out_factor(raw, buckets)
    payload: dict[int, dict[str, Any]] = {}
    for fold in folds:
        train = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        validation = np.flatnonzero((dates >= fold.validation_start) & (dates <= fold.validation_end))
        market_residual = np.full((len(validation), raw.shape[1]), np.nan)
        bucket_residual = np.full((len(validation), raw.shape[1]), np.nan)
        for position in range(raw.shape[1]):
            market_residual[:, position] = fit_apply_residual(raw[:, position], market_factor[:, [position]], train, validation)
            bucket_residual[:, position] = fit_apply_residual(raw[:, position], np.column_stack([market_factor[:, position], bucket_factor[:, position]]), train, validation)
        validation_raw = raw[validation]
        rank = pd.DataFrame(validation_raw).rank(axis=1, pct=True, method="average").to_numpy(float)
        payload[fold.fold_id] = {
            "fold": fold, "indices": validation, "dates": dates[validation],
            "variants": {"raw_return": validation_raw, "cross_demean": cross[validation], "market_residual": market_residual,
                         "bucket_residual": bucket_residual, "return_rank": rank, "direction_sign": np.sign(validation_raw)},
        }
    return payload


def compute_pair_maps(universe: Sequence[str], pairs: Sequence[tuple[int, int]], fold_payload: Mapping[int, Mapping[str, Any]], threads: int, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def one(pair: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        a, b = pair
        ticker_a, ticker_b = universe[a], universe[b]
        lag_rows, direction_rows, tail_rows = [], [], []
        for fold_id, payload in fold_payload.items():
            fold = payload["fold"]
            raw_a, raw_b = payload["variants"]["raw_return"][:, a], payload["variants"]["raw_return"][:, b]
            pearson, p_value, n = safe_corr(raw_a, raw_b, minimum=minimum)
            spearman, sp, _ = safe_corr(raw_a, raw_b, method="spearman", minimum=minimum)
            common = {"ticker_a": ticker_a, "ticker_b": ticker_b, "fold_id": fold_id, "role": role_for_fold(fold_id)}
            direction_rows.append(common | {"pearson": pearson, "pearson_p": p_value, "spearman": spearman, "spearman_p": sp, **sign_metrics(raw_a, raw_b)})
            tail_rows.append(common | tail_metrics(raw_a, raw_b))
            for variant, matrix in payload["variants"].items():
                for lag in range(-5, 6):
                    x, y = lag_align(matrix[:, a], matrix[:, b], lag)
                    corr, pv, valid_n = safe_corr(x, y, minimum=minimum)
                    lag_rows.append(common | {"variant": variant, "lag": lag, "correlation": corr, "p_value": pv, "observations": valid_n})
        return lag_rows, direction_rows, tail_rows

    lag_parts, direction_parts, tail_parts = [], [], []
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="pair-map") as pool:
        for done, (lags, directions, tails) in enumerate(pool.map(one, pairs), start=1):
            lag_parts.extend(lags); direction_parts.extend(directions); tail_parts.extend(tails)
            if done % 100 == 0 or done == len(pairs):
                log(f"pair maps {done}/{len(pairs)}")
    return pd.DataFrame(lag_parts), pd.DataFrame(direction_parts), pd.DataFrame(tail_parts)


def summarize_role_frame(frame: pd.DataFrame, metrics: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, part in frame.groupby(list(keys), sort=True, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(keys, key))
        for role in ["discovery", "development", "confirmation", "recent_audit"]:
            current = part.loc[part["role"].eq(role)]
            row[f"{role}_folds"] = int(current["fold_id"].nunique())
            for metric in metrics:
                values = pd.to_numeric(current[metric], errors="coerce")
                weights = pd.to_numeric(current.get("observations", pd.Series(1, index=current.index)), errors="coerce")
                row[f"{role}_{metric}"] = weighted_mean(values, weights)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_lags(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    discovery = by_fold.loc[by_fold["role"].eq("discovery")]
    group_columns = ["ticker_a", "ticker_b", "variant"]
    full_groups = {key: part for key, part in by_fold.groupby(group_columns, sort=False)}
    for (ticker_a, ticker_b, variant), part in discovery.groupby(group_columns, sort=True):
        lag_stats = []
        for lag, current in part.groupby("lag", sort=True):
            correlations = pd.to_numeric(current["correlation"], errors="coerce").to_numpy(float)
            valid = np.isfinite(correlations)
            mean_corr = weighted_mean(correlations, current["observations"].to_numpy(float))
            consistency = float(np.mean(np.sign(correlations[valid]) == np.sign(mean_corr))) if valid.any() and math.isfinite(mean_corr) else math.nan
            lag_stats.append({"lag": int(lag), "mean": mean_corr, "abs": abs(mean_corr) if math.isfinite(mean_corr) else -1,
                              "consistency": consistency, "folds": int(valid.sum()), "p": fisher_combine(current["p_value"].to_numpy(float)),
                              "observations": int(pd.to_numeric(current["observations"], errors="coerce").sum())})
        eligible = [row for row in lag_stats if row["folds"] >= 2]
        best = max(eligible, key=lambda row: (row["abs"], -abs(row["lag"]), -row["lag"])) if eligible else {"lag": 0, "mean": math.nan, "abs": math.nan, "consistency": math.nan, "folds": 0, "p": math.nan, "observations": 0}
        positive = max([row["abs"] for row in lag_stats if row["lag"] > 0], default=math.nan)
        negative = max([row["abs"] for row in lag_stats if row["lag"] < 0], default=math.nan)
        row = {"ticker_a": ticker_a, "ticker_b": ticker_b, "variant": variant, "discovery_best_lag": best["lag"],
               "discovery_best_lag_correlation": best["mean"], "discovery_abs_strength": best["abs"],
               "discovery_sign_consistency": best["consistency"], "discovery_valid_folds": best["folds"],
               "discovery_observations": best["observations"], "discovery_p_value": best["p"],
               "best_positive_lag_strength": positive, "best_negative_lag_strength": negative,
               "lead_advantage": positive - negative if math.isfinite(positive) and math.isfinite(negative) else math.nan}
        # Use the pre-indexed all-fold group for fixed-lag validation.  Scanning
        # the full ~600k-row table per group is quadratic, while filtering the
        # discovery-only `part` would incorrectly erase folds 3-7.
        all_part = full_groups[(ticker_a, ticker_b, variant)]
        fixed = all_part.loc[all_part["lag"].eq(best["lag"])]
        for role in ["development", "confirmation", "recent_audit"]:
            current = fixed.loc[fixed["role"].eq(role)]
            value = weighted_mean(current["correlation"], current["observations"])
            row[f"{role}_fixed_correlation"] = value
            row[f"{role}_same_direction"] = bool(math.isfinite(value) and math.isfinite(best["mean"]) and np.sign(value) == np.sign(best["mean"]))
            row[f"{role}_valid_folds"] = int(pd.to_numeric(current["correlation"], errors="coerce").notna().sum())
        if best["lag"] > 0:
            row.update({"leader": ticker_a, "follower": ticker_b, "directed_lag": best["lag"]})
        elif best["lag"] < 0:
            row.update({"leader": ticker_b, "follower": ticker_a, "directed_lag": -best["lag"]})
        else:
            row.update({"leader": "", "follower": "", "directed_lag": 0})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["discovery_q_value"] = np.nan
    for variant, index in result.groupby("variant").groups.items():
        result.loc[index, "discovery_q_value"] = bh_fdr(result.loc[index, "discovery_p_value"].to_numpy(float))
    result["strong_directed_edge"] = (
        result["directed_lag"].gt(0) & result["discovery_valid_folds"].ge(2)
        & result["discovery_sign_consistency"].ge(0.80) & result["discovery_observations"].ge(100)
        & result["discovery_q_value"].le(0.10) & result["development_same_direction"].astype(bool)
        & result["development_valid_folds"].ge(2)
    )
    return result


def compute_lagged_lifts(universe: Sequence[str], fold_payload: Mapping[int, Mapping[str, Any]], target_panel: pd.DataFrame, threads: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = [(a, b) for a in range(len(universe)) for b in range(len(universe)) if a != b]
    target_values = target_panel.to_numpy(float)
    def one(pair: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        a, b = pair; leader, follower = universe[a], universe[b]
        directional, tail = [], []
        for fold_id, payload in fold_payload.items():
            indices = payload["indices"]
            raw = payload["variants"]["raw_return"]
            for lag in [1, 2, 3, 5]:
                x, y = lag_align(raw[:, a], raw[:, b], lag)
                valid = np.isfinite(x) & np.isfinite(y)
                xx, yy = x[valid], y[valid]
                if len(xx) >= 20:
                    base_up, base_down = float(np.mean(yy > 0)), float(np.mean(yy < 0))
                    up = xx > 0; down = xx < 0
                    up_lift = float(np.mean(yy[up] > 0) - base_up) if up.sum() >= 5 else math.nan
                    down_lift = float(np.mean(yy[down] < 0) - base_down) if down.sum() >= 5 else math.nan
                else:
                    base_up = base_down = up_lift = down_lift = math.nan
                directional.append({"leader": leader, "follower": follower, "fold_id": fold_id, "role": role_for_fold(fold_id),
                                    "lag": lag, "observations": len(xx), "up_directional_lift": up_lift,
                                    "down_directional_lift": down_lift, "follower_up_baseline": base_up, "follower_down_baseline": base_down})
                tx, ty = lag_align(raw[:, a], target_values[indices, b], lag)
                for shock, mask in [("UP_1PCT", tx >= 0.01), ("UP_2PCT", tx >= 0.02), ("DOWN_1PCT", tx <= -0.01),
                                    ("DOWN_2PCT", tx <= -0.02), ("VOLATILITY_SHOCK", np.abs(tx) >= 0.03)]:
                    valid_target = np.isfinite(tx) & np.isfinite(ty)
                    baseline = float(np.mean(ty[valid_target])) if valid_target.sum() else math.nan
                    event = valid_target & mask
                    conditional = float(np.mean(ty[event])) if event.sum() >= 3 else math.nan
                    tail.append({"leader": leader, "follower": follower, "fold_id": fold_id, "role": role_for_fold(fold_id),
                                 "lag": lag, "shock": shock, "observations": int(valid_target.sum()), "event_n": int(event.sum()),
                                 "follower_surge3d_baseline": baseline, "follower_surge3d_conditional": conditional,
                                 "surge3d_lift_ratio": conditional / baseline if math.isfinite(conditional) and baseline > 0 else math.nan,
                                 "surge3d_lift_difference": conditional - baseline if math.isfinite(conditional) and math.isfinite(baseline) else math.nan})
        return directional, tail
    directional_rows, tail_rows = [], []
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="lag-lift") as pool:
        for done, (drows, trows) in enumerate(pool.map(one, ordered), start=1):
            directional_rows.extend(drows); tail_rows.extend(trows)
            if done % 300 == 0 or done == len(ordered): log(f"lagged lifts {done}/{len(ordered)}")
    return pd.DataFrame(directional_rows), pd.DataFrame(tail_rows)


def concatenate_role_segments(fold_payload: Mapping[int, Mapping[str, Any]], position: int, role: str, variant: str = "bucket_residual") -> list[np.ndarray]:
    return [payload["variants"][variant][:, position] for payload in fold_payload.values() if role_for_fold(payload["fold"].fold_id) == role]


def compute_event_and_nonlinear(universe: Sequence[str], fold_payload: Mapping[int, Mapping[str, Any]], threads: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = [(a, b) for a in range(len(universe)) for b in range(len(universe)) if a != b]
    def one(pair: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        a, b = pair; leader, follower = universe[a], universe[b]
        responses, granger_rows, mi_rows = [], [], []
        for role in ["discovery", "development", "confirmation", "recent_audit"]:
            segments_a = concatenate_role_segments(fold_payload, a, role)
            segments_b = concatenate_role_segments(fold_payload, b, role)
            for shock, threshold in [("UP_2PCT", 0.02), ("DOWN_2PCT", -0.02)]:
                for horizon in [0, 1, 2, 3, 5]:
                    event_values = []
                    for xa, yb in zip(segments_a, segments_b):
                        x, y = lag_align(xa, yb, horizon)
                        mask = x >= threshold if threshold > 0 else x <= threshold
                        event_values.extend(y[mask & np.isfinite(y)].tolist())
                    values = np.asarray(event_values, dtype=float)
                    responses.append({"leader": leader, "follower": follower, "role": role, "shock": shock, "horizon": horizon,
                                      "event_n": len(values), "mean_response": float(np.nanmean(values)) if len(values) else math.nan,
                                      "median_response": float(np.nanmedian(values)) if len(values) else math.nan,
                                      "positive_response_probability": float(np.mean(values > 0)) if len(values) else math.nan})
        discovery_a = concatenate_role_segments(fold_payload, a, "discovery")
        discovery_b = concatenate_role_segments(fold_payload, b, "discovery")
        for order in [1, 2, 3, 5]:
            parts = [granger_incremental(yb, xa, order) for xa, yb in zip(discovery_a, discovery_b)]
            granger_rows.append({"leader": leader, "follower": follower, "lag_order": order,
                                 "observations": sum(int(part["observations"]) for part in parts),
                                 "baseline_r2": weighted_mean([part["baseline_r2"] for part in parts], [part["observations"] for part in parts]),
                                 "augmented_r2": weighted_mean([part["augmented_r2"] for part in parts], [part["observations"] for part in parts]),
                                 "incremental_r2": weighted_mean([part["incremental_r2"] for part in parts], [part["observations"] for part in parts]),
                                 "p_value": fisher_combine([part["p_value"] for part in parts])})
        for lag in [1, 2, 3, 4, 5]:
            return_mi, sign_mi, total = [], [], 0
            for xa, yb in zip(discovery_a, discovery_b):
                x, y = lag_align(xa, yb, lag)
                mi, n = quantile_mutual_information(x, y); bmi, _ = binary_mutual_information(x, y)
                return_mi.append(mi); sign_mi.append(bmi); total += n
            mi_rows.append({"leader": leader, "follower": follower, "lag": lag, "observations": total,
                            "return_mutual_information": weighted_mean(return_mi, [len(x) for x in discovery_a]),
                            "direction_mutual_information": weighted_mean(sign_mi, [len(x) for x in discovery_a])})
        return responses, granger_rows, mi_rows
    response_rows, granger_rows, mi_rows = [], [], []
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="nonlinear") as pool:
        for done, (response, granger, mi) in enumerate(pool.map(one, ordered), start=1):
            response_rows.extend(response); granger_rows.extend(granger); mi_rows.extend(mi)
            if done % 300 == 0 or done == len(ordered): log(f"event/nonlinear {done}/{len(ordered)}")
    granger = pd.DataFrame(granger_rows)
    granger["q_value"] = bh_fdr(granger["p_value"].to_numpy(float))
    return pd.DataFrame(response_rows), granger, pd.DataFrame(mi_rows)


def classify_state(correlation: float, best_lag: int, lead_strength: float, spread_z: float, prior_spread_z: float) -> str:
    if not math.isfinite(correlation) or not math.isfinite(lead_strength): return "NO_EVIDENCE"
    if math.isfinite(spread_z) and abs(spread_z) >= 2.0: return "DIVERGING"
    if math.isfinite(spread_z) and math.isfinite(prior_spread_z) and abs(prior_spread_z) >= 1.5 and abs(spread_z) < abs(prior_spread_z) - 0.35: return "RECOUPLING"
    if correlation <= -0.35: return "OPPOSITE_DIRECTION"
    if best_lag > 0 and lead_strength >= 0.20: return "A_LEADS_B"
    if best_lag < 0 and lead_strength >= 0.20: return "B_LEADS_A"
    if correlation >= 0.50: return "STRONG_SYNCHRONOUS"
    if correlation >= 0.25: return "WEAK_SYNCHRONOUS"
    if abs(correlation) < 0.10 and abs(lead_strength) < 0.12: return "UNSTABLE"
    return "NO_EVIDENCE"


def compute_rolling(universe: Sequence[str], returns: pd.DataFrame, pairs: Sequence[tuple[int, int]], windows: Sequence[int], step: int, threads: int) -> pd.DataFrame:
    raw = returns.to_numpy(float)
    residual = raw - np.nanmean(raw, axis=1, keepdims=True)
    dates = pd.DatetimeIndex(returns.index)
    min_window = min(windows)
    endpoints = sorted(set(range(min_window - 1, len(dates), step)) | {len(dates) - 1})
    def rolling_corr(x: np.ndarray, y: np.ndarray, minimum: int) -> tuple[float, int]:
        """Pearson value only; rolling maps do not consume a p-value."""
        valid = np.isfinite(x) & np.isfinite(y)
        n = int(valid.sum())
        if n < minimum:
            return math.nan, n
        xv, yv = x[valid], y[valid]
        xv = xv - xv.mean(); yv = yv - yv.mean()
        denominator = float(np.sqrt(np.dot(xv, xv) * np.dot(yv, yv)))
        return (float(np.dot(xv, yv) / denominator), n) if denominator > 1e-12 else (math.nan, n)
    def one(pair: tuple[int, int]) -> list[dict[str, Any]]:
        a, b = pair; rows = []
        for window in windows:
            prior_spread = math.nan
            for endpoint in endpoints:
                if endpoint + 1 < window: continue
                start = endpoint + 1 - window
                xa, yb = residual[start:endpoint + 1, a], residual[start:endpoint + 1, b]
                corr, n = rolling_corr(xa, yb, minimum=max(20, window // 2))
                candidates = []
                for lag in range(-5, 6):
                    x, y = lag_align(xa, yb, lag); value, valid_n = rolling_corr(x, y, minimum=max(20, window // 2 - 5))
                    candidates.append((abs(value) if math.isfinite(value) else -1, -abs(lag), lag, value, valid_n))
                _, _, best_lag, best_corr, best_n = max(candidates)
                sign_valid = np.isfinite(xa) & np.isfinite(yb) & (xa != 0) & (yb != 0)
                sign = float(np.mean(np.sign(xa[sign_valid]) == np.sign(yb[sign_valid]))) if sign_valid.any() else math.nan
                valid = np.isfinite(xa) & np.isfinite(yb)
                if valid.sum() >= 20:
                    beta = float(np.dot(yb[valid], xa[valid]) / max(np.dot(yb[valid], yb[valid]), 1e-12))
                    spread_return = xa[valid] - beta * yb[valid]
                    cumulative = np.cumsum(spread_return)
                    spread_z = float((cumulative[-1] - np.mean(cumulative)) / max(np.std(cumulative), 1e-12))
                else:
                    beta = spread_z = math.nan
                state = classify_state(corr, best_lag, abs(best_corr) if math.isfinite(best_corr) else math.nan, spread_z, prior_spread)
                rows.append({"date": dates[endpoint], "ticker_a": universe[a], "ticker_b": universe[b], "window": window,
                             "observations": n, "contemporaneous_correlation": corr, "best_lag": best_lag,
                             "best_lag_correlation": best_corr, "lead_strength": abs(best_corr) if math.isfinite(best_corr) else math.nan,
                             "sign_agreement": sign, "spread_beta_trailing": beta, "spread_z": spread_z,
                             "relation_state": state, "uses_centered_window": False})
                prior_spread = spread_z
        return rows
    rows = []
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="rolling") as pool:
        for done, part in enumerate(pool.map(one, pairs), start=1):
            rows.extend(part)
            if done % 100 == 0 or done == len(pairs): log(f"rolling pairs {done}/{len(pairs)}")
    return pd.DataFrame(rows)


def divergence_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (a, b), part in rolling.groupby(["ticker_a", "ticker_b"], sort=True):
        row = {"ticker_a": a, "ticker_b": b}
        for window in [60, 120, 252]:
            current = part.loc[part["window"].eq(window)].sort_values("date")
            if current.empty: continue
            latest = current.iloc[-1]
            values = pd.to_numeric(current["spread_z"], errors="coerce")
            row[f"spread_z{window}"] = float(latest["spread_z"])
            row[f"spread_percentile_{window}"] = float((values <= latest["spread_z"]).mean()) if math.isfinite(latest["spread_z"]) else math.nan
            row[f"relation_state_{window}"] = str(latest["relation_state"])
            row[f"spread_widening_speed_{window}"] = float(values.iloc[-1] - values.iloc[-2]) if len(values) >= 2 else math.nan
        recent = part.loc[part["window"].eq(60)].sort_values("date")
        diverged = recent["relation_state"].eq("DIVERGING")
        if diverged.any():
            last_date = pd.Timestamp(recent.loc[diverged, "date"].iloc[-1]); row["days_since_divergence"] = int((pd.Timestamp(recent["date"].iloc[-1]) - last_date).days)
        else: row["days_since_divergence"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_self_map(universe: Sequence[str], fold_payload: Mapping[int, Mapping[str, Any]], target_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = target_panel.to_numpy(float)
    for position, ticker in enumerate(universe):
        ticker_rows = []
        for fold_id, payload in fold_payload.items():
            values = payload["variants"]["raw_return"][:, position]
            for lag in [1, 2, 3, 5]:
                x, y = lag_align(values, values, lag)
                corr, p_value, n = safe_corr(x, y, minimum=20)
                valid = np.isfinite(x) & np.isfinite(y)
                up, down = valid & (x > 0), valid & (x < 0)
                surge_x, surge_y = lag_align(values, target[payload["indices"], position], 0)
                valid_target = np.isfinite(surge_x) & np.isfinite(surge_y)
                ticker_rows.append({"ticker": ticker, "fold_id": fold_id, "role": role_for_fold(fold_id), "lag": lag,
                                    "correlation": corr, "p_value": p_value, "observations": n,
                                    "p_next_up_given_current_up": float(np.mean(y[up] > 0)) if up.sum() else math.nan,
                                    "p_next_down_given_current_down": float(np.mean(y[down] < 0)) if down.sum() else math.nan,
                                    "p_surge3d_given_current_up2": float(np.mean(surge_y[valid_target & (surge_x >= 0.02)])) if (valid_target & (surge_x >= 0.02)).sum() else math.nan,
                                    "p_surge3d_given_current_down2": float(np.mean(surge_y[valid_target & (surge_x <= -0.02)])) if (valid_target & (surge_x <= -0.02)).sum() else math.nan})
        rows.extend(ticker_rows)
    frame = pd.DataFrame(rows)
    discovery = frame.loc[(frame["role"].eq("discovery")) & frame["lag"].eq(1)].groupby("ticker")["correlation"].mean()
    frame["self_class"] = frame["ticker"].map(lambda ticker: "SELF_MOMENTUM" if discovery.get(ticker, 0) >= 0.05 else "SELF_MEAN_REVERSION" if discovery.get(ticker, 0) <= -0.05 else "SELF_NEUTRAL")
    return frame


def build_directed_network(lag_summary: pd.DataFrame, directional: pd.DataFrame, tail_lift: pd.DataFrame, granger: pd.DataFrame, mutual: pd.DataFrame) -> pd.DataFrame:
    base = lag_summary.loc[lag_summary["variant"].eq("bucket_residual") & lag_summary["directed_lag"].gt(0)].copy()
    d = directional.loc[directional["role"].eq("discovery")].groupby(["leader", "follower", "lag"], as_index=False).agg(
        directional_lift=("up_directional_lift", "mean"))
    t = tail_lift.loc[tail_lift["role"].eq("discovery") & tail_lift["shock"].eq("UP_2PCT")].groupby(["leader", "follower", "lag"], as_index=False).agg(
        surge3d_lift=("surge3d_lift_ratio", "mean"))
    g = granger.sort_values(["leader", "follower", "q_value", "incremental_r2"], ascending=[True, True, True, False], kind="mergesort").groupby(["leader", "follower"], as_index=False).first()
    mi = mutual.sort_values(["leader", "follower", "return_mutual_information"], ascending=[True, True, False], kind="mergesort").groupby(["leader", "follower"], as_index=False).first()
    edges = base.merge(d, left_on=["leader", "follower", "directed_lag"], right_on=["leader", "follower", "lag"], how="left").drop(columns=["lag"], errors="ignore")
    edges = edges.merge(t, left_on=["leader", "follower", "directed_lag"], right_on=["leader", "follower", "lag"], how="left").drop(columns=["lag"], errors="ignore")
    edges = edges.merge(g[["leader", "follower", "lag_order", "incremental_r2", "q_value"]].rename(columns={"q_value": "granger_q_value"}), on=["leader", "follower"], how="left")
    edges = edges.merge(mi[["leader", "follower", "lag", "return_mutual_information", "direction_mutual_information"]].rename(columns={"lag": "mi_best_lag"}), on=["leader", "follower"], how="left")
    for metric in ["discovery_abs_strength", "directional_lift", "surge3d_lift", "incremental_r2", "return_mutual_information"]:
        values = pd.to_numeric(edges[metric], errors="coerce")
        if metric == "surge3d_lift": values = values - 1.0
        edges[f"{metric}_rank"] = percentile_rank(values.fillna(values.min() if values.notna().any() else 0.0))
    edges["consensus_score"] = (
        0.30 * edges["discovery_abs_strength_rank"] + 0.15 * edges["directional_lift_rank"]
        + 0.20 * edges["surge3d_lift_rank"] + 0.15 * edges["incremental_r2_rank"]
        + 0.10 * edges["return_mutual_information_rank"] + 0.10 * edges["discovery_sign_consistency"].fillna(0)
    )
    edges["edge_status"] = np.where(edges["strong_directed_edge"].astype(bool), "STRONG_DIRECTED", "DISCOVERY_CANDIDATE")
    return edges.sort_values(["edge_status", "consensus_score", "leader", "follower"], ascending=[False, False, True, True], kind="mergesort")


def leader_scores(universe: Sequence[str], edges: pd.DataFrame) -> pd.DataFrame:
    strong = edges.loc[edges["strong_directed_edge"].astype(bool)].copy()
    rows = []
    for ticker in universe:
        outgoing = strong.loc[strong["leader"].eq(ticker)]
        incoming = strong.loc[strong["follower"].eq(ticker)]
        out_strength = float(outgoing["consensus_score"].sum())
        in_strength = float(incoming["consensus_score"].sum())
        up_out = float((outgoing["consensus_score"] * pd.to_numeric(outgoing["directional_lift"], errors="coerce").clip(lower=0).fillna(0)).sum())
        down_out = float((outgoing["consensus_score"] * pd.to_numeric(outgoing["directional_lift"], errors="coerce").clip(upper=0).abs().fillna(0)).sum())
        surge_out = float((outgoing["consensus_score"] * (pd.to_numeric(outgoing["surge3d_lift"], errors="coerce") - 1).clip(lower=0).fillna(0)).sum())
        rows.append({"ticker": ticker, "outgoing_edges": len(outgoing), "incoming_edges": len(incoming),
                     "outgoing_lead_strength": out_strength, "incoming_lead_strength": in_strength,
                     "leader_score": out_strength - in_strength, "up_leader_score": up_out,
                     "down_leader_score": down_out, "surge_leader_score": surge_out})
    return pd.DataFrame(rows).sort_values(["leader_score", "ticker"], ascending=[False, True], kind="mergesort")


def cascades(edges: pd.DataFrame) -> pd.DataFrame:
    strong = edges.loc[edges["strong_directed_edge"].astype(bool)]
    rows = []
    for left in strong.itertuples(index=False):
        for right in strong.loc[strong["leader"].eq(left.follower)].itertuples(index=False):
            if left.leader == right.follower: continue
            total_lag = int(left.directed_lag) + int(right.directed_lag)
            if total_lag > 5: continue
            rows.append({"ticker_a": left.leader, "ticker_b": left.follower, "ticker_c": right.follower,
                         "lag_ab": int(left.directed_lag), "lag_bc": int(right.directed_lag), "total_lag": total_lag,
                         "cascade_score": float(math.sqrt(max(left.consensus_score, 0) * max(right.consensus_score, 0)))})
    return pd.DataFrame(rows).sort_values(["cascade_score", "ticker_a", "ticker_b", "ticker_c"], ascending=[False, True, True, True], kind="mergesort") if rows else pd.DataFrame(columns=["ticker_a","ticker_b","ticker_c","lag_ab","lag_bc","total_lag","cascade_score"])


def feature_manifest(edges: pd.DataFrame, maximum_incoming: int = 5) -> dict[str, Any]:
    stable = edges.loc[edges["strong_directed_edge"].astype(bool)].copy()
    if stable.empty:
        stable = edges.loc[edges["development_same_direction"].astype(bool)].sort_values("consensus_score", ascending=False).head(40)
    features, ticker_sources = [], {}
    for follower, part in stable.groupby("follower", sort=True):
        part = part.sort_values("consensus_score", ascending=False).head(maximum_incoming)
        ticker_sources[str(follower)] = []
        for row in part.itertuples(index=False):
            name = f"lead_{row.leader}_ret1"
            features.append({"feature": name, "leader": row.leader, "follower": row.follower, "formula": f"return_1d[{row.leader}, t]",
                             "available_at_t": True, "discovery_lag": int(row.directed_lag), "consensus_score": float(row.consensus_score)})
            ticker_sources[str(follower)].append(str(row.leader))
        for name, formula in [("leader_consensus_return", "weighted mean of upstream return_1d at t"),
                              ("incoming_lead_pressure", "weighted signed upstream return_1d at t"),
                              ("upstream_up_count", "count(upstream return_1d[t] > 0)"),
                              ("upstream_down_count", "count(upstream return_1d[t] < 0)")]:
            features.append({"feature": name, "follower": follower, "formula": formula, "available_at_t": True})
    return {"schema": SCHEMA_VERSION, "selection_source": "folds 0-2 discovery with fixed validation on folds 3-4",
            "future_values_used": False, "features": features, "ticker_upstream_sources": ticker_sources}


def build_lead_feature_frame(frame: pd.DataFrame, manifest: Mapping[str, Any], edges: pd.DataFrame) -> pd.DataFrame:
    panel = frame.pivot(index="date", columns="ticker", values="return_1d")
    weights = edges.set_index(["leader", "follower"])["consensus_score"].to_dict()
    rows = []
    for follower, leaders in manifest.get("ticker_upstream_sources", {}).items():
        if follower not in panel.columns: continue
        for date in panel.index:
            values = np.asarray([panel.at[date, leader] if leader in panel.columns else math.nan for leader in leaders], dtype=float)
            ws = np.asarray([weights.get((leader, follower), 0.0) for leader in leaders], dtype=float)
            valid = np.isfinite(values)
            if valid.any() and ws[valid].sum() > 0:
                consensus = float(np.average(values[valid], weights=ws[valid])); pressure = float(np.sum(values[valid] * ws[valid]))
            else: consensus = pressure = math.nan
            row = {"date": date, "ticker": follower, "leader_consensus_return": consensus, "incoming_lead_pressure": pressure,
                   "upstream_up_count": int(np.sum(values[valid] > 0)), "upstream_down_count": int(np.sum(values[valid] < 0))}
            for leader, value in zip(leaders, values): row[f"lead_{leader}_ret1"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def run_probe(frame: pd.DataFrame, folds: Sequence[FoldSpec], lead_features: pd.DataFrame, v10_output: Path, target_precision: float, minimum_alerts: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if lead_features.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = pd.read_csv(v10_output / "ticker_base_oof_predictions_v10_2.csv", dtype={"ticker": str})
    base = base[["source_row_id", "ticker", "fold_id", "base_score_raw"]]
    data = frame.merge(lead_features, on=["date", "ticker"], how="inner").merge(base, on=["source_row_id", "ticker"], how="left")
    feature_columns = [column for column in lead_features.columns if column not in {"date", "ticker"}]
    prediction_rows, metric_rows = [], []
    for ticker, ticker_frame in data.groupby("ticker", sort=True):
        ticker_columns = [column for column in feature_columns if column in ticker_frame and (not column.startswith("lead_") or ticker_frame[column].notna().any())]
        for fold in folds:
            if fold.fold_id < 3: continue
            validation = ticker_frame.loc[ticker_frame["fold_id"].eq(fold.fold_id)].copy()
            training = ticker_frame.loc[ticker_frame["fold_id"].lt(fold.fold_id) & ticker_frame["fold_id"].ge(0)].copy()
            if validation.empty: continue
            for profile in ["BASE_TICKER", "BASE_PLUS_LEAD_LAG"]:
                if profile == "BASE_TICKER":
                    score = validation["base_score_raw"].to_numpy(float); status = "OK"; feature_count = 0
                elif len(training) >= 120 and len(ticker_columns) >= 2 and training["label_abs_surge_3d_5pct"].nunique() == 2:
                    columns = ["base_score_raw", *ticker_columns]
                    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.25, class_weight="balanced", max_iter=500, random_state=17))
                    model.fit(training[columns], training["label_abs_surge_3d_5pct"].astype(int))
                    score = model.predict_proba(validation[columns])[:, 1]; status = "OK"; feature_count = len(columns)
                else:
                    score = np.full(len(validation), np.nan); status = "LOW_EVIDENCE_NO_MODEL"; feature_count = len(ticker_columns)
                y = validation["label_abs_surge_3d_5pct"].to_numpy(int)
                metrics = binary_metrics(y, score); policy = best_precision_policy(y, score, target_precision=target_precision, minimum_alerts=3)
                metric_rows.append({"ticker": ticker, "fold_id": fold.fold_id, "role": role_for_fold(fold.fold_id), "profile": profile,
                                    "status": status, "feature_count": feature_count, **metrics, **{f"policy_{key}": value for key, value in policy.items()}})
                for row, value in zip(validation.itertuples(index=False), score):
                    prediction_rows.append({"source_row_id": int(row.source_row_id), "ticker": ticker, "fold_id": fold.fold_id, "role": role_for_fold(fold.fold_id),
                                            "profile": profile, "target": int(row.label_abs_surge_3d_5pct), "score": float(value) if math.isfinite(value) else math.nan, "status": status})
    predictions, metrics = pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)
    selection = metrics.loc[metrics["fold_id"].isin([3, 4]) & metrics["status"].eq("OK")]
    champions = []
    for ticker, part in selection.groupby("ticker", sort=True):
        agg = part.groupby("profile", as_index=False).agg(mean_pr_auc=("pr_auc", "mean"), worst_pr_auc=("pr_auc", "min"), folds=("fold_id", "nunique"))
        agg = agg.loc[agg["folds"].ge(2)].sort_values(["mean_pr_auc", "worst_pr_auc", "profile"], ascending=[False, False, True], kind="mergesort")
        champions.append({"ticker": ticker, "champion_profile": str(agg.iloc[0]["profile"]) if len(agg) else "BASE_TICKER"})
    champions_frame = pd.DataFrame(champions)
    routed = predictions.merge(champions_frame, on="ticker", how="inner")
    routed = routed.loc[routed["profile"].eq(routed["champion_profile"])]
    portfolios = []
    for fold_id, part in routed.groupby("fold_id", sort=True):
        metrics_row = binary_metrics(part["target"].to_numpy(float), part["score"].to_numpy(float))
        policy = best_precision_policy(part["target"].to_numpy(float), part["score"].to_numpy(float), target_precision=target_precision, minimum_alerts=minimum_alerts)
        portfolios.append({"fold_id": int(fold_id), "role": role_for_fold(int(fold_id)), **metrics_row, **{f"policy_{key}": value for key, value in policy.items()}})
    return predictions, metrics, champions_frame, pd.DataFrame(portfolios)


def save_graphs_and_plots(output: Path, universe: Sequence[str], metadata: pd.DataFrame, contemporaneous: pd.DataFrame, lag_summary: pd.DataFrame, edges: pd.DataFrame, rolling: pd.DataFrame, event: pd.DataFrame) -> None:
    meta = metadata.set_index("ticker")
    strong = edges.loc[edges["strong_directed_edge"].astype(bool)]
    directed = nx.DiGraph()
    undirected = nx.Graph()
    for ticker in universe:
        attributes = meta.loc[ticker].to_dict() if ticker in meta.index else {}
        directed.add_node(ticker, **{key: str(value) for key, value in attributes.items()})
        undirected.add_node(ticker, **{key: str(value) for key, value in attributes.items()})
    for row in strong.itertuples(index=False):
        directed.add_edge(row.leader, row.follower, lag=int(row.directed_lag), strength=float(row.consensus_score), correlation=float(row.discovery_best_lag_correlation))
    contemp = contemporaneous.sort_values("discovery_pearson", key=lambda s: s.abs(), ascending=False).head(200)
    for row in contemp.itertuples(index=False):
        if math.isfinite(row.discovery_pearson): undirected.add_edge(row.ticker_a, row.ticker_b, correlation=float(row.discovery_pearson))
    nx.write_graphml(directed, output / "ticker_lead_lag_network.graphml")
    nx.write_graphml(undirected, output / "ticker_contemporaneous_network.graphml")

    def heatmap(matrix: np.ndarray, title: str, path: Path, *, text: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(14, 12)); image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(universe))); ax.set_yticks(range(len(universe)))
        ax.set_xticklabels(universe, rotation=90, fontsize=5); ax.set_yticklabels(universe, fontsize=5); ax.set_title(title); fig.colorbar(image, ax=ax, shrink=.7)
        if text and len(universe) <= 15:
            for i in range(len(universe)):
                for j in range(len(universe)): ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=6)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    corr = np.eye(len(universe)); sign = np.eye(len(universe)); lag = np.zeros((len(universe), len(universe))); strength = np.zeros_like(lag)
    position = {ticker: index for index, ticker in enumerate(universe)}
    for row in contemporaneous.itertuples(index=False):
        a, b = position[row.ticker_a], position[row.ticker_b]; corr[a,b]=corr[b,a]=row.discovery_pearson; sign[a,b]=sign[b,a]=row.discovery_sign_agreement
    primary = lag_summary.loc[lag_summary["variant"].eq("bucket_residual")]
    for row in primary.itertuples(index=False):
        a,b=position[row.ticker_a],position[row.ticker_b]; lag[a,b]=row.discovery_best_lag; lag[b,a]=-row.discovery_best_lag; strength[a,b]=strength[b,a]=row.discovery_abs_strength
    heatmap(corr,"Discovery contemporaneous return correlation",output/"ticker_contemporaneous_correlation_heatmap.png")
    heatmap(sign,"Discovery direction agreement",output/"ticker_direction_agreement_heatmap.png")
    heatmap(lag,"Best lag (+ means row leads column)",output/"ticker_best_lag_heatmap.png",text=True)
    heatmap(strength,"Discovery lead-lag strength",output/"ticker_lead_strength_heatmap.png")
    fig, ax = plt.subplots(figsize=(15, 12)); pos = nx.spring_layout(directed, seed=17)
    nx.draw_networkx_nodes(directed,pos,ax=ax,node_size=350,node_color="#4c78a8"); nx.draw_networkx_labels(directed,pos,ax=ax,font_size=6,font_color="white")
    nx.draw_networkx_edges(directed,pos,ax=ax,arrows=True,width=[1+3*directed[u][v]["strength"] for u,v in directed.edges()],alpha=.6)
    labels={(u,v):f"+{directed[u][v]['lag']}d" for u,v in directed.edges()}; nx.draw_networkx_edge_labels(directed,pos,edge_labels=labels,font_size=5,ax=ax)
    ax.set_title("V11 strong directed lead-lag network"); ax.axis("off"); fig.tight_layout(); fig.savefig(output/"ticker_lead_lag_network.png",dpi=160); plt.close(fig)

    # Keep detailed timelines and event curves to the strongest ten pairs so
    # full-universe output remains practical while retaining inspectable cases.
    timeline_root=output/"pair_relationship_timeline"; response_root=output/"event_response"
    timeline_root.mkdir(exist_ok=True); response_root.mkdir(exist_ok=True)
    for root in [timeline_root, response_root]:
        for stale_plot in root.glob("*.png"):
            stale_plot.unlink()
    top_edges=edges.sort_values(["strong_directed_edge","consensus_score"],ascending=[False,False],kind="mergesort").head(10)
    for edge in top_edges.itertuples(index=False):
        pair=rolling.loc[(rolling["ticker_a"].isin([edge.leader,edge.follower])) & (rolling["ticker_b"].isin([edge.leader,edge.follower])) & rolling["window"].eq(60)].sort_values("date")
        if not pair.empty:
            fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
            axes[0].plot(pd.to_datetime(pair["date"]),pair["contemporaneous_correlation"]); axes[0].set_ylabel("corr")
            axes[1].plot(pd.to_datetime(pair["date"]),pair["best_lag"]); axes[1].set_ylabel("best lag")
            axes[2].plot(pd.to_datetime(pair["date"]),pair["spread_z"]); axes[2].axhline(2,color="red",ls="--"); axes[2].axhline(-2,color="red",ls="--"); axes[2].set_ylabel("spread z")
            fig.suptitle(f"{edge.leader} -> {edge.follower} rolling relationship"); fig.tight_layout(); fig.savefig(timeline_root/f"{edge.leader}_TO_{edge.follower}_timeline.png",dpi=130); plt.close(fig)
        current=event.loc[event["leader"].eq(edge.leader)&event["follower"].eq(edge.follower)&event["role"].eq("discovery")]
        for shock in ["UP_2PCT","DOWN_2PCT"]:
            curve=current.loc[current["shock"].eq(shock)].sort_values("horizon")
            if curve.empty: continue
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot(curve["horizon"],curve["mean_response"],marker="o"); ax.axhline(0,color="black",lw=.8)
            ax.set_xlabel("trading-day horizon"); ax.set_ylabel("mean bucket-residual response"); ax.set_title(f"{edge.leader} -> {edge.follower} {shock}"); fig.tight_layout()
            suffix="upshock" if shock=="UP_2PCT" else "downshock"; fig.savefig(response_root/f"{edge.leader}_TO_{edge.follower}_{suffix}.png",dpi=130); plt.close(fig)


def run_pipeline(args: Any) -> dict[str, Any]:
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    atomic_json(status_path, {"schema": SCHEMA_VERSION, "status": "RUNNING", "started_utc": utc_now()})
    try:
        folds = load_folds(Path(args.folds))
        frame = load_frame(Path(args.dataset), Path(args.target_sidecar))
        universe = choose_universe(frame, folds, tickers=args.tickers, top_n=args.top_n, all_tickers=args.all_tickers)
        frame = frame.loc[frame["ticker"].isin(universe)].copy()
        dates, returns, target_panel, source_ids = make_panel(frame, universe)
        metadata = ticker_metadata(frame, universe)
        atomic_csv(output / "ticker_universe.csv", metadata)
        pairs = list(itertools.combinations(range(len(universe)), 2))
        input_hashes = {"dataset": sha256_file(Path(args.dataset)), "target": sha256_file(Path(args.target_sidecar)), "folds": sha256_file(Path(args.folds))}
        parameters = {key: value for key, value in vars(args).items() if key not in {"resume"}}
        cache_meta = {"schema": SCHEMA_VERSION, "input_hashes": input_hashes, "universe_hash": stable_hash(universe),
                      "fold_hash": stable_hash([fold.__dict__ for fold in folds]), "parameter_hash": stable_hash(parameters), "tickers": universe}
        previous_meta_path = output / "CACHE_META_V11.json"
        if args.resume and previous_meta_path.exists():
            previous = json.loads(previous_meta_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != cache_meta.get(key) for key in ["input_hashes","universe_hash","fold_hash","parameter_hash"]):
                raise RuntimeError("--resume cache metadata mismatch")
        atomic_json(previous_meta_path, cache_meta)
        log(f"loaded rows={len(frame)} tickers={len(universe)} pairs={len(pairs)}")
        fold_payload = build_fold_variants(dates, returns, metadata, folds)

        lag_path = output / "lead_lag_cross_correlation_by_fold.csv"
        direction_fold_path = output / "_direction_by_fold_cache.csv"
        tail_fold_path = output / "_tail_by_fold_cache.csv"
        if args.resume and all(path.exists() for path in [lag_path,direction_fold_path,tail_fold_path]):
            lag_by_fold=pd.read_csv(lag_path,dtype={"ticker_a":str,"ticker_b":str}); direction_fold=pd.read_csv(direction_fold_path,dtype={"ticker_a":str,"ticker_b":str}); tail_fold=pd.read_csv(tail_fold_path,dtype={"ticker_a":str,"ticker_b":str})
            log("resumed pair maps")
        else:
            lag_by_fold,direction_fold,tail_fold=compute_pair_maps(universe,pairs,fold_payload,args.threads,args.minimum_pair_observations)
            atomic_csv(lag_path,lag_by_fold); atomic_csv(direction_fold_path,direction_fold); atomic_csv(tail_fold_path,tail_fold)
        lag_summary=summarize_lags(lag_by_fold); atomic_csv(output/"lead_lag_cross_correlation_summary.csv",lag_summary)
        contemporaneous=summarize_role_frame(direction_fold,["pearson","spearman","sign_agreement","up_up_probability","down_down_probability","opposite_probability","phi"],["ticker_a","ticker_b"])
        atomic_csv(output/"ticker_pair_contemporaneous_map.csv",contemporaneous[[c for c in contemporaneous.columns if "pearson" in c or "spearman" in c or c in ["ticker_a","ticker_b","discovery_folds","development_folds","confirmation_folds","recent_audit_folds"]]])
        atomic_csv(output/"ticker_pair_direction_agreement.csv",contemporaneous[[c for c in contemporaneous.columns if any(token in c for token in ["sign_agreement","up_up","down_down","opposite","phi"]) or c in ["ticker_a","ticker_b"]]])
        tail_summary=summarize_role_frame(tail_fold,["upper_tail_dependence","lower_tail_dependence","joint_up_2pct","joint_down_2pct","joint_up_5pct"],["ticker_a","ticker_b"]); atomic_csv(output/"ticker_pair_tail_dependence.csv",tail_summary)

        directional_path=output/"lagged_directional_lift.csv"; tail_lift_path=output/"lagged_tail_event_lift.csv"
        if args.resume and directional_path.exists() and tail_lift_path.exists():
            directional=pd.read_csv(directional_path,dtype={"leader":str,"follower":str}); tail_lift=pd.read_csv(tail_lift_path,dtype={"leader":str,"follower":str}); log("resumed lagged lifts")
        else:
            directional,tail_lift=compute_lagged_lifts(universe,fold_payload,target_panel,args.threads); atomic_csv(directional_path,directional); atomic_csv(tail_lift_path,tail_lift)

        event_path=output/"event_response_curves.csv"; granger_path=output/"pairwise_granger_diagnostics.csv"; mi_path=output/"lagged_mutual_information.csv"
        if args.resume and all(path.exists() for path in [event_path,granger_path,mi_path]):
            event=pd.read_csv(event_path,dtype={"leader":str,"follower":str}); granger=pd.read_csv(granger_path,dtype={"leader":str,"follower":str}); mutual=pd.read_csv(mi_path,dtype={"leader":str,"follower":str}); log("resumed nonlinear diagnostics")
        else:
            event,granger,mutual=compute_event_and_nonlinear(universe,fold_payload,args.threads); atomic_csv(event_path,event); atomic_csv(granger_path,granger); atomic_csv(mi_path,mutual)
        event_summary=event.loc[event["role"].eq("discovery")].copy(); atomic_csv(output/"event_response_summary.csv",event_summary)

        rolling_path=output/"rolling_lead_lag.csv"
        if args.resume and rolling_path.exists(): rolling=pd.read_csv(rolling_path,dtype={"ticker_a":str,"ticker_b":str},parse_dates=["date"]); log("resumed rolling map")
        else: rolling=compute_rolling(universe,returns,pairs,[60,120,252],args.rolling_step,args.threads); atomic_csv(rolling_path,rolling)
        atomic_csv(output/"rolling_pair_state.csv",rolling[["date","ticker_a","ticker_b","window","contemporaneous_correlation","best_lag","lead_strength","spread_z","relation_state","uses_centered_window"]])
        divergence=divergence_summary(rolling); atomic_csv(output/"ticker_pair_divergence_map.csv",divergence)
        self_map=compute_self_map(universe,fold_payload,target_panel); atomic_csv(output/"ticker_self_leadlag_map.csv",self_map)
        lead_surge=tail_lift.rename(columns={"follower_surge3d_conditional":"lead_surge_probability","follower_surge3d_baseline":"baseline_surge_probability","surge3d_lift_ratio":"lead_surge_lift"})
        atomic_csv(output/"lead_to_surge3d_map.csv",lead_surge)

        edges=build_directed_network(lag_summary,directional,tail_lift,granger,mutual); atomic_csv(output/"directed_network_edges.csv",edges)
        scores=leader_scores(universe,edges); nodes=metadata.merge(scores,on="ticker",how="left"); atomic_csv(output/"ticker_leader_follower_scores.csv",scores); atomic_csv(output/"directed_network_nodes.csv",nodes)
        cascade_frame=cascades(edges); atomic_csv(output/"lead_lag_cascades.csv",cascade_frame)
        features=feature_manifest(edges,args.maximum_incoming_features); atomic_json(output/"LEADLAG_FEATURE_MANIFEST_V11.json",features)
        lead_feature_frame=build_lead_feature_frame(frame,features,edges)
        probe_predictions=probe_metrics=probe_champions=probe_portfolio=pd.DataFrame()
        if args.run_probe:
            probe_predictions,probe_metrics,probe_champions,probe_portfolio=run_probe(frame,folds,lead_feature_frame,Path(args.v10_2_output),args.target_precision,args.minimum_portfolio_alerts)
            atomic_csv(output/"leadlag_probe_predictions.csv",probe_predictions); atomic_csv(output/"leadlag_probe_metrics.csv",probe_metrics); atomic_csv(output/"leadlag_probe_champions.csv",probe_champions); atomic_csv(output/"leadlag_probe_portfolio_by_fold.csv",probe_portfolio)

        save_graphs_and_plots(output,universe,metadata,contemporaneous,lag_summary,edges,rolling,event)
        strong=edges.loc[edges["strong_directed_edge"].astype(bool)]
        recommendation={"schema":SCHEMA_VERSION,"status":"LEADLAG_MAP_COMPLETE","rows":len(frame),"tickers":len(universe),"pairs":len(pairs),
                        "strong_directed_edges":len(strong),"confirmation_supported_edges":int(strong["confirmation_same_direction"].astype(bool).sum()) if len(strong) else 0,
                        "recent_supported_edges":int(strong["recent_audit_same_direction"].astype(bool).sum()) if len(strong) else 0,
                        "cascades":len(cascade_frame),"industry_residual_used":False,"bucket_residual_used":True,"probe_run":bool(args.run_probe),
                        "probe_portfolio_gate_passes":int(probe_portfolio["policy_gate_pass"].astype(bool).sum()) if not probe_portfolio.empty else 0,
                        "causal_claim":False,"production_action":"RESEARCH_MAP_ONLY_REVIEW_FUTURE_HOLDOUT"}
        atomic_json(output/"LEADLAG_RECOMMENDATION_V11.json",recommendation)
        manifest={"schema":SCHEMA_VERSION,"created_utc":utc_now(),"inputs":input_hashes,"fold_contract":{"discovery":[0,1,2],"development":[3,4],"confirmation":[5,6],"recent_audit":[7]},
                  "target":"label_abs_surge_3d_5pct","return_source":"t_price_ret_1 with return_pct/100 fallback","return_level_correlation_forbidden":True,
                  "residual_parameters_train_only":True,"rolling_windows_trailing_only":True,"rolling_step":args.rolling_step,"threads":args.threads,
                  "universe":universe,"parameters":parameters,"recommendation":recommendation}
        atomic_json(output/"LEADLAG_MANIFEST_V11.json",manifest)
        top_edges=edges.head(20)[[c for c in ["leader","follower","directed_lag","discovery_best_lag_correlation","consensus_score","edge_status","confirmation_same_direction","recent_audit_same_direction"] if c in edges]].to_markdown(index=False)
        top_leaders=scores.head(15).to_markdown(index=False)
        report=f"""# CrashWatch Surge V11 Lead-Lag Map Report

## Summary

- Rows: {len(frame):,}; tickers: {len(universe)}; unordered pairs: {len(pairs):,}
- Strong directed edges: {len(strong):,}
- Confirmation same-direction strong edges: {recommendation['confirmation_supported_edges']:,}
- Recent same-direction strong edges: {recommendation['recent_supported_edges']:,}
- Cascades: {len(cascade_frame):,}
- Industry metadata was too sparse, so market and bucket residuals were used.
- This is predictive lead-lag evidence, not proof of economic causality.

## Top directed candidates

{top_edges}

## Leader / follower ranking

{top_leaders}

## Leakage contract

Direction and lag were discovered only on folds 0-2, frozen for folds 3-4, and diagnosed on folds 5-7. Residual coefficients were estimated on each fold's train interval. Rolling windows are trailing-only and never centered.
"""
        atomic_text(output/"LEADLAG_MAP_REPORT_KO.md",report)
        excluded={"RUN_STATUS.json","OUTPUT_INVENTORY_V11.json"}; inventory=[]
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in excluded: inventory.append({"file":str(path.relative_to(output)).replace("\\", "/"),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
        atomic_json(output/"OUTPUT_INVENTORY_V11.json",inventory)
        atomic_json(status_path,{"schema":SCHEMA_VERSION,"status":"SUCCESS","finished_utc":utc_now(),"recommendation":recommendation,"files":len(inventory)})
        log(json.dumps(recommendation,ensure_ascii=False))
        return recommendation
    except BaseException as exc:
        atomic_json(status_path,{"schema":SCHEMA_VERSION,"status":"FAILED","finished_utc":utc_now(),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        raise
