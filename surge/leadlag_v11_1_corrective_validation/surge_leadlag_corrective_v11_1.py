from __future__ import annotations

import itertools
import json
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except ImportError:  # unit/synthetic tests may use CSV without parquet support
    pq = None
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surge_leadlag_common_v11_1 import (
    SCHEMA_VERSION,
    FoldSpec,
    alignment_offsets,
    atomic_csv,
    atomic_json,
    atomic_text,
    bh_fdr,
    binary_metrics,
    finite_corr,
    lag_align,
    load_folds,
    normalize_ticker,
    role_for_fold,
    rowwise_corr_against_vector,
    select_frozen_portfolio_threshold,
    sha256_file,
    stable_int_seed,
    weighted_mean,
)


def log(message: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] V11.1 {message}", flush=True)


def _table_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        if pq is None:
            raise ImportError("pyarrow is required to read Parquet inputs. Install requirements_v11_1.txt")
        return list(pq.ParquetFile(path).schema.names)
    if suffix in {".csv", ".txt"}:
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported table format: {path}")


def _read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    raise ValueError(f"Unsupported table format: {path}")


def load_frame(dataset: Path, target_sidecar: Path) -> pd.DataFrame:
    requested = [
        "date",
        "ticker",
        "name",
        "market",
        "bucket",
        "industry_name",
        "market_cap",
        "t_price_ret_1",
        "return_pct",
    ]
    available = set(_table_columns(dataset))
    columns = [column for column in requested if column in available]
    required = {"date", "ticker"}
    if not required.issubset(columns):
        raise RuntimeError(f"Dataset missing required columns: {sorted(required - set(columns))}")
    source = _read_table(dataset, columns=columns).reset_index(drop=True)
    for column, default in [
        ("name", "UNKNOWN"),
        ("market", "UNKNOWN"),
        ("bucket", "UNKNOWN"),
        ("industry_name", "UNKNOWN"),
        ("market_cap", np.nan),
        ("t_price_ret_1", np.nan),
        ("return_pct", np.nan),
    ]:
        if column not in source:
            source[column] = default
    source["source_row_id"] = np.arange(len(source), dtype=np.int64)
    source["ticker"] = source["ticker"].map(normalize_ticker)
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    primary = pd.to_numeric(source["t_price_ret_1"], errors="coerce")
    fallback = pd.to_numeric(source["return_pct"], errors="coerce") / 100.0
    source["return_1d"] = primary.where(primary.notna(), fallback)

    target = _read_table(target_sidecar).copy()
    target["ticker"] = target["ticker"].map(normalize_ticker)
    target["date"] = pd.to_datetime(target["date"], errors="raise")
    target_columns = ["source_row_id", "date", "ticker", "label_abs_surge_3d_5pct", "target_valid"]
    missing_target = [column for column in target_columns if column not in target]
    if missing_target:
        raise RuntimeError(f"Target sidecar missing columns: {missing_target}")
    keep = target_columns + (["best_forward_return_3d"] if "best_forward_return_3d" in target else [])
    merged = source.merge(target[keep], on=["source_row_id", "date", "ticker"], how="inner", validate="one_to_one")
    merged = merged.loc[
        merged["target_valid"].astype(bool) & merged["label_abs_surge_3d_5pct"].isin([0, 1])
    ].copy()
    if merged.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Duplicate ticker/date rows after target merge")
    return merged.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def choose_universe(frame: pd.DataFrame, tickers: str = "") -> list[str]:
    available = sorted(frame["ticker"].astype(str).unique())
    if not tickers:
        return available
    requested = [normalize_ticker(value) for value in tickers.split(",") if value.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown tickers: {missing}")
    return list(dict.fromkeys(requested))


def make_panel(
    frame: pd.DataFrame, universe: Sequence[str]
) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = frame.loc[frame["ticker"].isin(universe)].copy()
    returns = (
        selected.pivot(index="date", columns="ticker", values="return_1d")
        .reindex(columns=universe)
        .sort_index()
    )
    target = selected.pivot(index="date", columns="ticker", values="label_abs_surge_3d_5pct").reindex(
        index=returns.index, columns=universe
    )
    source_ids = selected.pivot(index="date", columns="ticker", values="source_row_id").reindex(
        index=returns.index, columns=universe
    )
    return pd.DatetimeIndex(returns.index), returns, target, source_ids


def ticker_metadata(frame: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in universe:
        part = frame.loc[frame["ticker"].eq(ticker)].sort_values("date")

        def latest(column: str) -> str:
            values = part[column].dropna().astype(str)
            return values.iloc[-1] if len(values) else "UNKNOWN"

        rows.append(
            {
                "ticker": ticker,
                "name": latest("name"),
                "market": latest("market"),
                "bucket": latest("bucket"),
                "industry": latest("industry_name"),
                "rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_factor(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=str)
    output = np.full_like(values, np.nan)
    for group in sorted(set(groups)):
        positions = np.flatnonzero(groups == group)
        block = values[:, positions]
        valid = np.isfinite(block)
        total = np.nansum(block, axis=1)
        count = valid.sum(axis=1)
        for local, position in enumerate(positions):
            denominator = count - valid[:, local]
            numerator = total - np.where(valid[:, local], block[:, local], 0.0)
            output[:, position] = np.divide(
                numerator,
                denominator,
                out=np.full(len(values), np.nan),
                where=denominator > 0,
            )
    return output


def factor_arrays(returns: pd.DataFrame, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = returns.to_numpy(float)
    indexed = metadata.set_index("ticker").reindex(returns.columns)
    markets = indexed["market"].fillna("UNKNOWN").astype(str).to_numpy()
    buckets = indexed["bucket"].fillna("UNKNOWN").astype(str).to_numpy()
    market_factor = leave_one_out_factor(raw, markets)
    bucket_factor = leave_one_out_factor(raw, buckets)
    return raw, market_factor, bucket_factor


def fit_residual_series(
    y: np.ndarray,
    market_factor: np.ndarray,
    bucket_factor: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    *,
    minimum_fit: int = 60,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    market_factor = np.asarray(market_factor, dtype=float)
    bucket_factor = np.asarray(bucket_factor, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    eval_indices = np.asarray(eval_indices, dtype=int)
    x_train = np.column_stack([market_factor[train_indices], bucket_factor[train_indices]])
    y_train = y[train_indices]
    valid = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
    if int(valid.sum()) < int(minimum_fit):
        return np.full(len(eval_indices), np.nan)
    design = np.column_stack([np.ones(valid.sum()), x_train[valid]])
    coefficients, *_ = np.linalg.lstsq(design, y_train[valid], rcond=None)
    x_eval = np.column_stack([market_factor[eval_indices], bucket_factor[eval_indices]])
    result = np.full(len(eval_indices), np.nan)
    good = np.isfinite(y[eval_indices]) & np.all(np.isfinite(x_eval), axis=1)
    if good.any():
        result[good] = y[eval_indices][good] - np.column_stack([np.ones(good.sum()), x_eval[good]]) @ coefficients
    return result


def build_fold_residual_payload(
    dates: pd.DatetimeIndex,
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
    folds: Sequence[FoldSpec],
) -> tuple[dict[int, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    raw, market_factor, bucket_factor = factor_arrays(returns, metadata)
    payload: dict[int, dict[str, Any]] = {}
    all_indices = np.arange(len(dates), dtype=int)
    for fold in folds:
        train = np.flatnonzero((dates >= fold.train_start) & (dates <= fold.train_end))
        validation = np.flatnonzero((dates >= fold.validation_start) & (dates <= fold.validation_end))
        residual_all = np.full_like(raw, np.nan)
        for position in range(raw.shape[1]):
            residual_all[:, position] = fit_residual_series(
                raw[:, position],
                market_factor[:, position],
                bucket_factor[:, position],
                train,
                all_indices,
                minimum_fit=60,
            )
        payload[fold.fold_id] = {
            "fold": fold,
            "indices": validation,
            "dates": dates[validation],
            "bucket_residual": residual_all[validation],
            "bucket_residual_all": residual_all,
        }
    return payload, raw, market_factor, bucket_factor


def _weighted_lag_stats(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    lags: Sequence[int],
    minimum: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lag in lags:
        correlations: list[float] = []
        observations: list[int] = []
        for x0, y0 in segments:
            x, y = lag_align(x0, y0, int(lag))
            corr, n = finite_corr(x, y, minimum=minimum)
            correlations.append(corr)
            observations.append(n)
        valid = np.isfinite(correlations)
        mean_corr = weighted_mean(correlations, observations)
        consistency = (
            float(np.mean(np.sign(np.asarray(correlations)[valid]) == np.sign(mean_corr)))
            if valid.any() and math.isfinite(mean_corr)
            else math.nan
        )
        rows.append(
            {
                "lag": int(lag),
                "correlation": mean_corr,
                "abs_correlation": abs(mean_corr) if math.isfinite(mean_corr) else -1.0,
                "valid_folds": int(valid.sum()),
                "observations": int(sum(n for corr, n in zip(correlations, observations) if math.isfinite(corr))),
                "sign_consistency": consistency,
            }
        )
    return rows


def block_permutation_matrix(
    y: np.ndarray,
    permutations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    blocks = [np.arange(start, min(start + block_size, len(y)), dtype=int) for start in range(0, len(y), block_size)]
    output = np.empty((permutations, len(y)), dtype=float)
    for row in range(permutations):
        order = rng.permutation(len(blocks))
        indices = np.concatenate([blocks[index] for index in order])
        output[row] = y[indices]
    return output


def _estimate_maxstat_empirical_p(
    discovery_segments: Sequence[tuple[np.ndarray, np.ndarray]],
    observed_abs_stat: float,
    *,
    permutations: int,
    block_size: int,
    minimum: int,
    seed: int,
    batch_size: int = 256,
) -> tuple[float, int, int]:
    """
    Moving-block max-stat null. One shuffled follower path is reused across all 11 lags
    inside a permutation, so the null statistic is genuinely max_{h=-5..5}|corr_h|.
    """
    if not math.isfinite(observed_abs_stat) or permutations <= 0:
        return math.nan, 0, 0
    lags = list(range(-5, 6))
    rng = np.random.default_rng(seed)
    exceedances = 0
    valid_total = 0
    remaining = int(permutations)
    while remaining > 0:
        current = min(int(batch_size), remaining)
        shuffled_segments = [
            (x0, block_permutation_matrix(y0, current, block_size, rng))
            for x0, y0 in discovery_segments
        ]
        null_lag_values = np.full((current, len(lags)), np.nan, dtype=float)
        for lag_index, lag in enumerate(lags):
            weighted_sum = np.zeros(current, dtype=float)
            weight_sum = np.zeros(current, dtype=float)
            for x0, permuted in shuffled_segments:
                if lag > 0:
                    x = x0[:-lag]
                    y = permuted[:, lag:]
                elif lag < 0:
                    k = -lag
                    x = x0[k:]
                    y = permuted[:, :-k]
                else:
                    x = x0
                    y = permuted
                corr, n = rowwise_corr_against_vector(x, y, minimum=minimum)
                valid = np.isfinite(corr)
                weighted_sum[valid] += corr[valid] * n[valid]
                weight_sum[valid] += n[valid]
            good = weight_sum > 0
            null_lag_values[good, lag_index] = weighted_sum[good] / weight_sum[good]
        absolute_null = np.abs(null_lag_values)
        valid_any = np.isfinite(absolute_null).any(axis=1)
        null_max = np.full(current, np.nan, dtype=float)
        if valid_any.any():
            null_max[valid_any] = np.nanmax(absolute_null[valid_any], axis=1)
        valid_null = np.isfinite(null_max)
        exceedances += int(np.sum(null_max[valid_null] >= observed_abs_stat))
        valid_total += int(valid_null.sum())
        remaining -= current
    empirical_p = float((1 + exceedances) / (1 + valid_total)) if valid_total else math.nan
    return empirical_p, valid_total, exceedances


def maxstat_pair_test(
    ticker_a: str,
    ticker_b: str,
    position_a: int,
    position_b: int,
    payload: Mapping[int, Mapping[str, Any]],
    *,
    screening_permutations: int,
    adaptive_permutations: int,
    adaptive_trigger_p: float,
    block_size: int,
    minimum: int,
    seed: int,
) -> dict[str, Any]:
    lags = list(range(-5, 6))
    discovery_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in [0, 1, 2]:
        if fold_id not in payload:
            continue
        matrix = payload[fold_id]["bucket_residual"]
        discovery_segments.append((matrix[:, position_a], matrix[:, position_b]))
    lag_stats = _weighted_lag_stats(discovery_segments, lags=lags, minimum=minimum)
    eligible = [row for row in lag_stats if row["valid_folds"] >= 2]
    if eligible:
        best = max(eligible, key=lambda row: (row["abs_correlation"], -abs(row["lag"]), -row["lag"]))
    else:
        best = {
            "lag": 0,
            "correlation": math.nan,
            "abs_correlation": math.nan,
            "valid_folds": 0,
            "observations": 0,
            "sign_consistency": math.nan,
        }

    observed = float(best["abs_correlation"]) if math.isfinite(best["abs_correlation"]) else math.nan
    screen_seed = stable_int_seed(seed, ticker_a, ticker_b, "maxstat_screen")
    empirical_p, valid_permutations, exceedances = _estimate_maxstat_empirical_p(
        discovery_segments,
        observed,
        permutations=screening_permutations,
        block_size=block_size,
        minimum=minimum,
        seed=screen_seed,
    )
    adaptive_refined = bool(
        math.isfinite(empirical_p)
        and empirical_p <= adaptive_trigger_p
        and adaptive_permutations > screening_permutations
    )
    if adaptive_refined:
        refine_seed = stable_int_seed(seed, ticker_a, ticker_b, "maxstat_refine")
        empirical_p, valid_permutations, exceedances = _estimate_maxstat_empirical_p(
            discovery_segments,
            observed,
            permutations=adaptive_permutations,
            block_size=block_size,
            minimum=minimum,
            seed=refine_seed,
        )

    row: dict[str, Any] = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "variant": "bucket_residual",
        "discovery_best_lag": int(best["lag"]),
        "discovery_best_lag_correlation": float(best["correlation"]) if math.isfinite(best["correlation"]) else math.nan,
        "discovery_abs_strength": observed,
        "discovery_sign_consistency": float(best["sign_consistency"]) if math.isfinite(best["sign_consistency"]) else math.nan,
        "discovery_valid_folds": int(best["valid_folds"]),
        "discovery_observations": int(best["observations"]),
        "maxstat_empirical_p": empirical_p,
        "maxstat_permutations": int(valid_permutations),
        "maxstat_exceedances": int(exceedances),
        "adaptive_refined": adaptive_refined,
        "permutation_method": f"moving_block_shuffle_block{block_size}_max_abs_over_11_lags",
    }
    for role, fold_ids in [("development", [3, 4]), ("confirmation", [5, 6]), ("recent_audit", [7])]:
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_id in fold_ids:
            if fold_id in payload:
                matrix = payload[fold_id]["bucket_residual"]
                segments.append((matrix[:, position_a], matrix[:, position_b]))
        fixed_stats = _weighted_lag_stats(segments, lags=[int(best["lag"])], minimum=minimum)[0] if segments else None
        value = float(fixed_stats["correlation"]) if fixed_stats and math.isfinite(fixed_stats["correlation"]) else math.nan
        row[f"{role}_fixed_correlation"] = value
        row[f"{role}_valid_folds"] = int(fixed_stats["valid_folds"]) if fixed_stats else 0
        row[f"{role}_same_direction"] = bool(
            math.isfinite(value)
            and math.isfinite(row["discovery_best_lag_correlation"])
            and np.sign(value) == np.sign(row["discovery_best_lag_correlation"])
        )
    if int(best["lag"]) > 0:
        row.update({"leader": ticker_a, "follower": ticker_b, "directed_lag": int(best["lag"])})
    elif int(best["lag"]) < 0:
        row.update({"leader": ticker_b, "follower": ticker_a, "directed_lag": int(-best["lag"])})
    else:
        row.update({"leader": "", "follower": "", "directed_lag": 0})
    return row


_MAXSTAT_PROCESS_UNIVERSE: Sequence[str] | None = None
_MAXSTAT_PROCESS_PAYLOAD: Mapping[int, Mapping[str, Any]] | None = None
_MAXSTAT_PROCESS_OPTIONS: dict[str, Any] | None = None


def _initialize_maxstat_process(
    universe: Sequence[str],
    payload: Mapping[int, Mapping[str, Any]],
    options: Mapping[str, Any],
) -> None:
    """Install read-only inputs once in each worker instead of serializing them per pair."""
    global _MAXSTAT_PROCESS_UNIVERSE, _MAXSTAT_PROCESS_PAYLOAD, _MAXSTAT_PROCESS_OPTIONS
    _MAXSTAT_PROCESS_UNIVERSE = tuple(universe)
    _MAXSTAT_PROCESS_PAYLOAD = payload
    _MAXSTAT_PROCESS_OPTIONS = dict(options)


def _maxstat_process_one(pair: tuple[int, int]) -> dict[str, Any]:
    if _MAXSTAT_PROCESS_UNIVERSE is None or _MAXSTAT_PROCESS_PAYLOAD is None or _MAXSTAT_PROCESS_OPTIONS is None:
        raise RuntimeError("max-stat process worker was not initialized")
    a, b = pair
    return maxstat_pair_test(
        _MAXSTAT_PROCESS_UNIVERSE[a],
        _MAXSTAT_PROCESS_UNIVERSE[b],
        a,
        b,
        _MAXSTAT_PROCESS_PAYLOAD,
        **_MAXSTAT_PROCESS_OPTIONS,
    )


def run_maxstat_map(
    universe: Sequence[str],
    payload: Mapping[int, Mapping[str, Any]],
    *,
    screening_permutations: int,
    adaptive_permutations: int,
    adaptive_trigger_p: float,
    block_size: int,
    minimum: int,
    threads: int,
    executor_kind: str = "thread",
    seed: int,
    alpha: float,
    minimum_forward_abs_corr: float,
) -> pd.DataFrame:
    pairs = list(itertools.combinations(range(len(universe)), 2))

    def one(pair: tuple[int, int]) -> dict[str, Any]:
        a, b = pair
        return maxstat_pair_test(
            universe[a],
            universe[b],
            a,
            b,
            payload,
            screening_permutations=screening_permutations,
            adaptive_permutations=adaptive_permutations,
            adaptive_trigger_p=adaptive_trigger_p,
            block_size=block_size,
            minimum=minimum,
            seed=seed,
        )

    rows: list[dict[str, Any]] = []
    if executor_kind == "process":
        options = {
            "screening_permutations": screening_permutations,
            "adaptive_permutations": adaptive_permutations,
            "adaptive_trigger_p": adaptive_trigger_p,
            "block_size": block_size,
            "minimum": minimum,
            "seed": seed,
        }
        pool_context = ProcessPoolExecutor(
            max_workers=threads,
            initializer=_initialize_maxstat_process,
            initargs=(tuple(universe), payload, options),
        )
        iterator = pool_context.map(
            _maxstat_process_one,
            pairs,
            chunksize=max(1, len(pairs) // max(1, threads * 16)),
        )
    elif executor_kind == "thread":
        pool_context = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="v11_1-maxstat")
        iterator = pool_context.map(one, pairs)
    else:
        raise ValueError(f"Unsupported max-stat executor: {executor_kind}")
    with pool_context:
        for done, row in enumerate(iterator, start=1):
            rows.append(row)
            if done % 100 == 0 or done == len(pairs):
                log(f"max-stat pairs {done}/{len(pairs)}")
    result = pd.DataFrame(rows)
    result["maxstat_q_value"] = bh_fdr(result["maxstat_empirical_p"].to_numpy(float))
    result["corrected_discovery_dev_candidate"] = (
        result["directed_lag"].gt(0)
        & result["discovery_valid_folds"].ge(2)
        & result["discovery_sign_consistency"].ge(0.80)
        & result["discovery_observations"].ge(100)
        & result["maxstat_q_value"].le(alpha)
        & result["development_same_direction"].astype(bool)
        & result["development_valid_folds"].ge(2)
    )
    result["forward_sign_stable"] = (
        result["corrected_discovery_dev_candidate"].astype(bool)
        & result["confirmation_same_direction"].astype(bool)
        & result["recent_audit_same_direction"].astype(bool)
    )
    result["forward_effect_stable"] = (
        result["forward_sign_stable"].astype(bool)
        & result["development_fixed_correlation"].abs().ge(minimum_forward_abs_corr)
        & result["confirmation_fixed_correlation"].abs().ge(minimum_forward_abs_corr)
        & result["recent_audit_fixed_correlation"].abs().ge(minimum_forward_abs_corr)
    )
    return result.sort_values(
        ["corrected_discovery_dev_candidate", "forward_sign_stable", "maxstat_q_value", "discovery_abs_strength"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def load_v11_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["leader", "follower", "directed_lag"])
    frame = pd.read_csv(path, dtype={"leader": str, "follower": str})
    frame["leader"] = frame["leader"].map(normalize_ticker)
    frame["follower"] = frame["follower"].map(normalize_ticker)
    frame["directed_lag"] = pd.to_numeric(frame["directed_lag"], errors="coerce").fillna(0).astype(int)
    return frame


def mark_v11_reference(result: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = set(zip(reference["leader"], reference["follower"], reference["directed_lag"]))
    output = result.copy()
    output["was_v11_strong"] = [
        (str(leader), str(follower), int(lag)) in keys
        for leader, follower, lag in zip(output["leader"], output["follower"], output["directed_lag"])
    ]
    return output



def recheck_v11_reference_edges(
    reference: pd.DataFrame,
    universe: Sequence[str],
    payload: Mapping[int, Mapping[str, Any]],
    maxstat: pd.DataFrame,
    *,
    minimum: int,
    maxstat_alpha: float,
    minimum_forward_abs_corr: float,
) -> pd.DataFrame:
    """Re-evaluate the original seven V11 edges at their original frozen direction and lag."""
    if reference.empty:
        return pd.DataFrame()
    positions = {ticker: index for index, ticker in enumerate(universe)}
    pair_lookup = {
        tuple(sorted((str(row.ticker_a), str(row.ticker_b)))): row
        for row in maxstat.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for ref in reference.itertuples(index=False):
        leader = str(ref.leader)
        follower = str(ref.follower)
        lag = int(ref.directed_lag)
        if leader not in positions or follower not in positions or lag <= 0:
            continue
        a, b = positions[leader], positions[follower]
        row: dict[str, Any] = {
            "leader": leader,
            "follower": follower,
            "directed_lag": lag,
            "was_v11_strong": True,
            "reference_source": "V11_strong_directed_edges_7",
        }
        for role, fold_ids in [
            ("discovery", [0, 1, 2]),
            ("development", [3, 4]),
            ("confirmation", [5, 6]),
            ("recent_audit", [7]),
        ]:
            segments: list[tuple[np.ndarray, np.ndarray]] = []
            for fold_id in fold_ids:
                if fold_id in payload:
                    matrix = payload[fold_id]["bucket_residual"]
                    segments.append((matrix[:, a], matrix[:, b]))
            stats = _weighted_lag_stats(segments, lags=[lag], minimum=minimum)[0] if segments else None
            corr = float(stats["correlation"]) if stats and math.isfinite(stats["correlation"]) else math.nan
            row[f"{role}_fixed_correlation"] = corr
            row[f"{role}_valid_folds"] = int(stats["valid_folds"]) if stats else 0
            if role == "discovery":
                row["discovery_best_lag_correlation"] = corr
                row["discovery_abs_strength"] = abs(corr) if math.isfinite(corr) else math.nan
                row["discovery_sign_consistency"] = float(stats["sign_consistency"]) if stats and math.isfinite(stats["sign_consistency"]) else math.nan
                row["discovery_observations"] = int(stats["observations"]) if stats else 0
            else:
                row[f"{role}_same_direction"] = bool(
                    math.isfinite(corr)
                    and math.isfinite(row.get("discovery_best_lag_correlation", math.nan))
                    and np.sign(corr) == np.sign(row["discovery_best_lag_correlation"])
                )
        pair_row = pair_lookup.get(tuple(sorted((leader, follower))))
        if pair_row is not None:
            row["pair_discovery_selected_lag_v11_1"] = int(pair_row.discovery_best_lag)
            row["pair_discovery_selected_leader_v11_1"] = str(pair_row.leader)
            row["pair_discovery_selected_follower_v11_1"] = str(pair_row.follower)
            row["pair_maxstat_empirical_p"] = float(pair_row.maxstat_empirical_p)
            row["pair_maxstat_q_value"] = float(pair_row.maxstat_q_value)
            row["pair_adaptive_refined"] = bool(pair_row.adaptive_refined)
        else:
            row["pair_discovery_selected_lag_v11_1"] = math.nan
            row["pair_discovery_selected_leader_v11_1"] = ""
            row["pair_discovery_selected_follower_v11_1"] = ""
            row["pair_maxstat_empirical_p"] = math.nan
            row["pair_maxstat_q_value"] = math.nan
            row["pair_adaptive_refined"] = False
        row["pair_maxstat_significant"] = bool(
            math.isfinite(row["pair_maxstat_q_value"]) and row["pair_maxstat_q_value"] <= maxstat_alpha
        )
        row["reference_forward_sign_stable"] = bool(
            row.get("development_same_direction", False)
            and row.get("confirmation_same_direction", False)
            and row.get("recent_audit_same_direction", False)
        )
        row["reference_forward_effect_stable"] = bool(
            row["reference_forward_sign_stable"]
            and all(
                math.isfinite(row.get(f"{role}_fixed_correlation", math.nan))
                and abs(row[f"{role}_fixed_correlation"]) >= minimum_forward_abs_corr
                for role in ["development", "confirmation", "recent_audit"]
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _role_for_date(date: pd.Timestamp, folds: Sequence[FoldSpec]) -> tuple[int | None, str]:
    for fold in folds:
        if fold.validation_start <= date <= fold.validation_end:
            return fold.fold_id, role_for_fold(fold.fold_id)
    return None, "outside_validation"


def compute_frozen_lag_rolling(
    edges: pd.DataFrame,
    universe: Sequence[str],
    dates: pd.DatetimeIndex,
    raw: np.ndarray,
    market_factor: np.ndarray,
    bucket_factor: np.ndarray,
    folds: Sequence[FoldSpec],
    *,
    windows: Sequence[int],
    step: int,
    residual_fit_days: int,
    residual_min_fit: int,
    minimum_corr_observations: int,
    minimum_forward_abs_corr: float,
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    positions = {ticker: index for index, ticker in enumerate(universe)}
    rows: list[dict[str, Any]] = []
    for edge_no, edge in enumerate(edges.itertuples(index=False), start=1):
        leader = str(edge.leader)
        follower = str(edge.follower)
        lag = int(edge.directed_lag)
        if leader not in positions or follower not in positions or lag <= 0:
            continue
        a = positions[leader]
        b = positions[follower]
        expected_sign = int(np.sign(float(edge.discovery_best_lag_correlation)))
        for window in windows:
            first_end = residual_min_fit + int(window) - 1
            for end in range(first_end, len(dates), int(step)):
                window_start = end - int(window) + 1
                fit_end = window_start
                fit_start = max(0, fit_end - int(residual_fit_days))
                fit_idx = np.arange(fit_start, fit_end, dtype=int)
                if len(fit_idx) < residual_min_fit:
                    continue
                eval_idx = np.arange(window_start, end + 1, dtype=int)
                leader_resid = fit_residual_series(
                    raw[:, a], market_factor[:, a], bucket_factor[:, a], fit_idx, eval_idx, minimum_fit=residual_min_fit
                )
                follower_resid = fit_residual_series(
                    raw[:, b], market_factor[:, b], bucket_factor[:, b], fit_idx, eval_idx, minimum_fit=residual_min_fit
                )
                x, y = lag_align(leader_resid, follower_resid, lag)
                corr, n = finite_corr(x, y, minimum=minimum_corr_observations)
                same_sign = bool(math.isfinite(corr) and expected_sign != 0 and int(np.sign(corr)) == expected_sign)
                effect_pass = bool(same_sign and abs(corr) >= minimum_forward_abs_corr)
                if not math.isfinite(corr):
                    state = "NO_EVIDENCE"
                elif not same_sign:
                    state = "SIGN_FLIP"
                elif effect_pass:
                    state = "SUPPORTED"
                else:
                    state = "WEAK_SAME_SIGN"
                fold_id, role = _role_for_date(pd.Timestamp(dates[end]), folds)
                rows.append(
                    {
                        "date": dates[end],
                        "leader": leader,
                        "follower": follower,
                        "fixed_lag": lag,
                        "window": int(window),
                        "fit_days_requested": int(residual_fit_days),
                        "fit_observations_calendar": int(len(fit_idx)),
                        "observations": int(n),
                        "frozen_lag_correlation": corr,
                        "expected_sign": expected_sign,
                        "same_direction": same_sign,
                        "effect_pass": effect_pass,
                        "state": state,
                        "endpoint_fold_id": fold_id,
                        "endpoint_role": role,
                        "residual_variant": "point_in_time_bucket_residual",
                        "lag_reoptimized": False,
                    }
                )
        if edge_no % 10 == 0 or edge_no == len(edges):
            log(f"frozen-lag rolling edges {edge_no}/{len(edges)}")
    return pd.DataFrame(rows)


def summarize_frozen_rolling(rolling: pd.DataFrame) -> pd.DataFrame:
    if rolling.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (leader, follower, lag, window), part in rolling.groupby(["leader", "follower", "fixed_lag", "window"], sort=True):
        part = part.sort_values("date")
        finite = part.loc[np.isfinite(part["frozen_lag_correlation"])]
        latest = part.iloc[-1]
        rows.append(
            {
                "leader": leader,
                "follower": follower,
                "fixed_lag": int(lag),
                "window": int(window),
                "windows_total": int(len(part)),
                "windows_finite": int(len(finite)),
                "same_direction_rate": float(finite["same_direction"].mean()) if len(finite) else math.nan,
                "effect_pass_rate": float(finite["effect_pass"].mean()) if len(finite) else math.nan,
                "sign_flip_rate": float((finite["state"] == "SIGN_FLIP").mean()) if len(finite) else math.nan,
                "median_abs_correlation": float(finite["frozen_lag_correlation"].abs().median()) if len(finite) else math.nan,
                "latest_date": latest["date"],
                "latest_correlation": latest["frozen_lag_correlation"],
                "latest_state": latest["state"],
                "lag_reoptimized": False,
            }
        )
    return pd.DataFrame(rows)


def _edge_feature_prefix(leader: str, lag: int) -> str:
    return f"ll_{leader}_h{int(lag)}"


def build_target_aligned_feature_frame(
    edges: pd.DataFrame,
    universe: Sequence[str],
    dates: pd.DatetimeIndex,
    raw: np.ndarray,
    target_panel: pd.DataFrame,
    source_ids: pd.DataFrame,
    payload: Mapping[int, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if edges.empty:
        return pd.DataFrame(), {
            "schema": SCHEMA_VERSION,
            "target": "label_abs_surge_3d_5pct",
            "future_offsets_used": False,
            "edges": [],
            "feature_columns": [],
        }
    positions = {ticker: index for index, ticker in enumerate(universe)}
    row_map: dict[tuple[int, str, int], dict[str, Any]] = {}
    manifest_edges: list[dict[str, Any]] = []
    incoming_columns: dict[str, list[tuple[str, float]]] = {}

    for edge in edges.itertuples(index=False):
        leader = str(edge.leader)
        follower = str(edge.follower)
        lag = int(edge.directed_lag)
        if leader not in positions or follower not in positions or lag <= 0:
            continue
        sign = int(np.sign(float(edge.discovery_best_lag_correlation)))
        if sign == 0:
            continue
        mapping = alignment_offsets(lag, target_horizon=3)
        prefix = _edge_feature_prefix(leader, lag)
        weight = abs(float(edge.discovery_best_lag_correlation))
        incoming_columns.setdefault(follower, []).append((f"{prefix}_resid_pressure", weight))
        manifest_edges.append(
            {
                "leader": leader,
                "follower": follower,
                "directed_lag": lag,
                "discovery_correlation_sign": sign,
                "discovery_correlation": float(edge.discovery_best_lag_correlation),
                "alignment": [
                    {"follower_horizon_day": int(k), "leader_offset_from_forecast_t": int(offset)}
                    for k, offset in mapping
                ],
                "all_offsets_nonpositive": all(offset <= 0 for _, offset in mapping),
                "pressure_formula": "sign(discovery_corr) * mean(aligned leader bucket-residual returns available at t)",
            }
        )
        a = positions[leader]
        b = positions[follower]
        for fold_id, fold_payload in payload.items():
            residual_all = fold_payload["bucket_residual_all"]
            for global_idx in fold_payload["indices"]:
                source_id = source_ids.iat[int(global_idx), b]
                target_value = target_panel.iat[int(global_idx), b]
                if not math.isfinite(float(source_id)) or not math.isfinite(float(target_value)):
                    continue
                key = (int(source_id), follower, int(fold_id))
                row = row_map.setdefault(
                    key,
                    {
                        "source_row_id": int(source_id),
                        "date": dates[int(global_idx)],
                        "ticker": follower,
                        "fold_id": int(fold_id),
                        "role": role_for_fold(int(fold_id)),
                    },
                )
                raw_components: list[float] = []
                resid_components: list[float] = []
                for k, offset in mapping:
                    leader_idx = int(global_idx) + int(offset)
                    raw_value = raw[leader_idx, a] if 0 <= leader_idx < len(dates) else math.nan
                    resid_value = residual_all[leader_idx, a] if 0 <= leader_idx < len(dates) else math.nan
                    row[f"{prefix}_raw_d{k}"] = raw_value
                    row[f"{prefix}_resid_d{k}"] = resid_value
                    raw_components.append(raw_value)
                    resid_components.append(resid_value)
                raw_array = np.asarray(raw_components, dtype=float)
                resid_array = np.asarray(resid_components, dtype=float)
                raw_valid = np.isfinite(raw_array)
                resid_valid = np.isfinite(resid_array)
                row[f"{prefix}_raw_pressure"] = float(sign * np.nanmean(raw_array)) if raw_valid.any() else math.nan
                row[f"{prefix}_resid_pressure"] = float(sign * np.nanmean(resid_array)) if resid_valid.any() else math.nan
                row[f"{prefix}_raw_peak"] = float(np.nanmax(sign * raw_array)) if raw_valid.any() else math.nan
                row[f"{prefix}_resid_peak"] = float(np.nanmax(sign * resid_array)) if resid_valid.any() else math.nan
                row[f"{prefix}_aligned_n"] = int(resid_valid.sum())

    features = pd.DataFrame(row_map.values())
    if features.empty:
        return features, {
            "schema": SCHEMA_VERSION,
            "target": "label_abs_surge_3d_5pct",
            "future_offsets_used": False,
            "edges": manifest_edges,
            "feature_columns": [],
        }

    # Follower-level aggregate pressure. Weights are frozen from discovery strength.
    for follower, columns_and_weights in incoming_columns.items():
        mask = features["ticker"].eq(follower)
        indices = features.index[mask]
        for index in indices:
            values: list[float] = []
            weights: list[float] = []
            for column, weight in columns_and_weights:
                if column in features:
                    value = features.at[index, column]
                    if pd.notna(value) and math.isfinite(float(value)):
                        values.append(float(value))
                        weights.append(float(weight))
            if values and sum(weights) > 0:
                features.at[index, "ll_incoming_resid_pressure"] = float(np.average(values, weights=weights))
                features.at[index, "ll_incoming_available_edges"] = int(len(values))
                features.at[index, "ll_incoming_positive_edges"] = int(np.sum(np.asarray(values) > 0))
            else:
                features.at[index, "ll_incoming_resid_pressure"] = math.nan
                features.at[index, "ll_incoming_available_edges"] = 0
                features.at[index, "ll_incoming_positive_edges"] = 0

    key_columns = {"source_row_id", "date", "ticker", "fold_id", "role"}
    feature_columns = [column for column in features.columns if column not in key_columns]
    manifest = {
        "schema": SCHEMA_VERSION,
        "target": "label_abs_surge_3d_5pct",
        "selection_contract": "edge and directed lag selected in discovery; never reoptimized in folds 3-7",
        "future_offsets_used": False,
        "edges": manifest_edges,
        "feature_columns": feature_columns,
    }
    return features.sort_values(["fold_id", "ticker", "date"], kind="mergesort").reset_index(drop=True), manifest


def compute_target_aligned_lift(
    edges: pd.DataFrame,
    features: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    minimum_event_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty or features.empty:
        return pd.DataFrame(), pd.DataFrame()
    labels = frame[["source_row_id", "ticker", "label_abs_surge_3d_5pct"]].copy()
    data = features.merge(labels, on=["source_row_id", "ticker"], how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        leader = str(edge.leader)
        follower = str(edge.follower)
        lag = int(edge.directed_lag)
        pressure_col = f"{_edge_feature_prefix(leader, lag)}_resid_pressure"
        if pressure_col not in data:
            continue
        part = data.loc[data["ticker"].eq(follower) & np.isfinite(data[pressure_col])].copy()
        discovery = part.loc[part["fold_id"].isin([0, 1, 2]), pressure_col].dropna()
        if len(discovery) < 20:
            continue
        thresholds = {"DISCOVERY_P80": float(discovery.quantile(0.80)), "DISCOVERY_P90": float(discovery.quantile(0.90))}
        for threshold_name, threshold in thresholds.items():
            for fold_id, fold_part in part.groupby("fold_id", sort=True):
                y = fold_part["label_abs_surge_3d_5pct"].to_numpy(float)
                pressure = fold_part[pressure_col].to_numpy(float)
                valid = np.isfinite(y) & np.isfinite(pressure)
                baseline = float(np.mean(y[valid])) if valid.any() else math.nan
                event = valid & (pressure >= threshold)
                event_n = int(event.sum())
                conditional = float(np.mean(y[event])) if event_n >= minimum_event_n else math.nan
                rows.append(
                    {
                        "leader": leader,
                        "follower": follower,
                        "directed_lag": lag,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        "threshold_source": "folds_0_2_only",
                        "fold_id": int(fold_id),
                        "role": role_for_fold(int(fold_id)),
                        "observations": int(valid.sum()),
                        "event_n": event_n,
                        "baseline_surge_probability": baseline,
                        "conditional_surge_probability": conditional,
                        "surge_lift_ratio": conditional / baseline if math.isfinite(conditional) and baseline > 0 else math.nan,
                        "surge_lift_difference": conditional - baseline if math.isfinite(conditional) and math.isfinite(baseline) else math.nan,
                        "target_alignment_corrected": True,
                    }
                )
        edge_rows = pd.DataFrame([row for row in rows if row["leader"] == leader and row["follower"] == follower and row["directed_lag"] == lag])
        summary: dict[str, Any] = {"leader": leader, "follower": follower, "directed_lag": lag}
        for threshold_name in ["DISCOVERY_P80", "DISCOVERY_P90"]:
            current = edge_rows.loc[edge_rows["threshold_name"].eq(threshold_name)]
            for role in ["development", "confirmation", "recent_audit"]:
                role_part = current.loc[current["role"].eq(role)]
                weights = role_part["event_n"].to_numpy(float)
                lift = weighted_mean(role_part["surge_lift_ratio"].to_numpy(float), weights)
                summary[f"{threshold_name.lower()}_{role}_lift"] = lift
                summary[f"{threshold_name.lower()}_{role}_event_n"] = int(role_part["event_n"].sum())
        summary["p80_forward_target_supported"] = all(
            math.isfinite(summary.get(f"discovery_p80_{role}_lift", math.nan))
            and summary[f"discovery_p80_{role}_lift"] > 1.0
            and summary.get(f"discovery_p80_{role}_event_n", 0) >= minimum_event_n
            for role in ["development", "confirmation", "recent_audit"]
        )
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def load_base_oof(v10_output: Path) -> pd.DataFrame:
    path = v10_output / "ticker_base_oof_predictions_v10_2.csv"
    if not path.exists():
        raise FileNotFoundError(f"V10.2 base OOF not found: {path}")
    base = pd.read_csv(path, dtype={"ticker": str})
    needed = ["source_row_id", "ticker", "fold_id", "base_score_raw"]
    missing = [column for column in needed if column not in base]
    if missing:
        raise RuntimeError(f"Base OOF missing columns: {missing}")
    base = base[needed].copy()
    base["ticker"] = base["ticker"].map(normalize_ticker)
    return base


def run_matched_probe(
    edge_set_name: str,
    features: pd.DataFrame,
    frame: pd.DataFrame,
    base_oof: pd.DataFrame,
    *,
    target_precision: float,
    minimum_portfolio_alerts: int,
    minimum_training_rows: int,
    minimum_validation_base_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if features.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, {"edge_set": edge_set_name, "status": "NO_FEATURES"}
    labels = frame[["source_row_id", "ticker", "label_abs_surge_3d_5pct"]]
    data = features.merge(labels, on=["source_row_id", "ticker"], how="left", validate="many_to_one")
    data = data.merge(base_oof, on=["source_row_id", "ticker", "fold_id"], how="left", validate="many_to_one")
    key_columns = {"source_row_id", "date", "ticker", "fold_id", "role", "label_abs_surge_3d_5pct", "base_score_raw"}
    feature_columns = [column for column in data.columns if column not in key_columns]

    eligibility_rows: list[dict[str, Any]] = []
    complete_tickers: set[str] = set()
    for ticker, part in data.groupby("ticker", sort=True):
        counts = {
            fold_id: int(np.isfinite(part.loc[part["fold_id"].eq(fold_id), "base_score_raw"]).sum())
            for fold_id in [3, 4, 5, 6, 7]
        }
        complete = all(counts[fold_id] >= minimum_validation_base_rows for fold_id in [3, 4, 5, 6, 7])
        if complete:
            complete_tickers.add(str(ticker))
        eligibility_rows.append(
            {
                "edge_set": edge_set_name,
                "ticker": ticker,
                **{f"fold_{fold_id}_finite_base_rows": counts[fold_id] for fold_id in [3, 4, 5, 6, 7]},
                "complete_base_eval_folds_3_7": complete,
            }
        )
    eligibility = pd.DataFrame(eligibility_rows)

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for ticker, ticker_frame in data.groupby("ticker", sort=True):
        ticker_features = [
            column
            for column in feature_columns
            if column in ticker_frame and pd.to_numeric(ticker_frame[column], errors="coerce").notna().any()
        ]
        for fold_id in [3, 4, 5, 6, 7]:
            validation_all = ticker_frame.loc[ticker_frame["fold_id"].eq(fold_id)].copy()
            # Matched support contract: both BASE and PLUS are evaluated only where base_score_raw is finite.
            validation = validation_all.loc[np.isfinite(validation_all["base_score_raw"])].copy()
            if validation.empty:
                metric_rows.append(
                    {
                        "edge_set": edge_set_name,
                        "ticker": ticker,
                        "fold_id": fold_id,
                        "role": role_for_fold(fold_id),
                        "profile": "BASE_TICKER",
                        "status": "MATCHED_SUPPORT_EMPTY",
                        "comparison_eligible": False,
                        **binary_metrics(np.array([]), np.array([])),
                    }
                )
                continue
            y = validation["label_abs_surge_3d_5pct"].to_numpy(int)
            base_score = validation["base_score_raw"].to_numpy(float)
            base_metrics = binary_metrics(y, base_score)
            metric_rows.append(
                {
                    "edge_set": edge_set_name,
                    "ticker": ticker,
                    "fold_id": fold_id,
                    "role": role_for_fold(fold_id),
                    "profile": "BASE_TICKER",
                    "status": "OK",
                    "comparison_eligible": False,
                    "feature_count": 0,
                    **base_metrics,
                }
            )
            for row, score in zip(validation.itertuples(index=False), base_score):
                prediction_rows.append(
                    {
                        "edge_set": edge_set_name,
                        "source_row_id": int(row.source_row_id),
                        "ticker": ticker,
                        "fold_id": fold_id,
                        "role": role_for_fold(fold_id),
                        "profile": "BASE_TICKER",
                        "target": int(row.label_abs_surge_3d_5pct),
                        "score": float(score),
                        "status": "OK",
                    }
                )

            training = ticker_frame.loc[
                ticker_frame["fold_id"].lt(fold_id) & np.isfinite(ticker_frame["base_score_raw"])
            ].copy()
            train_features = [
                column
                for column in ticker_features
                if pd.to_numeric(training[column], errors="coerce").notna().any()
            ]
            plus_status = "OK"
            if len(training) < minimum_training_rows:
                plus_status = "LOW_EVIDENCE_TRAIN_ROWS"
            elif training["label_abs_surge_3d_5pct"].nunique() < 2:
                plus_status = "LOW_EVIDENCE_ONE_CLASS"
            elif not train_features:
                plus_status = "NO_ALIGNED_LEAD_FEATURE"
            if plus_status == "OK":
                columns = ["base_score_raw", *train_features]
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    LogisticRegression(C=0.25, class_weight="balanced", max_iter=800, random_state=17),
                )
                model.fit(training[columns], training["label_abs_surge_3d_5pct"].astype(int))
                plus_score = model.predict_proba(validation[columns])[:, 1]
                plus_metrics = binary_metrics(y, plus_score)
                metric_rows[-1]["comparison_eligible"] = True
                metric_rows.append(
                    {
                        "edge_set": edge_set_name,
                        "ticker": ticker,
                        "fold_id": fold_id,
                        "role": role_for_fold(fold_id),
                        "profile": "BASE_PLUS_ALIGNED_LEAD",
                        "status": "OK",
                        "comparison_eligible": True,
                        "feature_count": len(columns),
                        **plus_metrics,
                    }
                )
                for row, score in zip(validation.itertuples(index=False), plus_score):
                    prediction_rows.append(
                        {
                            "edge_set": edge_set_name,
                            "source_row_id": int(row.source_row_id),
                            "ticker": ticker,
                            "fold_id": fold_id,
                            "role": role_for_fold(fold_id),
                            "profile": "BASE_PLUS_ALIGNED_LEAD",
                            "target": int(row.label_abs_surge_3d_5pct),
                            "score": float(score),
                            "status": "OK",
                        }
                    )
            else:
                metric_rows.append(
                    {
                        "edge_set": edge_set_name,
                        "ticker": ticker,
                        "fold_id": fold_id,
                        "role": role_for_fold(fold_id),
                        "profile": "BASE_PLUS_ALIGNED_LEAD",
                        "status": plus_status,
                        "comparison_eligible": False,
                        "feature_count": len(train_features) + 1,
                        **binary_metrics(np.array([]), np.array([])),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)

    champion_rows: list[dict[str, Any]] = []
    for ticker in sorted(data["ticker"].unique()):
        dev = metrics.loc[
            metrics["ticker"].eq(ticker) & metrics["fold_id"].isin([3, 4]) & metrics["status"].eq("OK")
        ].copy()
        base = dev.loc[dev["profile"].eq("BASE_TICKER")].set_index("fold_id")
        plus = dev.loc[dev["profile"].eq("BASE_PLUS_ALIGNED_LEAD")].set_index("fold_id")
        complete = all(fold_id in base.index and fold_id in plus.index for fold_id in [3, 4])
        deltas: dict[int, float] = {}
        if complete:
            for fold_id in [3, 4]:
                b = float(base.loc[fold_id, "pr_auc"])
                p = float(plus.loc[fold_id, "pr_auc"])
                deltas[fold_id] = p - b if math.isfinite(b) and math.isfinite(p) else math.nan
        plus_wins_both = complete and all(math.isfinite(deltas[fid]) and deltas[fid] > 0 for fid in [3, 4])
        champion = "BASE_PLUS_ALIGNED_LEAD" if plus_wins_both else "BASE_TICKER"
        reason = "PLUS_WINS_BOTH_DEV_FOLDS" if plus_wins_both else ("BASE_DEFAULT_DEV_INCOMPLETE" if not complete else "PLUS_NOT_CONSISTENTLY_BETTER")
        champion_rows.append(
            {
                "edge_set": edge_set_name,
                "ticker": ticker,
                "champion_profile": champion,
                "champion_reason": reason,
                "development_complete_paired": complete,
                "fold3_pr_auc_delta_plus_minus_base": deltas.get(3, math.nan),
                "fold4_pr_auc_delta_plus_minus_base": deltas.get(4, math.nan),
                "complete_base_eval_folds_3_7": ticker in complete_tickers,
            }
        )
    champions = pd.DataFrame(champion_rows)

    # Fixed ticker universe for portfolio: same eligible ticker set for folds 3-7.
    eligible_tickers: list[str] = []
    for row in champions.itertuples(index=False):
        if not bool(row.complete_base_eval_folds_3_7) or not bool(row.development_complete_paired):
            continue
        profile = str(row.champion_profile)
        profile_metrics = metrics.loc[
            metrics["ticker"].eq(row.ticker)
            & metrics["profile"].eq(profile)
            & metrics["fold_id"].isin([3, 4, 5, 6, 7])
            & metrics["status"].eq("OK")
        ]
        if set(profile_metrics["fold_id"].unique()) == {3, 4, 5, 6, 7}:
            eligible_tickers.append(str(row.ticker))
    champion_map = champions.set_index("ticker")["champion_profile"].to_dict()
    routed_parts: list[pd.DataFrame] = []
    for ticker in eligible_tickers:
        profile = champion_map[ticker]
        routed_parts.append(
            predictions.loc[predictions["ticker"].eq(ticker) & predictions["profile"].eq(profile)].copy()
        )
    routed = pd.concat(routed_parts, ignore_index=True) if routed_parts else pd.DataFrame(columns=predictions.columns)

    policy = select_frozen_portfolio_threshold(
        routed,
        target_precision=target_precision,
        minimum_alerts_per_fold=minimum_portfolio_alerts,
        development_folds=(3, 4),
    )
    portfolio_rows: list[dict[str, Any]] = []
    threshold = float(policy.get("threshold", math.nan))
    for fold_id in [3, 4, 5, 6, 7]:
        part = routed.loc[routed["fold_id"].eq(fold_id)].copy()
        metrics_row = binary_metrics(part["target"].to_numpy(float), part["score"].to_numpy(float)) if len(part) else binary_metrics(np.array([]), np.array([]))
        if len(part) and math.isfinite(threshold):
            alert = part["score"].to_numpy(float) >= threshold
            alerts = int(alert.sum())
            precision = float(part.loc[alert, "target"].mean()) if alerts else math.nan
        else:
            alerts = 0
            precision = math.nan
        fold_gate = bool(
            policy.get("development_gate_pass", False)
            and alerts >= minimum_portfolio_alerts
            and math.isfinite(precision)
            and precision >= target_precision
        )
        portfolio_rows.append(
            {
                "edge_set": edge_set_name,
                "fold_id": fold_id,
                "role": role_for_fold(fold_id),
                "eligible_tickers_fixed": len(eligible_tickers),
                **metrics_row,
                "frozen_threshold": threshold,
                "alerts": alerts,
                "precision": precision,
                "development_gate_pass": bool(policy.get("development_gate_pass", False)),
                "policy_gate_pass": fold_gate,
                "threshold_selected_on": "folds_3_4_only",
            }
        )
    portfolio = pd.DataFrame(portfolio_rows)
    policy = policy | {
        "edge_set": edge_set_name,
        "eligible_tickers": eligible_tickers,
        "production_gate_pass": bool(
            policy.get("development_gate_pass", False)
            and not portfolio.empty
            and portfolio.loc[portfolio["fold_id"].isin([5, 6, 7]), "policy_gate_pass"].astype(bool).all()
        ),
    }
    return predictions, metrics, champions, portfolio, eligibility, policy


def _edge_subset(result: pd.DataFrame, name: str) -> pd.DataFrame:
    normalized = name.strip().lower()
    if normalized == "v11_strong_repaired":
        return result.loc[result["was_v11_strong"].astype(bool) & result["directed_lag"].gt(0)].copy()
    if normalized == "maxstat_corrected":
        return result.loc[result["corrected_discovery_dev_candidate"].astype(bool)].copy()
    if normalized == "forward_effect_stable":
        return result.loc[result["forward_effect_stable"].astype(bool)].copy()
    raise ValueError(f"Unknown edge set: {name}")


def run_pipeline(args: Any) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    try:
        folds = load_folds(Path(args.folds))
        frame = load_frame(Path(args.dataset), Path(args.target_sidecar))
        universe = choose_universe(frame, args.tickers)
        frame = frame.loc[frame["ticker"].isin(universe)].copy()
        dates, returns, target_panel, source_ids = make_panel(frame, universe)
        metadata = ticker_metadata(frame, universe)
        payload, raw, market_factor, bucket_factor = build_fold_residual_payload(dates, returns, metadata, folds)
        log(f"loaded rows={len(frame):,} tickers={len(universe)} pairs={len(universe)*(len(universe)-1)//2:,}")

        maxstat = run_maxstat_map(
            universe,
            payload,
            screening_permutations=args.screening_permutations,
            adaptive_permutations=args.adaptive_permutations,
            adaptive_trigger_p=args.adaptive_trigger_p,
            block_size=args.block_size,
            minimum=args.minimum_pair_observations,
            threads=args.threads,
            executor_kind=getattr(args, "maxstat_executor", "thread"),
            seed=args.seed,
            alpha=args.maxstat_alpha,
            minimum_forward_abs_corr=args.minimum_forward_abs_corr,
        )
        reference = load_v11_reference(Path(args.v11_reference_strong))
        maxstat = mark_v11_reference(maxstat, reference)
        atomic_csv(output / "maxstat_pair_results_v11_1.csv", maxstat)
        corrected = maxstat.loc[maxstat["corrected_discovery_dev_candidate"].astype(bool)].copy()
        atomic_csv(output / "corrected_directed_edges_v11_1.csv", corrected)
        v11_recheck = recheck_v11_reference_edges(
            reference,
            universe,
            payload,
            maxstat,
            minimum=args.minimum_pair_observations,
            maxstat_alpha=args.maxstat_alpha,
            minimum_forward_abs_corr=args.minimum_forward_abs_corr,
        )
        atomic_csv(output / "v11_strong_edge_recheck_v11_1.csv", v11_recheck)

        rolling_edges = pd.concat([v11_recheck, corrected], ignore_index=True, sort=False).drop_duplicates(
            ["leader", "follower", "directed_lag"]
        )
        rolling = compute_frozen_lag_rolling(
            rolling_edges,
            universe,
            dates,
            raw,
            market_factor,
            bucket_factor,
            folds,
            windows=args.rolling_windows,
            step=args.rolling_step,
            residual_fit_days=args.rolling_residual_fit_days,
            residual_min_fit=args.rolling_residual_min_fit,
            minimum_corr_observations=args.minimum_pair_observations,
            minimum_forward_abs_corr=args.minimum_forward_abs_corr,
        )
        atomic_csv(output / "frozen_lag_rolling_v11_1.csv", rolling)
        rolling_summary = summarize_frozen_rolling(rolling)
        atomic_csv(output / "frozen_lag_rolling_summary_v11_1.csv", rolling_summary)

        base_oof = load_base_oof(Path(args.v10_2_output))
        all_lifts: list[pd.DataFrame] = []
        all_lift_summaries: list[pd.DataFrame] = []
        all_predictions: list[pd.DataFrame] = []
        all_metrics: list[pd.DataFrame] = []
        all_champions: list[pd.DataFrame] = []
        all_portfolios: list[pd.DataFrame] = []
        all_eligibility: list[pd.DataFrame] = []
        policies: dict[str, Any] = {}
        feature_manifests: dict[str, Any] = {}

        edge_set_names = [name.strip() for name in args.probe_edge_sets.split(",") if name.strip()]
        for edge_set_name in edge_set_names:
            normalized_edge_set = edge_set_name.strip().lower()
            if normalized_edge_set == "v11_strong_repaired":
                edges = v11_recheck.copy()
            else:
                edges = _edge_subset(maxstat, edge_set_name)
            canonical_name = edge_set_name.upper()
            edge_dir = output / "edge_sets" / canonical_name
            edge_dir.mkdir(parents=True, exist_ok=True)
            atomic_csv(edge_dir / "edges.csv", edges)
            features, feature_manifest = build_target_aligned_feature_frame(
                edges, universe, dates, raw, target_panel, source_ids, payload
            )
            feature_manifest["edge_set"] = canonical_name
            feature_manifests[canonical_name] = feature_manifest
            atomic_json(edge_dir / "TARGET_ALIGNED_FEATURE_MANIFEST.json", feature_manifest)
            atomic_csv(edge_dir / "target_aligned_features.csv", features)
            lift, lift_summary = compute_target_aligned_lift(
                edges, features, frame, minimum_event_n=args.minimum_target_event_n
            )
            if not lift.empty:
                lift.insert(0, "edge_set", canonical_name)
                all_lifts.append(lift)
            if not lift_summary.empty:
                lift_summary.insert(0, "edge_set", canonical_name)
                all_lift_summaries.append(lift_summary)
            atomic_csv(edge_dir / "target_aligned_lift_by_fold.csv", lift)
            atomic_csv(edge_dir / "target_aligned_lift_summary.csv", lift_summary)

            predictions, metrics, champions, portfolio, eligibility, policy = run_matched_probe(
                canonical_name,
                features,
                frame,
                base_oof,
                target_precision=args.target_precision,
                minimum_portfolio_alerts=args.minimum_portfolio_alerts,
                minimum_training_rows=args.minimum_probe_training_rows,
                minimum_validation_base_rows=args.minimum_validation_base_rows,
            )
            policies[canonical_name] = policy
            for collection, item in [
                (all_predictions, predictions),
                (all_metrics, metrics),
                (all_champions, champions),
                (all_portfolios, portfolio),
                (all_eligibility, eligibility),
            ]:
                if not item.empty:
                    collection.append(item)
            atomic_csv(edge_dir / "matched_probe_predictions.csv", predictions)
            atomic_csv(edge_dir / "matched_probe_metrics.csv", metrics)
            atomic_csv(edge_dir / "matched_probe_champions.csv", champions)
            atomic_csv(edge_dir / "matched_probe_portfolio.csv", portfolio)
            atomic_csv(edge_dir / "probe_eligibility.csv", eligibility)
            atomic_json(edge_dir / "FROZEN_PORTFOLIO_POLICY.json", policy)
            log(f"edge set {canonical_name}: edges={len(edges)} features={len(feature_manifest.get('feature_columns', []))}")

        def concat_or_empty(items: list[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(items, ignore_index=True) if items else pd.DataFrame()

        combined_lift = concat_or_empty(all_lifts)
        combined_lift_summary = concat_or_empty(all_lift_summaries)
        combined_predictions = concat_or_empty(all_predictions)
        combined_metrics = concat_or_empty(all_metrics)
        combined_champions = concat_or_empty(all_champions)
        combined_portfolios = concat_or_empty(all_portfolios)
        combined_eligibility = concat_or_empty(all_eligibility)
        atomic_csv(output / "target_aligned_lift_v11_1.csv", combined_lift)
        atomic_csv(output / "target_aligned_lift_summary_v11_1.csv", combined_lift_summary)
        atomic_csv(output / "matched_probe_predictions_v11_1.csv", combined_predictions)
        atomic_csv(output / "matched_probe_metrics_v11_1.csv", combined_metrics)
        atomic_csv(output / "matched_probe_champions_v11_1.csv", combined_champions)
        atomic_csv(output / "matched_probe_portfolio_v11_1.csv", combined_portfolios)
        atomic_csv(output / "matched_probe_eligibility_v11_1.csv", combined_eligibility)
        atomic_json(output / "FROZEN_PORTFOLIO_POLICIES_V11_1.json", policies)
        atomic_json(output / "TARGET_ALIGNED_FEATURE_MANIFESTS_V11_1.json", feature_manifests)

        # Compact comparison table for the seven original V11 strong edges.
        comparison = v11_recheck.copy()
        if not combined_lift_summary.empty:
            repaired = combined_lift_summary.loc[combined_lift_summary["edge_set"].eq("V11_STRONG_REPAIRED")].drop(columns=["edge_set"])
            comparison = comparison.merge(repaired, on=["leader", "follower", "directed_lag"], how="left")
        if not rolling_summary.empty:
            roll60 = rolling_summary.loc[rolling_summary["window"].eq(60), [
                "leader", "follower", "fixed_lag", "same_direction_rate", "effect_pass_rate", "latest_correlation", "latest_state"
            ]].rename(columns={"fixed_lag": "directed_lag"})
            comparison = comparison.merge(roll60, on=["leader", "follower", "directed_lag"], how="left")
        atomic_csv(output / "V11_TO_V11_1_EDGE_COMPARISON.csv", comparison)

        recommendation = {
            "schema": SCHEMA_VERSION,
            "status": "CORRECTIVE_VALIDATION_COMPLETE",
            "rows": int(len(frame)),
            "tickers": int(len(universe)),
            "pairs": int(len(universe) * (len(universe) - 1) // 2),
            "maxstat_screening_permutations": int(args.screening_permutations),
            "maxstat_adaptive_permutations": int(args.adaptive_permutations),
            "v11_strong_edges_rechecked": int(v11_recheck.shape[0]),
            "maxstat_corrected_discovery_dev_edges": int(maxstat["corrected_discovery_dev_candidate"].astype(bool).sum()),
            "forward_sign_stable_edges": int(maxstat["forward_sign_stable"].astype(bool).sum()),
            "forward_effect_stable_edges": int(maxstat["forward_effect_stable"].astype(bool).sum()),
            "probe_policies": policies,
            "target_alignment_corrected": True,
            "matched_row_probe": True,
            "rolling_lag_reoptimized": False,
            "causal_claim": False,
            "production_action": "NO_ALERT_UNLESS_FUTURE_HOLDOUT_LATER_PASSES",
        }
        atomic_json(output / "LEADLAG_CORRECTIVE_RECOMMENDATION_V11_1.json", recommendation)
        manifest = {
            "schema": SCHEMA_VERSION,
            "inputs": {
                "dataset": {"path": str(args.dataset), "sha256": sha256_file(Path(args.dataset))},
                "target_sidecar": {"path": str(args.target_sidecar), "sha256": sha256_file(Path(args.target_sidecar))},
                "folds": {"path": str(args.folds), "sha256": sha256_file(Path(args.folds))},
                "v10_2_base_oof": str(Path(args.v10_2_output) / "ticker_base_oof_predictions_v10_2.csv"),
                "v11_reference_strong": str(args.v11_reference_strong),
            },
            "fold_contract": {"discovery": [0, 1, 2], "development": [3, 4], "confirmation": [5, 6], "recent_audit": [7]},
            "target": "label_abs_surge_3d_5pct",
            "correction_contract": {
                "max_lag_multiple_testing": "moving-block permutation max |corr| over lags -5..5, then BH-FDR across unordered pairs",
                "frozen_lag_rolling": "discovery lag fixed; point-in-time bucket residual; no lag reoptimization",
                "target_alignment": "B forecast t uses A[t+k-h] only when offset<=0 for B D+k, k=1..3",
                "probe_support": "BASE and PLUS use identical finite-base validation rows",
                "champion": "PLUS only if PR-AUC beats BASE in both development folds 3 and 4",
                "portfolio_threshold": "single threshold selected on folds 3-4 only and frozen for folds 5-7",
            },
            "parameters": {key: value for key, value in vars(args).items() if key not in {"output"}},
        }
        atomic_json(output / "LEADLAG_MANIFEST_V11_1.json", manifest)
        atomic_json(status_path, {"status": "SUCCESS", "recommendation": recommendation})
        report = f"""# CrashWatch Surge Lead-Lag V11.1 Corrective Validation\n\n- Rows: {len(frame):,}\n- Tickers: {len(universe)}\n- Pairs: {len(universe)*(len(universe)-1)//2:,}\n- V11 strong edges rechecked: {len(v11_recheck)}\n- Max-stat corrected discovery+development edges: {recommendation['maxstat_corrected_discovery_dev_edges']}\n- Forward sign-stable edges: {recommendation['forward_sign_stable_edges']}\n- Forward effect-stable edges (|corr| >= {args.minimum_forward_abs_corr:.3f}): {recommendation['forward_effect_stable_edges']}\n\nThis run corrects four V11 issues: max-lag multiplicity, frozen-lag stability, target-horizon alignment, and matched-row BASE-vs-PLUS probing. It does not make a causal claim and does not authorize production alerts.\n"""
        atomic_text(output / "REPORT_V11_1_KO.md", report)
        log("corrective validation complete")
    except Exception as exc:
        atomic_json(status_path, {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
