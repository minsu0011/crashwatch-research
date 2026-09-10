from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-8
TARGET_COLUMN = "label_abs_surge_3d_5pct"
MOVE_COLUMN = "label_abs_move_3d_5pct"
DIRECTION_COLUMN = "label_up_given_abs_move"

# V12's direction gate was contaminated by volatility/range/tail magnitude.
# V11 lag-0 features are therefore split conceptually: the primary Stage-2 block
# keeps signed/relational state only.  Absolute-distance and dispersion columns
# are still emitted for audit/diagnostics but are not primary model inputs.
V11_LAG0_PRIMARY_DIRECTION_COLUMNS = (
    "v11_lag0_peer_count",
    "v11_lag0_peer_signed_resid_mean",
    "v11_lag0_relation_gap_mean",
    "v11_lag0_divergence_z60_mean",
    "v11_lag0_divergence_z120_mean",
    "v11_lag0_recoupling_mean",
    "v11_lag0_agreement_fraction",
)


MAGNITUDE_TOKENS = (
    "rangevol", "vol_realized", "realized_vol", "volatility", "_vol_", "bipower",
    "parkinson", "garman_klass", "rogers_satchell", "yang_zhang",
    "taildep", "tailnet", "expected_shortfall", "tail_downside", "tail_sync",
    "network_tail", "cocrash", "crash_ratio", "left_tail", "extreme",
    "corwin_schultz", "high_low_pct", "zero_return_ratio",
    "network_market_corr", "network_bucket_corr", "network_centrality", "largest_eigen",
    "amihud", "illiquidity", "spread", "turnover_vol", "volume_vol",
    "atr", "true_range", "kurt", "jump", "variance", "std_",
)

DIRECTION_TOKENS = (
    "price_ret", "_ret_", "return", "ma_gap", "drawdown", "skew",
    "finflow", "foreign", "institution", "individual", "finshort",
    "short", "lending", "event", "pressure", "peer_relative",
    "close_location", "gap_open", "momentum", "trend", "rsi", "macd",
    "slope", "change", "rising", "falling", "balance", "buy", "sell",
    "date_rank", "market_rank", "bucket_rank", "ticker_rank", "rank60",
    "rank20", "z20", "z60", "z120", "relative", "divergence",
)

METADATA_COLUMNS = {
    "source_row_id", "row_index", "date", "ticker", "name", "market",
    "bucket", "industry", "industry_name", "target_valid", TARGET_COLUMN,
    MOVE_COLUMN, DIRECTION_COLUMN, "first_hit_day", "best_forward_return_3d",
}


@dataclass(frozen=True)
class ABSpec:
    ticker: str
    node_id: str
    source_feature: str
    direction: int
    weight: float
    rank: int


@dataclass
class RobustScale:
    median: float
    scale: float


@dataclass
class ABState:
    specs_by_ticker: dict[str, list[ABSpec]]
    scales: dict[tuple[str, str], RobustScale]
    max_slots: int


def normalize_ticker(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else (text or "UNKNOWN")


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        try:
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError("Parquet input requires pyarrow. Install requirements_v13.txt.") from exc
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    if suffix.endswith(".feather"):
        return pd.read_feather(path, columns=list(columns) if columns is not None else None)
    if suffix.endswith(".pkl") or suffix.endswith(".pickle"):
        frame = pd.read_pickle(path)
        return frame[list(columns)] if columns is not None else frame
    raise ValueError(f"Unsupported table format: {path}")


def table_columns(path: Path) -> list[str]:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading a parquet schema requires pyarrow.") from exc
        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return list(pd.read_csv(path, nrows=0).columns)
    return list(read_table(path).columns)


def return_series(frame: pd.DataFrame) -> tuple[pd.Series, str, float]:
    """Return decimal one-day return, source name, and applied scale.

    V11 used t_price_ret_1 with return_pct / 100 fallback. The same contract is
    retained here. A defensive scale check catches percent-valued columns.
    """
    primary = pd.to_numeric(frame.get("t_price_ret_1"), errors="coerce") if "t_price_ret_1" in frame else pd.Series(np.nan, index=frame.index)
    if "return_pct" in frame:
        fallback = pd.to_numeric(frame["return_pct"], errors="coerce") / 100.0
        values = primary.where(primary.notna(), fallback)
        source = "t_price_ret_1_with_return_pct_fallback"
    else:
        values = primary
        source = "t_price_ret_1"
    finite = values[np.isfinite(values)]
    scale = 1.0
    if len(finite):
        q99 = float(np.nanquantile(np.abs(finite), 0.99))
        # Daily equity returns above 1.0 are possible but systematic q99 > 1 usually
        # indicates percentage points rather than decimals.
        if q99 > 1.0:
            values = values / 100.0
            scale = 0.01
    return values.astype(float), source, scale


def derive_future_path_labels(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.05,
    target_column: str = TARGET_COLUMN,
    max_mismatch_rate: float = 0.01,
    allow_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive 3-day absolute-move and conditional direction labels.

    The official surge label remains the source of truth. Future cumulative
    returns are reconstructed from one-day returns only to create the symmetric
    large-move label and to audit the official target definition.
    """
    required = {"ticker", "date", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"derive_future_path_labels missing columns: {sorted(missing)}")
    work = frame.copy()
    work["ticker"] = work["ticker"].map(normalize_ticker)
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work["return_1d"], return_source, applied_scale = return_series(work)
    original_index = work.index.copy()
    work["__original_position"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["ticker", "date", "source_row_id" if "source_row_id" in work else "__original_position"], kind="mergesort")

    r1 = work.groupby("ticker", sort=False)["return_1d"].shift(-1)
    r2 = work.groupby("ticker", sort=False)["return_1d"].shift(-2)
    r3 = work.groupby("ticker", sort=False)["return_1d"].shift(-3)
    c1 = r1
    c2 = (1.0 + r1) * (1.0 + r2) - 1.0
    c3 = (1.0 + r1) * (1.0 + r2) * (1.0 + r3) - 1.0
    path = np.column_stack([c1.to_numpy(float), c2.to_numpy(float), c3.to_numpy(float)])
    valid = np.all(np.isfinite(path), axis=1)
    up = np.full(len(work), np.nan, dtype=float)
    down = np.full(len(work), np.nan, dtype=float)
    up[valid] = np.max(path[valid], axis=1)
    down[valid] = np.min(path[valid], axis=1)

    if "best_forward_return_3d" in work.columns:
        official_best = pd.to_numeric(work["best_forward_return_3d"], errors="coerce").to_numpy(float)
        replace = np.isfinite(official_best)
        up[replace] = official_best[replace]

    official = pd.to_numeric(work[target_column], errors="coerce").to_numpy(float)
    reconstructed = np.where(np.isfinite(up), (up >= float(threshold)).astype(float), np.nan)
    audit_mask = np.isfinite(official) & np.isfinite(reconstructed)
    mismatch = audit_mask & (official.astype(np.int8) != reconstructed.astype(np.int8))
    mismatch_rate = float(mismatch.sum() / max(audit_mask.sum(), 1))
    if mismatch_rate > float(max_mismatch_rate) and not allow_mismatch:
        raise RuntimeError(
            f"Official surge target reconstruction mismatch {mismatch_rate:.4%} exceeds "
            f"{max_mismatch_rate:.4%}. Check return scaling/target definition."
        )

    abs_excursion = np.maximum(up, -down)
    move = np.where(valid, (abs_excursion >= float(threshold)).astype(np.int8), -1)
    direction = np.where(move == 1, official.astype(np.int8), -1)
    work["future_cumret_d1"] = c1.to_numpy(float)
    work["future_cumret_d2"] = c2.to_numpy(float)
    work["future_cumret_d3"] = c3.to_numpy(float)
    work["future_up_excursion_3d"] = up
    work["future_down_excursion_3d"] = down
    work["future_abs_excursion_3d"] = abs_excursion
    work["future_direction_margin_3d"] = up - np.abs(down)
    work["future_path_valid"] = valid
    work[MOVE_COLUMN] = move
    work[DIRECTION_COLUMN] = direction
    work["surge_target_reconstructed"] = reconstructed
    work = work.sort_values("__original_position", kind="mergesort").drop(columns="__original_position")
    work.index = original_index

    audit = {
        "return_source": return_source,
        "return_scale_applied": applied_scale,
        "threshold": float(threshold),
        "rows": int(len(work)),
        "future_path_valid_rows": int(valid.sum()),
        "official_target_audit_rows": int(audit_mask.sum()),
        "official_target_mismatches": int(mismatch.sum()),
        "official_target_mismatch_rate": mismatch_rate,
        "absolute_move_positive": int(np.sum(move == 1)),
        "absolute_move_negative": int(np.sum(move == 0)),
        "surge_within_absolute_move": int(np.sum((move == 1) & (official == 1))),
        "down_or_non_surge_large_move": int(np.sum((move == 1) & (official == 0))),
    }
    return work, audit


def classify_feature(feature: str) -> str:
    name = str(feature).lower()
    if name in METADATA_COLUMNS or name.startswith("future_") or name.startswith("label_"):
        return "metadata"
    magnitude = any(token in name for token in MAGNITUDE_TOKENS)
    direction = any(token in name for token in DIRECTION_TOKENS)
    if magnitude and direction:
        # Skew is signed/asymmetric and remains eligible for the direction model
        # unless it is explicitly a volatility/range statistic.
        if "skew" in name and not any(token in name for token in ("rangevol", "vol_realized", "realized_vol", "_vol_", "volatility")):
            return "direction"
        # Explicit range/volatility/tail/liquidity names are treated as magnitude
        # even when they also include a generic token such as change or z20.
        if any(token in name for token in (
            "rangevol", "vol_realized", "realized_vol", "_vol_", "volatility",
            "taildep", "tailnet", "expected_shortfall", "parkinson", "garman_klass",
            "rogers_satchell", "yang_zhang", "bipower", "spread", "zero_return_ratio",
            "network_market_corr", "network_bucket_corr", "network_centrality", "largest_eigen",
        )):
            return "magnitude"
        return "mixed"
    if magnitude:
        return "magnitude"
    if direction:
        return "direction"
    return "unknown"


def feature_family_audit(features: Sequence[str]) -> pd.DataFrame:
    rows = [{"feature": str(f), "family": classify_feature(str(f))} for f in features]
    return pd.DataFrame(rows)


def _as_candidate(series: pd.Series) -> pd.Series:
    return parse_bool(series)


def build_ab_specs(
    candidate_map: pd.DataFrame,
    available_columns: Iterable[str],
    *,
    max_features_per_ticker: int = 12,
) -> dict[str, list[ABSpec]]:
    required = {"ticker", "axis", "node_id", "source_feature", "selection_direction", "precision_separator_score_v10_2"}
    missing = required - set(candidate_map.columns)
    if missing:
        raise ValueError(f"A/B candidate map missing columns: {sorted(missing)}")
    part = candidate_map.copy()
    part["ticker"] = part["ticker"].map(normalize_ticker)
    part = part.loc[part["axis"].astype(str).eq("AB")].copy()
    if "precision_separator_selection_candidate" in part:
        part = part.loc[_as_candidate(part["precision_separator_selection_candidate"])].copy()
    part["selection_direction"] = pd.to_numeric(part["selection_direction"], errors="coerce")
    part["precision_separator_score_v10_2"] = pd.to_numeric(part["precision_separator_score_v10_2"], errors="coerce")
    part = part.loc[part["selection_direction"].isin([-1, 1])].copy()
    available = {str(x) for x in available_columns}
    part = part.loc[part["node_id"].astype(str).isin(available)].copy()
    part = part.sort_values(
        ["ticker", "source_feature", "precision_separator_score_v10_2", "node_id"],
        ascending=[True, True, False, True], kind="mergesort",
    ).drop_duplicates(["ticker", "source_feature"], keep="first")
    part = part.sort_values(
        ["ticker", "precision_separator_score_v10_2", "source_feature"],
        ascending=[True, False, True], kind="mergesort",
    )
    output: dict[str, list[ABSpec]] = {}
    for ticker, grp in part.groupby("ticker", sort=True):
        specs: list[ABSpec] = []
        for rank, (_, row) in enumerate(grp.head(int(max_features_per_ticker)).iterrows(), start=1):
            score = float(row["precision_separator_score_v10_2"])
            specs.append(
                ABSpec(
                    ticker=str(ticker),
                    node_id=str(row["node_id"]),
                    source_feature=str(row["source_feature"]),
                    direction=int(row["selection_direction"]),
                    weight=max(score, 0.05),
                    rank=rank,
                )
            )
        if specs:
            output[str(ticker)] = specs
    return output


def serialize_ab_specs(specs: Mapping[str, Sequence[ABSpec]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in sorted(specs):
        for spec in specs[ticker]:
            rows.append({
                "ticker": spec.ticker,
                "node_id": spec.node_id,
                "source_feature": spec.source_feature,
                "selection_direction": spec.direction,
                "weight": spec.weight,
                "rank": spec.rank,
            })
    return pd.DataFrame(rows)


def robust_scale(values: Sequence[float]) -> RobustScale:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return RobustScale(0.0, 1.0)
    median = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(x)) if x.size > 1 else 1.0
    if not math.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return RobustScale(median, scale)


def fit_ab_state(train: pd.DataFrame, specs: Mapping[str, Sequence[ABSpec]], max_slots: int = 12) -> ABState:
    tick = train["ticker"].astype(str).map(normalize_ticker)
    scales: dict[tuple[str, str], RobustScale] = {}
    selected: dict[str, list[ABSpec]] = {}
    for ticker, values in specs.items():
        items = list(values)[: int(max_slots)]
        selected[ticker] = items
        rows = train.loc[tick.eq(ticker)]
        for spec in items:
            scales[(ticker, spec.node_id)] = robust_scale(pd.to_numeric(rows.get(spec.node_id), errors="coerce"))
    return ABState(specs_by_ticker=selected, scales=scales, max_slots=int(max_slots))


def transform_ab_evidence(frame: pd.DataFrame, state: ABState) -> pd.DataFrame:
    n = len(frame)
    result = pd.DataFrame(index=frame.index)
    aggregate_names = [
        "ab_evidence_count", "ab_weighted_mean", "ab_mean", "ab_min", "ab_max", "ab_std",
        "ab_favorable_fraction", "ab_adverse_fraction", "ab_strong_adverse_fraction", "ab_missing_fraction",
    ]
    for name in aggregate_names:
        result[name] = 0.0
    for slot in range(1, state.max_slots + 1):
        result[f"ab_slot_{slot:02d}"] = 0.0
        result[f"ab_slot_present_{slot:02d}"] = 0.0
    tick = frame["ticker"].astype(str).map(normalize_ticker).to_numpy()
    for ticker, specs in state.specs_by_ticker.items():
        pos = np.flatnonzero(tick == ticker)
        if pos.size == 0:
            continue
        mat = np.zeros((pos.size, state.max_slots), dtype=float)
        present = np.zeros_like(mat)
        weights = np.zeros(state.max_slots, dtype=float)
        for j, spec in enumerate(specs[: state.max_slots]):
            scale = state.scales[(ticker, spec.node_id)]
            values = pd.to_numeric(frame.iloc[pos][spec.node_id], errors="coerce").to_numpy(float)
            good = np.isfinite(values)
            z = np.zeros(pos.size, dtype=float)
            z[good] = spec.direction * (values[good] - scale.median) / max(scale.scale, EPS)
            z = np.clip(z, -8.0, 8.0)
            mat[:, j] = z
            present[:, j] = good.astype(float)
            weights[j] = spec.weight
            result.iloc[pos, result.columns.get_loc(f"ab_slot_{j+1:02d}")] = z
            result.iloc[pos, result.columns.get_loc(f"ab_slot_present_{j+1:02d}")] = good.astype(float)
        count = present.sum(axis=1)
        denom = np.maximum(count, 1.0)
        weighted_denom = np.maximum((present * weights).sum(axis=1), EPS)
        weighted_mean = (mat * present * weights).sum(axis=1) / weighted_denom
        mean = (mat * present).sum(axis=1) / denom
        centered = (mat - mean[:, None]) * present
        std = np.sqrt((centered * centered).sum(axis=1) / denom)
        minv = np.where(present > 0, mat, np.inf).min(axis=1)
        maxv = np.where(present > 0, mat, -np.inf).max(axis=1)
        minv[~np.isfinite(minv)] = 0.0
        maxv[~np.isfinite(maxv)] = 0.0
        values_map = {
            "ab_evidence_count": count,
            "ab_weighted_mean": weighted_mean,
            "ab_mean": mean,
            "ab_min": minv,
            "ab_max": maxv,
            "ab_std": std,
            "ab_favorable_fraction": ((mat > 0.5) * present).sum(axis=1) / denom,
            "ab_adverse_fraction": ((mat < -0.5) * present).sum(axis=1) / denom,
            "ab_strong_adverse_fraction": ((mat < -1.5) * present).sum(axis=1) / denom,
            "ab_missing_fraction": 1.0 - count / max(len(specs), 1),
        }
        for name, values in values_map.items():
            result.iloc[pos, result.columns.get_loc(name)] = values
    return result


def _edge_weight(correlation: float, q_value: float) -> float:
    corr = abs(float(correlation)) if math.isfinite(float(correlation)) else 0.0
    q = max(float(q_value), 1e-12) if math.isfinite(float(q_value)) else 1.0
    return max(corr, 0.05) * min(max(-math.log10(q), 1.0), 5.0)


def build_v11_lag0_features(frame: pd.DataFrame, edge_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point-in-time residual co-movement features from V11.1 lag-0 edges.

    Only discovery max-stat q<=0.10 lag-0 identities are used. No directed edge
    enters this primary block. Rolling means/stds are shifted by one observation.
    """
    required = {"date", "ticker", "bucket", "return_1d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"V11 lag0 features missing columns: {sorted(missing)}")
    edges = edge_manifest.copy()
    if "directed_lag" in edges:
        edges = edges.loc[pd.to_numeric(edges["directed_lag"], errors="coerce").fillna(0).eq(0)].copy()
    if "maxstat_q_value" in edges:
        edges = edges.loc[pd.to_numeric(edges["maxstat_q_value"], errors="coerce").le(0.10)].copy()
    edges["ticker_a"] = edges["ticker_a"].map(normalize_ticker)
    edges["ticker_b"] = edges["ticker_b"].map(normalize_ticker)
    edges = edges.drop_duplicates(["ticker_a", "ticker_b"]).copy()

    base = frame[["date", "ticker", "bucket", "return_1d"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    base["ticker"] = base["ticker"].map(normalize_ticker)
    base["return_1d"] = pd.to_numeric(base["return_1d"], errors="coerce")
    date_grouped = base.groupby("date", sort=False)["return_1d"]
    date_total = date_grouped.transform("sum")
    date_count = date_grouped.transform("count")
    date_peer_mean = (date_total - base["return_1d"]) / (date_count - 1).replace(0, np.nan)
    grouped = base.groupby(["date", "bucket"], sort=False)["return_1d"]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    peer_mean = (total - base["return_1d"]) / (count - 1).replace(0, np.nan)
    base["bucket_residual"] = base["return_1d"] - peer_mean
    fallback = count.le(1)
    base.loc[fallback, "bucket_residual"] = base.loc[fallback, "return_1d"] - date_peer_mean.loc[fallback]
    base["bucket_residual"] = base["bucket_residual"].fillna(0.0)
    panel = base.pivot(index="date", columns="ticker", values="bucket_residual").sort_index()

    long_parts: list[pd.DataFrame] = []
    used_rows: list[dict[str, Any]] = []
    for _, edge in edges.iterrows():
        a, b = str(edge["ticker_a"]), str(edge["ticker_b"])
        if a not in panel.columns or b not in panel.columns:
            continue
        corr = float(pd.to_numeric(pd.Series([edge.get("discovery_best_lag_correlation", np.nan)]), errors="coerce").iloc[0])
        if not math.isfinite(corr) or corr == 0:
            continue
        q = float(pd.to_numeric(pd.Series([edge.get("maxstat_q_value", 1.0)]), errors="coerce").iloc[0])
        sign = 1.0 if corr > 0 else -1.0
        weight = _edge_weight(corr, q)
        used_rows.append({"ticker_a": a, "ticker_b": b, "discovery_correlation": corr, "relation_sign": int(sign), "maxstat_q_value": q, "weight": weight})
        for own_ticker, peer_ticker in ((a, b), (b, a)):
            own = panel[own_ticker]
            signed_peer = sign * panel[peer_ticker]
            gap = own - signed_peer
            mean60 = gap.shift(1).rolling(60, min_periods=20).mean()
            std60 = gap.shift(1).rolling(60, min_periods=20).std(ddof=0).replace(0, np.nan)
            mean120 = gap.shift(1).rolling(120, min_periods=40).mean()
            std120 = gap.shift(1).rolling(120, min_periods=40).std(ddof=0).replace(0, np.nan)
            z60 = (gap - mean60) / std60
            z120 = (gap - mean120) / std120
            recoupling = np.abs(z60).shift(1) - np.abs(z60)
            part = pd.DataFrame({
                "date": panel.index,
                "ticker": own_ticker,
                "weight": weight,
                "peer_signed_resid": signed_peer.to_numpy(float),
                "alignment": (own * signed_peer).to_numpy(float),
                "relation_gap": gap.to_numpy(float),
                "gap_abs": np.abs(gap).to_numpy(float),
                "z60": z60.to_numpy(float),
                "z120": z120.to_numpy(float),
                "recoupling": recoupling.to_numpy(float),
                "agreement": ((np.sign(own) == np.sign(signed_peer)) & own.notna() & signed_peer.notna()).astype(float).to_numpy(),
            })
            long_parts.append(part)
    feature_names = [
        "v11_lag0_peer_count", "v11_lag0_peer_signed_resid_mean", "v11_lag0_peer_signed_resid_maxabs",
        "v11_lag0_alignment_mean", "v11_lag0_relation_gap_mean", "v11_lag0_relation_gap_abs_mean",
        "v11_lag0_divergence_z60_mean", "v11_lag0_divergence_z60_maxabs",
        "v11_lag0_divergence_z120_mean", "v11_lag0_divergence_z120_maxabs",
        "v11_lag0_recoupling_mean", "v11_lag0_agreement_fraction", "v11_lag0_network_dispersion",
    ]
    if not long_parts:
        return pd.DataFrame(0.0, index=frame.index, columns=feature_names), pd.DataFrame(used_rows)
    long = pd.concat(long_parts, ignore_index=True)
    long["valid_peer"] = np.isfinite(long["peer_signed_resid"]).astype(float)
    long["weighted_peer"] = long["weight"] * long["peer_signed_resid"]
    long["weighted_alignment"] = long["weight"] * long["alignment"]
    long["weighted_gap"] = long["weight"] * long["relation_gap"]
    long["weighted_gap_abs"] = long["weight"] * long["gap_abs"]
    long["weighted_z60"] = long["weight"] * long["z60"]
    long["weighted_z120"] = long["weight"] * long["z120"]
    long["weighted_recoupling"] = long["weight"] * long["recoupling"]
    long["weighted_agreement"] = long["weight"] * long["agreement"]

    def summarize(group: pd.DataFrame) -> pd.Series:
        valid = np.isfinite(group["peer_signed_resid"].to_numpy(float))
        if not valid.any():
            return pd.Series({name: 0.0 for name in feature_names})
        g = group.loc[valid]
        w = pd.to_numeric(g["weight"], errors="coerce").to_numpy(float)
        wsum = max(float(np.sum(w)), EPS)
        peer = g["peer_signed_resid"].to_numpy(float)
        z60 = g["z60"].to_numpy(float)
        z120 = g["z120"].to_numpy(float)
        def wmean(column: str) -> float:
            values = pd.to_numeric(g[column], errors="coerce").to_numpy(float)
            ok = np.isfinite(values) & np.isfinite(w)
            return float(np.sum(values[ok] * w[ok]) / max(np.sum(w[ok]), EPS)) if ok.any() else 0.0
        return pd.Series({
            "v11_lag0_peer_count": float(len(g)),
            "v11_lag0_peer_signed_resid_mean": wmean("peer_signed_resid"),
            "v11_lag0_peer_signed_resid_maxabs": float(np.nanmax(np.abs(peer))) if len(peer) else 0.0,
            "v11_lag0_alignment_mean": wmean("alignment"),
            "v11_lag0_relation_gap_mean": wmean("relation_gap"),
            "v11_lag0_relation_gap_abs_mean": wmean("gap_abs"),
            "v11_lag0_divergence_z60_mean": wmean("z60"),
            "v11_lag0_divergence_z60_maxabs": float(np.nanmax(np.abs(z60))) if np.isfinite(z60).any() else 0.0,
            "v11_lag0_divergence_z120_mean": wmean("z120"),
            "v11_lag0_divergence_z120_maxabs": float(np.nanmax(np.abs(z120))) if np.isfinite(z120).any() else 0.0,
            "v11_lag0_recoupling_mean": wmean("recoupling"),
            "v11_lag0_agreement_fraction": wmean("agreement"),
            "v11_lag0_network_dispersion": float(np.nanstd(peer)) if len(peer) > 1 else 0.0,
        })

    summary = long.groupby(["date", "ticker"], sort=False).apply(summarize, include_groups=False).reset_index()
    merged = base[["date", "ticker"]].merge(summary, on=["date", "ticker"], how="left", validate="one_to_one")
    features = merged[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    features.index = frame.index
    return features, pd.DataFrame(used_rows)


def build_preexposed_directed_features(frame: pd.DataFrame, directed_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build V11 directed features as an explicitly pre-exposed diagnostic block.

    These columns must never be eligible for the primary V13 champion. They are
    generated only for component diagnostics because V11/V11.1 inspected forward
    folds while choosing/interpreting these edges.
    """
    required = {"date", "ticker", "return_1d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"preexposed directed features missing columns: {sorted(missing)}")
    edges = directed_edges.copy()
    edges["leader"] = edges["leader"].map(normalize_ticker)
    edges["follower"] = edges["follower"].map(normalize_ticker)
    edges["directed_lag"] = pd.to_numeric(edges["directed_lag"], errors="coerce").astype("Int64")
    base = frame[["date", "ticker", "return_1d"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="raise")
    base["ticker"] = base["ticker"].map(normalize_ticker)
    panel = base.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for _, row in edges.dropna(subset=["leader", "follower", "directed_lag"]).iterrows():
        leader, follower, lag = str(row["leader"]), str(row["follower"]), int(row["directed_lag"])
        if lag <= 0 or leader not in panel or follower not in panel:
            continue
        series = panel[leader]
        # For follower forecast date t, horizon k aligns to leader[t + k - lag].
        # Components with k > lag require a future leader return and are therefore
        # unavailable at t; keep them NaN rather than repeating leader[t].
        aligned = [
            series.shift(lag - k) if lag - k >= 0 else pd.Series(np.nan, index=series.index)
            for k in (1, 2, 3)
        ]
        values = pd.concat(aligned, axis=1)
        values.columns = ["d1", "d2", "d3"]
        corr = float(pd.to_numeric(pd.Series([row.get("discovery_best_lag_correlation", row.get("discovery_correlation", np.nan))]), errors="coerce").iloc[0])
        sign = 1.0 if not math.isfinite(corr) or corr >= 0 else -1.0
        part = pd.DataFrame({
            "date": panel.index,
            "ticker": follower,
            "v11_preexp_incoming_count": 1.0,
            "v11_preexp_aligned_mean": values.mean(axis=1).to_numpy(float),
            "v11_preexp_aligned_max": values.max(axis=1).to_numpy(float),
            "v11_preexp_aligned_min": values.min(axis=1).to_numpy(float),
            "v11_preexp_oriented_mean": (sign * values.mean(axis=1)).to_numpy(float),
            "v11_preexp_upshock": (values.max(axis=1) >= 0.02).astype(float).to_numpy(),
            "v11_preexp_downshock": (values.min(axis=1) <= -0.02).astype(float).to_numpy(),
        })
        parts.append(part)
        manifest_rows.append({
            "leader": leader, "follower": follower, "directed_lag": lag,
            "discovery_correlation": corr, "relation_sign": int(sign),
            "available_horizon_components": int(min(lag, 3)),
            "preexposed": True, "eligible_for_primary_champion": False,
        })
    columns = [
        "v11_preexp_incoming_count", "v11_preexp_aligned_mean", "v11_preexp_aligned_max",
        "v11_preexp_aligned_min", "v11_preexp_oriented_mean", "v11_preexp_upshock", "v11_preexp_downshock",
    ]
    if not parts:
        return pd.DataFrame(0.0, index=frame.index, columns=columns), pd.DataFrame(manifest_rows)
    long = pd.concat(parts, ignore_index=True)
    agg = long.groupby(["date", "ticker"], sort=False).agg({
        "v11_preexp_incoming_count": "sum",
        "v11_preexp_aligned_mean": "mean",
        "v11_preexp_aligned_max": "max",
        "v11_preexp_aligned_min": "min",
        "v11_preexp_oriented_mean": "mean",
        "v11_preexp_upshock": "max",
        "v11_preexp_downshock": "max",
    }).reset_index()
    merged = base[["date", "ticker"]].merge(agg, on=["date", "ticker"], how="left", validate="one_to_one")
    features = merged[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    features.index = frame.index
    return features, pd.DataFrame(manifest_rows)


def empirical_rank(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    good = np.isfinite(x)
    if len(ref):
        out[good] = np.searchsorted(ref, x[good], side="right") / float(len(ref))
    return out


def grouped_historical_rank(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    score_column: str,
    group_column: str = "ticker",
    minimum_group_rows: int = 30,
) -> np.ndarray:
    global_ref = pd.to_numeric(train[score_column], errors="coerce").to_numpy(float)
    output = np.full(len(evaluation), np.nan, dtype=float)
    groups = evaluation[group_column].astype(str).to_numpy()
    train_groups = train[group_column].astype(str)
    for group in np.unique(groups):
        pos = np.flatnonzero(groups == group)
        ref = pd.to_numeric(train.loc[train_groups.eq(group), score_column], errors="coerce").to_numpy(float)
        if np.isfinite(ref).sum() < int(minimum_group_rows):
            ref = global_ref
        output[pos] = empirical_rank(ref, pd.to_numeric(evaluation.iloc[pos][score_column], errors="coerce"))
    return output


def date_cross_sectional_rank(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    """Percentile rank within date, independent of the caller's index labels."""
    work = frame[["date", score_column]].copy().reset_index(drop=True)
    ranked = work.groupby("date", sort=False)[score_column].rank(pct=True, method="average")
    return pd.to_numeric(ranked, errors="coerce").to_numpy(float)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_self_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Past-only ticker self momentum/reversion features inspired by V11.

    Rolling relationship estimates are shifted so that the relation used at date t
    is estimated only from completed return pairs ending no later than t-1.  The
    current return itself is known at the close of t and can be combined with that
    historical state for the D+1..D+3 forecast.
    """
    required = {"ticker", "date", "return_1d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"self-state features missing columns: {sorted(missing)}")
    work = frame[["ticker", "date", "return_1d"]].copy()
    work["__pos"] = np.arange(len(work), dtype=np.int64)
    work["ticker"] = work["ticker"].map(normalize_ticker)
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work["return_1d"] = pd.to_numeric(work["return_1d"], errors="coerce")
    work = work.sort_values(["ticker", "date", "__pos"], kind="mergesort")
    parts: list[pd.DataFrame] = []
    for _, grp in work.groupby("ticker", sort=False):
        r = grp["return_1d"].astype(float)
        past = r.shift(1)
        past2 = r.shift(2)
        past3 = r.shift(3)
        corr60 = past.rolling(60, min_periods=30).corr(past2)
        corr120 = past.rolling(120, min_periods=45).corr(past2)
        corr2_120 = past.rolling(120, min_periods=45).corr(past3)
        mean60 = past.rolling(60, min_periods=20).mean()
        std60 = past.rolling(60, min_periods=20).std(ddof=0).replace(0, np.nan)
        z60 = (r - mean60) / std60

        # Historical one-step responses.  At t, event_return=r[t-2] and
        # realized_follow=r[t-1], hence no t+1 information is used.
        event_return = r.shift(2)
        realized_follow = r.shift(1)
        up_mask = event_return.gt(0).astype(float)
        down_mask = event_return.lt(0).astype(float)
        up_count = up_mask.rolling(120, min_periods=20).sum().replace(0, np.nan)
        down_count = down_mask.rolling(120, min_periods=20).sum().replace(0, np.nan)
        up_follow = (realized_follow.where(event_return.gt(0), 0.0)).rolling(120, min_periods=20).sum() / up_count
        down_follow = (realized_follow.where(event_return.lt(0), 0.0)).rolling(120, min_periods=20).sum() / down_count
        expected_follow = np.where(r.to_numpy(float) >= 0, up_follow.to_numpy(float), down_follow.to_numpy(float))

        sign = np.sign(r)
        same2 = ((sign == np.sign(r.shift(1))) & r.notna() & r.shift(1).notna()).astype(float)
        same3 = ((sign == np.sign(r.shift(1))) & (sign == np.sign(r.shift(2))) & r.shift(2).notna()).astype(float)
        out = pd.DataFrame({
            "__pos": grp["__pos"].to_numpy(np.int64),
            "v11_self_autocorr_lag1_60": corr60.to_numpy(float),
            "v11_self_autocorr_lag1_120": corr120.to_numpy(float),
            "v11_self_autocorr_lag2_120": corr2_120.to_numpy(float),
            "v11_self_return_z60": z60.to_numpy(float),
            "v11_self_momentum_pressure_60": (r * corr60).to_numpy(float),
            "v11_self_momentum_pressure_120": (r * corr120).to_numpy(float),
            "v11_self_up_follow_mean_120": up_follow.to_numpy(float),
            "v11_self_down_follow_mean_120": down_follow.to_numpy(float),
            "v11_self_expected_follow_120": expected_follow,
            "v11_self_asymmetry_120": (up_follow - down_follow).to_numpy(float),
            "v11_self_same_sign_2d": same2.to_numpy(float),
            "v11_self_same_sign_3d": same3.to_numpy(float),
            "v11_self_return_sum_3": (r + r.shift(1) + r.shift(2)).to_numpy(float),
            "v11_self_return_sum_5": r.rolling(5, min_periods=2).sum().to_numpy(float),
        })
        parts.append(out)
    merged = pd.concat(parts, ignore_index=True).sort_values("__pos", kind="mergesort")
    merged = merged.drop(columns="__pos").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged.index = frame.index
    return merged


def derive_future_path_labels_from_history(
    frame: pd.DataFrame,
    return_history: pd.DataFrame,
    *,
    threshold: float = 0.05,
    target_column: str = TARGET_COLUMN,
    max_mismatch_rate: float = 0.01,
    allow_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive labels using an unfiltered source return history.

    The V10 loader intentionally keeps only target-valid rows.  The final three
    source rows per ticker are therefore absent from ``frame`` even though their
    returns are required to reconstruct the path for the last target-valid dates.
    This function computes D+1..D+3 paths on the unfiltered source table first and
    then joins the result to the target-valid model frame.
    """
    required_frame = {"ticker", "date", target_column}
    required_history = {"ticker", "date"}
    missing_frame = required_frame - set(frame.columns)
    missing_history = required_history - set(return_history.columns)
    if missing_frame:
        raise ValueError(f"target-valid frame missing columns: {sorted(missing_frame)}")
    if missing_history:
        raise ValueError(f"return history missing columns: {sorted(missing_history)}")

    history = return_history.copy()
    history["ticker"] = history["ticker"].map(normalize_ticker)
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history["return_1d"], return_source, applied_scale = return_series(history)
    history["__history_pos"] = np.arange(len(history), dtype=np.int64)
    order_cols = ["ticker", "date"]
    if "source_row_id" in history.columns:
        order_cols.append("source_row_id")
    else:
        order_cols.append("__history_pos")
    history = history.sort_values(order_cols, kind="mergesort")
    grouped = history.groupby("ticker", sort=False)["return_1d"]
    r1 = grouped.shift(-1)
    r2 = grouped.shift(-2)
    r3 = grouped.shift(-3)
    c1 = r1
    c2 = (1.0 + r1) * (1.0 + r2) - 1.0
    c3 = (1.0 + r1) * (1.0 + r2) * (1.0 + r3) - 1.0
    path = np.column_stack([c1.to_numpy(float), c2.to_numpy(float), c3.to_numpy(float)])
    valid = np.all(np.isfinite(path), axis=1)
    up = np.full(len(history), np.nan, dtype=float)
    down = np.full(len(history), np.nan, dtype=float)
    up[valid] = np.max(path[valid], axis=1)
    down[valid] = np.min(path[valid], axis=1)
    history["future_cumret_d1"] = c1.to_numpy(float)
    history["future_cumret_d2"] = c2.to_numpy(float)
    history["future_cumret_d3"] = c3.to_numpy(float)
    history["future_up_excursion_3d_reconstructed"] = up
    history["future_down_excursion_3d"] = down
    history["future_path_valid"] = valid
    join_keys = ["source_row_id"] if "source_row_id" in frame.columns and "source_row_id" in history.columns else ["ticker", "date"]
    payload_value_columns = [
        "return_1d", "future_cumret_d1", "future_cumret_d2", "future_cumret_d3",
        "future_up_excursion_3d_reconstructed", "future_down_excursion_3d", "future_path_valid",
    ]
    if join_keys == ["source_row_id"]:
        payload = history[["source_row_id", "ticker", "date", *payload_value_columns]].copy().rename(
            columns={"ticker": "__history_ticker", "date": "__history_date"}
        )
    else:
        payload = history[[*join_keys, *payload_value_columns]].copy()
    if payload.duplicated(join_keys).any():
        raise ValueError(f"Return-history join keys are not unique: {join_keys}")

    work = frame.copy()
    work["ticker"] = work["ticker"].map(normalize_ticker)
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work["__original_position"] = np.arange(len(work), dtype=np.int64)
    merged = work.merge(payload, on=join_keys, how="left", validate="one_to_one", suffixes=("", "_history"))
    history_metadata_mismatches = 0
    history_join_missing_rows = 0
    if join_keys == ["source_row_id"]:
        history_ticker = merged["__history_ticker"].map(normalize_ticker)
        history_date = pd.to_datetime(merged["__history_date"], errors="coerce")
        history_join_missing = merged["__history_ticker"].isna() | merged["__history_date"].isna()
        metadata_mismatch = (~history_join_missing) & (
            ~history_ticker.eq(merged["ticker"]) | ~history_date.eq(merged["date"])
        )
        history_join_missing_rows = int(history_join_missing.sum())
        history_metadata_mismatches = int(metadata_mismatch.sum())
        if history_join_missing_rows or history_metadata_mismatches:
            sample_columns = ["source_row_id", "ticker", "date", "__history_ticker", "__history_date"]
            sample = merged.loc[history_join_missing | metadata_mismatch, sample_columns].head(10).to_dict("records")
            raise RuntimeError(
                "Source-row identity mismatch while joining full return history: "
                f"missing={history_join_missing_rows}, metadata_mismatches={history_metadata_mismatches}, "
                f"sample={sample}"
            )
    reconstructed_up = pd.to_numeric(merged["future_up_excursion_3d_reconstructed"], errors="coerce").to_numpy(float)
    down_values = pd.to_numeric(merged["future_down_excursion_3d"], errors="coerce").to_numpy(float)
    valid_values = merged["future_path_valid"].fillna(False).astype(bool).to_numpy()
    up_values = reconstructed_up.copy()
    if "best_forward_return_3d" in merged.columns:
        official_best = pd.to_numeric(merged["best_forward_return_3d"], errors="coerce").to_numpy(float)
        replace = np.isfinite(official_best)
        up_values[replace] = official_best[replace]

    official = pd.to_numeric(merged[target_column], errors="coerce").to_numpy(float)
    reconstructed_target = np.where(np.isfinite(up_values), (up_values >= float(threshold)).astype(float), np.nan)
    audit_mask = np.isfinite(official) & np.isfinite(reconstructed_target)
    mismatch = audit_mask & (official.astype(np.int8) != reconstructed_target.astype(np.int8))
    mismatch_rate = float(mismatch.sum() / max(audit_mask.sum(), 1))
    if mismatch_rate > float(max_mismatch_rate) and not allow_mismatch:
        raise RuntimeError(
            f"Official surge target reconstruction mismatch {mismatch_rate:.4%} exceeds "
            f"{max_mismatch_rate:.4%}. Check source return scaling/target definition."
        )
    abs_excursion = np.maximum(up_values, -down_values)
    move = np.where(valid_values & np.isfinite(abs_excursion), (abs_excursion >= float(threshold)).astype(np.int8), -1)
    official_int = np.where(np.isfinite(official), official, -1).astype(np.int8)
    direction = np.where(move == 1, official_int, -1)
    merged["future_up_excursion_3d"] = up_values
    merged["future_abs_excursion_3d"] = abs_excursion
    merged["future_direction_margin_3d"] = up_values - np.abs(down_values)
    merged[MOVE_COLUMN] = move
    merged[DIRECTION_COLUMN] = direction
    merged["surge_target_reconstructed"] = reconstructed_target
    merged = merged.sort_values("__original_position", kind="mergesort").drop(
        columns=[
            "__original_position", "future_up_excursion_3d_reconstructed",
            "__history_ticker", "__history_date",
        ],
        errors="ignore",
    )
    merged.index = frame.index
    audit = {
        "return_source": return_source,
        "return_scale_applied": applied_scale,
        "threshold": float(threshold),
        "rows": int(len(merged)),
        "return_history_rows": int(len(history)),
        "future_path_valid_rows": int(valid_values.sum()),
        "future_path_invalid_rows": int((~valid_values).sum()),
        "official_target_audit_rows": int(audit_mask.sum()),
        "official_target_mismatches": int(mismatch.sum()),
        "official_target_mismatch_rate": mismatch_rate,
        "absolute_move_positive": int(np.sum(move == 1)),
        "absolute_move_negative": int(np.sum(move == 0)),
        "surge_within_absolute_move": int(np.sum((move == 1) & (official_int == 1))),
        "down_or_non_surge_large_move": int(np.sum((move == 1) & (official_int == 0))),
        "join_keys": join_keys,
        "history_join_missing_rows": int(history_join_missing_rows),
        "history_metadata_mismatches": int(history_metadata_mismatches),
    }
    return merged, audit
