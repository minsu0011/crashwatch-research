from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ..config import get_paths
from ..io_utils import atomic_csv, finite_float32, normalize_date, normalize_ticker
from .models import fit_model
from .runner import load_base_features, load_catalog, select_valid_features
from .splits import make_walk_forward_folds


def _shuffle_rows(frame: pd.DataFrame, columns: list[str], conditions: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(conditions, dropna=False, sort=False, observed=False) if conditions else [(None, out)]
    for _, group in grouped:
        if len(group) < 2:
            continue
        order = rng.permutation(group.index.to_numpy())
        out.loc[group.index, columns] = out.loc[order, columns].to_numpy()
    return out


def _group_shuffle(
    block: pd.DataFrame,
    columns: list[str],
    seed: int,
    conditional: bool,
    namespace: str = "ticker",
) -> pd.DataFrame:
    """Jointly permute a feature group without destroying its data geometry.

    Universe features are one value per date, so they must be shuffled as date blocks.
    Ticker features are shuffled inside each ticker. Conditional mode additionally keeps
    calendar year and broad market regime fixed. This avoids manufacturing impossible
    cross-sectional market states or mixing company-specific scales.
    """
    out = block.copy()
    rng = np.random.default_rng(seed)
    out["_year"] = pd.to_datetime(out["date"], errors="coerce").dt.year
    regime_column = next((c for c in [
        "u_market_kospi_ret_20", "u_global_kospi_ret_5", "u_global_kospi_ret_1",
    ] if c in out.columns), None)
    if regime_column:
        regime_source = pd.to_numeric(out[regime_column], errors="coerce")
        try:
            out["_regime_bin"] = pd.qcut(regime_source, 5, duplicates="drop")
        except ValueError:
            out["_regime_bin"] = "all"

    if namespace == "universe":
        context = [c for c in ["_year", "_regime_bin"] if conditional and c in out.columns]
        daily_cols = ["date", *columns, *context]
        daily = out[daily_cols].sort_values("date").drop_duplicates("date", keep="last")
        shuffled = _shuffle_rows(daily, columns, context, rng)
        replacement = shuffled[["date", *columns]]
        out = out.drop(columns=columns).merge(replacement, on="date", how="left", validate="many_to_one")
    else:
        context = ["ticker"] if "ticker" in out.columns else []
        if conditional:
            context += [c for c in ["_year", "_regime_bin"] if c in out.columns]
        out = _shuffle_rows(out, columns, context, rng)
    return out.drop(columns=["_year", "_regime_bin"], errors="ignore")


def _scores(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    target = pd.to_numeric(y, errors="coerce").astype(int).to_numpy()
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    two_classes = np.unique(target).size == 2
    return {
        "pr_auc": float(average_precision_score(target, probability)) if two_classes else np.nan,
        "roc_auc": float(roc_auc_score(target, probability)) if two_classes else np.nan,
        "brier": float(brier_score_loss(target, probability)),
        "logloss": float(log_loss(target, probability, labels=[0, 1])),
    }


def run_grouped_permutation(project: Path | None = None, dataset_path: Path | None = None, target: str = "label_abs_crash_20", seed: int = 17, conditional: bool = True) -> pd.DataFrame:
    paths = get_paths(project)
    dataset_path = dataset_path or paths.data_root / "development" / "training_dataset_dual.parquet"
    df = normalize_date(pd.read_parquet(dataset_path))
    df["ticker"] = normalize_ticker(df["ticker"])
    u = load_catalog(paths.feature_dual / "feature_catalog_universe.json")
    t = load_catalog(paths.feature_dual / "feature_catalog_ticker.json")
    catalogs = {**{f"universe::{k}": v for k, v in u.items()}, **{f"ticker::{k}": v for k, v in t.items()}}
    candidates = sorted(set(sum(catalogs.values(), []) + load_base_features(paths)))
    valid, _ = select_valid_features(df, candidates)
    valid_set = set(valid)
    folds = make_walk_forward_folds(df["date"], n_folds=1)
    fold = folds[0]
    train = df.loc[df["date"].isin(fold["train_dates_index"])]
    val = df.loc[df["date"].isin(fold["validation_dates_index"])]
    model = fit_model(finite_float32(train, valid), train[target].astype(int).to_numpy(), seed, prefer_gpu=True)
    baseline = model.predict_proba(finite_float32(val, valid))
    baseline_scores = _scores(val[target], baseline)
    rows = []
    for name, cols in catalogs.items():
        cols = [c for c in cols if c in valid_set]
        if not cols:
            continue
        namespace = name.split("::", 1)[0]
        permuted = _group_shuffle(val, cols, seed, conditional, namespace=namespace)
        scores = _scores(permuted[target], model.predict_proba(finite_float32(permuted, valid)))
        rows.append({
            "group": name,
            "namespace": namespace,
            "feature_count": len(cols),
            "baseline_pr_auc": baseline_scores["pr_auc"],
            "permuted_pr_auc": scores["pr_auc"],
            "pr_auc_loss": baseline_scores["pr_auc"] - scores["pr_auc"],
            "baseline_roc_auc": baseline_scores["roc_auc"],
            "permuted_roc_auc": scores["roc_auc"],
            "roc_auc_loss": baseline_scores["roc_auc"] - scores["roc_auc"],
            "baseline_brier": baseline_scores["brier"],
            "permuted_brier": scores["brier"],
            "brier_increase": scores["brier"] - baseline_scores["brier"],
            "baseline_logloss": baseline_scores["logloss"],
            "permuted_logloss": scores["logloss"],
            "logloss_increase": scores["logloss"] - baseline_scores["logloss"],
            "conditional": conditional,
            "fold": int(fold["fold_id"]),
            "seed": int(seed),
        })
    out = pd.DataFrame(rows).sort_values("pr_auc_loss", ascending=False)
    suffix = "conditional" if conditional else "unconditional"
    atomic_csv(out, paths.result_dual / f"grouped_{suffix}_permutation.csv")
    return out
