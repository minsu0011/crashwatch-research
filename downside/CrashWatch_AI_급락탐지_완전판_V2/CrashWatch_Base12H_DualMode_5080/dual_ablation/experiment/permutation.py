from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import get_paths
from ..io_utils import atomic_csv, finite_float32, normalize_date, normalize_ticker
from .models import fit_model
from .runner import load_catalog, select_valid_features
from .splits import make_walk_forward_folds


def _group_shuffle(block: pd.DataFrame, columns: list[str], seed: int, conditional: bool) -> pd.DataFrame:
    out = block.copy()
    rng = np.random.default_rng(seed)
    if conditional:
        conditions = [c for c in ["bucket", "market"] if c in out.columns]
        if "u_market_kospi_ret_20" in out.columns:
            out["_regime_bin"] = pd.qcut(out["u_market_kospi_ret_20"], 5, duplicates="drop")
            conditions.append("_regime_bin")
        grouped = out.groupby(conditions, dropna=False, sort=False) if conditions else [(None, out)]
        for _, g in grouped:
            order = rng.permutation(g.index.to_numpy())
            source = out.loc[order, columns].to_numpy()
            out.loc[g.index, columns] = source
        return out.drop(columns="_regime_bin", errors="ignore")
    order = rng.permutation(out.index.to_numpy())
    out.loc[:, columns] = out.loc[order, columns].to_numpy()
    return out


def run_grouped_permutation(project: Path | None = None, dataset_path: Path | None = None, target: str = "label_abs_crash_20", seed: int = 17, conditional: bool = True) -> pd.DataFrame:
    paths = get_paths(project)
    dataset_path = dataset_path or paths.data_root / "development" / "training_dataset_dual.parquet"
    df = normalize_date(pd.read_parquet(dataset_path))
    df["ticker"] = normalize_ticker(df["ticker"])
    u = load_catalog(paths.feature_dual / "feature_catalog_universe.json")
    t = load_catalog(paths.feature_dual / "feature_catalog_ticker.json")
    catalogs = {**{f"universe::{k}": v for k, v in u.items()}, **{f"ticker::{k}": v for k, v in t.items()}}
    candidates = sorted(set(sum(catalogs.values(), [])))
    valid, _ = select_valid_features(df, candidates)
    valid_set = set(valid)
    folds = make_walk_forward_folds(df["date"], n_folds=1)
    fold = folds[0]
    train = df.loc[df["date"].isin(fold["train_dates_index"])]
    val = df.loc[df["date"].isin(fold["validation_dates_index"])]
    model = fit_model(finite_float32(train, valid), train[target].astype(int).to_numpy(), seed, prefer_gpu=True)
    baseline = model.predict_proba(finite_float32(val, valid))
    baseline_score = average_precision_score(val[target], baseline) if val[target].nunique() == 2 else np.nan
    rows = []
    for name, cols in catalogs.items():
        cols = [c for c in cols if c in valid_set]
        if not cols:
            continue
        permuted = _group_shuffle(val, cols, seed, conditional)
        score = average_precision_score(permuted[target], model.predict_proba(finite_float32(permuted, valid))) if permuted[target].nunique() == 2 else np.nan
        rows.append({"group": name, "feature_count": len(cols), "baseline_pr_auc": baseline_score, "permuted_pr_auc": score, "pr_auc_loss": baseline_score - score, "conditional": conditional})
    out = pd.DataFrame(rows).sort_values("pr_auc_loss", ascending=False)
    atomic_csv(out, paths.result_dual / "grouped_conditional_permutation.csv")
    return out
