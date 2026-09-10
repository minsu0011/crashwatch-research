from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from surge_model_zoo_common import derive_forward_targets


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))

def build_synthetic_package(root: Path, date_count: int = 300, ticker_count: int = 6) -> dict[str, Path]:
    rng = np.random.default_rng(20260810)
    dates = pd.bdate_range("2020-01-01", periods=date_count)
    records: list[dict[str, object]] = []
    row_id = 0
    for ticker_id in range(ticker_count):
        ticker = f"T{ticker_id:02d}"
        signal = np.zeros(date_count, dtype=np.float64)
        regime = np.sin(np.linspace(0.0, 8.0 * np.pi, date_count) + ticker_id * 0.13)
        for index in range(1, date_count):
            signal[index] = 0.78 * signal[index - 1] + rng.normal(0.0, 0.72)
        returns = rng.normal(0.0002, 0.008, size=date_count)
        for index in range(1, date_count):
            up_probability = float(sigmoid(np.asarray([-3.05 + 1.05 * signal[index - 1] + 0.25 * regime[index - 1]]))[0])
            down_probability = float(sigmoid(np.asarray([-3.55 - 0.45 * signal[index - 1]]))[0])
            draw = rng.random()
            if draw < up_probability:
                returns[index] = 0.058 + rng.normal(0.0, 0.004)
            elif draw < up_probability + down_probability:
                returns[index] = -0.058 + rng.normal(0.0, 0.004)
        for index, date in enumerate(dates):
            records.append(
                {
                    "source_row_id": row_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sealed_do_not_train_or_tune": 0,
                    "industry_name": f"I{ticker_id % 2}",
                    "market": "KOSPI" if ticker_id % 2 == 0 else "KOSDAQ",
                    "bucket": f"B{ticker_id % 3}",
                    "t_price_ret_1": float(returns[index]),
                    "f_signal": float(signal[index]),
                    "f_signal_noise": float(signal[index] + rng.normal(0.0, 0.35)),
                    "f_regime": float(regime[index]),
                    "f_ret_lag": float(returns[index]),
                    "f_interaction": float(signal[index] * (1.0 + 0.4 * regime[index])),
                    "f_noise_1": float(rng.normal()),
                    "f_noise_2": float(rng.normal()),
                    "f_sparse": float(signal[index]) if rng.random() > 0.16 else np.nan,
                }
            )
            row_id += 1
    source = pd.DataFrame.from_records(records)
    derived = derive_forward_targets(
        source,
        return_column="t_price_ret_1",
        ticker_column="ticker",
        date_column="date",
        threshold=0.05,
    )
    sidecar = source[["source_row_id", "date", "ticker"]].copy()
    sidecar["target_valid"] = derived["forward_valid"].eq(1.0)
    sidecar["label_abs_surge_3d_5pct"] = derived["surge_d3"]
    first_hit: list[float] = []
    best_forward: list[float] = []
    for _, group in source.groupby("ticker", sort=False):
        local_returns = group["t_price_ret_1"].to_numpy(dtype=np.float64)
        for local in range(len(group)):
            if local + 3 >= len(group):
                first_hit.append(np.nan)
                best_forward.append(np.nan)
                continue
            cumulative = np.cumprod(1.0 + local_returns[local + 1 : local + 4]) - 1.0
            hits = np.flatnonzero(cumulative >= 0.05 - 1e-12)
            first_hit.append(float(hits[0] + 1) if len(hits) else np.nan)
            best_forward.append(float(np.max(cumulative)))
    # Match the production target sidecar convention: 0 means no +5% hit.
    sidecar["first_hit_day"] = pd.Series(first_hit).fillna(0).astype(np.int8)
    sidecar["best_forward_return_3d"] = best_forward

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "synthetic_training.csv"
    target_path = data_dir / "synthetic_target.csv"
    source.to_csv(dataset_path, index=False)
    sidecar.to_csv(target_path, index=False)

    date_values = list(dates)
    folds = []
    boundaries = [
        (0, 119, 120, 124, 125, 164),
        (0, 159, 160, 164, 165, 204),
        (0, 199, 200, 204, 205, 244),
        (0, 239, 240, 244, 245, 284),
    ]
    for fold_id, (tr0, tr1, p0, p1, va0, va1) in enumerate(boundaries):
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": date_values[tr0].strftime("%Y-%m-%d"),
                "train_end": date_values[tr1].strftime("%Y-%m-%d"),
                "purge_start": date_values[p0].strftime("%Y-%m-%d"),
                "purge_end": date_values[p1].strftime("%Y-%m-%d"),
                "validation_start": date_values[va0].strftime("%Y-%m-%d"),
                "validation_end": date_values[va1].strftime("%Y-%m-%d"),
            }
        )
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps({"folds": folds}, ensure_ascii=False, indent=2), encoding="utf-8")

    all_features = [
        "f_signal",
        "f_signal_noise",
        "f_regime",
        "f_ret_lag",
        "f_interaction",
        "f_noise_1",
        "f_noise_2",
        "f_sparse",
    ]
    profiles = {
        "profiles": {
            "P0_ALL_VALID": {"features": all_features},
            "P1_SELECTION_TOP": {"features": all_features[:6]},
            "P7S_UNIVARIATE_TOP3_LIFT": {"features": ["f_signal", "f_signal_noise", "f_interaction", "f_sparse"]},
            "P4S_METRIC_CONSISTENT_DIRECTIONAL": {"features": ["f_signal", "f_ret_lag", "f_regime"]},
        }
    }
    profiles_path = root / "profiles.json"
    profiles_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    recipes = {
        "schema": "synthetic_model_zoo_test_v1",
        "custom_profiles": {},
        "quick_recipe_names": ["lgb_signal", "extra_signal"],
        "recipes": [
            {
                "name": "lgb_signal",
                "family": "lightgbm",
                "profile": "P7S_UNIVARIATE_TOP3_LIFT",
                "target_variant": "surge_d3",
                "train_policy": "expanding",
                "positive_weight_mode": "sqrt_balance",
                "params": {
                    "learning_rate": 0.08,
                    "num_leaves": 15,
                    "min_data_in_leaf": 15,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                },
            },
            {
                "name": "extra_signal",
                "family": "extra_trees",
                "profile": "P1_SELECTION_TOP",
                "target_variant": "surge_d3",
                "train_policy": "rolling",
                "rolling_days": 100,
                "positive_weight_mode": "sqrt_balance",
                "params": {
                    "n_estimators": 80,
                    "max_features": 0.75,
                    "min_samples_leaf": 3,
                    "max_depth": 10,
                },
            },
        ],
    }
    recipes_path = root / "recipes.json"
    recipes_path.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset": dataset_path,
        "target": target_path,
        "folds": folds_path,
        "profiles": profiles_path,
        "recipes": recipes_path,
    }


