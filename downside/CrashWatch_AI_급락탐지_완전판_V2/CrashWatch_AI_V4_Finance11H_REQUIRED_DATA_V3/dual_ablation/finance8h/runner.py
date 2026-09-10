from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..finance11h.model import CANDIDATES, ModelConfig, fit_xgb
from ..finance11h.monitor import (
    ResourceMonitor,
    prevent_windows_sleep,
    restore_windows_sleep,
)
from ..finance11h.runner import (
    Experiment,
    Finance11HRunner,
    _exact_sign_p,
    _metrics,
)
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, stable_hash

LOGGER = logging.getLogger(__name__)

PRIMARY_SEEDS = [17, 43, 79, 101, 137, 211, 307, 419, 541]
METHOD_SEEDS = [17, 79, 137]
CORRELATION_THRESHOLD = 0.95

LOSS_METRICS = [
    "pr_auc_loss",
    "roc_auc_loss",
    "balanced_accuracy_loss",
    "accuracy_loss",
    "brier_increase",
    "logloss_increase",
    "top_3pct_precision_loss",
    "top_3pct_recall_loss",
]


def _add_finance_subclusters(catalog: dict[str, list[str]]) -> None:
    short_cols = catalog.get("t_financial_shorting", [])
    lending_cols = catalog.get("t_stock_lending", [])
    subclusters = {
        "t_finshort_volume_cluster": [
            c for c in short_cols if "volume" in c or "value_ratio" in c
        ],
        "t_finshort_balance_cluster": [
            c
            for c in short_cols
            if "balance" in c
            and not any(k in c for k in ["change", "slope", "acceleration"])
        ],
        "t_finshort_dynamic_cluster": [
            c
            for c in short_cols
            if any(k in c for k in ["change", "slope", "acceleration", "divergence"])
        ],
        "t_finshort_crowding_cluster": [
            c
            for c in short_cols
            if any(k in c for k in ["crowding", "squeeze", "stress", "crash_pressure"])
        ],
        "t_lending_transaction_cluster": [
            c
            for c in lending_cols
            if any(k in c for k in ["contract", "net_shares", "net_value"])
        ],
        "t_lending_balance_cluster": [
            c
            for c in lending_cols
            if "balance" in c
            and not any(k in c for k in ["change", "slope", "acceleration"])
        ],
        "t_lending_dynamic_cluster": [
            c
            for c in lending_cols
            if any(k in c for k in ["change", "slope", "acceleration", "divergence"])
        ],
        "t_lending_crowding_cluster": [
            c for c in lending_cols if any(k in c for k in ["crowding", "squeeze"])
        ],
    }
    for group, columns in subclusters.items():
        columns = sorted(set(columns))
        if columns:
            catalog[group] = columns


def _summarize_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    keys = [
        "experiment",
        "phase",
        "ablation_method",
        "model_config",
        "stage",
        "mode",
        "group",
        "target_bucket",
        "target_ticker",
        "scope_type",
        "scope_value",
    ]
    rows: list[dict] = []
    for values, block in delta.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        by_fold = block.groupby("fold", as_index=False)[LOSS_METRICS].mean(
            numeric_only=True
        )
        row["fold_count"] = int(by_fold["fold"].nunique())
        row["seed_count"] = int(block["seed"].nunique())
        for metric in LOSS_METRICS:
            sample = (
                pd.to_numeric(by_fold[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            row[f"{metric}_mean"] = float(np.mean(sample)) if len(sample) else np.nan
            row[f"{metric}_median"] = (
                float(np.median(sample)) if len(sample) else np.nan
            )
            row[f"{metric}_std"] = (
                float(np.std(sample, ddof=1)) if len(sample) > 1 else np.nan
            )
            row[f"{metric}_positive_fold_ratio"] = (
                float(np.mean(sample > 0)) if len(sample) else np.nan
            )
            row[f"{metric}_sign_p"] = _exact_sign_p(sample)
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_ticker(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _conditional_source_rows(
    dates: np.ndarray,
    tickers: np.ndarray,
    regime_values: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Map each row to the same ticker at a shuffled date in the same regime."""
    frame = pd.DataFrame(
        {
            "row": np.arange(len(dates), dtype=int),
            "date": pd.to_datetime(dates),
            "ticker": np.asarray(tickers, dtype=str),
            "regime": pd.to_numeric(regime_values, errors="coerce"),
        }
    )
    daily = frame.groupby("date", as_index=False)["regime"].median()
    valid = daily["regime"].replace([np.inf, -np.inf], np.nan)
    if valid.notna().sum() >= 10 and valid.nunique(dropna=True) >= 3:
        ranked = valid.rank(method="average", pct=True)
        daily["bin"] = np.minimum(4, np.floor(ranked.fillna(0.5) * 5)).astype(int)
    else:
        daily["bin"] = 0
    rng = np.random.default_rng(seed)
    date_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for _, block in daily.groupby("bin", sort=True):
        target_dates = block["date"].to_numpy()
        source_dates = rng.permutation(target_dates)
        date_map.update(
            {
                pd.Timestamp(target): pd.Timestamp(source)
                for target, source in zip(target_dates, source_dates)
            }
        )
    lookup = {
        (pd.Timestamp(date), str(ticker)): int(row)
        for row, date, ticker in frame[["row", "date", "ticker"]].itertuples(
            index=False, name=None
        )
    }
    source = np.arange(len(frame), dtype=int)
    for row, date, ticker in frame[["row", "date", "ticker"]].itertuples(
        index=False, name=None
    ):
        mapped_date = date_map.get(pd.Timestamp(date), pd.Timestamp(date))
        source[int(row)] = lookup.get((mapped_date, str(ticker)), int(row))
    return source


class Finance8HRunner:
    def __init__(
        self,
        project: Path,
        dataset: Path | None = None,
        *,
        hours: float = 8.0,
        threads: int = 16,
        cache_namespace: str = "finance11h_v3",
        output_name: str = "ablation_finance8h",
        prefer_gpu: bool = True,
        overwrite_cache: bool = False,
    ) -> None:
        self.project = project.resolve()
        self.hours = float(hours)
        self.threads = int(threads)
        self.started = time.monotonic()
        self.deadline = self.started + self.hours * 3600
        self.reserve_seconds = 12 * 60
        self.output = (
            self.project / "crashwatch_ai_data" / output_name
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache_namespace = cache_namespace
        self.base = Finance11HRunner(
            self.project,
            dataset,
            hours=hours,
            threads=threads,
            folds=8,
            seeds=PRIMARY_SEEDS,
            validation_days=60,
            purge_days=20,
            prefer_gpu=prefer_gpu,
            overwrite_cache=overwrite_cache,
            cache_namespace=cache_namespace,
        )
        # Store new result tables separately while pointing model predictions at
        # the proven V3 cache namespace. Exact experiment/config/feature hashes
        # prevent incompatible reuse.
        self.base.result_dir = self.output
        self.base.cache_dir = (
            self.project
            / "crashwatch_ai_data"
            / "ablation_finance11h"
            / "prediction_cache"
            / cache_namespace
        )
        self.base.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base.started = self.started
        self.base.deadline = self.deadline
        self.base.reserve_seconds = self.reserve_seconds
        self.monitor = ResourceMonitor(
            self.output / "resource_usage.csv",
            interval_seconds=10,
            hard_gpu_temp=88,
            min_available_ram_gb=1.5,
        )
        self.metric_frames: list[pd.DataFrame] = []
        self.records: list[dict] = []
        self.new_task_seconds: list[float] = []
        self.phase_counts: dict[str, int] = {}
        self.plan: dict = {}
        self.prior_summary = pd.DataFrame()
        self.original_catalog: dict[str, list[str]] = {}
        self.correlation_clusters = pd.DataFrame()
        self.group_pairs = pd.DataFrame()

    def remaining_seconds(self) -> float:
        return self.deadline - time.monotonic()

    def can_start(self) -> bool:
        if self.monitor.abort_event.is_set():
            return False
        estimate = (
            float(np.median(self.new_task_seconds[-30:]))
            if self.new_task_seconds
            else 30.0
        )
        return self.remaining_seconds() > self.reserve_seconds + max(
            120.0, estimate * 2.0
        )

    def _checkpoint(self, *, force: bool = False) -> None:
        if not force and len(self.records) % 8:
            return
        if self.metric_frames:
            atomic_csv(
                pd.concat(self.metric_frames, ignore_index=True, sort=False),
                self.output / "all_metrics_incremental.csv",
            )
        atomic_csv(pd.DataFrame(self.records), self.output / "task_manifest.csv")
        atomic_json(
            {
                "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                "remaining_hours": self.remaining_seconds() / 3600,
                "completed_tasks": int(
                    sum(row.get("status") == "completed" for row in self.records)
                ),
                "cache_hits": int(
                    sum(row.get("cache_status") == "hit" for row in self.records)
                ),
                "new_models": int(
                    sum(row.get("cache_status") == "miss" for row in self.records)
                ),
                "median_new_task_seconds": (
                    float(np.median(self.new_task_seconds))
                    if self.new_task_seconds
                    else None
                ),
                "phase_counts": self.phase_counts,
                "temperature_abort": self.monitor.abort_event.is_set(),
            },
            self.output / "run_state.json",
        )

    def prepare(self) -> None:
        self.base.load()
        if "sealed_do_not_train_or_tune" in self.base.df.columns:
            sentinel = pd.to_numeric(
                self.base.df["sealed_do_not_train_or_tune"], errors="coerce"
            ).fillna(0)
            if bool((sentinel != 0).any()):
                raise RuntimeError(
                    "development contains sealed_do_not_train_or_tune != 0"
                )
        metadata = [fold["metadata"] for fold in self.base.folds]
        if any(int(item["validation_dates"]) != 60 for item in metadata):
            raise RuntimeError("validation 기간이 60거래일이 아닙니다.")
        purge_lengths = []
        for fold in self.base.folds:
            purge_lengths.append(
                len(
                    pd.bdate_range(
                        fold["metadata"]["purge_start"],
                        fold["metadata"]["purge_end"],
                    )
                )
            )
        if min(purge_lengths) < 20:
            raise RuntimeError("purge 간격이 20거래일 미만입니다.")
        _add_finance_subclusters(self.base.catalog)
        self.original_catalog = {
            group: sorted(set(columns))
            for group, columns in self.base.catalog.items()
            if columns
        }
        prior_path = (
            self.project
            / "crashwatch_ai_data"
            / "ablation_finance11h"
            / "ablation_summary.csv"
        )
        if prior_path.exists():
            self.prior_summary = pd.read_csv(
                prior_path,
                dtype={"target_ticker": "string", "scope_value": "string"},
            )
        self._build_correlation_assets()

    def _prior_group_effects(self) -> dict[str, float]:
        if self.prior_summary.empty:
            return {}
        block = self.prior_summary.loc[
            self.prior_summary["scope_type"].eq("all")
            & self.prior_summary["scope_value"].eq("all_validation")
            & self.prior_summary["group"].notna()
        ].copy()
        if block.empty:
            return {}
        block["abs_effect"] = block["pr_auc_loss_mean"].abs()
        block = block.sort_values(
            ["group", "fold_count", "seed_count", "abs_effect"],
            ascending=[True, False, False, False],
        ).drop_duplicates("group")
        return dict(zip(block["group"], block["pr_auc_loss_mean"]))

    def _build_correlation_assets(self) -> None:
        features = self.base.features
        sample_size = min(24_000, len(self.base.X))
        sample_idx = np.linspace(
            0, len(self.base.X) - 1, sample_size, dtype=int
        )
        sampled = pd.DataFrame(
            self.base.X[sample_idx],
            columns=features,
        )
        ranked = (
            sampled.rank(axis=0, pct=True, method="average")
            .fillna(0.5)
            .to_numpy(dtype=np.float32)
        )
        correlation = np.corrcoef(ranked, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlation, 1.0)
        absolute = np.abs(correlation)
        matrix = pd.DataFrame(correlation, columns=features)
        matrix.insert(0, "feature", features)
        atomic_parquet(matrix, self.output / "feature_spearman_correlation.parquet")

        upper_i, upper_j = np.triu_indices(len(features), k=1)
        values = absolute[upper_i, upper_j]
        selected = values >= 0.80
        edges = pd.DataFrame(
            {
                "feature_a": np.asarray(features)[upper_i[selected]],
                "feature_b": np.asarray(features)[upper_j[selected]],
                "correlation": correlation[upper_i[selected], upper_j[selected]],
                "abs_correlation": values[selected],
            }
        ).sort_values("abs_correlation", ascending=False)
        atomic_csv(edges, self.output / "correlation_edges_abs_ge_080.csv")

        distance = 1.0 - absolute
        np.fill_diagonal(distance, 0.0)
        hierarchy = linkage(
            squareform(distance, checks=False),
            method="complete",
        )
        labels = fcluster(
            hierarchy,
            t=1.0 - CORRELATION_THRESHOLD,
            criterion="distance",
        )
        owners: dict[str, list[str]] = {}
        for group, columns in self.original_catalog.items():
            for column in columns:
                owners.setdefault(column, []).append(group)
        prior_effects = self._prior_group_effects()
        cluster_rows: list[dict] = []
        cluster_catalog: dict[str, list[str]] = {}
        for raw_label in sorted(set(labels)):
            indices = np.flatnonzero(labels == raw_label)
            if len(indices) < 2:
                continue
            columns = sorted(features[index] for index in indices)
            name = f"corr95__{stable_hash(columns)}"
            within = absolute[np.ix_(indices, indices)]
            pair_values = within[np.triu_indices(len(indices), k=1)]
            groups = sorted(
                set(
                    group
                    for column in columns
                    for group in owners.get(column, [])
                )
            )
            prior_strength = max(
                [abs(prior_effects.get(group, 0.0)) for group in groups] or [0.0]
            )
            score = (
                len(columns)
                + 0.4 * len(groups)
                + float(np.mean(pair_values))
                + 25.0 * prior_strength
            )
            cluster_catalog[name] = columns
            for column in columns:
                cluster_rows.append(
                    {
                        "cluster": name,
                        "feature": column,
                        "cluster_size": len(columns),
                        "owner_groups": "|".join(owners.get(column, [])),
                        "cluster_group_count": len(groups),
                        "mean_abs_correlation": float(np.mean(pair_values)),
                        "min_abs_correlation": float(np.min(pair_values)),
                        "max_abs_correlation": float(np.max(pair_values)),
                        "prior_group_abs_effect_max": prior_strength,
                        "priority_score": score,
                    }
                )
        self.correlation_clusters = pd.DataFrame(cluster_rows).sort_values(
            ["priority_score", "cluster", "feature"],
            ascending=[False, True, True],
        )
        atomic_csv(
            self.correlation_clusters,
            self.output / "correlation_clusters.csv",
        )
        atomic_json(
            {
                "threshold": CORRELATION_THRESHOLD,
                "method": "Spearman rank, complete-linkage on 1-|rho|",
                "sample_rows": sample_size,
                "clusters": cluster_catalog,
            },
            self.output / "correlation_cluster_catalog.json",
        )
        for name, columns in cluster_catalog.items():
            self.base.catalog[name] = columns

        feature_position = {feature: index for index, feature in enumerate(features)}
        pair_rows: list[dict] = []
        group_items = [
            (group, [c for c in columns if c in feature_position])
            for group, columns in self.original_catalog.items()
            if len([c for c in columns if c in feature_position]) >= 2
        ]
        for left_pos, (left, left_columns) in enumerate(group_items):
            left_set = set(left_columns)
            for right, right_columns in group_items[left_pos + 1 :]:
                right_set = set(right_columns)
                left_unique = sorted(left_set - right_set)
                right_unique = sorted(right_set - left_set)
                if len(left_unique) < 2 or len(right_unique) < 2:
                    continue
                left_idx = [feature_position[c] for c in left_unique]
                right_idx = [feature_position[c] for c in right_unique]
                cross = absolute[np.ix_(left_idx, right_idx)].ravel()
                if not len(cross):
                    continue
                top = np.sort(cross)[-min(10, len(cross)) :]
                combined = sorted(left_set | right_set)
                name = f"corrpair__{stable_hash([left, right, *combined])}"
                effect = max(
                    abs(prior_effects.get(left, 0.0)),
                    abs(prior_effects.get(right, 0.0)),
                )
                pair_rows.append(
                    {
                        "pair_group": name,
                        "group_a": left,
                        "group_b": right,
                        "feature_count": len(combined),
                        "max_abs_correlation": float(np.max(cross)),
                        "p95_abs_correlation": float(np.quantile(cross, 0.95)),
                        "top10_mean_abs_correlation": float(np.mean(top)),
                        "prior_group_abs_effect_max": effect,
                        "priority_score": (
                            float(np.max(cross))
                            + float(np.quantile(cross, 0.95))
                            + float(np.mean(top))
                            + 20.0 * effect
                        ),
                        "features": "|".join(combined),
                    }
                )
                self.base.catalog[name] = combined
        self.group_pairs = pd.DataFrame(pair_rows).sort_values(
            "priority_score", ascending=False
        )
        atomic_csv(
            self.group_pairs,
            self.output / "correlated_group_pair_candidates.csv",
        )

    def _seed_groups(self) -> list[str]:
        mandatory = [
            "u_financial_market",
            "t_financial_flow",
            "u_financial_shorting",
            "t_financial_shorting",
            "t_stock_lending",
            "u_etf_pressure",
            "t_limit_stress",
            "t_volume_price_pressure",
            "t_network_contagion",
            "t_microstructure_proxy",
        ]
        if self.prior_summary.empty:
            return [g for g in mandatory if self.base.catalog.get(g)]
        block = self.prior_summary.loc[
            self.prior_summary["scope_type"].eq("all")
            & self.prior_summary["scope_value"].eq("all_validation")
            & self.prior_summary["stage"].isin(
                ["financial_global", "supplemental"]
            )
            & self.prior_summary["group"].notna()
        ].copy()
        block = block.sort_values(
            ["fold_count", "seed_count", "pr_auc_loss_mean"],
            ascending=[False, False, False],
        ).drop_duplicates("group")
        positives = block.sort_values("pr_auc_loss_mean", ascending=False)[
            "group"
        ].head(6)
        negatives = block.sort_values("pr_auc_loss_mean", ascending=True)[
            "group"
        ].head(2)
        ordered = list(positives) + mandatory + list(negatives)
        result: list[str] = []
        for group in ordered:
            if group not in result and self.base.catalog.get(group):
                result.append(group)
            if len(result) == 10:
                break
        return result

    def _prior_experiment(self, group: str) -> Experiment:
        if not self.prior_summary.empty:
            block = self.prior_summary.loc[
                self.prior_summary["group"].eq(group)
                & self.prior_summary["scope_type"].eq("all")
                & self.prior_summary["scope_value"].eq("all_validation")
            ].sort_values(
                ["fold_count", "seed_count"],
                ascending=False,
            )
            if not block.empty:
                row = block.iloc[0]
                return Experiment(
                    str(row["experiment"]),
                    str(row["stage"]),
                    str(row["mode"]),
                    group,
                )
        return Experiment(
            f"confirm__{group}",
            "seed_stability",
            "global_drop",
            group,
        )

    def _target_experiments(
        self,
    ) -> tuple[list[Experiment], list[Experiment]]:
        if self.prior_summary.empty:
            return [], []
        bucket = self.prior_summary.loc[
            self.prior_summary["stage"].eq("bucket")
            & self.prior_summary["scope_type"].eq("target_bucket")
            & self.prior_summary["target_bucket"].notna()
            & self.prior_summary["scope_value"].eq(
                self.prior_summary["target_bucket"]
            )
        ].copy()
        bucket = (
            bucket.sort_values(
                ["target_bucket", "pr_auc_loss_mean"],
                ascending=[True, False],
            )
            .groupby("target_bucket", sort=True)
            .head(2)
        )
        bucket_experiments = [
            Experiment(
                str(row.experiment),
                "bucket",
                "bucket_mask",
                str(row.group),
                target_bucket=str(row.target_bucket),
            )
            for row in bucket.itertuples(index=False)
            if self.base.catalog.get(str(row.group))
        ]

        ticker = self.prior_summary.loc[
            self.prior_summary["stage"].eq("ticker")
            & self.prior_summary["scope_type"].eq("target_ticker")
            & self.prior_summary["target_ticker"].notna()
        ].copy()
        top = ticker.sort_values(
            "pr_auc_loss_mean", ascending=False, na_position="last"
        ).head(10)
        bottom = ticker.sort_values(
            "pr_auc_loss_mean", ascending=True, na_position="last"
        ).head(6)
        selected = pd.concat([top, bottom], ignore_index=True).drop_duplicates(
            "experiment"
        )
        ticker_experiments = [
            Experiment(
                str(row.experiment),
                "ticker",
                "ticker_mask",
                str(row.group),
                target_bucket=(
                    str(row.target_bucket)
                    if pd.notna(row.target_bucket)
                    else None
                ),
                target_ticker=_normalize_ticker(row.target_ticker),
            )
            for row in selected.itertuples(index=False)
            if self.base.catalog.get(str(row.group))
        ]
        return bucket_experiments, ticker_experiments

    def _record_metrics(
        self,
        metrics: pd.DataFrame,
        *,
        phase: str,
        method: str,
        config: ModelConfig,
        baseline_experiment: str,
    ) -> None:
        frame = metrics.copy()
        frame["phase"] = phase
        frame["ablation_method"] = method
        frame["model_config"] = config.name
        frame["baseline_experiment"] = baseline_experiment
        self.metric_frames.append(frame)

    def _run_retrain_phase(
        self,
        phase: str,
        experiments: list[tuple[Experiment, str]],
        fold_ids: Iterable[int],
        seeds: Iterable[int],
        config: ModelConfig,
    ) -> bool:
        for experiment, baseline_experiment in experiments:
            if (
                experiment.mode != "none"
                and not self.base.catalog.get(experiment.group or "")
            ):
                self.records.append(
                    {
                        **asdict(experiment),
                        "phase": phase,
                        "model_config": config.name,
                        "ablation_method": "retrain_drop",
                        "status": "skipped_no_features",
                    }
                )
                continue
            for fold_id in fold_ids:
                for seed in seeds:
                    if not self.can_start():
                        self._checkpoint(force=True)
                        return False
                    LOGGER.info(
                        "phase=%s experiment=%s fold=%s seed=%s remaining=%.2fh",
                        phase,
                        experiment.name,
                        fold_id,
                        seed,
                        self.remaining_seconds() / 3600,
                    )
                    metrics, result = self.base.execute(
                        experiment,
                        self.base.folds[int(fold_id)],
                        int(seed),
                        config,
                    )
                    self._record_metrics(
                        metrics,
                        phase=phase,
                        method="retrain_drop",
                        config=config,
                        baseline_experiment=baseline_experiment,
                    )
                    record = {
                        **asdict(result),
                        "phase": phase,
                        "model_config": config.name,
                        "ablation_method": "retrain_drop",
                        "baseline_experiment": baseline_experiment,
                        "status": "completed",
                    }
                    self.records.append(record)
                    self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
                    if result.cache_status == "miss":
                        self.new_task_seconds.append(result.elapsed_seconds)
                    self._checkpoint()
        self._checkpoint(force=True)
        return True

    def _permutation_cache(
        self,
        experiment: str,
        fold: int,
        seed: int,
        config: ModelConfig,
    ) -> Path:
        payload = {
            "experiment": experiment,
            "fold": fold,
            "seed": seed,
            "config": config.payload(),
            "dataset": self.base.dataset_signature,
            "method": "conditional_time_regime_permutation_v1",
        }
        # Keep the directory short for Windows installations whose project
        # root is already close to MAX_PATH.
        root = self.output / "pc_perm"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{stable_hash(payload)}.parquet"

    def _prediction_metrics(
        self,
        prediction: pd.DataFrame,
        experiment: Experiment,
        fold_id: int,
        seed: int,
    ) -> pd.DataFrame:
        row = {
            "experiment": experiment.name,
            **asdict(experiment),
            "fold": fold_id,
            "seed": seed,
            "scope_type": "all",
            "scope_value": "all_validation",
            **_metrics(
                prediction["target"].to_numpy(),
                prediction["prediction"].to_numpy(),
                prediction["date"].to_numpy(),
            ),
        }
        return pd.DataFrame([row])

    def _run_conditional_permutation(
        self,
        groups: list[str],
        config: ModelConfig,
    ) -> bool:
        phase = "conditional_permutation"
        baseline = Experiment(
            "perm__baseline",
            phase,
            "none",
        )
        experiments = [
            Experiment(f"perm__{group}", phase, "conditional_permutation", group)
            for group in groups
        ]
        regime_feature = (
            "u_market_kospi_ret_20"
            if "u_market_kospi_ret_20" in self.base.feature_index
            else next(
                (
                    feature
                    for feature in self.base.features
                    if feature.startswith("u_")
                ),
                self.base.features[0],
            )
        )
        regime_col = self.base.feature_index[regime_feature]
        for fold_id in range(4, 8):
            train_idx, val_idx = self.base.fold_indices[fold_id]
            for seed in METHOD_SEEDS:
                caches = {
                    exp.name: self._permutation_cache(
                        exp.name, fold_id, seed, config
                    )
                    for exp in [baseline, *experiments]
                }
                missing = [name for name, path in caches.items() if not path.exists()]
                model = None
                baseline_probability = None
                if missing:
                    if not self.can_start():
                        self._checkpoint(force=True)
                        return False
                    started = time.perf_counter()
                    model, backend = fit_xgb(
                        np.ascontiguousarray(self.base.X[train_idx]),
                        self.base.y[train_idx],
                        seed,
                        config,
                        threads=self.threads,
                        prefer_gpu=self.base.prefer_gpu,
                    )
                    elapsed = time.perf_counter() - started
                    self.new_task_seconds.append(elapsed)
                    baseline_probability = model.predict_proba(
                        np.ascontiguousarray(self.base.X[val_idx])
                    )[:, 1]
                else:
                    backend = "cache"

                for exp in [baseline, *experiments]:
                    if not self.can_start() and not caches[exp.name].exists():
                        self._checkpoint(force=True)
                        return False
                    started = time.perf_counter()
                    cache = caches[exp.name]
                    if cache.exists():
                        prediction = pd.read_parquet(cache)
                        cache_status = "hit"
                        task_backend = "cache"
                    else:
                        if model is None:
                            raise RuntimeError("permutation model was not fitted")
                        x_val = np.ascontiguousarray(self.base.X[val_idx])
                        if exp.name == baseline.name:
                            probability = baseline_probability
                        else:
                            columns = [
                                self.base.feature_index[column]
                                for column in self.base.catalog.get(
                                    exp.group or "", []
                                )
                                if column in self.base.feature_index
                            ]
                            derived_seed = int(
                                stable_hash(
                                    [exp.name, fold_id, seed, "conditional"]
                                )[:8],
                                16,
                            )
                            source_rows = _conditional_source_rows(
                                self.base.dates[val_idx],
                                self.base.tickers[val_idx],
                                self.base.X[val_idx, regime_col],
                                derived_seed,
                            )
                            if columns:
                                x_val[np.ix_(np.arange(len(x_val)), columns)] = (
                                    x_val[source_rows][:, columns]
                                )
                            probability = model.predict_proba(x_val)[:, 1]
                        prediction = self.base.df.loc[
                            val_idx, ["date", "ticker", "bucket"]
                        ].copy()
                        prediction["target"] = self.base.y[val_idx]
                        prediction["prediction"] = np.asarray(
                            probability, dtype=np.float32
                        )
                        atomic_parquet(prediction, cache)
                        cache_status = "miss"
                        task_backend = backend
                    elapsed = time.perf_counter() - started
                    metrics = self._prediction_metrics(
                        prediction,
                        exp,
                        fold_id,
                        seed,
                    )
                    self._record_metrics(
                        metrics,
                        phase=phase,
                        method="conditional_permutation",
                        config=config,
                        baseline_experiment=baseline.name,
                    )
                    self.records.append(
                        {
                            **asdict(exp),
                            "phase": phase,
                            "model_config": config.name,
                            "ablation_method": "conditional_permutation",
                            "baseline_experiment": baseline.name,
                            "fold": fold_id,
                            "seed": seed,
                            "feature_count": len(self.base.features),
                            "backend": task_backend,
                            "elapsed_seconds": elapsed,
                            "cache_status": cache_status,
                            "prediction_path": str(cache),
                            "status": "completed",
                        }
                    )
                    self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
                    self._checkpoint()
        self._checkpoint(force=True)
        return True

    def _pair_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame()
        work = metrics.copy()
        work["pair_scope_type"] = work["scope_type"].replace(
            {"target_bucket": "bucket", "target_ticker": "ticker"}
        )
        metric_columns = [
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "accuracy",
            "brier",
            "logloss",
            "top_3pct_precision",
            "top_3pct_recall",
        ]
        baseline = work.loc[
            work["experiment"].eq(work["baseline_experiment"])
        ].copy()
        baseline = baseline.rename(
            columns={
                "experiment": "baseline_row_experiment",
                **{column: f"baseline_{column}" for column in metric_columns},
            }
        )
        keys_left = [
            "baseline_experiment",
            "fold",
            "seed",
            "pair_scope_type",
            "scope_value",
        ]
        keys_right = [
            "baseline_row_experiment",
            "fold",
            "seed",
            "pair_scope_type",
            "scope_value",
        ]
        baseline = baseline[
            keys_right + [f"baseline_{column}" for column in metric_columns]
        ].drop_duplicates(keys_right)
        compared = work.loc[
            ~work["experiment"].eq(work["baseline_experiment"])
        ].merge(
            baseline,
            left_on=keys_left,
            right_on=keys_right,
            how="left",
            validate="many_to_one",
        )
        if compared["baseline_pr_auc"].isna().any():
            missing = compared.loc[
                compared["baseline_pr_auc"].isna(),
                [
                    "experiment",
                    "baseline_experiment",
                    "fold",
                    "seed",
                    "scope_type",
                    "scope_value",
                ],
            ].head(20)
            raise RuntimeError(
                "baseline pairing failed:\n" + missing.to_string(index=False)
            )
        compared["pr_auc_loss"] = (
            compared["baseline_pr_auc"] - compared["pr_auc"]
        )
        compared["roc_auc_loss"] = (
            compared["baseline_roc_auc"] - compared["roc_auc"]
        )
        compared["balanced_accuracy_loss"] = (
            compared["baseline_balanced_accuracy"]
            - compared["balanced_accuracy"]
        )
        compared["accuracy_loss"] = (
            compared["baseline_accuracy"] - compared["accuracy"]
        )
        compared["brier_increase"] = (
            compared["brier"] - compared["baseline_brier"]
        )
        compared["logloss_increase"] = (
            compared["logloss"] - compared["baseline_logloss"]
        )
        compared["top_3pct_precision_loss"] = (
            compared["baseline_top_3pct_precision"]
            - compared["top_3pct_precision"]
        )
        compared["top_3pct_recall_loss"] = (
            compared["baseline_top_3pct_recall"]
            - compared["top_3pct_recall"]
        )
        return compared

    def _seed_outputs(self, deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        block = deltas.loc[
            deltas["phase"].eq("seed_stability")
            & deltas["scope_type"].eq("all")
        ].copy()
        if block.empty:
            return pd.DataFrame(), pd.DataFrame()
        seed_effects = (
            block.groupby(["group", "seed"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                pr_auc_loss_mean=("pr_auc_loss", "mean"),
                pr_auc_loss_std=("pr_auc_loss", "std"),
                roc_auc_loss_mean=("roc_auc_loss", "mean"),
                positive_fold_ratio=("pr_auc_loss", lambda values: np.mean(values > 0)),
            )
            .sort_values(["group", "seed"])
        )
        full = seed_effects.groupby("group")["pr_auc_loss_mean"].mean()
        convergence_rows: list[dict] = []
        for count in range(2, len(PRIMARY_SEEDS) + 1):
            selected = PRIMARY_SEEDS[:count]
            prefix = (
                seed_effects.loc[seed_effects["seed"].isin(selected)]
                .groupby("group")["pr_auc_loss_mean"]
                .mean()
            )
            aligned = pd.concat(
                [prefix.rename("prefix"), full.rename("full")],
                axis=1,
            ).dropna()
            convergence_rows.append(
                {
                    "seed_count": count,
                    "seeds": ",".join(map(str, selected)),
                    "groups": len(aligned),
                    "rank_spearman_vs_all_seeds": (
                        float(aligned["prefix"].corr(aligned["full"], method="spearman"))
                        if len(aligned) >= 2
                        else np.nan
                    ),
                    "mean_absolute_effect_error": float(
                        np.mean(np.abs(aligned["prefix"] - aligned["full"]))
                    ),
                    "max_absolute_effect_error": float(
                        np.max(np.abs(aligned["prefix"] - aligned["full"]))
                    ),
                }
            )
        return seed_effects, pd.DataFrame(convergence_rows)

    def _method_outputs(self, deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_folds = [4, 5, 6, 7]
        common_seeds = METHOD_SEEDS
        block = deltas.loc[
            deltas["scope_type"].eq("all")
            & deltas["fold"].isin(common_folds)
            & deltas["seed"].isin(common_seeds)
            & deltas["phase"].isin(
                [
                    "seed_stability",
                    "model_config_sensitivity",
                    "conditional_permutation",
                ]
            )
        ].copy()
        if block.empty:
            return pd.DataFrame(), pd.DataFrame()
        block["method_label"] = np.where(
            block["ablation_method"].eq("conditional_permutation"),
            "conditional_permutation",
            "retrain_" + block["model_config"].astype(str),
        )
        effects = (
            block.groupby(["method_label", "group"], as_index=False)
            .agg(
                tasks=("pr_auc_loss", "size"),
                folds=("fold", "nunique"),
                seeds=("seed", "nunique"),
                pr_auc_loss_mean=("pr_auc_loss", "mean"),
                roc_auc_loss_mean=("roc_auc_loss", "mean"),
                positive_ratio=("pr_auc_loss", lambda values: np.mean(values > 0)),
            )
            .sort_values(["method_label", "pr_auc_loss_mean"], ascending=[True, False])
        )
        pivot = effects.pivot_table(
            index="group",
            columns="method_label",
            values="pr_auc_loss_mean",
            aggfunc="mean",
        )
        consistency = pivot.corr(method="spearman", min_periods=3)
        consistency.insert(0, "method_label", consistency.index)
        return effects, consistency.reset_index(drop=True)

    def _finalize(self) -> dict:
        metrics = (
            pd.concat(self.metric_frames, ignore_index=True, sort=False)
            if self.metric_frames
            else pd.DataFrame()
        )
        atomic_csv(metrics, self.output / "all_metrics.csv")
        atomic_csv(pd.DataFrame(self.records), self.output / "task_manifest.csv")
        deltas = self._pair_metrics(metrics)
        atomic_csv(deltas, self.output / "paired_ablation_deltas.csv")
        summary = _summarize_deltas(deltas)
        atomic_csv(summary, self.output / "ablation_summary.csv")

        seed_effects, convergence = self._seed_outputs(deltas)
        atomic_csv(seed_effects, self.output / "seed_group_effects.csv")
        atomic_csv(convergence, self.output / "seed_convergence.csv")
        method_effects, method_consistency = self._method_outputs(deltas)
        atomic_csv(method_effects, self.output / "method_group_effects.csv")
        atomic_csv(
            method_consistency,
            self.output / "method_rank_consistency.csv",
        )

        recommended_seed_count = len(PRIMARY_SEEDS)
        if not convergence.empty:
            eligible = convergence.loc[
                (convergence["rank_spearman_vs_all_seeds"] >= 0.90)
                & (convergence["mean_absolute_effect_error"] <= 0.003)
            ]
            if not eligible.empty:
                recommended_seed_count = int(eligible.iloc[0]["seed_count"])
        top_groups: list[dict] = []
        if not summary.empty:
            candidates = summary.loc[
                summary["scope_type"].eq("all")
                & summary["phase"].isin(
                    [
                        "seed_stability",
                        "correlation_cluster",
                        "correlated_group_pair",
                    ]
                )
            ].sort_values("pr_auc_loss_mean", ascending=False)
            top_groups = candidates[
                [
                    "experiment",
                    "phase",
                    "group",
                    "fold_count",
                    "seed_count",
                    "pr_auc_loss_mean",
                    "pr_auc_loss_positive_fold_ratio",
                    "pr_auc_loss_sign_p",
                ]
            ].head(30).to_dict("records")
        recommendations = {
            "seed_policy": {
                "do_not_pick_best_seed": True,
                "candidate_seed_order": PRIMARY_SEEDS,
                "recommended_seed_count": recommended_seed_count,
                "recommended_seeds": PRIMARY_SEEDS[:recommended_seed_count],
                "rule": "smallest prefix with Spearman>=0.90 and mean effect error<=0.003 versus all completed seeds",
            },
            "method_policy": {
                "primary": "retrained leave-group/cluster-out",
                "confirmation": "conditional time/regime permutation",
                "interpretation": "promote groups only when signs agree across retraining, seeds and permutation; do not select by the single best seed",
            },
            "top_completed_candidates": top_groups,
            "limitations": [
                "correlation clusters are based on a deterministic 24k-row Spearman sample",
                "bucket/ticker confirmations remain exploratory when completed folds or seeds are sparse",
                "multiple-comparison correction is not used for ranking; confirm finalists in a smaller preregistered run",
            ],
        }
        atomic_json(recommendations, self.output / "recommendations.json")

        result = {
            "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "budget_hours": self.hours,
            "elapsed_hours": (time.monotonic() - self.started) / 3600,
            "stopped_with_reserve": self.remaining_seconds() <= self.reserve_seconds + 300,
            "remaining_hours": self.remaining_seconds() / 3600,
            "temperature_abort": self.monitor.abort_event.is_set(),
            "completed_tasks": int(
                sum(row.get("status") == "completed" for row in self.records)
            ),
            "cache_hits": int(
                sum(row.get("cache_status") == "hit" for row in self.records)
            ),
            "new_predictions": int(
                sum(row.get("cache_status") == "miss" for row in self.records)
            ),
            "phase_counts": self.phase_counts,
            "metrics_rows": int(len(metrics)),
            "paired_delta_rows": int(len(deltas)),
            "summary_rows": int(len(summary)),
            "features": len(self.base.features),
            "correlation_clusters": int(
                self.correlation_clusters["cluster"].nunique()
                if not self.correlation_clusters.empty
                else 0
            ),
            "correlated_group_pairs": int(len(self.group_pairs)),
            "recommended_seed_count": recommended_seed_count,
            "recommended_seeds": PRIMARY_SEEDS[:recommended_seed_count],
            "output": str(self.output),
        }
        atomic_json(result, self.output / "run_summary.json")
        return result

    def run(self) -> dict:
        prevent_windows_sleep()
        self.monitor.start()
        try:
            self.prepare()
            config = next(item for item in CANDIDATES if item.name == "depth6_fast")
            seed_groups = self._seed_groups()
            cluster_names = (
                self.correlation_clusters[
                    ["cluster", "priority_score"]
                ]
                .drop_duplicates()
                .sort_values("priority_score", ascending=False)["cluster"]
                .head(36)
                .tolist()
            )
            pair_names = self.group_pairs["pair_group"].head(18).tolist()
            bucket_experiments, ticker_experiments = self._target_experiments()
            method_groups = seed_groups[:4]
            clustered_features = (
                self.correlation_clusters.sort_values(
                    ["priority_score", "mean_abs_correlation"],
                    ascending=False,
                )["feature"]
                .drop_duplicates()
                .head(120)
                .tolist()
            )
            for feature in clustered_features:
                self.base.catalog[f"singlecorr__{stable_hash(feature)}"] = [feature]

            self.plan = {
                "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                "hard_budget_hours": self.hours,
                "internal_finalize_reserve_minutes": self.reserve_seconds / 60,
                "cache_namespace": self.cache_namespace,
                "phases": [
                    {
                        "name": "seed_stability",
                        "groups": seed_groups,
                        "folds": list(range(8)),
                        "seeds": PRIMARY_SEEDS,
                    },
                    {
                        "name": "conditional_permutation",
                        "groups": seed_groups,
                        "folds": list(range(4, 8)),
                        "seeds": METHOD_SEEDS,
                    },
                    {
                        "name": "correlation_cluster",
                        "groups": cluster_names,
                        "folds": list(range(8)),
                        "seeds": METHOD_SEEDS,
                    },
                    {
                        "name": "correlated_group_pair",
                        "groups": pair_names,
                        "folds": list(range(2, 8)),
                        "seeds": METHOD_SEEDS,
                    },
                    {
                        "name": "model_config_sensitivity",
                        "groups": method_groups,
                        "folds": list(range(4, 8)),
                        "seeds": METHOD_SEEDS,
                        "configs": [item.name for item in CANDIDATES],
                    },
                    {
                        "name": "bucket_confirmation",
                        "experiments": [exp.name for exp in bucket_experiments],
                        "folds": list(range(8)),
                        "seeds": METHOD_SEEDS,
                    },
                    {
                        "name": "ticker_confirmation",
                        "experiments": [exp.name for exp in ticker_experiments],
                        "folds": list(range(8)),
                        "seeds": METHOD_SEEDS,
                    },
                    {
                        "name": "single_correlated_feature_extension",
                        "features": clustered_features,
                        "folds": list(range(2, 8)),
                        "seeds": METHOD_SEEDS,
                        "time_fill_only": True,
                    },
                ],
            }
            atomic_json(self.plan, self.output / "experiment_plan.json")

            baseline = Experiment("baseline", "baseline", "none")
            seed_experiments = [(baseline, baseline.name)] + [
                (self._prior_experiment(group), baseline.name)
                for group in seed_groups
            ]
            phases_ok = self._run_retrain_phase(
                "seed_stability",
                seed_experiments,
                range(8),
                PRIMARY_SEEDS,
                config,
            )

            if phases_ok and self.can_start():
                phases_ok = self._run_conditional_permutation(seed_groups, config)

            if phases_ok and self.can_start():
                phases_ok = self._run_retrain_phase(
                    "correlation_cluster",
                    [
                        (
                            Experiment(
                                f"cluster__{name}",
                                "correlation_cluster",
                                "global_drop",
                                name,
                            ),
                            baseline.name,
                        )
                        for name in cluster_names
                    ],
                    range(8),
                    METHOD_SEEDS,
                    config,
                )

            if phases_ok and self.can_start():
                phases_ok = self._run_retrain_phase(
                    "correlated_group_pair",
                    [
                        (
                            Experiment(
                                f"pair__{name}",
                                "correlated_group_pair",
                                "global_drop",
                                name,
                            ),
                            baseline.name,
                        )
                        for name in pair_names
                    ],
                    range(2, 8),
                    METHOD_SEEDS,
                    config,
                )

            if phases_ok and self.can_start():
                for candidate in CANDIDATES:
                    if candidate.name == config.name:
                        continue
                    config_baseline = Experiment(
                        f"config__{candidate.name}__baseline",
                        "model_config_sensitivity",
                        "none",
                    )
                    experiments = [(config_baseline, config_baseline.name)] + [
                        (
                            Experiment(
                                f"config__{candidate.name}__{group}",
                                "model_config_sensitivity",
                                "global_drop",
                                group,
                            ),
                            config_baseline.name,
                        )
                        for group in method_groups
                    ]
                    phases_ok = self._run_retrain_phase(
                        "model_config_sensitivity",
                        experiments,
                        range(4, 8),
                        METHOD_SEEDS,
                        candidate,
                    )
                    if not phases_ok:
                        break

            if phases_ok and self.can_start():
                phases_ok = self._run_retrain_phase(
                    "bucket_confirmation",
                    [(exp, baseline.name) for exp in bucket_experiments],
                    range(8),
                    METHOD_SEEDS,
                    config,
                )

            if phases_ok and self.can_start():
                phases_ok = self._run_retrain_phase(
                    "ticker_confirmation",
                    [(exp, baseline.name) for exp in ticker_experiments],
                    range(8),
                    METHOD_SEEDS,
                    config,
                )

            if phases_ok and self.can_start():
                self._run_retrain_phase(
                    "single_correlated_feature_extension",
                    [
                        (
                            Experiment(
                                f"singlecorr__{stable_hash(feature)}",
                                "single_correlated_feature_extension",
                                "global_drop",
                                f"singlecorr__{stable_hash(feature)}",
                            ),
                            baseline.name,
                        )
                        for feature in clustered_features
                    ],
                    range(2, 8),
                    METHOD_SEEDS,
                    config,
                )
            return self._finalize()
        finally:
            self._checkpoint(force=True)
            self.monitor.stop()
            restore_windows_sleep()
