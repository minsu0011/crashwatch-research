from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_paths, load_baskets


@dataclass
class Job:
    job_id: str
    stage: str
    modes: list[str]
    run_tag: str
    groups: list[str] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    optional: bool = False
    status: str = "pending"
    attempts: int = 0
    duration_seconds: float = 0.0
    return_code: int | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Job":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in value.items() if k in allowed})


def initial_jobs(project: Path, config: dict[str, Any]) -> list[Job]:
    paths = get_paths(project)
    baskets = load_baskets(paths)
    exp = config["experiment"]
    seeds = list(map(int, exp["seeds_primary"]))
    jobs = [
        Job(
            job_id="01_global_screen",
            stage="global_screen",
            modes=["universe", "ticker_global"],
            run_tag="01_global_screen",
            seeds=seeds,
        )
    ]
    for index, bucket in enumerate(sorted(baskets["bucket"].dropna().unique()), start=1):
        safe = str(bucket).replace(" ", "_")
        jobs.append(Job(
            job_id=f"02_bucket_{index:02d}_{safe}",
            stage="bucket_screen",
            modes=["bucket"],
            run_tag=f"02_bucket_{index:02d}_{safe}",
            buckets=[str(bucket)],
            seeds=seeds,
        ))
    return jobs


def _choose_groups(summary_path: Path, bucket: str, config: dict[str, Any]) -> list[str]:
    fallback = list(config["experiment"].get("fallback_ticker_groups", []))
    if not summary_path.exists():
        return fallback
    df = pd.read_csv(summary_path)
    if df.empty or "target_group" not in df.columns:
        return fallback
    mask = pd.Series(True, index=df.index)
    if "pair_scope_type" in df.columns:
        mask &= df["pair_scope_type"].astype(str).eq("bucket")
    elif "scope_type" in df.columns:
        mask &= df["scope_type"].astype(str).isin(["bucket", "target_bucket"])
    if "scope_value" in df.columns:
        mask &= df["scope_value"].astype(str).eq(bucket)
    block = df.loc[mask].copy()
    delta = "pr_auc_loss_when_removed_mean"
    if block.empty or delta not in block.columns:
        return fallback
    block[delta] = pd.to_numeric(block[delta], errors="coerce")
    block["abs_delta"] = block[delta].abs()
    if "pr_auc_fdr_q" in block.columns:
        block["pr_auc_fdr_q"] = pd.to_numeric(block["pr_auc_fdr_q"], errors="coerce")
    else:
        block["pr_auc_fdr_q"] = float("nan")
    min_abs = float(config["experiment"].get("ticker_min_abs_delta", 0.005))
    max_q = float(config["experiment"].get("ticker_max_fdr_q", 0.20))
    selected = block.loc[(block["abs_delta"] >= min_abs) | (block["pr_auc_fdr_q"] <= max_q)]
    selected = selected.sort_values(["abs_delta", delta], ascending=[False, False])
    min_groups = int(config["experiment"].get("ticker_groups_min", 4))
    max_groups = int(config["experiment"].get("ticker_groups_max", 6))
    if len(selected) < min_groups:
        selected = block.sort_values("abs_delta", ascending=False).head(min_groups)
    groups = selected["target_group"].dropna().astype(str).drop_duplicates().head(max_groups).tolist()
    for group in fallback:
        if len(groups) >= min_groups:
            break
        if group not in groups:
            groups.append(group)
    return groups[:max_groups]


def build_ticker_jobs(project: Path, config: dict[str, Any], ready_buckets: set[str] | None = None) -> list[Job]:
    paths = get_paths(project)
    baskets = load_baskets(paths)
    seeds = list(map(int, config["experiment"]["seeds_primary"]))
    jobs: list[Job] = []
    for index, (bucket, block) in enumerate(baskets.groupby("bucket", sort=True), start=1):
        if ready_buckets is not None and str(bucket) not in ready_buckets:
            continue
        safe = str(bucket).replace(" ", "_")
        bucket_tag = f"02_bucket_{index:02d}_{safe}"
        summary = paths.data_root / "ablation_longrun" / "runs" / bucket_tag / "ablation_statistical_summary.csv"
        groups = _choose_groups(summary, str(bucket), config)
        jobs.append(Job(
            job_id=f"03_ticker_{index:02d}_{safe}",
            stage="ticker_targeted",
            modes=["ticker"],
            run_tag=f"03_ticker_{index:02d}_{safe}",
            groups=groups,
            buckets=[str(bucket)],
            tickers=block["ticker"].astype(str).tolist(),
            seeds=seeds,
        ))
    return jobs


def build_robustness_jobs(project: Path, config: dict[str, Any], ticker_jobs: list[Job]) -> list[Job]:
    extra = list(map(int, config["experiment"].get("seeds_robustness", [])))
    if not extra:
        return []
    jobs = [Job(
        job_id="04_robust_global",
        stage="robustness",
        modes=["universe", "ticker_global"],
        run_tag="04_robust_global",
        seeds=extra,
        optional=True,
    )]
    for i, source in enumerate(ticker_jobs, start=1):
        jobs.append(Job(
            job_id=f"04_robust_{i:02d}_{source.job_id}",
            stage="robustness",
            modes=["bucket", "ticker"],
            run_tag=f"04_robust_{i:02d}_{source.run_tag}",
            groups=list(source.groups),
            buckets=list(source.buckets),
            tickers=list(source.tickers),
            seeds=extra,
            optional=True,
        ))
    return jobs


def save_jobs(path: Path, jobs: list[Job], metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "jobs": [job.to_dict() for job in jobs]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_jobs(path: Path) -> tuple[list[Job], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Job.from_dict(x) for x in payload.get("jobs", [])], payload.get("metadata", {})
