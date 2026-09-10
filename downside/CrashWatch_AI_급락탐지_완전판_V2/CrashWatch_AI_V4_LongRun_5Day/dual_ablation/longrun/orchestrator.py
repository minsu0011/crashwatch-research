from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_paths
from ..experiment.splits import make_walk_forward_folds
from .hardware import (
    append_hardware_log,
    apply_power_ratio,
    prevent_windows_sleep,
    restore_power_limit,
    set_process_limits,
    snapshot,
    terminate_process_tree,
)
from .planner import Job, build_robustness_jobs, build_ticker_jobs, initial_jobs, load_jobs, save_jobs

LOGGER = logging.getLogger("crashwatch.longrun")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_dataset(project: Path, configured: str | None) -> Path:
    if configured and configured not in {"auto", ""}:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (project / path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"설정한 dataset이 없습니다: {path}")
    local = project / "crashwatch_ai_data" / "development" / "training_dataset_dual.parquet"
    if local.exists():
        return local
    candidates = list(project.parent.glob("*/crashwatch_ai_data/development/training_dataset_dual.parquet"))
    candidates += list(project.parent.glob("*/crashwatch_ai_data/development/training_dataset.parquet"))
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        candidates += list(downloads.glob("CrashWatch*/**/crashwatch_ai_data/development/training_dataset_dual.parquet"))
        candidates += list(downloads.glob("CrashWatch*/**/crashwatch_ai_data/development/training_dataset.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "training_dataset_dual.parquet를 찾지 못했습니다. longrun_5day.json의 dataset에 실제 경로를 입력하세요."
        )
    return max(candidates, key=lambda p: (p.stat().st_size, p.stat().st_mtime_ns))


def _data_root_from_dataset(dataset: Path) -> Path:
    # .../crashwatch_ai_data/development/file.parquet
    if dataset.parent.name == "development":
        return dataset.parent.parent
    return dataset.parent


def _resolve_data_root(project: Path, configured: str | None, dataset: Path) -> Path:
    if not configured or configured in {"auto", ""}:
        return _data_root_from_dataset(dataset)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (project / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _worker_command(project: Path, dataset: Path, job: Job, config: dict[str, Any]) -> list[str]:
    exp = config["experiment"]
    command = [
        sys.executable,
        str(project / "03F_장기이탈테스트_워커.py"),
        "--dataset", str(dataset),
        "--run-tag", job.run_tag,
        "--modes", ",".join(job.modes),
        "--seeds", ",".join(map(str, job.seeds)),
        "--folds", str(exp["folds"]),
        "--validation-days", str(exp["validation_days"]),
        "--purge-days", str(exp["purge_days"]),
        "--min-train-days", str(exp["min_train_days"]),
        "--max-train-rows", str(exp.get("max_train_rows", 300000)),
        "--calibration", str(exp.get("calibration", "sigmoid")),
        "--calibration-days", str(exp.get("calibration_days", 60)),
    ]
    if job.groups:
        command += ["--groups", ",".join(job.groups)]
    if job.buckets:
        command += ["--buckets", ",".join(job.buckets)]
    if job.tickers:
        command += ["--tickers", ",".join(job.tickers)]
    if not bool(exp.get("prefer_gpu", True)):
        command.append("--cpu")
    return command


def _setup_env(data_root: Path, config: dict[str, Any]) -> dict[str, str]:
    safety = config["safety"]
    model = config["model"]
    env = os.environ.copy()
    env["CRASHWATCH_DATA_DIR"] = str(data_root)
    threads = str(int(safety["cpu_logical_cores"]))
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["NUMEXPR_NUM_THREADS"] = threads
    env["CRASHWATCH_XGB_N_JOBS"] = threads
    env["CRASHWATCH_XGB_N_ESTIMATORS"] = str(model["n_estimators"])
    env["CRASHWATCH_XGB_MAX_DEPTH"] = str(model["max_depth"])
    env["CRASHWATCH_XGB_LEARNING_RATE"] = str(model["learning_rate"])
    env["CRASHWATCH_XGB_MAX_BIN"] = str(model.get("max_bin", 256))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _resource_gate(project: Path, config: dict[str, Any], job_id: str, log_path: Path) -> None:
    safety = config["safety"]
    while True:
        snap = snapshot(project)
        append_hardware_log(log_path, snap, job_id)
        temp = snap.gpu_temperature_c
        ready = (
            snap.ram_available_gb >= float(safety["min_free_ram_gb"])
            and snap.disk_free_gb >= float(safety["min_disk_free_gb"])
            and (temp is None or temp <= float(safety["gpu_resume_temp_c"]))
        )
        if ready:
            return
        LOGGER.warning(
            "자원 대기: GPU=%s°C, free RAM=%.1fGB, free disk=%.1fGB",
            temp, snap.ram_available_gb, snap.disk_free_gb,
        )
        time.sleep(int(safety["cooldown_seconds"]))


def _run_job(
    project: Path,
    dataset: Path,
    data_root: Path,
    job: Job,
    config: dict[str, Any],
    root: Path,
    deadline_monotonic: float,
) -> tuple[int, str]:
    safety = config["safety"]
    hardware_log = root / "hardware_log.csv"
    _resource_gate(project, config, job.job_id, hardware_log)
    command = _worker_command(project, dataset, job, config)
    env = _setup_env(data_root, config)
    logs = root / "job_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / f"{job.job_id}.log"
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    with log_file.open("a", encoding="utf-8", buffering=1) as out:
        out.write("\n=== COMMAND ===\n" + subprocess.list2cmdline(command) + "\n")
        proc = subprocess.Popen(
            command, cwd=project, env=env, stdout=out, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        limits = set_process_limits(
            proc.pid, int(safety["cpu_logical_cores"]), str(safety.get("process_priority", "below_normal"))
        )
        out.write("PROCESS_LIMITS=" + json.dumps(limits, ensure_ascii=False) + "\n")
        warning_count = 0
        start = time.monotonic()
        reason = ""
        while proc.poll() is None:
            snap = snapshot(project, proc.pid)
            append_hardware_log(hardware_log, snap, job.job_id)
            temp = snap.gpu_temperature_c
            if temp is not None and temp >= float(safety["gpu_hard_temp_c"]):
                reason = f"GPU hard temperature {temp}C"
            elif temp is not None and temp >= float(safety["gpu_abort_temp_c"]):
                warning_count += 1
                if warning_count >= int(safety.get("gpu_abort_consecutive_checks", 3)):
                    reason = f"GPU sustained high temperature {temp}C"
            else:
                warning_count = 0
            if snap.ram_available_gb < float(safety["hard_min_free_ram_gb"]):
                reason = f"system RAM low {snap.ram_available_gb:.2f}GB"
            if snap.process_rss_gb > float(safety["max_process_ram_gb"]):
                reason = f"worker RAM high {snap.process_rss_gb:.2f}GB"
            if snap.disk_free_gb < float(safety["hard_min_disk_free_gb"]):
                reason = f"disk free low {snap.disk_free_gb:.2f}GB"
            if time.monotonic() - start > float(safety["max_single_job_hours"]) * 3600:
                reason = "single job runtime limit"
            if time.monotonic() >= deadline_monotonic:
                reason = "global runtime budget reached"
            stop_file = root / "STOP_LONGRUN.txt"
            if stop_file.exists():
                reason = "user stop file"
            if reason:
                out.write(f"\nSAFETY_STOP={reason}\n")
                terminate_process_tree(proc.pid)
                break
            time.sleep(int(safety["poll_seconds"]))
        return_code = proc.poll()
        if return_code is None:
            return_code = -9
        return int(return_code), reason


def _merge_summaries(data_root: Path, root: Path) -> None:
    runs = data_root / "ablation_longrun" / "runs"
    summary_frames = []
    run_frames = []
    for path in sorted(runs.glob("*/ablation_statistical_summary.csv")):
        try:
            df = pd.read_csv(path)
            df.insert(0, "run_tag", path.parent.name)
            summary_frames.append(df)
        except Exception:  # noqa: BLE001
            continue
    for path in sorted(runs.glob("*/run_summary.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["run_tag"] = path.parent.name
            run_frames.append(row)
        except Exception:  # noqa: BLE001
            continue
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True, sort=False).to_csv(
            root / "combined_ablation_summary.csv", index=False, encoding="utf-8-sig"
        )
    if run_frames:
        pd.DataFrame(run_frames).to_csv(root / "combined_run_summary.csv", index=False, encoding="utf-8-sig")


def _create_compact_package(project: Path, data_root: Path, root: Path) -> Path | None:
    """Create a compact result package without prediction parquet files."""
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = project
    stamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    output = desktop / f"CrashWatch_V4_LongRun_RESULTS_{stamp}_COMPACT.zip"
    candidates: list[tuple[Path, str]] = []
    for path in root.glob("*.json"):
        candidates.append((path, f"longrun/{path.name}"))
    for path in root.glob("*.csv"):
        candidates.append((path, f"longrun/{path.name}"))
    runs = root / "runs"
    allowed_run_names = {
        "run_summary.json", "walk_forward_folds.json", "ablation_statistical_summary.csv",
        "skipped_experiments.csv", "skipped_groups.csv", "valid_feature_audit.csv",
        "bucket_group_sensitivity_matrix.csv", "ticker_group_sensitivity_matrix.csv",
        "grouped_conditional_permutation.csv", "grouped_unconditional_permutation.csv",
    }
    if runs.exists():
        for path in runs.glob("*/*"):
            if path.is_file() and path.name in allowed_run_names:
                candidates.append((path, f"runs/{path.parent.name}/{path.name}"))
    feature_dir = data_root / "features" / "dual_ablation"
    for name in [
        "feature_catalog_universe.json", "feature_catalog_ticker.json",
        "feature_quality.csv", "feature_run_summary.json",
    ]:
        path = feature_dir / name
        if path.exists():
            candidates.append((path, f"features/{name}"))
    for name in ["longrun_5day.json", "research_feature_groups.csv", "feature_group_prefixes.json"]:
        path = project / "configs" / name
        if path.exists():
            candidates.append((path, f"configs/{name}"))

    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            seen: set[str] = set()
            for path, arcname in candidates:
                if arcname in seen or not path.exists():
                    continue
                # 개별 상세 파일은 12MB를 넘으면 요약 패키지에서 제외한다.
                if path.stat().st_size > 12 * 1024 * 1024:
                    continue
                archive.write(path, arcname)
                seen.add(arcname)
            log_dir = root / "job_logs"
            if log_dir.exists():
                for log in sorted(log_dir.glob("*.log")):
                    try:
                        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-250:]
                        archive.writestr(f"log_tails/{log.stem}_tail.txt", "\n".join(lines))
                    except OSError:
                        continue
        return output
    except OSError as exc:
        LOGGER.warning("compact 결과 압축 실패: %s", exc)
        return None



def _preflight(project: Path, dataset: Path, data_root: Path, config: dict[str, Any], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    feature_dir = data_root / "features" / "dual_ablation"
    u_path = feature_dir / "feature_catalog_universe.json"
    t_path = feature_dir / "feature_catalog_ticker.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not u_path.exists() or not t_path.exists():
        errors.append("V4 피처 카탈로그가 없습니다. 01C와 02B를 먼저 실행하세요.")
        u_catalog: dict[str, list[str]] = {}
        t_catalog: dict[str, list[str]] = {}
    else:
        u_catalog = json.loads(u_path.read_text(encoding="utf-8"))
        t_catalog = json.loads(t_path.read_text(encoding="utf-8"))

    required = ["date", "ticker", str(config["experiment"].get("target", "label_abs_crash_20"))]
    # 스키마를 먼저 읽고 필요한 열만 로드해 사전점검 메모리를 제한한다.
    try:
        import pyarrow.parquet as pq
        schema_columns = set(pq.ParquetFile(dataset).schema.names)
    except Exception:  # noqa: BLE001
        schema_columns = set(pd.read_parquet(dataset).columns)
    missing_required = [c for c in required if c not in schema_columns]
    if missing_required:
        errors.append(f"필수 열 누락: {missing_required}")
    cols_to_read = [c for c in ["date", "ticker", required[-1], "sealed_do_not_train_or_tune"] if c in schema_columns]
    probe = pd.read_parquet(dataset, columns=cols_to_read)
    if "sealed_do_not_train_or_tune" in probe and pd.to_numeric(probe["sealed_do_not_train_or_tune"], errors="coerce").fillna(0).ne(0).any():
        errors.append("development dataset에 sealed 행이 포함되어 있습니다.")
    if {"date", "ticker"}.issubset(probe.columns):
        duplicate_count = int(probe.duplicated(["date", "ticker"]).sum())
        if duplicate_count:
            errors.append(f"중복 ticker-date: {duplicate_count}")
    else:
        duplicate_count = -1
    if "date" in probe:
        try:
            folds = make_walk_forward_folds(
                probe["date"],
                int(config["experiment"]["folds"]),
                int(config["experiment"]["validation_days"]),
                int(config["experiment"]["purge_days"]),
                int(config["experiment"]["min_train_days"]),
            )
            fold_count = len(folds)
        except Exception as exc:  # noqa: BLE001
            fold_count = 0
            errors.append(f"워크포워드 생성 실패: {exc}")
    else:
        fold_count = 0

    v4_groups = {
        "u_market_microstructure", "u_tail_network", "u_credit_funding", "u_etf_pressure",
        "u_derivatives_risk", "u_attention", "t_range_volatility", "t_microstructure_proxy",
        "t_tail_dependence", "t_volume_price_pressure", "t_limit_stress", "t_attention",
        "t_ownership_governance", "t_fundamental_quality", "t_network_contagion",
    }
    catalog = {**u_catalog, **t_catalog}
    present_v4_groups = sorted(g for g in v4_groups if g in catalog and any(c in schema_columns for c in catalog[g]))
    if not present_v4_groups:
        errors.append("dataset에 V4 연구 피처가 없습니다. V4 크롤링/피처 생성 후 training_dataset_dual.parquet를 사용하세요.")
    elif len(present_v4_groups) < 5:
        warnings.append(f"유효 V4 연구 그룹이 적습니다: {present_v4_groups}")

    base_catalog_path = data_root / "meta" / "feature_catalog.json"
    base_catalog_features = 0
    if base_catalog_path.exists():
        try:
            base_catalog = json.loads(base_catalog_path.read_text(encoding="utf-8"))
            base_columns = set(
                item for values in base_catalog.values() if isinstance(values, list) for item in values
            ) if isinstance(base_catalog, dict) else set(base_catalog)
            base_catalog_features = len(base_columns & schema_columns)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"기존 봉인 피처 카탈로그 파싱 실패: {exc}")
    if bool(config["experiment"].get("require_base_catalog", False)) and base_catalog_features == 0:
        errors.append("기존 봉인 V2 피처 카탈로그가 없거나 dataset과 겹치는 피처가 없습니다.")

    snap = snapshot(project)
    if snap.disk_free_gb < float(config["safety"]["min_disk_free_gb"]):
        errors.append(f"디스크 여유 부족: {snap.disk_free_gb:.1f}GB")
    if snap.ram_available_gb < float(config["safety"]["min_free_ram_gb"]):
        warnings.append(f"현재 사용 가능 RAM이 낮습니다: {snap.ram_available_gb:.1f}GB")

    report = {
        "dataset": str(dataset),
        "dataset_size_gb": round(dataset.stat().st_size / (1024 ** 3), 3),
        "rows": int(len(probe)),
        "tickers": int(probe["ticker"].astype(str).nunique()) if "ticker" in probe else 0,
        "date_min": str(pd.to_datetime(probe["date"], errors="coerce").min()) if "date" in probe else None,
        "date_max": str(pd.to_datetime(probe["date"], errors="coerce").max()) if "date" in probe else None,
        "duplicate_ticker_date": duplicate_count,
        "fold_count": fold_count,
        "universe_group_count": len(u_catalog),
        "ticker_group_count": len(t_catalog),
        "base_catalog_features_present": base_catalog_features,
        "present_v4_groups": present_v4_groups,
        "hardware": asdict(snap),
        "warnings": warnings,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    (root / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("장기실험 사전점검 실패: " + " | ".join(errors))
    return report

def run_longrun(project: Path, config_path: Path, *, reset: bool = False) -> dict[str, Any]:
    project = project.resolve()
    config = load_config(config_path)
    dataset = _resolve_dataset(project, config.get("dataset"))
    data_root = _resolve_data_root(project, config.get("data_root"), dataset)
    os.environ["CRASHWATCH_DATA_DIR"] = str(data_root)
    paths = get_paths(project)
    root = paths.data_root / "ablation_longrun"
    root.mkdir(parents=True, exist_ok=True)
    _preflight(project, dataset, data_root, config, root)
    stop_file = root / "STOP_LONGRUN.txt"
    if reset and stop_file.exists():
        stop_file.unlink()
    elif stop_file.exists():
        raise RuntimeError(f"중지 요청 파일이 남아 있습니다. 삭제하거나 --reset으로 시작하세요: {stop_file}")
    state_path = root / "longrun_state.json"
    if reset and state_path.exists():
        state_path.unlink()
    if state_path.exists():
        jobs, metadata = load_jobs(state_path)
        saved_dataset = metadata.get("dataset")
        if saved_dataset and Path(saved_dataset).resolve() != dataset.resolve():
            raise RuntimeError(
                f"기존 상태의 dataset이 현재 설정과 다릅니다. --reset 필요: {saved_dataset} != {dataset}"
            )
        for job in jobs:
            if job.status == "running":
                job.status = "pending"
    else:
        jobs = initial_jobs(project, config)
        metadata = {
            "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "dataset": str(dataset),
            "config": str(config_path),
        }
        save_jobs(state_path, jobs, metadata)

    runtime_hours = float(config["runtime_hours"])
    metadata.setdefault("started_at_epoch", time.time())
    metadata["expected_end_at"] = pd.Timestamp.fromtimestamp(
        float(metadata["started_at_epoch"]) + runtime_hours * 3600,
        tz="Asia/Seoul",
    ).isoformat()
    elapsed_before_start = max(0.0, time.time() - float(metadata["started_at_epoch"]))
    remaining_budget_seconds = max(0.0, runtime_hours * 3600 - elapsed_before_start)
    deadline = time.monotonic() + remaining_budget_seconds
    safety = config["safety"]
    power_record = apply_power_ratio(float(safety["gpu_power_ratio"]))
    (root / "gpu_power_limit.json").write_text(json.dumps(power_record, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Dataset: %s", dataset)
    LOGGER.info("Runtime budget: %.1f hours (remaining %.1f hours)", runtime_hours, remaining_budget_seconds / 3600)
    LOGGER.info("GPU power setting: %s", power_record)
    save_jobs(state_path, jobs, metadata)

    ticker_jobs_added = any(job.stage == "ticker_targeted" for job in jobs)
    robustness_added = any(job.stage == "robustness" for job in jobs)
    completed = 0
    try:
        with prevent_windows_sleep():
            while time.monotonic() < deadline:
                # 버킷 선별이 끝나면 종목별 표적 실험을 동적으로 추가한다.
                bucket_jobs = [j for j in jobs if j.stage == "bucket_screen"]
                bucket_terminal = bucket_jobs and all(j.status in {"success", "failed", "skipped_budget"} for j in bucket_jobs)
                if bucket_terminal and not ticker_jobs_added:
                    ready_buckets = {j.buckets[0] for j in bucket_jobs if j.status == "success" and j.buckets}
                    new_jobs = build_ticker_jobs(project, config, ready_buckets=ready_buckets)
                    jobs.extend(new_jobs)
                    ticker_jobs_added = True
                    save_jobs(state_path, jobs, metadata)
                ticker_jobs = [j for j in jobs if j.stage == "ticker_targeted"]
                ticker_terminal = ticker_jobs and all(
                    j.status in {"success", "failed", "skipped_budget"} for j in ticker_jobs
                )
                if ticker_terminal and not robustness_added:
                    successful_ticker_jobs = [j for j in ticker_jobs if j.status == "success"]
                    jobs.extend(build_robustness_jobs(project, config, successful_ticker_jobs))
                    robustness_added = True
                    save_jobs(state_path, jobs, metadata)

                pending = [j for j in jobs if j.status in {"pending", "retry"}]
                if not pending:
                    break
                job = pending[0]
                remaining_hours = (deadline - time.monotonic()) / 3600
                if job.optional and remaining_hours < float(config.get("optional_stage_min_remaining_hours", 8)):
                    job.status = "skipped_budget"
                    save_jobs(state_path, jobs, metadata)
                    continue
                job.status = "running"
                job.attempts += 1
                started = time.monotonic()
                save_jobs(state_path, jobs, metadata)
                LOGGER.info("JOB START %s (%s), remaining %.1fh", job.job_id, job.stage, remaining_hours)
                code, reason = _run_job(project, dataset, data_root, job, config, root, deadline)
                job.duration_seconds += time.monotonic() - started
                job.return_code = code
                if code == 0:
                    job.status = "success"
                    job.error = ""
                    completed += 1
                    LOGGER.info("JOB SUCCESS %s", job.job_id)
                else:
                    job.error = reason or f"return code {code}"
                    if reason == "user stop file":
                        job.status = "pending"
                        save_jobs(state_path, jobs, metadata)
                        break
                    if job.attempts < int(safety["max_job_retries"]):
                        job.status = "retry"
                        LOGGER.warning("JOB RETRY %s: %s", job.job_id, job.error)
                        time.sleep(int(safety["cooldown_seconds"]))
                    else:
                        job.status = "failed"
                        LOGGER.error("JOB FAILED %s: %s", job.job_id, job.error)
                metadata["updated_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
                metadata["completed_jobs"] = sum(j.status == "success" for j in jobs)
                metadata["failed_jobs"] = sum(j.status == "failed" for j in jobs)
                save_jobs(state_path, jobs, metadata)
                _merge_summaries(data_root, root)
                # 고온이 아니더라도 작업 사이에 짧은 냉각 구간을 둔다.
                time.sleep(int(safety.get("between_job_cooldown_seconds", 30)))
    finally:
        restore_ok, restore_message = restore_power_limit(power_record)
        metadata["last_exit_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        metadata["power_restore_ok"] = restore_ok
        metadata["power_restore_message"] = restore_message
        save_jobs(state_path, jobs, metadata)
        _merge_summaries(data_root, root)

    result = {
        "dataset": str(dataset),
        "state": str(state_path),
        "completed_jobs": sum(j.status == "success" for j in jobs),
        "failed_jobs": sum(j.status == "failed" for j in jobs),
        "pending_jobs": sum(j.status in {"pending", "retry", "running"} for j in jobs),
        "skipped_budget": sum(j.status == "skipped_budget" for j in jobs),
        "runtime_budget_hours": runtime_hours,
        "result_root": str(root),
        "expected_end_at": metadata.get("expected_end_at"),
    }
    terminal = result["pending_jobs"] == 0 or time.time() >= float(metadata["started_at_epoch"]) + runtime_hours * 3600
    if terminal:
        metadata["finished_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        package = _create_compact_package(project, data_root, root)
        result["compact_package"] = str(package) if package else None
        save_jobs(state_path, jobs, metadata)
    (root / "longrun_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
