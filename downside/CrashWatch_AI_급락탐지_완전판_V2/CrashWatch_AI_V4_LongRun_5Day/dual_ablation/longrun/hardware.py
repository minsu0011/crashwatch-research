from __future__ import annotations

import contextlib
import csv
import ctypes
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import psutil


@dataclass
class HardwareSnapshot:
    timestamp: str
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    process_rss_gb: float
    disk_free_gb: float
    gpu_temperature_c: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_power_w: float | None = None
    gpu_power_limit_w: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None


def _float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _run_nvidia_smi(args: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.run(
            [exe, *args], capture_output=True, text=True, timeout=timeout, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def query_gpu() -> dict[str, float | None]:
    query = (
        "temperature.gpu,utilization.gpu,power.draw,power.limit,"
        "memory.used,memory.total"
    )
    proc = _run_nvidia_smi([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return {}
    fields = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
    if len(fields) < 6:
        return {}
    return {
        "gpu_temperature_c": _float(fields[0]),
        "gpu_utilization_percent": _float(fields[1]),
        "gpu_power_w": _float(fields[2]),
        "gpu_power_limit_w": _float(fields[3]),
        "gpu_memory_used_mb": _float(fields[4]),
        "gpu_memory_total_mb": _float(fields[5]),
    }


def query_power_limits() -> dict[str, float | None]:
    query = "power.limit,power.default_limit,power.min_limit,power.max_limit"
    proc = _run_nvidia_smi([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return {}
    values = [_float(x) for x in proc.stdout.splitlines()[0].split(",")]
    if len(values) < 4:
        return {}
    return {
        "current": values[0], "default": values[1], "minimum": values[2], "maximum": values[3]
    }


def set_power_limit_w(watts: float) -> tuple[bool, str]:
    proc = _run_nvidia_smi(["-pl", str(int(round(watts)))], timeout=20)
    if proc is None:
        return False, "nvidia-smi를 찾을 수 없음"
    message = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, message


def apply_power_ratio(ratio: float) -> dict[str, Any]:
    limits = query_power_limits()
    current = limits.get("current")
    default = limits.get("default") or current
    minimum = limits.get("minimum")
    maximum = limits.get("maximum")
    result: dict[str, Any] = {"before": limits, "applied": False, "message": ""}
    if default is None:
        result["message"] = "GPU 전력 한도 조회 실패"
        return result
    target = default * max(0.5, min(1.0, ratio))
    if minimum is not None:
        target = max(target, minimum)
    if maximum is not None:
        target = min(target, maximum)
    ok, message = set_power_limit_w(target)
    result.update({"applied": ok, "target_w": round(target, 1), "message": message})
    return result


def restore_power_limit(record: dict[str, Any]) -> tuple[bool, str]:
    before = record.get("before") or {}
    target = before.get("current") or before.get("default")
    if not record.get("applied") or target is None:
        return False, "복원할 전력 한도 없음"
    return set_power_limit_w(float(target))


def process_rss_gb(pid: int | None) -> float:
    if pid is None:
        return 0.0
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(p.memory_info().rss for p in processes if p.is_running()) / (1024 ** 3)
    except (psutil.Error, OSError):
        return 0.0


def snapshot(work_dir: Path, pid: int | None = None) -> HardwareSnapshot:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(work_dir.resolve().anchor or work_dir.resolve()))
    gpu = query_gpu()
    return HardwareSnapshot(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        cpu_percent=float(psutil.cpu_percent(interval=0.2)),
        ram_percent=float(vm.percent),
        ram_available_gb=float(vm.available / (1024 ** 3)),
        process_rss_gb=float(process_rss_gb(pid)),
        disk_free_gb=float(disk.free / (1024 ** 3)),
        **gpu,
    )


def append_hardware_log(path: Path, snap: HardwareSnapshot, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"job_id": job_id, **asdict(snap)}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_process_limits(pid: int, logical_cores: int, priority: str = "below_normal") -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid, "affinity": False, "priority": False}
    try:
        proc = psutil.Process(pid)
        available = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else list(range(psutil.cpu_count() or 1))
        use_count = max(1, min(logical_cores, len(available)))
        # 첫 코어만 몰아 쓰지 않도록 전체 논리 코어에서 균등 간격으로 선택한다.
        if use_count < len(available):
            step = len(available) / use_count
            selected = sorted({available[min(len(available) - 1, int(i * step))] for i in range(use_count)})
        else:
            selected = available
        if hasattr(proc, "cpu_affinity"):
            proc.cpu_affinity(selected)
            result["affinity"] = selected
        if os.name == "nt":
            mapping = {
                "idle": psutil.IDLE_PRIORITY_CLASS,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
            }
            proc.nice(mapping.get(priority, psutil.BELOW_NORMAL_PRIORITY_CLASS))
        else:
            proc.nice(5 if priority == "below_normal" else 10 if priority == "idle" else 0)
        result["priority"] = priority
    except (psutil.Error, OSError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def terminate_process_tree(pid: int, grace_seconds: int = 20) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    children = root.children(recursive=True)
    for proc in [*children, root]:
        with contextlib.suppress(psutil.Error):
            proc.terminate()
    _, alive = psutil.wait_procs([*children, root], timeout=grace_seconds)
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()


@contextlib.contextmanager
def prevent_windows_sleep() -> Iterator[None]:
    """화면은 꺼질 수 있지만 Windows 시스템 절전은 막는다."""
    if os.name != "nt":
        yield
        return
    kernel32 = ctypes.windll.kernel32
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    es_awaymode_required = 0x00000040
    kernel32.SetThreadExecutionState(es_continuous | es_system_required | es_awaymode_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)
