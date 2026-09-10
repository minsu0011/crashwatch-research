from __future__ import annotations

import ctypes
import csv
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

LOGGER = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    process_rss_gb: float
    gpu_util_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_temperature_c: float | None
    gpu_power_w: float | None
    gpu_monitor_error: str


def prevent_windows_sleep() -> None:
    if os.name != "nt":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)


def restore_windows_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def query_gpu() -> dict[str, float | str | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8)
        line = result.stdout.strip().splitlines()[0]
        values = [x.strip() for x in line.split(",")]
        parsed = [float(x) if x not in {"N/A", "[Not Supported]"} else None for x in values]
        return {
            "gpu_util_percent": parsed[0], "gpu_memory_used_mb": parsed[1],
            "gpu_memory_total_mb": parsed[2], "gpu_temperature_c": parsed[3], "gpu_power_w": parsed[4],
            "gpu_monitor_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **{k: None for k in ["gpu_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_temperature_c", "gpu_power_w"]},
            "gpu_monitor_error": f"{type(exc).__name__}: {exc}",
        }


def snapshot() -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    gpu = query_gpu()
    return ResourceSnapshot(
        timestamp=time.time(), cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=float(vm.percent), ram_available_gb=float(vm.available / 1024**3),
        process_rss_gb=float(proc.memory_info().rss / 1024**3), **gpu,
    )


class ResourceMonitor:
    def __init__(self, output: Path, *, interval_seconds: float = 10.0, hard_gpu_temp: float = 88.0, min_available_ram_gb: float = 1.5):
        self.output = output
        self.interval_seconds = interval_seconds
        self.hard_gpu_temp = hard_gpu_temp
        self.min_available_ram_gb = min_available_ram_gb
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_snapshot: ResourceSnapshot | None = None
        self.last_error = ""
        self.abort_reason = ""
        self.gpu_monitor_warning = ""

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=15)

    def _run(self) -> None:
        fields = list(ResourceSnapshot.__dataclass_fields__)
        new_file = not self.output.exists()
        with self.output.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            while not self.stop_event.is_set():
                try:
                    snap = snapshot()
                except Exception as exc:  # noqa: BLE001
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception("리소스 감시 실패")
                    self.stop_event.wait(self.interval_seconds)
                    continue
                self.last_snapshot = snap
                if snap.gpu_monitor_error:
                    self.gpu_monitor_warning = snap.gpu_monitor_error
                    LOGGER.warning("GPU 온도 감시 불가: %s", snap.gpu_monitor_error)
                writer.writerow(snap.__dict__)
                handle.flush()
                if snap.gpu_temperature_c is not None and snap.gpu_temperature_c >= self.hard_gpu_temp:
                    self.abort_reason = f"gpu_temperature_c={snap.gpu_temperature_c:.1f} >= {self.hard_gpu_temp:.1f}"
                    self.abort_event.set()
                if snap.ram_available_gb < self.min_available_ram_gb:
                    self.abort_reason = f"ram_available_gb={snap.ram_available_gb:.3f} < {self.min_available_ram_gb:.3f}"
                    self.abort_event.set()
                self.stop_event.wait(self.interval_seconds)

    def status(self) -> dict:
        return {
            "abort": self.abort_event.is_set(),
            "abort_reason": self.abort_reason,
            "monitor_error": self.last_error,
            "gpu_monitor_warning": self.gpu_monitor_warning,
            "last_snapshot": self.last_snapshot.__dict__ if self.last_snapshot else None,
        }
