from __future__ import annotations

import ctypes
import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


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


def query_gpu() -> dict[str, float | None]:
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
        }
    except Exception:
        return {k: None for k in ["gpu_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_temperature_c", "gpu_power_w"]}


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
                snap = snapshot()
                writer.writerow(snap.__dict__)
                handle.flush()
                if snap.gpu_temperature_c is not None and snap.gpu_temperature_c >= self.hard_gpu_temp:
                    self.abort_event.set()
                if snap.ram_available_gb < self.min_available_ram_gb:
                    self.abort_event.set()
                self.stop_event.wait(self.interval_seconds)
