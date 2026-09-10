from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

FIELDS = [
    "timestamp", "elapsed_seconds", "cpu_percent", "ram_total_gb", "ram_available_gb",
    "ram_percent", "process_rss_gb", "disk_free_gb", "gpu_name", "gpu_util_percent",
    "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_temperature_c"
]


def _gpu_sample() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        parts = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
        return {
            "gpu_name": parts[0],
            "gpu_util_percent": float(parts[1]),
            "gpu_memory_used_mb": float(parts[2]),
            "gpu_memory_total_mb": float(parts[3]),
            "gpu_temperature_c": float(parts[4]),
        }
    except Exception:
        return {
            "gpu_name": "unavailable",
            "gpu_util_percent": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
            "gpu_temperature_c": "",
        }


class ResourceMonitor:
    def __init__(self, output_path: Path, disk_path: Path, poll_seconds: float, started_epoch: float, sample_gpu: bool = True):
        self.output_path = output_path
        self.disk_path = disk_path
        self.poll_seconds = poll_seconds
        self.started_epoch = started_epoch
        self.sample_gpu = sample_gpu
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.poll_seconds * 2))

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        process = psutil.Process(os.getpid())
        with self.output_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            psutil.cpu_percent(interval=None)
            while not self._stop.is_set():
                vm = psutil.virtual_memory()
                du = psutil.disk_usage(str(self.disk_path))
                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "elapsed_seconds": time.time() - self.started_epoch,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_total_gb": vm.total / 1024**3,
                    "ram_available_gb": vm.available / 1024**3,
                    "ram_percent": vm.percent,
                    "process_rss_gb": process.memory_info().rss / 1024**3,
                    "disk_free_gb": du.free / 1024**3,
                    **(_gpu_sample() if self.sample_gpu else {
                        "gpu_name": "sealed_by_config",
                        "gpu_util_percent": "",
                        "gpu_memory_used_mb": "",
                        "gpu_memory_total_mb": "",
                        "gpu_temperature_c": "",
                    }),
                }
                writer.writerow(row)
                handle.flush()
                self._stop.wait(self.poll_seconds)
