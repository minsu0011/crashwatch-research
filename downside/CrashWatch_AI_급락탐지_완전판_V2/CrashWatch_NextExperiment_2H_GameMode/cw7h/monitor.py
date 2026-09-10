from __future__ import annotations

import csv
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

LOGGER = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(self, output_path: Path, interval_seconds: float = 5.0):
        self.output_path = Path(output_path)
        self.interval = max(1.0, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = time.time()

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2 + 2)

    def _gpu_stats(self) -> dict[str, Any]:
        result = {
            "gpu_name": "",
            "gpu_util_percent": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
            "gpu_temperature_c": "",
        }
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                parts = [p.strip() for p in proc.stdout.splitlines()[0].split(",")]
                if len(parts) >= 5:
                    result = {
                        "gpu_name": parts[0],
                        "gpu_util_percent": parts[1],
                        "gpu_memory_used_mb": parts[2],
                        "gpu_memory_total_mb": parts[3],
                        "gpu_temperature_c": parts[4],
                    }
        except Exception:
            pass
        return result

    def _run(self) -> None:
        fields = [
            "timestamp", "elapsed_seconds", "cpu_percent", "ram_total_gb", "ram_available_gb", "ram_percent",
            "process_rss_gb", "disk_free_gb", "gpu_name", "gpu_util_percent", "gpu_memory_used_mb",
            "gpu_memory_total_mb", "gpu_temperature_c",
        ]
        process = psutil.Process(os.getpid())
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        with self.output_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            psutil.cpu_percent(interval=None)
            while not self._stop.is_set():
                vm = psutil.virtual_memory()
                try:
                    rss = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except Exception:
                    rss = 0
                try:
                    disk_free = psutil.disk_usage(self.output_path.parent.resolve()).free / 1024**3
                except Exception:
                    disk_free = float("nan")
                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "elapsed_seconds": time.time() - self.started,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_total_gb": vm.total / 1024**3,
                    "ram_available_gb": vm.available / 1024**3,
                    "ram_percent": vm.percent,
                    "process_rss_gb": rss / 1024**3,
                    "disk_free_gb": disk_free,
                    **self._gpu_stats(),
                }
                writer.writerow(row)
                f.flush()
                self._stop.wait(self.interval)
