from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil


class RunLock:
    def __init__(self, path: Path, profile: str) -> None:
        self.path = path
        self.profile = profile
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(old.get("pid", -1))
                if pid > 0 and psutil.pid_exists(pid):
                    raise RuntimeError(
                        f"집중 Nested 실행이 이미 동작 중입니다. pid={pid}, profile={old.get('profile')}. "
                        "두 프로필을 동시에 실행하면 안 됩니다."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
            self.path.unlink(missing_ok=True)
        payload = {"pid": os.getpid(), "profile": self.profile, "created_at": time.time()}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self.acquired = True
        except FileExistsError as exc:
            raise RuntimeError("다른 집중 Nested 실행이 lock을 선점했습니다.") from exc

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if int(current.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
        except Exception:
            pass
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def apply_process_profile(profile: str, threads: int) -> dict:
    process = psutil.Process()
    available = process.cpu_affinity() if hasattr(process, "cpu_affinity") else list(range(psutil.cpu_count() or threads))
    selected = available
    if profile == "game":
        physical_first = available[::2] or available
        selected = physical_first[:threads]
        if len(selected) < threads:
            selected = available[:threads]
        if hasattr(process, "cpu_affinity"):
            process.cpu_affinity(selected)
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(10)
    else:
        if hasattr(process, "cpu_affinity"):
            process.cpu_affinity(available)
        if os.name == "nt":
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            try:
                process.nice(-5)
            except Exception:
                pass
    return {"profile": profile, "threads": threads, "cpu_affinity": selected}
