from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    stage: str
    family: str
    outer_fold: int
    seed: int
    priority: int
    required_profile: str
    backend: str
    threads: int
    payload: dict[str, Any]


class RefineTaskRegistry:
    """Persistent task queue shared by PUBG and full-load profiles.

    A claimed task keeps its backend and thread count until completion. Completed tasks
    are immutable and are skipped by later runs. Deep-learning tasks are marked full-only.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    family TEXT NOT NULL,
                    outer_fold INTEGER NOT NULL DEFAULT -1,
                    seed INTEGER NOT NULL DEFAULT -1,
                    priority INTEGER NOT NULL,
                    required_profile TEXT NOT NULL DEFAULT 'any',
                    status TEXT NOT NULL DEFAULT 'pending',
                    backend TEXT,
                    threads INTEGER,
                    created_profile TEXT,
                    payload TEXT NOT NULL,
                    claimed_by INTEGER,
                    claimed_worker TEXT,
                    claimed_at REAL,
                    heartbeat_at REAL,
                    completed_at REAL,
                    result_path TEXT,
                    record_path TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_refine_tasks_claim
                    ON tasks(status, required_profile, stage, priority DESC, outer_fold, seed);
                """
            )

    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def ensure_tasks(self, tasks: Iterable[dict[str, Any]]) -> None:
        rows = []
        for task in tasks:
            rows.append((
                task["task_id"], task["stage"], task.get("family", task["stage"]),
                int(task.get("outer_fold", -1)), int(task.get("seed", -1)), int(task.get("priority", 0)),
                task.get("required_profile", "any"),
                json.dumps(task.get("payload", {}), ensure_ascii=False, sort_keys=True, default=str),
            ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT INTO tasks(task_id,stage,family,outer_fold,seed,priority,required_profile,payload) "
                    "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET priority=excluded.priority "
                    "WHERE tasks.status='pending'",
                    rows,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            import psutil
            return bool(psutil.pid_exists(pid))
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def reset_stale_claims(self, stale_seconds: float = 600.0) -> int:
        now = time.time()
        reset = 0
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT task_id,claimed_by,heartbeat_at FROM tasks WHERE status='running'"
            ).fetchall()
            for row in rows:
                pid = int(row["claimed_by"] or -1)
                heartbeat = float(row["heartbeat_at"] or 0.0)
                if not self._pid_exists(pid) or now - heartbeat > stale_seconds:
                    conn.execute(
                        "UPDATE tasks SET status='pending',claimed_by=NULL,claimed_worker=NULL,"
                        "claimed_at=NULL,heartbeat_at=NULL,error=? WHERE task_id=?",
                        ("stale claim reset", row["task_id"]),
                    )
                    reset += 1
        return reset

    def claim_next(
        self,
        *,
        profile: str,
        stages: tuple[str, ...],
        worker_name: str,
        backend: str,
        threads: int,
    ) -> TaskClaim | None:
        if not stages:
            return None
        placeholders = ",".join("?" for _ in stages)
        allowed_profiles = ("any", "pubg") if profile == "pubg" else ("any", "pubg", "full")
        profile_placeholders = ",".join("?" for _ in allowed_profiles)
        params = [*stages, *allowed_profiles]
        now = time.time()
        pid = os.getpid()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT * FROM tasks WHERE status='pending' AND stage IN ({placeholders}) "
                    f"AND required_profile IN ({profile_placeholders}) "
                    "ORDER BY priority DESC, outer_fold, seed, task_id LIMIT 1",
                    params,
                ).fetchone()
                if row is None:
                    conn.commit()
                    return None
                locked_backend = str(row["backend"] or backend)
                locked_threads = int(row["threads"] or threads)
                created_profile = str(row["created_profile"] or profile)
                conn.execute(
                    "UPDATE tasks SET status='running',backend=?,threads=?,created_profile=?,claimed_by=?,"
                    "claimed_worker=?,claimed_at=?,heartbeat_at=?,error=NULL WHERE task_id=? AND status='pending'",
                    (locked_backend, locked_threads, created_profile, pid, worker_name, now, now, row["task_id"]),
                )
                changed = conn.execute("SELECT changes() AS n").fetchone()["n"]
                if changed != 1:
                    conn.rollback()
                    return None
                conn.commit()
                return TaskClaim(
                    task_id=str(row["task_id"]), stage=str(row["stage"]), family=str(row["family"]),
                    outer_fold=int(row["outer_fold"]), seed=int(row["seed"]), priority=int(row["priority"]),
                    required_profile=str(row["required_profile"]), backend=locked_backend, threads=locked_threads,
                    payload=json.loads(row["payload"]),
                )
            except Exception:
                conn.rollback()
                raise

    def heartbeat(self, task_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE tasks SET heartbeat_at=? WHERE task_id=? AND status='running'", (time.time(), task_id))

    def complete(self, task_id: str, *, result_path: str = "", record_path: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='completed',completed_at=?,result_path=?,record_path=?,"
                "claimed_by=NULL,claimed_worker=NULL,heartbeat_at=NULL,error=NULL WHERE task_id=?",
                (time.time(), result_path, record_path, task_id),
            )

    def release(self, task_id: str, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='pending',claimed_by=NULL,claimed_worker=NULL,claimed_at=NULL,"
                "heartbeat_at=NULL,error=? WHERE task_id=?",
                ((error or "")[-4000:] or None, task_id),
            )

    def fail(self, task_id: str, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='failed',completed_at=?,claimed_by=NULL,claimed_worker=NULL,"
                "heartbeat_at=NULL,error=? WHERE task_id=?",
                (time.time(), error[-8000:], task_id),
            )

    def is_completed(self, task_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return row is not None and row["status"] == "completed"

    def status(self) -> dict[str, Any]:
        with self.connect() as conn:
            grouped = conn.execute(
                "SELECT stage,status,backend,COUNT(*) AS n FROM tasks GROUP BY stage,status,backend ORDER BY stage,status"
            ).fetchall()
            total = int(conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"])
            completed = int(conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='completed'").fetchone()["n"])
            running = conn.execute(
                "SELECT task_id,stage,family,outer_fold,seed,backend,threads,claimed_worker,heartbeat_at "
                "FROM tasks WHERE status='running' ORDER BY claimed_worker"
            ).fetchall()
            failed = conn.execute(
                "SELECT task_id,stage,family,outer_fold,seed,error FROM tasks WHERE status='failed' ORDER BY completed_at DESC"
            ).fetchall()
        return {
            "tasks_total": total,
            "tasks_completed": completed,
            "progress_ratio": float(completed / total) if total else 0.0,
            "grouped": [dict(row) for row in grouped],
            "running": [dict(row) for row in running],
            "failed": [dict(row) for row in failed],
        }
