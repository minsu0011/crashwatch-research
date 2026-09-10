from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TickerTask:
    task_id: str
    stage: str
    ticker: str
    variant: str
    seed: int
    priority: int
    backend: str
    threads: int
    payload: dict[str, Any]
    attempts: int = 0


class TickerTaskRegistry:
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
                    ticker TEXT NOT NULL,
                    variant TEXT NOT NULL DEFAULT '',
                    seed INTEGER NOT NULL DEFAULT -1,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    backend TEXT,
                    threads INTEGER,
                    payload TEXT NOT NULL,
                    claimed_by INTEGER,
                    claimed_worker TEXT,
                    claimed_at REAL,
                    heartbeat_at REAL,
                    completed_at REAL,
                    result_path TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_ticker_map_claim
                ON tasks(status, stage, priority DESC, ticker, variant, seed);
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
            if "attempts" not in columns:
                conn.execute("ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

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
                task["task_id"], task["stage"], str(task["ticker"]), task.get("variant", ""),
                int(task.get("seed", -1)), int(task.get("priority", 0)),
                json.dumps(task.get("payload", {}), ensure_ascii=False, sort_keys=True, default=str),
            ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT OR IGNORE INTO tasks(task_id,stage,ticker,variant,seed,priority,payload) "
                    "VALUES(?,?,?,?,?,?,?)",
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

    def reset_stale(self, stale_seconds: float = 900.0) -> int:
        now = time.time()
        count = 0
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
                    count += 1
        return count

    def claim_next(
        self,
        *,
        stages: tuple[str, ...],
        worker_name: str,
        backend: str,
        threads: int,
    ) -> TickerTask | None:
        if not stages:
            return None
        placeholders = ",".join("?" for _ in stages)
        now = time.time()
        pid = os.getpid()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT * FROM tasks WHERE status='pending' AND stage IN ({placeholders}) "
                    "ORDER BY priority DESC,ticker,variant,seed LIMIT 1",
                    list(stages),
                ).fetchone()
                if row is None:
                    conn.commit()
                    return None
                locked_backend = str(row["backend"] or backend)
                locked_threads = int(row["threads"] or threads)
                conn.execute(
                    "UPDATE tasks SET status='running',backend=?,threads=?,claimed_by=?,claimed_worker=?,"
                    "claimed_at=?,heartbeat_at=?,error=NULL,attempts=attempts+1 WHERE task_id=? AND status='pending'",
                    (locked_backend, locked_threads, pid, worker_name, now, now, row["task_id"]),
                )
                changed = conn.execute("SELECT changes() AS n").fetchone()["n"]
                if changed != 1:
                    conn.rollback()
                    return None
                conn.commit()
                return TickerTask(
                    task_id=str(row["task_id"]), stage=str(row["stage"]), ticker=str(row["ticker"]),
                    variant=str(row["variant"]), seed=int(row["seed"]), priority=int(row["priority"]),
                    backend=locked_backend, threads=locked_threads, payload=json.loads(row["payload"]),
                    attempts=int(row["attempts"] or 0) + 1,
                )
            except Exception:
                conn.rollback()
                raise

    def heartbeat(self, task_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET heartbeat_at=? WHERE task_id=? AND status='running'",
                (time.time(), task_id),
            )

    def complete(self, task_id: str, result_path: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='completed',completed_at=?,result_path=?,claimed_by=NULL,"
                "claimed_worker=NULL,heartbeat_at=NULL,error=NULL WHERE task_id=?",
                (time.time(), result_path, task_id),
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

    def status(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"])
            completed = int(conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='completed'").fetchone()["n"])
            pending = int(conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='pending'").fetchone()["n"])
            running_count = int(conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='running'").fetchone()["n"])
            failed_count = int(conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status='failed'").fetchone()["n"])
            grouped = conn.execute(
                "SELECT stage,status,backend,COUNT(*) AS n FROM tasks "
                "GROUP BY stage,status,backend ORDER BY stage,status,backend"
            ).fetchall()
            running = conn.execute(
                "SELECT task_id,stage,ticker,variant,seed,backend,threads,claimed_worker,heartbeat_at "
                "FROM tasks WHERE status='running' ORDER BY claimed_worker"
            ).fetchall()
            failed = conn.execute(
                "SELECT task_id,stage,ticker,variant,seed,attempts,error FROM tasks WHERE status='failed' "
                "ORDER BY completed_at DESC"
            ).fetchall()
            duration = conn.execute(
                "SELECT AVG(completed_at-claimed_at) AS avg_seconds, "
                "MIN(completed_at-claimed_at) AS min_seconds, "
                "MAX(completed_at-claimed_at) AS max_seconds "
                "FROM tasks WHERE status='completed' AND claimed_at IS NOT NULL AND completed_at IS NOT NULL"
            ).fetchone()
        terminal = completed + failed_count
        return {
            "tasks_total": total,
            "tasks_completed": completed,
            "tasks_pending": pending,
            "tasks_running": running_count,
            "tasks_failed": failed_count,
            "tasks_terminal": terminal,
            "tasks_unfinished": pending + running_count,
            "progress_ratio": completed / total if total else 0.0,
            "terminal_ratio": terminal / total if total else 0.0,
            "average_completed_task_seconds": float(duration["avg_seconds"] or 0.0),
            "min_completed_task_seconds": float(duration["min_seconds"] or 0.0),
            "max_completed_task_seconds": float(duration["max_seconds"] or 0.0),
            "grouped": [dict(row) for row in grouped],
            "running": [dict(row) for row in running],
            "failed": [dict(row) for row in failed],
        }
