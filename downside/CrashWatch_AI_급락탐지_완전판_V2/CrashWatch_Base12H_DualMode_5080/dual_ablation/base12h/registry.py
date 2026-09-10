from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BlockClaim:
    block_id: str
    outer_fold: int
    seed: int
    backend: str
    threads: int
    created_profile: str


class TaskRegistry:
    """SQLite 작업대장.

    한 block은 outer_fold × seed 단위이며, block에 최초 할당된 backend/threads는
    완료될 때까지 유지한다. 따라서 배그 모드에서 중단한 CPU block을 풀로드로
    재개해도 남은 task는 CPU 4스레드로 마무리되고, 다음 block부터 CUDA를 사용한다.
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
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    outer_fold INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    backend TEXT,
                    threads INTEGER,
                    created_profile TEXT,
                    claimed_by INTEGER,
                    claimed_worker TEXT,
                    claimed_at REAL,
                    heartbeat_at REAL,
                    completed_at REAL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_blocks_status_priority
                    ON blocks(status, priority DESC, outer_fold, seed);
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    block_id TEXT NOT NULL,
                    experiment TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    backend TEXT NOT NULL,
                    threads INTEGER NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result_path TEXT,
                    record_path TEXT,
                    error TEXT,
                    FOREIGN KEY(block_id) REFERENCES blocks(block_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_block ON tasks(block_id, status);
                """
            )
            # block_id already contains the dataset signature.  The old unique
            # (fold, seed) index prevented a changed dataset from registering a
            # new generation of blocks in the shared registry.
            conn.execute("DROP INDEX IF EXISTS idx_blocks_fold_seed")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocks_fold_seed "
                "ON blocks(outer_fold, seed)"
            )

    def set_meta(self, key: str, value: object) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )

    def get_meta(self, key: str, default: object = None) -> object:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def ensure_blocks(self, blocks: Iterable[tuple[str, int, int, int]]) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT OR IGNORE INTO blocks(block_id,outer_fold,seed,priority,status) "
                    "VALUES(?,?,?,?, 'pending')",
                    list(blocks),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def reset_stale_claims(self, stale_seconds: float = 300.0) -> int:
        now = time.time()
        reset = 0
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT block_id, claimed_by, heartbeat_at FROM blocks WHERE status='running'"
            ).fetchall()
            for row in rows:
                pid = int(row["claimed_by"] or -1)
                heartbeat = float(row["heartbeat_at"] or 0.0)
                alive = pid > 0 and self._pid_exists(pid)
                if not alive or now - heartbeat > stale_seconds:
                    conn.execute(
                        "UPDATE blocks SET status='pending', claimed_by=NULL, claimed_worker=NULL, "
                        "claimed_at=NULL, heartbeat_at=NULL WHERE block_id=?",
                        (row["block_id"],),
                    )
                    conn.execute(
                        "UPDATE tasks SET status='pending', started_at=NULL "
                        "WHERE block_id=? AND status='running'",
                        (row["block_id"],),
                    )
                    reset += 1
        return reset

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

    def claim_next_block(
        self,
        *,
        worker_name: str,
        default_backend: str,
        default_threads: int,
        profile: str,
    ) -> BlockClaim | None:
        pid = os.getpid()
        now = time.time()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM blocks WHERE status='pending' "
                    "ORDER BY CASE WHEN backend IS NOT NULL THEN 0 ELSE 1 END, "
                    "priority DESC, outer_fold, seed LIMIT 1"
                ).fetchone()
                if row is None:
                    conn.commit()
                    return None
                backend = row["backend"] or default_backend
                threads = int(row["threads"] or default_threads)
                created_profile = row["created_profile"] or profile
                conn.execute(
                    "UPDATE blocks SET status='running', backend=?, threads=?, created_profile=?, "
                    "claimed_by=?, claimed_worker=?, claimed_at=?, heartbeat_at=?, error=NULL "
                    "WHERE block_id=? AND status='pending'",
                    (backend, threads, created_profile, pid, worker_name, now, now, row["block_id"]),
                )
                changed = conn.execute("SELECT changes() AS n").fetchone()["n"]
                if changed != 1:
                    conn.rollback()
                    return None
                conn.commit()
                return BlockClaim(
                    block_id=str(row["block_id"]),
                    outer_fold=int(row["outer_fold"]),
                    seed=int(row["seed"]),
                    backend=str(backend),
                    threads=threads,
                    created_profile=str(created_profile),
                )
            except Exception:
                conn.rollback()
                raise

    def heartbeat(self, block_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE blocks SET heartbeat_at=? WHERE block_id=? AND status='running'",
                (time.time(), block_id),
            )

    def task_completed(self, task_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return row is not None and row["status"] == "completed"

    def start_task(self, *, task_id: str, block_id: str, experiment: str, backend: str, threads: int) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tasks(task_id,block_id,experiment,status,backend,threads,started_at) "
                "VALUES(?,?,?,'running',?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET status='running', started_at=excluded.started_at, "
                "backend=excluded.backend, threads=excluded.threads, error=NULL",
                (task_id, block_id, experiment, backend, threads, now),
            )

    def complete_task(self, *, task_id: str, result_path: str, record_path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='completed', completed_at=?, result_path=?, record_path=?, error=NULL "
                "WHERE task_id=?",
                (time.time(), result_path, record_path, task_id),
            )

    def fail_task(self, task_id: str, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='failed', completed_at=?, error=? WHERE task_id=?",
                (time.time(), error[-4000:], task_id),
            )

    def finish_block(self, block_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE blocks SET status='completed', completed_at=?, claimed_by=NULL, "
                "claimed_worker=NULL, heartbeat_at=NULL, error=NULL WHERE block_id=?",
                (time.time(), block_id),
            )

    def release_block(self, block_id: str, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE blocks SET status='pending', claimed_by=NULL, claimed_worker=NULL, "
                "claimed_at=NULL, heartbeat_at=NULL, error=? WHERE block_id=?",
                ((error or "")[-4000:] or None, block_id),
            )

    def status(self) -> dict:
        with self.connect() as conn:
            block_rows = conn.execute(
                "SELECT status, backend, COUNT(*) AS n FROM blocks GROUP BY status, backend"
            ).fetchall()
            task_rows = conn.execute(
                "SELECT status, backend, COUNT(*) AS n FROM tasks GROUP BY status, backend"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS n FROM blocks").fetchone()["n"]
            completed = conn.execute("SELECT COUNT(*) AS n FROM blocks WHERE status='completed'").fetchone()["n"]
            running = conn.execute(
                "SELECT block_id, outer_fold, seed, backend, threads, claimed_worker, heartbeat_at "
                "FROM blocks WHERE status='running' ORDER BY claimed_worker"
            ).fetchall()
        return {
            "blocks_total": int(total),
            "blocks_completed": int(completed),
            "progress_ratio": float(completed / total) if total else 0.0,
            "blocks": [dict(r) for r in block_rows],
            "tasks": [dict(r) for r in task_rows],
            "running": [dict(r) for r in running],
        }
