"""SQLite store that keeps autonomous jobs across daemon restarts.

One row per job holding its full record (goal, steps, decomposition, scope
override). A job that was queued, decomposing or running when the daemon
stopped is marked interrupted; nothing resumes on its own. The person resumes
it with ``autonomous_resume`` or dismisses it with ``autonomous_cancel``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from pilot import config as pilot_config

AUTONOMOUS_DB_FILENAME = "autonomous_jobs.db"
ACTIVE_STATES = ("queued", "decomposing", "running")
INTERRUPTED_STATE = "interrupted"
INTERRUPTED_SUMMARY = "Interrupted when Heliox stopped. Resume it when you are ready."
KEEP_JOBS = 200


class AutonomousJobStore:
    """Persists autonomous job records; every method opens a short-lived connection."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_file = Path(db_file) if db_file is not None else pilot_config.DATA_DIR / AUTONOMOUS_DB_FILENAME
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._db_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self._db_file) as db:
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS autonomous_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        record_json TEXT NOT NULL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_autonomous_jobs_status ON autonomous_jobs(status, updated_at)"
                )
                await db.commit()
            self._initialized = True

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute("PRAGMA synchronous = NORMAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            yield db

    async def save(self, record: dict[str, Any]) -> None:
        """Insert or replace one job record."""
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO autonomous_jobs (job_id, status, updated_at, record_json) VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status, updated_at = excluded.updated_at, record_json = excluded.record_json
                """,
                (record["job_id"], record["status"], time.time(), json.dumps(record, ensure_ascii=False)),
            )
            await db.commit()

    async def load_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """The most recently updated job records, newest first."""
        async with (
            self._connect() as db,
            db.execute("SELECT record_json FROM autonomous_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)) as cursor,
        ):
            rows = await cursor.fetchall()
        records = []
        for (raw,) in rows:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("job_id"):
                records.append(record)
        return records

    async def mark_interrupted(self) -> list[str]:
        """Turn jobs that were cut off mid-run into interrupted jobs; returns their ids."""
        placeholders = ", ".join("?" for _ in ACTIVE_STATES)
        async with self._connect() as db:
            async with db.execute(
                f"SELECT job_id, record_json FROM autonomous_jobs WHERE status IN ({placeholders})",
                ACTIVE_STATES,
            ) as cursor:
                rows = await cursor.fetchall()
            interrupted: list[str] = []
            for job_id, raw in rows:
                try:
                    record = json.loads(raw)
                except ValueError:
                    record = {"job_id": job_id}
                record["status"] = INTERRUPTED_STATE
                record["result_summary"] = INTERRUPTED_SUMMARY
                for step in record.get("steps") or []:
                    if isinstance(step, dict) and step.get("status") == "running":
                        step["status"] = INTERRUPTED_STATE
                await db.execute(
                    "UPDATE autonomous_jobs SET status = ?, updated_at = ?, record_json = ? WHERE job_id = ?",
                    (INTERRUPTED_STATE, time.time(), json.dumps(record, ensure_ascii=False), job_id),
                )
                interrupted.append(job_id)
            await db.commit()
        return interrupted

    async def prune(self, keep: int = KEEP_JOBS) -> int:
        """Delete the oldest finished jobs beyond ``keep``; interrupted jobs stay until handled."""
        async with self._connect() as db:
            cursor = await db.execute(
                """
                DELETE FROM autonomous_jobs WHERE job_id IN (
                    SELECT job_id FROM autonomous_jobs
                    WHERE status NOT IN ('queued', 'decomposing', 'running', 'interrupted')
                    ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (keep,),
            )
            await db.commit()
            return cursor.rowcount or 0
