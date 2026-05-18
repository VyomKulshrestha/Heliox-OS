"""Async SQLite connection pool with read/write separation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite


class AsyncSqlitePool:
    """Small aiosqlite pool tuned for daemon-style shared databases.

    SQLite allows many readers but only one writer. The pool reflects that:
    read connections are leased from a bounded queue while writes are routed
    through one WAL-enabled connection protected by a lock.
    """

    def __init__(self, db_path: str | Path, *, read_pool_size: int = 4, timeout: float = 30.0) -> None:
        if read_pool_size < 1:
            raise ValueError("read_pool_size must be at least 1")

        self._db_path = Path(db_path)
        self._read_pool_size = read_pool_size
        self._timeout = timeout

        self._read_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=read_pool_size)

        self._write_conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

        self._started = False
        self._closed = False

    async def start(self) -> None:
        if self._started:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_conn = await self._open_connection()
        await self._configure_connection(self._write_conn, writable=True)

        for _ in range(self._read_pool_size):
            conn = await self._open_connection()
            await self._configure_connection(conn, writable=False)
            await self._read_pool.put(conn)

        self._started = True
        self._closed = False

    @contextlib.asynccontextmanager
    async def read(self) -> AsyncIterator[aiosqlite.Connection]:
        self._ensure_started()

        conn = await asyncio.wait_for(
            self._read_pool.get(),
            timeout=self._timeout,
        )

        try:
            yield conn
        finally:
            await self._read_pool.put(conn)

    @contextlib.asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        self._ensure_started()

        if self._write_conn is None:
            raise RuntimeError("SQLite pool writer is not initialized")

        async with self._write_lock:
            yield self._write_conn

    async def checkpoint(self, mode: str = "TRUNCATE") -> dict[str, int | str]:
        """Run a WAL checkpoint to flush WAL contents safely to disk."""
        self._ensure_started()

        if self._write_conn is None:
            raise RuntimeError("SQLite pool writer is not initialized")

        mode = mode.upper()
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError(f"Unsupported WAL checkpoint mode: {mode}")

        async with self._write_lock:
            cursor = await self._write_conn.execute(f"PRAGMA wal_checkpoint({mode})")
            row = await cursor.fetchone()
            await cursor.close()
            await self._write_conn.commit()

        busy, log_frames, checkpointed_frames = row
        return {
            "mode": mode,
            "busy": busy,
            "log_frames": log_frames,
            "checkpointed_frames": checkpointed_frames,
        }

    async def close(self) -> None:
        if not self._started or self._closed:
            return

        # Final checkpoint before shutdown
        if self._write_conn is not None:
            await self.checkpoint()
            await self._write_conn.close()
            self._write_conn = None

        while not self._read_pool.empty():
            conn = await self._read_pool.get()
            await conn.close()

        self._closed = True
        self._started = False

    @property
    def read_pool_size(self) -> int:
        return self._read_pool_size

    async def _open_connection(self) -> aiosqlite.Connection:
        return await aiosqlite.connect(
            str(self._db_path),
            timeout=self._timeout,
        )

    async def _configure_connection(
        self,
        conn: aiosqlite.Connection,
        *,
        writable: bool,
    ) -> None:
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.execute("PRAGMA busy_timeout = 5000")

        if not writable:
            await conn.execute("PRAGMA query_only = ON")

        await conn.commit()

    def _ensure_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("SQLite pool is not started")
