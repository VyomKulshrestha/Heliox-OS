from __future__ import annotations

import asyncio

import pytest

from pilot.db.sqlite_pool import AsyncSqlitePool


@pytest.mark.asyncio
async def test_sqlite_pool_reuses_read_connections_and_writes(tmp_path):
    pool = AsyncSqlitePool(tmp_path / "pool.db", read_pool_size=2)
    await pool.start()
    try:
        async with pool.write() as db:
            await db.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            await db.execute("INSERT INTO demo (value) VALUES (?)", ("ready",))
            await db.commit()

        async with pool.read() as db:
            cursor = await db.execute("SELECT value FROM demo")
            row = await cursor.fetchone()
            await cursor.close()

        assert row[0] == "ready"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_sqlite_pool_allows_concurrent_reads(tmp_path):
    pool = AsyncSqlitePool(tmp_path / "pool.db", read_pool_size=3)
    await pool.start()
    try:
        async with pool.write() as db:
            await db.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
            await db.executemany("INSERT INTO demo (value) VALUES (?)", [(1,), (2,), (3,)])
            await db.commit()

        async def read_total() -> int:
            async with pool.read() as db:
                cursor = await db.execute("SELECT SUM(value) FROM demo")
                row = await cursor.fetchone()
                await cursor.close()
                return int(row[0])

        assert await asyncio.gather(read_total(), read_total(), read_total()) == [6, 6, 6]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_sqlite_pool_serializes_writes(tmp_path):
    pool = AsyncSqlitePool(tmp_path / "pool.db", read_pool_size=1)
    await pool.start()
    try:
        async with pool.write() as db:
            await db.execute("CREATE TABLE demo (value INTEGER NOT NULL)")
            await db.commit()

        async def insert_value(value: int) -> None:
            async with pool.write() as db:
                await db.execute("INSERT INTO demo (value) VALUES (?)", (value,))
                await db.commit()

        await asyncio.gather(*(insert_value(value) for value in range(10)))

        async with pool.read() as db:
            cursor = await db.execute("SELECT COUNT(*), SUM(value) FROM demo")
            row = await cursor.fetchone()
            await cursor.close()

        assert row == (10, 45)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_sqlite_pool_close_rejects_new_leases(tmp_path):
    pool = AsyncSqlitePool(tmp_path / "pool.db")
    await pool.start()
    await pool.close()

    with pytest.raises(RuntimeError, match="not started"):
        async with pool.read():
            pass
