"""Tests for daemon_manager module."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pilot.system.daemon_manager import (
    MAX_BACKOFF_SECONDS,
    MAX_RESTART_ATTEMPTS,
    CrashRecord,
    DaemonManager,
)


class TestCrashRecord:
    """Tests for CrashRecord class."""

    def test_crash_record_creation(self):
        """Test CrashRecord can be created with all fields."""
        crash = CrashRecord(
            timestamp="2026-05-17T10:00:00",
            exit_code=1,
            exception_type="MemoryError",
            exception_message="Out of memory",
            stack_trace="Traceback...",
            restart_count=2,
        )
        assert crash.timestamp == "2026-05-17T10:00:00"
        assert crash.exit_code == 1
        assert crash.exception_type == "MemoryError"
        assert crash.restart_count == 2


class TestDaemonManager:
    """Tests for DaemonManager class."""

    @pytest.fixture
    def manager(self):
        """Create a DaemonManager instance."""
        return DaemonManager()

    def test_initial_state(self, manager):
        """Test manager starts with correct initial state."""
        assert manager._restart_count == 0
        assert manager._backoff == 1

    def test_calculate_backoff_first_attempt(self, manager):
        """Test backoff calculation for first restart attempt."""
        manager._restart_count = 0
        backoff = manager.calculate_backoff()
        assert backoff == 1

    def test_calculate_backoff_exponential(self, manager):
        """Test exponential backoff calculation."""
        manager._restart_count = 0
        backoffs = []
        for _i in range(5):
            backoffs.append(manager.calculate_backoff())
            manager._restart_count += 1
        assert backoffs[0] == 1
        assert backoffs[1] == 2
        assert backoffs[2] == 4
        assert backoffs[3] == 8
        assert backoffs[4] == 16

    def test_calculate_backoff_capped_at_max(self, manager):
        """Test backoff is capped at MAX_BACKOFF_SECONDS."""
        manager._restart_count = 10
        backoff = manager.calculate_backoff()
        assert backoff == MAX_BACKOFF_SECONDS

    def test_reset_backoff(self, manager):
        """Test backoff is reset after successful start."""
        manager._restart_count = 5
        manager._backoff = 30
        manager.reset_backoff()
        assert manager._restart_count == 0
        assert manager._backoff == 1

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self, manager, tmp_path):
        """Test crash database table is created on initialization."""
        with patch("pilot.system.daemon_manager.CRASH_DB_PATH", tmp_path / "test.db"):
            await manager.initialize()
            import aiosqlite

            async with (
                aiosqlite.connect(tmp_path / "test.db") as db,
                db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor,
            ):
                rows = await cursor.fetchall()
                table_names = [r[0] for r in rows]
                assert "crashes" in table_names

    @pytest.mark.asyncio
    async def test_save_and_get_crashes(self, manager, tmp_path):
        """Test crash records can be saved and retrieved."""
        with patch("pilot.system.daemon_manager.CRASH_DB_PATH", tmp_path / "test.db"):
            await manager.initialize()

            crash = CrashRecord(
                timestamp="2026-05-17T10:00:00",
                exit_code=1,
                exception_type="MemoryError",
                exception_message="Out of memory",
                stack_trace="Traceback (most recent call last):\n  File ...",
                restart_count=0,
            )
            await manager.save_crash(crash)

            crashes = await manager.get_recent_crashes(limit=10)
            assert len(crashes) == 1
            assert crashes[0].exception_type == "MemoryError"
            assert crashes[0].restart_count == 0

    @pytest.mark.asyncio
    async def test_run_with_auto_restart_success(self, manager):
        """Test daemon runs successfully without restart."""
        call_count = 0

        async def successful_start():
            nonlocal call_count
            call_count += 1
            return "started"

        await manager.run_with_auto_restart(successful_start)
        assert call_count == 1
        assert manager._restart_count == 0

    @pytest.mark.asyncio
    async def test_run_with_auto_restart_retries_on_failure(self, manager):
        """Test daemon retries on failure up to max attempts."""
        call_count = 0
        fail_count = 3

        async def failing_start():
            nonlocal call_count
            call_count += 1
            if call_count <= fail_count:
                raise Exception("Simulated crash")
            return "started"

        await manager.run_with_auto_restart(failing_start)
        assert call_count == fail_count + 1

    @pytest.mark.asyncio
    async def test_run_with_auto_restart_gives_up_after_max(self, manager):
        """Test daemon gives up after max restart attempts."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")

        await manager.run_with_auto_restart(always_fails)
        assert call_count == MAX_RESTART_ATTEMPTS
