"""Daemon Manager - handles auto-restart, crash capture, and resurrection."""

import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import aiosqlite

from pilot.config import DATA_DIR

logger = logging.getLogger("pilot.system.daemon_manager")

CRASH_DB_PATH = DATA_DIR / "crashes.db"

MAX_RESTART_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60


class CrashRecord:
    """Represents a crash event."""

    def __init__(
        self,
        timestamp: str,
        exit_code: int,
        exception_type: str,
        exception_message: str,
        stack_trace: str,
        restart_count: int,
    ):
        self.timestamp = timestamp
        self.exit_code = exit_code
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.stack_trace = stack_trace
        self.restart_count = restart_count


class DaemonManager:
    """Manages daemon lifecycle with auto-restart and crash dump analysis."""

    def __init__(self):
        self._restart_count = 0
        self._backoff = INITIAL_BACKOFF_SECONDS

    async def initialize(self) -> None:
        """Initialize the crash database."""
        async with aiosqlite.connect(CRASH_DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS crashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    exit_code INTEGER,
                    exception_type TEXT,
                    exception_message TEXT,
                    stack_trace TEXT,
                    restart_count INTEGER,
                    resolved BOOLEAN DEFAULT 0
                )
            """
            )
            await db.commit()
        logger.info("Daemon manager initialized, crash DB: %s", CRASH_DB_PATH)

    async def save_crash(self, crash: CrashRecord) -> None:
        """Save crash record to database."""
        async with aiosqlite.connect(CRASH_DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO crashes (timestamp, exit_code, exception_type, exception_message, stack_trace, restart_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    crash.timestamp,
                    crash.exit_code,
                    crash.exception_type,
                    crash.exception_message,
                    crash.stack_trace,
                    crash.restart_count,
                ),
            )
            await db.commit()
        logger.warning("Crash saved to database: %s", crash.exception_type)

    async def get_recent_crashes(self, limit: int = 10) -> list[CrashRecord]:
        """Get recent crash records."""
        async with aiosqlite.connect(CRASH_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM crashes ORDER BY timestamp DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    CrashRecord(
                        timestamp=row["timestamp"],
                        exit_code=row["exit_code"],
                        exception_type=row["exception_type"],
                        exception_message=row["exception_message"],
                        stack_trace=row["stack_trace"],
                        restart_count=row["restart_count"],
                    )
                    for row in rows
                ]

    def calculate_backoff(self) -> float:
        """Calculate next restart backoff with exponential backoff."""
        backoff = min(self._backoff * (2**self._restart_count), MAX_BACKOFF_SECONDS)
        return backoff

    def reset_backoff(self) -> None:
        """Reset backoff after successful start."""
        self._restart_count = 0
        self._backoff = INITIAL_BACKOFF_SECONDS
        logger.info("Backoff reset - daemon running stable")

    async def run_with_auto_restart(self, start_daemon) -> None:
        """Run daemon with auto-restart on crash."""
        await self.initialize()

        while self._restart_count < MAX_RESTART_ATTEMPTS:
            try:
                logger.info("Starting daemon (attempt %d/%d)", self._restart_count + 1, MAX_RESTART_ATTEMPTS)
                await start_daemon()
                self.reset_backoff()
                return

            except Exception as e:
                crash = CrashRecord(
                    timestamp=datetime.now().isoformat(),
                    exit_code=1,
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    stack_trace=traceback.format_exc(),
                    restart_count=self._restart_count,
                )
                await self.save_crash(crash)

                self._restart_count += 1

                if self._restart_count >= MAX_RESTART_ATTEMPTS:
                    logger.error(
                        "Max restart attempts (%d) reached. Daemon will not auto-restart.",
                        MAX_RESTART_ATTEMPTS,
                    )
                    break

                backoff = self.calculate_backoff()
                logger.error(
                    "Daemon crashed: %s. Restarting in %.1f seconds (attempt %d/%d)",
                    str(e)[:100],
                    backoff,
                    self._restart_count,
                    MAX_RESTART_ATTEMPTS,
                )
                await asyncio.sleep(backoff)

        logger.error("DaemonManager: Giving up after %d failed restart attempts", MAX_RESTART_ATTEMPTS)


def get_systemd_service_template() -> str:
    """Generate systemd service file content."""
    daemon_path = Path(__file__).parent.parent
    return f"""[Unit]
Description=Heliox OS Pilot Daemon
After=network.target

[Service]
Type=simple
User={os.environ.get('USER', 'root')}
WorkingDirectory={daemon_path}
ExecStart={sys.executable} -m pilot.server
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Memory limits to prevent OOM kills
MemoryMax=2G
MemoryHigh=1G

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""


def get_windows_service_info() -> dict:
    """Get Windows Service setup instructions."""
    return {
        "description": "Heliox OS Pilot Daemon",
        "display_name": "Heliox OS Pilot",
        "start_type": "auto",
        "instructions": """
To create a Windows Service for Heliox OS:

1. Install pywin32:
   pip install pywin32

2. Create service using NSSM (Non-Sucking Service Manager):
   - Download NSSM from https://nssm.cc/download
   - Run: nssm install HelioxOS "C:\\Python\\python.exe" "-m pilot.server"
   - Set: nssm set HelioxOS AppDirectory "C:\\path\\to\\daemon"
   - Set: nssm set HelioxOS Description "Heliox OS Pilot Daemon"
   - Set: nssm set HelioxOS Start SERVICE_AUTO_START

3. Or use Python's servicemanager:
   python -m pip install pywin32
   python -m win32serviceutil install pilot.server
""",
    }


async def main():
    """Test the daemon manager."""
    manager = DaemonManager()

    async def dummy_start():
        print("Daemon started!")
        await asyncio.sleep(1)

    await manager.run_with_auto_restart(dummy_start)


if __name__ == "__main__":
    asyncio.run(main())
