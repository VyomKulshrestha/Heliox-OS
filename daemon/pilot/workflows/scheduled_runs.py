"""Scheduled agent runs: a goal Heliox works on by itself on a timetable.

Each schedule names the goal, when it runs (every N minutes, or daily at a time
on chosen weekdays) and the action types it may use. A run is an ordinary
autonomous job limited to that approved list; actions on the list still follow
the normal approval rules, so read-only and user-level actions run unattended
and anything that needs approval waits for the person.

Schedules are saved in SQLite and survive restarts. A run that came due while
Heliox was off for longer than ``MISSED_GRACE_SECONDS`` is skipped and marked
missed rather than fired late, and a schedule never starts a second run while
its previous run is still going.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from pilot import config as pilot_config
from pilot.actions import ActionType

logger = logging.getLogger("pilot.workflows.scheduled_runs")

SCHEDULED_RUNS_DB_FILENAME = "scheduled_runs.db"
MIN_INTERVAL_MINUTES = 5
MAX_INTERVAL_MINUTES = 7 * 24 * 60
MAX_ALLOWED_ACTIONS = 60
MISSED_GRACE_SECONDS = 15 * 60
TICK_SECONDS = 20.0

_DAILY_AT = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class ScheduledRun:
    run_id: str
    name: str
    goal: str
    allowed_actions: list[str]
    every_minutes: int = 0  # interval schedule; 0 when daily_at is used
    daily_at: str = ""  # "HH:MM" local time
    weekdays: list[int] = field(default_factory=list)  # 0 = Monday ... 6 = Sunday; empty = every day
    enabled: bool = True
    next_run_at: float = 0.0
    last_run_at: float = 0.0
    last_job_id: str = ""
    last_status: str = ""  # started | missed | skipped_still_running | failed_to_start
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledRun:
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


def validate_schedule(
    *,
    name: str,
    goal: str,
    allowed_actions: list[str],
    every_minutes: int = 0,
    daily_at: str = "",
    weekdays: list[int] | None = None,
) -> tuple[str, str, list[str], int, str, list[int]]:
    """Normalize a schedule definition; raises ValueError with a message the person can act on."""
    name = name.strip() if isinstance(name, str) else ""
    goal = goal.strip() if isinstance(goal, str) else ""
    if not 1 <= len(name) <= 80:
        raise ValueError("Name must be 1 to 80 characters")
    if not 1 <= len(goal) <= 2000:
        raise ValueError("Goal must be 1 to 2000 characters")
    if not isinstance(allowed_actions, list) or not allowed_actions:
        raise ValueError("List at least one action the run may use")
    if len(allowed_actions) > MAX_ALLOWED_ACTIONS:
        raise ValueError(f"At most {MAX_ALLOWED_ACTIONS} actions can be approved")
    known = {action.value for action in ActionType}
    unknown = sorted({str(item) for item in allowed_actions} - known)
    if unknown:
        raise ValueError(f"Unknown action types: {', '.join(unknown)}")
    actions = sorted({str(item) for item in allowed_actions})

    every_minutes = int(every_minutes or 0)
    daily_at = daily_at.strip() if isinstance(daily_at, str) else ""
    if bool(every_minutes) == bool(daily_at):
        raise ValueError("Choose either every_minutes or daily_at")
    if every_minutes and not MIN_INTERVAL_MINUTES <= every_minutes <= MAX_INTERVAL_MINUTES:
        raise ValueError(f"every_minutes must be between {MIN_INTERVAL_MINUTES} and {MAX_INTERVAL_MINUTES}")
    if daily_at and not _DAILY_AT.fullmatch(daily_at):
        raise ValueError("daily_at must be HH:MM in 24-hour time")
    days = sorted({int(day) for day in weekdays or []})
    if any(day < 0 or day > 6 for day in days):
        raise ValueError("weekdays must be numbers from 0 (Monday) to 6 (Sunday)")
    if every_minutes and days:
        raise ValueError("weekdays only apply to daily_at schedules")
    return name, goal, actions, every_minutes, daily_at, days


def next_run_after(run: ScheduledRun, now: float) -> float:
    """The first scheduled time strictly after ``now``."""
    if run.every_minutes:
        return now + run.every_minutes * 60
    hour, minute = (int(part) for part in run.daily_at.split(":"))
    current = datetime.fromtimestamp(now)
    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    for _ in range(8):
        if candidate.timestamp() > now and (not run.weekdays or candidate.weekday() in run.weekdays):
            return candidate.timestamp()
        candidate += timedelta(days=1)
    raise ValueError("No upcoming day matches this schedule")


class ScheduledRunStore:
    """One row per schedule; every method opens a short-lived connection."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_file = Path(db_file) if db_file is not None else pilot_config.DATA_DIR / SCHEDULED_RUNS_DB_FILENAME
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
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS scheduled_runs (run_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
                )
                await db.commit()
            self._initialized = True

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            yield db

    async def save(self, run: ScheduledRun) -> None:
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO scheduled_runs (run_id, record_json) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET record_json = excluded.record_json",
                (run.run_id, json.dumps(run.to_dict(), ensure_ascii=False)),
            )
            await db.commit()

    async def list(self) -> list[ScheduledRun]:
        async with self._connect() as db, db.execute("SELECT record_json FROM scheduled_runs") as cursor:
            rows = await cursor.fetchall()
        runs = []
        for (raw,) in rows:
            try:
                runs.append(ScheduledRun.from_dict(json.loads(raw)))
            except (TypeError, ValueError):
                logger.warning("Skipping unreadable scheduled run record")
        return sorted(runs, key=lambda run: run.created_at)

    async def get(self, run_id: str) -> ScheduledRun | None:
        return next((run for run in await self.list() if run.run_id == run_id), None)

    async def delete(self, run_id: str) -> bool:
        async with self._connect() as db:
            cursor = await db.execute("DELETE FROM scheduled_runs WHERE run_id = ?", (run_id,))
            await db.commit()
            return bool(cursor.rowcount)


SubmitRun = Callable[[ScheduledRun], Awaitable[str]]
JobIsActive = Callable[[str], bool]


class ScheduledRunner:
    """Creates schedules and starts the ones that come due."""

    def __init__(self, store: ScheduledRunStore, submit: SubmitRun, job_is_active: JobIsActive) -> None:
        self._store = store
        self._submit = submit
        self._job_is_active = job_is_active
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    async def create(
        self,
        *,
        name: str,
        goal: str,
        allowed_actions: list[str],
        every_minutes: int = 0,
        daily_at: str = "",
        weekdays: list[int] | None = None,
        now: float | None = None,
    ) -> ScheduledRun:
        name, goal, actions, every_minutes, daily_at, days = validate_schedule(
            name=name,
            goal=goal,
            allowed_actions=allowed_actions,
            every_minutes=every_minutes,
            daily_at=daily_at,
            weekdays=weekdays,
        )
        now = time.time() if now is None else now
        run = ScheduledRun(
            run_id=uuid.uuid4().hex[:10],
            name=name,
            goal=goal,
            allowed_actions=actions,
            every_minutes=every_minutes,
            daily_at=daily_at,
            weekdays=days,
            created_at=now,
        )
        run.next_run_at = next_run_after(run, now)
        await self._store.save(run)
        return run

    async def list(self) -> list[ScheduledRun]:
        return await self._store.list()

    async def delete(self, run_id: str) -> bool:
        # Under the lock: a tick that is starting this run would otherwise save it back.
        async with self._lock:
            return await self._store.delete(run_id)

    async def set_enabled(self, run_id: str, enabled: bool, now: float | None = None) -> ScheduledRun | None:
        async with self._lock:
            run = await self._store.get(run_id)
            if run is None:
                return None
            run.enabled = enabled
            if enabled:
                run.next_run_at = next_run_after(run, time.time() if now is None else now)
            await self._store.save(run)
            return run

    async def run_now(self, run_id: str, now: float | None = None) -> ScheduledRun | None:
        """Start a schedule's goal immediately; its timetable is unchanged."""
        async with self._lock:
            run = await self._store.get(run_id)
            if run is None:
                return None
            await self._start(run, time.time() if now is None else now)
            await self._store.save(run)
            return run

    async def tick(self, now: float | None = None) -> list[str]:
        """Start every enabled schedule that is due; returns the run_ids started."""
        now = time.time() if now is None else now
        started: list[str] = []
        async with self._lock:
            for run in await self._store.list():
                if not run.enabled or run.next_run_at > now:
                    continue
                if now - run.next_run_at > MISSED_GRACE_SECONDS:
                    logger.info("Scheduled run %s came due while Heliox was off; skipping it", run.name)
                    run.last_status = "missed"
                elif run.last_job_id and self._job_is_active(run.last_job_id):
                    logger.info("Scheduled run %s is still running; skipping this occurrence", run.name)
                    run.last_status = "skipped_still_running"
                else:
                    await self._start(run, now)
                    if run.last_status == "started":
                        started.append(run.run_id)
                run.next_run_at = next_run_after(run, now)
                await self._store.save(run)
        return started

    async def _start(self, run: ScheduledRun, now: float) -> None:
        try:
            run.last_job_id = await self._submit(run)
            run.last_status = "started"
        except Exception as error:
            logger.warning("Scheduled run %s could not start: %s", run.name, error)
            run.last_status = "failed_to_start"
        run.last_run_at = now

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Scheduled run check failed", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)
