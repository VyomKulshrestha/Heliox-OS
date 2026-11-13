"""SQLite-backed checkpoints for resumable multi-step plans."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from pilot import config as pilot_config
from pilot.actions import ActionPlan, ActionResult

WORKFLOW_DB_FILENAME = "workflow_checkpoints.db"

# What the person decided about a checkpoint's plan. Only "approved" resumes as it is; "denied" never resumes;
# "pending" (saved before the approval step), "preview" (a dry run) and "unknown" (saved before 13 November 2026,
# when this was not recorded) resume only through the approval they needed the first time.
APPROVALS = frozenset({"pending", "approved", "denied", "preview", "unknown"})


@dataclass(frozen=True)
class PlanCheckpoint:
    plan_id: str
    user_input: str
    plan: ActionPlan
    results: list[ActionResult]
    completed_count: int
    last_output: str
    snapshot_ids: list[str]
    status: str
    updated_at: str
    approval: str = "unknown"

    @property
    def is_complete(self) -> bool:
        return self.completed_count >= len(self.plan.actions)


class WorkflowCheckpointStore:
    """Persists enough execution state to resume plans without replaying steps."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_file = Path(db_file) if db_file is not None else pilot_config.DATA_DIR / WORKFLOW_DB_FILENAME
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Every store method opens its own short-lived connection.
        busy_timeout + synchronous are connection-local settings, cheap and
        safe to set every time, matching db/sqlite_pool.py's own connection
        settings and this store's sibling VoiceGestureWorkflowStore -- lets
        two connections writing near-simultaneously wait for the lock
        instead of immediately raising "database is locked". journal_mode
        is deliberately NOT set here -- see initialize()."""
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute("PRAGMA synchronous = NORMAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            yield db

    async def initialize(self) -> None:
        """Switches the database file to WAL journal mode exactly once
        (guarded by _init_lock) -- SQLite requires no OTHER connection be
        open at the moment journal mode actually changes, a requirement
        busy_timeout/retries do NOT satisfy, so re-issuing this PRAGMA on
        every short-lived connection can raise "database is locked" under
        real concurrency even with busy_timeout set. WAL persists in the
        file's own header once set, so this only needs to run once."""
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
                    CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                        plan_id TEXT PRIMARY KEY,
                        user_input TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        results_json TEXT NOT NULL,
                        completed_count INTEGER NOT NULL,
                        last_output TEXT NOT NULL,
                        snapshot_ids_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cursor = await db.execute("PRAGMA table_info(workflow_checkpoints)")
                columns = {row[1] for row in await cursor.fetchall()}
                await cursor.close()
                if "approval" not in columns:  # a store from before 13 November 2026
                    await db.execute(
                        "ALTER TABLE workflow_checkpoints ADD COLUMN approval TEXT NOT NULL DEFAULT 'unknown'"
                    )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_status
                    ON workflow_checkpoints(status)
                    """
                )
                await db.commit()
            self._initialized = True

    async def start_plan(self, plan_id: str, user_input: str, plan: ActionPlan) -> PlanCheckpoint:
        checkpoint = PlanCheckpoint(
            plan_id=plan_id,
            user_input=user_input,
            plan=plan,
            results=[],
            completed_count=0,
            last_output="",
            snapshot_ids=[],
            status="running",
            updated_at=self._now(),
            approval="pending",
        )
        await self._upsert(checkpoint)
        return checkpoint

    async def mark_approval(self, plan_id: str, approval: str) -> PlanCheckpoint | None:
        """Record what the person decided about the plan (APPROVALS)."""
        if approval not in APPROVALS:
            raise ValueError(f"unknown approval: {approval}")
        checkpoint = await self.get(plan_id)
        if checkpoint is None:
            return None
        updated = replace(checkpoint, approval=approval, updated_at=self._now())
        await self._upsert(updated)
        return updated

    async def replace_plan(self, plan_id: str, plan: ActionPlan) -> PlanCheckpoint | None:
        """Keep the results so far and put ``plan`` in place: a resume approved only some of what was left."""
        checkpoint = await self.get(plan_id)
        if checkpoint is None:
            return None
        updated = replace(checkpoint, plan=plan, updated_at=self._now())
        await self._upsert(updated)
        return updated

    async def record_result(
        self,
        plan_id: str,
        result: ActionResult,
        *,
        status: str | None = None,
    ) -> PlanCheckpoint:
        checkpoint = await self.get(plan_id)
        if checkpoint is None:
            raise KeyError(f"No checkpoint exists for plan_id: {plan_id}")

        results = [*checkpoint.results, result]
        completed_count = checkpoint.completed_count + (1 if result.success else 0)
        snapshot_ids = list(checkpoint.snapshot_ids)
        if result.snapshot_id and result.snapshot_id not in snapshot_ids:
            snapshot_ids.append(result.snapshot_id)
        updated = replace(
            checkpoint,
            results=results,
            completed_count=completed_count,
            last_output=result.output or checkpoint.last_output,
            snapshot_ids=snapshot_ids,
            status=status or ("failed" if not result.success else "running"),
            updated_at=self._now(),
        )
        await self._upsert(updated)
        return updated

    async def mark_status(self, plan_id: str, status: str) -> PlanCheckpoint | None:
        checkpoint = await self.get(plan_id)
        if checkpoint is None:
            return None
        updated = replace(checkpoint, status=status, updated_at=self._now())
        await self._upsert(updated)
        return updated

    async def get(self, plan_id: str) -> PlanCheckpoint | None:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT
                    plan_id,
                    user_input,
                    plan_json,
                    results_json,
                    completed_count,
                    last_output,
                    snapshot_ids_json,
                    status,
                    updated_at,
                    approval
                FROM workflow_checkpoints
                WHERE plan_id = ?
                """,
                (plan_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        return self._from_row(row)

    async def _upsert(self, checkpoint: PlanCheckpoint) -> None:
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO workflow_checkpoints (
                    plan_id,
                    user_input,
                    plan_json,
                    results_json,
                    completed_count,
                    last_output,
                    snapshot_ids_json,
                    status,
                    updated_at,
                    approval
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    user_input = excluded.user_input,
                    plan_json = excluded.plan_json,
                    results_json = excluded.results_json,
                    completed_count = excluded.completed_count,
                    last_output = excluded.last_output,
                    snapshot_ids_json = excluded.snapshot_ids_json,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    approval = excluded.approval
                """,
                self._to_row(checkpoint),
            )
            await db.commit()

    @staticmethod
    def _to_row(checkpoint: PlanCheckpoint) -> tuple[Any, ...]:
        return (
            checkpoint.plan_id,
            checkpoint.user_input,
            checkpoint.plan.model_dump_json(),
            json.dumps([result.model_dump(mode="json") for result in checkpoint.results]),
            checkpoint.completed_count,
            checkpoint.last_output,
            json.dumps(checkpoint.snapshot_ids),
            checkpoint.status,
            checkpoint.updated_at,
            checkpoint.approval,
        )

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> PlanCheckpoint:
        results_raw = json.loads(row[3])
        return PlanCheckpoint(
            plan_id=row[0],
            user_input=row[1],
            plan=ActionPlan.model_validate_json(row[2]),
            results=[ActionResult.model_validate(result) for result in results_raw],
            completed_count=int(row[4]),
            last_output=row[5],
            snapshot_ids=list(json.loads(row[6])),
            status=row[7],
            updated_at=row[8],
            approval=row[9],
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
