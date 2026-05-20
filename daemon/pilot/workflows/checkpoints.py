"""SQLite-backed checkpoints for resumable multi-step plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from pilot import config as pilot_config
from pilot.actions import ActionPlan, ActionResult

WORKFLOW_DB_FILENAME = "workflow_checkpoints.db"


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

    @property
    def is_complete(self) -> bool:
        return self.completed_count >= len(self.plan.actions)


class WorkflowCheckpointStore:
    """Persists enough execution state to resume plans without replaying steps."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_file = Path(db_file) if db_file is not None else pilot_config.DATA_DIR / WORKFLOW_DB_FILENAME

    async def initialize(self) -> None:
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_file) as db:
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
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_status
                ON workflow_checkpoints(status)
                """
            )
            await db.commit()

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
        )
        await self._upsert(checkpoint)
        return checkpoint

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
        updated = PlanCheckpoint(
            plan_id=checkpoint.plan_id,
            user_input=checkpoint.user_input,
            plan=checkpoint.plan,
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
        updated = PlanCheckpoint(
            plan_id=checkpoint.plan_id,
            user_input=checkpoint.user_input,
            plan=checkpoint.plan,
            results=checkpoint.results,
            completed_count=checkpoint.completed_count,
            last_output=checkpoint.last_output,
            snapshot_ids=checkpoint.snapshot_ids,
            status=status,
            updated_at=self._now(),
        )
        await self._upsert(updated)
        return updated

    async def get(self, plan_id: str) -> PlanCheckpoint | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
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
                    updated_at
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
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
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
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    user_input = excluded.user_input,
                    plan_json = excluded.plan_json,
                    results_json = excluded.results_json,
                    completed_count = excluded.completed_count,
                    last_output = excluded.last_output,
                    snapshot_ids_json = excluded.snapshot_ids_json,
                    status = excluded.status,
                    updated_at = excluded.updated_at
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
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
