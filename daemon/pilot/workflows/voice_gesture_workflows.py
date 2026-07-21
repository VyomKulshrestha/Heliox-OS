"""SQLite-backed durable store for voice/gesture-triggered multi-step workflows.

A sibling to WorkflowCheckpointStore (checkpoints.py), not a replacement —
that store still checkpoints each step's own ActionPlan (keyed
"{workflow_id}:{step_index}") exactly as it already does for resume_plan.
This store holds the workflow-level shape: the ordered list of steps, which
one is current, and whether the workflow is running, paused (explicit user
request), waiting-for-trigger (a step boundary awaiting the next voice/
gesture input), or terminal.

See pilot.agents.voice_gesture_workflow.VoiceGestureWorkflowEngine for the
execution loop that drives this store's state transitions.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiosqlite

from pilot import config as pilot_config
from pilot.security.gateway import TaskScopeOverride

WORKFLOW_DB_FILENAME = "voice_gesture_workflows.db"


class WorkflowState(StrEnum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_TRIGGER = "waiting_for_trigger"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class WorkflowStepRecord:
    index: int
    title: str
    description: str
    status: str = "pending"
    sub_plan_id: str = ""
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "sub_plan_id": self.sub_plan_id,
            "output": self.output,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStepRecord:
        return cls(
            index=int(data["index"]),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=data.get("status", "pending"),
            sub_plan_id=data.get("sub_plan_id", ""),
            output=data.get("output", ""),
            error=data.get("error", ""),
        )


@dataclass(frozen=True)
class VoiceGestureWorkflow:
    workflow_id: str
    goal: str
    invocation_source: str
    scope_override_json: str
    steps: list[WorkflowStepRecord]
    current_step: int
    state: str
    created_at: str
    updated_at: str
    paused_at: str | None
    trigger_deadline: str | None

    @property
    def scope_override(self) -> TaskScopeOverride | None:
        if not self.scope_override_json:
            return None
        return TaskScopeOverride.model_validate_json(self.scope_override_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "invocation_source": self.invocation_source,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "paused_at": self.paused_at,
            "trigger_deadline": self.trigger_deadline,
        }


class VoiceGestureWorkflowStore:
    """Persists workflow-level progress (steps, current index, state) —
    durable across daemon restarts, one row per workflow."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_file = Path(db_file) if db_file is not None else pilot_config.DATA_DIR / WORKFLOW_DB_FILENAME
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Every store method opens its own short-lived connection (no
        persistent connection to keep alive across daemon restarts).
        busy_timeout + synchronous are connection-local settings, cheap and
        safe to set every time, matching db/sqlite_pool.py's own connection
        settings -- so two connections writing near-simultaneously (e.g. a
        workflow's own background _drive task and an explicit cancel/pause
        call racing each other) wait for the lock instead of immediately
        raising "database is locked". journal_mode is deliberately NOT set
        here -- see initialize()."""
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute("PRAGMA synchronous = NORMAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            yield db

    async def initialize(self) -> None:
        """Switches the database file to WAL journal mode exactly once
        (guarded by _init_lock) rather than on every _connect() call.
        SQLite requires no OTHER connection be open at the moment journal
        mode actually changes -- unlike a normal write, that requirement
        is NOT satisfied by busy_timeout/retries, so re-issuing this PRAGMA
        on every short-lived connection raised "database is locked" under
        real concurrency (many workflows' connections overlapping) even
        though busy_timeout was set. Once WAL is on, it's persisted in the
        file's own header, so this only needs to run the first time."""
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
                    CREATE TABLE IF NOT EXISTS voice_gesture_workflows (
                        workflow_id TEXT PRIMARY KEY,
                        goal TEXT NOT NULL,
                        invocation_source TEXT NOT NULL,
                        scope_override_json TEXT NOT NULL,
                        steps_json TEXT NOT NULL,
                        current_step INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        paused_at TEXT,
                        trigger_deadline TEXT
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_voice_gesture_workflows_source_state
                    ON voice_gesture_workflows(invocation_source, state)
                    """
                )
                await db.commit()
            self._initialized = True

    async def create(
        self,
        goal: str,
        invocation_source: str,
        scope_override: TaskScopeOverride | None = None,
    ) -> VoiceGestureWorkflow:
        now = self._now()
        workflow = VoiceGestureWorkflow(
            workflow_id=uuid.uuid4().hex[:10],
            goal=goal,
            invocation_source=invocation_source,
            scope_override_json=scope_override.model_dump_json() if scope_override else "",
            steps=[],
            current_step=0,
            state=WorkflowState.PENDING.value,
            created_at=now,
            updated_at=now,
            paused_at=None,
            trigger_deadline=None,
        )
        await self._upsert(workflow)
        return workflow

    async def set_steps(self, workflow_id: str, steps: list[WorkflowStepRecord]) -> VoiceGestureWorkflow:
        workflow = await self._require(workflow_id)
        updated = self._replace(workflow, steps=steps, updated_at=self._now())
        await self._upsert(updated)
        return updated

    async def update_step(self, workflow_id: str, step_index: int, **fields: Any) -> VoiceGestureWorkflow:
        workflow = await self._require(workflow_id)
        steps = list(workflow.steps)
        for i, step in enumerate(steps):
            if step.index == step_index:
                for key, value in fields.items():
                    setattr(step, key, value)
                steps[i] = step
                break
        else:
            raise KeyError(f"No step {step_index} in workflow {workflow_id}")
        updated = self._replace(workflow, steps=steps, updated_at=self._now())
        await self._upsert(updated)
        return updated

    async def set_state(
        self,
        workflow_id: str,
        state: WorkflowState,
        *,
        current_step: int | None = None,
        trigger_deadline: str | None = None,
    ) -> VoiceGestureWorkflow | None:
        workflow = await self.get(workflow_id)
        if workflow is None:
            return None
        now = self._now()
        updated = self._replace(
            workflow,
            state=state.value,
            current_step=current_step if current_step is not None else workflow.current_step,
            updated_at=now,
            paused_at=now if state == WorkflowState.PAUSED else workflow.paused_at,
            trigger_deadline=trigger_deadline
            if trigger_deadline is not None
            else (
                None
                if state not in (WorkflowState.PAUSED, WorkflowState.WAITING_FOR_TRIGGER)
                else workflow.trigger_deadline
            ),
        )
        await self._upsert(updated)
        return updated

    async def get(self, workflow_id: str) -> VoiceGestureWorkflow | None:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT workflow_id, goal, invocation_source, scope_override_json, steps_json,
                       current_step, state, created_at, updated_at, paused_at, trigger_deadline
                FROM voice_gesture_workflows WHERE workflow_id = ?
                """,
                (workflow_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._from_row(row) if row else None

    async def list(self, include_terminal: bool = False) -> list[VoiceGestureWorkflow]:
        query = "SELECT workflow_id, goal, invocation_source, scope_override_json, steps_json, current_step, state, created_at, updated_at, paused_at, trigger_deadline FROM voice_gesture_workflows"
        if not include_terminal:
            query += " WHERE state IN ('pending','decomposing','running','paused','waiting_for_trigger')"
        query += " ORDER BY updated_at DESC"
        async with self._connect() as db:
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._from_row(row) for row in rows]

    async def find_pending_for_source(
        self, invocation_source: str, within_seconds: float
    ) -> VoiceGestureWorkflow | None:
        """Most-recently-updated PAUSED/WAITING_FOR_TRIGGER workflow for this
        source. Returns None if there isn't one, or if the most recent one's
        last update is older than `within_seconds` (a defensive fallback —
        the engine is expected to proactively transition stale rows to
        EXPIRED via its own trigger_deadline check, see VoiceGestureWorkflowEngine)."""
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT workflow_id, goal, invocation_source, scope_override_json, steps_json,
                       current_step, state, created_at, updated_at, paused_at, trigger_deadline
                FROM voice_gesture_workflows
                WHERE invocation_source = ? AND state IN ('paused', 'waiting_for_trigger')
                ORDER BY updated_at DESC LIMIT 1
                """,
                (invocation_source,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None:
            return None
        workflow = self._from_row(row)
        updated_at = datetime.fromisoformat(workflow.updated_at)
        age_seconds = (datetime.now(UTC) - updated_at).total_seconds()
        if age_seconds > within_seconds:
            return None
        return workflow

    async def _require(self, workflow_id: str) -> VoiceGestureWorkflow:
        workflow = await self.get(workflow_id)
        if workflow is None:
            raise KeyError(f"No workflow exists for workflow_id: {workflow_id}")
        return workflow

    @staticmethod
    def _replace(workflow: VoiceGestureWorkflow, **changes: Any) -> VoiceGestureWorkflow:
        data = {
            "workflow_id": workflow.workflow_id,
            "goal": workflow.goal,
            "invocation_source": workflow.invocation_source,
            "scope_override_json": workflow.scope_override_json,
            "steps": workflow.steps,
            "current_step": workflow.current_step,
            "state": workflow.state,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "paused_at": workflow.paused_at,
            "trigger_deadline": workflow.trigger_deadline,
        }
        data.update(changes)
        return VoiceGestureWorkflow(**data)

    async def _upsert(self, workflow: VoiceGestureWorkflow) -> None:
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO voice_gesture_workflows (
                    workflow_id, goal, invocation_source, scope_override_json, steps_json,
                    current_step, state, created_at, updated_at, paused_at, trigger_deadline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    goal = excluded.goal,
                    invocation_source = excluded.invocation_source,
                    scope_override_json = excluded.scope_override_json,
                    steps_json = excluded.steps_json,
                    current_step = excluded.current_step,
                    state = excluded.state,
                    updated_at = excluded.updated_at,
                    paused_at = excluded.paused_at,
                    trigger_deadline = excluded.trigger_deadline
                """,
                self._to_row(workflow),
            )
            await db.commit()

    @staticmethod
    def _to_row(workflow: VoiceGestureWorkflow) -> tuple[Any, ...]:
        return (
            workflow.workflow_id,
            workflow.goal,
            workflow.invocation_source,
            workflow.scope_override_json,
            json.dumps([s.to_dict() for s in workflow.steps]),
            workflow.current_step,
            workflow.state,
            workflow.created_at,
            workflow.updated_at,
            workflow.paused_at,
            workflow.trigger_deadline,
        )

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> VoiceGestureWorkflow:
        steps_raw = json.loads(row[4])
        return VoiceGestureWorkflow(
            workflow_id=row[0],
            goal=row[1],
            invocation_source=row[2],
            scope_override_json=row[3],
            steps=[WorkflowStepRecord.from_dict(s) for s in steps_raw],
            current_step=int(row[5]),
            state=row[6],
            created_at=row[7],
            updated_at=row[8],
            paused_at=row[9],
            trigger_deadline=row[10],
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
