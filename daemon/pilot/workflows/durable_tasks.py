"""Durable task, approval, and action-execution state.

The journal separates immutable transitions from mutable projections. Action
claims deliberately fail closed after an interrupted lease: Heliox must
reconcile the real-world effect before it can retry a possibly completed
side-effect.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pilot import config as pilot_config
from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.intelligence.experience import redact_for_persistence


class DurableTaskError(RuntimeError):
    """Base error for invalid durable-task operations."""


class InvalidTaskTransition(DurableTaskError):
    """Raised when a task state transition violates the state machine."""


class StaleTaskVersion(DurableTaskError):
    """Raised when an optimistic task update loses a race."""


class ApprovalConflict(DurableTaskError):
    """Raised when an approval is resolved differently more than once."""


class ActionClaimConflict(DurableTaskError):
    """Raised when a worker attempts to finish another worker's claim."""


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"
    SUPERSEDED = "superseded"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    DISCONNECTED = "disconnected"


class ActionExecutionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


class ActionClaimDecision(StrEnum):
    CLAIMED = "claimed"
    ALREADY_COMPLETED = "already_completed"
    BUSY = "busy"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    RETRY_REQUIRED = "retry_required"


TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.PARTIAL,
        TaskStatus.SUPERSEDED,
    }
)

_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.PLANNING, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.INTERRUPTED}),
    TaskStatus.PLANNING: frozenset(
        {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.EXECUTING,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.INTERRUPTED,
            TaskStatus.SUPERSEDED,
        }
    ),
    TaskStatus.AWAITING_APPROVAL: frozenset(
        {
            TaskStatus.EXECUTING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.EXECUTING: frozenset(
        {
            TaskStatus.VERIFYING,
            TaskStatus.PARTIAL,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.VERIFYING: frozenset(
        {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.EXECUTING,
            TaskStatus.SUCCEEDED,
            TaskStatus.PARTIAL,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.INTERRUPTED: frozenset(
        {
            TaskStatus.PLANNING,
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.EXECUTING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.PARTIAL: frozenset(),
    TaskStatus.SUPERSEDED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DurableTask:
    task_id: str
    session_id: str
    user_id: str
    user_input: str
    status: TaskStatus
    plan_id: str
    cancellation_requested: bool
    terminal_response: dict[str, Any] | None
    version: int
    created_at: str
    updated_at: str

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES


@dataclass(frozen=True, slots=True)
class CreatedTask:
    task: DurableTask
    resume_token: str


@dataclass(frozen=True, slots=True)
class DurableApproval:
    approval_id: str
    task_id: str
    plan_id: str
    status: ApprovalStatus
    request: dict[str, Any]
    approved_indices: list[int]
    expires_at: str
    created_at: str
    resolved_at: str


@dataclass(frozen=True, slots=True)
class ActionClaim:
    decision: ActionClaimDecision
    task_id: str
    plan_id: str
    action_key: str
    status: ActionExecutionStatus
    attempt_count: int
    result: dict[str, Any] | None = None
    lease_owner: str = ""
    lease_until: str = ""


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    interrupted_tasks: int
    uncertain_actions: int
    expired_approvals: int


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS durable_tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_input TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_id TEXT NOT NULL DEFAULT '',
    resume_token_hash TEXT NOT NULL UNIQUE,
    cancellation_requested INTEGER NOT NULL DEFAULT 0,
    terminal_response_json TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_durable_tasks_session_updated
    ON durable_tasks(session_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_durable_tasks_status_updated
    ON durable_tasks(status, updated_at);

CREATE TABLE IF NOT EXISTS task_transitions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES durable_tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_task_transitions_task_sequence
    ON task_transitions(task_id, sequence);

CREATE TRIGGER IF NOT EXISTS task_transitions_no_update
BEFORE UPDATE ON task_transitions
BEGIN
    SELECT RAISE(ABORT, 'task transitions are append-only');
END;

CREATE TRIGGER IF NOT EXISTS task_transitions_no_delete
BEFORE DELETE ON task_transitions
BEGIN
    SELECT RAISE(ABORT, 'task transitions are append-only');
END;

CREATE TABLE IF NOT EXISTS durable_approvals (
    approval_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    plan_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    approved_indices_json TEXT NOT NULL DEFAULT '[]',
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY(task_id) REFERENCES durable_tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_durable_approvals_task
    ON durable_approvals(task_id);
CREATE INDEX IF NOT EXISTS idx_durable_approvals_status_expiry
    ON durable_approvals(status, expires_at);

CREATE TABLE IF NOT EXISTS action_execution_claims (
    task_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    action_key TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    lease_owner TEXT NOT NULL,
    lease_until TEXT NOT NULL,
    result_json TEXT,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id, action_key),
    FOREIGN KEY(task_id) REFERENCES durable_tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_action_claims_status_lease
    ON action_execution_claims(status, lease_until);
"""


class DurableTaskStore:
    """SQLite journal used to resume tasks without replaying side effects."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_path = Path(db_file) if db_file is not None else pilot_config.DURABLE_TASK_DB_FILE
        self._pool: AsyncSqlitePool | None = None

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        pool = AsyncSqlitePool(self._db_path)
        await pool.start()
        async with pool.write() as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.executescript(_SCHEMA_SQL)
            await db.commit()
        self._pool = pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create_task(
        self,
        *,
        session_id: str,
        user_input: str,
        user_id: str = "local",
        task_id: str | None = None,
    ) -> CreatedTask:
        task_id = task_id or str(uuid.uuid4())
        resume_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(resume_token)
        now = self._now()
        safe_input = str(redact_for_persistence(user_input))
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    """
                    INSERT INTO durable_tasks (
                        task_id, session_id, user_id, user_input, status,
                        resume_token_hash, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        session_id,
                        user_id,
                        safe_input,
                        TaskStatus.QUEUED.value,
                        token_hash,
                        now,
                        now,
                    ),
                )
                await db.execute(
                    """
                    INSERT INTO task_transitions (
                        task_id, from_status, to_status, reason, occurred_at
                    )
                    VALUES (?, NULL, ?, ?, ?)
                    """,
                    (task_id, TaskStatus.QUEUED.value, "created", now),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        task = await self.get(task_id)
        if task is None:  # pragma: no cover - defensive storage invariant
            raise DurableTaskError(f"Created task disappeared: {task_id}")
        return CreatedTask(task=task, resume_token=resume_token)

    async def get(self, task_id: str) -> DurableTask | None:
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT task_id, session_id, user_id, user_input, status, plan_id,
                       cancellation_requested, terminal_response_json, version,
                       created_at, updated_at
                FROM durable_tasks
                WHERE task_id = ?
                """,
                (task_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._task_from_row(row) if row is not None else None

    async def get_by_resume_token(self, resume_token: str) -> DurableTask | None:
        token_hash = self._hash_token(resume_token)
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT task_id, session_id, user_id, user_input, status, plan_id,
                       cancellation_requested, terminal_response_json, version,
                       created_at, updated_at, resume_token_hash
                FROM durable_tasks
                WHERE resume_token_hash = ?
                """,
                (token_hash,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        if row is None or not hmac.compare_digest(row[11], token_hash):
            return None
        return self._task_from_row(row[:11])

    async def get_by_plan_id(self, plan_id: str) -> DurableTask | None:
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT task_id, session_id, user_id, user_input, status, plan_id,
                       cancellation_requested, terminal_response_json, version,
                       created_at, updated_at
                FROM durable_tasks
                WHERE plan_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (plan_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._task_from_row(row) if row is not None else None

    async def transition(
        self,
        task_id: str,
        to_status: TaskStatus,
        *,
        reason: str = "",
        plan_id: str | None = None,
        terminal_response: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> DurableTask:
        pool = self._require_pool()
        now = self._now()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT status, version, plan_id, terminal_response_json
                    FROM durable_tasks
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise KeyError(f"No durable task exists for task_id: {task_id}")
                from_status = TaskStatus(row[0])
                version = int(row[1])
                if expected_version is not None and version != expected_version:
                    raise StaleTaskVersion(f"Task {task_id} is version {version}, expected {expected_version}")
                if from_status == to_status:
                    await db.rollback()
                    task = await self.get(task_id)
                    if task is None:  # pragma: no cover - defensive
                        raise DurableTaskError(f"Task disappeared: {task_id}")
                    return task
                if to_status not in _ALLOWED_TRANSITIONS[from_status]:
                    raise InvalidTaskTransition(
                        f"Cannot transition task {task_id} from {from_status.value} to {to_status.value}"
                    )
                response_json = row[3]
                if terminal_response is not None:
                    response_json = json.dumps(
                        redact_for_persistence(terminal_response),
                        sort_keys=True,
                    )
                next_plan_id = row[2] if plan_id is None else plan_id
                await db.execute(
                    """
                    UPDATE durable_tasks
                    SET status = ?, plan_id = ?, terminal_response_json = ?,
                        version = version + 1, updated_at = ?
                    WHERE task_id = ? AND version = ?
                    """,
                    (
                        to_status.value,
                        next_plan_id,
                        response_json,
                        now,
                        task_id,
                        version,
                    ),
                )
                await db.execute(
                    """
                    INSERT INTO task_transitions (
                        task_id, from_status, to_status, reason, occurred_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        from_status.value,
                        to_status.value,
                        str(redact_for_persistence(reason)),
                        now,
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        task = await self.get(task_id)
        if task is None:  # pragma: no cover - defensive
            raise DurableTaskError(f"Task disappeared: {task_id}")
        return task

    async def request_cancel(self, task_id: str) -> DurableTask:
        pool = self._require_pool()
        async with pool.write() as db:
            cursor = await db.execute(
                """
                UPDATE durable_tasks
                SET cancellation_requested = 1, version = version + 1, updated_at = ?
                WHERE task_id = ?
                """,
                (self._now(), task_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No durable task exists for task_id: {task_id}")
            await cursor.close()
            await db.commit()
        task = await self.get(task_id)
        if task is None:  # pragma: no cover - defensive
            raise DurableTaskError(f"Task disappeared: {task_id}")
        return task

    async def create_approval(
        self,
        *,
        task_id: str,
        plan_id: str,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> DurableApproval:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=max(0.0, timeout_seconds))).isoformat()
        approval_id = str(uuid.uuid4())
        safe_request = json.dumps(redact_for_persistence(request), sort_keys=True)
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute(
                """
                INSERT INTO durable_approvals (
                    approval_id, task_id, plan_id, status, request_json,
                    approved_indices_json, expires_at, created_at, resolved_at
                )
                VALUES (?, ?, ?, ?, ?, '[]', ?, ?, '')
                ON CONFLICT(plan_id) DO NOTHING
                """,
                (
                    approval_id,
                    task_id,
                    plan_id,
                    ApprovalStatus.PENDING.value,
                    safe_request,
                    expires_at,
                    now,
                ),
            )
            await db.commit()
        approval = await self.get_approval(plan_id)
        if approval is None:  # pragma: no cover - defensive
            raise DurableTaskError(f"Approval disappeared for plan: {plan_id}")
        return approval

    async def get_approval(self, plan_id: str) -> DurableApproval | None:
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT approval_id, task_id, plan_id, status, request_json,
                       approved_indices_json, expires_at, created_at, resolved_at
                FROM durable_approvals
                WHERE plan_id = ?
                """,
                (plan_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._approval_from_row(row) if row is not None else None

    async def resolve_approval(
        self,
        plan_id: str,
        status: ApprovalStatus,
        *,
        approved_indices: list[int] | None = None,
    ) -> DurableApproval:
        if status == ApprovalStatus.PENDING:
            raise ValueError("An approval cannot be resolved back to pending")
        indices = sorted(set(approved_indices or []))
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT status, approved_indices_json
                    FROM durable_approvals
                    WHERE plan_id = ?
                    """,
                    (plan_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise KeyError(f"No durable approval exists for plan_id: {plan_id}")
                current = ApprovalStatus(row[0])
                current_indices = list(json.loads(row[1]))
                if current != ApprovalStatus.PENDING:
                    if current == status and current_indices == indices:
                        await db.rollback()
                        approval = await self.get_approval(plan_id)
                        if approval is None:  # pragma: no cover - defensive
                            raise DurableTaskError(f"Approval disappeared: {plan_id}")
                        return approval
                    raise ApprovalConflict(f"Approval {plan_id} was already resolved as {current.value}")
                await db.execute(
                    """
                    UPDATE durable_approvals
                    SET status = ?, approved_indices_json = ?, resolved_at = ?
                    WHERE plan_id = ? AND status = ?
                    """,
                    (
                        status.value,
                        json.dumps(indices),
                        self._now(),
                        plan_id,
                        ApprovalStatus.PENDING.value,
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        approval = await self.get_approval(plan_id)
        if approval is None:  # pragma: no cover - defensive
            raise DurableTaskError(f"Approval disappeared: {plan_id}")
        return approval

    async def claim_action(
        self,
        *,
        task_id: str,
        plan_id: str,
        action_key: str,
        lease_owner: str,
        lease_seconds: float = 60.0,
        allow_retry: bool = False,
    ) -> ActionClaim:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        lease_until = (now_dt + timedelta(seconds=max(1.0, lease_seconds))).isoformat()
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT status, attempt_count, lease_owner, lease_until, result_json
                    FROM action_execution_claims
                    WHERE task_id = ? AND action_key = ?
                    """,
                    (task_id, action_key),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    await db.execute(
                        """
                        INSERT INTO action_execution_claims (
                            task_id, plan_id, action_key, status, attempt_count,
                            lease_owner, lease_until, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            plan_id,
                            action_key,
                            ActionExecutionStatus.RUNNING.value,
                            lease_owner,
                            lease_until,
                            now,
                            now,
                        ),
                    )
                    await db.commit()
                    return ActionClaim(
                        decision=ActionClaimDecision.CLAIMED,
                        task_id=task_id,
                        plan_id=plan_id,
                        action_key=action_key,
                        status=ActionExecutionStatus.RUNNING,
                        attempt_count=1,
                        lease_owner=lease_owner,
                        lease_until=lease_until,
                    )

                status = ActionExecutionStatus(row[0])
                attempts = int(row[1])
                if status == ActionExecutionStatus.COMPLETED:
                    await db.rollback()
                    return ActionClaim(
                        decision=ActionClaimDecision.ALREADY_COMPLETED,
                        task_id=task_id,
                        plan_id=plan_id,
                        action_key=action_key,
                        status=status,
                        attempt_count=attempts,
                        result=json.loads(row[4]) if row[4] else None,
                    )
                if status == ActionExecutionStatus.RUNNING:
                    if row[3] > now:
                        await db.rollback()
                        return ActionClaim(
                            decision=ActionClaimDecision.BUSY,
                            task_id=task_id,
                            plan_id=plan_id,
                            action_key=action_key,
                            status=status,
                            attempt_count=attempts,
                            lease_owner=row[2],
                            lease_until=row[3],
                        )
                    await db.execute(
                        """
                        UPDATE action_execution_claims
                        SET status = ?, lease_owner = '', lease_until = '', updated_at = ?
                        WHERE task_id = ? AND action_key = ?
                        """,
                        (
                            ActionExecutionStatus.UNCERTAIN.value,
                            now,
                            task_id,
                            action_key,
                        ),
                    )
                    await db.commit()
                    return ActionClaim(
                        decision=ActionClaimDecision.RECONCILIATION_REQUIRED,
                        task_id=task_id,
                        plan_id=plan_id,
                        action_key=action_key,
                        status=ActionExecutionStatus.UNCERTAIN,
                        attempt_count=attempts,
                    )
                if status == ActionExecutionStatus.UNCERTAIN:
                    await db.rollback()
                    return ActionClaim(
                        decision=ActionClaimDecision.RECONCILIATION_REQUIRED,
                        task_id=task_id,
                        plan_id=plan_id,
                        action_key=action_key,
                        status=status,
                        attempt_count=attempts,
                    )
                if not allow_retry:
                    await db.rollback()
                    return ActionClaim(
                        decision=ActionClaimDecision.RETRY_REQUIRED,
                        task_id=task_id,
                        plan_id=plan_id,
                        action_key=action_key,
                        status=status,
                        attempt_count=attempts,
                    )
                await db.execute(
                    """
                    UPDATE action_execution_claims
                    SET status = ?, attempt_count = attempt_count + 1,
                        lease_owner = ?, lease_until = ?, result_json = NULL,
                        error = '', updated_at = ?
                    WHERE task_id = ? AND action_key = ?
                    """,
                    (
                        ActionExecutionStatus.RUNNING.value,
                        lease_owner,
                        lease_until,
                        now,
                        task_id,
                        action_key,
                    ),
                )
                await db.commit()
                return ActionClaim(
                    decision=ActionClaimDecision.CLAIMED,
                    task_id=task_id,
                    plan_id=plan_id,
                    action_key=action_key,
                    status=ActionExecutionStatus.RUNNING,
                    attempt_count=attempts + 1,
                    lease_owner=lease_owner,
                    lease_until=lease_until,
                )
            except Exception:
                await db.rollback()
                raise

    async def finish_action(
        self,
        *,
        task_id: str,
        action_key: str,
        lease_owner: str,
        status: ActionExecutionStatus,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> ActionClaim:
        if status not in {
            ActionExecutionStatus.COMPLETED,
            ActionExecutionStatus.FAILED,
            ActionExecutionStatus.CANCELLED,
        }:
            raise ValueError(f"Cannot finish an action with status: {status.value}")
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT plan_id, status, attempt_count, lease_owner
                    FROM action_execution_claims
                    WHERE task_id = ? AND action_key = ?
                    """,
                    (task_id, action_key),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise KeyError(f"No action claim exists for key: {action_key}")
                if row[1] != ActionExecutionStatus.RUNNING.value or not hmac.compare_digest(row[3], lease_owner):
                    raise ActionClaimConflict(f"Worker {lease_owner!r} does not own running action {action_key}")
                safe_result = json.dumps(redact_for_persistence(result), sort_keys=True) if result is not None else None
                await db.execute(
                    """
                    UPDATE action_execution_claims
                    SET status = ?, result_json = ?, error = ?,
                        lease_owner = '', lease_until = '', updated_at = ?
                    WHERE task_id = ? AND action_key = ?
                    """,
                    (
                        status.value,
                        safe_result,
                        str(redact_for_persistence(error)),
                        self._now(),
                        task_id,
                        action_key,
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return ActionClaim(
            decision=(
                ActionClaimDecision.ALREADY_COMPLETED
                if status == ActionExecutionStatus.COMPLETED
                else ActionClaimDecision.RETRY_REQUIRED
            ),
            task_id=task_id,
            plan_id=row[0],
            action_key=action_key,
            status=status,
            attempt_count=int(row[2]),
            result=redact_for_persistence(result) if result is not None else None,
        )

    async def mark_action_uncertain(
        self,
        *,
        task_id: str,
        action_key: str,
        lease_owner: str,
        error: str = "",
    ) -> None:
        """Release a claim only into the fail-closed reconciliation state."""

        pool = self._require_pool()
        async with pool.write() as db:
            cursor = await db.execute(
                """
                UPDATE action_execution_claims
                SET status = ?, lease_owner = '', lease_until = '', error = ?,
                    updated_at = ?
                WHERE task_id = ? AND action_key = ? AND status = ?
                      AND lease_owner = ?
                """,
                (
                    ActionExecutionStatus.UNCERTAIN.value,
                    str(redact_for_persistence(error)),
                    self._now(),
                    task_id,
                    action_key,
                    ActionExecutionStatus.RUNNING.value,
                    lease_owner,
                ),
            )
            if cursor.rowcount == 0:
                raise ActionClaimConflict(f"Worker {lease_owner!r} does not own running action {action_key}")
            await cursor.close()
            await db.commit()

    async def recover_incomplete(self) -> RecoverySummary:
        """Recover after daemon startup without repeating uncertain effects."""

        now = self._now()
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT task_id, status
                    FROM durable_tasks
                    WHERE status IN (?, ?, ?, ?)
                    """,
                    (
                        TaskStatus.QUEUED.value,
                        TaskStatus.PLANNING.value,
                        TaskStatus.EXECUTING.value,
                        TaskStatus.VERIFYING.value,
                    ),
                )
                interrupted = await cursor.fetchall()
                await cursor.close()
                for task_id, old_status in interrupted:
                    await db.execute(
                        """
                        UPDATE durable_tasks
                        SET status = ?, version = version + 1, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (TaskStatus.INTERRUPTED.value, now, task_id),
                    )
                    await db.execute(
                        """
                        INSERT INTO task_transitions (
                            task_id, from_status, to_status, reason, occurred_at
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            old_status,
                            TaskStatus.INTERRUPTED.value,
                            "daemon_recovery",
                            now,
                        ),
                    )
                action_cursor = await db.execute(
                    """
                    UPDATE action_execution_claims
                    SET status = ?, lease_owner = '', lease_until = '', updated_at = ?
                    WHERE status = ?
                    """,
                    (
                        ActionExecutionStatus.UNCERTAIN.value,
                        now,
                        ActionExecutionStatus.RUNNING.value,
                    ),
                )
                uncertain_actions = action_cursor.rowcount
                await action_cursor.close()
                approval_cursor = await db.execute(
                    """
                    UPDATE durable_approvals
                    SET status = ?, resolved_at = ?
                    WHERE status = ? AND expires_at <= ?
                    """,
                    (
                        ApprovalStatus.EXPIRED.value,
                        now,
                        ApprovalStatus.PENDING.value,
                        now,
                    ),
                )
                expired_approvals = approval_cursor.rowcount
                await approval_cursor.close()
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return RecoverySummary(
            interrupted_tasks=len(interrupted),
            uncertain_actions=uncertain_actions,
            expired_approvals=expired_approvals,
        )

    def _require_pool(self) -> AsyncSqlitePool:
        if self._pool is None:
            raise RuntimeError("DurableTaskStore is not initialized")
        return self._pool

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _task_from_row(row: tuple[Any, ...]) -> DurableTask:
        return DurableTask(
            task_id=row[0],
            session_id=row[1],
            user_id=row[2],
            user_input=row[3],
            status=TaskStatus(row[4]),
            plan_id=row[5],
            cancellation_requested=bool(row[6]),
            terminal_response=json.loads(row[7]) if row[7] else None,
            version=int(row[8]),
            created_at=row[9],
            updated_at=row[10],
        )

    @staticmethod
    def _approval_from_row(row: tuple[Any, ...]) -> DurableApproval:
        return DurableApproval(
            approval_id=row[0],
            task_id=row[1],
            plan_id=row[2],
            status=ApprovalStatus(row[3]),
            request=dict(json.loads(row[4])),
            approved_indices=list(json.loads(row[5])),
            expires_at=row[6],
            created_at=row[7],
            resolved_at=row[8],
        )
