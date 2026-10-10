"""Typed, append-only experience ledger for Heliox intelligence systems."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pilot.db.sqlite_pool import AsyncSqlitePool

SCHEMA_VERSION = 1
logger = logging.getLogger("pilot.intelligence.experience")


class ExperienceEventType(StrEnum):
    """Canonical event vocabulary shared by every intelligence component."""

    OBSERVATION = "observation"
    INTENT = "intent"
    PLAN_CREATED = "plan_created"
    CANDIDATE_ACTION = "candidate_action"
    WORLD_PREDICTION = "world_prediction"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    OUTCOME_VERIFIED = "outcome_verified"
    PREDICTION_ERROR = "prediction_error"
    USER_CORRECTION = "user_correction"
    SUGGESTION_SHOWN = "suggestion_shown"
    SUGGESTION_FEEDBACK = "suggestion_feedback"
    MEMORY_PROMOTED = "memory_promoted"
    STRATEGY_CANDIDATE = "strategy_candidate"
    EVOLUTION_CANDIDATE = "evolution_candidate"
    EVOLUTION_EVALUATION = "evolution_evaluation"
    EVOLUTION_PROMOTION_REQUEST = "evolution_promotion_request"
    AGENT_MESH_OUTCOME = "agent_mesh_outcome"
    AGENT_MESH_HANDOFF = "agent_mesh_handoff"


class PrivacyClass(StrEnum):
    """Storage sensitivity attached to each event."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    BIOMETRIC_DERIVED = "biometric_derived"


@dataclass(frozen=True, slots=True)
class ExperienceContext:
    """Causal identity propagated safely across asynchronous task boundaries."""

    session_id: str = ""
    task_id: str = ""
    user_id: str = "local"
    parent_event_id: str = ""
    plan_id: str = ""
    action_index_offset: int = 0


_CURRENT_CONTEXT: ContextVar[ExperienceContext | None] = ContextVar(
    "heliox_experience_context",
    default=None,
)


@contextmanager
def experience_scope(
    *,
    session_id: str = "",
    task_id: str = "",
    user_id: str = "local",
    parent_event_id: str = "",
    plan_id: str = "",
    action_index_offset: int | None = None,
) -> Iterator[ExperienceContext]:
    """Propagate ledger identity through nested agents and executor tasks."""

    inherited = _CURRENT_CONTEXT.get() or ExperienceContext()
    context = ExperienceContext(
        session_id=session_id or inherited.session_id,
        task_id=task_id or inherited.task_id,
        user_id=user_id or inherited.user_id,
        parent_event_id=parent_event_id or inherited.parent_event_id,
        plan_id=plan_id or inherited.plan_id,
        action_index_offset=(inherited.action_index_offset if action_index_offset is None else action_index_offset),
    )
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


def get_experience_context() -> ExperienceContext:
    """Return the experience context active in this async execution tree."""

    return _CURRENT_CONTEXT.get() or ExperienceContext()


_SECRET_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "cookie",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
_RAW_MEDIA_KEYS = {
    "audio",
    "audio_bytes",
    "audio_data",
    "camera_frame",
    "face_landmarks",
    "frame",
    "hand_landmarks",
    "image",
    "image_bytes",
    "image_data",
    "electrode_data",
    "eeg",
    "eeg_samples",
    "neural_samples",
    "raw_audio",
    "raw_camera",
    "raw_screen",
    "raw_eeg",
    "screen_capture",
    "screenshot",
    "video",
    "waveform",
}
_TOKEN_PATTERNS = (
    re.compile(r"\b(?:sk|gh[opsu]|github_pat)_[A-Za-z0-9_-]{12,}\b"),
    # OpenAI / Anthropic / OpenRouter style keys: sk-..., sk-ant-..., sk-or-...
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b"),
    # key=value / key: value assignments of credentials in logs and tool output
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)\b\s*[:=]\s*[\"']?[^\s\"',;]{6,}"
    ),
)


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_for_persistence(value: Any) -> Any:
    """Remove credentials and raw sensor media before anything reaches SQLite."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalized_key(key)
            if normalized in _RAW_MEDIA_KEYS or normalized.startswith(
                ("raw_audio", "raw_camera", "raw_screen", "raw_eeg", "raw_neural")
            ):
                redacted[str(key)] = "[EXCLUDED_RAW_MEDIA]"
            elif normalized in _SECRET_KEY_PARTS or any(normalized.endswith(f"_{part}") for part in _SECRET_KEY_PARTS):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_for_persistence(item)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_for_persistence(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[EXCLUDED_BINARY]"
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    """One immutable event from the canonical experience stream."""

    event_id: str
    sequence: int
    event_type: ExperienceEventType
    occurred_at: str
    schema_version: int
    session_id: str
    task_id: str
    user_id: str
    plan_id: str
    action_id: str
    parent_event_id: str
    idempotency_key: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    privacy_class: PrivacyClass = PrivacyClass.INTERNAL


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experience_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    parent_event_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    confidence REAL,
    privacy_class TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experience_session_sequence
    ON experience_events(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_experience_task_sequence
    ON experience_events(task_id, sequence);
CREATE INDEX IF NOT EXISTS idx_experience_plan_sequence
    ON experience_events(plan_id, sequence);
CREATE INDEX IF NOT EXISTS idx_experience_action_sequence
    ON experience_events(action_id, sequence);
CREATE INDEX IF NOT EXISTS idx_experience_type_sequence
    ON experience_events(event_type, sequence);

CREATE TRIGGER IF NOT EXISTS experience_events_no_update
BEFORE UPDATE ON experience_events
BEGIN
    SELECT RAISE(ABORT, 'experience ledger is append-only');
END;

CREATE TRIGGER IF NOT EXISTS experience_events_no_delete
BEFORE DELETE ON experience_events
BEGIN
    SELECT RAISE(ABORT, 'experience ledger is append-only');
END;
"""

_SELECT_COLUMNS = """
sequence, event_id, event_type, occurred_at, schema_version, session_id,
task_id, user_id, plan_id, action_id, parent_event_id, idempotency_key,
source, payload_json, provenance_json, confidence, privacy_class
"""


def stable_action_idempotency_key(plan_id: str, index: int, action: Any) -> str:
    """Build the stable identity used to detect duplicate action execution."""

    if hasattr(action, "model_dump"):
        action = action.model_dump(mode="json")
    canonical = json.dumps(redact_for_persistence(action), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{plan_id}:{index}:{canonical}".encode()).hexdigest()
    return f"action:{digest}"


class ExperienceLedger:
    """Append-only SQLite event store with idempotent inserts and redaction."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._pool: AsyncSqlitePool | None = None
        self._subscribers: list[Callable[[ExperienceEvent], Awaitable[None]]] = []
        self._subscriber_tasks: set[asyncio.Task[None]] = set()

    def subscribe(self, callback: Callable[[ExperienceEvent], Awaitable[None]]) -> None:
        """Receive newly inserted events without gaining mutation authority."""

        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ExperienceEvent], Awaitable[None]]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        pool = AsyncSqlitePool(self._db_path)
        await pool.start()
        async with pool.write() as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
        self._pool = pool

    async def append(
        self,
        event_type: ExperienceEventType | str,
        *,
        payload: Mapping[str, Any] | None = None,
        session_id: str = "",
        task_id: str = "",
        user_id: str = "",
        plan_id: str = "",
        action_id: str = "",
        parent_event_id: str = "",
        idempotency_key: str = "",
        source: str = "daemon",
        provenance: Mapping[str, Any] | None = None,
        confidence: float | None = None,
        privacy_class: PrivacyClass | str = PrivacyClass.INTERNAL,
        occurred_at: str = "",
    ) -> ExperienceEvent:
        """Redact and append one event, returning the existing row on retry."""

        pool = self._require_pool()
        event_kind = ExperienceEventType(event_type)
        privacy = PrivacyClass(privacy_class)
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        context = get_experience_context()
        event_id = str(uuid.uuid4())
        event_key = idempotency_key or f"event:{event_id}"
        values = (
            event_id,
            event_kind.value,
            occurred_at or datetime.now(timezone.utc).isoformat(),
            SCHEMA_VERSION,
            session_id or context.session_id or "system",
            task_id or context.task_id or plan_id or "unscoped",
            user_id or context.user_id or "local",
            plan_id,
            action_id,
            parent_event_id or context.parent_event_id,
            event_key,
            source,
            json.dumps(redact_for_persistence(dict(payload or {})), sort_keys=True, separators=(",", ":")),
            json.dumps(
                redact_for_persistence(dict(provenance or {})),
                sort_keys=True,
                separators=(",", ":"),
            ),
            confidence,
            privacy.value,
        )
        async with pool.write() as db:
            cursor = await db.execute(
                """
                INSERT OR IGNORE INTO experience_events (
                    event_id, event_type, occurred_at, schema_version,
                    session_id, task_id, user_id, plan_id, action_id,
                    parent_event_id, idempotency_key, source, payload_json,
                    provenance_json, confidence, privacy_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            inserted = cursor.rowcount == 1
            await cursor.close()
            if inserted:
                sequence_cursor = await db.execute("SELECT last_insert_rowid()")
                sequence_row = await sequence_cursor.fetchone()
                await sequence_cursor.close()
                await db.commit()
                event = ExperienceEvent(
                    event_id=event_id,
                    sequence=int(sequence_row[0]),
                    event_type=event_kind,
                    occurred_at=values[2],
                    schema_version=SCHEMA_VERSION,
                    session_id=values[4],
                    task_id=values[5],
                    user_id=values[6],
                    plan_id=plan_id,
                    action_id=action_id,
                    parent_event_id=values[9],
                    idempotency_key=event_key,
                    source=source,
                    payload=json.loads(values[12]),
                    provenance=json.loads(values[13]),
                    confidence=confidence,
                    privacy_class=privacy,
                )
                self._publish(event)
                return event

            existing_cursor = await db.execute(
                f"SELECT {_SELECT_COLUMNS} FROM experience_events WHERE idempotency_key = ?",
                (event_key,),
            )
            row = await existing_cursor.fetchone()
            await existing_cursor.close()
            await db.commit()
        if row is None:
            raise RuntimeError("Experience event was not inserted and could not be recovered")
        existing = self._event_from_row(row)
        expected_identity = (
            event_kind,
            values[4],
            values[5],
            values[6],
            plan_id,
            action_id,
        )
        existing_identity = (
            existing.event_type,
            existing.session_id,
            existing.task_id,
            existing.user_id,
            existing.plan_id,
            existing.action_id,
        )
        if existing_identity != expected_identity:
            raise ValueError(f"idempotency key {event_key!r} is already bound to a different event identity")
        return existing

    async def list_events(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        plan_id: str = "",
        action_id: str = "",
        event_type: ExperienceEventType | str | None = None,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> list[ExperienceEvent]:
        """Read an ordered trace without exposing mutation operations."""

        pool = self._require_pool()
        clauses = ["sequence > ?"]
        params: list[Any] = [max(0, after_sequence)]
        for column, value in (
            ("session_id", session_id),
            ("task_id", task_id),
            ("plan_id", plan_id),
            ("action_id", action_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(ExperienceEventType(event_type).value)
        params.append(max(1, min(limit, 1000)))
        query = (
            f"SELECT {_SELECT_COLUMNS} FROM experience_events "
            f"WHERE {' AND '.join(clauses)} ORDER BY sequence ASC LIMIT ?"
        )
        async with pool.read() as db:
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._event_from_row(row) for row in rows]

    async def close(self) -> None:
        await self.drain_subscribers()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def drain_subscribers(self) -> None:
        """Wait for advisory consumers without allowing their failure to block shutdown."""

        if self._subscriber_tasks:
            await asyncio.gather(*tuple(self._subscriber_tasks), return_exceptions=True)

    def _publish(self, event: ExperienceEvent) -> None:
        for callback in tuple(self._subscribers):
            task = asyncio.create_task(callback(event))
            self._subscriber_tasks.add(task)
            task.add_done_callback(self._subscriber_done)

    def _subscriber_done(self, task: asyncio.Task[None]) -> None:
        self._subscriber_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.warning("Experience subscriber failed", exc_info=True)

    def _require_pool(self) -> AsyncSqlitePool:
        if self._pool is None:
            raise RuntimeError("ExperienceLedger is not initialized")
        return self._pool

    @staticmethod
    def _event_from_row(row: Sequence[Any]) -> ExperienceEvent:
        return ExperienceEvent(
            sequence=int(row[0]),
            event_id=str(row[1]),
            event_type=ExperienceEventType(row[2]),
            occurred_at=str(row[3]),
            schema_version=int(row[4]),
            session_id=str(row[5]),
            task_id=str(row[6]),
            user_id=str(row[7]),
            plan_id=str(row[8]),
            action_id=str(row[9]),
            parent_event_id=str(row[10]),
            idempotency_key=str(row[11]),
            source=str(row[12]),
            payload=json.loads(row[13]),
            provenance=json.loads(row[14]),
            confidence=None if row[15] is None else float(row[15]),
            privacy_class=PrivacyClass(row[16]),
        )
