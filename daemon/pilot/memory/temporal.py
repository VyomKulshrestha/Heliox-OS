"""Temporal memory graph with provenance, evidence, and validity windows."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pilot import config as pilot_config
from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.intelligence.experience import PrivacyClass, redact_for_persistence


class MemoryScope(StrEnum):
    USER = "user"
    SESSION = "session"
    TASK = "task"
    SYSTEM = "system"


class MemoryProvenance(StrEnum):
    EXPLICIT_USER = "explicit_user"
    VERIFIED_OUTCOME = "verified_outcome"
    SYSTEM_OBSERVATION = "system_observation"
    REPEATED_BEHAVIOR = "repeated_behavior"
    INFERRED = "inferred"
    MODEL_SYNTHESIS = "model_synthesis"


class FactStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


PROVENANCE_WEIGHT: dict[MemoryProvenance, float] = {
    MemoryProvenance.EXPLICIT_USER: 1.0,
    MemoryProvenance.VERIFIED_OUTCOME: 0.95,
    MemoryProvenance.SYSTEM_OBSERVATION: 0.85,
    MemoryProvenance.REPEATED_BEHAVIOR: 0.7,
    MemoryProvenance.INFERRED: 0.45,
    MemoryProvenance.MODEL_SYNTHESIS: 0.3,
}


@dataclass(frozen=True, slots=True)
class TemporalFact:
    fact_id: str
    subject: str
    predicate: str
    value: Any
    scope: MemoryScope
    session_id: str
    task_id: str
    status: FactStatus
    confidence: float
    provenance: MemoryProvenance
    evidence_count: int
    valid_from: str
    valid_until: str
    updated_at: str
    privacy_class: PrivacyClass


@dataclass(frozen=True, slots=True)
class EpisodicMemory:
    episode_id: str
    session_id: str
    task_id: str
    summary: str
    outcome: str
    tags: list[str]
    importance: float
    occurred_at: str
    provenance_event_id: str


@dataclass(frozen=True, slots=True)
class WorkingMemory:
    session_id: str
    task_id: str
    key: str
    value: Any
    priority: float
    expires_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class RankedMemory:
    kind: str
    text: str
    score: float
    confidence: float
    provenance: str
    memory_id: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS temporal_facts (
    fact_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    provenance TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    privacy_class TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_temporal_facts_lookup
    ON temporal_facts(subject, predicate, scope, status);
CREATE INDEX IF NOT EXISTS idx_temporal_facts_session
    ON temporal_facts(session_id, status, updated_at);

CREATE TABLE IF NOT EXISTS fact_evidence (
    evidence_id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL,
    event_id TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL,
    provenance TEXT NOT NULL,
    confidence REAL NOT NULL,
    observed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(fact_id) REFERENCES temporal_facts(fact_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_evidence_fact
    ON fact_evidence(fact_id, observed_at);

CREATE TRIGGER IF NOT EXISTS fact_evidence_no_update
BEFORE UPDATE ON fact_evidence
BEGIN
    SELECT RAISE(ABORT, 'fact evidence is append-only');
END;

CREATE TRIGGER IF NOT EXISTS fact_evidence_no_delete
BEFORE DELETE ON fact_evidence
BEGIN
    SELECT RAISE(ABORT, 'fact evidence is append-only');
END;

CREATE TABLE IF NOT EXISTS fact_contradictions (
    contradiction_id TEXT PRIMARY KEY,
    prior_fact_id TEXT NOT NULL,
    new_fact_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    resolution TEXT NOT NULL,
    FOREIGN KEY(prior_fact_id) REFERENCES temporal_facts(fact_id),
    FOREIGN KEY(new_fact_id) REFERENCES temporal_facts(fact_id)
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance REAL NOT NULL,
    occurred_at TEXT NOT NULL,
    provenance_event_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'inferred'
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_time
    ON episodic_memories(session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome_time
    ON episodic_memories(outcome, occurred_at);

CREATE TABLE IF NOT EXISTS working_memory (
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    priority REAL NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(session_id, task_id, key)
);

CREATE INDEX IF NOT EXISTS idx_working_memory_expiry
    ON working_memory(expires_at);
"""


class TemporalMemoryStore:
    """Stores changing facts without erasing their evidence or history."""

    def __init__(self, db_file: str | Path | None = None) -> None:
        self._db_path = Path(db_file) if db_file is not None else pilot_config.TEMPORAL_MEMORY_DB_FILE
        self._pool: AsyncSqlitePool | None = None

    async def initialize(self) -> None:
        if self._pool is not None:
            return
        pool = AsyncSqlitePool(self._db_path)
        await pool.start()
        async with pool.write() as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.executescript(_SCHEMA_SQL)
            cursor = await db.execute("PRAGMA table_info(episodic_memories)")
            columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()
            if "provenance" not in columns:
                await db.execute("ALTER TABLE episodic_memories ADD COLUMN provenance TEXT NOT NULL DEFAULT 'inferred'")
            await db.commit()
        self._pool = pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def remember_fact(
        self,
        *,
        subject: str,
        predicate: str,
        value: Any,
        scope: MemoryScope = MemoryScope.USER,
        session_id: str = "",
        task_id: str = "",
        confidence: float = 0.5,
        provenance: MemoryProvenance = MemoryProvenance.INFERRED,
        event_id: str = "",
        valid_from: str | None = None,
        privacy_class: PrivacyClass = PrivacyClass.SENSITIVE,
        evidence_payload: dict[str, Any] | None = None,
    ) -> TemporalFact:
        normalized_subject = self._normalize_term(subject)
        normalized_predicate = self._normalize_term(predicate)
        if not normalized_subject or not normalized_predicate:
            raise ValueError("subject and predicate are required")
        confidence = max(0.0, min(1.0, float(confidence)))
        safe_value = redact_for_persistence(value)
        value_json = json.dumps(
            safe_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        now = self._now()
        fact_valid_from = valid_from or now
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT fact_id, value_json, status, confidence, provenance,
                           evidence_count
                    FROM temporal_facts
                    WHERE subject = ? AND predicate = ? AND scope = ?
                      AND session_id = ? AND task_id = ?
                      AND status IN (?, ?)
                    ORDER BY updated_at DESC
                    """,
                    (
                        normalized_subject,
                        normalized_predicate,
                        scope.value,
                        session_id,
                        task_id,
                        FactStatus.ACTIVE.value,
                        FactStatus.CANDIDATE.value,
                    ),
                )
                current_rows = await cursor.fetchall()
                await cursor.close()
                matching = next((row for row in current_rows if row[1] == value_json), None)
                if matching is not None:
                    fact_id = matching[0]
                    evidence_count = int(matching[5]) + 1
                    combined_confidence = self._combine_confidence(
                        float(matching[3]),
                        confidence,
                        provenance,
                    )
                    next_status = FactStatus(matching[2])
                    if self._eligible_for_activation(
                        provenance,
                        evidence_count,
                        combined_confidence,
                    ):
                        next_status = FactStatus.ACTIVE
                        await self._supersede_conflicts(
                            db,
                            fact_id=fact_id,
                            rows=current_rows,
                            now=now,
                        )
                        await db.execute(
                            """
                            UPDATE fact_contradictions
                            SET resolution = 'candidate_promoted'
                            WHERE new_fact_id = ?
                              AND resolution = 'candidate_requires_more_evidence'
                            """,
                            (fact_id,),
                        )
                    await db.execute(
                        """
                        UPDATE temporal_facts
                        SET status = ?, confidence = ?, provenance = ?,
                            evidence_count = ?, updated_at = ?
                        WHERE fact_id = ?
                        """,
                        (
                            next_status.value,
                            combined_confidence,
                            provenance.value,
                            evidence_count,
                            now,
                            fact_id,
                        ),
                    )
                else:
                    fact_id = str(uuid.uuid4())
                    evidence_count = 1
                    next_status = (
                        FactStatus.ACTIVE
                        if self._eligible_for_activation(
                            provenance,
                            evidence_count,
                            confidence,
                        )
                        else FactStatus.CANDIDATE
                    )
                    await db.execute(
                        """
                        INSERT INTO temporal_facts (
                            fact_id, subject, predicate, value_json, scope,
                            session_id, task_id, status, confidence, provenance,
                            evidence_count, valid_from, created_at, updated_at,
                            privacy_class
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fact_id,
                            normalized_subject,
                            normalized_predicate,
                            value_json,
                            scope.value,
                            session_id,
                            task_id,
                            next_status.value,
                            confidence,
                            provenance.value,
                            evidence_count,
                            fact_valid_from,
                            now,
                            now,
                            privacy_class.value,
                        ),
                    )
                    for row in current_rows:
                        if row[1] == value_json:
                            continue
                        resolution = (
                            "new_fact_activated"
                            if next_status == FactStatus.ACTIVE
                            else "candidate_requires_more_evidence"
                        )
                        await db.execute(
                            """
                            INSERT INTO fact_contradictions (
                                contradiction_id, prior_fact_id, new_fact_id,
                                observed_at, resolution
                            )
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (str(uuid.uuid4()), row[0], fact_id, now, resolution),
                        )
                    if next_status == FactStatus.ACTIVE:
                        await self._supersede_conflicts(
                            db,
                            fact_id=fact_id,
                            rows=current_rows,
                            now=now,
                        )
                await db.execute(
                    """
                    INSERT INTO fact_evidence (
                        evidence_id, fact_id, event_id, direction, provenance,
                        confidence, observed_at, payload_json
                    )
                    VALUES (?, ?, ?, 'supports', ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        fact_id,
                        event_id,
                        provenance.value,
                        confidence,
                        now,
                        json.dumps(
                            redact_for_persistence(evidence_payload or {}),
                            sort_keys=True,
                        ),
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        fact = await self.get_fact(fact_id)
        if fact is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Fact disappeared after write: {fact_id}")
        return fact

    async def get_fact(self, fact_id: str) -> TemporalFact | None:
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT fact_id, subject, predicate, value_json, scope, session_id,
                       task_id, status, confidence, provenance, evidence_count,
                       valid_from, valid_until, updated_at, privacy_class
                FROM temporal_facts
                WHERE fact_id = ?
                """,
                (fact_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._fact_from_row(row) if row is not None else None

    async def list_facts(
        self,
        *,
        statuses: tuple[FactStatus, ...] = (
            FactStatus.ACTIVE,
            FactStatus.CANDIDATE,
        ),
        limit: int = 100,
    ) -> list[TemporalFact]:
        """List reviewable facts without treating candidates as active context."""
        if not statuses or limit <= 0:
            return []
        pool = self._require_pool()
        placeholders = ",".join("?" for _ in statuses)
        async with pool.read() as db:
            cursor = await db.execute(
                f"""
                SELECT fact_id, subject, predicate, value_json, scope, session_id,
                       task_id, status, confidence, provenance, evidence_count,
                       valid_from, valid_until, updated_at, privacy_class
                FROM temporal_facts
                WHERE status IN ({placeholders})
                ORDER BY
                    CASE status WHEN 'active' THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (*[status.value for status in statuses], min(500, int(limit))),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._fact_from_row(row) for row in rows]

    async def stats(self) -> dict[str, Any]:
        """Return non-sensitive counts for product status and diagnostics."""
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute("SELECT status, COUNT(*) FROM temporal_facts GROUP BY status")
            status_counts = {str(row[0]): int(row[1]) for row in await cursor.fetchall()}
            await cursor.close()
            cursor = await db.execute("SELECT COUNT(*) FROM episodic_memories")
            episode_count = int((await cursor.fetchone())[0])
            await cursor.close()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM working_memory WHERE expires_at > ?",
                (self._now(),),
            )
            working_count = int((await cursor.fetchone())[0])
            await cursor.close()
        return {
            "facts": {status.value: status_counts.get(status.value, 0) for status in FactStatus},
            "episodes": episode_count,
            "working_items": working_count,
        }

    async def retract_fact(self, fact_id: str, *, reason: str = "") -> TemporalFact:
        pool = self._require_pool()
        now = self._now()
        async with pool.write() as db:
            cursor = await db.execute(
                """
                UPDATE temporal_facts
                SET status = ?, valid_until = ?, updated_at = ?
                WHERE fact_id = ? AND status IN (?, ?)
                """,
                (
                    FactStatus.RETRACTED.value,
                    now,
                    now,
                    fact_id,
                    FactStatus.ACTIVE.value,
                    FactStatus.CANDIDATE.value,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No active or candidate fact exists: {fact_id}")
            await cursor.close()
            await db.execute(
                """
                INSERT INTO fact_evidence (
                    evidence_id, fact_id, direction, provenance, confidence,
                    observed_at, payload_json
                )
                VALUES (?, ?, 'retracts', ?, 1.0, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    fact_id,
                    MemoryProvenance.EXPLICIT_USER.value,
                    now,
                    json.dumps({"reason": str(redact_for_persistence(reason))}),
                ),
            )
            await db.commit()
        fact = await self.get_fact(fact_id)
        if fact is None:  # pragma: no cover
            raise RuntimeError(f"Fact disappeared after retraction: {fact_id}")
        return fact

    async def query_facts(
        self,
        query: str,
        *,
        session_id: str = "",
        task_id: str = "",
        limit: int = 20,
        now: datetime | None = None,
    ) -> list[RankedMemory]:
        current = now or datetime.now(UTC)
        current_iso = current.isoformat()
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT fact_id, subject, predicate, value_json, scope, session_id,
                       task_id, confidence, provenance, updated_at
                FROM temporal_facts
                WHERE status = ? AND valid_from <= ?
                  AND (valid_until = '' OR valid_until > ?)
                ORDER BY updated_at DESC
                LIMIT 500
                """,
                (FactStatus.ACTIVE.value, current_iso, current_iso),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        query_terms = self._terms(query)
        ranked: list[RankedMemory] = []
        for row in rows:
            scope = MemoryScope(row[4])
            if scope == MemoryScope.SESSION and row[5] != session_id:
                continue
            if scope == MemoryScope.TASK and row[6] != task_id:
                continue
            value = json.loads(row[3])
            text = f"{row[1]} {row[2]} {self._display_value(value)}"
            relevance = self._lexical_relevance(query_terms, self._terms(text))
            age_days = max(
                0.0,
                (current - datetime.fromisoformat(row[9])).total_seconds() / 86400,
            )
            recency = math.exp(-age_days / 45.0)
            provenance = MemoryProvenance(row[8])
            scope_utility = {
                MemoryScope.TASK: 1.0,
                MemoryScope.SESSION: 0.95,
                MemoryScope.USER: 0.85,
                MemoryScope.SYSTEM: 0.75,
            }[scope]
            score = (
                (0.2 + 0.8 * relevance)
                * (0.35 + 0.65 * recency)
                * float(row[7])
                * PROVENANCE_WEIGHT[provenance]
                * scope_utility
            )
            ranked.append(
                RankedMemory(
                    kind="fact",
                    text=text,
                    score=score,
                    confidence=float(row[7]),
                    provenance=provenance.value,
                    memory_id=row[0],
                )
            )
        return sorted(ranked, key=lambda item: (-item.score, item.memory_id))[: max(0, limit)]

    async def record_episode(
        self,
        *,
        session_id: str,
        task_id: str,
        summary: str,
        outcome: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        provenance_event_id: str = "",
        payload: dict[str, Any] | None = None,
        provenance: MemoryProvenance = MemoryProvenance.INFERRED,
        occurred_at: str | None = None,
    ) -> EpisodicMemory:
        episode = EpisodicMemory(
            episode_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            summary=str(redact_for_persistence(summary)),
            outcome=outcome,
            tags=sorted(set(tags or [])),
            importance=max(0.0, min(1.0, float(importance))),
            occurred_at=occurred_at or self._now(),
            provenance_event_id=provenance_event_id,
        )
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute(
                """
                INSERT INTO episodic_memories (
                    episode_id, session_id, task_id, summary, outcome, tags_json,
                    importance, occurred_at, provenance_event_id, payload_json, provenance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.session_id,
                    episode.task_id,
                    episode.summary,
                    episode.outcome,
                    json.dumps(episode.tags),
                    episode.importance,
                    episode.occurred_at,
                    episode.provenance_event_id,
                    json.dumps(redact_for_persistence(payload or {}), sort_keys=True),
                    MemoryProvenance(provenance).value,
                ),
            )
            await db.commit()
        return episode

    async def query_episodes(
        self,
        query: str,
        *,
        session_id: str = "",
        include_other_sessions: bool = True,
        limit: int = 20,
        now: datetime | None = None,
    ) -> list[RankedMemory]:
        current = now or datetime.now(UTC)
        pool = self._require_pool()
        async with pool.read() as db:
            if include_other_sessions:
                cursor = await db.execute(
                    """
                    SELECT episode_id, session_id, summary, outcome, tags_json,
                           importance, occurred_at, provenance
                    FROM episodic_memories
                    ORDER BY occurred_at DESC
                    LIMIT 500
                    """
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT episode_id, session_id, summary, outcome, tags_json,
                           importance, occurred_at, provenance
                    FROM episodic_memories
                    WHERE session_id = ?
                    ORDER BY occurred_at DESC
                    LIMIT 500
                    """,
                    (session_id,),
                )
            rows = await cursor.fetchall()
            await cursor.close()
        query_terms = self._terms(query)
        ranked: list[RankedMemory] = []
        for row in rows:
            tags = list(json.loads(row[4]))
            text = f"{row[2]} Outcome: {row[3]}. Tags: {', '.join(tags)}"
            relevance = self._lexical_relevance(query_terms, self._terms(text))
            age_days = max(
                0.0,
                (current - datetime.fromisoformat(row[6])).total_seconds() / 86400,
            )
            recency = math.exp(-age_days / 21.0)
            session_utility = 1.0 if row[1] == session_id else 0.7
            outcome_weight = {"success": 1.0, "in_progress": 0.55, "unverified": 0.65}.get(row[3], 0.8)
            try:
                provenance = MemoryProvenance(row[7])
            except ValueError:
                provenance = MemoryProvenance.INFERRED
            score = (
                (0.1 + 0.9 * relevance)
                * (0.25 + 0.75 * recency)
                * float(row[5])
                * PROVENANCE_WEIGHT[provenance]
                * session_utility
                * outcome_weight
            )
            ranked.append(
                RankedMemory(
                    kind="episode",
                    text=text,
                    score=score,
                    confidence=1.0,
                    provenance=provenance.value,
                    memory_id=row[0],
                )
            )
        return sorted(ranked, key=lambda item: (-item.score, item.memory_id))[: max(0, limit)]

    async def put_working(
        self,
        *,
        session_id: str,
        task_id: str,
        key: str,
        value: Any,
        priority: float = 0.5,
        ttl_seconds: float = 3600,
    ) -> WorkingMemory:
        now_dt = datetime.now(UTC)
        item = WorkingMemory(
            session_id=session_id,
            task_id=task_id,
            key=key,
            value=redact_for_persistence(value),
            priority=max(0.0, min(1.0, float(priority))),
            expires_at=(now_dt + timedelta(seconds=max(1.0, ttl_seconds))).isoformat(),
            updated_at=now_dt.isoformat(),
        )
        pool = self._require_pool()
        async with pool.write() as db:
            await db.execute(
                """
                INSERT INTO working_memory (
                    session_id, task_id, key, value_json, priority, expires_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, task_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    priority = excluded.priority,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    item.session_id,
                    item.task_id,
                    item.key,
                    json.dumps(item.value, sort_keys=True),
                    item.priority,
                    item.expires_at,
                    item.updated_at,
                ),
            )
            await db.commit()
        return item

    async def get_working(
        self,
        *,
        session_id: str,
        task_id: str = "",
        now: datetime | None = None,
    ) -> list[WorkingMemory]:
        current_iso = (now or datetime.now(UTC)).isoformat()
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT session_id, task_id, key, value_json, priority, expires_at,
                       updated_at
                FROM working_memory
                WHERE session_id = ? AND expires_at > ?
                  AND (task_id = '' OR task_id = ?)
                ORDER BY priority DESC, updated_at DESC
                """,
                (session_id, current_iso, task_id),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            WorkingMemory(
                session_id=row[0],
                task_id=row[1],
                key=row[2],
                value=json.loads(row[3]),
                priority=float(row[4]),
                expires_at=row[5],
                updated_at=row[6],
            )
            for row in rows
        ]

    async def clear_task_working(self, *, session_id: str, task_id: str) -> int:
        pool = self._require_pool()
        async with pool.write() as db:
            cursor = await db.execute(
                "DELETE FROM working_memory WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            )
            count = cursor.rowcount
            await cursor.close()
            await db.commit()
        return count

    async def purge_expired_working(self, *, now: datetime | None = None) -> int:
        current_iso = (now or datetime.now(UTC)).isoformat()
        pool = self._require_pool()
        async with pool.write() as db:
            cursor = await db.execute(
                "DELETE FROM working_memory WHERE expires_at <= ?",
                (current_iso,),
            )
            count = cursor.rowcount
            await cursor.close()
            await db.commit()
        return count

    async def contradiction_history(
        self,
        *,
        subject: str,
        predicate: str,
    ) -> list[dict[str, Any]]:
        pool = self._require_pool()
        async with pool.read() as db:
            cursor = await db.execute(
                """
                SELECT c.prior_fact_id, c.new_fact_id, c.observed_at, c.resolution
                FROM fact_contradictions c
                JOIN temporal_facts f ON f.fact_id = c.new_fact_id
                WHERE f.subject = ? AND f.predicate = ?
                ORDER BY c.observed_at
                """,
                (self._normalize_term(subject), self._normalize_term(predicate)),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [
            {
                "prior_fact_id": row[0],
                "new_fact_id": row[1],
                "observed_at": row[2],
                "resolution": row[3],
            }
            for row in rows
        ]

    async def _supersede_conflicts(
        self,
        db: Any,
        *,
        fact_id: str,
        rows: list[tuple[Any, ...]],
        now: str,
    ) -> None:
        for row in rows:
            if row[0] == fact_id or row[2] != FactStatus.ACTIVE.value:
                continue
            await db.execute(
                """
                UPDATE temporal_facts
                SET status = ?, valid_until = ?, updated_at = ?
                WHERE fact_id = ?
                """,
                (FactStatus.SUPERSEDED.value, now, now, row[0]),
            )

    @staticmethod
    def _eligible_for_activation(
        provenance: MemoryProvenance,
        evidence_count: int,
        confidence: float,
    ) -> bool:
        if provenance in {
            MemoryProvenance.EXPLICIT_USER,
            MemoryProvenance.VERIFIED_OUTCOME,
            MemoryProvenance.SYSTEM_OBSERVATION,
        }:
            return confidence >= 0.5
        if provenance == MemoryProvenance.REPEATED_BEHAVIOR:
            return evidence_count >= 2 and confidence >= 0.6
        return evidence_count >= 3 and confidence >= 0.65

    @staticmethod
    def _combine_confidence(
        prior: float,
        evidence: float,
        provenance: MemoryProvenance,
    ) -> float:
        support = evidence * PROVENANCE_WEIGHT[provenance] * 0.5
        return min(0.999, 1.0 - (1.0 - prior) * (1.0 - support))

    def _require_pool(self) -> AsyncSqlitePool:
        if self._pool is None:
            raise RuntimeError("TemporalMemoryStore is not initialized")
        return self._pool

    @staticmethod
    def _normalize_term(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9_]{2,}", value.lower())
            if term
            not in {
                "the",
                "and",
                "for",
                "with",
                "that",
                "this",
                "from",
                "user",
            }
        }

    @staticmethod
    def _lexical_relevance(query_terms: set[str], memory_terms: set[str]) -> float:
        if not query_terms:
            return 0.5
        return len(query_terms & memory_terms) / max(1, len(query_terms))

    @staticmethod
    def _display_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _fact_from_row(row: tuple[Any, ...]) -> TemporalFact:
        return TemporalFact(
            fact_id=row[0],
            subject=row[1],
            predicate=row[2],
            value=json.loads(row[3]),
            scope=MemoryScope(row[4]),
            session_id=row[5],
            task_id=row[6],
            status=FactStatus(row[7]),
            confidence=float(row[8]),
            provenance=MemoryProvenance(row[9]),
            evidence_count=int(row[10]),
            valid_from=row[11],
            valid_until=row[12],
            updated_at=row[13],
            privacy_class=PrivacyClass(row[14]),
        )
