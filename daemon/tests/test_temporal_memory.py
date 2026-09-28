from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from pilot.intelligence.experience import PrivacyClass
from pilot.memory.assembler import TemporalContextAssembler
from pilot.memory.sliding_window import get_token_count
from pilot.memory.temporal import (
    FactStatus,
    MemoryProvenance,
    MemoryScope,
    TemporalMemoryStore,
)


@pytest.fixture
async def store(tmp_path):
    temporal = TemporalMemoryStore(tmp_path / "temporal.db")
    await temporal.initialize()
    try:
        yield temporal
    finally:
        await temporal.close()


@pytest.mark.asyncio
async def test_explicit_user_fact_is_active_immediately(store):
    fact = await store.remember_fact(
        subject="User",
        predicate="preferred editor",
        value="VS Code",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
        event_id="event-1",
    )

    assert fact.status == FactStatus.ACTIVE
    assert fact.subject == "user"
    assert fact.predicate == "preferred editor"
    assert fact.evidence_count == 1


@pytest.mark.asyncio
async def test_inferred_fact_needs_three_supporting_observations(store):
    first = await store.remember_fact(
        subject="user",
        predicate="prefers concise responses",
        value=True,
        confidence=0.8,
        provenance=MemoryProvenance.INFERRED,
    )
    second = await store.remember_fact(
        subject="user",
        predicate="prefers concise responses",
        value=True,
        confidence=0.8,
        provenance=MemoryProvenance.INFERRED,
    )
    third = await store.remember_fact(
        subject="user",
        predicate="prefers concise responses",
        value=True,
        confidence=0.8,
        provenance=MemoryProvenance.INFERRED,
    )

    assert first.status == FactStatus.CANDIDATE
    assert second.status == FactStatus.CANDIDATE
    assert third.status == FactStatus.ACTIVE
    assert third.evidence_count == 3


@pytest.mark.asyncio
async def test_repeated_behavior_needs_two_observations(store):
    first = await store.remember_fact(
        subject="user",
        predicate="usual browser",
        value="Firefox",
        confidence=0.7,
        provenance=MemoryProvenance.REPEATED_BEHAVIOR,
    )
    second = await store.remember_fact(
        subject="user",
        predicate="usual browser",
        value="Firefox",
        confidence=0.7,
        provenance=MemoryProvenance.REPEATED_BEHAVIOR,
    )

    assert first.status == FactStatus.CANDIDATE
    assert second.status == FactStatus.ACTIVE


@pytest.mark.asyncio
async def test_low_evidence_contradiction_does_not_replace_active_fact(store):
    original = await store.remember_fact(
        subject="user",
        predicate="theme",
        value="dark",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    candidate = await store.remember_fact(
        subject="user",
        predicate="theme",
        value="light",
        confidence=0.9,
        provenance=MemoryProvenance.MODEL_SYNTHESIS,
    )

    assert candidate.status == FactStatus.CANDIDATE
    assert (await store.get_fact(original.fact_id)).status == FactStatus.ACTIVE
    ranked = await store.query_facts("theme")
    assert [item.text for item in ranked] == ["user theme dark"]


@pytest.mark.asyncio
async def test_explicit_correction_closes_old_validity_and_preserves_history(store):
    original = await store.remember_fact(
        subject="user",
        predicate="theme",
        value="dark",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    corrected = await store.remember_fact(
        subject="user",
        predicate="theme",
        value="light",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    prior = await store.get_fact(original.fact_id)
    history = await store.contradiction_history(subject="user", predicate="theme")

    assert corrected.status == FactStatus.ACTIVE
    assert prior is not None
    assert prior.status == FactStatus.SUPERSEDED
    assert prior.valid_until
    assert history[-1]["resolution"] == "new_fact_activated"


@pytest.mark.asyncio
async def test_candidate_promotion_updates_contradiction_resolution(store):
    await store.remember_fact(
        subject="user",
        predicate="editor",
        value="vim",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    for _ in range(3):
        promoted = await store.remember_fact(
            subject="user",
            predicate="editor",
            value="code",
            confidence=0.8,
            provenance=MemoryProvenance.INFERRED,
        )

    history = await store.contradiction_history(subject="user", predicate="editor")
    assert promoted.status == FactStatus.ACTIVE
    assert history[-1]["resolution"] == "candidate_promoted"


@pytest.mark.asyncio
async def test_retracted_fact_is_excluded_from_context(store):
    fact = await store.remember_fact(
        subject="user",
        predicate="preferred shell",
        value="PowerShell",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    retracted = await store.retract_fact(fact.fact_id, reason="user corrected it")

    assert retracted.status == FactStatus.RETRACTED
    assert await store.query_facts("shell") == []


@pytest.mark.asyncio
async def test_fact_evidence_is_append_only(store, tmp_path):
    await store.remember_fact(
        subject="system",
        predicate="platform",
        value="windows",
        confidence=1.0,
        provenance=MemoryProvenance.SYSTEM_OBSERVATION,
    )

    connection = sqlite3.connect(tmp_path / "temporal.db")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM fact_evidence")
    connection.close()


@pytest.mark.asyncio
async def test_secret_and_raw_media_are_redacted_before_memory_storage(store):
    fact = await store.remember_fact(
        subject="task",
        predicate="context",
        value={
            "api_key": "private",
            "camera_frame": b"pixels",
            "safe": "ok",
        },
        scope=MemoryScope.TASK,
        session_id="s1",
        task_id="t1",
        confidence=1.0,
        provenance=MemoryProvenance.SYSTEM_OBSERVATION,
        privacy_class=PrivacyClass.SENSITIVE,
    )

    assert fact.value == {
        "api_key": "[REDACTED]",
        "camera_frame": "[EXCLUDED_RAW_MEDIA]",
        "safe": "ok",
    }


@pytest.mark.asyncio
async def test_working_memory_is_task_scoped_and_expires(store):
    now = datetime.now(UTC)
    await store.put_working(
        session_id="s1",
        task_id="t1",
        key="pending approval",
        value={"plan_id": "p1"},
        priority=1.0,
        ttl_seconds=60,
    )

    assert len(await store.get_working(session_id="s1", task_id="t1", now=now)) == 1
    assert await store.get_working(session_id="s1", task_id="other", now=now) == []
    assert (
        await store.get_working(
            session_id="s1",
            task_id="t1",
            now=now + timedelta(minutes=2),
        )
        == []
    )


@pytest.mark.asyncio
async def test_clear_task_working_does_not_remove_session_memory(store):
    await store.put_working(
        session_id="s1",
        task_id="",
        key="active workspace",
        value="pilot",
    )
    await store.put_working(
        session_id="s1",
        task_id="t1",
        key="current plan",
        value="inspect",
    )

    assert await store.clear_task_working(session_id="s1", task_id="t1") == 1
    remaining = await store.get_working(session_id="s1", task_id="t1")
    assert [item.key for item in remaining] == ["active workspace"]


@pytest.mark.asyncio
async def test_episode_ranking_prefers_relevant_same_session(store):
    now = datetime.now(UTC)
    relevant = await store.record_episode(
        session_id="s1",
        task_id="t1",
        summary="Opened the project README and summarized release steps",
        outcome="success",
        tags=["file_read", "release"],
        importance=0.9,
        occurred_at=now.isoformat(),
    )
    await store.record_episode(
        session_id="s2",
        task_id="t2",
        summary="Adjusted speaker volume",
        outcome="success",
        tags=["volume"],
        importance=1.0,
        occurred_at=now.isoformat(),
    )

    ranked = await store.query_episodes(
        "summarize release README",
        session_id="s1",
        now=now,
    )
    assert ranked[0].memory_id == relevant.episode_id


@pytest.mark.asyncio
async def test_legacy_episodes_migrate_with_conservative_provenance(tmp_path):
    db_path = tmp_path / "legacy-temporal.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE episodic_memories (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            outcome TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            importance REAL NOT NULL,
            occurred_at TEXT NOT NULL,
            provenance_event_id TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO episodic_memories (
            episode_id, session_id, task_id, summary, outcome, tags_json,
            importance, occurred_at, provenance_event_id, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-episode",
            "s1",
            "t1",
            "Legacy release workflow",
            "success",
            "[]",
            0.8,
            datetime.now(UTC).isoformat(),
            "",
            "{}",
        ),
    )
    connection.commit()
    connection.close()

    temporal = TemporalMemoryStore(db_path)
    await temporal.initialize()
    try:
        episodes = await temporal.query_episodes("release workflow", session_id="s1")
        assert len(episodes) == 1
        assert episodes[0].provenance == MemoryProvenance.INFERRED.value
    finally:
        await temporal.close()


@pytest.mark.asyncio
async def test_context_assembler_enforces_token_budget_and_provenance_labels(store):
    await store.put_working(
        session_id="s1",
        task_id="t1",
        key="pending decision",
        value="approve plan p1",
        priority=1.0,
    )
    for index in range(8):
        await store.remember_fact(
            subject="user",
            predicate=f"preference {index}",
            value="concise " * 20,
            confidence=1.0,
            provenance=MemoryProvenance.EXPLICIT_USER,
        )
    assembler = TemporalContextAssembler(store)

    context = await assembler.assemble(
        "preference",
        session_id="s1",
        task_id="t1",
        max_tokens=120,
    )

    assert context.items
    assert context.items[0].kind == "working"
    assert context.token_count <= 120
    assert get_token_count(context.text) <= 120
    assert "advisory context only" in context.text
    assert "source=" in context.text
    assert context.omitted_count > 0


@pytest.mark.asyncio
async def test_review_listing_and_stats_include_candidates_without_activating_them(store):
    active = await store.remember_fact(
        subject="user",
        predicate="editor",
        value="VS Code",
        confidence=1.0,
        provenance=MemoryProvenance.EXPLICIT_USER,
    )
    candidate = await store.remember_fact(
        subject="user",
        predicate="theme",
        value="dark",
        confidence=0.8,
        provenance=MemoryProvenance.INFERRED,
    )
    await store.record_episode(
        session_id="s1",
        task_id="t1",
        summary="Inspected settings",
        outcome="success",
    )
    await store.put_working(
        session_id="s1",
        task_id="t1",
        key="intent",
        value="review memory",
    )

    listed = await store.list_facts()
    stats = await store.stats()

    assert [item.fact_id for item in listed] == [active.fact_id, candidate.fact_id]
    assert stats["facts"]["active"] == 1
    assert stats["facts"]["candidate"] == 1
    assert stats["episodes"] == 1
    assert stats["working_items"] == 1
