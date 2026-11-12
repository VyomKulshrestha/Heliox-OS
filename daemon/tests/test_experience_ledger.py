"""Tests for the typed, append-only Heliox experience ledger."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
import pytest_asyncio

from pilot.intelligence.experience import (
    SCHEMA_VERSION,
    ExperienceEventType,
    ExperienceLedger,
    PrivacyClass,
    experience_scope,
    stable_action_idempotency_key,
)


@pytest_asyncio.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    yield instance
    await instance.close()


@pytest.mark.asyncio
async def test_records_typed_ordered_events_with_context_and_provenance(ledger):
    with experience_scope(session_id="session-1", task_id="task-1", user_id="user-1"):
        intent = await ledger.append(
            ExperienceEventType.INTENT,
            idempotency_key="intent-1",
            source="text",
            payload={"input": "open the browser"},
            provenance={"component": "test"},
            confidence=0.9,
        )
        plan = await ledger.append(
            ExperienceEventType.PLAN_CREATED,
            idempotency_key="plan-1",
            source="planner",
            parent_event_id=intent.event_id,
            plan_id="p1",
            payload={"action_count": 1},
        )

    events = await ledger.list_events(task_id="task-1")
    assert [event.event_type for event in events] == [
        ExperienceEventType.INTENT,
        ExperienceEventType.PLAN_CREATED,
    ]
    assert events[0].sequence < events[1].sequence
    assert events[0].schema_version == SCHEMA_VERSION
    assert events[0].session_id == "session-1"
    assert events[0].user_id == "user-1"
    from pilot import __version__

    # The caller's provenance, and which Heliox wrote the row in what kind of process (the suite: its test root).
    assert events[0].provenance == {"component": "test", "written_by": {"version": __version__, "run": "test root"}}
    assert events[0].confidence == 0.9
    assert events[1].parent_event_id == intent.event_id
    assert plan.plan_id == "p1"


@pytest.mark.asyncio
async def test_idempotency_returns_existing_event_without_duplicate(ledger):
    first = await ledger.append(
        ExperienceEventType.ACTION_STARTED,
        idempotency_key="same-action-start",
        action_id="action-1",
        payload={"attempt": 1},
    )
    retry = await ledger.append(
        ExperienceEventType.ACTION_STARTED,
        idempotency_key="same-action-start",
        action_id="action-1",
        payload={"attempt": 2},
    )

    events = await ledger.list_events(action_id="action-1")
    assert len(events) == 1
    assert retry.event_id == first.event_id
    assert retry.payload == {"attempt": 1}


@pytest.mark.asyncio
async def test_subscribers_receive_only_newly_inserted_events(ledger):
    received = []

    async def consume(event):
        received.append(event)

    ledger.subscribe(consume)
    await ledger.append(
        ExperienceEventType.OBSERVATION,
        idempotency_key="subscriber-observation",
        payload={"active_app": "editor"},
    )
    await ledger.append(
        ExperienceEventType.OBSERVATION,
        idempotency_key="subscriber-observation",
        payload={"active_app": "different retry payload"},
    )
    await ledger.drain_subscribers()

    assert len(received) == 1
    assert received[0].payload == {"active_app": "editor"}


@pytest.mark.asyncio
async def test_idempotency_key_cannot_alias_a_different_event_identity(ledger):
    await ledger.append(
        ExperienceEventType.INTENT,
        idempotency_key="collision",
        task_id="task-1",
    )

    with pytest.raises(ValueError, match="different event identity"):
        await ledger.append(
            ExperienceEventType.ACTION_STARTED,
            idempotency_key="collision",
            task_id="task-1",
        )


@pytest.mark.asyncio
async def test_redacts_credentials_binary_and_raw_sensor_media(ledger):
    event = await ledger.append(
        ExperienceEventType.OBSERVATION,
        idempotency_key="private-observation",
        privacy_class=PrivacyClass.BIOMETRIC_DERIVED,
        payload={
            "api_key": "secret-value",
            "nested": {"authorization": "Bearer abcdefghijklmnop"},
            "camera_frame": b"camera bytes",
            "audio_data": [1, 2, 3],
            "screenshot": "base64 pixels",
            "raw_eeg": [[1.0, 2.0]],
            "transcript": "use sk_abcdefghijklmnopqrstuvwxyz now",
            "binary": b"other bytes",
            "region": "left",
        },
    )

    assert event.payload["api_key"] == "[REDACTED]"
    assert event.payload["nested"]["authorization"] == "[REDACTED]"
    assert event.payload["camera_frame"] == "[EXCLUDED_RAW_MEDIA]"
    assert event.payload["audio_data"] == "[EXCLUDED_RAW_MEDIA]"
    assert event.payload["screenshot"] == "[EXCLUDED_RAW_MEDIA]"
    assert event.payload["raw_eeg"] == "[EXCLUDED_RAW_MEDIA]"
    assert "[REDACTED]" in event.payload["transcript"]
    assert event.payload["binary"] == "[EXCLUDED_BINARY]"
    assert event.payload["region"] == "left"
    assert event.privacy_class == PrivacyClass.BIOMETRIC_DERIVED


@pytest.mark.asyncio
async def test_database_rejects_update_and_delete(ledger):
    event = await ledger.append(
        ExperienceEventType.INTENT,
        idempotency_key="immutable-event",
        payload={"input": "keep me"},
    )
    pool = ledger._pool
    assert pool is not None

    async with pool.write() as db:
        with pytest.raises(aiosqlite.IntegrityError, match="append-only"):
            await db.execute(
                "UPDATE experience_events SET source = ? WHERE event_id = ?",
                ("changed", event.event_id),
            )
        await db.rollback()
        with pytest.raises(aiosqlite.IntegrityError, match="append-only"):
            await db.execute(
                "DELETE FROM experience_events WHERE event_id = ?",
                (event.event_id,),
            )
        await db.rollback()


@pytest.mark.asyncio
async def test_concurrent_appends_receive_unique_monotonic_sequences(ledger):
    events = await asyncio.gather(
        *[
            ledger.append(
                ExperienceEventType.OBSERVATION,
                idempotency_key=f"concurrent-{index}",
                payload={"index": index},
            )
            for index in range(20)
        ]
    )

    sequences = [event.sequence for event in events]
    assert len(set(sequences)) == 20
    persisted = await ledger.list_events(limit=100)
    assert [event.sequence for event in persisted] == sorted(sequences)


def test_stable_action_key_changes_with_plan_index_or_action():
    action = {"action_type": "open_url", "target": "https://example.com"}
    same = dict(action)
    changed = {**action, "target": "https://openai.com"}

    key = stable_action_idempotency_key("plan-1", 0, action)
    assert key == stable_action_idempotency_key("plan-1", 0, same)
    assert key != stable_action_idempotency_key("plan-1", 1, same)
    assert key != stable_action_idempotency_key("plan-1", 0, changed)


@pytest.mark.asyncio
async def test_rejects_invalid_confidence(ledger):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        await ledger.append(
            ExperienceEventType.WORLD_PREDICTION,
            confidence=1.1,
        )


def test_the_stamp_says_daemon_in_the_daemon_and_tool_on_the_real_folders(monkeypatch):
    from pilot import __version__, real_folders
    from pilot.intelligence import experience

    experience._written_by.cache_clear()
    monkeypatch.setattr(real_folders, "_daemon_entry", True)
    assert experience.written_by() == {"version": __version__, "run": "daemon"}

    monkeypatch.setattr(real_folders, "_daemon_entry", False)
    monkeypatch.setattr(real_folders, "where_folders_are", lambda: ("real", []))
    experience._written_by.cache_clear()
    assert experience.written_by()["run"] == "tool"

    monkeypatch.setattr(real_folders, "where_folders_are", lambda: ("fake home", []))
    experience._written_by.cache_clear()
    assert experience.written_by()["run"] == "fake home"
    experience._written_by.cache_clear()
