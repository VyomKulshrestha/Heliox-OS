from __future__ import annotations

import asyncio

import pytest

from pilot.security.gateway import TaskScopeOverride
from pilot.workflows.voice_gesture_workflows import (
    VoiceGestureWorkflowStore,
    WorkflowState,
    WorkflowStepRecord,
)


def _store(tmp_path) -> VoiceGestureWorkflowStore:
    return VoiceGestureWorkflowStore(db_file=tmp_path / "voice_gesture_workflows.db")


@pytest.mark.asyncio
async def test_create_returns_pending_workflow_with_no_steps(tmp_path):
    store = _store(tmp_path)
    workflow = await store.create("back up today's screenshots", "voice")
    assert workflow.state == WorkflowState.PENDING.value
    assert workflow.steps == []
    assert workflow.current_step == 0
    assert workflow.scope_override is None


@pytest.mark.asyncio
async def test_create_round_trips_scope_override(tmp_path):
    store = _store(tmp_path)
    override = TaskScopeOverride(max_tier={"shell": 0}, deny_action_types=["browser_execute_js"])
    workflow = await store.create("goal", "gesture", scope_override=override)

    reloaded = await store.get(workflow.workflow_id)
    assert reloaded is not None
    assert reloaded.scope_override is not None
    assert reloaded.scope_override.max_tier == {"shell": 0}
    assert reloaded.scope_override.deny_action_types == ["browser_execute_js"]


@pytest.mark.asyncio
async def test_set_steps_and_update_step_round_trip(tmp_path):
    store = _store(tmp_path)
    workflow = await store.create("goal", "voice")
    steps = [
        WorkflowStepRecord(index=0, title="Step 1", description="do the first thing"),
        WorkflowStepRecord(index=1, title="Step 2", description="do the second thing"),
    ]
    workflow = await store.set_steps(workflow.workflow_id, steps)
    assert len(workflow.steps) == 2

    updated = await store.update_step(workflow.workflow_id, 0, status="success", output="done")
    assert updated.steps[0].status == "success"
    assert updated.steps[0].output == "done"
    assert updated.steps[1].status == "pending"  # untouched


@pytest.mark.asyncio
async def test_update_step_unknown_index_raises(tmp_path):
    store = _store(tmp_path)
    workflow = await store.create("goal", "voice")
    await store.set_steps(workflow.workflow_id, [WorkflowStepRecord(index=0, title="x", description="y")])
    with pytest.raises(KeyError):
        await store.update_step(workflow.workflow_id, 5, status="success")


@pytest.mark.asyncio
async def test_set_state_paused_records_paused_at_and_deadline(tmp_path):
    store = _store(tmp_path)
    workflow = await store.create("goal", "voice")
    updated = await store.set_state(
        workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline="2026-01-01T00:00:00+00:00"
    )
    assert updated is not None
    assert updated.state == WorkflowState.PAUSED.value
    assert updated.paused_at is not None
    assert updated.trigger_deadline == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_set_state_terminal_clears_trigger_deadline(tmp_path):
    store = _store(tmp_path)
    workflow = await store.create("goal", "voice")
    await store.set_state(workflow.workflow_id, WorkflowState.PAUSED, trigger_deadline="2026-01-01T00:00:00+00:00")
    updated = await store.set_state(workflow.workflow_id, WorkflowState.SUCCESS)
    assert updated is not None
    assert updated.trigger_deadline is None


@pytest.mark.asyncio
async def test_set_state_unknown_workflow_returns_none(tmp_path):
    store = _store(tmp_path)
    result = await store.set_state("does-not-exist", WorkflowState.RUNNING)
    assert result is None


@pytest.mark.asyncio
async def test_list_excludes_terminal_by_default(tmp_path):
    store = _store(tmp_path)
    running = await store.create("running goal", "voice")
    done = await store.create("done goal", "voice")
    await store.set_state(done.workflow_id, WorkflowState.SUCCESS)

    active = await store.list()
    assert {w.workflow_id for w in active} == {running.workflow_id}

    everything = await store.list(include_terminal=True)
    assert {w.workflow_id for w in everything} == {running.workflow_id, done.workflow_id}


class TestFindPendingForSource:
    @pytest.mark.asyncio
    async def test_finds_paused_workflow_for_matching_source(self, tmp_path):
        store = _store(tmp_path)
        workflow = await store.create("goal", "voice")
        await store.set_state(workflow.workflow_id, WorkflowState.PAUSED)

        found = await store.find_pending_for_source("voice", within_seconds=60)
        assert found is not None
        assert found.workflow_id == workflow.workflow_id

    @pytest.mark.asyncio
    async def test_ignores_other_sources(self, tmp_path):
        store = _store(tmp_path)
        workflow = await store.create("goal", "gesture")
        await store.set_state(workflow.workflow_id, WorkflowState.PAUSED)

        found = await store.find_pending_for_source("voice", within_seconds=60)
        assert found is None

    @pytest.mark.asyncio
    async def test_ignores_running_and_terminal_states(self, tmp_path):
        store = _store(tmp_path)
        workflow = await store.create("goal", "voice")  # stays PENDING

        found = await store.find_pending_for_source("voice", within_seconds=60)
        assert found is None

    @pytest.mark.asyncio
    async def test_stale_workflow_outside_window_is_not_returned(self, tmp_path):
        store = _store(tmp_path)
        workflow = await store.create("goal", "voice")
        await store.set_state(workflow.workflow_id, WorkflowState.WAITING_FOR_TRIGGER)

        # A within_seconds of 0 means "must have been updated this instant" —
        # any real elapsed time (however small) should fall outside the window.
        found = await store.find_pending_for_source("voice", within_seconds=0)
        assert found is None


@pytest.mark.asyncio
async def test_connect_sets_wal_and_busy_timeout(tmp_path):
    """Regression test for "database is locked": every connection this
    store opens must have a nonzero busy_timeout (and WAL journal mode) so
    two connections writing near-simultaneously wait for the lock instead
    of immediately raising sqlite3.OperationalError."""
    store = _store(tmp_path)
    await store.initialize()
    async with store._connect() as db:
        cursor = await db.execute("PRAGMA busy_timeout")
        (busy_timeout,) = await cursor.fetchone()
        await cursor.close()

        cursor = await db.execute("PRAGMA journal_mode")
        (journal_mode,) = await cursor.fetchone()
        await cursor.close()

    assert busy_timeout > 0
    assert journal_mode.lower() == "wal"


@pytest.mark.asyncio
async def test_concurrent_writes_to_different_workflows_do_not_lock(tmp_path):
    """Real concurrency, not a mocked race: many separate workflows on the
    same store, each doing several overlapping writes via asyncio.gather,
    must not raise "database is locked" now that every connection sets
    busy_timeout."""
    store = _store(tmp_path)

    async def _create_and_advance(i: int) -> None:
        workflow = await store.create(f"goal {i}", "voice")
        await store.set_state(workflow.workflow_id, WorkflowState.RUNNING)
        await store.set_steps(workflow.workflow_id, [WorkflowStepRecord(index=0, title="step", description="step")])
        await store.set_state(workflow.workflow_id, WorkflowState.SUCCESS)

    await asyncio.gather(*(_create_and_advance(i) for i in range(15)))  # must not raise

    workflows = await store.list(include_terminal=True)
    assert len(workflows) == 15
    assert all(w.state == WorkflowState.SUCCESS.value for w in workflows)
