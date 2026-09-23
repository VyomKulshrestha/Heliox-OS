from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from pilot.workflows.durable_tasks import (
    ActionClaimConflict,
    ActionClaimDecision,
    ActionExecutionStatus,
    ApprovalConflict,
    ApprovalStatus,
    DurableTaskStore,
    InvalidTaskTransition,
    StaleTaskVersion,
    TaskStatus,
)


@pytest.fixture
async def store(tmp_path):
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    try:
        yield durable
    finally:
        await durable.close()


@pytest.mark.asyncio
async def test_resume_token_is_returned_once_and_only_hash_is_stored(store, tmp_path):
    created = await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="open github",
    )

    assert created.resume_token
    assert await store.get_by_resume_token(created.resume_token) == created.task
    assert await store.get_by_resume_token("wrong-token") is None

    connection = sqlite3.connect(tmp_path / "durable.db")
    stored_hash = connection.execute("SELECT resume_token_hash FROM durable_tasks WHERE task_id = 'task-1'").fetchone()[
        0
    ]
    connection.close()

    assert stored_hash != created.resume_token
    assert created.resume_token not in (tmp_path / "durable.db").read_bytes().decode(errors="ignore")


@pytest.mark.asyncio
async def test_task_state_machine_is_versioned_and_idempotent(store):
    created = await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="run plan",
    )
    planning = await store.transition(
        "task-1",
        TaskStatus.PLANNING,
        expected_version=created.task.version,
    )
    same = await store.transition("task-1", TaskStatus.PLANNING)

    assert planning.version == created.task.version + 1
    assert same.version == planning.version

    with pytest.raises(StaleTaskVersion):
        await store.transition(
            "task-1",
            TaskStatus.EXECUTING,
            expected_version=created.task.version,
        )
    with pytest.raises(InvalidTaskTransition):
        await store.transition("task-1", TaskStatus.VERIFYING)


@pytest.mark.asyncio
async def test_terminal_response_is_redacted_and_replayable(store):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="inspect",
    )
    await store.transition("task-1", TaskStatus.PLANNING)
    completed = await store.transition(
        "task-1",
        TaskStatus.SUCCEEDED,
        terminal_response={"status": "success", "api_key": "secret-value"},
    )

    assert completed.is_terminal
    assert completed.terminal_response == {
        "api_key": "[REDACTED]",
        "status": "success",
    }
    assert (await store.get("task-1")).terminal_response == completed.terminal_response


@pytest.mark.asyncio
async def test_task_transition_history_is_append_only(store, tmp_path):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="inspect",
    )
    await store.transition("task-1", TaskStatus.PLANNING)

    connection = sqlite3.connect(tmp_path / "durable.db")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute("DELETE FROM task_transitions WHERE task_id = 'task-1'")
    connection.close()


@pytest.mark.asyncio
async def test_cancellation_request_survives_reopen(tmp_path):
    db_path = tmp_path / "durable.db"
    first = DurableTaskStore(db_path)
    await first.initialize()
    await first.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="long task",
    )
    await first.request_cancel("task-1")
    await first.close()

    second = DurableTaskStore(db_path)
    await second.initialize()
    try:
        task = await second.get("task-1")
        assert task is not None
        assert task.cancellation_requested is True
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_approval_survives_reopen_and_resolution_is_idempotent(tmp_path):
    db_path = tmp_path / "durable.db"
    first = DurableTaskStore(db_path)
    await first.initialize()
    await first.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="delete file",
    )
    await first.create_approval(
        task_id="task-1",
        plan_id="plan-1",
        request={"reason": "destructive"},
        timeout_seconds=300,
    )
    await first.close()

    second = DurableTaskStore(db_path)
    await second.initialize()
    try:
        persisted = await second.get_approval("plan-1")
        assert persisted is not None
        assert persisted.status == ApprovalStatus.PENDING

        approved = await second.resolve_approval(
            "plan-1",
            ApprovalStatus.APPROVED,
            approved_indices=[2, 0, 2],
        )
        duplicate = await second.resolve_approval(
            "plan-1",
            ApprovalStatus.APPROVED,
            approved_indices=[0, 2],
        )
        assert approved == duplicate
        assert approved.approved_indices == [0, 2]

        with pytest.raises(ApprovalConflict):
            await second.resolve_approval("plan-1", ApprovalStatus.DENIED)
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_completed_action_is_replayed_without_a_second_claim(store):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="notify",
    )
    first = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-1",
    )
    await store.finish_action(
        task_id="task-1",
        action_key="action-1",
        lease_owner="worker-1",
        status=ActionExecutionStatus.COMPLETED,
        result={"success": True, "output": "done"},
    )
    duplicate = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-2",
    )

    assert first.decision == ActionClaimDecision.CLAIMED
    assert duplicate.decision == ActionClaimDecision.ALREADY_COMPLETED
    assert duplicate.attempt_count == 1
    assert duplicate.result == {"output": "done", "success": True}


@pytest.mark.asyncio
async def test_active_action_lease_is_busy_and_cannot_be_stolen(store):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="notify",
    )
    await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-1",
        lease_seconds=120,
    )

    duplicate = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-2",
    )

    assert duplicate.decision == ActionClaimDecision.BUSY
    assert duplicate.lease_owner == "worker-1"
    with pytest.raises(ActionClaimConflict):
        await store.finish_action(
            task_id="task-1",
            action_key="action-1",
            lease_owner="worker-2",
            status=ActionExecutionStatus.COMPLETED,
            result={"success": True},
        )


@pytest.mark.asyncio
async def test_expired_running_claim_requires_reconciliation(store):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="notify",
    )
    await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-1",
    )
    pool = store._require_pool()
    async with pool.write() as db:
        await db.execute(
            """
            UPDATE action_execution_claims
            SET lease_until = ?
            WHERE task_id = 'task-1' AND action_key = 'action-1'
            """,
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(),),
        )
        await db.commit()

    recovered = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-2",
    )

    assert recovered.decision == ActionClaimDecision.RECONCILIATION_REQUIRED
    assert recovered.status == ActionExecutionStatus.UNCERTAIN
    assert recovered.attempt_count == 1


@pytest.mark.asyncio
async def test_failed_action_needs_explicit_retry(store):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="notify",
    )
    await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-1",
    )
    await store.finish_action(
        task_id="task-1",
        action_key="action-1",
        lease_owner="worker-1",
        status=ActionExecutionStatus.FAILED,
        error="temporary error",
    )

    refused = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-2",
    )
    retried = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="worker-2",
        allow_retry=True,
    )

    assert refused.decision == ActionClaimDecision.RETRY_REQUIRED
    assert retried.decision == ActionClaimDecision.CLAIMED
    assert retried.attempt_count == 2


@pytest.mark.asyncio
async def test_startup_recovery_interrupts_tasks_and_never_replays_running_actions(
    store,
):
    await store.create_task(
        task_id="task-1",
        session_id="session-1",
        user_input="long task",
    )
    await store.transition("task-1", TaskStatus.PLANNING)
    await store.transition("task-1", TaskStatus.EXECUTING)
    await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="dead-worker",
    )
    await store.create_approval(
        task_id="task-1",
        plan_id="plan-1",
        request={},
        timeout_seconds=0,
    )

    recovered = await store.recover_incomplete()
    task = await store.get("task-1")
    action = await store.claim_action(
        task_id="task-1",
        plan_id="plan-1",
        action_key="action-1",
        lease_owner="new-worker",
    )
    approval = await store.get_approval("plan-1")

    assert recovered.interrupted_tasks == 1
    assert recovered.uncertain_actions == 1
    assert recovered.expired_approvals == 1
    assert task is not None
    assert task.status == TaskStatus.INTERRUPTED
    assert action.decision == ActionClaimDecision.RECONCILIATION_REQUIRED
    assert approval is not None
    assert approval.status == ApprovalStatus.EXPIRED


@pytest.mark.asyncio
async def test_startup_recovery_does_not_silently_run_queued_user_task(store):
    await store.create_task(task_id="queued-task", session_id="session-1", user_input="next task")

    recovered = await store.recover_incomplete()
    task = await store.get("queued-task")

    assert recovered.interrupted_tasks == 1
    assert task is not None
    assert task.status == TaskStatus.INTERRUPTED
