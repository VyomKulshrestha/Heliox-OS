from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    EmptyParams,
    NotifyParams,
    VerificationResult,
)
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.intelligence.experience import experience_scope, stable_action_idempotency_key
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import (
    ActionClaimDecision,
    ApprovalStatus,
    DurableTaskStore,
    TaskStatus,
)


async def _durable_store(tmp_path) -> DurableTaskStore:
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    return store


def _executor(tmp_path) -> Executor:
    config = PilotConfig()
    return Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )


def _cpu_plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.CPU_USAGE,
                target="",
                parameters=EmptyParams(),
            )
        ],
        explanation="inspect cpu",
        raw_input="inspect cpu",
    )


def _notify_plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.NOTIFY,
                target="done",
                parameters=NotifyParams(summary="Test", body="done"),
            )
        ],
        explanation="send notification",
        raw_input="notify me",
    )


@pytest.mark.asyncio
async def test_executor_replays_completed_durable_action_without_second_side_effect(tmp_path):
    store = await _durable_store(tmp_path)
    try:
        await store.create_task(
            task_id="task-1",
            session_id="session-1",
            user_input="inspect cpu",
        )
        await store.transition("task-1", TaskStatus.PLANNING)
        await store.transition("task-1", TaskStatus.EXECUTING, plan_id="plan-1")
        executor = _executor(tmp_path)
        executor.set_durable_task_store(store)
        outcome_recorder = MagicMock()
        executor.set_world_model_outcome_recorder(outcome_recorder)
        plan = _cpu_plan()
        effect = AsyncMock(
            return_value=ActionResult(
                action=plan.actions[0],
                success=True,
                output="CPU 12%",
            )
        )
        executor._execute_single = effect

        with experience_scope(session_id="session-1", task_id="task-1"):
            first = await executor.execute(plan, plan_id="plan-1")
            replayed = await executor.execute(plan, plan_id="plan-1")

        assert first[0].success is True
        assert first[0].executed is True
        assert replayed[0].success is True
        assert replayed[0].executed is False
        assert replayed[0].skip_reason == "durable_replay"
        assert replayed[0].output == first[0].output
        effect.assert_awaited_once()
        outcome_recorder.assert_called_once()
        (recorded,) = outcome_recorder.call_args.args  # the first run's result; the replay records nothing
        assert recorded.action.action_type is ActionType.CPU_USAGE and recorded.success is True
        audit_entries = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()]
        assert [entry["event_type"] for entry in audit_entries] == [
            "action_start",
            "action_complete",
            "action_complete",
        ]
        assert audit_entries[-1]["details"]["executed"] is False
        assert audit_entries[-1]["details"]["skip_reason"] == "durable_replay"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pre_cancelled_batch_reports_every_action_as_not_executed(tmp_path):
    executor = _executor(tmp_path)
    plan = _cpu_plan()
    cancel_event = asyncio.Event()
    cancel_event.set()

    results = await executor.execute(plan, cancel_event=cancel_event)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].executed is False
    assert results[0].skip_reason == "cancelled"


@pytest.mark.asyncio
async def test_cancelled_side_effect_becomes_uncertain_and_cannot_auto_retry(tmp_path):
    store = await _durable_store(tmp_path)
    try:
        await store.create_task(
            task_id="task-1",
            session_id="session-1",
            user_input="inspect cpu",
        )
        await store.transition("task-1", TaskStatus.PLANNING)
        await store.transition("task-1", TaskStatus.EXECUTING, plan_id="plan-1")
        executor = _executor(tmp_path)
        executor.set_durable_task_store(store)
        plan = _cpu_plan()
        effect = AsyncMock(side_effect=asyncio.CancelledError)
        executor._execute_single = effect

        with experience_scope(session_id="session-1", task_id="task-1"):
            cancelled = await executor.execute(plan, plan_id="plan-1")
            retry = await executor.execute(plan, plan_id="plan-1")

        assert cancelled[0].success is False
        assert retry[0].success is False
        assert "reconcile its effect" in (retry[0].error or "")
        effect.assert_awaited_once()
        action_key = stable_action_idempotency_key("plan-1", 0, plan.actions[0])
        claim = await store.claim_action(
            task_id="task-1",
            plan_id="plan-1",
            action_key=action_key,
            lease_owner="new-worker",
        )
        assert claim.decision == ActionClaimDecision.RECONCILIATION_REQUIRED
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_duplicate_execute_replays_terminal_response_with_resume_token(tmp_path):
    store = await _durable_store(tmp_path)
    try:
        server = PilotServer(PilotConfig())
        server._durable_tasks = store
        server._handle_execute_inner = AsyncMock(
            return_value={
                "status": "success",
                "conversational": True,
                "explanation": "Hello.",
                "results": [],
            }
        )
        ws = AsyncMock()

        first = await server._handle_execute(
            {
                "input": "hello",
                "session_id": "session-1",
                "task_id": "task-1",
            },
            ws,
        )
        replayed = await server._handle_execute(
            {
                "input": "hello",
                "session_id": "session-1",
                "task_id": "task-1",
                "resume_token": first["resume_token"],
            },
            ws,
        )
        refused = await server._handle_execute(
            {
                "input": "hello",
                "session_id": "session-1",
                "task_id": "task-1",
                "resume_token": "wrong",
            },
            ws,
        )

        assert first["task_id"] == "task-1"
        assert first["resume_token"]
        assert replayed["status"] == "success"
        assert replayed["replayed"] is True
        assert "resume_token" not in replayed
        assert refused["status"] == "error"
        server._handle_execute_inner.assert_awaited_once()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_disconnected_approval_can_be_resolved_and_resumed_after_restart(tmp_path):
    durable = await _durable_store(tmp_path)
    checkpoints = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    try:
        created = await durable.create_task(
            task_id="task-1",
            session_id="session-1",
            user_input="notify me",
        )
        await durable.transition("task-1", TaskStatus.PLANNING)
        await durable.transition(
            "task-1",
            TaskStatus.AWAITING_APPROVAL,
            plan_id="plan-1",
        )
        plan = _notify_plan()
        await checkpoints.start_plan("plan-1", "notify me", plan)
        await durable.create_approval(
            task_id="task-1",
            plan_id="plan-1",
            request={
                "action_indices": [0],
                "actions": [{"index": 0, "action_type": "notify"}],
            },
            timeout_seconds=300,
        )

        server = PilotServer(PilotConfig())
        server._durable_tasks = durable
        server._checkpoint_store = checkpoints
        server._verifier = SimpleNamespace(
            verify=AsyncMock(
                return_value=VerificationResult(
                    passed=True,
                    details=["verified"],
                )
            )
        )

        async def _execute(
            remaining_plan,
            *,
            on_action_start,
            on_action_complete,
            action_index_offset,
            **_kwargs,
        ):
            assert action_index_offset == 0
            result = ActionResult(
                action=remaining_plan.actions[0],
                success=True,
                output="sent",
            )
            await on_action_start(remaining_plan.actions[0])
            await on_action_complete(result)
            return [result]

        server._executor = SimpleNamespace(execute=_execute)
        ws = MagicMock()
        ws.send = AsyncMock()

        confirmation = await server._handle_confirm(
            {"plan_id": "plan-1", "confirmed": True},
            ws,
        )
        resumed = await server._handle_resume_task(
            {
                "task_id": "task-1",
                "resume_token": created.resume_token,
            },
            ws,
        )
        final = await durable.get("task-1")
        approval = await durable.get_approval("plan-1")

        assert confirmation == {
            "status": "ok",
            "confirmed": True,
            "resume_required": True,
            "task_id": "task-1",
        }
        assert resumed["status"] == "success"
        assert resumed["resumed"] is True
        assert final is not None
        assert final.status == TaskStatus.SUCCEEDED
        assert approval is not None
        assert approval.status == ApprovalStatus.APPROVED
    finally:
        await durable.close()


@pytest.mark.asyncio
async def test_resume_task_returns_pending_approval_without_executing(tmp_path):
    durable = await _durable_store(tmp_path)
    try:
        created = await durable.create_task(
            task_id="task-1",
            session_id="session-1",
            user_input="notify me",
        )
        await durable.transition("task-1", TaskStatus.PLANNING)
        await durable.transition(
            "task-1",
            TaskStatus.AWAITING_APPROVAL,
            plan_id="plan-1",
        )
        await durable.create_approval(
            task_id="task-1",
            plan_id="plan-1",
            request={"action_indices": [0]},
            timeout_seconds=300,
        )
        server = PilotServer(PilotConfig())
        server._durable_tasks = durable
        server._handle_resume_plan = AsyncMock()

        response = await server._handle_resume_task(
            {
                "task_id": "task-1",
                "resume_token": created.resume_token,
            },
            AsyncMock(),
        )

        assert response["status"] == "awaiting_approval"
        assert response["approval"] == {"action_indices": [0]}
        server._handle_resume_plan.assert_not_awaited()
    finally:
        await durable.close()
