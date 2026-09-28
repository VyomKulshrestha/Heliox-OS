from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, NotifyParams, VerificationResult
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore


def _notify_action(message: str) -> Action:
    return Action(
        action_type=ActionType.NOTIFY,
        target=message,
        parameters=NotifyParams(summary="Test", body=message),
    )


@pytest.mark.asyncio
async def test_checkpoint_store_round_trips_plan_results_and_resume_state(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    plan = ActionPlan(
        actions=[_notify_action("one"), _notify_action("two")],
        explanation="notify twice",
        raw_input="send notifications",
    )

    await store.start_plan("plan-1", "send notifications", plan)
    await store.record_result(
        "plan-1",
        ActionResult(action=plan.actions[0], success=True, output="first output", snapshot_id="snap-1"),
    )

    checkpoint = await store.get("plan-1")

    assert checkpoint is not None
    assert checkpoint.plan_id == "plan-1"
    assert checkpoint.user_input == "send notifications"
    assert checkpoint.completed_count == 1
    assert checkpoint.last_output == "first output"
    assert checkpoint.snapshot_ids == ["snap-1"]
    assert checkpoint.plan.actions[1].target == "two"
    assert checkpoint.results[0].output == "first output"
    assert not checkpoint.is_complete


@pytest.mark.asyncio
async def test_resume_plan_skips_completed_actions_and_verifies_full_plan(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    plan = ActionPlan(
        actions=[_notify_action("one"), _notify_action("two")],
        explanation="notify twice",
        raw_input="send notifications",
    )
    first_result = ActionResult(action=plan.actions[0], success=True, output="first output")
    await store.start_plan("plan-1", "send notifications", plan)
    await store.record_result("plan-1", first_result)

    server = PilotServer(PilotConfig())
    server._checkpoint_store = store
    server._verifier = SimpleNamespace(
        verify=AsyncMock(
            return_value=VerificationResult(
                passed=True,
                details=["all actions completed"],
                failed_actions=[],
                rollback_triggered=False,
            )
        )
    )

    async def _execute(
        remaining_plan,
        *,
        on_action_start,
        on_action_complete,
        cancel_event,
        plan_id,
        initial_last_output,
    ):
        assert plan_id == "plan-1"
        assert initial_last_output == "first output"
        assert [action.target for action in remaining_plan.actions] == ["two"]
        result = ActionResult(action=remaining_plan.actions[0], success=True, output="second output")
        await on_action_start(remaining_plan.actions[0])
        await on_action_complete(result)
        return [result]

    server._executor = SimpleNamespace(execute=_execute)
    ws = MagicMock()
    ws.send = AsyncMock()

    response = await server._handle_resume_plan({"plan_id": "plan-1"}, ws)
    checkpoint = await store.get("plan-1")

    assert response["status"] == "success"
    assert response["resumed"] is True
    assert response["skipped_actions"] == 1
    assert response["executed_actions"] == 1
    assert response["verification"]["passed"] is True
    assert [result["output"] for result in response["results"]] == ["first output", "second output"]
    assert checkpoint is not None
    assert checkpoint.status == "complete"
    assert checkpoint.completed_count == 2
    server._verifier.verify.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_plan_returns_success_when_checkpoint_already_complete(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    plan = ActionPlan(actions=[_notify_action("one")], explanation="done")
    checkpoint_result = ActionResult(action=plan.actions[0], success=True, output="done")
    await store.start_plan("plan-1", "send notification", plan)
    await store.record_result("plan-1", checkpoint_result)
    await store.mark_status("plan-1", "complete")

    server = PilotServer(PilotConfig())
    server._checkpoint_store = store
    server._executor = SimpleNamespace(execute=AsyncMock())
    server._verifier = SimpleNamespace(
        verify=AsyncMock(
            return_value=VerificationResult(
                passed=True,
                details=["checkpoint postconditions verified"],
                failed_actions=[],
                independently_observed_actions=[0],
            )
        )
    )
    ws = MagicMock()
    ws.send = AsyncMock()

    response = await server._handle_resume_plan({"plan_id": "plan-1"}, ws)

    assert response["status"] == "success"
    assert response["resumed"] is False
    assert response["message"] == "done"
    assert response["verification"]["passed"] is True
    server._executor.execute.assert_not_called()
    server._verifier.verify.assert_awaited_once_with(plan, [checkpoint_result])


@pytest.mark.asyncio
async def test_complete_checkpoint_fails_closed_when_postcondition_is_not_verified(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    plan = ActionPlan(actions=[_notify_action("one")], explanation="done")
    checkpoint_result = ActionResult(action=plan.actions[0], success=True, output="done")
    await store.start_plan("plan-1", "send notification", plan)
    await store.record_result("plan-1", checkpoint_result)
    await store.mark_status("plan-1", "complete")

    server = PilotServer(PilotConfig())
    server._checkpoint_store = store
    server._executor = SimpleNamespace(execute=AsyncMock())
    server._verifier = SimpleNamespace(
        verify=AsyncMock(
            return_value=VerificationResult(
                passed=False,
                details=["Action 0 (notify): MISMATCH — notification was not observed"],
                failed_actions=[0],
            )
        )
    )
    ws = MagicMock()
    ws.send = AsyncMock()

    response = await server._handle_resume_plan({"plan_id": "plan-1"}, ws)
    checkpoint = await store.get("plan-1")

    assert response["status"] == "partial_failure"
    assert response["verification"]["passed"] is False
    assert "notification was not observed" in response["message"]
    assert checkpoint is not None
    assert checkpoint.status == "failed"
    server._executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_failed_resume_reports_actual_failure_instead_of_plan_intent(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints.db")
    plan = ActionPlan(
        actions=[_notify_action("one")],
        explanation="notify successfully",
        raw_input="send notification",
    )
    await store.start_plan("plan-1", "send notification", plan)

    server = PilotServer(PilotConfig())
    server._checkpoint_store = store
    server._verifier = SimpleNamespace(verify=AsyncMock())

    async def _execute(remaining_plan, **_kwargs):
        return [
            ActionResult(
                action=remaining_plan.actions[0],
                success=False,
                error="Safety stop: interrupted action requires reconciliation.",
            )
        ]

    server._executor = SimpleNamespace(execute=_execute)
    ws = MagicMock()
    ws.send = AsyncMock()

    response = await server._handle_resume_plan({"plan_id": "plan-1"}, ws)

    assert response["status"] == "partial_failure"
    assert "interrupted action requires reconciliation" in response["message"]
    assert response["message"] != plan.explanation
    assert "explanation" not in response
    server._verifier.verify.assert_not_awaited()
