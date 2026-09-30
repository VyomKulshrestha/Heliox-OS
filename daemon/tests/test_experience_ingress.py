"""Integration coverage for experience-ledger ingress wiring."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams, SystemInfoParams
from pilot.agents.executor import Executor
from pilot.agents.proactive import ProactiveSuggestionEngine
from pilot.config import PilotConfig
from pilot.intelligence.experience import (
    ExperienceEventType,
    ExperienceLedger,
    experience_scope,
)
from pilot.intelligence.online_learning import VerifiedOnlineLearner
from pilot.network.collab_executor import CollabExecutor
from pilot.security.audit import AuditLogger
from pilot.security.gateway import InvocationSource
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer


@pytest_asyncio.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    yield instance
    await instance.close()


def _executor(tmp_path, ledger: ExperienceLedger) -> Executor:
    config = PilotConfig()
    config.security.dry_run = True
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )
    executor.set_experience_ledger(ledger)
    return executor


def _plan(tmp_path) -> ActionPlan:
    target = tmp_path / "would-write.txt"
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_WRITE,
                target=str(target),
                parameters=FileParams(path=str(target), content="hello"),
            )
        ],
        explanation="write a test file",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        InvocationSource.INTERACTIVE,
        InvocationSource.VOICE,
        InvocationSource.GESTURE,
        InvocationSource.AUTONOMOUS,
    ],
)
async def test_all_executor_ingress_sources_emit_action_lifecycle(tmp_path, ledger, source):
    executor = _executor(tmp_path, ledger)
    starts = AsyncMock()
    completions = AsyncMock()

    with experience_scope(session_id="session-a", task_id="task-a", user_id="user-a"):
        results = await executor.execute(
            _plan(tmp_path),
            plan_id=f"plan-{source.value}",
            invocation_source=source,
            on_action_start=starts,
            on_action_complete=completions,
        )

    assert results[0].success is True
    starts.assert_awaited_once()
    completions.assert_awaited_once()
    events = await ledger.list_events(plan_id=f"plan-{source.value}")
    assert [event.event_type for event in events] == [
        ExperienceEventType.PLAN_CREATED,
        ExperienceEventType.CANDIDATE_ACTION,
        ExperienceEventType.ACTION_STARTED,
        ExperienceEventType.ACTION_COMPLETED,
    ]
    assert {event.source for event in events} == {source.value}
    assert {event.session_id for event in events} == {"session-a"}
    assert {event.task_id for event in events} == {"task-a"}
    assert events[1].action_id == events[2].action_id == events[3].action_id
    assert events[1].payload["action_idempotency_key"] == events[1].action_id
    assert events[3].payload["executed"] is False
    assert events[3].payload["skip_reason"] == "dry_run"


@pytest.mark.asyncio
async def test_reexecution_keeps_action_identity_and_records_each_attempt(tmp_path, ledger):
    executor = _executor(tmp_path, ledger)
    plan = _plan(tmp_path)

    await executor.execute(plan, plan_id="repeat-plan")
    await executor.execute(plan, plan_id="repeat-plan")

    events = await ledger.list_events(plan_id="repeat-plan")
    assert sum(event.event_type == ExperienceEventType.INTENT for event in events) == 1
    assert sum(event.event_type == ExperienceEventType.PLAN_CREATED for event in events) == 1
    assert sum(event.event_type == ExperienceEventType.CANDIDATE_ACTION for event in events) == 1
    candidate = next(event for event in events if event.event_type == ExperienceEventType.CANDIDATE_ACTION)
    starts = [event for event in events if event.event_type == ExperienceEventType.ACTION_STARTED]
    completions = [event for event in events if event.event_type == ExperienceEventType.ACTION_COMPLETED]
    assert len(starts) == len(completions) == 2
    assert {event.action_id for event in starts + completions} == {candidate.action_id}


@pytest.mark.asyncio
async def test_scoped_subplans_replay_as_one_verified_parent_plan(tmp_path, ledger):
    config = PilotConfig()
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "scoped-audit.jsonl"),
    )
    executor.set_experience_ledger(ledger)
    learner = VerifiedOnlineLearner(tmp_path / "scoped-learning-state.json")
    ledger.subscribe(learner.consume)

    # Safe read-only actions run against fixture files and preserve the full
    # parent-plan indices while each specialist is dispatched independently.
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for index, path in enumerate(paths):
        path.write_text(f"fixture {index}", encoding="utf-8")
    actions = [
        Action(action_type=ActionType.FILE_READ, target=str(path), parameters=FileParams(path=str(path)))
        for path in paths
    ]
    for index, action in enumerate(actions):
        subplan = ActionPlan(actions=[action], explanation="read fixture")
        with experience_scope(
            task_id="parent-plan",
            plan_id="parent-plan",
            action_index_offset=index,
        ):
            result = await executor.execute(subplan)
        assert result[0].success is True

    await ledger.append(
        ExperienceEventType.OUTCOME_VERIFIED,
        plan_id="parent-plan",
        source="verifier",
        idempotency_key="parent-plan:verified",
        payload={
            "result_count": 2,
            "verification": {
                "passed": True,
                "failed_actions": [],
                "independently_observed_actions": [0, 1],
            },
        },
    )
    await ledger.drain_subscribers()

    events = await ledger.list_events(plan_id="parent-plan")
    candidates = [event for event in events if event.event_type == ExperienceEventType.CANDIDATE_ACTION]
    starts = [event for event in events if event.event_type == ExperienceEventType.ACTION_STARTED]
    completions = [event for event in events if event.event_type == ExperienceEventType.ACTION_COMPLETED]
    assert [event.payload["index"] for event in candidates] == [0, 1]
    assert [event.payload["index"] for event in starts] == [0, 1]
    assert [event.payload["index"] for event in completions] == [0, 1]
    assert [event.action_id for event in candidates] == [event.action_id for event in starts]
    assert [event.action_id for event in candidates] == [event.action_id for event in completions]
    assert learner.status()["transitions"]["labels"] == 2
    assert learner.status()["workflow_patterns"] == 1


@pytest.mark.asyncio
async def test_remote_collaboration_records_original_parent_action_indices(tmp_path, ledger):
    config = PilotConfig()
    config.security.dry_run = False
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "remote-collab-audit.jsonl"),
    )
    executor.set_experience_ledger(ledger)
    learner = VerifiedOnlineLearner(tmp_path / "remote-collab-learning.json")
    ledger.subscribe(learner.consume)

    actions = [
        Action(action_type=ActionType.SYSTEM_INFO, parameters=SystemInfoParams()),
        Action(action_type=ActionType.SYSTEM_INFO, parameters=SystemInfoParams()),
    ]
    plan = ActionPlan(actions=actions, raw_input="review system status")
    collab = None

    async def send_to(peer_id, message_type, payload):
        if message_type == "task_delegate":
            remote_actions = [Action.model_validate(item) for item in payload["actions"]]
            raw_results = [
                ActionResult(action=action, success=True, output=f"remote result {index}").model_dump(mode="json")
                for index, action in enumerate(remote_actions)
            ]
            asyncio.create_task(
                collab.handle_task_result(peer_id, {"task_id": payload["task_id"], "results": raw_results})
            )
        return True

    mesh = SimpleNamespace(
        peer_ids=["peer-a"],
        get_connection=lambda _peer: SimpleNamespace(connected=True, peer_capabilities=None),
        send_to=send_to,
    )
    collab = CollabExecutor(mesh=mesh, local_executor=executor)
    executor.set_collab_executor(collab)

    results = await executor.execute(plan, plan_id="remote-parent")

    assert [result.output for result in results] == ["remote result 0", "remote result 1"]
    events = await ledger.list_events(plan_id="remote-parent")
    candidates = [event for event in events if event.event_type == ExperienceEventType.CANDIDATE_ACTION]
    starts = [event for event in events if event.event_type == ExperienceEventType.ACTION_STARTED]
    completions = [event for event in events if event.event_type == ExperienceEventType.ACTION_COMPLETED]
    assert [event.payload["index"] for event in candidates] == [0, 1]
    assert [event.payload["index"] for event in starts] == [0, 1]
    assert [event.payload["index"] for event in completions] == [0, 1]
    assert [event.action_id for event in candidates] == [event.action_id for event in starts]
    assert [event.action_id for event in candidates] == [event.action_id for event in completions]

    await ledger.append(
        ExperienceEventType.OUTCOME_VERIFIED,
        plan_id="remote-parent",
        source="verifier",
        idempotency_key="remote-parent:verified",
        payload={
            "result_count": 2,
            "verification": {"passed": True, "failed_actions": [], "independently_observed_actions": [0, 1]},
        },
    )
    await ledger.drain_subscribers()
    assert learner.status()["transitions"]["labels"] == 2
    assert learner.status()["workflow_patterns"] == 1


@pytest.mark.asyncio
async def test_local_collaboration_batches_share_one_experience_trace(tmp_path, ledger):
    config = PilotConfig()
    config.security.dry_run = False
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "local-collab-audit.jsonl"),
    )
    executor.set_experience_ledger(ledger)
    action = Action(action_type=ActionType.SYSTEM_INFO, parameters=SystemInfoParams())
    plan = ActionPlan(actions=[action], raw_input="review system status")
    collab = CollabExecutor(mesh=SimpleNamespace(peer_ids=[]), local_executor=executor, enabled=False)
    # Exercise the distributed ingress and its local-only fallback.
    collab.should_distribute = lambda _plan, _batches: True
    executor.set_collab_executor(collab)

    results = await executor.execute(plan, plan_id="local-parent")

    assert len(results) == 1 and results[0].success
    events = await ledger.list_events(plan_id="local-parent")
    assert sum(event.event_type == ExperienceEventType.PLAN_CREATED for event in events) == 1
    assert sum(event.event_type == ExperienceEventType.CANDIDATE_ACTION for event in events) == 1
    assert sum(event.event_type == ExperienceEventType.ACTION_STARTED for event in events) == 1
    assert sum(event.event_type == ExperienceEventType.ACTION_COMPLETED for event in events) == 1


@pytest.mark.asyncio
async def test_approval_request_and_resolution_share_original_task_context(ledger):
    server = PilotServer(PilotConfig())
    server._experience_ledger = ledger
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_DELETE,
                target="C:/safe-test.txt",
                parameters=FileParams(path="C:/safe-test.txt"),
                destructive=True,
            )
        ],
        explanation="delete a file",
    )
    ws = AsyncMock()

    with experience_scope(session_id="session-approval", task_id="task-approval"):
        waiter = asyncio.create_task(server._wait_for_confirmation("approval-plan", plan, ws))
        await asyncio.sleep(0)
        response = await server._handle_confirm(
            {"plan_id": "approval-plan", "confirmed": True},
            ws,
        )
        confirmed, approved, required = await waiter

    assert response == {"status": "ok", "confirmed": True}
    assert confirmed is True
    assert approved == required == {0}
    sent = json.loads(ws.send.await_args.args[0])
    assert sent["method"] == "confirm_required"
    events = await ledger.list_events(plan_id="approval-plan")
    assert [event.event_type for event in events] == [
        ExperienceEventType.APPROVAL_REQUESTED,
        ExperienceEventType.APPROVAL_RESOLVED,
    ]
    assert {event.session_id for event in events} == {"session-approval"}
    assert {event.task_id for event in events} == {"task-approval"}
    assert events[1].payload["decision"] == "approved"


@pytest.mark.asyncio
async def test_empty_interactive_request_still_has_terminal_trace(ledger):
    server = PilotServer(PilotConfig())
    server._experience_ledger = ledger

    result = await server._handle_execute(
        {"input": "", "session_id": "empty-session", "task_id": "empty-task"},
        ws=AsyncMock(),
    )

    assert result == {"status": "error", "message": "Empty input"}
    events = await ledger.list_events(task_id="empty-task")
    assert [event.event_type for event in events] == [
        ExperienceEventType.INTENT,
        ExperienceEventType.OUTCOME_VERIFIED,
    ]
    assert events[1].payload["status"] == "error"


@pytest.mark.asyncio
async def test_unhandled_interactive_error_is_recorded_before_propagation(ledger):
    server = PilotServer(PilotConfig())
    server._experience_ledger = ledger
    server._handle_execute_inner = AsyncMock(side_effect=RuntimeError("planner exploded"))

    with pytest.raises(RuntimeError, match="planner exploded"):
        await server._handle_execute(
            {
                "input": "do something",
                "session_id": "error-session",
                "task_id": "error-task",
            },
            ws=AsyncMock(),
        )

    events = await ledger.list_events(task_id="error-task")
    assert len(events) == 1
    assert events[0].event_type == ExperienceEventType.OUTCOME_VERIFIED
    assert events[0].payload == {
        "error_type": "RuntimeError",
        "message": "planner exploded",
        "status": "internal_error",
    }


@pytest.mark.asyncio
async def test_full_interactive_pipeline_produces_one_causal_trace(ledger, tmp_path):
    config = PilotConfig()
    config.security.dry_run = True
    executor = Executor(
        config,
        ActionValidator(config),
        PermissionChecker(config),
        AuditLogger(audit_file=tmp_path / "full-flow-audit.jsonl"),
    )
    executor.set_experience_ledger(ledger)
    target = tmp_path / "must-not-exist.txt"
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_WRITE,
                target=str(target),
                parameters=FileParams(path=str(target), content="not written in dry run"),
            )
        ],
        explanation="write one file",
        raw_input="write the test file",
    )

    server = PilotServer(config)
    server._experience_ledger = ledger
    server._executor = executor
    server._planner = AsyncMock()
    server._planner.plan.return_value = plan
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = MagicMock()
    server._permission_checker.plan_requires_confirmation.return_value = False
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()

    result = await server._handle_execute(
        {
            "input": "write the test file",
            "session_id": "full-flow-session",
            "task_id": "full-flow-task",
            "dry_run": True,
        },
        ws=AsyncMock(),
    )

    assert result["status"] == "success"
    assert result["dry_run"] is True
    assert not target.exists()
    events = await ledger.list_events(task_id="full-flow-task")
    assert [event.event_type for event in events] == [
        ExperienceEventType.INTENT,
        ExperienceEventType.PLAN_CREATED,
        ExperienceEventType.CANDIDATE_ACTION,
        ExperienceEventType.WORLD_PREDICTION,
        ExperienceEventType.ACTION_STARTED,
        ExperienceEventType.ACTION_COMPLETED,
        ExperienceEventType.OUTCOME_VERIFIED,
        ExperienceEventType.OUTCOME_VERIFIED,
    ]
    candidate = events[2]
    assert candidate.parent_event_id == events[1].event_id
    assert candidate.action_id == events[4].action_id == events[5].action_id
    assert events[6].source == "verifier"
    assert events[7].payload["status"] == "success"


@pytest.mark.asyncio
async def test_proactive_observation_suggestion_and_feedback_are_linked(ledger, tmp_path):
    engine = ProactiveSuggestionEngine(feedback_path=tmp_path / "feedback.json")
    engine.set_experience_ledger(ledger)
    current = SimpleNamespace(
        active_app="PowerShell",
        active_window_title="Fatal error traceback",
    )
    dwell_key = "powershell:fatal error traceback"
    engine._app_dwell_tracker[dwell_key] = time.time() - 60

    await engine._check_patterns(current)
    suggestion = engine._pending_suggestions[0]
    assert await engine.dismiss_suggestion(suggestion.suggestion_id) is True

    events = await ledger.list_events()
    assert [event.event_type for event in events] == [
        ExperienceEventType.OBSERVATION,
        ExperienceEventType.SUGGESTION_SHOWN,
        ExperienceEventType.SUGGESTION_FEEDBACK,
    ]
    assert events[0].payload["raw_media_excluded"] is True
    assert events[1].payload["suggestion_id"] == suggestion.suggestion_id
    assert events[2].payload == {
        "context_app": "powershell",
        "decision": "dismissed",
        "pattern_id": suggestion.pattern_id,
        "priority": suggestion.priority,
        "suggestion_id": suggestion.suggestion_id,
    }


@pytest.mark.asyncio
async def test_gesture_observation_excludes_frontend_landmarks(ledger):
    server = PilotServer(PilotConfig())
    server._experience_ledger = ledger
    server._fusion = AsyncMock()
    server._fusion.on_gesture_event.return_value = None

    result = await server._handle_gesture_event(
        {
            "gesture": "thumbs_up",
            "confidence": 0.91,
            "data": {"hand_landmarks": [[0.1, 0.2, 0.3]], "camera_frame": "pixels"},
        },
        ws=None,
    )

    assert result == {"status": "buffered"}
    events = await ledger.list_events(event_type=ExperienceEventType.OBSERVATION)
    assert len(events) == 1
    assert events[0].payload == {
        "gesture": "thumbs_up",
        "raw_sensor_data_excluded": True,
    }
    assert events[0].confidence == 0.91
