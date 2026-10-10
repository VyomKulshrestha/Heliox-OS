"""Autonomous work streams each tool call to the chat with redacted arguments and an output preview."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, FileParams
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator


def _autonomous(**kwargs) -> AutonomousExecutor:
    autonomous = AutonomousExecutor(planner=None, executor=None, verifier=None, decomposer=None, **kwargs)
    autonomous.set_broadcast(AsyncMock())
    return autonomous


def _events(autonomous: AutonomousExecutor) -> list[dict]:
    return [call.args[1] for call in autonomous._broadcast.await_args_list if call.args[0] == "agent_tool_call"]


async def test_start_and_complete_share_a_call_id_and_hide_credentials():
    autonomous = _autonomous()
    job = AutonomousJob(job_id="j1", goal="check the API", source="scheduled")
    on_start, on_complete = autonomous._tool_call_reporter(job, JobStep(index=0, title="t", description="d"), 1)
    action = Action(
        action_type=ActionType.API_REQUEST,
        target="https://api.example.com/status",
        parameters=ApiRequestParams(
            url="https://api.example.com/status", headers={"Authorization": "Bearer sk-abcdefghijklmnop123456"}
        ),
    )

    await on_start(action)
    await on_complete(ActionResult(action=action, success=True, output="status ok " + "x" * 2000))

    start, complete = _events(autonomous)
    assert start["phase"] == "start" and complete["phase"] == "complete"
    assert start["call_id"] == complete["call_id"]
    assert (start["job_id"], start["round"], start["source"]) == ("j1", 2, "scheduled")
    assert start["action_type"] == "api_request" and start["tier"] == "read_only"
    assert "api.example.com/status" in start["arguments"]
    assert "sk-abcdefghijklmnop123456" not in start["arguments"]
    assert complete["success"] is True and complete["output"].startswith("status ok")
    assert len(complete["output"]) <= 600
    assert complete["duration_ms"] >= 0


async def test_a_call_refused_before_it_started_still_gets_a_card():
    autonomous = _autonomous()
    job = AutonomousJob(job_id="j2", goal="g")
    _on_start, on_complete = autonomous._tool_call_reporter(job, JobStep(index=0, title="t", description="d"), 0)
    action = Action(action_type=ActionType.FILE_DELETE, target="C:/x", parameters=FileParams(path="C:/x"))

    await on_complete(ActionResult(action=action, success=False, error="denied by policy", executed=False))

    [event] = _events(autonomous)
    assert event["success"] is False and event["executed"] is False
    assert event["error"] == "denied by policy" and event["call_id"]


async def test_the_real_executor_reports_every_action_through_the_callbacks(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("buy milk", encoding="utf-8")
    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    autonomous = _autonomous()
    job = AutonomousJob(job_id="j3", goal="read my notes")
    on_start, on_complete = autonomous._tool_call_reporter(job, JobStep(index=0, title="t", description="d"), 0)
    action = Action(action_type=ActionType.FILE_READ, target=str(notes), parameters=FileParams(path=str(notes)))

    results = await executor.execute(
        ActionPlan(actions=[action], explanation="read"), on_action_start=on_start, on_action_complete=on_complete
    )

    assert results[0].success
    start, complete = _events(autonomous)
    assert start["call_id"] == complete["call_id"]
    assert "buy milk" in complete["output"]
    assert str(notes).replace("\\", "\\\\") in start["arguments"] or "notes.txt" in start["arguments"]


async def test_the_goal_loop_passes_the_reporter_to_the_executor():
    seen = {}

    async def plan(goal, **_kwargs):
        return ActionPlan(
            actions=[Action(action_type=ActionType.SYSTEM_INFO, target="", parameters=FileParams(path=""))],
            explanation="info",
        )

    async def execute(plan, *, on_action_start, on_action_complete, **_kwargs):
        seen["callbacks"] = (on_action_start, on_action_complete)
        await on_action_start(plan.actions[0])
        result = ActionResult(action=plan.actions[0], success=True, output="8 cores")
        await on_action_complete(result)
        return [result]

    async def verify(plan, results):
        return SimpleNamespace(passed=True, details=["ok"], independently_observed_actions=[])

    autonomous = AutonomousExecutor(
        planner=SimpleNamespace(plan=plan),
        executor=SimpleNamespace(execute=execute),
        verifier=SimpleNamespace(verify=verify),
        decomposer=None,
    )
    autonomous.set_broadcast(AsyncMock())
    job = AutonomousJob(job_id="j4", goal="system info")
    step = JobStep(index=0, title="Execute", description=job.goal)

    await autonomous._execute_goal_loop(job, step, job.goal)

    assert step.status == "success"
    assert [event["phase"] for event in _events(autonomous)] == ["start", "complete"]
