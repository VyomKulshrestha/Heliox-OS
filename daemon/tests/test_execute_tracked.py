"""Tests for PilotServer._execute_tracked -- the real, cancellable
asyncio.Task wrapper around Executor.execute() that lets _handle_abort
(Part 3) cancel the CURRENTLY in-flight interactive execution, not just
signal cancel_event for the next action boundary.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    EmptyParams,
    FileParams,
    OpenApplicationParams,
    PowerParams,
    ScreenVisionParams,
    SystemInfoParams,
    VerificationResult,
    WindowParams,
)
from pilot.agents.destructive_critic import CriticVerdict
from pilot.agents.execution_companion import CompanionReview
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.rpc_identity import RpcClientRole
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer
from pilot.system import sysinfo
from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus


class _SlowExecutor:
    """Fake Executor.execute() that blocks until cancelled or released."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def execute(self, plan, **kwargs):
        self.started.set()
        try:
            await self.release.wait()
            return ["done"]
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def _server() -> PilotServer:
    return PilotServer(PilotConfig())


@pytest.mark.asyncio
async def test_execute_tracked_tracks_the_task_while_running():
    server = _server()
    executor = _SlowExecutor()
    server._executor = executor

    # `task` is the OUTER task wrapping _execute_tracked() itself; the task it
    # stores in _active_execution_task is the INNER one wrapping
    # executor.execute() -- these are deliberately two different Task objects.
    task = asyncio.ensure_future(server._execute_tracked(None))
    await executor.started.wait()

    assert server._active_execution_task is not None
    assert not server._active_execution_task.done()

    executor.release.set()
    result = await task
    assert result == ["done"]


@pytest.mark.asyncio
async def test_execute_tracked_clears_slot_after_normal_completion():
    server = _server()
    executor = _SlowExecutor()
    server._executor = executor

    task = asyncio.ensure_future(server._execute_tracked(None))
    await executor.started.wait()
    executor.release.set()
    await task

    assert server._active_execution_task is None


@pytest.mark.asyncio
async def test_cancelling_active_execution_task_propagates_to_executor():
    server = _server()
    executor = _SlowExecutor()
    server._executor = executor

    task = asyncio.ensure_future(server._execute_tracked(None))
    await executor.started.wait()

    server._active_execution_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert executor.cancelled is True
    assert server._active_execution_task is None


@pytest.mark.asyncio
async def test_execute_tracked_clears_slot_even_when_cancelled():
    server = _server()
    executor = _SlowExecutor()
    server._executor = executor

    task = asyncio.ensure_future(server._execute_tracked(None))
    await executor.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert server._active_execution_task is None


@pytest.mark.asyncio
async def test_execute_tracked_passes_through_kwargs():
    server = _server()

    received = {}

    class _Recording:
        async def execute(self, plan, **kwargs):
            received.update(kwargs)
            return []

    server._executor = _Recording()
    await server._execute_tracked(None, plan_id="abc", critic_already_reviewed=True)

    assert received == {"plan_id": "abc", "critic_already_reviewed": True}


class _FakeWs:
    """Minimal stand-in for ServerConnection -- _handle_execute only calls
    ws.send(json_string); nothing needs to actually go anywhere."""

    def __init__(self, confirmation: bool | None = None):
        self.sent: list[str] = []
        self.confirmation = confirmation
        self.server: PilotServer | None = None

    async def send(self, message):
        self.sent.append(message)
        payload = json.loads(message)
        if payload.get("method") == "confirm_required" and self.confirmation is not None:
            assert self.server is not None
            await self.server._handle_confirm(
                {
                    "plan_id": payload["params"]["plan_id"],
                    "confirmed": self.confirmation,
                },
                self,
            )


class _FakeReflector:
    async def get_improvement_context(self, user_input):
        return ""

    async def reflect(self, *args, **kwargs):
        return None


class _FakeMultiAgent:
    def get_routing_summary(self, user_input):
        return {"assigned_agents": []}


class _FakePermissionChecker:
    def __init__(self, requires_confirmation: bool = False):
        self.requires_confirmation = requires_confirmation

    def plan_requires_confirmation(self, plan):
        return self.requires_confirmation


class _FakeMemory:
    async def record(self, *args, **kwargs):
        return None

    async def put_working(self, *args, **kwargs):
        return None


def _server_ready_for_handle_execute(executor, plan: ActionPlan | None = None) -> PilotServer:
    """Builds a PilotServer with just enough wired up to drive
    _handle_execute's fresh-plan path through _execute_tracked, without
    running the real (heavy, ML-loading) PilotServer.initialize()."""
    server = _server()
    server._reflector = _FakeReflector()
    server._multi_agent = _FakeMultiAgent()
    server._permission_checker = _FakePermissionChecker()
    server._executor = executor
    server._memory = _FakeMemory()
    server._orchestrator = None
    server._destructive_critic = None
    server.test_broadcasts = []

    async def _capture_broadcast(method, params):
        server.test_broadcasts.append((method, params))

    server._broadcast_notification = _capture_broadcast

    resolved_plan = plan or ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(),
            )
        ],
        explanation="Mocked plan",
    )

    class _FakePlanner:
        async def plan(self, user_input, error_context="", screen_context="", stream_callback=None, **kwargs):
            return resolved_plan

    server._planner = _FakePlanner()
    return server


@pytest.mark.asyncio
async def test_all_system_info_sections_survive_executor_and_response_serialization(monkeypatch, tmp_path):
    headings = ("System Time", "Operating System", "CPU", "Memory", "Disk", "Network", "Battery")

    async def section(heading):
        return f"=== {heading} ===\n  sample: {heading}"

    monkeypatch.setattr(sysinfo, "get_platform_info", lambda: {"sample": "Operating System"})
    for category, heading in zip(
        ("time", "cpu", "memory", "disk", "network", "battery"),
        ("System Time", "CPU", "Memory", "Disk", "Network", "Battery"),
        strict=True,
    ):
        monkeypatch.setattr(sysinfo, f"_{category}_info", lambda heading=heading: section(heading))

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(categories=["all"]),
            )
        ],
        explanation="Read all system information.",
    )
    server = _server_ready_for_handle_execute(None, plan)
    server._executor = Executor(
        server.config,
        ActionValidator(server.config),
        PermissionChecker(server.config),
        AuditLogger(audit_file=tmp_path / "audit.jsonl"),
    )

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=all(result.success for result in results))

    server._verifier = _Verifier()
    ws = _FakeWs()
    response = await server._handle_execute({"input": "show all system information"}, ws)
    serialized = json.loads(json.dumps(response))
    completed = [
        json.loads(message)["params"]["result"]
        for message in ws.sent
        if json.loads(message).get("method") == "action_complete"
    ]

    assert serialized["status"] == "success"
    assert len(serialized["results"]) == len(completed) == 1
    assert serialized["results"][0] == completed[0]
    assert serialized["results"][0]["action"]["parameters"]["categories"] == ["all"]
    assert serialized["results"][0]["success"] is True
    assert [line for line in serialized["results"][0]["output"].splitlines() if line.startswith("===")] == [
        f"=== {heading} ===" for heading in headings
    ]
    assert "Battery" in serialized["message"]


@pytest.mark.asyncio
async def test_verified_screen_observation_is_grounding_for_the_next_turn():
    screen_plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SCREEN_ANALYZE,
                target="screen",
                parameters=ScreenVisionParams(prompt="Identify what is visible"),
            )
        ],
        explanation="Inspect the current screen.",
    )

    class _Executor:
        async def execute(self, plan, **kwargs):
            return [
                ActionResult(
                    action=plan.actions[0],
                    success=True,
                    output="Two climbers are standing on the Empire State Building antenna.",
                )
            ]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=True, details=["Screen result verified"], failed_actions=[])

    server = _server_ready_for_handle_execute(_Executor(), screen_plan)
    server._verifier = _Verifier()
    await server._handle_execute(
        {"input": "what is on my screen", "session_id": "voice"},
        _FakeWs(),
    )

    captured: dict[str, str] = {}

    class _FollowThroughPlanner:
        async def plan(self, user_input, **kwargs):
            captured["screen_context"] = kwargs["screen_context"]
            return ActionPlan(actions=[], explanation="I can research that subject next.")

    server._planner = _FollowThroughPlanner()
    await server._handle_execute(
        {"input": "find out what this is about", "session_id": "voice"},
        _FakeWs(),
    )

    assert "[RECENT COMPANION CONTEXT]" in captured["screen_context"]
    assert "Two climbers" in captured["screen_context"]
    assert "Empire State Building" in captured["screen_context"]


@pytest.mark.asyncio
async def test_handle_execute_returns_conversation_without_preview_or_execution():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a zero-action conversation must not reach the executor")

    response = "Hello! I’m ready to help."
    server = _server_ready_for_handle_execute(
        _ExecutorThatMustNotRun(),
        ActionPlan(actions=[], explanation=response, raw_input="Hello Heliox"),
    )
    ws = _FakeWs()

    result = await server._handle_execute({"input": "Hello Heliox"}, ws)

    assert result == {
        "status": "success",
        "conversational": True,
        "dry_run": False,
        "results": [],
        "explanation": response,
        "agent_routing": {"assigned_agents": []},
    }
    assert any('"method": "conversation_response"' in message for message in ws.sent)
    assert not any('"method": "plan_preview"' in message for message in ws.sent)
    assert not any('"method": "confirm_required"' in message for message in ws.sent)


@pytest.mark.asyncio
async def test_handle_execute_forwards_chat_session_to_planner():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a zero-action conversation must not reach the executor")

    captured = {}

    class _SessionAwarePlanner:
        async def plan(self, user_input, **kwargs):
            captured.update(kwargs)
            return ActionPlan(actions=[], explanation="Session-aware response", raw_input=user_input)

    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun())
    server._planner = _SessionAwarePlanner()

    result = await server._handle_execute(
        {"input": "Continue this chat", "session_id": "chat-123"},
        _FakeWs(),
    )

    assert result["status"] == "success"
    assert captured["session_id"] == "chat-123"


@pytest.mark.asyncio
async def test_reflection_context_does_not_disable_first_attempt_fast_paths():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a zero-action response must not reach the executor")

    captured = {}

    class _ReflectorWithHistory:
        async def get_improvement_context(self, user_input):
            return "Prefer the dedicated read-only status action."

        async def reflect(self, *args, **kwargs):
            return None

    class _CapturingPlanner:
        async def plan(self, user_input, **kwargs):
            captured.update(kwargs)
            return ActionPlan(actions=[], explanation="Ready", raw_input=user_input)

    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun())
    server._reflector = _ReflectorWithHistory()
    server._planner = _CapturingPlanner()

    result = await server._handle_execute(
        {"input": "Show current battery status without changing anything."},
        _FakeWs(),
    )

    assert result["status"] == "success"
    assert captured["error_context"] == ""
    assert "[PRIOR VERIFIED IMPROVEMENT CONTEXT]" in captured["screen_context"]
    assert "dedicated read-only status action" in captured["screen_context"]


@pytest.mark.asyncio
async def test_handle_execute_returns_clean_response_when_cancelled_mid_flight():
    """End-to-end (bypassing the real, ML-heavy PilotServer.initialize()):
    drives a real 'execute' RPC through _handle_execute and confirms that
    cancelling the tracked task mid-flight -- exactly as Part 3's
    _handle_abort will do -- returns a clean {"status": "cancelled"} dict
    rather than letting the CancelledError escape the RPC handler."""
    executor = _SlowExecutor()
    server = _server_ready_for_handle_execute(executor)
    ws = _FakeWs()

    handle_task = asyncio.ensure_future(server._handle_execute({"input": "do something"}, ws))
    await executor.started.wait()

    # Mirrors _handle_abort's Part-3 ordering: set the cooperative cancel
    # token first, then cancel the tracked task -- by the time the
    # CancelledError reaches _handle_execute's try/except, cancel_event is
    # already set, so it falls through to the pre-existing "Cancel Token"
    # response path instead of needing new response-shaping logic.
    server._cancel_event.set()
    server._active_execution_task.cancel()

    result = await asyncio.wait_for(handle_task, timeout=10)

    assert result["status"] == "cancelled"
    assert "stopped by user" in result["message"]
    assert executor.cancelled is True
    assert server._active_execution_task is None
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1


@pytest.mark.asyncio
async def test_handle_execute_denial_has_one_truthful_terminal_response_and_does_not_execute():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a denied plan must not reach the executor")

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BROWSER_NAVIGATE,
                target="https://example.com",
                parameters=BrowserParams(url="https://example.com"),
            )
        ],
        explanation="Open example.com.",
    )
    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun(), plan)
    server._permission_checker = _FakePermissionChecker(requires_confirmation=True)
    ws = _FakeWs(confirmation=False)
    ws.server = server

    result = await server._handle_execute({"input": "open example.com"}, ws)

    assert result["status"] == "cancelled"
    assert result["message"] == "Cancelled before execution: the plan was denied."
    assert sum('"method": "confirm_required"' in message for message in ws.sent) == 1
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1


@pytest.mark.asyncio
async def test_mcp_source_forces_approval_even_when_normal_policy_does_not():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("an MCP plan denied by the user must not execute")

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target="system",
                parameters=SystemInfoParams(),
            )
        ],
        explanation="Inspect system information.",
    )
    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun(), plan)
    server._permission_checker = _FakePermissionChecker(requires_confirmation=False)
    ws = _FakeWs(confirmation=False)
    ws.server = server

    result = await server._handle_execute(
        {"input": "show system information", "source": "mcp"},
        ws,
    )

    assert result["status"] == "cancelled"
    request = next(json.loads(message) for message in ws.sent if '"method": "confirm_required"' in message)
    assert request["params"]["actions"][0]["index"] == 0
    assert "local MCP client" in request["params"]["reason"]


@pytest.mark.asyncio
async def test_handle_execute_approved_success_reports_verified_outcome():
    class _Executor:
        async def execute(self, plan, **kwargs):
            return [ActionResult(action=plan.actions[0], success=True, output="opened")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=True,
                details=["Action 0 (browser_navigate): VERIFIED"],
                failed_actions=[],
                independently_observed_actions=[0],
            )

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BROWSER_NAVIGATE,
                target="https://example.com",
                parameters=BrowserParams(url="https://example.com"),
            )
        ],
        explanation="Open example.com.",
    )
    server = _server_ready_for_handle_execute(_Executor(), plan)
    server._permission_checker = _FakePermissionChecker(requires_confirmation=True)
    server._verifier = _Verifier()
    ws = _FakeWs(confirmation=True)
    ws.server = server

    result = await server._handle_execute({"input": "open example.com"}, ws)

    assert result["status"] == "success"
    assert result["verification"]["passed"] is True
    assert result["message"] == "opened"
    assert server._recent_companion_context_by_session["default"].startswith("Previous request: open example.com\nExecution result:")
    assert sum('"method": "confirm_required"' in message for message in ws.sent) == 1
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1


@pytest.mark.asyncio
async def test_handle_execute_partial_failure_reports_execution_error_not_plan_intent():
    class _Executor:
        async def execute(self, plan, **kwargs):
            return [ActionResult(action=plan.actions[0], success=False, error="Process exited with code 125")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=False,
                details=["Action 0 (system_info): FAILED — Process exited with code 125"],
                failed_actions=[0],
            )

    server = _server_ready_for_handle_execute(_Executor())
    server._verifier = _Verifier()
    server.MAX_RETRIES = 0
    ws = _FakeWs()

    result = await server._handle_execute({"input": "run the task"}, ws)

    assert result["status"] == "partial_failure"
    assert "Process exited with code 125" in result["message"]
    assert result["message"] != result["explanation"]
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1


@pytest.mark.asyncio
async def test_partial_failure_retries_only_failed_action_and_returns_full_verified_results():
    actions = [
        Action(
            action_type=ActionType.FILE_READ,
            target=f"file{index}.txt",
            parameters=FileParams(path=f"file{index}.txt"),
        )
        for index in range(1, 4)
    ]
    plan = ActionPlan(actions=actions, explanation="Read three files.", raw_input="read the files")

    class _Executor:
        def __init__(self):
            self.targets_by_call: list[list[str]] = []

        async def execute(self, current_plan, **kwargs):
            targets = [action.target for action in current_plan.actions]
            self.targets_by_call.append(targets)
            if len(self.targets_by_call) == 1:
                return [
                    ActionResult(action=actions[0], success=True, output="one"),
                    ActionResult(action=actions[1], success=False, error="Transient read error"),
                    ActionResult(action=actions[2], success=True, output="three"),
                ]
            return [ActionResult(action=current_plan.actions[0], success=True, output="two")]

    class _Verifier:
        async def verify(self, current_plan, results):
            failed = [index for index, result in enumerate(results) if not result.success]
            return VerificationResult(
                passed=not failed,
                details=[
                    f"Action {index}: {'FAILED' if index in failed else 'VERIFIED'}" for index in range(len(results))
                ],
                failed_actions=failed,
            )

    executor = _Executor()
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    planner_calls = 0

    class _Planner:
        async def plan(self, user_input, **kwargs):
            nonlocal planner_calls
            planner_calls += 1
            return plan

    server._planner = _Planner()

    result = await server._handle_execute({"input": "read the files"}, _FakeWs())

    assert result["status"] == "success"
    assert result["verification"]["passed"] is True
    assert [item["action"]["target"] for item in result["results"]] == [
        "file1.txt",
        "file2.txt",
        "file3.txt",
    ]
    assert [item["output"] for item in result["results"]] == ["one", "two", "three"]
    assert executor.targets_by_call == [
        ["file1.txt", "file2.txt", "file3.txt"],
        ["file2.txt"],
    ]
    assert planner_calls == 1


@pytest.mark.asyncio
async def test_missing_file_read_returns_truthful_failure_without_llm_retry():
    class _Executor:
        def __init__(self):
            self.calls = 0

        async def execute(self, plan, **kwargs):
            self.calls += 1
            return [ActionResult(action=plan.actions[0], success=False, error="File not found: C:\\missing.txt")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=False,
                details=["Action 0 (file_read): FAILED — File not found"],
                failed_actions=[0],
            )

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_READ,
                target="C:\\missing.txt",
                parameters=FileParams(path="C:\\missing.txt"),
            )
        ],
        explanation="Read the requested file.",
    )
    executor = _Executor()
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    planner_calls = 0

    class _Planner:
        async def plan(self, user_input, **kwargs):
            nonlocal planner_calls
            planner_calls += 1
            return plan

    server._planner = _Planner()

    result = await server._handle_execute(
        {"input": "Read C:\\missing.txt and do not create it"},
        _FakeWs(),
    )

    assert result["status"] == "partial_failure"
    assert "File not found" in result["message"]
    assert planner_calls == 1
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_missing_application_returns_truthful_failure_without_llm_retry():
    class _Executor:
        def __init__(self):
            self.calls = 0

        async def execute(self, plan, **kwargs):
            self.calls += 1
            return [
                ActionResult(
                    action=plan.actions[0],
                    success=False,
                    error=(
                        "Application 'Definitely Missing Product' was not found in the Start menu, "
                        "PATH, or Windows app registry."
                    ),
                )
            ]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=False,
                details=["Action 0 (open_application): FAILED - application was not found"],
                failed_actions=[0],
            )

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="Definitely Missing Product",
                parameters=OpenApplicationParams(name="Definitely Missing Product"),
            )
        ],
        explanation="Open the requested application.",
    )
    executor = _Executor()
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    planner_calls = 0

    class _Planner:
        async def plan(self, user_input, **kwargs):
            nonlocal planner_calls
            planner_calls += 1
            return plan

    server._planner = _Planner()

    result = await server._handle_execute(
        {"input": "Open Definitely Missing Product"},
        _FakeWs(),
    )

    assert result["status"] == "partial_failure"
    assert "was not found" in result["message"]
    assert planner_calls == 1
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_exhausted_provider_failure_does_not_repeat_planning():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a failed plan must not reach the executor")

    planner_calls = 0

    class _Planner:
        async def plan(self, user_input, **kwargs):
            nonlocal planner_calls
            planner_calls += 1
            return ActionPlan(
                error="Gemini API unavailable (400): the configured API key is invalid.",
                explanation="",
            )

    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun())
    server._planner = _Planner()

    result = await server._handle_execute(
        {"input": "Summarize the current workspace"},
        _FakeWs(),
    )

    assert result["status"] == "error"
    assert "configured API key is invalid" in result["message"]
    assert planner_calls == 1


@pytest.mark.asyncio
async def test_handle_execute_critic_block_is_terminal_and_never_executes():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("a critic-blocked plan must not reach the executor")

    class _BlockingCritic:
        async def review(self, user_input, plan):
            return CriticVerdict(
                verdict="BLOCK",
                risk_score=1.0,
                recommendation="Unsafe power operation.",
            )

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.POWER_SHUTDOWN,
                target="system",
                parameters=PowerParams(),
                requires_root=True,
            )
        ],
        explanation="Shut down the computer.",
    )
    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun(), plan)
    server._destructive_critic = _BlockingCritic()
    ws = _FakeWs()

    result = await server._handle_execute({"input": "shut down"}, ws)

    assert result["status"] == "blocked_by_critic"
    assert result["message"] == "Blocked before execution by safety review: Unsafe power operation."
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1


@pytest.mark.asyncio
async def test_high_heuristic_low_authority_plan_skips_destructive_llm_critic():
    class _Executor:
        async def execute(self, plan, **kwargs):
            return [ActionResult(action=action, success=True, output=f"{action.target} ok") for action in plan.actions]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(
                passed=True,
                details=[f"Action {index}: VERIFIED" for index in range(len(results))],
                failed_actions=[],
            )

    class _CriticThatMustNotRun:
        async def review(self, user_input, plan):
            raise AssertionError("non-destructive plans must not wait on the destructive critic")

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.SYSTEM_INFO,
                target=target,
                parameters=SystemInfoParams(),
            )
            for target in ("os", "cpu", "memory", "disk")
        ],
        explanation="Inspect four system findings.",
    )
    server = _server_ready_for_handle_execute(_Executor(), plan)
    server._verifier = _Verifier()
    server._destructive_critic = _CriticThatMustNotRun()

    result = await server._handle_execute({"input": "inspect four metrics"}, _FakeWs())

    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_live_correction_cancels_current_step_and_replans_in_same_request():
    class _CorrectableExecutor:
        def __init__(self):
            self.calls = 0
            self.first_started = asyncio.Event()
            self.first_cancelled = False

        async def execute(self, plan, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.first_cancelled = True
                    raise
            return [ActionResult(action=plan.actions[0], success=True, output="revised")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=True, details=["Revised action verified"], failed_actions=[])

    executor = _CorrectableExecutor()
    server = _server_ready_for_handle_execute(executor)
    server._verifier = _Verifier()
    planned_inputs: list[str] = []
    original_planner = server._planner

    class _RecordingPlanner:
        async def plan(self, user_input, error_context="", screen_context="", stream_callback=None):
            planned_inputs.append(user_input)
            return await original_planner.plan(user_input, error_context, screen_context, stream_callback)

    server._planner = _RecordingPlanner()
    ws = _FakeWs()

    running = asyncio.create_task(server._handle_execute({"input": "show system information"}, ws))
    await executor.first_started.wait()

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        response = await server._handle_interject(
            {"input": "Also include only the operating-system version."},
            MagicMock(),
        )

    result = await asyncio.wait_for(running, timeout=10)

    assert response["status"] == "revising"
    assert result["status"] == "success"
    assert executor.first_cancelled is True
    assert executor.calls == 2
    assert len(planned_inputs) == 2
    assert "[LIVE USER CORRECTION]" in planned_inputs[1]
    assert "operating-system version" in planned_inputs[1]
    assert any(method == "companion_revision_started" for method, _ in server.test_broadcasts)
    assert [method for method, _ in server.test_broadcasts].count("task_complete") == 1
    assert server._interactive_request_active is False


@pytest.mark.asyncio
async def test_live_correction_reports_no_active_task_instead_of_queueing_work():
    server = _server()

    result = await server._handle_interject({"input": "change the target"}, MagicMock())

    assert result["status"] == "no_active_execution"


@pytest.mark.asyncio
async def test_in_task_question_answers_without_cancelling_execution():
    server = _server()
    server._interactive_request_active = True
    server._active_task_id = "task-running"
    server._cancel_event = asyncio.Event()
    server._model_router = MagicMock()
    server._model_router.generate = AsyncMock(return_value="I can help while the task continues.")

    result = await server._handle_interject(
        {"input": "Can you explain the current plan?", "mode": "ask"}, MagicMock()
    )

    assert result == {
        "status": "answered",
        "message": "I can help while the task continues.",
        "task_id": "task-running",
    }
    assert server._cancel_event.is_set() is False
    assert server._live_correction is None
    server._model_router.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_in_task_answer_uses_personal_context_without_granting_actions():
    server = _server()
    server._interactive_request_active = True
    server._model_router = MagicMock()
    server._model_router.generate = AsyncMock(return_value="You prefer concise updates.")
    server._subconscious = MagicMock()
    server._subconscious.get_persona_context = AsyncMock(return_value="User prefers concise updates")
    server._cancel_event = asyncio.Event()

    result = await server._handle_interject(
        {"input": "How should you update me?", "mode": "ask"}, MagicMock()
    )

    prompt = server._model_router.generate.await_args.args[0]
    assert "User prefers concise updates" in prompt
    assert result["status"] == "answered"
    assert server._cancel_event.is_set() is False


@pytest.mark.asyncio
async def test_in_task_status_question_uses_no_model_or_screen_read():
    server = _server()
    server._interactive_request_active = True
    server._model_router = MagicMock()
    server._model_router.generate = AsyncMock()

    result = await server._handle_interject({"input": "status", "mode": "ask"}, MagicMock())

    assert result["status"] == "answered"
    assert "current task" in result["message"].lower()
    server._model_router.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_task_screen_question_reads_ocr_only_on_explicit_request():
    server = _server()
    server._interactive_request_active = True
    server._model_router = MagicMock()
    server._model_router.generate = AsyncMock()

    with patch("pilot.system.vision.screen_ocr", new_callable=AsyncMock, return_value="Visible document text") as ocr:
        result = await server._handle_interject({"input": "read my screen", "mode": "ask"}, MagicMock())

    assert result["status"] == "answered"
    assert "Visible document text" in result["message"]
    ocr.assert_awaited_once()
    server._model_router.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_interactive_tasks_run_in_order_after_current_task(tmp_path):
    server = _server()
    store = DurableTaskStore(tmp_path / "tasks.db")
    await store.initialize()
    server._durable_tasks = store
    server._broadcast_notification = AsyncMock()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    observed: list[str] = []

    async def execute_inner(params, _ws):
        user_input = params["input"]
        observed.append(user_input)
        if user_input == "first":
            first_started.set()
            await release_first.wait()
        return {"status": "success", "message": f"Completed {user_input}"}

    server._handle_execute_inner = AsyncMock(side_effect=execute_inner)
    ws = _FakeWs()
    server._client_roles[ws] = RpcClientRole.UI
    try:
        active = asyncio.create_task(server._handle_execute({"input": "first", "session_id": "chat-1"}, ws))
        await first_started.wait()
        second = await server._handle_queue_execute({"input": "second", "session_id": "chat-1"}, ws)
        third = await server._handle_queue_execute({"input": "third", "session_id": "chat-1"}, ws)

        assert second["status"] == third["status"] == "queued"
        assert observed == ["first"]
        release_first.set()
        assert (await active)["status"] == "success"
        worker = server._queued_execute_worker
        assert worker is not None
        await asyncio.wait_for(worker, timeout=5)

        assert observed == ["first", "second", "third"]
        assert (await store.get(second["task_id"])).status == TaskStatus.SUCCEEDED
        assert (await store.get(third["task_id"])).status == TaskStatus.SUCCEEDED
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_queued_task_can_be_cancelled_without_affecting_current_task(tmp_path):
    server = _server()
    store = DurableTaskStore(tmp_path / "tasks.db")
    await store.initialize()
    server._durable_tasks = store
    server._interactive_request_active = True
    server._broadcast_notification = AsyncMock()
    ws = _FakeWs()
    server._client_roles[ws] = RpcClientRole.UI
    try:
        queued = await server._handle_queue_execute({"input": "later", "session_id": "chat-1"}, ws)
        denied = await server._handle_cancel_queued_execute(
            {"task_id": queued["task_id"], "resume_token": "wrong"}, ws
        )
        cancelled = await server._handle_cancel_queued_execute(
            {"task_id": queued["task_id"], "resume_token": queued["resume_token"]}, ws
        )

        assert denied["status"] == "error"
        assert cancelled["status"] == "cancelled"
        assert server._interactive_request_active is True
        assert len(server._queued_execute_requests) == 0
        assert (await store.get(queued["task_id"])).status == TaskStatus.CANCELLED
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_external_execute_cannot_claim_reserved_queued_task_slot():
    server = _server()
    worker = asyncio.create_task(asyncio.sleep(10))
    server._queued_execute_worker = worker
    server._queued_reserved_task_id = "queued-1"
    try:
        result = await server._handle_execute({"task_id": "queued-1", "input": "run"}, _FakeWs())
        assert result["status"] == "busy"
        assert server._interactive_request_active is False
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_next_queued_task_stays_reserved_while_previous_result_is_published(tmp_path):
    server = _server()
    store = DurableTaskStore(tmp_path / "tasks.db")
    await store.initialize()
    server._durable_tasks = store
    server._interactive_request_active = True
    first_result = asyncio.Event()
    release_result = asyncio.Event()
    observed: list[str] = []

    async def broadcast(method, params):
        if method == "queued_task_result" and params["task_id"] == first["task_id"]:
            first_result.set()
            await release_result.wait()

    async def execute_inner(params, _ws):
        observed.append(params["input"])
        return {"status": "success", "message": "Done"}

    server._broadcast_notification = AsyncMock(side_effect=broadcast)
    server._handle_execute_inner = AsyncMock(side_effect=execute_inner)
    ws = _FakeWs()
    server._client_roles[ws] = RpcClientRole.UI
    try:
        first = await server._handle_queue_execute({"input": "first"}, ws)
        second = await server._handle_queue_execute({"input": "second"}, ws)
        server._interactive_request_active = False
        server._start_queued_execute_worker()
        await asyncio.wait_for(first_result.wait(), timeout=5)

        assert server._queued_reserved_task_id == second["task_id"]
        outsider = await server._handle_execute({"input": "jump ahead"}, ws)
        assert outsider["status"] == "busy"
        release_result.set()
        worker = server._queued_execute_worker
        assert worker is not None
        await asyncio.wait_for(worker, timeout=5)
        assert observed == ["first", "second"]
    finally:
        release_result.set()
        await store.close()


@pytest.mark.asyncio
async def test_external_execute_cannot_claim_reserved_mcp_task_slot():
    server = _server()
    worker = asyncio.create_task(asyncio.sleep(10))
    server._mcp_tasks["mcp-1"] = worker
    server._mcp_reserved_task_id = "mcp-1"
    try:
        result = await server._handle_execute({"task_id": "mcp-1", "input": "run"}, _FakeWs())
        assert result["status"] == "busy"
        assert server._interactive_request_active is False
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_queue_submissions_obey_ten_task_limit(tmp_path):
    server = _server()
    store = DurableTaskStore(tmp_path / "tasks.db")
    await store.initialize()
    server._durable_tasks = store
    server._interactive_request_active = True
    server._broadcast_notification = AsyncMock()
    ws = _FakeWs()
    server._client_roles[ws] = RpcClientRole.UI
    try:
        results = await asyncio.gather(*(
            server._handle_queue_execute({"input": f"task {index}"}, ws) for index in range(11)
        ))
        assert sum(result["status"] == "queued" for result in results) == 10
        assert sum(result["status"] == "error" for result in results) == 1
        assert len(server._queued_execute_requests) == 10
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_interactive_slot_is_reserved_before_first_setup_await():
    server = _server()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_setup(_params, _ws):
        started.set()
        await release.wait()
        return {"status": "success"}

    server._handle_execute_reserved = AsyncMock(side_effect=delayed_setup)
    first = asyncio.create_task(server._handle_execute({"input": "first"}, _FakeWs()))
    await started.wait()
    second = await server._handle_execute({"input": "second"}, _FakeWs())
    assert second["status"] == "busy"
    release.set()
    assert (await first)["status"] == "success"
    assert server._interactive_request_active is False


@pytest.mark.asyncio
async def test_live_correction_also_cancels_the_orchestrated_execution_path():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("the fake orchestrator owns this test path")

    class _CorrectableOrchestrator:
        def __init__(self):
            self.calls = 0
            self.first_started = asyncio.Event()
            self.first_cancelled = False

        def get_routing_summary(self, plan):
            return {"assigned_agents": [], "is_multi_agent": False}

        async def execute_plan(self, user_input, plan, on_action_complete=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.first_cancelled = True
                    raise
            result = ActionResult(action=plan.actions[0], success=True, output="orchestrated revision")
            if on_action_complete:
                await on_action_complete(result)
            return [result]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=True, details=["Orchestrated revision verified"], failed_actions=[])

    orchestrator = _CorrectableOrchestrator()
    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun())
    server._orchestrator = orchestrator
    server._verifier = _Verifier()
    ws = _FakeWs()

    running = asyncio.create_task(server._handle_execute({"input": "inspect the system"}, ws))
    await orchestrator.first_started.wait()
    assert server._active_execution_task is not None

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        response = await server._handle_interject({"input": "only inspect storage"}, MagicMock())

    result = await asyncio.wait_for(running, timeout=10)

    assert response["status"] == "revising"
    assert result["status"] == "success"
    assert orchestrator.first_cancelled is True
    assert orchestrator.calls == 2
    assert server._active_execution_task is None


@pytest.mark.asyncio
async def test_explicit_stop_interjection_uses_terminal_abort_not_replanning():
    server = _server()
    server._interactive_request_active = True
    server._cancel_event = asyncio.Event()

    async def _never_ends():
        await asyncio.Event().wait()

    task = asyncio.create_task(_never_ends())
    server._active_execution_task = task
    broadcasts: list[tuple[str, dict]] = []

    async def _broadcast(method, params):
        broadcasts.append((method, params))

    server._broadcast_notification = _broadcast
    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        result = await server._handle_interject({"input": "stop this task"}, MagicMock())

    assert result["status"] == "aborted"
    assert server._live_correction is None
    assert server._cancel_event.is_set()
    assert broadcasts[-1][1]["mode"] == "stop"
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_explicit_stop_interrupts_planning_without_waiting_for_model():
    class _ExecutorThatMustNotRun:
        async def execute(self, plan, **kwargs):
            raise AssertionError("stopped planning must not reach execution")

    class _SlowPlanner:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def plan(self, *args, **kwargs):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    server = _server_ready_for_handle_execute(_ExecutorThatMustNotRun())
    planner = _SlowPlanner()
    server._planner = planner
    ws = _FakeWs()
    running = asyncio.create_task(server._handle_execute({"input": "inspect everything"}, ws))
    await planner.started.wait()

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        stop_response = await server._handle_interject({"input": "stop"}, MagicMock())

    result = await asyncio.wait_for(running, timeout=1)

    assert stop_response["status"] == "aborted"
    assert result["status"] == "cancelled"
    assert result["message"] == "Execution stopped by user during planning."
    assert planner.cancelled is True
    assert server._active_execution_task is None


@pytest.mark.asyncio
async def test_proactive_companion_revises_plan_before_any_action_runs():
    class _Executor:
        def __init__(self):
            self.calls = 0

        async def execute(self, plan, **kwargs):
            self.calls += 1
            return [ActionResult(action=plan.actions[0], success=True, output="Windows 11")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=True, details=["OS result verified"], failed_actions=[])

    class _Companion:
        def __init__(self):
            self.calls = 0
            self.contexts = []

        async def review(self, user_input, plan, *, screen_context=""):
            self.calls += 1
            self.contexts.append(screen_context)
            if self.calls == 1:
                return CompanionReview(
                    decision="REVISE",
                    reason="The plan does more work than the request needs.",
                    planner_feedback="Use one direct file-read action and report its output.",
                )
            return CompanionReview(decision="CONTINUE", reason="The revised plan is minimal.")

    executor = _Executor()
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_READ,
                target="notes.txt",
                parameters=FileParams(path="notes.txt"),
            )
        ],
        explanation="Read the requested notes.",
    )
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    companion = _Companion()
    server._execution_companion = companion

    class _ScreenVision:
        def get_context_for_planner(self):
            return 'Currently viewing: Browser — "Requested page"'

        def get_recent_text_context(self):
            return self.get_context_for_planner()

    server._screen_vision = _ScreenVision()
    server._recent_companion_context_by_session["default"] = "Prior private follow-up"
    planned_inputs: list[str] = []
    original_planner = server._planner

    class _RecordingPlanner:
        async def plan(self, user_input, error_context="", screen_context="", stream_callback=None):
            planned_inputs.append(user_input)
            return await original_planner.plan(user_input, error_context, screen_context, stream_callback)

    server._planner = _RecordingPlanner()
    ws = _FakeWs()

    result = await server._handle_execute({"input": "read my notes"}, ws)

    assert result["status"] == "success"
    assert companion.calls == 2
    assert companion.contexts == ['Currently viewing: Browser — "Requested page"'] * 2
    assert executor.calls == 1
    assert len(planned_inputs) == 2
    assert "[INDEPENDENT COMPANION REVIEW]" in planned_inputs[1]
    assert "Use one direct file-read action" in planned_inputs[1]
    assert any(method == "companion_plan_intervention" for method, _ in server.test_broadcasts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_input", "source", "proposed", "corrected", "observed_output"),
    [
        (
            "Show current battery status without changing anything.",
            "mcp",
            Action(
                action_type=ActionType.SYSTEM_INFO, target="system", parameters=SystemInfoParams(categories=["all"])
            ),
            Action(action_type=ActionType.BATTERY_INFO, target="battery", parameters=EmptyParams()),
            "Battery: 73%, plugged in",
        ),
        (
            "Close ChatGPT",
            "text",
            Action(
                action_type=ActionType.OPEN_APPLICATION,
                target="ChatGPT",
                parameters=OpenApplicationParams(name="ChatGPT"),
            ),
            Action(action_type=ActionType.WINDOW_CLOSE, target="ChatGPT", parameters=WindowParams(title="ChatGPT")),
            "Closed ChatGPT window",
        ),
    ],
)
async def test_mismatched_local_plan_revises_to_one_approved_truthful_result(
    user_input, source, proposed, corrected, observed_output
):
    plans = [
        ActionPlan(actions=[proposed], explanation="Proposed action"),
        ActionPlan(actions=[corrected], explanation="Corrected action"),
    ]
    planned_inputs = []
    executed = []

    class _Planner:
        async def plan(self, user_input, **kwargs):
            planned_inputs.append(user_input)
            return plans[len(planned_inputs) - 1]

    class _Executor:
        async def execute(self, plan, **kwargs):
            assert kwargs["user_confirmed"] is True
            executed.append(plan.actions[0].action_type)
            result = ActionResult(action=plan.actions[0], success=True, output=observed_output)
            await kwargs["on_action_start"](plan.actions[0])
            await kwargs["on_action_complete"](result)
            return [result]

    class _Verifier:
        async def verify(self, plan, results):
            assert [result.action.action_type for result in results] == [corrected.action_type]
            return VerificationResult(
                passed=True,
                details=["Observed action verified"],
                independently_observed_actions=[0],
            )

    class _Companion:
        async def review(self, user_input, plan, *, screen_context=""):
            return CompanionReview(decision="CONTINUE", reason="Corrected plan matches the request.")

    server = _server_ready_for_handle_execute(_Executor(), plans[0])
    server._planner = _Planner()
    server._verifier = _Verifier()
    server._execution_companion = _Companion()

    class _PermissionChecker:
        def plan_requires_confirmation(self, plan):
            assert plan.actions[0].action_type == corrected.action_type
            return corrected.action_type == ActionType.WINDOW_CLOSE

    server._permission_checker = _PermissionChecker()
    ws = _FakeWs(confirmation=True)
    ws.server = server

    response = await server._handle_execute({"input": user_input, "source": source}, ws)
    messages = [json.loads(message) for message in ws.sent]

    assert len(planned_inputs) == 2
    assert "[INDEPENDENT COMPANION REVIEW]" in planned_inputs[1]
    assert executed == [corrected.action_type]
    assert [
        message["params"]["decision"] for message in messages if message.get("method") == "companion_plan_review"
    ] == ["REVISE", "CONTINUE"]
    assert len([message for message in messages if message.get("method") == "confirm_required"]) == 1
    assert len([message for message in messages if message.get("method") == "action_complete"]) == 1
    assert response["status"] == "success"
    assert response["message"] == observed_output
    assert len(response["results"]) == 1
    assert response["results"][0]["action"]["action_type"] == corrected.action_type.value
    terminal = [payload for method, payload in server.test_broadcasts if method == "task_complete"]
    assert len(terminal) == 1
    assert terminal[0]["summary"] == observed_output


@pytest.mark.asyncio
async def test_companion_revision_preserves_mcp_approval_boundary():
    class _Executor:
        async def execute(self, plan, **kwargs):
            return [ActionResult(action=plan.actions[0], success=True, output="Windows 11")]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=True, details=["OS result verified"], failed_actions=[])

    class _Companion:
        def __init__(self):
            self.calls = 0

        async def review(self, user_input, plan, *, screen_context=""):
            self.calls += 1
            if self.calls == 1:
                return CompanionReview(
                    decision="REVISE",
                    reason="Use the bounded system information action.",
                    planner_feedback="Keep one direct system-information action.",
                )
            return CompanionReview(decision="CONTINUE", reason="The revised plan is bounded.")

    server = _server_ready_for_handle_execute(_Executor())
    server._verifier = _Verifier()
    server._execution_companion = _Companion()
    server._permission_checker = _FakePermissionChecker(requires_confirmation=False)
    ws = _FakeWs(confirmation=True)
    ws.server = server

    result = await server._handle_execute(
        {"input": "show system information", "source": "mcp"},
        ws,
    )

    assert result["status"] == "success"
    requests = [json.loads(message) for message in ws.sent if '"method": "confirm_required"' in message]
    assert len(requests) == 1
    assert "local MCP client" in requests[0]["params"]["reason"]
