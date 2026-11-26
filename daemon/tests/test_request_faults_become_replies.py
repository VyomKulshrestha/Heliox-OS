"""A failure comes back as a reply, never as a crash.

Live run, 26 November: "Do an HTTP GET request to https://example.com/this-page-does-not-exist-404 and tell me the
status" ended as RPC error -32603 "Internal error: 1 validation error for ActionResult" and the window got no reply.
The GET had gone to the RSS agent (it declared api_request), whose answer could not be built, and the exception rose
through the orchestrator and the execute handler to the RPC layer. Now: the GET reaches the Executor, its 404 comes
back as a failed step with the sentence written for it, an agent that raises fails its steps plainly, and any request
that raises ends as a failed result with a plain sentence. Fakes only: no address is fetched.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, EmptyParams
from pilot.agents.agent_mesh import AgentMesh
from pilot.config import PilotConfig
from tests.test_action_routing_contract import _build_agents
from tests.test_failed_actions_are_failures import _executor as _one_handler_executor
from tests.test_failed_actions_are_failures import http  # noqa: F401 - fixture

URL = "https://example.com/this-page-does-not-exist-404"
REQUEST = f"Do an HTTP GET request to {URL} and tell me the status"


class _HandlerExecutor:
    """The shared Executor as far as one handler goes: each step runs through the real ``_execute_single``."""

    def __init__(self, action_type: ActionType, handler: str) -> None:
        self._real = _one_handler_executor(action_type, handler)
        self.ran: list[ActionType] = []

    async def execute(self, plan: ActionPlan, **_kwargs: Any) -> list[ActionResult]:
        results = []
        for action in plan.actions:
            self.ran.append(action.action_type)
            results.append(await self._real._execute_single(action, snapshot_id=None))
        return results


async def test_the_404_get_from_the_run_comes_back_as_a_failed_step_with_its_sentence(http: dict) -> None:  # noqa: F811
    from pilot.agents.orchestrator import AgentOrchestrator
    from pilot.response_contract import partial_failure_message

    http["status"] = 404
    http["text"] = "<html>Not here</html>"
    executor = _HandlerExecutor(ActionType.API_REQUEST, "_exec_api_request")
    orchestrator = AgentOrchestrator(MagicMock(name="model_router"), agent_mesh=AgentMesh())
    try:
        for agent in _build_agents(executor).values():
            orchestrator.register_agent(agent)
        step = Action(
            action_type=ActionType.API_REQUEST, target=URL, parameters=ApiRequestParams(method="GET", url=URL)
        )
        [result] = await orchestrator.execute_plan(REQUEST, ActionPlan(actions=[step], explanation="GET the page"))
    finally:
        await orchestrator.stop()

    assert executor.ran == [ActionType.API_REQUEST]  # the Executor ran it, not the RSS agent
    assert [request[:2] for request in http["requests"]] == [("GET", URL)]
    assert result.success is False
    assert "HTTP 404" in (result.error or "")
    hint = "the server has nothing at this address; the same request will get the same answer"
    assert hint in (result.error or "")
    assert hint in partial_failure_message([result], None)  # what the reply is built from


async def test_an_agent_that_raises_fails_its_steps_plainly_and_later_steps_do_not_run() -> None:
    """The RSS agent's crash, as any agent's: a failed step with a plain error, the plan's later steps not run."""
    from pilot.agents.base_agent import AgentCapability, AgentRole, BaseAgent
    from pilot.agents.orchestrator import AGENT_FAULT_ERROR, AgentOrchestrator

    class _Raises(BaseAgent):
        def __init__(self, action_type: ActionType) -> None:
            super().__init__(role=AgentRole.RSS)
            self._type = action_type

        def get_capabilities(self) -> list[AgentCapability]:
            return [AgentCapability(action_type=self._type, description="raises")]

        def get_system_prompt(self) -> str:
            return ""

        def can_handle(self, action_type: ActionType) -> bool:
            return action_type == self._type

        async def handle_task(self, user_input, plan, context=None, scope_override=None):
            ActionResult(action_type=self._type, success=False, output="unrecognized")  # the run's exact mistake
            raise AssertionError("unreachable")

    class _Runs(_Raises):
        async def handle_task(self, user_input, plan, context=None, scope_override=None):
            raise AssertionError("a step after a failed one must not run")

    orchestrator = AgentOrchestrator(MagicMock(name="model_router"), agent_mesh=AgentMesh())
    try:
        orchestrator.register_agent(_Raises(ActionType.API_REQUEST))
        orchestrator.register_agent(_Runs(ActionType.NOTIFY))
        steps = [
            Action(action_type=ActionType.API_REQUEST, target=URL, parameters=ApiRequestParams(url=URL)),
            Action(action_type=ActionType.NOTIFY, target="done", parameters=EmptyParams()),
        ]
        done: list[ActionResult] = []
        results = await orchestrator.execute_plan(
            REQUEST, ActionPlan(actions=steps), on_action_complete=AsyncMock(side_effect=done.append)
        )
    finally:
        await orchestrator.stop()

    assert [(r.success, r.error, r.skip_reason) for r in results] == [
        (False, AGENT_FAULT_ERROR, None),
        (False, "Skipped due to earlier batch failure", "dependency_failed"),
    ]
    assert done == results[:1]  # the window heard the failed step finish


async def test_a_request_that_raises_is_answered_with_a_failed_result_not_an_internal_error(monkeypatch, tmp_path):
    """Over the socket, as the window sends it: execute, resume_task and the other request-running methods answer a
    raised exception with a failed result and the plain sentence; a method that runs no request keeps -32603."""
    import websockets

    from pilot.server import REQUEST_FAULT_SENTENCE, REQUEST_RUNNING_METHODS, PilotServer

    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "test-token"
    server = PilotServer(config)
    server.initialize = AsyncMock()
    server._broadcast_notification = AsyncMock()

    async def broken(_params, _ws):
        raise ValueError("1 validation error for ActionResult")

    methods = ["execute", "resume_task", "phone_request_run", "scheduled_run_now", "todo_add"]
    assert set(methods[:-1]) <= REQUEST_RUNNING_METHODS and "todo_add" not in REQUEST_RUNNING_METHODS
    await server.start()
    server._handlers = {method: broken for method in methods}
    try:
        port = server._server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": 0}))
            for number, method in enumerate(methods, start=1):
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": {}, "id": number}))
            replies: dict[int, dict] = {}
            while len(replies) < len(methods):
                message = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if message.get("id") in range(1, len(methods) + 1):
                    replies[message["id"]] = message
    finally:
        await server.stop()

    for number, method in enumerate(methods[:-1], start=1):
        answer = replies[number]["result"]
        assert answer["status"] == "error", method
        assert answer["message"] == answer["explanation"] == REQUEST_FAULT_SENTENCE, method
        assert "validation" not in json.dumps(answer), method
    assert replies[len(methods)]["error"]["code"] == -32603


async def test_execute_itself_answers_a_request_that_raised() -> None:
    """The handler, not only the socket layer: voice, the queue and MCP call it directly and speak its message."""
    from pilot.server import REQUEST_FAULT_SENTENCE, PilotServer

    server = PilotServer(PilotConfig())
    server._handle_execute_inner = AsyncMock(side_effect=RuntimeError("1 validation error for ActionResult"))
    answer = await server._handle_execute({"input": REQUEST, "session_id": "s", "task_id": "t"}, ws=AsyncMock())
    assert answer["status"] == "error" and answer["message"] == REQUEST_FAULT_SENTENCE
    assert answer["request_fault"] is True and answer["error_type"] == "RuntimeError"
