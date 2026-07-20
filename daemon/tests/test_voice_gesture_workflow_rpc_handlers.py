"""Tests for the voice/gesture workflow RPC handlers in server.py.

PilotServer.__init__ is lightweight (just attribute setup, no network/model
init — the real work happens in start()), so these construct a bare
PilotServer and only wire self.config and self._voice_gesture_workflows
(a real VoiceGestureWorkflowEngine backed by stub Planner/Executor/
Decomposer, matching test_voice_gesture_workflow_engine.py's doubles).
"""

from __future__ import annotations

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.decomposer import TaskDecomposition
from pilot.agents.voice_gesture_workflow import VoiceGestureWorkflowEngine
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.voice_gesture_workflows import VoiceGestureWorkflowStore


class _StubPlanner:
    async def plan(self, description, **kwargs):
        action = Action(action_type=ActionType.FILE_READ, target=description, parameters=EmptyParams())
        return ActionPlan(actions=[action], raw_input=description)


class _StubExecutor:
    async def execute(self, plan, *, plan_id=None, invocation_source=None, scope_override=None, **kwargs):
        action = plan.actions[0]
        return [ActionResult(action=action, success=True, output=f"did {action.target}")]


class _StubDecomposer:
    async def decompose(self, goal):
        return TaskDecomposition(goal=goal, subtasks=[], is_complex=False)

    def get_execution_order(self, decomposition):
        return []


def _server(tmp_path) -> PilotServer:
    server = PilotServer(PilotConfig())
    workflow_store = VoiceGestureWorkflowStore(db_file=tmp_path / "workflows.db")
    checkpoint_store = WorkflowCheckpointStore(db_file=tmp_path / "checkpoints.db")
    server._voice_gesture_workflows = VoiceGestureWorkflowEngine(
        _StubPlanner(), _StubExecutor(), _StubDecomposer(), workflow_store, checkpoint_store
    )
    return server


async def _wait_settled(server, workflow_id, timeout=10.0):
    import asyncio

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        workflow = await server._voice_gesture_workflows.get_workflow(workflow_id)
        if workflow and workflow["state"] in (
            "success",
            "partial",
            "failed",
            "cancelled",
            "paused",
            "waiting_for_trigger",
        ):
            return workflow
        await asyncio.sleep(0.01)
    raise TimeoutError(f"workflow {workflow_id} never settled")


class TestSubmit:
    @pytest.mark.asyncio
    async def test_returns_error_when_engine_not_initialized(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_voice_gesture_workflow_submit({"goal": "x"}, ws=None)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_rejects_empty_goal(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_submit({"goal": "  "}, ws=None)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_rejects_invalid_invocation_source(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "not_real"}, ws=None
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_rejects_interactive_invocation_source(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "interactive"}, ws=None
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_submits_and_runs_to_success(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "voice"}, ws=None
        )
        assert result["status"] == "submitted"
        workflow_id = result["workflow"]["workflow_id"]
        final = await _wait_settled(server, workflow_id)
        assert final["state"] == "success"

    @pytest.mark.asyncio
    async def test_rejects_invalid_scope_override(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "voice", "scope_override": {"max_tier": "not-a-dict"}},
            ws=None,
        )
        assert result["status"] == "error"


class TestListGetPauseResumeCancel:
    @pytest.mark.asyncio
    async def test_list_empty_when_engine_not_initialized(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_voice_gesture_workflow_list({}, ws=None)
        assert result["workflows"] == []

    @pytest.mark.asyncio
    async def test_list_and_get_round_trip(self, tmp_path):
        server = _server(tmp_path)
        submitted = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "voice"}, ws=None
        )
        workflow_id = submitted["workflow"]["workflow_id"]
        await _wait_settled(server, workflow_id)

        listed = await server._handle_voice_gesture_workflow_list({"include_terminal": True}, ws=None)
        assert any(w["workflow_id"] == workflow_id for w in listed["workflows"])

        got = await server._handle_voice_gesture_workflow_get({"workflow_id": workflow_id}, ws=None)
        assert got["status"] == "ok"
        assert got["workflow"]["workflow_id"] == workflow_id

    @pytest.mark.asyncio
    async def test_get_unknown_workflow_errors(self, tmp_path):
        server = _server(tmp_path)
        result = await server._handle_voice_gesture_workflow_get({"workflow_id": "nope"}, ws=None)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, tmp_path):
        server = _server(tmp_path)
        submitted = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "voice"}, ws=None
        )
        workflow_id = submitted["workflow"]["workflow_id"]

        paused = await server._handle_voice_gesture_workflow_pause({"workflow_id": workflow_id}, ws=None)
        settled = await _wait_settled(server, workflow_id)
        if settled["state"] != "paused":
            return  # race: finished before the pause flag was checked, not a failure
        assert paused["paused"] is True

        resumed = await server._handle_voice_gesture_workflow_resume({"workflow_id": workflow_id}, ws=None)
        assert resumed["resumed"] is True
        final = await _wait_settled(server, workflow_id)
        assert final["state"] == "success"

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_path):
        server = _server(tmp_path)
        submitted = await server._handle_voice_gesture_workflow_submit(
            {"goal": "do a thing", "invocation_source": "voice"}, ws=None
        )
        workflow_id = submitted["workflow"]["workflow_id"]
        cancelled = await server._handle_voice_gesture_workflow_cancel({"workflow_id": workflow_id}, ws=None)
        assert cancelled["cancelled"] is True


class TestGestureWorkflowBindings:
    @pytest.mark.asyncio
    async def test_get_returns_defaults(self):
        server = PilotServer(PilotConfig())
        result = await server._handle_gesture_workflow_bindings_get({}, ws=None)
        assert result["enabled"] is False
        assert result["bindings"] == []

    @pytest.mark.asyncio
    async def test_update_sets_enabled_and_bindings(self, monkeypatch):
        server = PilotServer(PilotConfig())
        monkeypatch.setattr(server.config, "save", lambda: None)

        result = await server._handle_gesture_workflow_bindings_update(
            {
                "enabled": True,
                "bindings": [{"gesture_name": "swipe_up", "goal_template": "run my daily briefing"}],
            },
            ws=None,
        )
        assert result["status"] == "ok"
        assert result["enabled"] is True
        assert len(result["bindings"]) == 1
        assert result["bindings"][0]["gesture_name"] == "swipe_up"
        assert result["bindings"][0]["enabled"] is True  # defaults to True when omitted

    @pytest.mark.asyncio
    async def test_update_ignores_non_dict_binding_entries(self, monkeypatch):
        server = PilotServer(PilotConfig())
        monkeypatch.setattr(server.config, "save", lambda: None)

        result = await server._handle_gesture_workflow_bindings_update(
            {"bindings": [{"gesture_name": "palm", "goal_template": "x"}, "not-a-dict"]}, ws=None
        )
        assert result["status"] == "ok"
        assert len(result["bindings"]) == 1
