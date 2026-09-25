"""Recorder RPCs preserve consent and guarded autonomous replay."""

from datetime import UTC, datetime, timedelta

import pytest

from pilot.actions import Action, ActionResult, ActionType, BrowserParams
from pilot.config import PilotConfig
from pilot.security.rpc_identity import RpcClientRole, rpc_method_allowed
from pilot.server import PilotServer
from pilot.workflows.recorded_workflows import RecordedWorkflowStore


class _Job:
    job_id = "job-1"

    def to_dict(self) -> dict:
        return {"job_id": self.job_id}


class _Autonomous:
    def __init__(self) -> None:
        self.submitted = []

    async def submit(self, prompt: str, **kwargs) -> _Job:
        self.submitted.append((prompt, kwargs))
        return _Job()


@pytest.mark.asyncio
async def test_ui_only_start_and_guarded_replay(tmp_path) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32
    )
    server._autonomous = _Autonomous()
    assert not rpc_method_allowed(RpcClientRole.MCP_LOCAL, "recorded_workflow_start")
    assert not rpc_method_allowed(RpcClientRole.NEURAL_SIDECAR, "recorded_workflow_start")

    denied = await server._handle_recorded_workflow_start({"name": "Review", "consent": False}, None)
    assert denied["status"] == "error"
    started = await server._handle_recorded_workflow_start(
        {"name": "Review", "goal": "Open the review page", "consent": True}, None
    )
    assert started["active"] is True
    action = Action(
        action_type=ActionType.BROWSER_CLICK,
        parameters=BrowserParams(selector="#review"),
    )
    token = await server._recorded_workflows.before(action, single_action_batch=True)
    await server._recorded_workflows.after(token, ActionResult(action=action, success=True))
    saved = await server._handle_recorded_workflow_stop({"save": True}, None)
    assert saved["status"] == "ok"
    workflow_id = saved["id"]

    result = await server._handle_recorded_workflow_run({"workflow_id": workflow_id}, None)
    assert result["status"] == "submitted"
    prompt, options = server._autonomous.submitted[0]
    assert "browser_click" in prompt
    assert "normal permission and confirmation checks" in prompt
    assert options["source"] == "recorded_workflow"

    scheduled = await server._handle_recorded_workflow_schedule(
        {"workflow_id": workflow_id, "run_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat()}, None
    )
    assert scheduled["status"] == "ok"
