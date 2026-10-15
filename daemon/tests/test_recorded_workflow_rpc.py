"""Recorder RPCs preserve consent and guarded autonomous replay."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from pilot.actions import Action, ActionResult, ActionType, BrowserParams
from pilot.agents.autonomous import JobStatus
from pilot.config import PilotConfig
from pilot.security.rpc_identity import RpcClientRole, rpc_method_allowed
from pilot.server import PilotServer
from pilot.workflows.manual_workflow_hook import ManualWorkflowHook
from pilot.workflows.recorded_workflows import RecordedWorkflowStore


class _Job:
    job_id = "job-1"
    status = JobStatus.QUEUED
    completed_at = 0.0

    def to_dict(self) -> dict:
        return {"job_id": self.job_id}


class _Autonomous:
    def __init__(self) -> None:
        self.submitted = []
        self.job = _Job()

    async def submit(self, prompt: str, **kwargs) -> _Job:
        self.submitted.append((prompt, kwargs))
        return self.job

    def get_job(self, job_id: str) -> _Job | None:
        return self.job if job_id == self.job.job_id else None


@pytest.mark.asyncio
async def test_ui_only_start_and_guarded_replay(tmp_path) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
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


@pytest.mark.asyncio
async def test_manual_recording_requires_separate_consent_and_run_time_text(tmp_path) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    server._autonomous = _Autonomous()
    denied = await server._handle_recorded_workflow_start(
        {"name": "Manual", "consent": True, "capture_manual": True}, None
    )
    assert denied["status"] == "error"
    assert server._recorded_workflows.status()["active"] is False
    server._recorded_workflows.start(name="Manual", consent=True, capture_manual=True, manual_consent=True)
    server._recorded_workflows.record_manual_event("text")
    saved = server._recorded_workflows.stop(save=True)
    missing = await server._handle_recorded_workflow_run({"workflow_id": saved["id"], "reviewed": True}, None)
    assert missing["status"] == "error"
    submitted = await server._handle_recorded_workflow_run(
        {"workflow_id": saved["id"], "reviewed": True, "text_values": {"0": "status report"}}, None
    )
    assert submitted["status"] == "submitted"
    assert "status report" in server._autonomous.submitted[0][0]


@pytest.mark.asyncio
async def test_manual_hook_failure_leaves_recording_off(tmp_path, monkeypatch) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)

    def fail_start(_self):
        raise RuntimeError("OS hook unavailable")

    monkeypatch.setattr(ManualWorkflowHook, "start", fail_start)
    result = await server._handle_recorded_workflow_start(
        {"name": "Manual", "consent": True, "capture_manual": True, "manual_consent": True}, None
    )
    assert result["status"] == "error"
    assert server._recorded_workflows.status()["active"] is False


@pytest.mark.asyncio
async def test_manual_visual_rpc_starts_and_stops_rolling_capture(tmp_path, monkeypatch) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {"embedding": [0.5] * 256, "left": 0, "top": 0, "width": 100, "height": 100}

    monkeypatch.setattr(ManualWorkflowHook, "start", lambda self: None)
    monkeypatch.setattr(ManualWorkflowHook, "stop", lambda self: None)
    monkeypatch.setattr(ManualWorkflowHook, "healthy", lambda self: True)
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    started = await server._handle_recorded_workflow_start(
        {
            "name": "Manual visual",
            "consent": True,
            "capture_manual": True,
            "manual_consent": True,
            "capture_visual": True,
        },
        None,
    )
    assert started["status"] == "ok"
    for _ in range(50):
        if captures:
            break
        await asyncio.sleep(0.01)
    assert captures > 0
    press_feature = server._recorded_workflows.manual_press_feature(time.monotonic())
    assert press_feature is not None
    server._recorded_workflows.record_manual_event("click", x=25, y=75, button="left", press_feature=press_feature)
    saved = await server._handle_recorded_workflow_stop({"save": True}, None)
    assert saved["status"] == "ok"
    assert saved["sample_count"] == 1
    assert server._recorded_workflows._manual_visual_task is None


@pytest.mark.asyncio
async def test_feedback_requires_matching_completed_workflow_job(tmp_path) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    server._autonomous = _Autonomous()
    assert not rpc_method_allowed(RpcClientRole.MCP_LOCAL, "recorded_workflow_feedback")
    server._recorded_workflows.start(name="Review", consent=True)
    action = Action(action_type=ActionType.BROWSER_CLICK, parameters=BrowserParams(selector="#review"))
    token = await server._recorded_workflows.before(action, single_action_batch=True)
    await server._recorded_workflows.after(token, ActionResult(action=action, success=True))
    workflow_id = server._recorded_workflows.stop(save=True)["id"]
    await server._handle_recorded_workflow_run({"workflow_id": workflow_id}, None)
    params = {"workflow_id": workflow_id, "job_id": "job-1", "outcome": "not_achieved"}
    assert (await server._handle_recorded_workflow_feedback(params, None))["status"] == "error"
    server._autonomous.job.status = JobStatus.SUCCESS
    server._autonomous.job.completed_at = time.time()
    assert (await server._handle_recorded_workflow_feedback({**params, "job_id": "other"}, None))["status"] == "error"
    saved = await server._handle_recorded_workflow_feedback(params, None)
    assert saved["status"] == "ok"
    assert saved["workflow"]["feedback_count"] == 1
    assert (await server._handle_recorded_workflow_feedback(params, None))["status"] == "error"


@pytest.mark.asyncio
async def test_replay_visual_capture_is_scoped_and_opt_in(tmp_path) -> None:
    async def capture() -> dict:
        return {"embedding": [0.5] * 256, "left": 0, "top": 0, "width": 100, "height": 100}

    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    server._autonomous = _Autonomous()
    store = server._recorded_workflows
    store.start(name="Review", consent=True)
    action = Action(action_type=ActionType.BROWSER_CLICK, parameters=BrowserParams(selector="#review"))
    token = await store.before(action, single_action_batch=True)
    await store.after(token, ActionResult(action=action, success=True))
    workflow_id = store.stop(save=True)["id"]

    observed: list[str] = []

    async def submit(_prompt: str, **_kwargs):
        observed.append(store._replay_visual_workflow.get())
        return server._autonomous.job

    server._autonomous.submit = submit
    ordinary = await server._handle_recorded_workflow_run({"workflow_id": workflow_id}, None)
    opted_in = await server._handle_recorded_workflow_run({"workflow_id": workflow_id, "capture_visual": True}, None)
    assert ordinary["status"] == opted_in["status"] == "submitted"
    assert observed == ["", workflow_id]
    assert store._replay_visual_workflow.get() == ""


class _Broadcasts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, object]] = []

    async def __call__(self, method: str, params: dict, *, exclude: object = None) -> None:
        self.calls.append((method, params, exclude))

    def status_changes(self) -> list[dict]:
        return [params for method, params, _ in self.calls if method == "recorded_workflow_status_changed"]


@pytest.mark.asyncio
async def test_start_and_stop_announce_the_new_recording_state(tmp_path) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    broadcasts = _Broadcasts()
    server._broadcast_notification = broadcasts

    started = await server._handle_recorded_workflow_start({"name": "Review", "consent": True}, None)
    assert started["status"] == "ok"
    assert broadcasts.status_changes() == [{"active": True}]

    stopped = await server._handle_recorded_workflow_stop({"save": False}, None)
    assert stopped["status"] == "ok"
    assert broadcasts.status_changes() == [{"active": True}, {"active": False}]


@pytest.mark.asyncio
async def test_failed_start_or_stop_announces_nothing(tmp_path, monkeypatch) -> None:
    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    broadcasts = _Broadcasts()
    server._broadcast_notification = broadcasts

    no_consent = await server._handle_recorded_workflow_start({"name": "Review", "consent": False}, None)

    def fail_start(_self):
        raise RuntimeError("OS hook unavailable")

    monkeypatch.setattr(ManualWorkflowHook, "start", fail_start)
    hook_failed = await server._handle_recorded_workflow_start(
        {"name": "Manual", "consent": True, "capture_manual": True, "manual_consent": True}, None
    )
    nothing_to_stop = await server._handle_recorded_workflow_stop({"save": False}, None)

    assert [no_consent["status"], hook_failed["status"], nothing_to_stop["status"]] == ["error", "error", "error"]
    assert broadcasts.status_changes() == []


@pytest.mark.asyncio
async def test_the_announcement_skips_only_the_callers_socket(tmp_path) -> None:
    class _Client:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    server = PilotServer(PilotConfig())
    server._recorded_workflows = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    caller, other = _Client(), _Client()
    server._clients = {caller, other}

    await server._handle_recorded_workflow_start({"name": "Review", "consent": True}, caller)

    assert caller.sent == []
    assert other.sent == [{"jsonrpc": "2.0", "method": "recorded_workflow_status_changed", "params": {"active": True}}]
