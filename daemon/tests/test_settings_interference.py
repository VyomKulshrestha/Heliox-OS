"""Changing a setting must not break work already in progress."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep
from pilot.config import PilotConfig
from pilot.intelligence.gnomon_reviewer import GnomonReviewer
from pilot.mcp_client import McpClientManager, McpTool
from pilot.models.router import ModelRouter
from pilot.server import PilotServer
from pilot.workflows.scheduled_runs import ScheduledRunner, ScheduledRunStore


async def test_a_job_keeps_the_dry_run_setting_it_started_with():
    config = PilotConfig()
    config.security.dry_run = True
    calls = []

    async def execute(plan, **kwargs):
        calls.append(kwargs)
        return [ActionResult(action=plan.actions[0], success=True, output="simulated")]

    async def plan(goal, **kwargs):
        return ActionPlan(
            actions=[Action(action_type=ActionType.SYSTEM_INFO, target="", parameters=EmptyParams())], explanation="p"
        )

    async def verify(plan, results):
        return SimpleNamespace(passed=True, details=["ok"], independently_observed_actions=[])

    orchestrator = SimpleNamespace(execute_plan=AsyncMock())
    autonomous = AutonomousExecutor(
        planner=SimpleNamespace(plan=plan),
        executor=SimpleNamespace(execute=execute, _config=config),
        verifier=SimpleNamespace(verify=verify),
        decomposer=None,
        orchestrator=orchestrator,
    )
    job = await autonomous.submit("report system info")
    autonomous._active_tasks.pop(job.job_id).cancel()
    config.security.dry_run = False  # the person turns dry run off while the job is running
    step = JobStep(index=0, title="Execute", description=job.goal)

    await autonomous._execute_goal_loop(job, step, job.goal)

    assert job.dry_run is True
    assert calls[0]["dry_run"] is True
    orchestrator.execute_plan.assert_not_awaited()
    assert AutonomousJob.from_record(job.to_record()).dry_run is True


async def test_supervision_reads_nothing_while_switched_off(monkeypatch):
    from pilot.agents import user_supervision

    ocr = AsyncMock(return_value="password reset form")
    monkeypatch.setattr(user_supervision, "screen_ocr", ocr)
    config = PilotConfig()
    config.supervision.enabled = False
    config.supervision.keyboard_mouse_hook_enabled = True
    task = user_supervision.UserSupervisionEngine.__new__(user_supervision.UserSupervisionEngine)
    task._cfg = lambda: config.supervision
    task._hook = SimpleNamespace(snapshot=MagicMock())

    result = await task.tick()

    assert result == {"triggered": False, "signals": []}
    ocr.assert_not_awaited()
    task._hook.snapshot.assert_not_called()


async def test_changing_the_microphone_keeps_voice_tasks_running(monkeypatch):
    server = PilotServer(PilotConfig())
    running = asyncio.create_task(asyncio.sleep(5))

    class FakeListener:
        def __init__(self, **kwargs):
            self.is_running = False
            self.wake_words = []
            self._command_tasks = set()
            self.adopted = set()

        async def start(self):
            self.is_running = True
            return "started"

        async def stop(self, *, cancel_commands=True):
            self.is_running = False
            if cancel_commands:
                for task in self._command_tasks:
                    task.cancel()

        def adopt_command_tasks(self, tasks):
            self.adopted = set(tasks)

    previous = FakeListener()
    previous.is_running = True
    previous._command_tasks = {running}
    server._voice_listener = previous
    monkeypatch.setattr("pilot.system.voice.ContinuousVoiceListener", FakeListener)

    await server._replace_voice_listener(["heliox"])

    assert not running.cancelled() and not running.done()
    assert server._voice_listener.adopted == {running}
    running.cancel()


async def test_a_replaced_model_client_stays_open_for_calls_already_using_it():
    config = PilotConfig()
    router = ModelRouter.__new__(ModelRouter)
    router._config = config
    old_cloud = SimpleNamespace(close=AsyncMock())
    router._cloud = old_cloud
    router._vault = None
    router._clear_cloud_circuit = MagicMock()
    router._rate_limiter = SimpleNamespace(reconfigure=MagicMock())
    config.model.cloud_provider = ""

    await router.reconfigure({"cloud_provider"})

    old_cloud.close.assert_not_awaited()  # a generation in flight keeps working
    router._cancel_llamacpp_idle_unload = AsyncMock()
    router._llamacpp = None
    router._subscription = SimpleNamespace(close=AsyncMock())
    router._cache = SimpleNamespace(close=AsyncMock())
    router._ollama = SimpleNamespace(close=AsyncMock())
    router._cloud = None
    await router.close()
    old_cloud.close.assert_awaited_once()  # shutting down closes it at once


async def test_gnomon_releases_the_model_it_loaded_even_after_the_config_changed():
    loaded = SimpleNamespace(load=MagicMock(), unload=MagicMock(), score=MagicMock())
    current = {"engine": loaded}
    reviewer = GnomonReviewer(lambda: current["engine"])
    reviewer._ensure_idle_watch = lambda: None
    reviewer.warm()
    for _ in range(100):
        if reviewer.status()["state"] == "ready":
            break
        await asyncio.sleep(0.01)
    current["engine"] = None  # switched off in Settings

    assert reviewer.status()["state"] == "disabled"
    await reviewer.unload()
    loaded.unload.assert_called_once()


async def test_deleting_a_schedule_waits_for_a_tick_that_is_starting_it(tmp_path):
    runner = ScheduledRunner(ScheduledRunStore(tmp_path / "s.db"), submit=AsyncMock(), job_is_active=lambda _: False)
    run = await runner.create(name="n", goal="g", allowed_actions=["notify"], every_minutes=10)

    async with runner._lock:  # a tick in progress
        deletion = asyncio.create_task(runner.delete(run.run_id))
        await asyncio.sleep(0.05)
        assert not deletion.done()
    assert await deletion is True
    assert await runner.list() == []


async def test_reloading_mcp_servers_lets_a_running_call_finish(monkeypatch):
    import pilot.mcp_client as mcp_client

    monkeypatch.setattr(mcp_client, "load_server_specs", lambda path=None: [])
    finished = asyncio.Event()

    async def call_tool(name, arguments):
        await asyncio.sleep(0.2)
        finished.set()
        return "done", False

    session = SimpleNamespace(running=True, call_tool=call_tool, close=AsyncMock())
    manager = McpClientManager()
    manager._sessions = {"files": session}
    manager._tools = {("files", "write"): McpTool("files", "write", "", {}, False, False)}

    call = asyncio.create_task(manager.call("files", "write", {}))
    await asyncio.sleep(0.02)
    await manager.refresh()

    assert finished.is_set()  # the call completed before the server was closed
    assert "done" in await call
    session.close.assert_awaited_once()
    with pytest.raises(mcp_client.McpClientError, match="not a tool"):
        await manager.call("files", "write", {})
