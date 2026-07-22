"""Tests for PilotServer._execute_tracked -- the real, cancellable
asyncio.Task wrapper around Executor.execute() that lets _handle_abort
(Part 3) cancel the CURRENTLY in-flight interactive execution, not just
signal cancel_event for the next action boundary.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


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

    def __init__(self):
        self.sent: list[str] = []

    async def send(self, message):
        self.sent.append(message)


class _FakeReflector:
    async def get_improvement_context(self, user_input):
        return ""


class _FakeMultiAgent:
    def get_routing_summary(self, user_input):
        return {"assigned_agents": []}


class _FakePermissionChecker:
    def plan_requires_confirmation(self, plan):
        return False


def _server_ready_for_handle_execute(executor) -> PilotServer:
    """Builds a PilotServer with just enough wired up to drive
    _handle_execute's fresh-plan path through _execute_tracked, without
    running the real (heavy, ML-loading) PilotServer.initialize()."""
    server = _server()
    server._reflector = _FakeReflector()
    server._multi_agent = _FakeMultiAgent()
    server._permission_checker = _FakePermissionChecker()
    server._executor = executor

    class _FakePlanner:
        async def plan(self, user_input, error_context="", screen_context="", stream_callback=None):
            return MagicMock(error=None, actions=[], explanation="Mocked plan")

    server._planner = _FakePlanner()
    return server


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
    assert executor.cancelled is True
    assert server._active_execution_task is None
