"""Tests for PilotServer._handle_abort -- real mid-flight cancellation
(Issue #92, extended). Prior to this fix, abort only set cancel_event
(a cooperative, boundary-only signal); this locks in that it now also
cancels the currently tracked interactive execution task and interrupts
every live PTY session.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


def _server() -> PilotServer:
    return PilotServer(PilotConfig())


@pytest.mark.asyncio
async def test_abort_cancels_active_execution_task():
    server = _server()
    server._cancel_event = asyncio.Event()

    async def _never_ends():
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_never_ends())
    server._active_execution_task = task

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as mock_interrupt_all:
        result = await server._handle_abort({}, MagicMock())

    assert result == {"status": "aborted"}
    assert server._cancel_event.is_set()
    assert task.cancelled() or task.cancelling() > 0
    mock_interrupt_all.assert_called_once()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_abort_sets_cancel_event_even_with_no_active_task():
    server = _server()
    server._cancel_event = asyncio.Event()
    server._active_execution_task = None

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as mock_interrupt_all:
        result = await server._handle_abort({}, MagicMock())

    assert result == {"status": "aborted"}
    assert server._cancel_event.is_set()
    mock_interrupt_all.assert_called_once()


@pytest.mark.asyncio
async def test_abort_does_not_cancel_an_already_completed_task():
    server = _server()
    server._cancel_event = asyncio.Event()

    async def _quick():
        return "done"

    task = asyncio.ensure_future(_quick())
    await task
    server._active_execution_task = task

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        result = await server._handle_abort({}, MagicMock())

    assert result == {"status": "aborted"}  # cancel_event alone still counts
    assert task.exception() is None
    assert task.result() == "done"


@pytest.mark.asyncio
async def test_abort_reports_no_active_execution_when_nothing_is_running():
    server = _server()
    server._cancel_event = None
    server._active_execution_task = None

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as mock_interrupt_all:
        result = await server._handle_abort({}, MagicMock())

    assert result == {"status": "no_active_execution"}
    mock_interrupt_all.assert_called_once()


@pytest.mark.asyncio
async def test_abort_called_twice_in_a_row_does_not_error():
    server = _server()
    server._cancel_event = asyncio.Event()

    async def _never_ends():
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_never_ends())
    server._active_execution_task = task

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all"):
        first = await server._handle_abort({}, MagicMock())

        # Let the event loop actually deliver the cancellation before the
        # second call, or task.done() would still read False (cancel() only
        # schedules delivery -- it doesn't happen synchronously) and the
        # second call would wrongly count as "aborting" it all over again.
        with pytest.raises(asyncio.CancelledError):
            await task

        second = await server._handle_abort({}, MagicMock())

    assert first == {"status": "aborted"}
    # cancel_event is already set and the task is already done by the
    # second call -- nothing left to newly abort, but it must not raise.
    assert second == {"status": "no_active_execution"}


@pytest.mark.asyncio
async def test_abort_interrupts_pty_sessions_even_when_nothing_else_is_active():
    """PtySessionManager.interrupt_all() must run unconditionally -- a
    pty_exec command can be in flight with no _active_execution_task set
    (e.g. mid-command inside a single _executor.execute() call whose outer
    task hasn't been cancelled yet is not the scenario here, but the PTY
    interrupt is cheap and side-effect-free when nothing is running, so it
    always fires rather than being gated behind aborted_something)."""
    server = _server()
    server._cancel_event = None
    server._active_execution_task = None

    with patch("pilot.system.pty_session.PtySessionManager.interrupt_all") as mock_interrupt_all:
        await server._handle_abort({}, MagicMock())

    mock_interrupt_all.assert_called_once()
