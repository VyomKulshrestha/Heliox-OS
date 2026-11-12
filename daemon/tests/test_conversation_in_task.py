"""What a person says while a task they handed over is running: which words reach the task as a correction, which stop
it, which are questions to it. Voice can stop or deny; it never approves."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer


def running_server() -> PilotServer:
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._handle_interject = AsyncMock(return_value={"status": "ok", "message": ""})
    server._interaction_runtime.transition = AsyncMock()
    server._interactive_request_active = True
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "mode"),
    [
        ("No, the other one.", None),  # a correction: the task is replanned with it
        ("Actually make it five.", None),
        ("Not that file.", None),
        ("Wait.", "stop"),  # was a correction "wait" that replanned and carried on
        ("Hold on.", "stop"),
        ("That's enough.", "stop"),
        ("Not that.", "stop"),
        ("No.", "stop"),
        ("Stop.", "stop"),
        ("What are you doing?", "ask"),
    ],
)
async def test_what_reaches_the_running_task(said, mode):
    server = running_server()
    await server._voice_command_dispatch(said)
    server._handle_execute.assert_not_awaited()  # nothing said during a task starts a new request
    server._handle_interject.assert_awaited_once()
    params = server._handle_interject.await_args.args[0]
    assert params.get("mode") == mode
    sent = server._broadcast_notification.await_args.args[1]
    assert sent["coordinated_correction"] is (mode is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Yes.", "Go ahead.", "Approve it.", "Yes, the other one."])
async def test_nothing_said_during_a_task_approves(said):
    server = running_server()
    await server._voice_command_dispatch(said)
    server._handle_execute.assert_not_awaited()
    params = server._handle_interject.await_args.args[0]
    assert params.get("mode") is None  # the correction path, which refuses a go-ahead while an approval waits
    # Approval goes only through the dialog: the correction path refuses a go-ahead while an approval waits
    # (tests/test_interject_approval_guard.py), and nothing here calls confirm.
    assert not any(c.args[0] == "plan_confirmed" for c in server._broadcast_notification.await_args_list)
