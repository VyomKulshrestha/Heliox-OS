"""A task told to stop while it waits ends, on every Python Heliox installs on (3.11 to 3.13).

Before 3.12, asyncio.wait_for could lose a cancellation that arrived as the awaited thing finished, and the task
then waited on: on CI's Windows Python 3.11 the session "a correction, then a stop" hung for 300 s inside the wait
for an approval (reproduced here under 3.11.9 on the first run). The stop paths now use asyncio.timeout."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from pilot.config import PilotConfig
from pilot.server import PendingConfirmation, PilotServer


@pytest.mark.parametrize("set_first", [False, True])
async def test_a_wait_for_approval_that_is_stopped_ends(set_first):
    server = PilotServer(PilotConfig())

    async def nowhere(*_args, **_kwargs):
        return None

    server._broadcast_notification = nowhere
    pending = PendingConfirmation(plan_id="p1", event=asyncio.Event())
    server._pending_confirms["p1"] = pending
    task = asyncio.ensure_future(server._consume_confirmation(pending))
    await asyncio.sleep(0)
    if set_first:
        pending.event.set()  # the denial lands in the same moment as the stop
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=2.0)
    assert done, "a cancelled wait for approval did not end"
    assert "p1" not in server._pending_confirms


def test_no_wait_for_is_left_on_the_stop_and_conversation_paths():
    from pilot.conversation import adapters, layer, turns

    for module in (layer, adapters, turns):
        assert "wait_for(" not in inspect.getsource(module), module.__name__
    assert "wait_for(" not in inspect.getsource(PilotServer._consume_confirmation)
