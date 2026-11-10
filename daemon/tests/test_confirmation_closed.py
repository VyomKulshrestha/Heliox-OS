"""Every answer to a pending approval, including one the dialog never gave, tells the UI to close that dialog."""

from __future__ import annotations

import asyncio

import pytest

from tests.test_confirmation_unanswered import _server, _start_waiter


def _closed(server) -> list[dict]:
    return [
        call.args[1]
        for call in server._broadcast_notification.await_args_list
        if call.args and call.args[0] == "confirmation_closed"
    ]


@pytest.mark.asyncio
async def test_a_correction_closes_the_dialog_for_the_plan_it_withdrew():
    server = _server()
    waiter, _pending = await _start_waiter(server)
    assert _closed(server) == []

    assert await server._resolve_pending_denied("plan-1", resolution_path="interject_correction")
    assert (await asyncio.wait_for(waiter, 2.0))[0] is False
    assert _closed(server) == [{"plan_id": "plan-1", "resolution": "interject_correction"}]


@pytest.mark.asyncio
async def test_stop_closes_the_dialog():
    server = _server()
    waiter, _pending = await _start_waiter(server)
    assert await server._resolve_pending_denied("plan-1", resolution_path="abort")
    await asyncio.wait_for(waiter, 2.0)
    assert _closed(server) == [{"plan_id": "plan-1", "resolution": "abort"}]


@pytest.mark.asyncio
async def test_the_timeout_closes_the_dialog(monkeypatch):
    monkeypatch.setattr("pilot.server.CONFIRM_TIMEOUT_SECONDS", 0.05)
    server = _server()
    waiter, _pending = await _start_waiter(server)
    assert (await asyncio.wait_for(waiter, 2.0))[0] is False
    assert _closed(server) == [{"plan_id": "plan-1", "resolution": "timeout"}]


@pytest.mark.asyncio
async def test_an_answer_from_the_dialog_closes_it_once():
    server = _server()
    waiter, _pending = await _start_waiter(server)
    reply = await server._handle_confirm({"plan_id": "plan-1", "confirmed": False, "input_channel": "dialog"}, None)
    assert reply["status"] == "ok"
    await asyncio.wait_for(waiter, 2.0)
    assert _closed(server) == [{"plan_id": "plan-1", "resolution": "confirm_rpc"}]
