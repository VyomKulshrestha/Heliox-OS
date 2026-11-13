"""A reminder counts as shown only in a window that took it, and a window that connects mid-delivery gets it.

CI on a13a496 (Real faults, Ubuntu): the frozen-daemon check's reminder reached no window and was no longer waiting.
Two ways that can happen, both closed here, through the real broadcast and the real store:
- a window closed just before the computer slept (the check closes its first connection, then freezes the daemon):
  its connection was still in the daemon's list when the reminder came due, so "a window is open" was true and the
  reminder was done, sent to a connection that refused it;
- a window connected while a delivery was under way (saying it takes seconds): it asked for the held reminders
  before this one was held, and only the window after it would have got it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.workflows.reminders import ReminderStore

NINE = datetime(2026, 11, 12, 9, 0)


class Clock:
    def __init__(self, when: datetime) -> None:
        self.t = when.timestamp()

    def __call__(self) -> float:
        return self.t


class Window:
    def __init__(self, *, closed: bool = False) -> None:
        self.closed = closed
        self.got: list[dict] = []

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("the window closed")
        self.got.append(json.loads(message))


def _server(tmp_path, clock: Clock) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._reminders = ReminderStore(
        tmp_path / "reminders.json",
        deliver=server._deliver_reminder,
        deliver_missed=server._deliver_missed_reminders,
        deliver_held=server._deliver_held_reminders,
        clock=clock,
    )
    return server


def _reminders(window: Window) -> list[dict]:
    return [message["params"] for message in window.got if message.get("method") == "reminder"]


@pytest.mark.asyncio
async def test_a_window_that_closed_but_is_still_listed_does_not_count_as_shown(tmp_path):
    clock = Clock(NINE - timedelta(minutes=1))
    server = _server(tmp_path, clock)
    server._reminders.add(NINE.timestamp(), "the frozen daemon check")
    server._clients.add(Window(closed=True))  # closed a moment before the sleep; not let go of yet
    clock.t = NINE.timestamp() + 40

    await server._reminders.tick()

    assert [item["text"] for item in server._reminders.held()] == ["the frozen daemon check"]
    server._clients.clear()
    window = Window()
    server._clients.add(window)
    server._on_window_connected()
    for task in list(server.__dict__.get("_notice_tasks", ())):
        await task
    (shown,) = _reminders(window)
    assert shown["late"] is True and shown["message"].startswith("This was due ")
    assert server._reminders.held() == []


@pytest.mark.asyncio
async def test_a_window_that_connects_during_a_delivery_gets_what_it_held(tmp_path):
    clock = Clock(NINE - timedelta(minutes=1))
    server = _server(tmp_path, clock)
    server._reminders.add(NINE.timestamp(), "send the report")
    clock.t = NINE.timestamp() + 40
    window = Window()
    broadcast = server._broadcast_notification

    async def slow_broadcast(method, params, **kwargs):
        handed = await broadcast(method, params, **kwargs)
        if method == "reminder" and window not in server._clients:
            # The window connects after the notification went to nobody, while the delivery is still under way.
            server._clients.add(window)
            server._on_window_connected()
            await asyncio.sleep(0)
        return handed

    server._broadcast_notification = slow_broadcast  # type: ignore[method-assign]

    await server._reminders.tick()
    for task in list(server.__dict__.get("_notice_tasks", ())):
        await task

    (shown,) = _reminders(window)
    assert shown["text"] == "send the report" and shown["late"] is True
    assert server._reminders.held() == [] and server._reminders.list() == []


@pytest.mark.asyncio
async def test_broadcast_counts_the_windows_it_reached(tmp_path):
    server = PilotServer(PilotConfig())
    open_window, closed_window = Window(), Window(closed=True)
    server._clients.update({open_window, closed_window})
    assert await server._broadcast_notification("reminder", {"text": "x"}) == 1
    server._clients.discard(open_window)
    assert await server._broadcast_notification("reminder", {"text": "x"}) == 0


@pytest.mark.asyncio
async def test_a_voice_that_never_finishes_holds_the_lock_a_bounded_time(tmp_path, monkeypatch):
    """A window that connects mid-delivery waits at most REMINDER_SAY_SECONDS per reminder said, never forever: the
    reminder said into a stuck voice counts as not said and is held for that window."""
    from types import SimpleNamespace

    from pilot import server as server_module
    from pilot.config import PilotConfig
    from pilot.server import PilotServer
    from pilot.workflows.reminders import ReminderStore

    monkeypatch.setattr(server_module, "REMINDER_SAY_SECONDS", 0.2)
    server = PilotServer(PilotConfig())
    server._voice_listener = SimpleNamespace(is_running=True)

    async def stuck(*_args, **_kwargs):
        await asyncio.Event().wait()

    server._speak_companion_text = stuck
    shown: list[tuple[str, dict]] = []

    async def broadcast(method, params, **_kwargs):
        if server._clients:
            shown.append((method, params))
            return len(server._clients)
        return 0

    server._broadcast_notification = broadcast
    now = [datetime(2026, 11, 16, 9, 0).timestamp()]
    store = ReminderStore(
        tmp_path / "r.json", deliver=server._deliver_reminder, deliver_missed=server._deliver_missed_reminders,
        deliver_held=server._deliver_held_reminders, clock=lambda: now[0],
    )  # fmt: skip
    server._reminders = store
    store.add(now[0] + 60, "send the report")
    now[0] += 90
    ticking = asyncio.ensure_future(store.tick())
    await asyncio.sleep(0.05)
    server._clients.add(object())  # a window connects while the voice is stuck
    from pilot.timeouts import bounded

    given = await bounded(store.deliver_held(), 5.0)  # bounded: a hang fails the test instead of the run
    await ticking
    assert given == 1 and shown[-1][1]["text"] == "send the report"
