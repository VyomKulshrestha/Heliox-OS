"""A reminder is not done until it reached the person (pilot.workflows.reminders, pilot.server).

CI's frozen-daemon check (tests/platform_real/frozen_daemon_check.py, run 43337605746) froze a real daemon past a
reminder's time on Linux, macOS and Windows: when it was let go the reminder fired with no window open and nothing
said aloud, was removed, and no window that connected afterwards got it. Now one that reaches nobody is held, and the
next window that connects (or the wake, with a window open) gets every held one together, saying when each was due.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.workflows.daily_brief import DailyBriefScheduler
from pilot.workflows.reminders import ReminderStore, describe_held

NINE = datetime(2026, 11, 12, 9, 0)


class Clock:
    def __init__(self, when: datetime) -> None:
        self.t = when.timestamp()

    def __call__(self) -> float:
        return self.t


# ── the store ──


@pytest.mark.asyncio
async def test_one_that_reached_nobody_is_held_and_given_once(tmp_path):
    clock, given, reach = Clock(NINE - timedelta(minutes=1)), [], {"now": False}

    async def deliver(item):
        return reach["now"]

    async def deliver_held(items):
        if not reach["now"]:
            return False
        given.append([item["text"] for item in items])
        return True

    store = ReminderStore(tmp_path / "r.json", deliver=deliver, deliver_held=deliver_held, clock=clock)
    store.add(NINE.timestamp(), "send the report")
    clock.t = NINE.timestamp() + 40  # frozen past it
    await store.tick()
    assert store.list() == [] and [item["text"] for item in store.held()] == ["send the report"]
    assert await store.deliver_held() == 0  # still nobody: still held
    reach["now"] = True
    assert await store.deliver_held() == 1 and given == [["send the report"]]
    assert store.held() == [] and await store.deliver_held() == 0  # once
    await store.tick()
    assert given == [["send the report"]]  # and never fired again


@pytest.mark.asyncio
async def test_several_go_out_together_and_a_repeating_one_keeps_its_time(tmp_path):
    clock, given = Clock(NINE - timedelta(minutes=5)), []

    async def nobody(*_args):
        return False

    async def deliver_held(items):
        given.append(sorted(item["text"] for item in items))
        return True

    store = ReminderStore(
        tmp_path / "r.json", deliver=nobody, deliver_missed=nobody, deliver_held=deliver_held, clock=clock
    )
    store.add(NINE.timestamp(), "send the report")
    store.add(NINE.timestamp() + 60, "stand up", repeat={"minutes": 60})
    clock.t = NINE.timestamp() + 600  # one is "missed" (over two minutes late), one not: both reached nobody
    await store.tick()
    assert len(store.held()) == 2
    assert [item["text"] for item in store.list()] == ["stand up"]  # the next one is still waiting
    assert await store.deliver_held() == 2 and given == [["send the report", "stand up"]]


@pytest.mark.asyncio
async def test_a_reminder_someone_saw_is_not_held(tmp_path):
    clock = Clock(NINE - timedelta(minutes=1))
    store = ReminderStore(
        tmp_path / "r.json", deliver=AsyncMock(return_value=True), deliver_held=AsyncMock(), clock=clock
    )
    store.add(NINE.timestamp(), "send the report")
    clock.t = NINE.timestamp() + 5
    await store.tick()
    assert store.held() == [] and store.list() == []


def test_the_sentence_says_when_each_was_due():
    now = (NINE + timedelta(hours=1)).timestamp()
    assert describe_held([{"due": NINE.timestamp(), "text": "send the report"}], now=now) == (
        "This was due at 9:00 AM: send the report."
    )
    items = [{"due": NINE.timestamp() + 60 * i, "text": f"thing {i}"} for i in range(5)]
    assert describe_held(items, now=now) == (
        "These came due while you were away: at 9:00 AM, thing 0; at 9:01 AM, thing 1; at 9:02 AM, thing 2; and two"
        " more."
    )
    yesterday = (NINE - timedelta(days=1)).timestamp()
    assert describe_held([{"due": yesterday, "text": "x"}], now=now) == "This was due yesterday at 9:00 AM: x."


# ── through the server ──


@pytest.fixture
def server(tmp_path):
    server = PilotServer(PilotConfig())
    sent: list[tuple[str, dict]] = []

    async def broadcast(method, params, **_kwargs):
        if server._clients:  # a notification reaches only the windows that are open
            sent.append((method, params))

    server._broadcast_notification = broadcast
    server._speak_companion_text = AsyncMock(
        return_value=SpeechOutcome("no_voice", SpeechChannel.PROACTIVE_SUGGESTION, "")
    )
    server._sent = sent
    server._reminder_clock = Clock(NINE - timedelta(minutes=1))
    server._reminders = ReminderStore(
        tmp_path / "reminders.json",
        deliver=server._deliver_reminder,
        deliver_missed=server._deliver_missed_reminders,
        deliver_held=server._deliver_held_reminders,
        clock=server._reminder_clock,
    )
    return server


async def _settle(server) -> None:
    for task in list(server.__dict__.get("_notice_tasks", ())):
        await task


@pytest.mark.asyncio
async def test_frozen_with_no_window_then_a_window_connects(server):
    server._reminders.add(NINE.timestamp(), "the frozen daemon check")
    server._reminder_clock.t = NINE.timestamp() + 40
    await server._reminders.tick()  # fired at nobody
    assert server._sent == [] and server._reminders.list() == [] and len(server._reminders.held()) == 1
    tried = server._speak_companion_text.await_count  # it tried to say it, and nothing could
    server._clients.add(object())  # a window connects
    server._on_window_connected()
    await _settle(server)
    method, params = server._sent[-1]
    assert method == "reminder" and params["text"] == "the frozen daemon check" and params["late"] is True
    assert params["message"].startswith("This was due ") and params["message"].endswith(": the frozen daemon check.")
    assert server._reminders.held() == []
    assert server._speak_companion_text.await_count == tried  # the voice is not in use: shown, not said


@pytest.mark.asyncio
async def test_on_wake_with_a_window_open_it_is_shown_at_once(server):
    server._reminders.add(NINE.timestamp(), "send the report")
    server._clients.add(object())
    server._reminder_clock.t = NINE.timestamp() + 40
    await server._on_wake(SimpleNamespace(kind="wake", seconds=100))
    assert [m for m, p in server._sent if m == "reminder" and p["text"] == "send the report"]
    assert server._reminders.held() == [] and server._reminders.list() == []


@pytest.mark.asyncio
async def test_said_aloud_with_no_window_and_no_answer_is_shown_to_the_next_window_as_said(server):
    """A desktop that woke by itself speaks to an empty room: said is not heard."""
    server._voice_listener = SimpleNamespace(is_running=True)
    server._speak_companion_text.return_value = SpeechOutcome("spoken", SpeechChannel.PROACTIVE_SUGGESTION, "")
    server._reminders.add(NINE.timestamp(), "send the report")
    server._reminder_clock.t = NINE.timestamp() + 40
    await server._reminders.tick()
    held = server._reminders.held()
    assert [i["text"] for i in held] == ["send the report"] and held[0]["said_at"]
    spoken_before = server._speak_companion_text.await_count
    server._clients.add(object())
    server._on_window_connected()
    await _settle(server)
    method, params = server._sent[-1]
    assert method == "reminder" and params["message"].startswith("I said this ")
    assert server._speak_companion_text.await_count == spoken_before  # shown, not said again
    assert server._reminders.held() == []


@pytest.mark.asyncio
async def test_answered_after_it_was_said_it_is_done(server):
    server._voice_listener = SimpleNamespace(is_running=True)
    server._speak_companion_text.return_value = SpeechOutcome("spoken", SpeechChannel.PROACTIVE_SUGGESTION, "")
    server._reminders.add(NINE.timestamp(), "send the report")
    server._reminder_clock.t = NINE.timestamp() + 40
    await server._reminders.tick()
    assert len(server._reminders.held()) == 1
    await server._reminder_heard()  # "thanks", or anything addressed to Heliox, within the follow-up window
    assert server._reminders.held() == []


@pytest.mark.asyncio
async def test_late_says_when_it_was_due_and_the_voice_only_when_in_use(server):
    server._clients.add(object())
    server._reminders.add(NINE.timestamp(), "send the report")
    server._reminder_clock.t = NINE.timestamp() + 600  # ten minutes late, delivered on its own (no missed path)
    server._reminders._deliver_missed = None
    await server._reminders.tick()
    method, params = server._sent[-1]
    assert params["late"] is True and params["message"].startswith("This was due ")
    server._speak_companion_text.assert_not_awaited()  # the voice is off: shown, not said
    server._voice_listener = SimpleNamespace(is_running=True)
    server._reminders.add(NINE.timestamp() + 700, "stretch")
    server._reminder_clock.t = NINE.timestamp() + 705  # on time
    await server._reminders.tick()
    assert server._sent[-1][1]["message"] == "Reminder: stretch"
    assert server._speak_companion_text.await_args.args[0] == "Reminder: stretch"


# ── the morning brief ──


@pytest.mark.asyncio
async def test_the_brief_waits_for_someone_the_same_day(tmp_path):
    reach, delivered = {"now": False}, []
    config = SimpleNamespace(enabled=True, time="08:00")

    async def build():
        return {"day": "today", "brief": "Morning."}

    async def deliver(overview):
        delivered.append(overview)
        return reach["now"]

    brief = DailyBriefScheduler(
        config=lambda: config, build=build, deliver=deliver, state_file=tmp_path / "b.json",
        now=lambda: datetime(2026, 11, 12, 8, 30), reachable=lambda: reach["now"],
    )  # fmt: skip
    assert await brief.tick() is False and delivered == []  # nobody there: not built, not marked
    reach["now"] = True
    assert await brief.tick() is True and len(delivered) == 1
    assert await brief.tick() is False  # once a day


@pytest.mark.asyncio
async def test_the_brief_that_reached_nobody_after_all_is_given_again(tmp_path):
    delivered, answers = [], [False, True]
    config = SimpleNamespace(enabled=True, time="08:00")

    async def build():
        return {"day": "today", "brief": "Morning."}

    async def deliver(overview):
        delivered.append(overview)
        return answers.pop(0)

    brief = DailyBriefScheduler(
        config=lambda: config, build=build, deliver=deliver, state_file=tmp_path / "b.json",
        now=lambda: datetime(2026, 11, 12, 8, 30),
    )  # fmt: skip
    assert await brief.tick() is False
    assert await brief.tick() is True and len(delivered) == 2
    assert await asyncio.sleep(0) is None
