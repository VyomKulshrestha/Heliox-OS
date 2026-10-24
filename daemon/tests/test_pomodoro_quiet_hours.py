"""Pomodoros, quiet hours and "say that again"."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.workflows.focus import FocusSession, QuietHours
from pilot.workflows.reminders import ReminderStore


class Clock:
    def __init__(self, when: datetime) -> None:
        self.when = when

    def now(self) -> datetime:
        return self.when

    def time(self) -> float:
        return self.when.timestamp()


@pytest.mark.asyncio
async def test_quiet_hours_start_a_session_inside_the_window_only(tmp_path):
    clock = Clock(datetime(2026, 10, 26, 8, 59))  # a Monday
    focus = FocusSession(tmp_path / "focus.json", on_end=AsyncMock(), clock=clock.time)
    started = []

    async def on_start(status):
        started.append(status)

    quiet = QuietHours(tmp_path / "quiet.json", focus=focus, on_start=on_start, now=clock.now)
    quiet.add({"days": [0, 1, 2, 3, 4], "start": 9 * 60, "end": 12 * 60})

    assert await quiet.tick() is False
    clock.when = datetime(2026, 10, 26, 9, 0)
    assert await quiet.tick() is True
    assert focus.status()["remaining_seconds"] == 3 * 3600
    assert started[0]["quiet_hours"] == "weekdays 09:00 to 12:00"
    assert await quiet.tick() is False  # already focusing


@pytest.mark.asyncio
async def test_stopping_inside_a_window_keeps_it_off_until_the_window_ends(tmp_path):
    clock = Clock(datetime(2026, 10, 26, 9, 30))
    focus = FocusSession(tmp_path / "focus.json", on_end=AsyncMock(), clock=clock.time)
    quiet = QuietHours(tmp_path / "quiet.json", focus=focus, on_start=AsyncMock(), now=clock.now)
    quiet.add({"days": [0], "start": 9 * 60, "end": 12 * 60})

    assert await quiet.tick() is True
    await focus.stop()
    quiet.note_stopped()

    assert await quiet.tick() is False
    clock.when = datetime(2026, 11, 2, 9, 30)  # next Monday
    assert await quiet.tick() is True


def test_quiet_hours_validation(tmp_path):
    focus = FocusSession(tmp_path / "focus.json", on_end=AsyncMock())
    quiet = QuietHours(tmp_path / "quiet.json", focus=focus, on_start=AsyncMock())

    with pytest.raises(ValueError):
        quiet.add({"days": [0], "start": 600, "end": 500})
    with pytest.raises(ValueError):
        quiet.add({"days": [9], "start": 1, "end": 2})


def _server(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speech_coordinator = MagicMock()
    server._speech_coordinator.speak = AsyncMock(return_value=SpeechOutcome("spoken", SpeechChannel.FINAL_ANSWER, ""))
    server._focus = FocusSession(tmp_path / "focus.json", on_end=server._on_focus_end)
    server._reminders = ReminderStore(tmp_path / "reminders.json")
    server._quiet_hours = QuietHours(tmp_path / "quiet.json", focus=server._focus, on_start=AsyncMock())
    return server


@pytest.mark.asyncio
async def test_a_pomodoro_ends_with_a_break_reminder(tmp_path):
    server = _server(tmp_path)

    reply = await server._control_reply("start a pomodoro")
    assert reply.startswith("Pomodoro started: 25 minutes")
    assert server._focus.status()["remaining_seconds"] > 24 * 60

    server._focus._until = server._focus._clock() + 0.01  # the 25 minutes are up
    summary = await server._focus.stop(reason="time")
    assert "Take a 5-minute break" in summary["text"]
    reminder = server._reminders.list()[0]
    assert reminder["text"].startswith("Break is over")
    assert 4 * 60 < reminder["due"] - server._focus._clock() <= 5 * 60


@pytest.mark.asyncio
async def test_quiet_hours_by_voice(tmp_path):
    server = _server(tmp_path)

    assert await server._control_reply("focus every weekday from 9 to 12") == (
        "Quiet hours set: weekdays 09:00 to 12:00. I'll start focusing then."
    )
    assert await server._control_reply("what is my focus schedule") == "Your quiet hours: weekdays 09:00 to 12:00."
    assert await server._control_reply("clear my quiet hours") == "Cleared 1 quiet-hour window."
    assert await server._control_reply("what is my focus schedule") == "You have no quiet hours."


@pytest.mark.asyncio
async def test_say_that_again_repeats_the_last_spoken_answer(tmp_path):
    server = _server(tmp_path)

    assert await server._control_reply("say that again") == "I haven't said anything yet."
    await server._speak_companion_text("Your disk is 61% full.", channel=SpeechChannel.FINAL_ANSWER)
    assert await server._control_reply("what did you say") == "Your disk is 61% full."


@pytest.mark.asyncio
async def test_quiet_hours_rpcs(tmp_path):
    server = _server(tmp_path)

    added = await server._handle_focus_schedule_add({"days": [5, 6], "start": 600, "end": 720}, MagicMock())
    assert added["status"] == "ok" and added["windows"][0]["days"] == [5, 6]
    assert (await server._handle_focus_schedule_list({}, MagicMock()))["windows"][0]["start"] == 600
    window_id = added["windows"][0]["id"]
    assert (await server._handle_focus_schedule_delete({"id": window_id}, MagicMock()))["windows"] == []
    assert (await server._handle_focus_schedule_add({"days": [], "start": 1, "end": 2}, MagicMock()))[
        "status"
    ] == "error"
