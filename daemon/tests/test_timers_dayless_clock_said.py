"""Where know8 meets the earlier series: a timer set by voice never becomes grounds; a to-do with a time and no day,
followed by a decision said back without the wake phrase, gets no invented day in what is said; the time in front of
the model is said the way people say it around noon and midnight. Fakes, scripted replies and the server's own clocks
set by hand; nothing waits on real time."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.fast_model import HttpFastModel
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.todos import TodoStore
from tests.test_addressing_said_back import answered, say

DECISION = "I'm going to watch a series tonight and do the report in the morning"


def _server(tmp_path, now: list[datetime], reply: str = "Okay.") -> tuple[PilotServer, list[str], list[str]]:
    server = PilotServer(PilotConfig())
    sent: list[str] = []
    heard: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def execute(params, ws):
        raise AssertionError(f"planned: {params.get('input')}")

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": reply}}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute = execute
    server._model_router = object()
    server._wall_clock = lambda: now[0]
    server._todos = TodoStore(tmp_path / "todos.json")
    server._reminders = ReminderStore(tmp_path / "reminders.json", clock=lambda: now[0].timestamp())
    model = HttpFastModel(
        provider="claude",
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )
    layer = layer_for(server)
    layer._model = lambda: model
    layer.sources.now = lambda: now[0]
    return server, sent, heard


async def _say(server: PilotServer, words: str, addressed: str = "wake") -> None:
    voice_turn_addressed.set(addressed)
    await server._voice_command_dispatch(words)


# ── a timer set by voice, then talk about a deadline that names a time ──


async def test_a_timer_set_by_voice_is_never_grounds_before_or_after_it_goes_off(tmp_path):
    now = [datetime(2026, 11, 24, 17, 30)]
    server, sent, heard = _server(tmp_path, now)
    await _say(server, "set a timer for ten minutes")
    assert not sent  # set in code
    assert [r.get("kind") for r in server._reminders.list()] == ["timer"]
    await _say(server, "the report is due at nine tomorrow and it's about four hours of work")
    now[0] += timedelta(minutes=11)  # the timer has gone off
    await _say(server, DECISION, "follow_up")
    assert len(sent) == 2
    for body in sent:
        assert "timer" not in body.casefold()
    assert not any("timer" in line.casefold() for line in heard[1:])  # only the code's own "Timer set" said it


# ── a to-do with a time and no day, then a decision said back without the wake phrase ──


async def test_a_dayless_to_do_gets_no_invented_day_in_a_reply_to_a_follow_up(tmp_path):
    now = [datetime(2026, 11, 24, 17, 40)]
    server, sent, heard = _server(tmp_path, now, reply="Your report is due at nine tomorrow morning, so start early.")
    await _say(server, "add finish the quarterly report by nine to my to-do list")
    assert server._todos.open_items()[0]["text"] == "finish the quarterly report by nine"
    rule, clock = answered()
    assert say(rule, clock, DECISION, 1.0).act  # said back without the wake phrase, and taken
    heard.clear()
    await _say(server, DECISION, "follow_up")
    assert "day not said" in sent[-1]
    assert heard and not any("tomorrow" in line.casefold() for line in heard), heard
    # The sentence that gave it a day goes whole, and the question about the day is asked in its place.
    assert "Which day is it due?" in heard, heard


# ── the time in front of the model, around noon and midnight ──


@pytest.mark.parametrize(
    ("hour", "minute", "said"),
    [
        (11, 59, "11:59 in the morning"),
        (12, 0, "noon"),
        (12, 1, "12:01 in the afternoon"),
        (23, 59, "11:59 at night"),
        (0, 0, "midnight"),
        (0, 1, "12:01 at night"),
    ],
)
async def test_the_request_says_the_time_as_people_say_it_around_noon_and_midnight(tmp_path, hour, minute, said):
    now = [datetime(2026, 11, 24, hour, minute)]
    server, sent, _heard = _server(tmp_path, now)
    await _say(server, "what should I cook for dinner?")
    assert f"Tuesday 24 November 2026, {said}." in sent[-1]
    for wrong in ("23:59", "00:00", "00:01", "12:00,"):
        assert f"2026, {wrong}" not in sent[-1]
