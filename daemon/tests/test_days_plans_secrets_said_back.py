"""Where know7 meets the earlier series: a to-do with a weekday said by voice and a follow-up without the wake phrase
days before it is due; "Yes." to the person's own plan said back without the wake phrase; a secret in a turn that
also asks to note it. Fakes, scripted replies and the server's wall clock set by hand; nothing waits on real time."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.layer import GIVE_WAY, SECRET_LEFT_OUT, SECRET_NOT_USED
from pilot.conversation.withheld import forget_recent
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.notes import NoteStore
from pilot.workflows.todos import TodoStore
from tests.test_addressing_said_back import answered, say
from tests.test_conversation_due import _server as _due_server
from tests.test_conversation_own_plan_and_days import PLAN, Replies
from tests.test_conversation_secrets import shows

TUESDAY = datetime(2026, 11, 24, 8, 30)
DECISION = "I'm going to watch a series tonight and do the report in the morning"


@pytest.fixture(autouse=True)
def _fresh():
    forget_recent()
    yield
    forget_recent()


# ── a to-do with a weekday, then a follow-up without the wake phrase days before it is due ──


async def test_a_weekday_to_do_is_grounds_with_its_own_day_for_a_follow_up_days_ahead(tmp_path):
    server, sent = _due_server(tmp_path, "claude")
    server._wall_clock = lambda: TUESDAY
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("add finish the quarterly report by nine on Friday morning to my to-do list")
    assert not sent  # kept in code
    assert server._todos.open_items()[0]["text"] == "finish the quarterly report by nine on Friday morning"
    rule, clock = answered()
    assert say(rule, clock, DECISION, 1.2).act  # said back without the wake phrase, and taken
    voice_turn_addressed.set("follow_up")
    await server._voice_command_dispatch(DECISION)
    body = sent[-1]
    assert "due Friday 27 November at 9:00 in the morning" in body
    grounds = body[body.index("to-do about") :][:160]
    assert "tomorrow" not in grounds and "today" not in grounds  # never a day the to-do does not carry


# ── "Yes." to the person's own plan, said back without the wake phrase ──


def _server(model, tmp_path) -> tuple[PilotServer, list[str], list[str]]:
    server = PilotServer(PilotConfig())
    heard: list[str] = []
    planned: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def execute(params, ws):
        planned.append(str(params.get("input")))
        return {"status": "success", "message": ""}

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute = execute
    server._model_router = object()
    server._notes = NoteStore(tmp_path / "notes.json")
    server._todos = TodoStore(tmp_path / "todos.json")
    layer_for(server)._model = lambda: model
    return server, heard, planned


async def test_yes_said_back_to_their_own_plan_holds_it_and_nothing_reaches_the_planner(tmp_path):
    act = (
        'Okay, your call.\nACT {"verb": "delete", "object": "the thesis", '
        '"request": "Delete the only copy of the thesis"}'
    )
    model = Replies(
        "That's your only copy of the thesis, so deleting it can't be undone. Do you have a backup?"
        "\nSTANCE: only thesis copy",
        act,
    )
    server, heard, planned = _server(model, tmp_path)
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(PLAN)
    rule, clock = answered(asked=True)  # Heliox questioned their plan and asked
    assert say(rule, clock, "Yes.", 1.0).act  # taken without the wake phrase
    voice_turn_addressed.set("follow_up")
    heard.clear()
    await server._voice_command_dispatch("Yes")
    assert planned == []  # their plan is theirs: nothing is handed over, even with an ACT line written
    assert heard == [GIVE_WAY]


# ── a secret in a turn that also asks to note it ──


@pytest.mark.parametrize("words", ["My PIN is 4471, note that down.", "Note that my door code is 2580."])
async def test_a_secret_asked_to_be_noted_is_left_out_and_never_noted(words, tmp_path, caplog):
    model = Replies("Got it, noted.", "Got it, noted.")
    server, heard, planned = _server(model, tmp_path)
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(words)
    assert server._notes.list() == [] and planned == []
    assert heard and all("noted" not in line.casefold() for line in heard), heard
    assert heard[-1] in (SECRET_LEFT_OUT, SECRET_NOT_USED), heard
    value = "4471" if "4471" in words else "2580"
    for place in (*heard, caplog.text, json.dumps(server._notes.list())):
        assert not shows(value, place), place
    assert not any(shows(value, turn.text) for turn in layer_for(server).memory.recent())
