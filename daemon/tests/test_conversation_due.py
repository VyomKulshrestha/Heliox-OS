"""To-dos and reminders as grounds the talk model can see (pilot.conversation.due): a small line with what is due soon
and bears on the words, under the rule that decides what of the person reaches a cloud model. And the spoken "add ...
to my to-do list" and "remind me ..." are answered in code, never planned."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.due import due_line, todo_due
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.reminders import ReminderStore
from pilot.workflows.todos import TodoStore

NOW = datetime(2026, 11, 24, 18, 30)
REPORT = {"text": "Finish the quarterly report by 9 am tomorrow (about four hours of work left)", "done": False}
DECISION = "I'm going to watch a series tonight and do the report in the morning"


def _reminder(text: str, at: datetime) -> dict:
    return {"id": "r1", "text": text, "due": at.timestamp()}


def test_a_to_do_names_its_own_time():
    assert todo_due(REPORT["text"], NOW) == datetime(2026, 11, 25, 9, 0)
    assert todo_due("buy milk", NOW) is None


def test_a_model_on_this_computer_gets_the_to_do_in_its_words():
    line = due_line(DECISION, todos=[REPORT], reminders=[], now=NOW, local=True)
    assert REPORT["text"] in line and "due tomorrow at 9:00 in the morning" in line


def test_a_cloud_model_gets_only_the_words_the_person_just_said_and_the_time():
    line = due_line(DECISION, todos=[REPORT], reminders=[], now=NOW, local=False)
    assert 'a to-do about "report", due tomorrow at 9:00 in the morning' in line
    assert "quarterly" not in line and "four hours" not in line


def test_only_what_bears_on_the_words():
    pasta = {"text": "Renew the passport", "done": False}
    assert due_line("what's the capital of Australia?", todos=[REPORT, pasta], reminders=[], now=NOW, local=True) == ""
    # A reminder later today bears on a plan for tonight, but not on a question about Canberra.
    call = _reminder("call mum", datetime(2026, 11, 24, 20, 0))
    assert "call mum" in due_line("I'll go to the gym tonight", todos=[], reminders=[call], now=NOW, local=True)
    cloud = due_line("I'll go to the gym tonight", todos=[], reminders=[call], now=NOW, local=False)
    assert "a reminder, due today at 8:00 in the evening" in cloud and "mum" not in cloud
    assert due_line("why is the sky blue?", todos=[], reminders=[call], now=NOW, local=True) == ""


def test_far_off_items_count_only_by_a_shared_word_and_done_or_private_items_never():
    far = {"text": "walk the dog by Friday 4 December", "done": False}
    done = {**REPORT, "done": True}
    health = {"text": "take the medication for the report deadline stress tomorrow", "done": False}
    later = _reminder("dentist review", datetime(2026, 11, 27, 9, 0))
    assert due_line(DECISION, todos=[far, done, health], reminders=[later], now=NOW, local=True) == ""
    shared = {"text": "finish the report by Friday 4 December", "done": False}
    assert "Friday 04 December" in due_line(DECISION, todos=[shared], reminders=[], now=NOW, local=True)


def test_bounded():
    many = [{"text": f"report part {n} by 9 am tomorrow", "done": False} for n in range(6)]
    line = due_line(DECISION, todos=many, reminders=[], now=NOW, local=True)
    assert line.count('to-do "') == 2 and len(line) <= 300


# ── the request body, through the server's own sources ──


def _server(tmp_path, provider: str, share: str = "auto") -> tuple[PilotServer, list[str]]:
    config = PilotConfig()
    config.conversation.share_profile = share
    server = PilotServer(config)
    sent: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def execute(params, ws):
        raise AssertionError(f"planned: {params.get('input')}")

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute = execute
    server._model_router = object()
    server._wall_clock = lambda: NOW
    server._todos = TodoStore(tmp_path / "todos.json")
    server._reminders = ReminderStore(tmp_path / "reminders.json", clock=lambda: NOW.timestamp())

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        if provider == "ollama":
            body = json.dumps({"message": {"content": "Okay."}, "done": True}) + "\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Okay."}}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    from pilot.conversation.fast_model import HttpFastModel

    model = HttpFastModel(
        provider=provider,
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
    )
    layer_for(server)._model = lambda: model
    return server, sent


async def _say(server: PilotServer, words: str) -> None:
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(words)


@pytest.mark.parametrize(("provider", "share", "expected"), [
    ("claude", "auto", 'a to-do about \\"report\\", due tomorrow at 9:00 in the morning'),
    ("ollama", "auto", "four hours of work left"),
    ("claude", "off", None),
])  # fmt: skip
async def test_the_to_do_added_by_voice_reaches_the_request_where_it_bears(tmp_path, provider, share, expected):
    server, sent = _server(tmp_path, provider, share)
    await _say(
        server, "add finish the quarterly report by 9 am tomorrow (about four hours of work left) to my to-do list"
    )
    assert not sent  # answered in code: no model, no planner
    assert server._todos.open_items()[0]["text"].startswith("finish the quarterly report")
    await _say(server, DECISION)
    body = sent[-1]
    if expected is None:
        assert "to-do about" not in body and 'report\\", due' not in body
    else:
        assert expected in body
    if provider == "claude":
        assert "quarterly" not in body.split("[The person said]")[0].split("[Context")[-1]
    await _say(server, "what's the capital of Australia?")
    assert "On their to-do list" not in sent[-1]  # nothing about the report bears on Canberra


async def test_remind_me_is_set_in_code_not_planned(tmp_path):
    server, sent = _server(tmp_path, "claude")
    await _say(server, "remind me tomorrow at 4 pm to call the dentist")
    await _say(server, "remind me on Monday 30 November to call the dentist")
    assert not sent
    assert len(server._reminders.list()) >= 1
