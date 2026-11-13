"""An answer Heliox gives in code joins the conversation, but reaches a model in the cloud only when it holds none of
the person's own data (sums, units, dates, the time). Checked on the request body the model is sent."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.server import ANSWERED_ON_THIS_COMPUTER, PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.quick_answers import QuickAnswers


def server_with(provider: str) -> tuple[PilotServer, list[str], list[str]]:
    server = PilotServer(PilotConfig())
    sent: list[str] = []
    heard: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._model_router = object()
    server._quick_answers = QuickAnswers(now=lambda: datetime(2026, 11, 12, 9, 0))
    server._wall_clock = lambda: datetime(2026, 11, 12, 9, 0)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        if provider == "ollama":
            body = json.dumps({"message": {"content": "Okay."}, "done": False}) + "\n"
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
    return server, sent, heard


async def say(server: PilotServer, words: str) -> None:
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(words)


@pytest.mark.parametrize(("provider", "shared"), [("claude", False), ("ollama", True)])
async def test_the_days_answer_goes_to_a_local_model_but_never_to_a_cloud_one(provider, shared):
    server, sent, heard = server_with(provider)
    await say(server, "good morning")
    day_answer = heard[-1]
    assert day_answer.startswith("Good morning")
    await say(server, "why is the sky blue?")
    body = sent[-1]
    # The person's words go with the answer or not at all: to a cloud model, only the kind of answer it was.
    assert ("good morning" in body.casefold()) is shared
    assert (day_answer in body) is shared
    assert ("(answered on this computer: the day's plan)" in body) is not shared
    assert ("(said to Heliox, about the day's plan)" in body) is not shared


async def test_two_turns_at_once_each_keep_their_own_answerer():
    """Which handler answered is the turn's own: a sum and the day's answer handled at the same time each go to
    memory by their own rule (the sum as said, the day's answer as answered on this computer)."""
    import asyncio

    from pilot.server import CODE_ANSWER_BY

    server, _sent, heard = server_with("claude")
    seen: dict[str, str] = {}
    original = server._control_reply

    async def control_reply(text: str):
        reply = await original(text)
        await asyncio.sleep(0.01)  # the other turn answers in between
        seen[text] = CODE_ANSWER_BY.get()
        return reply

    server._control_reply = control_reply
    await asyncio.gather(say(server, "what's two plus two"), say(server, "good morning"))
    assert seen == {"what's two plus two": "_local_answer:QuickAnswers", "good morning": seen["good morning"]}
    assert seen["good morning"] not in ("", "_local_answer:QuickAnswers")
    said_by_heliox = [turn.text for turn in layer_for(server).memory.recent() if turn.role == "assistant"]
    assert any("That's 4." in text for text in said_by_heliox)
    assert "(answered on this computer: the day's plan)" in said_by_heliox
    assert not any(text.startswith("Good morning") for text in said_by_heliox)
    assert CODE_ANSWER_BY.get() == ""  # nothing leaks out of the turns into the caller


@pytest.mark.parametrize("provider", ["claude", "ollama"])
async def test_a_sum_answered_in_code_is_part_of_the_conversation_for_every_model(provider):
    server, sent, heard = server_with(provider)
    await say(server, "what's two plus two")
    assert "That's 4." in heard[-1]
    await say(server, "and times ten?")
    assert "That's 4." in sent[-1]
