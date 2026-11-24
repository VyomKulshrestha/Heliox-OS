"""A secret said aloud (a PIN, a password, a code, a card number, a key) stays on this computer: no request to any
model holds it, in that turn or any later one, nothing keeps it, and a request to do something with it is not handed
over. Checked on the exact request body each provider would be sent (pilot.conversation.withheld)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.context import ConversationMemory, build_messages
from pilot.conversation.fast_model import HttpFastModel, RouterFastModel
from pilot.conversation.layer import SECRET_NOT_USED, ConversationLayer
from pilot.conversation.withheld import withhold
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed

PIN = "4471"


@pytest.mark.parametrize(
    ("said", "kind", "secret"),
    [
        ("oh, and my bank pin is 4471", "a PIN", "4471"),
        ("Oh, and my bank PIN is four four seven one.", "a PIN", "four four seven one"),
        ("my wifi password is swordfish", "a password", "swordfish"),
        ("my card number is 4111 1111 1111 1111", "a card number", "4111 1111 1111 1111"),
        ("the door code is 2580, remind me later", "a code", "2580"),
        ("my account number is 12345678 and the sort code is 20-00-00", "an account number", "12345678"),
        ("here's the key sk-ant-abcdefghijklmnopqrstuvwxyz123", "a key", "sk-ant-abcdefghijklmnopqrstuvwxyz123"),
        ("4111 1111 1111 1111", "a card number", "4111 1111 1111 1111"),
    ],
)
def test_what_is_a_secret_said_aloud(said, kind, secret):
    found = withhold(said)
    assert found is not None and found.kind == kind
    assert secret not in found.text
    assert f"[left out: {kind}]" in found.text


@pytest.mark.parametrize(
    "said",
    [
        "what's the difference between a PIN and a password?",
        "my password is wrong",
        "my PIN is on a card in my wallet",
        "pin the 3 tabs",
        "I need to login at 9:30",
        "my report is due at 9 tomorrow and it's about four hours of work",
        "call 555 1234 please",
        "the meeting is in room 4471",
        "I have two reports on my desktop, q3-summary.docx and q4-summary.docx.",
    ],
)
def test_ordinary_words_are_not_taken_out(said):
    assert withhold(said) is None


def test_the_conversation_keeps_only_the_left_out_form_and_forgets_the_value_with_a_new_conversation():
    memory = ConversationMemory()
    memory.add_user("oh, and my bank pin is 4471")
    memory.add_heliox("I won't keep 4471 or say it again.")  # a reply that repeated it
    memory.add_user("was it 4471?")  # said again without naming it
    texts = [turn.text for turn in memory.recent()]
    assert not any(PIN in text for text in texts)
    assert texts[0] == "oh, and my bank pin is [left out: a PIN]"
    memory.clear()
    memory.add_user("room 4471 please")  # a new conversation: the value is no longer known
    assert memory.recent()[-1].text == "room 4471 please"


# ── every provider path, the exact request body ──

PROVIDERS = ["claude", "openai", "openrouter", "gemini", "ollama"]


def _http_model(provider: str, sent: list[str]) -> HttpFastModel:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        if provider == "ollama":
            body = json.dumps({"message": {"content": "Okay."}, "done": True}) + "\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})
        if provider == "gemini":
            event = {"candidates": [{"content": {"parts": [{"text": "Okay."}]}}]}
        elif provider == "claude":
            event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Okay."}}
        else:
            event = {"choices": [{"delta": {"content": "Okay."}}]}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    return HttpFastModel(
        provider=provider,
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
    )


class _Router:
    def __init__(self, sent: list[str]) -> None:
        self.sent = sent

    async def generate(self, messages, system="", temperature=0.3):
        self.sent.append(json.dumps({"system": system, "messages": messages}))
        return "Okay."


async def _silent(text, channel, key):
    return "spoken"


async def _two_turns(model) -> ConversationLayer:
    layer = ConversationLayer(sink=_silent, model=lambda: model, disagree=lambda: "when_it_matters")
    await layer.take("Oh, and my bank PIN is 4471.")
    await layer.take("What's the capital of Australia?")
    return layer


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_no_request_to_any_provider_holds_the_pin(provider):
    sent: list[str] = []
    await _two_turns(_http_model(provider, sent))
    assert len(sent) >= 2  # the turn it was said in, and the next one
    assert not any(PIN in body for body in sent)
    assert "[left out: a PIN]" in sent[0]  # in its own turn: the words with the value taken out
    assert "[left out: a PIN]" in sent[-1]  # later: in the history, so the conversation still reads in order
    assert "they said a secret" in sent[-1]  # and the model is told what the marker means


async def test_no_request_through_the_model_router_holds_the_pin():
    sent: list[str] = []
    await _two_turns(RouterFastModel(router=_Router(sent)))
    assert len(sent) == 2 and not any(PIN in body for body in sent)


async def test_a_request_to_do_something_with_it_is_not_handed_over():
    sent: list[str] = []
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: _http_model("claude", sent))
    result = await layer.take("type my PIN 4471 into the bank form")
    assert result.done and result.intent is None and result.context is None
    assert said == [SECRET_NOT_USED]
    assert PIN not in result.planner_input
    assert not sent  # no model was asked either


async def test_an_act_line_written_in_that_turn_hands_nothing_over():
    class _Acts:
        local, streams, name, provider = False, True, "fake", "fake"

        async def stream(self, system, messages):
            yield 'ACT {"verb": "type", "object": "the PIN", "request": "Type [left out: a PIN] into the form"}'

    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: _Acts())
    result = await layer.take("my PIN is 4471, can you put it in the form?")
    assert result.done and result.intent is None
    assert said[-1] == SECRET_NOT_USED


def test_an_early_request_never_starts_on_words_holding_a_secret():
    layer = ConversationLayer(sink=_silent, model=lambda: _http_model("claude", []), provisional_setting=lambda: "on")
    assert layer.begin_provisional("t1", "oh and my bank pin is 4471", "wake") is False


def test_build_messages_takes_it_out_whoever_calls_it():
    memory = ConversationMemory()
    from pilot.conversation.context import ContextSources

    messages = build_messages("my wifi password is swordfish, why won't it connect?", ContextSources(), memory)
    assert "swordfish" not in json.dumps(messages)


# ── the voice path in the server: no handler that keeps things, no planner, no log ──


async def test_the_spoken_turn_reaches_no_store_no_planner_and_no_log(caplog):
    server = PilotServer(PilotConfig())
    sent: list[str] = []
    heard: list[str] = []
    asked_handlers: list[str] = []
    planned: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def control_reply(text):
        asked_handlers.append(text)
        return None

    async def execute(params, ws):
        planned.append(params.get("input", ""))
        return {"status": "success", "message": "done"}

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._model_router = object()
    server._control_reply = control_reply
    server._handle_execute = execute
    layer_for(server)._model = lambda: _http_model("claude", sent)
    voice_turn_addressed.set("wake")
    with caplog.at_level(logging.DEBUG):
        await server._voice_command_dispatch("Oh, and my bank PIN is 4471.")
        await server._voice_command_dispatch("take a note that my door code is 2580")
        await server._voice_command_dispatch("What's the capital of Australia?")
    assert asked_handlers == ["What's the capital of Australia?"]  # notes, to-dos and the profile never saw them
    assert planned == []
    assert not any(PIN in body or "2580" in body for body in sent)
    assert PIN not in caplog.text and "2580" not in caplog.text
    turns = [turn.text for turn in layer_for(server).memory.recent()]
    assert not any(PIN in text or "2580" in text for text in turns)
