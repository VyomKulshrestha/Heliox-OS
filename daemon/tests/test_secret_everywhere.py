"""A secret is taken out where the words first exist: the recogniser's final words, the caption, a typed message and
a phone's request. Everything after that point (logs, notices, stores, Live, the planner and every model) sees only
"[left out: a password]" (pilot.conversation.withheld.left_out)."""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from types import SimpleNamespace

import httpx
import pytest

from pilot.air_handoff import AirHandoffManager, PairedDevice, _b64u
from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.fast_model import HttpFastModel
from pilot.conversation.withheld import forget_recent, left_out
from pilot.server import PilotServer
from pilot.system import voice
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import ContinuousVoiceListener, voice_turn_addressed
from pilot.workflows.notes import NoteStore
from tests.test_conversation_secrets import shows

HUNTER = "My password is hunter two, don't tell anyone."


@pytest.fixture(autouse=True)
def _fresh():
    forget_recent()
    yield
    forget_recent()


def test_the_value_said_again_on_its_own_is_taken_out_too():
    assert left_out("oh, and my bank PIN is 4471") == "oh, and my bank PIN is [left out: a PIN]"
    assert "4471" not in left_out("was it 4471?")
    forget_recent()  # a new conversation
    assert left_out("room 4471") == "room 4471"


# ── the listener: one call where the final words are handed on ──


async def test_the_listeners_final_words_never_hold_the_secret():
    async def recognised(**_kwargs):
        return HUNTER

    fake = SimpleNamespace(_record_and_transcribe=recognised)
    words = await ContinuousVoiceListener._final_words(fake, duration=3, timeout=None, keep_backlog=True)
    assert "hunter" not in words and "[left out: a password]" in words


def test_every_final_pass_goes_through_it():
    source = inspect.getsource(voice.ContinuousVoiceListener)
    calls = re.findall(r"await self\._record_and_transcribe\(", source)
    assert len(calls) == 1  # only inside _final_words itself
    assert len(re.findall(r"await self\._final_words\(", source)) == 3  # the loop, the wake-only turn, the rest


async def test_the_caption_shows_the_left_out_form():
    server = PilotServer(PilotConfig())
    sent: list[tuple[str, dict]] = []

    async def broadcast(method, params):
        sent.append((method, params))

    server._broadcast_notification = broadcast
    server._prewarm_voice = lambda **_kwargs: None
    await server._voice_partial("my PIN is 4471")
    assert sent == [("voice_partial", {"text": "my PIN is [left out: a PIN]"})]


# ── the combination with the turn-taking series: a follow-up with no wake phrase ──


def _server_with_model() -> tuple[PilotServer, list[str]]:
    server = PilotServer(PilotConfig())
    sent: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "I won't keep that."}}
        return httpx.Response(
            200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
        )

    async def key() -> str:
        return "k"

    model = HttpFastModel(
        provider="claude",
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=key,
    )
    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._model_router = object()
    layer_for(server)._model = lambda: model
    return server, sent


@pytest.mark.parametrize("taken_out_by_listener", [True, False])
async def test_a_follow_up_without_the_wake_phrase_reaches_no_model_with_the_value(taken_out_by_listener, caplog):
    server, sent = _server_with_model()
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("What's the capital of Australia?")
    voice_turn_addressed.set("follow_up")  # said in the window, no "Hey Heliox"
    words = left_out(HUNTER) if taken_out_by_listener else HUNTER
    with caplog.at_level(logging.DEBUG):
        await server._voice_command_dispatch(words)
        voice_turn_addressed.set("follow_up")
        await server._voice_command_dispatch("Why not?")
    assert len(sent) >= 2
    assert not any("hunter" in body.casefold() for body in sent)
    assert "[left out: a password]" in sent[-1]
    assert "hunter" not in caplog.text.casefold()
    assert not any("hunter" in turn.text.casefold() for turn in layer_for(server).memory.recent())


# ── a typed message ──


async def test_a_typed_secret_merely_mentioned_is_left_out_before_anything_else(tmp_path):
    server = PilotServer(PilotConfig())
    seen: list[str] = []

    async def reserved(params, ws):
        seen.append(params["input"])
        return {"status": "success"}

    server._handle_execute_reserved = reserved
    server._notes = NoteStore(tmp_path / "notes.json")
    await server._handle_execute({"input": "my wifi password is swordfish, why won't my laptop connect?"}, None)
    assert seen == ["my wifi password is [left out: a password], why won't my laptop connect?"]
    assert server._notes.list() == []


async def test_a_typed_request_to_keep_it_is_kept_on_this_computer_and_goes_nowhere_else(tmp_path):
    server = PilotServer(PilotConfig())
    seen: list[str] = []

    async def reserved(params, ws):
        seen.append(params["input"])
        return {"status": "success"}

    server._handle_execute_reserved = reserved
    server._notes = NoteStore(tmp_path / "notes.json")
    reply = await server._handle_execute({"input": "take a note: the wifi password is swordfish"}, None)
    assert seen == []  # no task, no history, no planner, no model
    assert "swordfish" in json.dumps(server._notes.list())  # kept here, as asked
    assert "swordfish" not in json.dumps(reply)  # the reply leaves it out


async def test_a_spoken_request_to_keep_it_is_not_honoured_the_same_way(tmp_path):
    """Voice goes through the listener, which takes it out first: a note said aloud keeps only the left-out form."""
    server = PilotServer(PilotConfig())
    seen: list[str] = []

    async def reserved(params, ws):
        seen.append(params["input"])
        return {"status": "success"}

    server._handle_execute_reserved = reserved
    server._notes = NoteStore(tmp_path / "notes.json")
    await server._handle_execute({"input": "take a note: the wifi password is swordfish", "source": "voice"}, None)
    assert server._notes.list() == [] and seen == ["take a note: the wifi password is [left out: a password]"]


# ── a phone's request ──


async def test_a_phone_request_is_kept_and_planned_only_in_the_left_out_form(tmp_path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    manager = AirHandoffManager(SimpleNamespace(), data_dir=tmp_path)
    secret = os.urandom(32)

    async def device_secret(_device_id):
        return secret

    manager._device_secret = device_secret
    manager._devices["d1"] = PairedDevice(device_id="d1", name="phone", created_at=0.0, last_seen_at=0.0)
    told = []

    async def on_request(request):
        told.append(request.text)

    manager.set_request_handler(accepting=lambda: True, on_request=on_request)
    nonce = os.urandom(12)
    sealed = AESGCM(secret).encrypt(
        nonce, json.dumps({"text": "my door code is 2580, open the gate"}).encode(), b"request-v1"
    )
    request = await manager.submit_request("d1", {"nonce": _b64u(nonce), "ciphertext": _b64u(sealed)})
    assert not shows("2580", request.text) and not shows("2580", json.dumps(manager.list_requests()))
    assert not shows("2580", told[0])
