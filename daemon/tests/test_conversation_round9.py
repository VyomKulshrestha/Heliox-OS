"""What the whole-session suite found, fixed: a command misheard right after the wake phrase, "undo" said during a
task, "again" and "shorter" about the whole of a reply, and the no-model sentence."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.conversation.layer import FALLBACK, ConversationLayer
from pilot.conversation.router import misheard_command
from pilot.server import PilotServer


@pytest.mark.parametrize(
    ("heard", "meant"),
    [
        ("opened Notepad", "Open Notepad"),
        ("Closed Chrome.", "Close Chrome."),
        ("opens Spotify", "Open Spotify"),
        ("played the next song", "Play the next song"),
        ("muted", "Mute"),
    ],
)
@pytest.mark.parametrize("addressed", ["wake", "button"])
def test_a_subjectless_past_or_s_form_right_after_the_wake_phrase_is_the_command(heard, meant, addressed):
    assert misheard_command(heard, addressed=addressed) == meant


@pytest.mark.parametrize(
    "heard", ["I opened Notepad earlier", "opened it earlier", "closed and then I left", "open Notepad", "opened"]
)
def test_a_remark_or_an_imperative_is_left_as_it_is(heard):
    assert misheard_command(heard, addressed="wake") is None


@pytest.mark.parametrize("addressed", ["follow_up", "doubtful_follow_up", ""])
def test_only_after_the_wake_phrase_or_the_button(addressed):
    assert misheard_command("opened Notepad", addressed=addressed) is None


async def test_the_layer_hands_the_misheard_command_to_the_quick_road():
    said = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: None, addressed=lambda: "wake")
    result = await layer.take("opened Notepad")
    assert not result.done and result.planner_input == "Open Notepad" and result.quick is not None
    remark = ConversationLayer(sink=sink, model=lambda: None, addressed=lambda: "follow_up")
    assert (await remark.take("opened Notepad")).done  # a follow-up keeps its words: a remark to talk about


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["Undo that.", "Undo the last two tasks."])
async def test_undo_said_during_a_task_stops_it_and_says_one_sentence(said):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock()
    server._handle_interject = AsyncMock(return_value={"status": "ok", "message": ""})
    server._interaction_runtime.transition = AsyncMock()
    server._interactive_request_active = True
    await server._voice_command_dispatch(said)
    assert server._handle_interject.await_args.args[0]["mode"] == "stop"
    spoken = [call.args[0] for call in server._speak_voice_response.await_args_list]
    assert spoken == ["There is nothing I can undo right now."]


def test_say_that_again_and_shorter_are_about_the_whole_of_a_reply_said_sentence_by_sentence():
    server = PilotServer(PilotConfig())
    import asyncio

    from pilot.system.companion_speech import SpeechChannel, SpeechOutcome

    async def speak(text, channel, dedupe_key="", recorder=None):
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    server._speech_coordinator.speak = speak

    async def say():
        await server._speak_companion_text("Probably the processor.", "final_answer", "conv:7:0")
        await server._speak_companion_text("A browser tab is the usual reason.", "final_answer", "conv:7:1")

    asyncio.run(say())
    assert server._last_spoken == "Probably the processor. A browser tab is the usual reason."
    server._last_spoken_at = time.monotonic()
    server._last_turn_kind = "answer"
    assert asyncio.run(server._say_again_request("Shorter.")) == "Probably the processor."
    asyncio.run(server._speak_companion_text("It's nine o'clock.", "final_answer", "voice-result:time"))
    assert server._last_spoken == "It's nine o'clock."  # a new reply starts afresh


def test_the_no_model_sentence_is_short_and_says_where_to_fix_it():
    assert len(FALLBACK["none"].split()) <= 25
    assert "Settings, under Model" in FALLBACK["none"]


async def test_the_no_model_sentence_waits_for_a_person_still_talking():
    said = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: None, user_speaking=lambda: True)
    result = await layer.take("what's the capital of")
    assert result.done and said == []
