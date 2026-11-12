"""Every addressed turn goes to Heliox Live before any road takes it (conv_interfaces.md, section a): a held turn
runs nothing, its one question is said once by the conversation side, the answer comes back through the same call,
and stops, denials, doubted turns and corrections during a task never go to Live.

Live's side (pilot.personal.wiring.live_hold) is faked here with the shape it has: ``say`` and ``proceed``."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system.voice import voice_turn_addressed

QUESTION = "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?"


@dataclass(frozen=True)
class Outcome:
    say: str
    proceed: str | None


class FakeLive:
    """Holds "put the review in for Monday the nineteenth" once; "Thursday" picks the corrected request."""

    def __init__(self):
        self.seen: list[str] = []
        self.waiting = False
        self.dropped = 0

    async def hold(self, server, text):
        self.seen.append(text)
        if self.waiting:
            self.waiting = False
            if "never mind" in text.casefold() or "stop" in text.casefold():
                return Outcome("Okay, I've left it.", None)
            return Outcome("", "Open the review for Thursday the nineteenth.")
        if "monday the nineteenth" in text.casefold():
            self.waiting = True
            return Outcome(QUESTION, None)
        return None

    def drop_held(self):
        self.dropped += 1
        self.waiting = False


@pytest.fixture
def live(monkeypatch):
    fake = FakeLive()
    monkeypatch.setattr("pilot.personal.wiring.live_hold", fake.hold, raising=False)
    return fake


def voice_server(live: FakeLive) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._live = MagicMock(drop_held=live.drop_held, handle_reply=AsyncMock(return_value=None))
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()

    async def silent(text, channel="final_answer", dedupe_key="", recorder=None):
        from pilot.system.companion_speech import SpeechChannel, SpeechOutcome

        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    server._speech_coordinator.speak = silent  # the layer's own lines ("Sorry?") never reach a real voice
    server._spawn_interaction_speech = MagicMock()
    server._handle_execute = AsyncMock(return_value={"status": "success", "message": "Done.", "results": []})
    server._model_router = object()
    from pilot.conversation import layer_for

    layer_for(server)._model = lambda: None  # no model on this machine: nothing waits on one
    return server


def spoken(server) -> list[str]:
    return [call.args[0] for call in server._speak_voice_response.await_args_list]


async def say(server, words, how="wake"):
    voice_turn_addressed.set(how)
    await server._voice_command_dispatch(words)


async def test_a_held_turn_runs_nothing_and_its_question_is_said_once(live):
    server = voice_server(live)
    await say(server, "Open the review for Monday the nineteenth")
    server._handle_execute.assert_not_awaited()
    assert spoken(server) == [QUESTION]
    channel = server._speak_voice_response.await_args.kwargs["channel"]
    assert str(channel.value if hasattr(channel, "value") else channel) == "final_answer"


async def test_the_answer_comes_back_through_the_same_call_and_the_chosen_version_runs(live):
    server = voice_server(live)
    await say(server, "Open the review for Monday the nineteenth")
    await say(server, "Thursday", how="follow_up")
    server._handle_execute.assert_awaited_once()
    assert server._handle_execute.await_args.args[0]["input"] == "Open the review for Thursday the nineteenth."
    assert spoken(server).count(QUESTION) == 1


async def test_never_mind_drops_it_and_nothing_runs(live):
    server = voice_server(live)
    await say(server, "Open the review for Monday the nineteenth")
    await say(server, "Never mind that", how="follow_up")
    server._handle_execute.assert_not_awaited()
    assert spoken(server) == [QUESTION, "Okay, I've left it."]


@pytest.mark.parametrize("words", ["Stop.", "Never mind."])
async def test_a_stop_answers_a_held_question_and_with_none_held_still_stops(live, words):
    server = voice_server(live)
    await say(server, "Open the review for Monday the nineteenth")
    await say(server, words, how="follow_up")
    server._handle_execute.assert_not_awaited()
    assert spoken(server)[-1] == "Okay, I've left it."
    plain = voice_server(FakeLive())
    await say(plain, words)
    plain._handle_execute.assert_not_awaited()  # nothing held: Live lets it go, and it stops as before


async def test_a_turn_the_listener_doubted_never_goes_to_live(live):
    await say(voice_server(live), "the blue one by the thing", how="doubtful_follow_up")
    assert live.seen == []


async def test_a_correction_during_a_task_keeps_its_own_road(live):
    server = voice_server(live)
    server._handle_interject = AsyncMock(return_value={"status": "ok", "message": ""})
    server._interaction_runtime.transition = AsyncMock()
    server._interactive_request_active = True
    await say(server, "Actually make it Monday the nineteenth", how="follow_up")
    assert live.seen == []
    server._handle_interject.assert_awaited_once()


async def test_a_new_conversation_drops_what_live_was_holding(live):
    server = voice_server(live)
    server._memory = None
    await say(server, "Open the review for Monday the nineteenth")
    await server._handle_voice_new_conversation({}, None)
    assert live.dropped == 1 and live.waiting is False
