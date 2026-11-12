"""A provisional turn (conv_interfaces.md, section c): the fast model starts on the listener's words before the turn is
confirmed, and nothing from it is said, kept or acted on unless the final words are the same.

On the real server with the real streaming client over a stubbed transport, driven by a fake listener that calls the
interface as the speech agent's listener will. The request log is what reached the model."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation import provisional as prov
from pilot.conversation.fast_model import HttpFastModel
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed, voice_turn_id
from pilot.workflows.quick_answers import QuickAnswers

FIRST_WORD = 0.4  # the stub's first word, after the request arrives
OVERLAP = 0.3  # from the provisional start to the final dispatch (the endpoint and the accurate pass deciding)


def _reply(body: dict) -> str:
    asked = json.dumps(body["messages"]).casefold()
    if "peru" in asked:
        return "Lima is the capital of Peru."
    if "chile" in asked:
        return "Santiago is the capital of Chile."
    return "Okay."


class Rig:
    """The server, what reached the model, and what was said (with when)."""

    def __init__(self, setting: str = "on") -> None:
        config = PilotConfig()
        config.conversation.provisional_answers = setting
        self.server = server = PilotServer(config)
        self.requests: list[dict] = []
        self.heard: list[tuple[float, str]] = []

        async def speak(text, channel, dedupe_key="", recorder=None):
            self.heard.append((time.monotonic(), text))
            return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

        async def nowhere(*_args, **_kwargs):
            return None

        server._speech_coordinator.speak = speak
        server._broadcast_notification = nowhere
        server._model_router = object()
        server._quick_answers = QuickAnswers(now=lambda: datetime(2026, 11, 12, 9, 0))
        server._wall_clock = lambda: datetime(2026, 11, 12, 9, 0)

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            self.requests.append(body)
            await asyncio.sleep(FIRST_WORD)
            event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": _reply(body)}}
            return httpx.Response(
                200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
            )

        async def key() -> str:
            return "k"

        self.model = HttpFastModel(
            provider="claude",
            name="claude-haiku-5-5",
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            api_key=key,
        )
        self.layer = layer_for(server)
        self.layer._model = lambda: self.model

    @property
    def said(self) -> list[str]:
        return [text for _at, text in self.heard]

    def provisional(self, turn_id: str, words: str, addressed: str = "wake") -> bool:
        return self.server.provisional_voice_turn(turn_id, words, addressed)

    async def dispatch(self, turn_id: str, words: str, addressed: str = "wake") -> float:
        """The final words, dispatched as the listener does; returns when the dispatch started."""

        async def run() -> None:
            voice_turn_addressed.set(addressed)
            voice_turn_id.set(turn_id)
            await self.server._voice_turn(words)

        started = time.monotonic()
        await asyncio.create_task(run())
        return started

    def first_sound_after(self, started: float) -> float:
        return min(at for at, _text in self.heard if at >= started) - started

    async def settle(self, seconds: float = FIRST_WORD + 0.3) -> None:
        await asyncio.sleep(seconds)


async def test_adopted_the_first_sound_comes_earlier_by_the_overlap():
    plain = Rig()
    started = await plain.dispatch("v1", "What's the capital of Peru?")
    without = plain.first_sound_after(started)

    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await asyncio.sleep(OVERLAP)
    assert rig.said == []  # nothing is said before the final words
    started = await rig.dispatch("v1", "What's the capital of Peru?")
    with_provisional = rig.first_sound_after(started)

    assert rig.said == ["Lima is the capital of Peru."] == plain.said
    assert len(rig.requests) == 1  # the running answer was taken over, not asked again
    assert rig.layer.provisional.adopted == 1 and rig.layer.provisional.dropped == 0
    assert without >= FIRST_WORD - 0.02
    assert with_provisional <= without - OVERLAP + 0.15, (without, with_provisional)
    # Kept as the turn's own: the question and the answer, once each.
    kept = [(turn.role, turn.text) for turn in rig.layer.memory.recent()]
    assert kept == [("user", "What's the capital of Peru?"), ("assistant", "Lima is the capital of Peru.")]


async def test_numbers_said_as_words_or_digits_are_the_same_words():
    rig = Rig()
    assert rig.provisional("v1", "um what is the capital of peru in twenty twenty six")
    await asyncio.sleep(OVERLAP)
    await rig.dispatch("v1", "What is the capital of Peru in 2026?")
    assert len(rig.requests) == 1 and rig.layer.provisional.adopted == 1


async def test_words_changed_the_answer_starts_again_and_the_provisional_one_is_never_said():
    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await asyncio.sleep(OVERLAP)
    await rig.dispatch("v1", "What's the capital of Chile?")
    await rig.settle()
    assert rig.said == ["Santiago is the capital of Chile."]
    assert len(rig.requests) == 2
    assert rig.layer.provisional.dropped == 1 and rig.layer.provisional.adopted == 0
    assert not any("peru" in turn.text.casefold() for turn in rig.layer.memory.recent())


async def test_the_person_kept_talking_nothing_is_said_or_kept():
    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await asyncio.sleep(0.05)
    rig.server.cancel_provisional_voice_turn("v1", "carried_on")
    await rig.settle()
    assert rig.said == []
    assert rig.layer.memory.recent() == []
    assert rig.layer.provisional.dropped == 1


@pytest.mark.parametrize("addressed", ["follow_up", "doubtful_follow_up", "continuation", "barge_in", "live_reply", ""])
async def test_speech_not_known_to_be_addressed_never_starts_a_provisional_call(addressed):
    rig = Rig()
    assert not rig.provisional("v1", "what's the capital of peru", addressed=addressed)
    await rig.settle()
    assert rig.requests == []  # nothing reached the model
    assert rig.layer.provisional.started == 0


async def test_not_taken_by_the_addressing_rule_nothing_is_said():
    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await asyncio.sleep(0.05)
    rig.server.cancel_provisional_voice_turn("v1", "not_taken")
    await rig.settle()
    assert rig.said == [] and rig.layer.memory.recent() == []


@pytest.mark.parametrize(
    "words",
    [
        "open notepad for me",  # a request to act: the planner's road never starts early
        "what's two plus two",  # answered in code
        "stop stop stop",  # control words, however many
        "never mind that one",
        "hi",  # too short to start on
    ],
)
async def test_only_the_spoken_fast_model_road_starts_early(words):
    rig = Rig()
    assert not rig.provisional("v1", words)
    assert rig.requests == []


async def test_nothing_starts_while_a_task_runs_or_an_approval_waits():
    rig = Rig()
    rig.server._pending_confirms["plan-1"] = object()
    assert not rig.provisional("v1", "what's the capital of peru")
    assert rig.requests == []


async def test_a_stop_during_a_provisional_answer_drops_it():
    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await asyncio.sleep(0.05)
    await rig.dispatch("v2", "stop")
    await rig.settle()
    assert not any("Lima" in text for text in rig.said)
    assert rig.layer.provisional.dropped == 1
    assert not any("peru" in turn.text.casefold() for turn in rig.layer.memory.recent())


async def test_no_final_words_in_time_drops_it(monkeypatch):
    monkeypatch.setattr(prov, "FINAL_WITHIN_SECONDS", 0.1)
    rig = Rig()
    assert rig.provisional("v1", "what's the capital of peru")
    await rig.settle()
    assert rig.said == [] and rig.layer.provisional.dropped == 1


async def test_budget_near_the_monthly_cap_turns_it_off_for_the_session():
    rig = Rig()
    rig.model.budget = SimpleNamespace(_monthly_limit=10.0, _monthly_cost=9.2, check_budget=lambda provider: None)
    assert not rig.provisional("v1", "what's the capital of peru")
    rig.model.budget._monthly_cost = 1.0  # a reset later in the session does not turn it back on
    assert not rig.provisional("v2", "what's the capital of peru")
    assert rig.requests == []


async def test_at_most_six_dropped_starts_a_minute():
    rig = Rig()
    for n in range(prov.DROPS_PER_MINUTE):
        assert rig.provisional(f"v{n}", "what's the capital of peru")
        rig.server.cancel_provisional_voice_turn(f"v{n}", "carried_on")
    assert not rig.provisional("v9", "what's the capital of peru")
    await rig.settle()
    assert rig.said == []


async def test_auto_waits_for_the_measured_adoption_rate(monkeypatch):
    rig = Rig(setting="auto")
    assert not rig.provisional("v1", "what's the capital of peru")
    monkeypatch.setattr(prov, "MEASURED_ADOPTION_RATE", 0.79)
    assert not rig.provisional("v1", "what's the capital of peru")
    monkeypatch.setattr(prov, "MEASURED_ADOPTION_RATE", 0.85)
    assert rig.provisional("v1", "what's the capital of peru")
    assert not rig.layer.provisional_on(SimpleNamespace(local=True))  # a model on this computer: off under "auto"
    assert not Rig(setting="off").provisional("v1", "what's the capital of peru")


# ── the money caps, on the registry itself ──


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _dropped(answers: prov.ProvisionalAnswers, model, clock: Clock, n: int) -> None:
    for i in range(n):
        held = prov.Provisional(
            turn_id=f"t{clock.t}-{i}", words="w w w", addressed="wake", started=clock.t, model=model, before=[]
        )
        held.record = SimpleNamespace(input_tokens=1, output_tokens=1, started=clock.t)
        model.calls.append(held.record)
        answers.add(held)
        answers.drop(held.turn_id, "carried_on")
        clock.t += 61.0  # past the per-minute cap


def test_dropped_starts_may_cost_at_most_five_cents_an_hour_whatever_the_model(monkeypatch):
    monkeypatch.setattr(prov, "call_cost", lambda provider, model, record: 0.02)  # a model fifty times dearer
    clock = Clock()
    clock.t = 3600.0 * 5  # the start of an hour
    answers = prov.ProvisionalAnswers(clock=clock)
    model = SimpleNamespace(provider="claude", name="m", calls=[], budget=None)
    _dropped(answers, model, clock, 3)  # $0.06
    assert answers.refusal(model) == "money_this_hour"
    clock.t = 3600.0 * 6 + 1  # the hour turns
    assert answers.refusal(model) == ""


def test_dropped_starts_never_cost_more_than_a_tenth_of_the_rest_once_ten_calls_old(monkeypatch):
    monkeypatch.setattr(prov, "call_cost", lambda provider, model, record: 0.001)
    clock = Clock()
    answers = prov.ProvisionalAnswers(clock=clock)
    model = SimpleNamespace(provider="claude", name="m", calls=[], budget=None)
    model.calls.extend(SimpleNamespace(input_tokens=1, output_tokens=1, started=0.0) for _ in range(9))  # $0.009
    _dropped(answers, model, clock, 1)  # $0.001 dropped, ten calls in all: a tenth is $0.0009
    assert answers.refusal(model) == "money_share"
    model.calls.extend(SimpleNamespace(input_tokens=1, output_tokens=1, started=0.0) for _ in range(5))
    assert answers.refusal(model) == "money_this_hour"  # closed for the rest of this hour
    clock.t += 3600.0
    assert answers.refusal(model) == ""  # the next hour: $0.001 against a tenth of $0.014


def test_the_same_words_are_strict():
    assert prov.same_words("What's the capital of Peru?", "whats the capital of peru")
    assert prov.same_words("um, twenty two times three", "22 times 3")
    assert prov.same_words("one hundred and five", "105")
    assert not prov.same_words("what's the capital of peru", "what's the capital of peru again")
    assert not prov.same_words("open notepad", "open notepad please")
    assert not prov.same_words("what's the time in paris", "what's the time in perth")
    assert not prov.same_words("anything", "")
    assert not prov.same_words("one two", "three")  # two numbers, never their sum
    assert prov.same_words("nineteen eighty four", "1984")
