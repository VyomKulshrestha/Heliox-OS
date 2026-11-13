"""The acknowledgement (pilot.conversation.acknowledge): a short sound about half a second after the last word while
a spoken answer from the model is on its way, never when the answer is ready, and words that commit to nothing.

Timed on a clock the test holds: the person's last word is at 100.0 on the layer's clock, so "due" is arithmetic, and
the model's answer is released by the test (an event), not by a sleep."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pilot.conversation import trace
from pilot.conversation.acknowledge import (
    ACK_AFTER_SECONDS,
    ACKNOWLEDGEMENTS,
    next_acknowledgement,
    wants_acknowledgement,
)
from pilot.conversation.layer import HOLD_PHRASES, ConversationLayer

QUESTION = "Why is my laptop fan so loud?"


class Gate:
    """A model whose answer comes only when the test releases it."""

    name, provider, streams, local = "stub", "claude", True, False

    def __init__(self, text: str = "Mostly the processor working hard.") -> None:
        self.text, self.calls, self.release = text, [], asyncio.Event()

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        await self.release.wait()
        yield self.text + " "


class Ear:
    def __init__(self) -> None:
        self.heard: list[tuple[str, str]] = []

    async def __call__(self, text: str, channel: str, key: str) -> str:
        self.heard.append((text, channel))
        return "spoken"


def layer_for(model, ear, *, ended: float | None = 100.0, now: list[float] | None = None, on: bool = True):
    clock = now if now is not None else [100.0 + ACK_AFTER_SECONDS]  # the turn reaches the layer when it is due
    return ConversationLayer(
        sink=ear,
        model=lambda: model,
        hold_after=60.0,
        clock=lambda: clock[0],
        turn_ended=lambda: ended,
        acknowledge=lambda: on,
    )


async def until(condition, ceiling: float = 5.0) -> None:
    async with asyncio.timeout(ceiling):  # a hang ceiling only
        while not condition():
            await asyncio.sleep(0.005)


async def test_an_answer_on_its_way_gets_a_short_sound_first_on_the_lower_channel():
    model, ear = Gate(), Ear()
    stamps: list[tuple[str, dict]] = []
    trace.listener = lambda name, at, **data: stamps.append((name, data))
    try:
        turn = asyncio.ensure_future(layer_for(model, ear).take(QUESTION))
        await until(lambda: ear.heard)
        assert ear.heard == [(ACKNOWLEDGEMENTS[0], "task_narration")]  # below the answer's priority: it can be cut
        model.release.set()
        await turn
    finally:
        trace.listener = None
    assert ear.heard[1] == ("Mostly the processor working hard.", "final_answer")
    kinds = [data.get("kind") for name, data in stamps if name == "sentence to voice"]
    assert kinds == ["acknowledgement", "content"]


async def test_never_when_the_answer_is_ready_by_then():
    model, ear = Gate(), Ear()
    model.release.set()  # the answer is there before the acknowledgement is due
    now = [100.0]
    layer = layer_for(model, ear, now=now)
    turn = asyncio.ensure_future(layer.take(QUESTION))
    await until(lambda: ear.heard)
    now[0] = 100.0 + ACK_AFTER_SECONDS + 1.0
    await turn
    assert ear.heard == [("Mostly the processor working hard.", "final_answer")]


async def test_never_twice_in_a_row_and_never_the_same_sound_twice_running():
    ear = Ear()
    current = [Gate()]
    layer = layer_for(None, ear)
    layer._model = lambda: current[0]  # one conversation, a fresh answer each turn
    replies = ("Mostly the processor working hard.", "Warmer rooms make it spin faster.", "Charging adds heat too.")
    questions = (QUESTION, "And why does it get worse at night?", "What about when it is charging?")
    for words, reply in zip(questions, replies, strict=True):
        current[0] = Gate(reply)
        turn = asyncio.ensure_future(layer.take(words))
        await asyncio.sleep(0.05)  # the acknowledgement is due at once (the clock is past it), or not at all
        current[0].release.set()
        await turn
    layer_sounds = [text for text, _ch in ear.heard if text in ACKNOWLEDGEMENTS]
    assert len(layer_sounds) == 2  # turns one and three; never the turn after one
    assert layer_sounds[0] != layer_sounds[1]  # and never the same sound twice running


async def test_a_turn_with_an_acknowledgement_gets_no_holding_phrase_as_well():
    model, ear = Gate(), Ear()
    layer = ConversationLayer(
        sink=ear,
        model=lambda: model,
        hold_after=0.05,
        clock=lambda: 100.0 + ACK_AFTER_SECONDS,
        turn_ended=lambda: 100.0,
        acknowledge=lambda: True,
    )
    turn = asyncio.ensure_future(layer.take(QUESTION))
    await asyncio.sleep(0.2)  # well past the holding phrase's time
    model.release.set()
    await turn
    said = [text for text, _ch in ear.heard]
    assert said[0] in ACKNOWLEDGEMENTS and not set(HOLD_PHRASES) & set(said)


@pytest.mark.parametrize(
    ("text", "kwargs"),
    [
        (QUESTION, {"typed": True}),
        (QUESTION, {"ended": None}),  # the last word's time is unknown: it cannot be timed
        (QUESTION, {"statement": True}),
        (QUESTION, {"acked_last_turn": True}),
        (QUESTION, {"enabled": False}),
        ("Why?", {}),
        ("And tomorrow?", {}),
        ("Thank you so much", {}),
        ("Okay", {}),
        ("Stop", {}),
        ("Never mind", {}),
    ],
)
def test_never_on_these(text, kwargs):
    base = {"typed": False, "ended": 1.0, "statement": False, "acked_last_turn": False, "enabled": True}
    assert not wants_acknowledgement(text, **{**base, **kwargs})


def test_a_question_of_a_few_words_qualifies():
    assert wants_acknowledgement(QUESTION, typed=False, ended=1.0, statement=False, acked_last_turn=False, enabled=True)


def test_the_words_commit_to_nothing_and_rotate():
    assent = {"okay.", "ok.", "right.", "sure.", "yes.", "got it.", "of course.", "alright."}
    assert not {sound.casefold() for sound in ACKNOWLEDGEMENTS} & assent
    assert next_acknowledgement(ACKNOWLEDGEMENTS[0]) == ACKNOWLEDGEMENTS[1]
    assert next_acknowledgement(ACKNOWLEDGEMENTS[-1]) == ACKNOWLEDGEMENTS[0]


def test_the_store_makes_them_ahead():
    from pilot.system.fixed_speech import gather

    assert set(ACKNOWLEDGEMENTS) <= set(gather())


def test_the_setting_is_kept_and_read_back(tmp_path, monkeypatch):
    import pilot.config as config_module
    from pilot.config import PilotConfig

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config_module, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")
    config = PilotConfig()
    assert config.conversation.acknowledge is False  # off until the sounds have been heard in the local voice
    config.conversation.acknowledge = True
    config.save()
    assert PilotConfig.load().conversation.acknowledge is True


def test_the_daemon_layer_reads_the_setting_off_by_default():
    from pilot.config import PilotConfig
    from pilot.conversation import layer_for
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    layer = layer_for(server)
    assert layer._acknowledge() is False
    server.config.conversation.acknowledge = True
    assert layer._acknowledge() is True
