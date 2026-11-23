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
    AFTER_ONE_CHANCE,
    AFTER_TWO_CHANCE,
    MAX_RUN,
    hum_allowed,
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


def layer_for(
    model,
    ear,
    *,
    ended: float | None = 100.0,
    now: list[float] | None = None,
    on: bool = True,
    draw=lambda: 0.0,  # the spacing's draw: 0.0 always passes, so a test decides each turn itself
    ready=lambda text: True,
):
    clock = now if now is not None else [100.0 + ACK_AFTER_SECONDS]  # the turn reaches the layer when it is due
    return ConversationLayer(
        sink=ear,
        model=lambda: model,
        hold_after=60.0,
        clock=lambda: clock[0],
        turn_ended=lambda: ended,
        acknowledge=lambda: on,
        ack_draw=draw,
        ready_made=ready,
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


async def test_late_turns_in_a_row_hum_and_never_the_same_sound_twice_running():
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
    assert layer_sounds == ["Mm.", "Hmm.", "Mm."]  # late every time and the draw passes: three, alternating


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
    assert said[0] in ACKNOWLEDGEMENTS and not set(HOLD_PHRASES) & set(said[1:])


@pytest.mark.parametrize(
    ("text", "kwargs"),
    [
        (QUESTION, {"typed": True}),
        (QUESTION, {"ended": None}),  # the last word's time is unknown: it cannot be timed
        (QUESTION, {"statement": True}),
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
    base = {"typed": False, "ended": 1.0, "statement": False, "enabled": True}
    assert not wants_acknowledgement(text, **{**base, **kwargs})


def test_a_question_of_a_few_words_qualifies():
    assert wants_acknowledgement(QUESTION, typed=False, ended=1.0, statement=False, enabled=True)


def test_the_words_commit_to_nothing_and_rotate():
    assent = {"okay.", "ok.", "right.", "sure.", "yes.", "got it.", "of course.", "alright."}
    assert not {sound.casefold() for sound in ACKNOWLEDGEMENTS} & assent
    assert next_acknowledgement(ACKNOWLEDGEMENTS[0]) == ACKNOWLEDGEMENTS[1]
    assert next_acknowledgement(ACKNOWLEDGEMENTS[-1]) == ACKNOWLEDGEMENTS[0]


def test_the_set_is_the_two_hums_and_never_repeats_running():
    """Heard in Heliox's voice, the hums sounded like a person thinking and the four phrases did not."""
    assert ACKNOWLEDGEMENTS == ("Mm.", "Hmm.")
    assert not {"Let me see.", "One sec.", "One moment.", "Let me check.", "Mhm."} & set(ACKNOWLEDGEMENTS)
    said, last = [], ""
    for _turn in range(3 * len(ACKNOWLEDGEMENTS)):
        last = next_acknowledgement(last)
        said.append(last)
    assert all(a != b for a, b in zip(said, said[1:], strict=False))
    assert set(said) == set(ACKNOWLEDGEMENTS)  # each is used in turn
    assert next_acknowledgement("Working on it.") == ACKNOWLEDGEMENTS[0]  # after an action's own words


async def test_the_holding_phrase_never_repeats_the_acknowledgement_s_words():
    from pilot.conversation.layer import TurnResult

    ear = Ear()
    layer = layer_for(Gate(), ear)
    layer._last_ack = "One moment."  # also the first holding phrase
    await layer._hold_phrase(TurnResult(done=False), 100.0)
    assert ear.heard == [(HOLD_PHRASES[1], "task_narration")]
    assert HOLD_PHRASES[0] == "One moment."


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
    assert config.conversation.acknowledge is True  # on by default
    config.conversation.acknowledge = False
    config.save()
    assert PilotConfig.load().conversation.acknowledge is False


def test_the_daemon_layer_reads_the_setting_on_by_default():
    from pilot.config import PilotConfig
    from pilot.conversation import layer_for
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    layer = layer_for(server)
    assert layer._acknowledge() is True
    server.config.conversation.acknowledge = False
    assert layer._acknowledge() is False


# ── not a habit: spacing and alternation, counted turn by turn ──


async def settle() -> None:
    """Let every task that is ready run (loop turns, not time): an acknowledgement due at once is said by then."""
    for _ in range(20):
        await asyncio.sleep(0)


class Turns:
    """One conversation of slow answers: each turn's answer comes only after the acknowledgement had its chance."""

    def __init__(self, draws: list[float] | None = None, ready=lambda text: True) -> None:
        self.ear = Ear()
        self.draws = list(draws or [])
        self.drawn = 0
        self.asked: list[str] = []  # every sentence ready_made was asked about, in order

        def draw() -> float:
            self.drawn += 1
            return self.draws.pop(0)

        def made(text: str) -> bool:
            self.asked.append(text)
            return ready(text)

        self.current = Gate()
        self.layer = layer_for(None, self.ear, draw=draw, ready=made)
        self.layer._model = lambda: self.current
        self.marks: list[str] = []  # per turn: the hum said, or "-"

    async def turn(self, n: int) -> None:
        self.current = Gate(f"Answer number {n} is here.")
        before = len(self.ear.heard)
        running = asyncio.ensure_future(self.layer.take(f"Tell me about thing number {n} please"))
        await settle()
        self.current.release.set()
        await running
        heard = [text for text, _ch in self.ear.heard[before:]]
        assert heard[-1] == f"Answer number {n} is here."  # the answer always comes, and last
        hums = [text for text in heard if text in ACKNOWLEDGEMENTS]
        assert len(hums) <= 1
        self.marks.append(hums[0] if hums else "-")


def test_hum_allowed_after_none_always_after_one_mostly_after_two_half_never_a_fourth():
    draws: list[float] = []

    def draw() -> float:
        draws.append(0.0)
        return 0.0

    assert hum_allowed(0, draw) and draws == []  # after a quiet turn: no draw, always
    assert not hum_allowed(MAX_RUN, draw) and not hum_allowed(MAX_RUN + 2, draw) and draws == []
    assert hum_allowed(1, lambda: AFTER_ONE_CHANCE - 0.01) and not hum_allowed(1, lambda: AFTER_ONE_CHANCE)
    assert hum_allowed(2, lambda: AFTER_TWO_CHANCE - 0.01) and not hum_allowed(2, lambda: AFTER_TWO_CHANCE)


async def test_a_run_of_slow_answers_hears_a_hum_on_most_turns_never_the_same_twice_running():
    # Twelve slow turns, scripted draws: after one hum pass, fail; after two in a row the second draw fails.
    turns = Turns(draws=[0.1, 0.6, 0.9, 0.1, 0.1, 0.9])
    for n in range(12):
        await turns.turn(n)
    # 0: none before, hum. 1: after one, 0.1 passes. 2: after two, 0.6 fails (needs < 0.5). 3: hum. 4: after one,
    # 0.9 fails. 5: hum. 6: after one, 0.1. 7: after two, 0.1. 8: never a fourth. 9: hum. 10: after one, 0.9 fails.
    # 11: hum.
    assert turns.marks == ["Mm.", "Hmm.", "-", "Mm.", "-", "Hmm.", "Mm.", "Hmm.", "-", "Mm.", "-", "Hmm."]
    assert turns.drawn == 6 and turns.draws == []
    said = [mark for mark in turns.marks if mark != "-"]
    assert all(a != b for a, b in zip(said, said[1:], strict=False))  # never the same hum twice running


def test_counted_over_three_hundred_late_turns_about_two_in_three_and_no_fixed_beat():
    """The rule alone, over a scripted run of late turns with a seeded draw: how often, how long the runs, where the
    quiet turns fall."""
    import random

    rng = random.Random(24)
    run, last, marks = 0, "", []
    for _ in range(300):
        if hum_allowed(run, rng.random):
            last = next_acknowledgement(last)
            marks.append(last)
            run += 1
        else:
            marks.append("-")
            run = 0
    hums = [m for m in marks if m != "-"]
    assert 0.64 * 300 <= len(hums) <= 0.74 * 300  # about 69 % in expectation
    assert all(a != b for a, b in zip(hums, hums[1:], strict=False))  # never the same sound twice running
    longest = max(len(part) for part in "".join("x" if m != "-" else " " for m in marks).split())
    assert longest == MAX_RUN  # runs of up to three, never four
    quiet = [i for i, m in enumerate(marks) if m == "-"]
    gaps = {b - a for a, b in zip(quiet, quiet[1:], strict=False)}
    assert len(gaps) >= 3  # quiet turns at irregular places: two, three and four turns apart, not a fixed beat
    assert "--" not in "".join("-" if m == "-" else "h" for m in marks)  # never two late turns in a row without one


async def test_an_action_s_own_words_between_two_hums_do_not_bring_the_same_hum_back():
    turns = Turns(draws=[0.9])  # the second turn's draw fails: the third comes after a quiet turn
    await turns.turn(0)
    turns.layer._last_ack = "Working on it."  # a hand-over's acknowledgement on a turn of its own
    await turns.turn(1)
    await turns.turn(2)
    assert [mark for mark in turns.marks if mark != "-"] == ["Mm.", "Hmm."]


async def test_with_no_audio_made_ahead_nothing_is_said_and_nothing_waits():
    # The system voice, or a local voice that has not made the hums yet: ready_made says no, and the turn goes on as
    # if the setting were off. The answer is the only thing heard, and no draw is spent.
    turns = Turns(draws=[], ready=lambda text: False)
    for n in range(4):
        await turns.turn(n)
    assert turns.marks == ["-"] * 4
    assert [text for text, _ch in turns.ear.heard] == [f"Answer number {n} is here." for n in range(4)]
    assert turns.asked == ["Mm."] * 4 and turns.drawn == 0


async def test_once_the_audio_is_made_the_first_hum_comes():
    made: set[str] = set()
    turns = Turns(ready=lambda text: text in made)
    await turns.turn(0)
    made.add("Mm.")
    await turns.turn(1)
    assert turns.marks == ["-", "Mm."]


def test_the_hums_are_the_first_audio_the_store_makes():
    from pilot.system.fixed_speech import gather

    assert list(gather()[:2]) == list(ACKNOWLEDGEMENTS)


def test_ready_made_reads_the_store_of_the_chosen_local_voice_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pilot.conversation.adapters import _ready_made
    from pilot.system import local_tts_worker
    from pilot.system.local_tts_worker import SpeechStore

    store = SpeechStore(tmp_path / "voice_heads")
    monkeypatch.setattr(local_tts_worker._manager, "head_cache", store)
    audio = tmp_path / "mm.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    assert store.add_fixed("Mm.", SpeechStore.signature("kokoro_tts", "af_heart"), audio)

    def server(engine: str, voice: str = "af_heart"):
        return SimpleNamespace(config=SimpleNamespace(voice=SimpleNamespace(tts_engine=engine, tts_voice=voice)))

    monkeypatch.setattr(local_tts_worker._manager, "speaks_locally_now", lambda engine: False)
    assert _ready_made(server("kokoro_tts"), "Mm.") is False  # Heliox's voice is not speaking now: not ready-made
    monkeypatch.setattr(local_tts_worker._manager, "speaks_locally_now", lambda engine: True)
    assert _ready_made(server("kokoro_tts"), "Mm.") is True
    assert _ready_made(server("kokoro_tts"), "Hmm.") is False  # not made yet
    assert _ready_made(server("kokoro_tts", "am_adam"), "Mm.") is False  # made with another voice
    assert _ready_made(server("pocket_tts"), "Mm.") is False  # another local voice
    assert _ready_made(server("os_native"), "Mm.") is False  # the system voice never says it
    monkeypatch.setattr(local_tts_worker._manager, "head_cache", None)
    assert _ready_made(server("kokoro_tts"), "Mm.") is False  # no local voice set up yet


# ── decided when the listener's turn ends, before the turn reaches the layer ──


async def test_a_hum_decided_at_the_turn_end_is_said_before_the_hand_over_and_only_once():
    from pilot.conversation import provisional as prov

    model, ear = Gate(), Ear()
    layer = layer_for(model, ear)  # the clock is at the due time: said at once
    assert layer.hum_at_turn_end("v1", QUESTION, 100.0)
    await settle()
    assert ear.heard == [("Mm.", "task_narration")]  # before take(): the dispatch has not reached the layer yet
    prov.dispatch_turn.set("v1")
    turn = asyncio.ensure_future(layer.take(QUESTION))
    await settle()
    model.release.set()
    await turn
    said = [text for text, _ch in ear.heard]
    assert said == ["Mm.", "Mostly the processor working hard."]  # one hum, then the answer, and no second hum
    assert layer._hum_run == 1


async def test_no_hum_at_the_turn_end_when_the_early_request_already_has_text():
    from pilot.conversation import provisional as prov

    layer = layer_for(Gate(), Ear())
    held = prov.Provisional(turn_id="v1", words=QUESTION, addressed="wake", started=100.0, model=None, before=[])
    held.first_text_at = 100.3
    layer.provisional.add(held)
    assert not layer.hum_at_turn_end("v1", QUESTION, 100.0)


async def test_text_arriving_before_the_hum_is_due_keeps_it_unsaid():
    from pilot.conversation import provisional as prov

    ear = Ear()
    layer = layer_for(Gate(), ear, now=[100.0])  # due 0.45 s from now
    held = prov.Provisional(turn_id="v1", words=QUESTION, addressed="wake", started=100.0, model=None, before=[])
    layer.provisional.add(held)
    assert layer.hum_at_turn_end("v1", QUESTION, 100.0)
    held.first_text_at = 100.2  # the early request's first text came first
    await layer._early_hums["v1"].task
    assert ear.heard == []


@pytest.mark.parametrize("ending", ["done", "not_addressed"])
async def test_a_turn_that_goes_nowhere_drops_a_hum_not_yet_said(ending):
    ear = Ear()
    layer = layer_for(Gate(), ear, now=[100.0])
    assert layer.hum_at_turn_end("v1", QUESTION, 100.0)
    task = layer._early_hums["v1"].task
    if ending == "done":
        layer.provisional_done("v1")  # the dispatch ended on another road
    else:
        layer.cancel_provisional("v1", "not_addressed")
    await settle()
    assert task.cancelled() and ear.heard == []


async def test_the_dispatch_of_the_same_turn_sees_its_hum_playing():
    release = asyncio.Event()
    heard: list[str] = []

    async def slow_ear(text: str, channel: str, key: str) -> str:
        heard.append(text)
        await release.wait()
        return "spoken"

    layer = layer_for(Gate(), slow_ear)
    assert layer.hum_at_turn_end("v1", QUESTION, 100.0)
    await settle()
    assert heard == ["Mm."] and layer.hum_playing("v1") and not layer.hum_playing("v2")
    release.set()
    await settle()
    assert not layer.hum_playing("v1")


@pytest.mark.parametrize("words", ["Open Notepad for me please", "Stop that right now", "Thank you so much"])
def test_no_hum_at_the_turn_end_for_actions_control_words_or_thanks(words):
    assert not layer_for(Gate(), Ear()).hum_at_turn_end("v1", words, 100.0)
