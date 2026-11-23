"""The listener's early request (pilot.system.voice): the talk model is asked on the words of a speculative pass
already known to be addressed, before the turn settles, and the request is dropped when the person goes on, the words
change, nothing is dispatched or the speech was not for Heliox. Fakes only; events are counted, never timed."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.voice import REQUEST_POINT_FRAMES, ContinuousVoiceListener

WAKE = {"addressed": "wake", "said": "Hey Heliocs, what's the capital of Peru?", "wake_text": "Hey Heliox, what's the"}


class Hooks:
    def __init__(self, start: bool = True) -> None:
        self.started: list[tuple[str, str, str]] = []
        self.dropped: list[tuple[str, str]] = []
        self.start = start

    def begin(self, turn_id: str, words: str, addressed: str) -> bool:
        self.started.append((turn_id, words, addressed))
        return self.start

    def cancel(self, turn_id: str, reason: str) -> None:
        self.dropped.append((turn_id, reason))


def listener(hooks: Hooks | None = None) -> ContinuousVoiceListener:
    item = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    hooks = hooks or Hooks()
    item.early_request = hooks.begin
    item.cancel_early_request = hooks.cancel
    return item


@pytest.mark.parametrize(
    ("text", "proof", "expected"),
    [
        ("Hey Heliox, what's the capital of Peru?", WAKE, ("what's the capital of Peru", "wake")),
        ("Hey Heliox, the time?", WAKE, None),  # fewer than three words after the phrase
        ("Hey Heliox, remind me to", {**WAKE, "said": "Hey Heliocs, remind me to"}, None),  # stops mid-sentence
        ("Hey Heliox, what's the capital of Peru?", {}, None),  # not known to be addressed
        ("Hey Heliox, what's the capital of Peru?", {**WAKE, "addressed": "follow_up"}, None),  # judged on final words
        # A spelling of the name needs the second recogniser to agree on the first word after it.
        ("Hey Heliocs, what's the capital of Peru?", {**WAKE, "wake_text": "Hey Heliocs, open"}, None),
    ],
)
def test_only_addressed_finished_words_of_three_or_more_start_one(text, proof, expected):
    assert listener().early_request_words(text, proof) == expected


def test_never_in_a_call_while_heliox_talks_or_over_heliox():
    item = listener()
    item.in_call = "Zoom"
    assert item.early_request_words("Hey Heliox, what's the capital of Peru?", WAKE) is None
    item.in_call = ""
    item._wake_free_suppression_depth = 1  # Heliox is talking: possibly its own voice
    assert item.early_request_words("Hey Heliox, what's the capital of Peru?", WAKE) is None
    item._wake_free_suppression_depth = 0
    item.note_interruption()  # said over Heliox: the cut-off rule decides on the final words
    assert item.early_request_words("Hey Heliox, what's the capital of Peru?", WAKE) is None


def test_the_microphone_button_starts_one_on_its_words():
    item = listener()
    item.note_interruption(by_button=True)
    found = item.early_request_words("What's the capital of Peru?", {"addressed": "button", "said": "x"})
    assert found == ("What's the capital of Peru", "button")


def test_the_same_words_keep_the_running_request_and_new_words_ask_again_up_to_the_bound():
    hooks = Hooks()
    item = listener(hooks)
    item._begin_early_request("Hey Heliox, what's the capital of Peru?", WAKE)
    item._begin_early_request("Hey Heliox, what's the capital of Peru.", WAKE)  # same words after normalising
    assert hooks.started == [("v1", "what's the capital of Peru", "wake")] and hooks.dropped == []
    for n, place in enumerate(("Chile", "Bolivia", "Ecuador")):
        item._begin_early_request(f"Hey Heliox, what's the capital of {place}?", WAKE)
        assert hooks.dropped[-1] == ("v1", "words_changed") and len(hooks.dropped) == n + 1
    assert len(hooks.started) == 3  # the bound: three a turn; the fourth words drop the third and start nothing
    assert item._early_turn == ""


def test_the_bound_is_a_setting_and_zero_turns_it_off():
    hooks = Hooks()
    item = listener(hooks)
    item.config.conversation.early_requests_per_turn = 0
    item._begin_early_request("Hey Heliox, what's the capital of Peru?", WAKE)
    assert hooks.started == []


def test_a_refused_start_is_not_counted():
    hooks = Hooks(start=False)
    item = listener(hooks)
    for _ in range(5):
        item._begin_early_request("Hey Heliox, what's the capital of Peru?", WAKE)
    assert len(hooks.started) == 5 and item._early_count == 0 and item._early_turn == ""


@pytest.mark.asyncio
async def test_a_pass_with_a_proof_starts_one_and_carrying_on_drops_it(monkeypatch):
    hooks = Hooks()
    item = listener(hooks)

    async def transcribe(_path, *, speculative=False):
        voice._pass_proof.get().update(WAKE)
        return "Hey Heliox, what's the capital of Peru?"

    item._transcribe_path = transcribe
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    monkeypatch.setattr(voice, "_write_wav", lambda _frames, _rate: "candidate.wav")
    item._speculate_final_pass()
    item._recorder.on_candidate_end([np.zeros(800, dtype=np.int16)] * 30)
    for _ in range(5):
        await asyncio.sleep(0)
    assert hooks.started == [("v1", "what's the capital of Peru", "wake")]
    item._recorder.on_candidate_cancel()  # the person went on
    assert hooks.dropped == [("v1", "carried_on")]


@pytest.mark.asyncio
async def test_a_pass_without_a_proof_starts_nothing(monkeypatch):
    hooks = Hooks()
    item = listener(hooks)

    async def transcribe(_path, *, speculative=False):
        return "Hey Heliox, what's the capital of Peru?"  # the whole-utterance check has not said yet

    item._transcribe_path = transcribe
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 0)
    monkeypatch.setattr(voice, "_write_wav", lambda _frames, _rate: "candidate.wav")
    item._speculate_final_pass()
    item._recorder.on_candidate_end([np.zeros(800, dtype=np.int16)] * 30)
    for _ in range(5):
        await asyncio.sleep(0)
    assert hooks.started == []


async def run_one_utterance(item: ContinuousVoiceListener, transcript: str) -> list[str]:
    """The listen loop over one utterance whose speculative pass had started an early request as v1."""
    dispatched: list[str] = []

    async def on_command(text: str) -> None:
        dispatched.append(f"{voice.voice_turn_id.get()}:{text}")

    item._on_command = on_command
    calls = {"n": 0}

    async def record(**_kwargs) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            item._early_turn, item._early_words, item._early_count = "v1", "x", 1
            return transcript
        item._running = False
        return ""

    item._record_and_transcribe = record
    item._running = True
    await item._listen_loop()
    for _ in range(5):
        await asyncio.sleep(0)
    return dispatched


@pytest.mark.asyncio
async def test_the_dispatch_with_its_turn_id_takes_the_early_request_over():
    hooks = Hooks()
    item = listener(hooks)
    dispatched = await run_one_utterance(item, "Hey Heliox, what's the capital of Peru?")
    assert dispatched == ["v1:what's the capital of Peru"]
    assert hooks.dropped == []  # the layer keeps it or drops it on the settled words


@pytest.mark.asyncio
async def test_speech_not_for_heliox_drops_the_early_request_at_once():
    hooks = Hooks()
    item = listener(hooks)
    dispatched = await run_one_utterance(item, "so I told her the meeting moved to Monday")
    assert dispatched == [] and hooks.dropped == [("v1", "not_addressed")]


def test_known_addressed_speech_moves_the_request_point_to_100_ms():
    item = listener()
    loop = asyncio.new_event_loop()
    try:
        confirmed = loop.create_future()
        confirmed.set_result("Hey Heliox, what's the")
        alone = loop.create_future()
        alone.set_result("Hey Heliox.")
    finally:
        loop.close()
    item._recorder.speculate_frames = 4
    item._request_point_if_addressed(alone)
    assert item._recorder.speculate_frames == 4  # the phrase alone confirms nothing
    item._request_point_if_addressed(confirmed)
    assert item._recorder.speculate_frames == REQUEST_POINT_FRAMES == 2
    item._recorder.speculate_frames = 4
    item.early_request = None  # no early requests: the pass keeps its own timing
    item._request_point_if_addressed(confirmed)
    assert item._recorder.speculate_frames == 4


@pytest.mark.asyncio
async def test_the_turn_end_is_reported_on_addressed_words_as_soon_as_the_kept_pass_returns():
    item = listener()
    ended: list[tuple] = []
    item.turn_ended = lambda *args: ended.append(args) or True
    item._recorder.last_turn_ended_at = 50.0
    kept = asyncio.get_running_loop().create_future()
    item._hum_at_turn_end(kept, dict(WAKE))
    assert ended == []  # the pass is still running: reported when it returns
    kept.set_result("Hey Heliox, what's the capital of Peru?")
    await asyncio.sleep(0)
    assert ended == [("v1", "what's the capital of Peru", "wake", 50.0)]


@pytest.mark.asyncio
async def test_words_not_known_to_be_addressed_report_nothing():
    item = listener()
    ended: list[tuple] = []
    item.turn_ended = lambda *args: ended.append(args) or True
    item._recorder.last_turn_ended_at = 50.0
    kept = asyncio.get_running_loop().create_future()
    kept.set_result("so I told her the meeting moved to Monday")
    item._hum_at_turn_end(kept, {})
    assert ended == []
