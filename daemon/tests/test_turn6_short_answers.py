"""Complete short utterances do not wait.

In the acceptance run (24 November) "What am I training for?" waited 2.5 s for the rest of a sentence (the "?" was
dropped before the unfinished-sentence check, leaving "for"), and "No." during an approval waited out Smart Turn's
2 s hold. Fakes; nothing waits on real time; no audio library is imported."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pilot.system.voice import _ContinuousRecorder, sounds_unfinished
from tests.test_voice_conversation import make_listener


@pytest.mark.parametrize(
    "words",
    ["What am I training for", "Who is this from", "What are you looking at", "Which one are you talking about",
     "What is it for"],
)  # fmt: skip
def test_a_whole_question_ending_on_its_preposition_is_finished(words):
    assert not sounds_unfinished(words)


@pytest.mark.parametrize(
    "words",
    ["set a timer for", "What is the capital of", "What do you want me to", "remind me to", "send it to"],
)  # fmt: skip
def test_real_trail_offs_still_wait(words):
    assert sounds_unfinished(words)


@pytest.mark.asyncio
async def test_a_question_mark_from_the_recogniser_is_kept_for_the_check():
    listener, run = make_listener(["What am I training for?"], follow_up=30)
    waited: list[str] = []

    async def finish(first):
        waited.append(first)
        return first

    listener._finish_sentence = finish
    dispatched, _ = await run()
    assert dispatched == ["What am I training for"] and waited == []


class _Done:
    def __init__(self, text: str) -> None:
        loop = asyncio.get_running_loop()
        self.task = loop.create_future()
        self.task.set_result(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("words", "approval_waiting", "question", "ends"),
    [("No.", True, False, True), ("Yes.", True, False, True), ("Stop.", False, True, True),
     ("Cancel", True, False, True), ("No.", False, False, False), ("No problem, go ahead", True, False, False)],
)  # fmt: skip
async def test_a_short_answer_ends_the_turn_while_heliox_waits_on_a_decision(words, approval_waiting, question, ends):
    listener, _ = make_listener([], task_open=approval_waiting)
    listener._question_until = float("inf") if question else 0.0
    listener._speculation = _Done(words)
    assert listener._short_answer_heard() is ends


@pytest.mark.asyncio
async def test_the_recorder_asks_the_hint_after_the_fast_recogniser_s_window():
    # The speculative pass's words come about 0.8 s into the quiet, after the 400 to 800 ms window the fast
    # recogniser has: with the wider window the turn ends then, not after the 2.5 s hold.
    rec = _ContinuousRecorder(start_frames=2, silence_frames=50)
    rec._queue = asyncio.Queue()
    asked = [0]

    def hint() -> bool:
        asked[0] += 1
        return asked[0] >= 12  # the pass is done on the twelfth frame of quiet it is asked on

    rec.complete_hint = hint
    rec.complete_hint_frames = 50
    loud, quiet = np.full(800, 10000, dtype=np.int16), np.zeros(800, dtype=np.int16)
    for frame in [loud] * 6 + [quiet] * 55:
        rec._queue.put_nowait(frame)
    assert await rec.record_utterance(timeout=1.0, keep_backlog=True)
    assert rec.last_endpoint_reason == "short_answer" and rec.last_endpoint_ms > 800
