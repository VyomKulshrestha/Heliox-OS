"""A sentence the person paused in the middle of. Words that stop mid-sentence ("remind me to", "...during exams,")
are not dispatched: the listener waits for the rest and dispatches the whole sentence once. A sentence the server
joined after a split can be continued again by a third piece. "Hey Heliox" said over Heliox starts a request and is
not a stop. A request that began before the wait ran out is heard to its end. Injected clocks; no audio, no models."""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.addressing import Addressing, Heard
from pilot.system.voice import ContinuousVoiceListener, _ContinuousRecorder, sounds_unfinished


@pytest.mark.parametrize(
    "text",
    [
        "remind me to",
        "remind me too",  # how a recogniser writes "to" when the sentence stops there
        "Remind me to.",
        "open the",
        "email the invoice to",
        "The library extended its opening hours during exams,",
        "Close the Chrome window -",
        "set a timer for",
        "turn off the lights and",
        "I want to...",
    ],
)
def test_words_that_stop_mid_sentence_sound_unfinished(text):
    assert sounds_unfinished(text)


@pytest.mark.parametrize(
    "text",
    [
        "remind me to call the bank",
        "Close Chrome.",
        "me too",
        "Is it raising the price too?",
        "What are you up to?",
        "turn it on",
        "open Spotify",
        "The",
        "and",
        "",
        "Stop.",
    ],
)
def test_finished_sentences_do_not(text):
    assert not sounds_unfinished(text)


def listener() -> ContinuousVoiceListener:
    item = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    item._wake_calibrator = MagicMock()
    item._wake_calibrator.promoted.return_value = []
    item._wake_calibrator.check_near_miss.return_value = None
    return item


async def hear(
    said: list[tuple[str, float]],
    withdraw=None,
    expect: int = 1,
    *,
    task_open: bool = False,
    real_wait: bool = False,
    on_command=None,
    elapsed: list[float] | None = None,
):
    """Feed the listener utterances (text, seconds after the previous one ended) and return what it dispatched,
    the withdraw calls it made and the waits it asked the recorder for. The recorder's stamps start from the
    monotonic clock, as the listener's own do. ``real_wait``: a wait for the rest of a sentence takes its full
    time, as with a microphone that hears nothing more; ``elapsed`` gets the seconds to each dispatch."""
    item = listener()
    dispatched: list[tuple[str, str, str]] = []
    calls: list[tuple[str, str]] = []
    waits: list[float | None] = []
    done = asyncio.Event()

    began = time.monotonic()
    forward = on_command

    async def on_command(text: str) -> None:
        if elapsed is not None:
            elapsed.append(time.monotonic() - began)
        dispatched.append((text, voice.voice_turn_addressed.get(), voice.voice_turn_id.get()))
        if forward is not None:
            await forward(text)
        if len(dispatched) + len(calls) >= expect:
            done.set()

    async def withdraw_turn(turn_id: str, joined: str):
        calls.append((turn_id, joined))
        if len(dispatched) + len(calls) >= expect:
            done.set()
        return SimpleNamespace(status=withdraw)

    item._on_command = on_command
    if task_open:
        item.task_open = lambda: True  # a task runs or an approval waits: replies need no wake phrase
    if withdraw is not None:
        item.withdraw_turn = withdraw_turn
    recorder = item._recorder
    pending = iter(said)
    clock = [time.monotonic()]

    async def transcribe(**kwargs):
        waits.append(kwargs.get("timeout"))
        try:
            text, after = next(pending)
        except StopIteration:
            if kwargs.get("timeout") is not None:
                if real_wait:
                    await asyncio.sleep(kwargs["timeout"])
                return "No speech detected"  # the wait for the rest of a sentence ran out
            await asyncio.wait_for(done.wait(), 2)
            item._running = False
            return "No speech detected"
        clock[0] += after
        recorder.speech_started_at = clock[0]
        recorder.last_endpoint_at = clock[0] + 1.0
        recorder.last_turn_ended_at = clock[0] + 0.6
        clock[0] += 1.0
        return text

    item._record_and_transcribe = transcribe
    item._running = True
    await asyncio.wait_for(item._listen_loop(), timeout=8)
    await asyncio.sleep(0.05)
    return dispatched, calls, waits


@pytest.mark.asyncio
async def test_a_request_paused_after_to_is_dispatched_once_whole():
    dispatched, calls, waits = await hear(
        [("Hey Heliox, remind me too", 0.0), ("Call the bank tomorrow morning.", 1.7)], withdraw="withdrawn"
    )
    assert [d[0] for d in dispatched] == ["remind me to call the bank tomorrow morning"]  # "too" read as "to"
    assert dispatched[0][1] == "wake"
    assert calls == []  # nothing to withdraw: half a sentence was never sent
    assert voice.UNFINISHED_WAIT_SECONDS in waits


@pytest.mark.asyncio
async def test_a_sentence_cut_at_a_comma_is_dispatched_whole():
    dispatched, _calls, _waits = await hear(
        [
            ("Hey Heliox, the library extended its opening hours during exams,", 0.0),
            ("and the reading room was full until midnight.", 1.1),
        ]
    )
    assert [d[0] for d in dispatched] == [
        "the library extended its opening hours during exams and the reading room was full until midnight"
    ]


@pytest.mark.asyncio
async def test_with_nothing_more_the_words_are_dispatched_as_they_were():
    dispatched, _calls, _waits = await hear([("Hey Heliox, remind me to", 0.0)])
    assert [d[0] for d in dispatched] == ["remind me to"]


@pytest.mark.asyncio
async def test_a_stop_word_or_a_new_request_after_half_a_sentence_is_taken_alone():
    stopped, _c, _w = await hear([("Hey Heliox, open the", 0.0), ("Never mind.", 1.0)])
    assert [d[0] for d in stopped] == ["Never mind"]
    fresh, _c, _w = await hear([("Hey Heliox, open the", 0.0), ("Hey Heliox, open Spotify.", 1.0)])
    assert [d[0] for d in fresh] == ["open Spotify"]


@pytest.mark.asyncio
async def test_a_sentence_the_server_joined_is_continued_by_a_third_piece():
    """ "The recipe says to bake the bread for forty minutes | but my oven runs hot | so I take it out at 35":
    the second piece joins the first on the server, and the third joins that joined turn ("v1+"), where it was
    judged alone and overheard."""
    dispatched, calls, _waits = await hear(
        [
            ("Hey Heliox, the recipe says to bake the bread for forty minutes.", 0.0),
            ("But my oven runs hot.", 0.5),
            ("So I take it out at 35.", 0.5),
        ],
        withdraw="withdrawn",
        expect=3,
    )
    assert [d[0] for d in dispatched] == ["the recipe says to bake the bread for forty minutes"]
    assert calls[0][0] == "v1" and calls[0][1].endswith("but my oven runs hot")
    assert calls[1][0] == "v1+" and calls[1][1].endswith("but my oven runs hot so I take it out at 35")


def test_hey_heliox_over_heliox_starts_a_request_and_a_short_cut_off_still_stops_it():
    rule = Addressing(clock=lambda: 10.0)
    wake = rule.decide(Heard("Hay heliox", 10.0, 10.7, 0.65, heliox_was_speaking=True))
    assert wake.act and wake.why == "wake" and wake.command == ""
    stop = rule.decide(Heard("Wait.", 20.0, 20.5, 0.4, heliox_was_speaking=True))
    assert stop.act and stop.command == "Stop" and stop.why == "cut-off"


@pytest.mark.asyncio
async def test_a_request_that_began_before_the_wait_ran_out_is_heard_to_its_end():
    recorder = _ContinuousRecorder(start_frames=1, silence_frames=2, sample_rate=16000)
    recorder._queue = asyncio.Queue()
    loud, quiet = np.full(800, 10000, dtype=np.int16), np.zeros(800, dtype=np.int16)

    async def feed() -> None:
        # Speech begins at once and lasts about 1.5 s, past the 1.0 s wait for it to begin.
        for frame in [loud] * 30 + [quiet] * 3:
            await asyncio.sleep(0.05)
            recorder._queue.put_nowait(frame)

    feeder = asyncio.create_task(feed())
    path = await asyncio.wait_for(recorder.record_utterance(timeout=1.0), 5)
    await feeder
    assert path is not None and os.path.exists(path)
    os.remove(path)


WHOLE_ALREADY = [
    "Stop,",
    "stop -",
    "No,",
    "Yes,",
    "Cancel that,",
    "Undo,",
    "Never mind,",
    "wait,",
    "shut up,",
    "Go on,",
]


@pytest.mark.parametrize("text", [*WHOLE_ALREADY, "Stop.", "Pause.", "What is this for?"])
def test_words_already_whole_never_wait_whatever_mark_ends_them(text):
    assert not sounds_unfinished(text)


@pytest.mark.parametrize(
    "text", ["Open the,", "remind me to,", "Hey Heliox, when I was at school during exams,", "No, I want Priya to,"]
)
def test_a_mark_after_words_that_cannot_stand_alone_still_waits(text):
    assert sounds_unfinished(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", WHOLE_ALREADY)
async def test_a_stop_word_or_an_answer_with_a_trailing_mark_is_dispatched_at_once(text):
    """Said with the wake phrase, the recogniser ending it on a comma or a dash: dispatched at once, not after the
    2.5 s wait for a rest that never comes."""
    elapsed: list[float] = []
    dispatched, _calls, waits = await hear([(f"Hey Heliox, {text}", 0.0)], real_wait=True, elapsed=elapsed)
    assert len(dispatched) == 1 and elapsed[0] < 1.0
    assert voice.UNFINISHED_WAIT_SECONDS not in waits


@pytest.mark.asyncio
async def test_stop_while_a_task_runs_silently_is_dispatched_at_once():
    elapsed: list[float] = []
    dispatched, _calls, waits = await hear([("Stop,", 0.0)], task_open=True, real_wait=True, elapsed=elapsed)
    assert [d[0] for d in dispatched] == ["Stop"] and elapsed[0] < 1.0
    assert voice.UNFINISHED_WAIT_SECONDS not in waits


@pytest.mark.asyncio
async def test_no_while_an_approval_waits_reaches_the_denial_at_once():
    from unittest.mock import AsyncMock

    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._pending_confirms["plan-1"] = {"plan": None}
    server._handle_confirm = AsyncMock(return_value={"status": "ok"})
    elapsed: list[float] = []
    dispatched, _calls, waits = await hear(
        [("No,", 0.0)], task_open=True, real_wait=True, on_command=server._voice_command_dispatch, elapsed=elapsed
    )
    assert [d[0] for d in dispatched] == ["No"] and elapsed[0] < 1.0
    server._handle_confirm.assert_awaited_once()
    assert server._handle_confirm.await_args.args[0] == {
        "plan_id": "plan-1", "confirmed": False, "input_channel": "voice_deny"
    }  # fmt: skip
    assert voice.UNFINISHED_WAIT_SECONDS not in waits


@pytest.mark.asyncio
async def test_a_finished_request_ending_on_about_waits_and_then_goes_as_it_was():
    """The cost of the rule: "Tell me what this is about." waits the full UNFINISHED_WAIT_SECONDS for a rest that
    never comes, then goes unchanged."""
    elapsed: list[float] = []
    dispatched, _calls, _waits = await hear(
        [("Hey Heliox, tell me what this is about.", 0.0)], real_wait=True, elapsed=elapsed
    )
    assert [d[0] for d in dispatched] == ["tell me what this is about"]
    assert elapsed[0] >= voice.UNFINISHED_WAIT_SECONDS - 0.05


@pytest.mark.parametrize(
    "text", ["remind me the launch call is Monday the twelfth at 10.", "Set a timer for 5.", "at 10:30"]
)
def test_a_number_after_a_preposition_finishes_the_sentence(text):
    assert not sounds_unfinished(text)
