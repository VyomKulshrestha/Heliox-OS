"""A turn the listener ended too early: the person goes on, Heliox yields its voice at once (not a stop), and the
continuation is joined to the turn it continues. Stop words still stop; Heliox talking for a while still means a
short cut-off is "stop". A wake-phrased turn is addressed whatever state the listener is in. Injected clocks only."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.system import voice
from pilot.system.addressing import Addressing, Heard
from pilot.system.voice import BargeInWatcher, ContinuousVoiceListener


def listener() -> ContinuousVoiceListener:
    item = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    item._wake_calibrator = MagicMock()
    item._wake_calibrator.promoted.return_value = []
    item._wake_calibrator.check_near_miss.return_value = None
    return item


def test_speech_soon_after_the_person_s_own_turn_continues_it_while_heliox_has_only_just_begun():
    item = listener()
    item._last_turn = (100.0, "remind me to", "v1")
    assert item.continuation_due(101.5)  # Heliox silent
    assert not item.continuation_due(103.5)  # too late: a new turn
    item.suppress_wake_free_commands()  # Heliox starts talking
    item._heliox_started_at = 100.8
    assert item.continuation_due(102.0)  # Heliox began 1.2 s before: it came in too early
    item._heliox_started_at = 99.0
    assert not item.continuation_due(101.0)  # Heliox was already talking before the person's turn ended
    item._heliox_started_at = 100.2
    assert not item.continuation_due(102.6)  # Heliox has been talking 2.4 s: the cut-off rule applies


@pytest.mark.asyncio
async def test_heliox_yields_at_once_to_a_continuation_without_a_stop():
    item = MagicMock()
    item.is_running = True
    item._recorder.is_active = True
    item._recorder.last_frame_rms = 0.05
    item._recorder.speech_started_at = 10.0
    item.user_is_speaking = True
    item.continuation_due.return_value = True
    now = [10.05]
    watcher = BargeInWatcher(item, echo_free=False, clock=lambda: now[0])
    assert await asyncio.wait_for(watcher.wait_for_speech_start(timeout=1.0), 1.0)
    item.note_yield.assert_called_once()
    item.note_interruption.assert_not_called()  # a yield is not a stop


@pytest.mark.asyncio
async def test_without_the_server_s_withdraw_the_second_half_is_one_new_turn_never_a_stop():
    """No withdraw_turn wired: the first half may already have run, so the second half runs alone, as a follow-up.
    One action each, and no "Stop", even when the second half began over Heliox's acknowledgement."""
    item = listener()
    seen: list[tuple[str, str, str, int, int, str]] = []
    done = asyncio.Event()

    async def on_command(text: str) -> None:
        seen.append(
            (
                text,
                voice.voice_turn_addressed.get(),
                voice.voice_turn_continues.get(),
                voice.voice_turn_id.get(),
                voice.voice_turn_continues_id.get(),
                voice.voice_turn_tail.get(),
            )
        )
        if len(seen) == 2:
            done.set()

    item._on_command = on_command
    recorder = item._recorder
    said = iter([("Hey Heliox, remind me to", 0.0), ("call the bank tomorrow morning.", 1.4)])
    # The recorder's stamps here are compared with stamps the listener takes from the real monotonic clock, so they
    # start from it: a made-up 1000.0 is in the future on a machine that booted five minutes ago (every CI runner).
    clock = [time.monotonic()]

    async def transcribe(**_kwargs):
        try:
            text, after = next(said)
        except StopIteration:
            await done.wait()
            item._running = False
            return "No speech detected"
        clock[0] += after
        recorder.speech_started_at = clock[0]
        recorder.last_endpoint_at = clock[0] + 1.0
        recorder.last_turn_ended_at = clock[0] + 0.6
        recorder.last_voiced_seconds = 1.2  # short enough to be a cut-off if it were read as one
        if text.startswith("call"):
            item.suppress_wake_free_commands()  # Heliox's acknowledgement has begun
            item._heliox_started_at = clock[0] - 0.5
            item.note_yield()  # the barge-in watcher yielded to the continuation
        clock[0] += 1.0
        return text

    item._record_and_transcribe = transcribe
    item._running = True
    await asyncio.wait_for(item._listen_loop(), timeout=5)
    assert seen[0][:2] == ("remind me to", "wake") and seen[0][3] == "v1"
    assert seen[1] == ("call the bank tomorrow morning", "follow_up", "", "v2", "", "")
    assert len(seen) == 2 and all(text != "Stop" for text, *_ in seen)


def test_a_stop_word_in_the_continuation_window_is_not_joined():
    item = listener()
    item._last_turn = (100.0, "remind me to", "v1")
    item._recorder.speech_started_at = 101.0
    assert item._continuation("Stop.") is None
    assert item._continuation("Wait, stop.") is None
    assert item._continuation("call the bank") is not None


def test_a_cut_off_is_not_taken_from_a_continuation_but_still_is_when_heliox_talked_for_a_while():
    item = listener()
    item.note_interruption()
    item._recorder.speech_started_at = 101.0
    item._recorder.last_voiced_seconds = 1.2
    item._last_turn = (100.0, "remind me to", "v1")
    item.suppress_wake_free_commands()
    item._heliox_started_at = 100.4
    assert not item._cutting_off(final=True)  # finishing the sentence: the words are needed
    item._heliox_started_at = 95.0
    assert item._cutting_off(final=True)  # Heliox was talking long before: a short cut-off is "stop"


def test_a_wake_phrased_turn_is_addressed_when_the_wake_check_misreads_what_follows_the_name():
    # The bench's 4.2: Whisper heard "Hey Heliox and do that." (talk about Heliox, by the rule); the words acted on
    # (name from Whisper, the rest from Parakeet) are "Hey Heliox, undo that."
    rule = Addressing(clock=lambda: 10.0)
    rule.task_open = True
    decision = rule.decide(
        Heard(
            "Hey Heliox, undo that.",
            10.0,
            11.0,
            1.35,
            other_text="Hey Heliocs, undo that.",
            wake_text="Hey Heliox and do that.",
        )
    )
    assert decision.act and decision.why == "wake" and decision.command == "undo that"
    # Third person stays third person in both readings.
    talk = rule.decide(
        Heard(
            "Heliox, is a nice name for a project.", 20.0, 21.0, 1.5, wake_text="Heliox is a nice name for a project."
        )
    )
    assert not talk.act


async def split_sentence(withdraw, pause: float = 1.4, withdraw_turn=None):
    """ "Hey Heliox, remind me to" | ``pause`` | "call the bank tomorrow morning." through the listener, with the
    server's withdraw call answering ``withdraw`` (or ``withdraw_turn`` itself). Returns (dispatched, withdraw calls)."""
    item = listener()
    dispatched: list[tuple[str, str]] = []
    calls: list[tuple[str, str]] = []
    done = asyncio.Event()

    async def on_command(text: str) -> None:
        dispatched.append((text, voice.voice_turn_addressed.get()))

    async def two_arguments(turn_id: str, joined: str):  # an older server, or a double without the gap
        calls.append((turn_id, joined))
        done.set()
        return withdraw

    async def counted(*args, **kwargs):
        calls.append(args + tuple(kwargs.items()))
        done.set()
        return await custom(*args, **kwargs)

    custom = withdraw_turn
    item._on_command = on_command
    item.withdraw_turn = two_arguments if custom is None else counted
    recorder = item._recorder
    said = iter([("Hey Heliox, remind me to", 0.0), ("call the bank tomorrow morning.", pause)])
    # The recorder's stamps here are compared with stamps the listener takes from the real monotonic clock, so they
    # start from it: a made-up 1000.0 is in the future on a machine that booted five minutes ago (every CI runner).
    clock = [time.monotonic()]

    async def transcribe(**_kwargs):
        try:
            text, after = next(said)
        except StopIteration:
            await done.wait()
            await asyncio.sleep(0.05)
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
    await asyncio.wait_for(item._listen_loop(), timeout=5)
    await asyncio.sleep(0.05)
    return dispatched, calls


@pytest.mark.asyncio
async def test_a_withdrawn_first_half_is_replaced_by_the_joined_sentence_on_the_server():
    from types import SimpleNamespace

    dispatched, calls = await split_sentence(SimpleNamespace(status="withdrawn"))
    assert calls == [("v1", "remind me to call the bank tomorrow morning")]
    assert dispatched == [("remind me to", "wake")]  # the server dispatched the joined sentence in its place


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["too_late", "unknown", "nonsense"])
async def test_too_late_or_unknown_runs_the_second_half_as_a_new_turn(status):
    from types import SimpleNamespace

    dispatched, _calls = await split_sentence(SimpleNamespace(status=status, said="I've already set it."))
    assert dispatched == [("remind me to", "wake"), ("call the bank tomorrow morning", "follow_up")]


@pytest.mark.asyncio
async def test_the_gap_on_the_capture_clock_reaches_the_server_so_a_slow_recogniser_does_not_split_the_request():
    """The bench's 2.0 s pause: the second half began 2.6 s after the first half's last word, but was decided 6.1 s
    after the first half's dispatch. Counted on the gap the person left, it is withdrawn and joined."""
    from types import SimpleNamespace

    seen: list[float | None] = []

    async def server_withdraw(turn_id: str, joined: str, gap: float | None = None):
        seen.append(gap)
        processing = 6.1  # what the server would count without the gap
        waited = gap if gap is not None else processing
        return SimpleNamespace(status="withdrawn" if waited <= 6.0 else "unknown")

    dispatched, _calls = await split_sentence(None, pause=2.2, withdraw_turn=server_withdraw)
    assert seen and abs(seen[0] - 2.6) < 1e-6  # last word at +0.6, second half at +3.2
    assert dispatched == [("remind me to", "wake")]  # withdrawn: the server runs the joined sentence
