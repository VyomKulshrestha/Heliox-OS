"""Talking over Heliox: a stop stops, a question is answered, and speech begun between two pieces of a reply stops the
next piece and is heard.

The companion acceptance run (24 November, spoken through the real listener) found two faults. "Actually, how tall
is it?" (about 1.2 s of voice) said over Heliox was taken as a cut-off "Stop": Heliox said "Okay, stopped." and never
answered. And the same question begun in the 0.25 s gap between two pieces of a sentence was not noticed: the next
piece played 2.7 s over it, no barge-in fired, and the words were dropped. Fakes, scripted voice activity and a clock
the test controls; nothing waits on real time."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.system.addressing import Addressing, Heard
from pilot.system.voice import BargeInGate, BargeInWatcher, ContinuousVoiceListener

# ── the rule: what speech over Heliox says ──


class Clock:
    def __init__(self, now: float = 300.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def replying() -> tuple[Addressing, Clock]:
    """The person asked Heliox something and Heliox is in the middle of its reply."""
    clock = Clock()
    rule = Addressing(clock=clock)
    assert rule.decide(Heard("Hey Heliox, tell me how the Eiffel Tower was built.", clock.now, clock.now + 4, 3.6)).act
    clock.now += 7.0
    return rule, clock


def over(rule: Addressing, clock: Clock, text: str, voiced: float, **kwargs):
    return rule.decide(Heard(text, clock.now, clock.now + voiced, voiced, heliox_was_speaking=True, **kwargs))


@pytest.mark.parametrize(
    ("text", "voiced"),
    [
        ("Actually, how tall is it?", 1.2),  # the acceptance run's line
        ("How tall is it?", 0.9),
        ("Wait, how tall is it?", 1.3),
        ("Open my mail.", 0.9),
        ("Actually, make it Friday.", 1.1),
        ("How tall is it, roughly, in metres and in feet?", 2.4),
        ("And how long did it take to build?", 2.1),
    ],
)
def test_a_question_or_request_over_heliox_is_an_interruption_with_words(text, voiced):
    rule, clock = replying()
    decision = over(rule, clock, text, voiced)
    assert decision.act and decision.why == "interruption" and decision.command == text.strip(" .?!")


@pytest.mark.parametrize(
    "text",
    [
        "Wait, stop.",
        "Stop.",
        "That's enough.",
        "Okay, that's enough",
        "Hold on.",
        "No.",
        "Never mind.",
        "(cut off)",
        "",
    ],
)
def test_a_stop_phrase_over_heliox_is_a_stop(text):
    rule, clock = replying()
    decision = over(rule, clock, text, 0.8)
    assert decision.act and decision.command == "Stop" and decision.why == "cut-off"


# "Wait, stop." said over Heliox, about a second of voice, as the recognisers wrote it in four rehearsals.
CUT_OFF_READINGS = ["Brutes gold", "Bruins go home", "Very discovered.", "Good score.", "Visual Studio Code.",
                    "Where we store home", "PowerShell", "Thanks for watching."]  # fmt: skip


@pytest.mark.parametrize("text", CUT_OFF_READINGS)
def test_a_short_reading_that_is_not_clearly_words_is_still_a_stop(text):
    rule, clock = replying()
    decision = over(rule, clock, text, 1.0)
    assert decision.act and decision.command == "Stop" and decision.why == "cut-off"


def test_over_a_remark_live_made_on_its_own_a_question_is_still_a_stop():
    rule = Addressing(clock=Clock())  # no conversation open: Live spoke on its own
    decision = over(rule, rule.clock, "Actually, how tall is it?", 1.2)
    assert decision.act and decision.command == "Stop"
    assert not over(rule, rule.clock, "How tall is it, roughly, in metres and in feet?", 2.4).act


def test_a_sentence_over_heliox_said_to_someone_else_or_talk_is_not_taken():
    rule, clock = replying()
    assert not over(rule, clock, "Priya, could you open the deck for everyone?", 2.5).act
    assert not over(rule, clock, "The meeting moved to Thursday afternoon.", 2.2).act
    assert not over(rule, clock, "How tall is it?", 1.0, in_call=True).act  # a call: nothing at all


def test_hey_heliox_over_heliox_is_a_request_as_before():
    rule, clock = replying()
    decision = over(rule, clock, "Hey Heliox, how tall is it?", 1.5)
    assert decision.act and decision.why == "wake" and decision.command == "how tall is it"


# ── the listener: the words of a cut-off are read; a stop phrase still lands at once ──


def cut_off_listener(final_words: str, streaming_words: str = "") -> ContinuousVoiceListener:
    listener = ContinuousVoiceListener(wake_words=["hey heliox"], config=PilotConfig())
    listener.config.voice.transcription_engine = "whisper"
    listener._recorder._soundcard_recorder = object()  # the stream is "open"

    class Streaming:
        async def finish(self):
            return streaming_words

    async def record(**_kwargs):
        listener._recorder.last_voiced_seconds = 1.2
        listener._recorder.stream_sink = Streaming()  # the fast recogniser heard the utterance as it was said
        return "utterance.wav"

    listener._recorder.record_utterance = record
    listener._transcribe_path = AsyncMock(return_value=final_words)
    listener.note_interruption()  # the person talked over Heliox and stopped it
    return listener


@pytest.mark.asyncio
async def test_a_short_cut_off_is_read_by_the_accurate_pass():
    listener = cut_off_listener("Actually, how tall is it?")
    assert await listener._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == (
        "Actually, how tall is it?"
    )
    listener._transcribe_path.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_stop_heard_by_the_fast_recogniser_waits_for_no_other():
    listener = cut_off_listener("Wait, stop.", streaming_words="Wait, stop")
    assert await listener._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == "Wait, stop"
    listener._transcribe_path.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cut_off_with_no_readable_words_is_still_a_cut_off():
    listener = cut_off_listener("")
    assert await listener._record_and_transcribe(duration=3, timeout=None, keep_backlog=True) == "(cut off)"


@pytest.mark.asyncio
async def test_the_interrupting_question_is_dispatched_and_a_stop_phrase_is_a_stop():
    from tests.test_voice_conversation import make_listener

    listener, run = make_listener(["Actually, how tall is it?"], suppressed=True, follow_up=30)
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 1.2
    statuses: list[dict] = []

    async def status(name, data):
        if name == "follow_up_detected":
            statuses.append(data)

    listener._on_status = status
    dispatched, overheard = await run()
    assert dispatched == ["Actually, how tall is it"] and overheard == []
    assert statuses == [{"transcript": "Actually, how tall is it?", "wake_free": True, "cut_off": False,
                         "addressed": "follow_up"}]  # fmt: skip
    stop, run_stop = make_listener(["That's enough."], suppressed=True, follow_up=30)
    stop.note_interruption()
    stop._recorder.last_voiced_seconds = 0.9
    assert await run_stop() == (["Stop"], [])


@pytest.mark.asyncio
async def test_the_server_answers_the_interrupting_question_instead_of_saying_stopped(monkeypatch):
    import time

    from pilot.server import PilotServer
    from tests.test_voice_conversation import make_listener

    asked: list[str] = []
    interrupted: list[bool] = []

    class Memory:
        def last_heliox(self):
            return "It was built for the 1889 World's Fair,"

        def snapshot(self):
            return ()

    class Layer:  # the conversation layer itself is tested elsewhere: here, what reaches it
        memory = Memory()

        def interrupt(self):
            interrupted.append(True)

        def provisional_dispatch(self, _turn):
            pass

        def provisional_done(self, _turn):
            pass

        async def take(self, text, **_kwargs):
            asked.append(text)
            return None

    layer = Layer()
    monkeypatch.setattr("pilot.conversation.layer_for", lambda _server: layer)
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_execute = AsyncMock(return_value={"status": "success", "message": ""})
    server._voice_cut_off_at = time.monotonic() - 0.5  # the barge-in has just stopped Heliox
    listener, run = make_listener(["Actually, how tall is it?"], suppressed=True, follow_up=30)
    listener._on_command = server._voice_command_dispatch
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = 1.2
    await run()
    assert asked == ["Actually, how tall is it"] and interrupted
    said = [call.args[0] for call in server._speak_voice_response.await_args_list]
    assert "Okay, stopped." not in said


# ── speech begun between two pieces of a reply ──


def frames(start: float, end: float, speaking: bool, level: float, step: float = 0.02):
    return [(start + i * step, speaking, level) for i in range(round((end - start) / step))]


def first_stop(gate: BargeInGate, samples) -> float | None:
    for when, speaking, level in samples:
        if gate.push(when, speaking, level):
            return when
    return None


def test_open_speakers_a_gate_seeded_with_the_last_piece_judges_speech_at_the_start_of_the_next():
    # Piece one: Heliox alone in the microphone at 0.02 for 2.4 s. The person starts 0.05 s into piece two.
    last = BargeInGate(echo_free=False)
    first_stop(last, frames(0.0, 2.4, False, 0.02))
    person = frames(2.65, 2.70, False, 0.02) + frames(2.70, 4.0, True, 0.09)
    unseeded = BargeInGate(echo_free=False)
    assert first_stop(unseeded, person) is None  # before: no idea how loud Heliox is, so never
    seeded = BargeInGate(echo_free=False)
    seeded.seed(last.echo_samples())
    stopped = first_stop(seeded, person)
    assert stopped is not None and stopped - 2.70 <= 0.45  # the open-speaker hold, measured from the first word
    echo = BargeInGate(echo_free=False)
    echo.seed(last.echo_samples())
    assert first_stop(echo, frames(2.65, 5.0, True, 0.025)) is None  # Heliox's own voice still does not stop it


class _Recorder:
    """Voice-activity flags as the listener keeps them; each poll of the watcher is one 20 ms step of the clock."""

    def __init__(self, clock: Clock, talk_from: float | None, level: float = 0.09) -> None:
        self.clock, self.talk_from, self.level = clock, talk_from, level
        self.is_active = True

    @property
    def capture_speech_since(self) -> float | None:
        return self.talk_from if self.talk_from is not None and self.clock.now >= self.talk_from else None

    @property
    def last_frame_rms(self) -> float:
        self.clock.now += 0.02  # read once per poll
        return self.level if self.capture_speech_since is not None else 0.02


class _Listener:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.is_running = True
        self.user_is_speaking = False
        self.barge_echo = None
        self.interruptions = 0

    def continuation_due(self, _started):
        return False

    def note_interruption(self):
        self.interruptions += 1


async def watch(listener: _Listener, clock: Clock, *, until: float) -> float | None:
    watcher = BargeInWatcher(listener, echo_free=False, clock=clock)
    watcher.POLL_SECONDS = 0.0
    stopped = await watcher.wait_for_speech_start(timeout=until - clock.now)
    return watcher.detected_at if stopped else None


@pytest.mark.asyncio
async def test_speech_begun_in_the_gap_stops_the_next_piece_and_is_marked_as_an_interruption():
    clock = Clock(1000.0)
    recorder = _Recorder(clock, talk_from=None)
    listener = _Listener(recorder)
    assert await watch(listener, clock, until=1002.4) is None  # piece one plays to its end; nobody talks
    assert listener.barge_echo is not None  # what the microphone heard of Heliox, for the next piece
    clock.now += 0.12  # the gap between the pieces
    recorder.talk_from = clock.now + 0.05  # the person's first word, just before piece two
    clock.now += 0.13
    stopped = await watch(listener, clock, until=clock.now + 3.0)
    assert stopped is not None and stopped - recorder.talk_from <= 0.45
    assert listener.interruptions == 1


@pytest.mark.asyncio
async def test_the_echo_of_the_last_piece_running_on_does_not_stop_the_next():
    clock = Clock(1000.0)
    recorder = _Recorder(clock, talk_from=None, level=0.02)
    listener = _Listener(recorder)
    assert await watch(listener, clock, until=1002.4) is None
    recorder.talk_from = clock.now - 0.3  # "voice" that began during piece one: its echo, as loud as Heliox
    clock.now += 0.1
    assert await watch(listener, clock, until=clock.now + 2.0) is None
    assert listener.interruptions == 0
