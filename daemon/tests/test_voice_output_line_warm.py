"""The output line is opened again as soon as speech is known to be said to Heliox, beside recognition, so a reply
after the line has closed (ten idle minutes) does not wait for the device. A fake device whose opening takes as long
as a cold one; nothing is played."""

from __future__ import annotations

import sys
import time
import types

import pytest

from pilot.config import PilotConfig
from pilot.system import voice

REAL_WARM_OUTPUT_LINE = voice.warm_output_line  # conftest stubs it for every test; these tests use the real one
OPEN_SECONDS = 0.4  # a cold open on the owner's speakers took 0.54 s to the first callback


class FakeStream:
    opened = 0

    def __init__(self, **_kwargs) -> None:
        self.active = False
        self.latency = 0.0

    def start(self) -> None:
        time.sleep(OPEN_SECONDS)
        FakeStream.opened += 1
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


@pytest.fixture
def cold_line(monkeypatch):
    """A line that closed after ten idle minutes, on a fake device."""
    FakeStream.opened = 0
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr("pilot.system.audio_health.default_speaker_name", lambda: "Speakers")
    line = voice._OutputLine()
    monkeypatch.setattr(voice, "_output_line", line)
    monkeypatch.setattr(voice, "warm_output_line", REAL_WARM_OUTPUT_LINE)
    line.ensure(voice.LOCAL_VOICE_RATE, 1)
    line.last_used = time.monotonic() - 600
    line._close_if_idle()
    assert line.stream is None
    yield line
    line.close()


def server(engine: str = "kokoro_tts"):
    from pilot.conversation import layer_for
    from pilot.server import PilotServer

    config = PilotConfig()
    config.voice.tts_engine = engine
    item = PilotServer(config)
    layer_for(item).prewarm = lambda: None  # no model connection here
    return item


def reply_waits(line: voice._OutputLine) -> float:
    """How long the reply's first piece waits for the line (what _Playback.play does first)."""
    started = time.perf_counter()
    line.ensure(voice.LOCAL_VOICE_RATE, 1)
    return time.perf_counter() - started


def wait_for_open(line: voice._OutputLine, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while line.stream is None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_speech_said_to_heliox_reopens_the_line_before_the_reply(cold_line):
    assert server().warm_for_speech("wake") is True
    wait_for_open(cold_line)  # recognition and the layer take about half a second before a reply exists
    assert reply_waits(cold_line) < 0.05
    assert FakeStream.opened == 2  # the first open, and the reopening at "wake"; none for the reply


def test_without_it_the_reply_waits_for_the_device(cold_line):
    assert reply_waits(cold_line) >= OPEN_SECONDS * 0.9


@pytest.mark.parametrize("addressed", ["", "doubtful_follow_up", "overheard"])
def test_speech_not_said_to_heliox_opens_nothing(cold_line, addressed):
    assert server().warm_for_speech(addressed) is False
    time.sleep(OPEN_SECONDS + 0.1)
    assert cold_line.stream is None and FakeStream.opened == 1


def test_the_system_voice_needs_no_line(cold_line):
    assert server("os_native").warm_for_speech("wake") is True
    time.sleep(OPEN_SECONDS + 0.1)
    assert cold_line.stream is None


def test_an_open_line_is_not_opened_again(cold_line):
    cold_line.ensure(voice.LOCAL_VOICE_RATE, 1)
    assert FakeStream.opened == 2
    server().warm_for_speech("follow_up")
    time.sleep(0.1)
    assert FakeStream.opened == 2
