"""The output line is opened again as soon as speech is known to be said to Heliox, beside recognition, so a reply
after the line has closed (ten idle minutes) does not wait for the device. A fake device that notes which thread
opened it; nothing is played. Whether the reply waits for the device is read from where the opening ran (the reply's
own call, or the background line-opening thread, which the tests join), not from how long a call took."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from pilot.config import PilotConfig
from pilot.system import voice

REAL_WARM_OUTPUT_LINE = voice.warm_output_line  # conftest stubs it for every test; these tests use the real one
# A cold open on one development laptop's speakers took 0.54 s to the first callback: a reply that opens the line itself waits that.


class FakeStream:
    opened = 0
    opened_on: list[str] = []  # the thread each opening ran on

    def __init__(self, **_kwargs) -> None:
        self.active = False
        self.latency = 0.0

    def start(self) -> None:
        FakeStream.opened += 1
        FakeStream.opened_on.append(threading.current_thread().name)
        self.active = True

    def abort(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


@pytest.fixture
def cold_line(monkeypatch):
    """A line that closed after ten idle minutes, on a fake device."""
    FakeStream.opened = 0
    FakeStream.opened_on = []
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


def background_opening_done() -> None:
    """Wait for the line-opening thread warm_output_line starts (if one was started) to end: the event itself."""
    for thread in threading.enumerate():
        if thread.name == "heliox-output-line":
            thread.join(30)
            assert not thread.is_alive(), "the background opening never ended"


def reply_opens_the_line(line: voice._OutputLine) -> bool:
    """What _Playback.play does first; True when the reply's own call had to open the device (and so waited for it)."""
    before = FakeStream.opened
    line.ensure(voice.LOCAL_VOICE_RATE, 1)
    return FakeStream.opened > before


def test_speech_said_to_heliox_reopens_the_line_before_the_reply(cold_line):
    assert server().warm_for_speech("wake") is True
    background_opening_done()  # recognition and the layer take about half a second before a reply exists
    assert FakeStream.opened_on == [threading.current_thread().name, "heliox-output-line"]
    assert not reply_opens_the_line(cold_line)
    assert FakeStream.opened == 2  # the first open, and the reopening at "wake"; none for the reply


def test_without_it_the_reply_waits_for_the_device(cold_line):
    assert reply_opens_the_line(cold_line)
    assert FakeStream.opened_on[-1] == threading.current_thread().name  # on the reply's own call


@pytest.mark.parametrize("addressed", ["", "doubtful_follow_up", "overheard"])
def test_speech_not_said_to_heliox_opens_nothing(cold_line, addressed):
    assert server().warm_for_speech(addressed) is False
    background_opening_done()
    assert cold_line.stream is None and FakeStream.opened == 1


def test_the_system_voice_needs_no_line(cold_line):
    assert server("os_native").warm_for_speech("wake") is True
    background_opening_done()
    assert cold_line.stream is None


def test_an_open_line_is_not_opened_again(cold_line):
    cold_line.ensure(voice.LOCAL_VOICE_RATE, 1)
    assert FakeStream.opened == 2
    server().warm_for_speech("follow_up")
    background_opening_done()
    assert FakeStream.opened == 2
