"""The speech stamps (pilot.system.speech_trace): the listener's candidate ends and final passes, the voice's pieces, the
output line's open and the first sample. Data only; stand-ins for every model and device."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.system import local_tts_worker, speech_trace, voice


@pytest.fixture
def stamps(monkeypatch):
    seen: list[tuple[str, dict]] = []
    lock = threading.Lock()

    def sink(event, data):
        with lock:
            seen.append((event, data))

    monkeypatch.setattr(speech_trace, "sink", sink)
    return seen


def names(seen):
    return [event for event, _data in seen]


def test_no_sink_is_a_no_op_and_a_failing_sink_never_reaches_the_caller(monkeypatch):
    monkeypatch.setattr(speech_trace, "sink", None)
    speech_trace.emit("candidate end", at=1.0)

    def broken(event, data):
        raise RuntimeError("bench gone")

    monkeypatch.setattr(speech_trace, "sink", broken)
    speech_trace.emit("candidate end", at=1.0)  # no exception


class Listener:
    """What _Speculation needs of the listener: a recorder's rate, a pass, and note_pass."""

    def __init__(self, release: asyncio.Event) -> None:
        self._recorder = SimpleNamespace(sample_rate=16000)
        self.release = release
        self.passes: list[float] = []

    def _transcribe_path(self, path, speculative=False):
        async def run():
            await self.release.wait()
            return "hey heliox open notepad"

        return run()

    def note_pass(self, seconds, audio_seconds):
        self.passes.append(audio_seconds)


@pytest.mark.asyncio
async def test_a_speculative_pass_is_stamped_started_done_and_kept(stamps):
    release = asyncio.Event()
    speculation = voice._Speculation(Listener(release))
    speculation.start([np.zeros(800, dtype=np.int16)] * 20)  # 1 s of audio
    await asyncio.sleep(0)
    task = speculation.take()
    release.set()
    assert await task == "hey heliox open notepad"
    assert names(stamps) == ["final pass started", "speculative pass kept", "final pass done"]
    started, kept, done = (data for _event, data in stamps)
    assert started["speculative"] and started["audio_s"] == pytest.approx(1.0)
    assert kept["id"] == started["id"] == done["id"] and not done["failed"]
    assert done["done"] >= done["started"]


@pytest.mark.asyncio
async def test_a_pass_the_person_talked_past_is_stamped_dropped_while_running(stamps):
    release = asyncio.Event()
    speculation = voice._Speculation(Listener(release))
    speculation.start([np.zeros(800, dtype=np.int16)] * 4)
    await asyncio.sleep(0)
    speculation.cancel()
    await asyncio.sleep(0)
    dropped = [data for event, data in stamps if event == "speculative pass dropped"]
    assert len(dropped) == 1 and dropped[0]["running"]


def test_an_early_pass_held_back_by_a_running_recogniser_is_stamped(stamps, monkeypatch):
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 1)
    speculation = voice._Speculation(Listener(asyncio.Event()))
    speculation.start([np.zeros(800, dtype=np.int16)])
    assert names(stamps) == ["speculation held back"] and stamps[0][1]["recognisers"] == 1
    assert speculation.task is None


@pytest.mark.asyncio
async def test_the_candidate_end_and_cancel_are_listener_events(stamps, monkeypatch):
    from pilot.config import PilotConfig

    listener = voice.ContinuousVoiceListener(config=PilotConfig())
    monkeypatch.setattr(voice, "_recognisers_running", lambda: 1)  # no pass is started here
    listener._speculate_final_pass()
    listener._recorder.on_candidate_end([np.zeros(800, dtype=np.int16)] * 6)
    listener._recorder.on_candidate_cancel()
    assert names(stamps)[0] == "candidate end" and stamps[0][1]["audio_s"] == pytest.approx(0.3)
    assert "candidate cancel" in names(stamps)


class Line:
    """A stand-in output line: attach plays the piece through its callback at once."""

    def rate_for(self, rate):
        return rate

    def channels_for(self, channels):
        return channels

    def ensure(self, rate, channels):
        return SimpleNamespace(latency=0.0)

    def attach(self, playback):
        out = np.zeros((len(playback.audio), 1), dtype=np.float32)
        playback.callback(out, len(out), SimpleNamespace(outputBufferDacTime=2.04, currentTime=2.0), None)

    def detach(self, playback):
        pass

    def device_lost(self, reason):
        pass


def test_playback_stamps_its_start_and_the_first_sample_with_the_device_delay(stamps, monkeypatch):
    monkeypatch.setattr(voice, "_output_line", Line())
    voice._Playback(np.zeros(2400, dtype=np.float32), 24000).play()
    assert names(stamps) == ["playback start", "first sample"]
    first = stamps[1][1]
    assert first["first_block_at"] >= first["handed"] and first["dac_delay_s"] == pytest.approx(0.04)


def test_opening_the_output_line_is_stamped(stamps, monkeypatch):
    class Stream:
        latency = 0.09

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Stream))
    monkeypatch.setattr("pilot.system.audio_health.default_speaker_name", lambda: "Speakers")
    line = voice._OutputLine()
    try:
        line.ensure(24000, 1)
    finally:
        line.close()
    opened = [data for event, data in stamps if event == "output line open"]
    assert len(opened) == 1 and opened[0]["latency_s"] == pytest.approx(0.09) and opened[0]["rate"] == 24000
    assert opened[0]["opened"] >= opened[0]["started"]


@pytest.mark.asyncio
async def test_each_piece_is_stamped_ready_with_its_source_then_handed_to_playback(stamps, tmp_path):
    manager = local_tts_worker.LocalTTSWorkerManager()
    manager.head_cache = None
    made: list[str] = []

    async def make(engine, text, voice_name, target, folder):
        speech_trace.emit(
            "piece synthesized", words=len(text.split()), kind="reply", requested=0, worker_started=0, done=0
        )
        target.write_bytes(b"x")
        made.append(text)
        return target

    manager._make = make
    played: list[Path] = []

    async def play(path):
        played.append(path)

    await manager._speak_sentences("kokoro_tts", ["Okay.", "It is done."], "af_heart", tmp_path, play)
    assert len(played) == 2
    sequence = [(event, data.get("index")) for event, data in stamps if event.startswith("piece")]
    assert sequence[:3] == [("piece synthesized", None), ("piece ready", 0), ("piece plays", 0)]
    ready = [data for event, data in stamps if event == "piece ready"]
    assert [r["source"] for r in ready] == ["made", "made"] and [r["words"] for r in ready] == [1, 3]
