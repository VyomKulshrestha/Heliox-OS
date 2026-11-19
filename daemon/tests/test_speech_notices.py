"""The window hears when Heliox speaks: one notice per piece as it goes to the speaker, one at the end of the reply,
and one as a cut-off starts its fade. Speech never waits for them, and the mouth data in them is made only while a
face is chosen.

The audio packages are optional and absent in CI: the output device and the file reader are fakes here, as in
test_voice_playback_cut. Order is checked on ordered logs, never on clock stamps."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pilot.system import kokoro_tts, local_tts_worker, speech_notices, voice


class Window:
    """What the window received, in order."""

    def __init__(self) -> None:
        self.got: list[tuple[str, dict[str, Any]]] = []
        self.gate: asyncio.Event | None = None

    async def send(self, method: str, params: dict[str, Any]) -> None:
        if self.gate is not None:
            await self.gate.wait()  # a window that is slow to read
        self.got.append((method, params))

    def methods(self) -> list[str]:
        return [method for method, _params in self.got]


class Phases:
    def __init__(self) -> None:
        self.phase, self.message, self.sequence = "completed", "Done", 4
        self.log: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"phase": self.phase, "message": self.message, "sequence": self.sequence}

    async def transition(self, phase: str, *, message: str = "") -> dict[str, Any]:
        self.phase, self.message = str(phase), message
        self.sequence += 1
        self.log.append(str(phase))
        return self.status()


async def settle(window: Window, count: int) -> None:
    for _ in range(2000):
        if len(window.got) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"the window got {window.methods()}, expected {count} notices")


@pytest.fixture
async def notices():
    window = Window()
    face = {"on": False}

    def install(phases: Any = None) -> Window:
        speech_notices.install(window.send, face_on=lambda: face["on"], phase=phases)
        return window

    yield install, face
    task = speech_notices._hub.task
    speech_notices.uninstall()
    if task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await task


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, samplerate, channels, dtype, callback, latency=None):
        self.callback = callback
        self.channels = channels
        self.latency = 0.05
        self._stop = threading.Event()
        FakeStream.instances.append(self)

    def start(self):
        def run():
            while not self._stop.is_set():
                out = np.zeros((480, self.channels), dtype=np.float32)
                self.callback(out, 480, None, None)
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()

    def abort(self):
        self._stop.set()

    def close(self):
        self._stop.set()


@pytest.fixture
def line(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr(voice, "_output_line", voice._OutputLine())
    yield
    voice._output_line.close()


def fake_audio(monkeypatch, samples: int, rate: int = 24000) -> None:
    monkeypatch.setattr(voice, "_read_audio_file", lambda _path: (np.full(samples, 0.3, dtype=np.float32), rate))


# ── Notices, their order and the phase ─────────────────────────────


@pytest.mark.asyncio
async def test_a_reply_sends_its_pieces_then_one_end_and_the_phase_reads_speaking_meanwhile(notices):
    install, _face = notices
    phases = Phases()
    window = install(phases)
    with speech_notices.reply() as reply:
        for seconds in (1.0, 2.0):
            speech_notices.handed(None, seconds)(0.04)
    await settle(window, 3)
    assert window.methods() == ["voice_piece", "voice_piece", "voice_end"]
    first, second = window.got[0][1], window.got[1][1]
    assert first["reply"] == second["reply"] == reply.number
    assert (first["index"], second["index"]) == (0, 1)
    assert first["seconds"] == 1.0 and first["lead_ms"] == 40.0 and first["mouth"] == "off"
    assert "visemes" not in first and "envelope" not in first  # no face: nothing made for one
    assert window.got[2][1]["pieces"] == 2
    for _ in range(200):
        if len(phases.log) >= 2:
            break
        await asyncio.sleep(0.005)
    assert phases.log == ["speaking", "completed"] and phases.message == "Done"  # back to what it was


@pytest.mark.asyncio
async def test_the_phase_is_left_alone_when_something_else_moved_it_meanwhile(notices):
    install, _face = notices
    phases = Phases()
    window = install(phases)
    with speech_notices.reply():
        speech_notices.handed(None, 1.0)(0.0)
        await settle(window, 1)
        for _ in range(200):
            if phases.log:
                break
            await asyncio.sleep(0.005)
        await phases.transition("acting", message="Working")
    await settle(window, 2)
    await asyncio.sleep(0.05)
    assert phases.log == ["speaking", "acting"]


@pytest.mark.asyncio
async def test_a_reply_that_never_played_sends_nothing(notices):
    install, _face = notices
    window = install()
    with speech_notices.reply():
        pass
    with pytest.raises(asyncio.CancelledError), speech_notices.reply():
        raise asyncio.CancelledError
    await asyncio.sleep(0.05)
    assert window.got == []


@pytest.mark.asyncio
async def test_a_cancelled_reply_is_cut_once_and_never_ended(notices):
    install, _face = notices
    window = install()
    with pytest.raises(asyncio.CancelledError), speech_notices.reply():
        speech_notices.handed(None, 1.0)(0.0)
        speech_notices.cut_now()
        raise asyncio.CancelledError
    await settle(window, 2)
    await asyncio.sleep(0.05)
    assert window.methods() == ["voice_piece", "voice_cut"]
    assert window.got[1][1]["index"] == 0


@pytest.mark.asyncio
async def test_a_piece_handed_over_after_the_cut_is_never_announced(notices):
    install, _face = notices
    window = install()
    with pytest.raises(asyncio.CancelledError), speech_notices.reply():
        speech_notices.handed(None, 1.0)(0.0)
        late = speech_notices.handed(None, 1.0)
        speech_notices.cut_now()
        threading.Thread(target=late, args=(0.0,)).start()  # the player's thread, after the cut
        await asyncio.sleep(0.05)
        raise asyncio.CancelledError
    await settle(window, 2)
    await asyncio.sleep(0.05)
    assert window.methods() == ["voice_piece", "voice_cut"]


@pytest.mark.asyncio
async def test_a_nested_reply_is_the_same_reply(notices):
    install, _face = notices
    window = install()
    with speech_notices.reply() as outer:
        with speech_notices.reply() as inner:
            assert inner is outer
            speech_notices.handed(None, 1.0)(0.0)
        speech_notices.handed(None, 1.0)(0.0)
    await settle(window, 3)
    assert window.methods() == ["voice_piece", "voice_piece", "voice_end"]


@pytest.mark.asyncio
async def test_speech_never_waits_for_a_slow_window(notices):
    install, _face = notices
    window = install()
    window.gate = asyncio.Event()  # the window reads nothing until opened
    with speech_notices.reply():
        for _ in range(5):
            speech_notices.handed(None, 1.0)(0.0)  # returns at once each time
    assert window.got == []
    window.gate.set()
    await settle(window, 6)
    assert window.methods() == ["voice_piece"] * 5 + ["voice_end"]


def test_without_an_installed_window_nothing_is_made_or_sent():
    speech_notices.uninstall()
    assert not speech_notices.face_on()
    with speech_notices.reply():
        assert speech_notices.handed(None, 1.0) is None
        speech_notices.unvoiced_piece()
    assert speech_notices.mouth_for(None, np.zeros(10), 24000) is None


@pytest.mark.asyncio
async def test_the_system_voice_says_it_speaks_with_no_mouth_data(notices):
    install, face = notices
    face["on"] = True
    window = install()
    with speech_notices.reply():
        speech_notices.unvoiced_piece()
    await settle(window, 2)
    piece = window.got[0][1]
    assert piece["voice"] == "system" and piece["mouth"] == "none" and piece["lead_ms"] is None


# ── Phoneme timing beside the audio ─────────────────────────────────


def write_timing(path: Path, phonemes: list[list[Any]], samples: int, words: list[list[Any]] | None = None) -> None:
    timing = {"rate": 24000, "samples": samples, "phonemes": phonemes, "words": words or []}
    speech_notices.timing_path(path).write_text(json.dumps(timing), encoding="utf-8")


def test_the_trim_moves_every_time_to_the_first_sample_kept(tmp_path):
    wav = tmp_path / "piece.wav"
    write_timing(wav, [["h", 0.30, 0.35], ["a", 0.35, 0.60], ["ɪ", 0.60, 0.80]], 24000, [["Hi", 0.30, 0.80]])
    speech_notices.shift_timing(wav, int(0.27 * 24000), int(0.70 * 24000))
    timing = json.loads(speech_notices.timing_path(wav).read_text(encoding="utf-8"))
    assert timing["samples"] == int(0.70 * 24000) - int(0.27 * 24000)
    assert timing["phonemes"] == [["h", 0.03, 0.08], ["a", 0.08, 0.33], ["ɪ", 0.33, 0.43]]
    assert timing["words"] == [["Hi", 0.03, 0.43]]
    speech_notices.shift_timing(tmp_path / "none.wav", 0, 10)  # no timing file: nothing happens


@pytest.mark.asyncio
async def test_mouth_data_is_the_timeline_when_timing_matches_and_the_envelope_otherwise(notices, tmp_path):
    install, face = notices
    install()
    audio = np.full(24000, 0.3, dtype=np.float32)
    wav = tmp_path / "piece.wav"
    write_timing(wav, [["m", 0.0, 0.2], ["a", 0.2, 0.9]], 24000)
    assert speech_notices.mouth_for(wav, audio, 24000) is None  # no face chosen: nothing made
    face["on"] = True
    mouth = speech_notices.mouth_for(wav, audio, 24000)
    assert mouth["mouth"] == "visemes"
    assert mouth["visemes"] == [["PP", 0, 200], ["aa", 200, 900], ["sil", 900, 1000]]
    # The loudness goes with the timeline, so the face can close where the sound has died inside a phoneme.
    assert mouth["envelope_rate"] == 25 and len(mouth["envelope"]) == 25 and max(mouth["envelope"]) == 15
    write_timing(wav, [["m", 0.0, 0.2]], 48000)  # made for other audio: not trusted
    assert speech_notices.mouth_for(wav, audio, 24000)["mouth"] == "envelope"
    other = speech_notices.mouth_for(tmp_path / "store.wav", audio, 24000)  # a kept piece has no timing file
    assert other["mouth"] == "envelope" and other["envelope_rate"] == 25 and len(other["envelope"]) == 25


@pytest.mark.asyncio
async def test_the_first_phoneme_is_moved_to_the_first_audible_sample(notices, tmp_path):
    install, face = notices
    install()
    face["on"] = True
    audio = np.zeros(24000, dtype=np.float32)
    audio[2400:] = 0.3  # sound from 0.1 s
    wav = tmp_path / "piece.wav"
    write_timing(wav, [["ˈ", 0.15, 0.2], ["m", 0.2, 0.3], ["a", 0.3, 0.9]], 24000, [["ma", 0.2, 0.9]])
    mouth = speech_notices.mouth_for(wav, audio, 24000)  # the voice's times run 0.1 s late: moved back
    assert mouth["onset_ms"] == 100
    assert mouth["visemes"][:3] == [["sil", 0, 50], ["PP", 50, 200], ["aa", 200, 800]]
    assert mouth["words"] == [["ma", 100, 800]]
    write_timing(wav, [["m", 0.6, 0.7], ["a", 0.7, 0.9]], 24000)  # 0.5 s apart: something else is wrong
    assert speech_notices.mouth_for(wav, audio, 24000)["onset_ms"] == 0


class FakeResult:
    """Kokoro's Result: phonemes, durations per phoneme (with one before and one after), English word times."""

    def __init__(self, phonemes: str, frames: list[int], samples: int, tokens: list[Any]):
        self.phonemes = phonemes
        self.pred_dur = frames
        self.tokens = tokens
        self._audio = np.full(samples, 0.1, dtype=np.float32)

    def __getitem__(self, index: int) -> Any:
        return [None, self.phonemes, self._audio][index]


def test_kokoro_keeps_phoneme_durations_scaled_to_each_chunk_and_offset_across_chunks(monkeypatch):
    first = FakeResult("hˈaɪ", [2, 1, 1, 2, 2, 2], 600 * 10, [SimpleNamespace(text="Hi", start_ts=0.05, end_ts=0.2)])
    second = FakeResult("ok", [1, 2, 1, 1], 600 * 5, [])

    class Pipeline:
        model = SimpleNamespace(vocab={"h": 1, "ˈ": 2, "a": 3, "ɪ": 4, "o": 5, "k": 6})

        def __call__(self, text, **kwargs):
            return [first, second]

    monkeypatch.setattr(kokoro_tts, "_get_pipeline", lambda _lang: Pipeline())
    monkeypatch.setattr(kokoro_tts, "_allow_fetching_missing_voice", lambda _voice: None)
    audio, rate, timing = kokoro_tts._generate_timed("Hi. Ok.", "af_heart")
    assert rate == 24000 and len(audio) == 600 * 15
    assert timing["samples"] == 600 * 15
    assert [p[0] for p in timing["phonemes"]] == ["h", "ˈ", "a", "ɪ", "o", "k"]
    assert timing["phonemes"][0][1:] == [0.05, 0.075]  # after the 2 frames before the first phoneme
    assert timing["phonemes"][4][1] == pytest.approx(0.25 + 0.025)  # second chunk starts at 6000 samples
    assert timing["words"] == [["Hi", 0.05, 0.2]]
    assert kokoro_tts._generate("Hi. Ok.", "af_heart")[0].shape == audio.shape  # the old call is unchanged


def test_durations_that_do_not_line_up_with_the_phonemes_give_no_timing(monkeypatch):
    bad = FakeResult("abc", [1, 1], 600, [])
    monkeypatch.setattr(kokoro_tts, "_get_pipeline", lambda _lang: lambda text, **kw: [bad])
    monkeypatch.setattr(kokoro_tts, "_allow_fetching_missing_voice", lambda _voice: None)
    _audio, _rate, timing = kokoro_tts._generate_timed("abc", "af_heart")
    assert timing is None
    plain = (None, "x", np.zeros(10, dtype=np.float32))  # an engine that gives only audio
    monkeypatch.setattr(kokoro_tts, "_get_pipeline", lambda _lang: lambda text, **kw: [plain])
    assert kokoro_tts._generate_timed("x", "af_heart")[2] is None


def test_the_worker_writes_timing_only_when_asked_and_only_for_an_engine_that_has_it(monkeypatch, tmp_path):
    made: list[tuple[str, ...]] = []
    timed = SimpleNamespace(
        synthesize_to_file_here=lambda *a: made.append(("plain", *a)),
        synthesize_with_timing_here=lambda *a: made.append(("timed", *a)),
    )
    plain = SimpleNamespace(synthesize_to_file_here=lambda *a: made.append(("plain", *a)))
    monkeypatch.setattr(local_tts_worker, "_use_voice_threads", lambda: None)
    monkeypatch.setattr(local_tts_worker, "_engine_module", lambda engine: timed if engine == "kokoro_tts" else plain)
    local_tts_worker._synthesize("kokoro_tts", "Hi.", "af_heart", "a.wav")
    local_tts_worker._synthesize("kokoro_tts", "Hi.", "af_heart", "a.wav", "a.wav.timing.json")
    local_tts_worker._synthesize("pocket_tts", "Hi.", "alba", "b.wav", "b.wav.timing.json")
    assert [entry[0] for entry in made] == ["plain", "timed", "plain"]
    assert made[1][-1] == "a.wav.timing.json"


# ── Through the player ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_played_piece_is_announced_with_its_length_and_lead_and_the_reply_ends(notices, line, monkeypatch):
    install, face = notices
    face["on"] = True
    window = install()
    fake_audio(monkeypatch, 2400)
    with speech_notices.reply():
        await voice._play_local_tts_file(Path("piece.wav"))
    await settle(window, 2)
    assert window.methods() == ["voice_piece", "voice_end"]
    piece = window.got[0][1]
    assert piece["seconds"] == 0.1 and piece["mouth"] == "envelope" and len(piece["envelope"]) == 3
    assert piece["lead_ms"] == pytest.approx(50.0)  # the line's latency once the device runs


@pytest.mark.asyncio
async def test_the_cut_notice_is_posted_before_the_fade_starts(notices, line, monkeypatch):
    install, _face = notices
    window = install()
    order: list[str] = []
    real_post = speech_notices._post
    monkeypatch.setattr(speech_notices, "_post", lambda item: (order.append(item[0]), real_post(item)))
    real_cut = voice._Playback.cut
    monkeypatch.setattr(voice._Playback, "cut", lambda self: (order.append("fade"), real_cut(self)))
    fake_audio(monkeypatch, 24000 * 60)
    task = asyncio.ensure_future(voice._play_local_tts_file(Path("piece.wav")))
    await settle(window, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await settle(window, 2)
    assert order[order.index("voice_cut") + 1 :].count("fade") == 1
    assert order.index("voice_cut") < order.index("fade")
    assert window.methods() == ["voice_piece", "voice_cut"]


@pytest.mark.asyncio
async def test_the_worker_is_asked_for_timing_only_while_a_face_is_chosen(notices, tmp_path):
    install, face = notices
    install()
    requests: list[dict[str, str]] = []

    class Manager(local_tts_worker.LocalTTSWorkerManager):
        async def _usable_process(self, engine, voice):
            return object()

        async def _request(self, process, engine, request, *, timeout):
            requests.append(dict(request))
            Path(request["output_file"]).write_bytes(b"RIFF")

    manager = Manager()
    await manager.synthesize("kokoro_tts", "Hi.", "af_heart", tmp_path / "a.wav", tmp_path)
    face["on"] = True
    await manager.synthesize("kokoro_tts", "Hi.", "af_heart", tmp_path / "b.wav", tmp_path)
    assert "timing_file" not in requests[0]
    assert requests[1]["timing_file"] == str(tmp_path / "b.wav.timing.json")
