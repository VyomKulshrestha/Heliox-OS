"""Duplex voice on the CPU: Smart Turn endpointing, streaming transcription and live captions.

Every model here is a stand-in, so nothing downloads in CI; the last test exercises the real ones when
HELIOX_REAL_VOICE_MODELS=1.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.system import streaming_asr
from pilot.system.streaming_asr import StreamingTranscriber, tidy
from pilot.system.turn_detector import SmartTurnDetector, window
from pilot.system.voice import ContinuousVoiceListener, _ContinuousRecorder


def loud(size: int = 800) -> np.ndarray:
    return np.full(size, 10000, dtype=np.int16)


def quiet(size: int = 800) -> np.ndarray:
    return np.zeros(size, dtype=np.int16)


async def feed(queue: asyncio.Queue, frames: list[np.ndarray]) -> None:
    for frame in frames:
        await asyncio.sleep(0.005)
        queue.put_nowait(frame)


class FakeDetector:
    def __init__(self, answers: list[float]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def probability(self, audio: np.ndarray) -> float:
        self.calls += 1
        return self.answers.pop(0) if self.answers else self.answers_last

    @property
    def answers_last(self) -> float:
        return 0.0


def recorder(**overrides) -> _ContinuousRecorder:
    rec = _ContinuousRecorder(start_frames=2, silence_frames=9, **overrides)
    rec._queue = asyncio.Queue()
    return rec


# ── Smart Turn ──


def test_window_keeps_the_last_eight_seconds_padded_at_the_front():
    short = window(np.full(16000, 16384, dtype=np.int16))
    assert short.shape == (128000,) and short[0] == 0 and short[-1] == pytest.approx(0.5)
    long = window(np.arange(16000 * 10, dtype=np.float32))
    assert long.shape == (128000,) and long[0] == 16000 * 2


def test_features_match_the_reference_shape_and_probability_needs_a_model():
    pytest.importorskip("faster_whisper")
    detector = SmartTurnDetector()
    assert detector.features(np.zeros(8000, dtype=np.int16)).shape == (1, 80, 800)
    assert detector.probability(np.zeros(8000, dtype=np.int16)) is None  # not loaded yet: plain endpointing


def test_a_failed_load_is_reported_not_raised(tmp_path):
    detector = SmartTurnDetector(model_path=str(tmp_path / "missing.onnx"))
    assert detector.load() is False and detector.error


@pytest.mark.asyncio
async def test_a_finished_sentence_ends_the_turn_after_400_ms_not_the_full_silence():
    rec = recorder()
    rec.turn_detector = FakeDetector([0.93])
    frames = [loud()] * 6 + [quiet()] * 12
    feeder = asyncio.create_task(feed(rec._queue, frames))
    path = await rec.record_utterance(timeout=3)
    feeder.cancel()
    assert path and rec.last_endpoint_reason == "smart_turn"
    # Asked after 400 ms, not 200: real pauses between words on a hands-free headset last 250 to 350 ms.
    assert rec.last_endpoint_ms == 400 and rec.turn_detector.calls == 1


@pytest.mark.asyncio
async def test_a_pause_mid_thought_holds_the_turn_and_says_so_once():
    held = []
    rec = recorder()
    rec.turn_detector = FakeDetector([0.05, 0.05, 0.05])
    rec.turn_hold_frames = 30  # 1.5 s
    rec.on_hold = lambda: held.append(True)
    # 0.75 s of silence would end a plain turn (9 frames); here the person resumes after 1 s and then finishes.
    frames = [loud()] * 6 + [quiet()] * 20 + [loud()] * 4 + [quiet()] * 35
    feeder = asyncio.create_task(feed(rec._queue, frames))
    path = await rec.record_utterance(timeout=5)
    feeder.cancel()
    assert path and rec.last_endpoint_reason == "silence" and rec.last_endpoint_ms == 1500
    assert held == [True]


@pytest.mark.asyncio
async def test_without_a_detector_the_plain_silence_endpoint_is_unchanged():
    rec = recorder()
    frames = [loud()] * 6 + [quiet()] * 12
    feeder = asyncio.create_task(feed(rec._queue, frames))
    assert await rec.record_utterance(timeout=3)
    feeder.cancel()
    assert rec.last_endpoint_reason == "silence" and rec.last_endpoint_ms == 450


# ── streaming recognition ──


class FakeStream:
    def __init__(self) -> None:
        self.samples = 0
        self.finished = False

    def accept_waveform(self, rate, samples) -> None:
        self.samples += len(samples)
        self.last_chunk = samples

    def input_finished(self) -> None:
        self.finished = True
        self.tail = self.last_chunk


class FakeRecognizer:
    """Hears one word per 0.25 s of audio."""

    words = ["OPEN", "CHROME", "AND", "I'M", "DONE"]

    def create_stream(self) -> FakeStream:
        return FakeStream()

    def is_ready(self, stream) -> bool:
        return False

    def decode_stream(self, stream) -> None:
        pass

    def get_result(self, stream) -> str:
        return " ".join(self.words[: min(len(self.words), stream.samples // 4000)])


def test_tidy_reads_like_a_sentence():
    assert tidy("OPEN CHROME AND I'M DONE") == "Open chrome and I'm done"
    assert tidy("  ") == ""


@pytest.mark.asyncio
async def test_streaming_text_is_ready_when_the_turn_ends_and_partials_flow(monkeypatch):
    monkeypatch.setattr(streaming_asr, "PARTIAL_EVERY_SECONDS", 0.0)
    transcriber = StreamingTranscriber(recognizer=FakeRecognizer())
    partials: list[str] = []
    transcriber.on_partial = partials.append
    transcriber.begin([loud()])
    for _ in range(24):
        transcriber.feed(loud())
    text = await transcriber.finish()
    assert text == "Open chrome and I'm done"
    assert partials[0] == "Open" and partials[-1] == "Open chrome and I'm done"


@pytest.mark.asyncio
async def test_an_unready_transcriber_returns_nothing_so_whisper_runs():
    transcriber = StreamingTranscriber()
    transcriber.begin([loud()])
    assert await transcriber.finish() == ""


@pytest.mark.asyncio
async def test_streamed_text_is_the_caption_and_the_final_pass_gives_the_words(monkeypatch):
    config = PilotConfig()
    config.voice.transcription_engine = "streaming"
    config.voice.turn_detection = "silence"
    captions: list[str] = []

    async def on_partial(text: str) -> None:
        captions.append(text)

    async def whisper(path: str) -> str:
        return "Open Chrome, and I'm done."

    def build():
        listener = ContinuousVoiceListener(config=config, on_partial=on_partial)
        listener._streaming = StreamingTranscriber(recognizer=FakeRecognizer())
        listener._recorder._queue = asyncio.Queue()
        listener._recorder._stream = object()  # is_active
        listener._transcribe_path = whisper
        return listener

    monkeypatch.setattr(streaming_asr, "PARTIAL_EVERY_SECONDS", 0.0)
    listener = build()
    frames = [loud()] * 20 + [quiet()] * 12
    feeder = asyncio.create_task(feed(listener._recorder._queue, frames))
    text = await listener._record_and_transcribe(timeout=3)
    feeder.cancel()
    await asyncio.sleep(0.05)
    assert text == "Open Chrome, and I'm done."  # the accurate pass's words are the ones acted on
    assert listener.last_transcription_engine == "whisper"
    assert captions[0].startswith("Open") and captions[-1] == "Open Chrome, and I'm done."  # caption corrected

    # With no local Whisper installed, the streamed words are all there is.
    monkeypatch.setattr(ContinuousVoiceListener, "_final_pass_available", lambda self: False)

    async def nothing(path: str) -> str:
        return ""

    listener = build()
    listener._transcribe_path = nothing
    feeder = asyncio.create_task(feed(listener._recorder._queue, frames))
    assert await listener._record_and_transcribe(timeout=3) == "Open chrome and I'm done"
    feeder.cancel()
    assert listener.last_transcription_engine == "streaming"


def test_settings_validate_the_new_voice_fields():
    from unittest.mock import MagicMock

    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    async def update(values):
        return await server._handle_update_config({"section": "voice", "values": values}, None)

    assert asyncio.run(update({"transcription_engine": "streaming"}))["status"] == "ok"
    assert asyncio.run(update({"transcription_engine": "magic"}))["status"] == "error"
    assert asyncio.run(update({"turn_detection": "silence", "turn_hold_ms": 3000}))["status"] == "ok"
    assert asyncio.run(update({"turn_hold_ms": 50}))["status"] == "error"


@pytest.mark.skipif(os.environ.get("HELIOX_REAL_VOICE_MODELS") != "1", reason="downloads ~80 MB of models")
def test_real_models_agree_on_a_finished_and_an_unfinished_sentence():
    import wave

    from huggingface_hub import hf_hub_download

    wav = hf_hub_download(streaming_asr.REPO, "test_wavs/0.wav")
    with wave.open(wav) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    detector = SmartTurnDetector()
    assert detector.load()
    assert detector.probability(np.concatenate([audio, np.zeros(3200, np.int16)])) > 0.5
    assert detector.probability(audio[: 4 * 16000]) < 0.5


# ── the Whisper path these build on ──


def _wav(tmp_path) -> str:
    import wave

    path = tmp_path / "turn.wav"
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.sin(np.linspace(0, 400, 16000)) * 8000).astype(np.int16).tobytes())
    return str(path)


class _Segment:
    text = " open chrome"


@pytest.mark.asyncio
async def test_faster_whisper_gets_decoded_samples_not_a_path(tmp_path, monkeypatch):
    from pilot.system import voice

    seen = []

    class Model:
        def transcribe(self, audio, **kwargs):
            seen.append(audio)
            return [_Segment()], type("Info", (), {"language": "en"})()

    monkeypatch.setattr(voice, "_get_faster_whisper_model", lambda name: Model())
    result = await voice._transcribe_faster_whisper(_wav(tmp_path), "en", "small")
    assert result["text"] == "open chrome"
    assert isinstance(seen[0], np.ndarray)  # PyAV is never asked to decode the file


@pytest.mark.asyncio
async def test_a_gpu_without_cuda_libraries_falls_back_to_the_cpu(tmp_path, monkeypatch):
    from pilot.system import voice

    monkeypatch.setattr(voice, "_faster_whisper_cuda_unusable", False)
    built = []

    class Broken:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    class Cpu:
        def transcribe(self, audio, **kwargs):
            return [_Segment()], type("Info", (), {"language": "en"})()

    def model(name):
        built.append(voice._faster_whisper_cuda_unusable)
        return Cpu() if voice._faster_whisper_cuda_unusable else Broken()

    monkeypatch.setattr(voice, "_get_faster_whisper_model", model)
    result = await voice._transcribe_faster_whisper(_wav(tmp_path), "en", "small")
    assert result["text"] == "open chrome" and built == [False, True]


@pytest.mark.asyncio
async def test_the_last_words_get_a_moment_of_silence_to_be_decoded():
    # Without it the streaming recogniser dropped "at ten" from "...the twelfth of November at ten".
    recognizer = FakeRecognizer()
    streams: list[FakeStream] = []
    create = recognizer.create_stream
    recognizer.create_stream = lambda: streams.append(create()) or streams[-1]
    transcriber = StreamingTranscriber(recognizer=recognizer)
    transcriber.begin([loud()])
    await transcriber.finish()
    tail = streams[0].tail
    assert len(tail) == int(streaming_asr.SAMPLE_RATE * streaming_asr.TAIL_SECONDS) and not tail.any()


@pytest.mark.asyncio
async def test_a_wait_that_began_before_the_room_was_learned_hears_a_quiet_voice():
    # The headset in hands-free mode: near-silence, then a voice at about 0.01, below the 0.02 starting default.
    rec = recorder(energy_threshold=0.02)
    quiet_voice = np.full(800, 330, dtype=np.int16)  # rms about 0.01
    near_silence = np.full(800, 20, dtype=np.int16)
    frames = [near_silence] * 30 + [quiet_voice] * 8 + [quiet()] * 12
    rec._loop = asyncio.get_running_loop()

    async def publish():  # through the recorder, so it learns the room as the microphone thread would
        for frame in frames:
            await asyncio.sleep(0.005)
            rec._publish_audio_block(frame)

    feeder = asyncio.create_task(publish())
    path = await rec.record_utterance(timeout=3)
    feeder.cancel()
    assert path, "the quiet voice was heard once the threshold followed the room"
