"""When the local voice worker starts, how long it stays loaded, and what happens when it fails.

A fake worker and a fake clock stand in for Kokoro, so nothing here loads a model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import kokoro_tts, local_tts_worker, voice
from pilot.system.local_tts_worker import LocalTTSUnavailable, LocalTTSWorkerManager, split_for_speech
from pilot.system.platform_detect import Platform


class FakeClock:
    """Time that only moves when the test says so; sleepers wake when it passes their deadline."""

    def __init__(self) -> None:
        self.now = 1000.0
        self._moved = asyncio.Event()

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        wake = self.now + seconds
        while self.now < wake:
            self._moved.clear()
            await self._moved.wait()

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        self._moved.set()
        for _ in range(20):
            await asyncio.sleep(0)


class FakeWorker:
    def __init__(self, *, load_reply: bytes = b'{"status":"ok","op":"load"}\n') -> None:
        self.load_reply = load_reply
        self.requests: list[dict] = []
        self.returncode: int | None = None
        self._payload = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, close=lambda: None)

    def _write(self, payload: bytes) -> None:
        self._payload = payload

    def _close(self) -> None:
        self.stdin.closed = True
        self.returncode = 0

    def _readline(self) -> bytes:
        request = json.loads(self._payload)
        self.requests.append(request)
        if request.get("op") == "load":
            return self.load_reply
        Path(request["output_file"]).write_bytes(b"RIFF")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class Spawner:
    def __init__(self, *workers: FakeWorker) -> None:
        self.workers = list(workers)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, engine: str, device: str) -> FakeWorker:
        self.calls.append((engine, device))
        return self.workers.pop(0) if len(self.workers) > 1 else self.workers[0]


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0.01)


def _manager(spawner: Spawner, clock: FakeClock, keep: float = 300.0) -> LocalTTSWorkerManager:
    return LocalTTSWorkerManager(spawn=spawner, clock=clock, sleep=clock.sleep, keep_loaded_seconds=keep)


@pytest.mark.asyncio
async def test_the_voice_stays_loaded_through_a_conversation_and_is_released_after_real_inactivity() -> None:
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock)
    play = AsyncMock()

    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, play)
    await clock.advance(240)  # a pause in the conversation
    assert manager.running
    manager.note_activity()  # the person speaks again
    await clock.advance(240)
    await manager.speak("kokoro_tts", "Done.", "af_heart", None, play)
    await clock.advance(299)
    assert manager.running and worker.stdin.closed is False

    await clock.advance(2)  # five quiet minutes after Heliox last spoke
    await _settle()
    assert not manager.running and worker.stdin.closed is True
    assert [r.get("op") for r in worker.requests].count("load") == 1  # loaded once for the whole conversation


@pytest.mark.asyncio
async def test_a_long_reply_is_not_released_while_it_is_still_playing() -> None:
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock, keep=10)
    gate = asyncio.Event()

    async def play(path: Path) -> None:
        await gate.wait()

    speech = asyncio.ensure_future(manager.speak("kokoro_tts", "A long story.", "af_heart", None, play))
    await _settle()
    await clock.advance(60)
    assert manager.running
    gate.set()
    await speech
    await clock.advance(11)
    await _settle()
    assert not manager.running
    await manager.shutdown()


@pytest.mark.asyncio
async def test_hearing_the_person_loads_the_voice_before_there_is_anything_to_say() -> None:
    clock, worker = FakeClock(), FakeWorker()
    spawner = Spawner(worker)
    manager = _manager(spawner, clock)
    statuses: list[str] = []

    async def on_status(status: str, data: dict) -> None:
        statuses.append(status)

    manager.on_status = on_status
    manager.prewarm("kokoro_tts", "af_heart")
    manager.prewarm("kokoro_tts", "af_heart")  # every partial caption calls it; one worker starts
    await _settle()
    assert manager.loaded and worker.requests == [{"op": "load", "voice": "af_heart"}]

    await manager.speak("kokoro_tts", "Sure.", "af_heart", None, AsyncMock())
    assert len(spawner.calls) == 1
    assert statuses == []  # already loaded: no "getting ready" sign, it just speaks
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_cold_reply_says_the_voice_is_getting_ready_and_then_speaks() -> None:
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock)
    statuses: list[tuple[str, dict]] = []

    async def on_status(status: str, data: dict) -> None:
        statuses.append((status, data))

    manager.on_status = on_status
    play = AsyncMock()
    await manager.speak("kokoro_tts", "Okay.", "af_heart", None, play)

    assert statuses == [("voice_loading", {"message": "Getting my voice ready…"}), ("voice_ready", {})]
    play.assert_awaited_once()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_prewarm_ignores_the_system_voice() -> None:
    clock = FakeClock()
    spawner = Spawner(FakeWorker())
    manager = _manager(spawner, clock)
    manager.prewarm("os_native", "")
    await _settle()
    assert spawner.calls == []


@pytest.mark.asyncio
async def test_a_worker_that_fails_to_load_is_reported_once_and_retried_later(caplog) -> None:
    clock = FakeClock()
    broken = FakeWorker(load_reply=b'{"status":"error","error":"RuntimeError: CUDA driver fault"}\n')
    healthy = FakeWorker()
    spawner = Spawner(broken, healthy)
    manager = _manager(spawner, clock)
    statuses: list[str] = []

    async def on_status(status: str, data: dict) -> None:
        statuses.append(status)

    manager.on_status = on_status
    play = AsyncMock()
    caplog.set_level(logging.INFO, logger="pilot.system.local_tts_worker")

    for _ in range(3):  # three sentences in a row, each falls back at once
        with pytest.raises(LocalTTSUnavailable) as raised:
            await manager.speak("kokoro_tts", "Hello.", "af_heart", None, play)
        assert raised.value.already_logged is True
    assert len(spawner.calls) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "CUDA driver fault" in warnings[0].getMessage()
    assert statuses == ["voice_loading", "voice_unavailable"]

    await clock.advance(121)
    with pytest.raises(LocalTTSUnavailable):  # still the system voice for this one; a retry starts behind it
        await manager.speak("kokoro_tts", "Hello.", "af_heart", None, play)
    await _settle()
    assert len(spawner.calls) == 2 and manager.loaded

    await manager.speak("kokoro_tts", "Back.", "af_heart", None, play)
    play.assert_awaited_once()
    assert any("recovered" in r.getMessage() for r in caplog.records)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_worker_that_dies_mid_conversation_falls_back_without_waiting() -> None:
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock)
    await manager.speak("kokoro_tts", "One.", "af_heart", None, AsyncMock())
    worker.stdout.readline = lambda: b""  # crashed
    with pytest.raises(LocalTTSUnavailable):
        await manager.speak("kokoro_tts", "Two.", "af_heart", None, AsyncMock())
    assert not manager.running and manager.failing
    await manager.shutdown()


@pytest.mark.asyncio
async def test_speak_impl_uses_the_system_voice_without_logging_the_failure_again(monkeypatch, caplog) -> None:
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine="kokoro_tts", tts_voice="af_heart")
    monkeypatch.setattr(PilotConfig, "load", lambda: config)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)
    caplog.set_level(logging.WARNING)
    with (
        patch.object(voice, "_run_local_tts_worker", new=AsyncMock(side_effect=LocalTTSUnavailable("down"))),
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="Spoken: hi...")) as system_voice,
    ):
        result = await voice._speak_impl("hi")
    assert result == "Spoken: hi..."
    system_voice.assert_awaited_once()
    assert not [r for r in caplog.records if r.name == voice.logger.name]


@pytest.mark.asyncio
async def test_changing_engine_or_device_releases_the_running_worker(monkeypatch) -> None:
    clock, worker = FakeClock(), FakeWorker()
    manager = _manager(Spawner(worker), clock)
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    await manager.ensure_ready("kokoro_tts", "af_heart")

    local_tts_worker.configure_local_voice(VoiceConfig(tts_engine="kokoro_tts", tts_keep_loaded_seconds=60))
    await _settle()
    assert manager.running and manager.keep_loaded_seconds == 60

    local_tts_worker.configure_local_voice(VoiceConfig(tts_engine="kokoro_tts", tts_device="cpu"))
    await _settle()
    assert not manager.running


def test_settings_are_clamped_to_sensible_values() -> None:
    manager = LocalTTSWorkerManager()
    manager.configure(keep_loaded_seconds=-5, device="tpu")
    assert manager.keep_loaded_seconds == 300.0 and manager.device == "auto"
    manager.configure(keep_loaded_seconds=10**9)
    assert manager.keep_loaded_seconds == 4 * 3600.0
    manager.configure(keep_loaded_seconds=0, device="cpu")
    assert manager.keep_loaded_seconds == 0 and manager.device == "cpu"


def test_default_settings_work_without_a_config_file() -> None:
    assert VoiceConfig().tts_keep_loaded_seconds == 300.0
    assert VoiceConfig().tts_device == "auto"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Okay.", ["Okay."]),
        ("Done. I moved three files! Want more?", ["Done.", "I moved three files!", "Want more?"]),
        ("Dr. Smith called at 3.5 p.m. today.", ["Dr. Smith called at 3.5 p.m. today."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        ("Line one\nLine two", ["Line one", "Line two"]),
        ("A. Then more.", ["A. Then more."]),
        ("No ending punctuation", ["No ending punctuation"]),
        ("   ", []),
    ],
)
def test_split_for_speech(text: str, expected: list[str]) -> None:
    assert split_for_speech(text) == expected


def _fake_hub(cached: set[str]) -> ModuleType:
    hub = ModuleType("huggingface_hub")
    hub.constants = SimpleNamespace(HF_HUB_OFFLINE=False)
    hub.try_to_load_from_cache = lambda repo, name: f"/cache/{name}" if name in cached else None
    return hub


def test_cached_weights_skip_the_network_check(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(kokoro_tts, "_offline_by_us", False)
    hub = _fake_hub({"config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"})
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        assert kokoro_tts.use_cached_weights_offline("af_heart") is True
        assert hub.constants.HF_HUB_OFFLINE is True
        kokoro_tts._allow_fetching_missing_voice("bf_emma")  # a voice picked later, not on disk yet
    assert hub.constants.HF_HUB_OFFLINE is False
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


def test_missing_weights_keep_the_network_check(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    hub = _fake_hub({"config.json"})
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        assert kokoro_tts.use_cached_weights_offline("af_heart") is False
    assert hub.constants.HF_HUB_OFFLINE is False


def test_a_broken_gpu_falls_back_to_the_cpu(monkeypatch) -> None:
    pipeline = MagicMock()
    module = ModuleType("kokoro")
    module.KPipeline = MagicMock(side_effect=[RuntimeError("CUDA error: unknown error"), pipeline])
    monkeypatch.setattr(kokoro_tts, "_device", "auto")
    kokoro_tts._pipeline_cache.clear()
    with patch.dict(sys.modules, {"kokoro": module}):
        assert kokoro_tts._get_pipeline() is pipeline
    assert module.KPipeline.call_args.kwargs["device"] == "cpu"
    kokoro_tts._pipeline_cache.clear()


def test_prepare_loads_and_makes_one_throwaway_sentence(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(kokoro_tts, "use_cached_weights_offline", lambda v: calls.append("offline") or True)
    monkeypatch.setattr(kokoro_tts, "_get_pipeline", lambda: calls.append("load"))
    monkeypatch.setattr(kokoro_tts, "_generate", lambda text, v: calls.append(f"say {text}"))
    kokoro_tts.prepare("af_heart")
    assert calls == ["offline", "load", "say Okay."]


def test_cpu_device_is_passed_to_kokoro(monkeypatch) -> None:
    module = ModuleType("kokoro")
    module.KPipeline = MagicMock()
    kokoro_tts._pipeline_cache.clear()
    monkeypatch.setattr(kokoro_tts, "_device", "auto")
    kokoro_tts.configure(device="cpu")
    with patch.dict(sys.modules, {"kokoro": module}):
        kokoro_tts._get_pipeline()
    assert module.KPipeline.call_args.kwargs == {"lang_code": "a", "repo_id": "hexgrad/Kokoro-82M", "device": "cpu"}
    kokoro_tts._pipeline_cache.clear()
