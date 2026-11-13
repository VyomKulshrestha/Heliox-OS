"""No reply waits for the local voice to load: when it is not ready within LOAD_GRACE_SECONDS the system voice speaks
that reply whole and the local voice takes over from the next one; a reply whose every piece is in the store of made
audio plays from it with the worker released. Fake workers only: no model, no audio device."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import local_tts_worker, voice
from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady, SpeechStore
from pilot.system.platform_detect import Platform


class SlowWorker:
    """Loads only when ``loaded`` is set (a cold load of 20 s or more, held as long as the test likes)."""

    def __init__(self) -> None:
        self.loaded = threading.Event()
        self.texts: list[str] = []
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
        if request.get("op") == "load":
            self.loaded.wait(10)
            return b'{"status":"ok","op":"load"}\n'
        text = Path(request["text_file"]).read_text(encoding="utf-8")
        self.texts.append(text)
        Path(request["output_file"]).write_bytes(f"RIFF-{text}".encode())
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


def manager_for(worker: SlowWorker, statuses: list[str]) -> LocalTTSWorkerManager:
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: worker)

    async def on_status(status: str, data: dict) -> None:
        statuses.append(status)

    manager.on_status = on_status
    return manager


async def settle(condition, seconds: float = 3.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_reply_does_not_wait_for_a_cold_load_and_the_next_one_is_in_heliox_s_voice(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker, statuses = SlowWorker(), []
    manager = manager_for(worker, statuses)
    play = AsyncMock()
    with pytest.raises(LocalVoiceNotReady):
        await manager.speak("kokoro_tts", "It's 10:42.", "af_heart", None, play)
    assert not worker.loaded.is_set()  # handed back while the load was still held: it did not wait for it
    play.assert_not_awaited()
    assert not manager.failing
    assert statuses == ["voice_loading"]  # the bar says the voice is getting ready; the system voice speaks

    worker.loaded.set()  # the load finishes in the background
    await settle(lambda: manager.loaded and "voice_ready" in statuses)
    assert statuses == ["voice_loading", "voice_ready"]
    await manager.speak("kokoro_tts", "Done.", "af_heart", None, play)
    assert worker.texts == ["Done."]  # the next reply in Heliox's own voice
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_reply_that_is_all_in_the_store_plays_with_the_worker_released(monkeypatch, tmp_path):
    spawned: list[str] = []
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: spawned.append(engine))
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    for text in ("Okay.", "Still on the plan."):
        made = tmp_path / f"{len(text)}.wav"
        made.write_bytes(f"RIFF-{text}".encode())
        assert manager.head_cache.add_fixed(text, sig, made)
    played: list[bytes] = []

    async def play(path: Path) -> None:
        played.append(path.read_bytes())

    await manager.speak("kokoro_tts", "Okay. Still on the plan.", "af_heart", None, play)
    assert played == [b"RIFF-Okay.", b"RIFF-Still on the plan."] and spawned == []  # no worker, no load

    # One piece not kept: one voice per reply, so the whole reply goes to the system voice while the voice loads.
    worker = SlowWorker()
    manager._spawn = lambda engine, device: worker
    with pytest.raises(LocalVoiceNotReady):
        await manager.speak("kokoro_tts", "Okay. Your files are in Downloads.", "af_heart", None, play)
    assert played == [b"RIFF-Okay.", b"RIFF-Still on the plan."]  # nothing of it in the local voice
    worker.loaded.set()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_system_voice_speaks_the_whole_reply_without_logging_a_failure(monkeypatch, caplog):
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine="kokoro_tts", tts_voice="af_heart")
    monkeypatch.setattr(PilotConfig, "load", lambda: config)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)
    caplog.set_level(logging.WARNING)
    with (
        patch.object(voice, "_run_local_tts_worker", new=AsyncMock(side_effect=LocalVoiceNotReady("loading"))),
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="Spoken: It's 10:42....")) as system,
    ):
        result = await voice._speak_impl("It's 10:42.")
    assert result.startswith("Spoken:")
    assert system.await_args.args[0] == "It's 10:42."  # the whole reply, in one voice
    assert not [r for r in caplog.records if r.name == voice.logger.name]


@pytest.mark.asyncio
async def test_with_no_system_voice_the_reply_waits_for_heliox_s_own_voice_never_silence(monkeypatch):
    """A Linux computer without speech-dispatcher or espeak (or a system voice that fails): the reply is not dropped.
    It waits for the local voice, and is spoken by it once loaded."""
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine="kokoro_tts", tts_voice="af_heart")
    monkeypatch.setattr(PilotConfig, "load", lambda: config)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.LINUX)
    calls: list[bool] = []

    async def local(engine, text, voice_name, output_file, *, wait_for_load=False):
        calls.append(wait_for_load)
        if not wait_for_load:
            raise LocalVoiceNotReady("loading")

    no_voice = "TTS unavailable: no speech program on this computer"
    with (
        patch.object(voice, "_run_local_tts_worker", new=local),
        patch.object(voice, "_tts_linux", new=AsyncMock(return_value=no_voice)),
    ):
        result = await voice._speak_impl("It's 10:42.")
    assert calls == [False, True]  # tried within the grace, then waited for the load
    assert result.startswith("Spoken:") and not voice.unheard_reason(result)


@pytest.mark.asyncio
async def test_waiting_for_the_load_speaks_in_the_local_voice_once_it_is_loaded(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker, statuses = SlowWorker(), []
    manager = manager_for(worker, statuses)
    played: list[bytes] = []

    async def play(path: Path) -> None:
        played.append(path.read_bytes())

    speaking = asyncio.ensure_future(
        manager.speak("kokoro_tts", "It's 10:42.", "af_heart", None, play, wait_for_load=True)
    )
    await asyncio.sleep(local_tts_worker.LOAD_GRACE_SECONDS + 0.2)
    assert not speaking.done() and played == []  # past the grace: still waiting, not dropped
    worker.loaded.set()
    await asyncio.wait_for(speaking, 5)
    assert played == [b"RIFF-It's 10:42."] and statuses == ["voice_loading", "voice_ready"]
    await manager.shutdown()
