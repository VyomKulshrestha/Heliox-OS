"""Data the bench reads about the voice, without changing what it does: a public release that runs what the idle
release runs, which voice spoke each reply and by which branch, when loads and releases happened, and when each
synthesis was asked for, taken and done. Fake workers only."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import local_tts_worker, voice
from pilot.system.companion_speech import CompanionSpeechCoordinator
from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady, SpeechStore
from pilot.system.platform_detect import Platform


class Worker:
    def __init__(self) -> None:
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
        if request.get("op") != "load":
            Path(request["output_file"]).write_bytes(b"RIFF")
        return b'{"status":"ok"}\n'

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def manager(monkeypatch) -> LocalTTSWorkerManager:
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker = Worker()
    return LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)


@pytest.mark.asyncio
async def test_release_runs_the_idle_release_and_force_overrides_the_keep_rule(monkeypatch):
    item = manager(monkeypatch)
    await item.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    item.keep_while = lambda: True  # memory to spare: the keep rule would keep it
    assert await item.release("bench") is False  # just used, and kept by memory
    assert item.running
    assert await item.release("bench", force=True) is True
    assert not item.running and item.reload_log[-1][:2] == ("released", "bench")
    assert await item.release("bench", force=True) is False  # nothing left to release


@pytest.mark.asyncio
async def test_never_released_while_a_reply_is_being_spoken(monkeypatch):
    item = manager(monkeypatch)
    gate = asyncio.Event()

    async def play(_path: Path) -> None:
        await gate.wait()

    speaking = asyncio.ensure_future(item.speak("kokoro_tts", "A long answer.", "af_heart", None, play))
    for _ in range(50):
        await asyncio.sleep(0.01)
    assert await item.release("bench", force=True) is False and item.running
    gate.set()
    await speaking
    await item.shutdown()


@pytest.mark.asyncio
async def test_loads_releases_and_syntheses_are_stamped(monkeypatch):
    item = manager(monkeypatch)
    await item.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())
    events = [event for event, _detail, _at in item.reload_log]
    assert events[:2] == ["load_started", "load_ended"]
    kind, words, requested, started, done = item.synth_log[-1]
    assert (kind, words) == ("reply", 1) and requested <= started <= done
    await item.release("bench", force=True)
    assert [event for event, _d, _a in item.reload_log][-1] == "released"


@pytest.mark.asyncio
async def test_which_voice_spoke_each_reply_and_by_which_branch(monkeypatch, tmp_path):
    item = manager(monkeypatch)
    monkeypatch.setattr(local_tts_worker, "_manager", item)
    await item.speak("kokoro_tts", "Okay.", "af_heart", None, AsyncMock())  # loaded within the grace (cold)
    assert (item.last_reply.voice, item.last_reply.branch) == ("local", "loaded_within_grace")
    await item.speak("kokoro_tts", "Done.", "af_heart", None, AsyncMock())
    assert (item.last_reply.voice, item.last_reply.branch) == ("local", "ready")
    item.head_cache = SpeechStore(tmp_path / "voice_heads")
    made = tmp_path / "a.wav"
    made.write_bytes(b"RIFF")
    item.head_cache.add_fixed("Next track.", SpeechStore.signature("kokoro_tts", "af_heart"), made)
    await item.speak("kokoro_tts", "Next track.", "af_heart", None, AsyncMock())
    assert (item.last_reply.voice, item.last_reply.branch) == ("store", "store")
    await item.shutdown()


@pytest.mark.asyncio
async def test_the_system_voice_is_recorded_with_why(monkeypatch, caplog):
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine="kokoro_tts", tts_voice="af_heart")
    monkeypatch.setattr(PilotConfig, "load", lambda: config)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)
    caplog.set_level(logging.WARNING)
    with (
        patch.object(voice, "_run_local_tts_worker", new=AsyncMock(side_effect=LocalVoiceNotReady("loading"))),
        patch.object(voice, "_tts_windows", new=AsyncMock(return_value="Spoken: hi...")),
    ):
        await voice._speak_impl("hi")
    record = local_tts_worker.manager().last_reply
    assert (record.voice, record.branch) == ("system", "system_still_loading") and record.reason
    config.voice = VoiceConfig(tts_engine="os_native")
    with patch.object(voice, "_tts_windows", new=AsyncMock(return_value="Spoken: hi...")):
        await voice._speak_impl("hi")
    assert local_tts_worker.manager().last_reply.branch == "system_chosen"


@pytest.mark.asyncio
async def test_the_speech_result_carries_the_voice():
    async def speaker(text: str, recorder) -> bool:
        local_tts_worker.record_reply_voice("system", "system_still_loading", "Heliox's own voice was still loading")
        return False

    coordinator = CompanionSpeechCoordinator(speaker=speaker)
    outcome = await coordinator.speak("It's 10:42.")
    assert (outcome.voice, outcome.voice_branch) == ("system", "system_still_loading") and outcome.voice_reason

    async def silent(text: str, recorder) -> bool:
        return False  # nothing recorded for this request

    coordinator_two = CompanionSpeechCoordinator(speaker=silent)
    assert (await coordinator_two.speak("Okay.")).voice is None
    await coordinator.close()
    await coordinator_two.close()


def test_the_vocabulary_is_closed():
    assert set(local_tts_worker.REPLY_VOICES) == {"local", "store", "system"}
    assert "waited_no_system_voice" in local_tts_worker.REPLY_BRANCHES
