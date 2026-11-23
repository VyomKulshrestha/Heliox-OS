"""One voice per reply: a reply that began in the system voice ends in it, even when Heliox's own voice finishes
loading partway, and even for sentences the store could play; the next reply (the person spoke, or Heliox was quiet
for REPLY_GAP_SECONDS) is in Heliox's voice. Audio made ahead counts as ready only while Heliox's voice speaks, so no
hum comes in one voice before an answer in another.

Fake workers and a clock the test moves: no model, no audio device, no real elapsed time decides anything."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig, VoiceConfig
from pilot.system import local_tts_worker, voice
from pilot.system.local_tts_worker import LocalTTSWorkerManager, LocalVoiceNotReady, SpeechStore
from pilot.system.platform_detect import Platform


class Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


class HeldLoadWorker:
    """A worker whose load finishes only when the test sets ``loaded``."""

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


async def settle(condition) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("never happened")


@pytest.fixture
def cold(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 0.01)  # the grace is not what is tested here
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker = HeldLoadWorker()
    clock = Clock()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: worker, clock=clock)
    manager.environ = {}
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    return manager, worker, clock


async def say(manager: LocalTTSWorkerManager, text: str) -> list[str]:
    played: list[str] = []

    async def play(piece) -> None:
        played.append(piece.text)

    await manager.speak("kokoro_tts", text, "af_heart", None, play)
    return played


@pytest.mark.asyncio
async def test_a_reply_that_began_in_the_system_voice_ends_in_it_and_the_next_one_is_heliox_s(cold):
    manager, worker, _clock = cold
    with pytest.raises(LocalVoiceNotReady) as first:
        await say(manager, "First sentence.")
    assert first.value.branch == "system_still_loading"
    worker.loaded.set()  # the load ends between two sentences of the same reply
    await settle(lambda: manager.loaded)
    with pytest.raises(LocalVoiceNotReady) as second:
        await say(manager, "Second sentence.")
    assert second.value.branch == "system_same_reply" and worker.texts == []
    assert manager.speaks_locally_now("kokoro_tts") is False

    manager.begin_reply()  # the person speaks again
    assert manager.speaks_locally_now("kokoro_tts") is True
    assert await say(manager, "The next reply.") == ["The next reply."]
    assert local_tts_worker.manager().last_reply.branch == "ready"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_person_being_heard_is_a_new_reply(cold):
    manager, worker, _clock = cold
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "First sentence.")
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    manager.prewarm("kokoro_tts", "af_heart")  # what the listener does when the person speaks to Heliox
    assert await say(manager, "In Heliox's voice.") == ["In Heliox's voice."]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_quiet_gap_ends_the_reply(cold):
    manager, worker, clock = cold
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "First sentence.")
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    clock.now += local_tts_worker.REPLY_GAP_SECONDS - 1  # a slow model: still the same reply
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "Still this reply.")
    clock.now += local_tts_worker.REPLY_GAP_SECONDS + 1  # Heliox said nothing for longer: a new reply
    assert await say(manager, "Something new.") == ["Something new."]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_kept_pieces_are_not_played_while_the_reply_is_in_the_system_voice(cold, tmp_path):
    manager, worker, _clock = cold
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    signature = SpeechStore.signature("kokoro_tts", "af_heart")
    kept = tmp_path / "okay.wav"
    kept.write_bytes(b"RIFF-kept")
    assert manager.head_cache.add_fixed("Okay.", signature, kept)
    config = SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart")
    assert local_tts_worker.made_ahead(config, "Okay.") is False  # not loaded: the hum would be the system voice's
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "Okay.")  # in the store, and still the system voice: no switch inside the reply
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    assert local_tts_worker.made_ahead(config, "Okay.") is False  # loaded, but this reply began in the system voice
    with pytest.raises(LocalVoiceNotReady):
        await say(manager, "Okay.")
    manager.begin_reply()
    assert local_tts_worker.made_ahead(config, "Okay.") is True
    assert await say(manager, "Okay.") == ["Okay."]
    assert worker.texts == [] and local_tts_worker.manager().last_reply.voice == "store"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_system_voice_s_own_speech_does_not_count_as_quiet(cold, monkeypatch):
    """The system voice speaks while _speak_impl waits for it: a long sentence of it is not a gap between replies."""
    manager, worker, clock = cold
    config = PilotConfig()
    config.voice = VoiceConfig(tts_engine="kokoro_tts", tts_voice="af_heart")
    monkeypatch.setattr(PilotConfig, "load", lambda: config)
    monkeypatch.setattr(voice, "CURRENT_PLATFORM", Platform.WINDOWS)

    async def long_system_sentence(*_args, **_kwargs) -> str:
        clock.now += local_tts_worker.REPLY_GAP_SECONDS * 3  # it takes a while to say
        return "Spoken"

    monkeypatch.setattr(voice, "_tts_windows", AsyncMock(side_effect=long_system_sentence))
    await voice._speak_impl("A long first sentence in the system voice.")
    worker.loaded.set()
    await settle(lambda: manager.loaded)
    await voice._speak_impl("The second sentence.")
    assert worker.texts == []  # still the system voice
    assert manager.last_reply.voice == "system" and manager.last_reply.branch == "system_same_reply"
    await manager.shutdown()
