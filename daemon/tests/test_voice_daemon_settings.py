"""The voice speaks with the settings the daemon holds, reads no file per reply, and a change of engine while a piece
is being made never leaves the voice failed.

In a measurement on 24 November the daemon ran Pocket TTS while the settings file said Kokoro: every reply read the
file, moved the worker to Kokoro, and the warm-up moved it back ("Local voice pocket_tts is unavailable (ValueError:
write to closed file)", then "kokoro_tts loaded in 11 s"). Fake workers and made-up settings only.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from pilot.system import local_tts_worker, voice
from pilot.system.local_tts_worker import LocalTTSWorkerManager


@pytest.mark.asyncio
async def test_a_reply_speaks_with_the_daemon_s_settings_and_reads_no_file(monkeypatch):
    def no_file():
        raise AssertionError("the settings file was read for a reply")

    monkeypatch.setattr(voice.PilotConfig, "load", staticmethod(no_file))
    held = SimpleNamespace(voice=SimpleNamespace(tts_engine="pocket_tts", tts_voice="alba", system_voice=""))
    monkeypatch.setattr(voice, "_daemon_config", None)
    voice.use_daemon_config(held)
    spoken: list[tuple[str, str]] = []

    async def worker(engine, text, voice_name, output_file, *, wait_for_load=False):
        spoken.append((engine, voice_name))

    monkeypatch.setattr(voice, "_run_local_tts_worker", worker)
    await voice._speak_impl("Hello there.")
    held.voice.tts_engine, held.voice.tts_voice = "kokoro_tts", "af_heart"  # Settings changed it in place
    await voice._speak_impl("And again.")
    assert spoken == [("pocket_tts", "alba"), ("kokoro_tts", "af_heart")]


def test_outside_the_daemon_the_file_is_read_as_before(monkeypatch):
    monkeypatch.setattr(voice, "_daemon_config", None)
    marker = object()
    monkeypatch.setattr(voice.PilotConfig, "load", staticmethod(lambda: marker))
    assert voice._speech_config() is marker


@pytest.mark.asyncio
async def test_a_request_on_a_worker_closed_on_purpose_does_not_fail_the_voice():
    manager = LocalTTSWorkerManager(spawn=lambda engine, device: None)
    old = SimpleNamespace(poll=lambda: 0, stdin=None, stdout=None, wait=lambda timeout=None: 0)
    manager.process, manager.engine = old, "pocket_tts"
    await manager._close(old)  # the engine changed: the old worker is closed
    error = await manager._fail(old, "pocket_tts", "ValueError: write to closed file")
    assert isinstance(error, local_tts_worker.LocalTTSUnavailable)
    assert not manager.failing and manager.unavailable_reason == ""


class GatedWorker:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.asked: list[str] = []
        self.returncode: int | None = None
        self._payload = b""
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=self._close, closed=False)
        self.stdout = SimpleNamespace(readline=self._readline, read=lambda size: b"", close=lambda: None)

    def _write(self, payload: bytes) -> None:
        if self.stdin.closed:
            raise ValueError("write to closed file")
        self._payload = payload

    def _close(self) -> None:
        self.stdin.closed = True
        self.returncode = 0

    def _readline(self) -> bytes:
        request = json.loads(self._payload)
        if request.get("op") == "load":
            return b'{"status":"ok","op":"load"}\n'
        self.asked.append(request["text"])
        self.gate.wait(10)
        from pathlib import Path

        Path(request["output_file"]).write_bytes(b"RIFF-audio")
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_changing_engine_while_a_piece_is_made_waits_for_it_and_the_voice_stays_usable(monkeypatch):
    monkeypatch.setattr(local_tts_worker, "LOAD_GRACE_SECONDS", 60.0)
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    workers: list[GatedWorker] = []

    def spawn(engine, device, gpu_lease=False):
        made = GatedWorker()
        if workers:
            made.gate.set()  # the new engine's worker answers at once
        workers.append(made)
        return made

    manager = LocalTTSWorkerManager(spawn=spawn)
    manager.environ = {}
    monkeypatch.setattr(local_tts_worker, "_manager", manager)
    played: list[str] = []

    async def play(piece) -> None:
        played.append(piece.text)

    speaking = asyncio.ensure_future(manager.speak("kokoro_tts", "A long answer.", "af_heart", None, play))
    for _ in range(500):
        if workers and workers[0].asked:
            break
        await asyncio.sleep(0.01)
    # The setting changes under the piece being made (here the device; a change of engine takes the same path).
    local_tts_worker.configure_local_voice(
        SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart", tts_device="cpu")
    )
    manager.free_memory = manager.light_reason = manager.keep_while = None  # a made-up machine with room
    manager.choose_device = None
    for _ in range(10):
        await asyncio.sleep(0)
    assert manager.process is workers[0] and not workers[0].stdin.closed  # not closed under the piece being made
    workers[0].gate.set()
    await speaking
    for _ in range(500):
        if workers[0].stdin.closed:
            break
        await asyncio.sleep(0.01)
    assert played == ["A long answer."] and workers[0].stdin.closed and not manager.failing
    await manager.speak("kokoro_tts", "Hello.", "af_heart", None, play)  # the new worker speaks at once
    assert len(workers) == 2 and played[-1] == "Hello." and not manager.failing
    await manager.shutdown()
