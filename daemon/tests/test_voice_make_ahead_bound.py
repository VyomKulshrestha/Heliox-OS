"""Making fixed sentences ahead never delays a reply: no piece starts while Heliox has been used in the last
MAKE_AHEAD_IDLE_SECONDS, a reply in the store plays at once even while a piece is being made, and a piece already
being made when the first request after a quiet stretch arrives adds at most one short piece's synthesis. Stand-in
workers whose synthesis takes a set time; no model, no audio device."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.system import local_tts_worker
from pilot.system.local_tts_worker import LocalTTSWorkerManager, SpeechStore

PIECE_SECONDS = 0.3  # a stand-in synthesis time for every piece


class TimedWorker:
    """Synthesis takes PIECE_SECONDS; records (text, started, ended) for every piece made."""

    def __init__(self) -> None:
        self.made: list[tuple[str, float, float]] = []
        self.returncode: int | None = None
        self._payload = b""
        self._lock = threading.Lock()
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
            return b'{"status":"ok","op":"load"}\n'
        text = Path(request["text_file"]).read_text(encoding="utf-8")
        started = time.monotonic()
        time.sleep(PIECE_SECONDS)
        Path(request["output_file"]).write_bytes(f"RIFF-{text}".encode())
        self.made.append((text, started, time.monotonic()))
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


async def ready_manager(tmp_path, monkeypatch) -> tuple[LocalTTSWorkerManager, TimedWorker]:
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    worker = TimedWorker()
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    await manager.ensure_ready("kokoro_tts", "af_heart")
    return manager, worker


async def first_play(manager: LocalTTSWorkerManager, text: str) -> float:
    """Seconds from handing ``text`` to the voice to its first piece reaching playback."""
    handed = time.monotonic()
    played: list[float] = []

    async def play(_path: Path) -> None:
        played.append(time.monotonic())

    await manager.speak("kokoro_tts", text, "af_heart", None, play)
    return played[0] - handed


PIECES = [f"Piece number {n}." for n in range(40)]


@pytest.mark.asyncio
async def test_a_reply_right_after_a_cut_never_waits_for_a_piece_made_ahead(tmp_path, monkeypatch):
    """CI's control scenarios: a reply cut off, then "Okay." asked for 50 ms later, with make-ahead allowed (the room
    quiet). Before the bound, a piece started a second after the cut and the reply waited for it."""
    manager, worker = await ready_manager(tmp_path, monkeypatch)
    making = asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", PIECES, quiet=lambda: True, poll_seconds=0.01)
    )
    await first_play(manager, "Here is a long answer that the person cuts off.")  # the reply that is cut
    await asyncio.sleep(0.05)
    waited = await first_play(manager, "Okay.")
    assert waited < PIECE_SECONDS + 0.2  # its own synthesis only
    assert [text for text, _s, _e in worker.made if text in PIECES] == []  # nothing made within the idle stretch
    making.cancel()
    await asyncio.gather(making, return_exceptions=True)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_rule_before_the_bound_reproduces_the_ci_delay(tmp_path, monkeypatch):
    """The cause, reproduced: with no idle stretch (the rule on f6a1f8e waited for one second of microphone quiet,
    which the bench's room has right after a cut), a piece starts as soon as the cut reply ends, and the "Okay." asked
    for 50 ms later waits for it: its first sound comes a piece's synthesis late."""
    manager, worker = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    making = asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", PIECES, quiet=lambda: True, poll_seconds=0.01, rest_factor=0)
    )
    await first_play(manager, "Here is a long answer that the person cuts off.")
    await asyncio.sleep(0.05)
    waited = await first_play(manager, "Okay.")
    assert waited > PIECE_SECONDS + 0.1  # its own synthesis and most of a piece made ahead
    assert any(text in PIECES for text, _s, _e in worker.made)
    making.cancel()
    await asyncio.gather(making, return_exceptions=True)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_reply_in_the_store_plays_at_once_even_while_a_piece_is_being_made(tmp_path, monkeypatch):
    manager, worker = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)  # past the quiet stretch
    sig = SpeechStore.signature("kokoro_tts", "af_heart")
    made = tmp_path / "okay.wav"
    made.write_bytes(b"RIFF-Okay.")
    manager.head_cache.add_fixed("Okay.", sig, made)
    making = asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", PIECES, quiet=lambda: True, poll_seconds=0.01, rest_factor=0)
    )
    await asyncio.sleep(0.1)  # a piece is being made now
    assert await first_play(manager, "Okay.") < 0.05  # played from the store; the worker is not needed
    making.cancel()
    await asyncio.gather(making, return_exceptions=True)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_the_first_request_after_a_quiet_stretch_waits_at_most_one_short_piece(tmp_path, monkeypatch):
    manager, worker = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    making = asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", PIECES, quiet=lambda: True, poll_seconds=0.01, rest_factor=0)
    )
    await asyncio.sleep(0.1)  # the worst moment: a piece has just started
    waited = await first_play(manager, "It's 10:42.")  # a reply with a changing part, not in the store
    assert waited < 2 * PIECE_SECONDS + 0.2  # its own synthesis, plus at most the one piece in progress
    making.cancel()
    await asyncio.gather(making, return_exceptions=True)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_only_short_pieces_are_made_ahead(tmp_path, monkeypatch):
    manager, worker = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    long_piece = "This fixed sentence has far too many words to be made ahead of time by the voice."
    made = await manager.make_ahead(
        "kokoro_tts", "af_heart", ["Done.", long_piece], quiet=lambda: True, poll_seconds=0.01, rest_factor=0
    )
    assert made == 1 and [text for text, _s, _e in worker.made] == ["Done."]
    assert len(long_piece.split()) > local_tts_worker.MAKE_AHEAD_MAX_WORDS
    await manager.shutdown()


def test_a_conversation_keeps_make_ahead_waiting(monkeypatch):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    item = PilotServer(PilotConfig())
    recorder = SimpleNamespace(last_above_threshold_at=0.0, user_is_speaking=False)
    item._voice_listener = SimpleNamespace(_recorder=recorder, busy=False, conversation_open=lambda: True)
    assert not item._nobody_talking_to_heliox()  # a conversation is open
    item._voice_listener.conversation_open = lambda: False
    assert item._nobody_talking_to_heliox()  # quiet for long, no conversation
    recorder.last_above_threshold_at = time.time() - 5
    assert not item._nobody_talking_to_heliox()  # heard 5 s ago: under the 30 s stretch
