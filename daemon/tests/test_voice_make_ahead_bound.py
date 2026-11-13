"""Making fixed sentences ahead never delays a reply: no piece starts while Heliox has been used in the last
MAKE_AHEAD_IDLE_SECONDS, a reply in the store plays at once even while a piece is being made, and a piece already
being made when the first request after a quiet stretch arrives adds at most one short piece's synthesis. Stand-in
workers whose synthesis takes a set time; no model, no audio device.

Every check reads one ordered log of events (asked, synthesis start and end, playback), never the clock: on Windows
before Python 3.13 time.monotonic() moves in 15 ms steps, and on a busy machine wall times stretch."""

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
PIECES = [f"Piece number {n}." for n in range(40)]


class TimedWorker:
    """Synthesis takes PIECE_SECONDS; every start and end goes into ``log`` in the order it happened."""

    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.log = log
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
            return b'{"status":"ok","op":"load"}\n'
        text = Path(request["text_file"]).read_text(encoding="utf-8")
        self.log.append(("start", text))
        time.sleep(PIECE_SECONDS)
        Path(request["output_file"]).write_bytes(f"RIFF-{text}".encode())
        self.log.append(("end", text))
        return b'{"status":"ok"}\n'

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


class Run:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def note(self, kind: str, text: str) -> None:
        with self.lock:
            self.log.append((kind, text))

    def between(self, first: tuple[str, str], last: tuple[str, str]) -> list[tuple[str, str]]:
        """The events after ``first`` and before ``last``."""
        start = self.log.index(first)
        return self.log[start + 1 : self.log.index(last, start)]


async def ready_manager(tmp_path, monkeypatch) -> tuple[LocalTTSWorkerManager, Run]:
    monkeypatch.setattr(local_tts_worker, "trim_silence", lambda path, lead, tail: None)
    run = Run()
    worker = TimedWorker(run.log)
    manager = LocalTTSWorkerManager(spawn=lambda engine, device, gpu_lease=False: worker)
    manager.head_cache = SpeechStore(tmp_path / "voice_heads")
    await manager.ensure_ready("kokoro_tts", "af_heart")
    return manager, run


async def say(manager: LocalTTSWorkerManager, run: Run, text: str) -> None:
    """Hand ``text`` to the voice; the log gets ("asked", text) and ("play", text) for its first piece."""
    run.note("asked", text)
    played: list[bool] = []

    async def play(_path: Path) -> None:
        if not played:
            played.append(True)
            run.note("play", text)

    await manager.speak("kokoro_tts", text, "af_heart", None, play)


def pieces_in(events: list[tuple[str, str]], kind: str) -> list[str]:
    return [text for event, text in events if event == kind and text in PIECES]


async def making(manager: LocalTTSWorkerManager, **options) -> asyncio.Future[int]:
    return asyncio.ensure_future(
        manager.make_ahead("kokoro_tts", "af_heart", PIECES, quiet=lambda: True, poll_seconds=0.01, **options)
    )


async def stop(task: asyncio.Future[int], manager: LocalTTSWorkerManager) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_a_reply_right_after_a_cut_never_waits_for_a_piece_made_ahead(tmp_path, monkeypatch):
    """CI's control scenarios: a reply cut off, then "Okay." asked for 50 ms later, with make-ahead allowed (the room
    quiet). Before the bound, a piece started a second after the cut and the reply waited for it."""
    manager, run = await ready_manager(tmp_path, monkeypatch)
    task = await making(manager)
    await say(manager, run, "Here is a long answer that the person cuts off.")
    await asyncio.sleep(0.05)
    await say(manager, run, "Okay.")
    waited = run.between(("asked", "Okay."), ("start", "Okay."))
    assert pieces_in(waited, "end") == [] and pieces_in(run.log, "start") == []  # nothing made in the idle stretch
    await stop(task, manager)


@pytest.mark.asyncio
async def test_the_rule_before_the_bound_reproduces_the_ci_delay(tmp_path, monkeypatch):
    """The cause, reproduced: with no idle stretch (the rule on f6a1f8e waited for one second of microphone quiet,
    which the bench's room has right after a cut), a piece starts as soon as the cut reply ends, and the "Okay." asked
    for 50 ms later waits for it: its synthesis starts only after that piece ends."""
    manager, run = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    task = await making(manager, rest_factor=0)
    await say(manager, run, "Here is a long answer that the person cuts off.")
    for _ in range(500):  # a piece starts as soon as the cut reply has ended
        await asyncio.sleep(0.01)
        if pieces_in(run.log, "start"):
            break
    await say(manager, run, "Okay.")
    waited = run.between(("asked", "Okay."), ("start", "Okay."))
    assert len(pieces_in(waited, "end")) == 1  # the reply's synthesis waited for a piece made ahead to end
    await stop(task, manager)


@pytest.mark.asyncio
async def test_a_reply_in_the_store_plays_at_once_even_while_a_piece_is_being_made(tmp_path, monkeypatch):
    manager, run = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)  # past the quiet stretch
    made = tmp_path / "okay.wav"
    made.write_bytes(b"RIFF-Okay.")
    manager.head_cache.add_fixed("Okay.", SpeechStore.signature("kokoro_tts", "af_heart"), made)
    task = await making(manager, rest_factor=0)
    for _ in range(500):  # until a piece is being made
        await asyncio.sleep(0.01)
        if pieces_in(run.log, "start"):
            break
    piece = pieces_in(run.log, "start")[-1]
    await say(manager, run, "Okay.")
    assert ("end", piece) not in run.between(("start", piece), ("play", "Okay."))  # played while it was being made
    assert ("start", "Okay.") not in run.log  # from the store: never synthesised
    await stop(task, manager)


@pytest.mark.asyncio
async def test_the_first_request_after_a_quiet_stretch_waits_at_most_one_short_piece(tmp_path, monkeypatch):
    manager, run = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    task = await making(manager, rest_factor=0)
    for _ in range(500):  # the worst moment: a piece has just started
        await asyncio.sleep(0.01)
        if pieces_in(run.log, "start"):
            break
    await say(manager, run, "It's 10:42.")  # a reply with a changing part, not in the store
    waited = run.between(("asked", "It's 10:42."), ("start", "It's 10:42."))
    assert len(pieces_in(waited, "end")) <= 1 and pieces_in(waited, "start") == []  # one piece at most, none begun
    await stop(task, manager)


@pytest.mark.asyncio
async def test_only_short_pieces_are_made_ahead(tmp_path, monkeypatch):
    manager, run = await ready_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(local_tts_worker, "MAKE_AHEAD_IDLE_SECONDS", 0.0)
    long_piece = "This fixed sentence has far too many words to be made ahead of time by the voice."
    made = await manager.make_ahead(
        "kokoro_tts", "af_heart", ["Done.", long_piece], quiet=lambda: True, poll_seconds=0.01, rest_factor=0
    )
    assert made == 1 and [text for event, text in run.log if event == "start"] == ["Done."]
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
