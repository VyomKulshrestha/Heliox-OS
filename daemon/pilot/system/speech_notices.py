"""What the window is told about Heliox's own speech: a notice per piece as it goes to the speaker, one at the end of
a reply, and one at the moment a cut-off starts its fade.

The daemon plays every sound itself, so without these the window cannot know when Heliox speaks. They are for a face
that moves its mouth with the voice (Settings, Appearance, "Face"; docs/FACE_PACKS.md) and for anything else in the
window that follows speech. Three rules hold everywhere here:

* Speech never waits for the window. A notice is put on a queue and sent by its own task; nothing awaits it.
* The cut notice goes out when the fade starts (voice._play_local_tts_file), not after the speech task has ended.
* The mouth data (a timeline of mouth shapes from the voice's own phoneme timing, or else a loudness envelope) is made
  and sent only while a face is shown (``face_on``). Without one each piece's notice carries its length and timing.

IPC_MESSAGE_FORMATS.md has the notices' fields (voice_piece, voice_end, voice_cut).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pilot.system import mouth_shapes

logger = logging.getLogger("pilot.system.speech_notices")

PIECE = "voice_piece"
END = "voice_end"
CUT = "voice_cut"
QUEUE_LIMIT = 256  # notices not yet sent; past it new ones are dropped (a window that stopped reading)
# The phoneme times must describe the audio played: a timing file whose length differs from the audio's by more than
# this is not used (the envelope is, from the samples themselves).
TIMING_TOLERANCE_SECONDS = 0.05

Send = Callable[[str, dict[str, Any]], Awaitable[Any]]


class _Hub:
    """Where notices go: one queue drained in order by one task on the daemon's event loop."""

    def __init__(self) -> None:
        self.send: Send | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue[tuple[str, Any]] | None = None
        self.task: asyncio.Task[None] | None = None
        self.face_on: Callable[[], bool] = lambda: False
        self.phase: Any = None  # the interaction runtime: .status() and async .transition(phase, message=)
        self.lock = threading.Lock()
        self.replies = 0
        self.dropped = 0


_hub = _Hub()
_current: ContextVar[Reply | None] = ContextVar("speech_notices_reply", default=None)


def install(send: Send, *, face_on: Callable[[], bool] | None = None, phase: Any = None) -> None:
    """Send notices with ``send(method, params)`` from now on (the server's broadcast). Call on the event loop."""
    uninstall()
    _hub.send = send
    _hub.loop = asyncio.get_running_loop()
    _hub.queue = asyncio.Queue()
    _hub.face_on = face_on or (lambda: False)
    _hub.phase = phase
    _hub.task = _hub.loop.create_task(_drain(_hub.queue))


def uninstall() -> None:
    task = _hub.task
    _hub.send = None
    _hub.loop = None
    _hub.queue = None
    _hub.task = None
    _hub.face_on = lambda: False
    _hub.phase = None
    if task is not None and not task.done():
        with contextlib.suppress(RuntimeError):  # its loop has already closed
            task.cancel()


def installed() -> bool:
    return _hub.send is not None and _hub.loop is not None and not _hub.loop.is_closed()


def face_on() -> bool:
    """A face is shown, so mouth data is worth making. False when nobody installed the notices."""
    if not installed():
        return False
    try:
        return bool(_hub.face_on())
    except Exception:
        return False


def _post(item: tuple[str, Any]) -> None:
    """Queue ``item`` without waiting, from the event loop or any other thread. Caller holds _hub.lock."""
    loop, queue = _hub.loop, _hub.queue
    if loop is None or queue is None or loop.is_closed():
        return

    def put() -> None:
        if queue.qsize() >= QUEUE_LIMIT:
            _hub.dropped += 1
            return
        queue.put_nowait(item)

    # Always through the loop's own queue of callbacks, also from the loop's thread: it keeps the order in which the
    # notices were posted (under _hub.lock) whatever thread posted them.
    with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
        loop.call_soon_threadsafe(put)


async def _drain(queue: asyncio.Queue[tuple[str, Any]]) -> None:
    while True:
        kind, payload = await queue.get()
        try:
            if kind == "phase":
                await _move_phase(*payload)
            elif _hub.send is not None:
                await _hub.send(kind, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("A speech notice could not be sent", exc_info=True)


async def _move_phase(reply: Reply, speaking: bool) -> None:
    """The interaction phase reads "speaking" while a reply plays, and goes back to what it was afterwards unless
    something else moved it meanwhile."""
    runtime = _hub.phase
    if runtime is None:
        return
    if speaking:
        before = runtime.status()
        if before.get("phase") == "speaking":
            return
        after = await runtime.transition("speaking", message="Speaking")
        reply.phase_before = (str(before.get("phase") or "idle"), str(before.get("message") or ""))
        reply.phase_sequence = after.get("sequence")
        return
    if reply.phase_before is None or runtime.status().get("sequence") != reply.phase_sequence:
        return
    phase, message = reply.phase_before
    await runtime.transition(phase, message=message)


class Reply:
    """One spoken reply (one speak()): its pieces, and whether it ended or was cut. Every notice of a reply goes out
    in the order it happened; a piece handed over after its reply was cut is never announced."""

    def __init__(self, number: int, voice: str = "") -> None:
        self.number = number
        self.voice = voice
        self.next_index = 0
        self.sent = 0  # pieces announced
        self.playing = -1  # the index of the last piece announced
        self.state = "open"  # "open" | "ended" | "cut"
        self.phase_before: tuple[str, str] | None = None
        self.phase_sequence: Any = None

    def take_index(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index

    def piece(self, params: dict[str, Any]) -> bool:
        """Announce a piece now (any thread). False when the reply is already over."""
        with _hub.lock:
            if self.state != "open":
                return False
            params = {"reply": self.number, **params, "sent_ms": round(time.time() * 1000, 1)}
            first = self.sent == 0
            self.sent += 1
            self.playing = int(params.get("index", self.sent - 1))
            _post((PIECE, params))
            if first:
                _post(("phase", (self, True)))
        return True

    def end(self) -> None:
        with _hub.lock:
            if self.state != "open":
                return
            self.state = "ended"
            if self.sent:
                _post((END, {"reply": self.number, "pieces": self.sent, "sent_ms": round(time.time() * 1000, 1)}))
                _post(("phase", (self, False)))

    def cut(self) -> None:
        """The voice is being cut off now: the window stops the mouth at once."""
        with _hub.lock:
            if self.state != "open":
                return
            self.state = "cut"
            if self.sent:
                _post((CUT, {"reply": self.number, "index": self.playing, "sent_ms": round(time.time() * 1000, 1)}))
                _post(("phase", (self, False)))


def current() -> Reply | None:
    return _current.get()


@contextlib.contextmanager
def reply(voice: str = "") -> Iterator[Reply]:
    """The reply being spoken: the one already open in this task, or a new one that ends when the block does (cut
    when it is cancelled)."""
    existing = _current.get()
    if existing is not None and existing.state == "open":
        yield existing
        return
    with _hub.lock:
        _hub.replies += 1
        made = Reply(_hub.replies, voice)
    token = _current.set(made)
    try:
        yield made
    except asyncio.CancelledError:
        made.cut()
        raise
    except BaseException:
        made.end()
        raise
    else:
        made.end()
    finally:
        _current.reset(token)


def unvoiced_piece() -> None:
    """The system voice is about to speak: it plays by itself, so there is no audio here to time a mouth by. The face
    stays still; the window still learns that Heliox speaks."""
    made = _current.get()
    if made is None or not installed():
        return
    made.piece({"index": made.take_index(), "seconds": None, "lead_ms": None, "voice": "system", "mouth": "none"})


def handed(mouth: dict[str, Any] | None, seconds: float) -> Callable[[float | None], None] | None:
    """For one piece about to play: the call the player makes when the piece is on the output line, with its
    estimate of the seconds until the first sample is heard. None when nobody listens."""
    made = _current.get()
    if made is None or not installed():
        return None
    index = made.take_index()

    def on_handed(lead: float | None) -> None:
        params: dict[str, Any] = {
            "index": index,
            "seconds": round(seconds, 3),
            "lead_ms": None if lead is None else round(max(0.0, lead) * 1000, 1),
            "voice": made.voice or "local",
        }
        params.update(mouth or {"mouth": "off"})
        made.piece(params)

    return on_handed


def cut_now() -> None:
    """The fade starts now (barge-in, or a newer reply took the voice)."""
    made = _current.get()
    if made is not None:
        made.cut()


# ── Phoneme timing beside the audio ─────────────────────────────────


def timing_path(audio_path: Path | str) -> Path:
    """Where the voice worker writes a piece's phoneme timing: beside its WAV."""
    path = Path(audio_path)
    return path.with_name(path.name + ".timing.json")


def shift_timing(audio_path: Path | str, start: int, stop: int) -> None:
    """The WAV was trimmed to samples [start, stop) of the audio as made: move every time so it counts from the
    first sample kept, and drop what was cut. Nothing to do without a timing file."""
    path = timing_path(audio_path)
    try:
        timing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    try:
        rate = int(timing.get("rate") or 24000)
        lead, length = start / rate, max(0, stop - start) / rate
        for key in ("phonemes", "words"):
            moved = []
            for name, begin, end in timing.get(key) or []:
                begin, end = max(0.0, float(begin) - lead), min(length, float(end) - lead)
                if end > begin:
                    moved.append([name, round(begin, 4), round(end, 4)])
            timing[key] = moved
        timing["samples"] = max(0, stop - start)
        path.write_text(json.dumps(timing, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        with contextlib.suppress(OSError):
            path.unlink()


# Kokoro's predicted durations place every phoneme about 0.1 s after the sound it names: on the real model the first
# phoneme started 0.099 s after the first audible sample, and stop closures and the pause at a colon showed the same
# lag through the sentence (scratch check, 19 November). Kokoro's own word times subtract a fixed 3 frames for this.
# Here the lag is measured on each piece instead: the first phoneme is moved to the first audible sample, by at most
# ONSET_LIMIT_SECONDS either way (beyond that the timing and the audio disagree for another reason: nothing moves).
ONSET_LIMIT_SECONDS = 0.2
_AUDIBLE = 0.002  # as local_tts_worker.SILENCE_LEVEL


def _onset_offset(phonemes: list[Any], audio: Any, rate: int) -> float:
    """Seconds the phoneme times run after the audio: first sounding phoneme's start minus first audible sample."""
    import numpy as np

    first = next((float(p[1]) for p in phonemes if mouth_shapes.sounds(str(p[0]))), None)
    level = np.abs(np.asarray(audio, dtype=np.float32))
    if level.ndim > 1:
        level = level.max(axis=1)
    loud = np.flatnonzero(level > _AUDIBLE)
    if first is None or loud.size == 0:
        return 0.0
    offset = first - int(loud[0]) / max(1, int(rate))
    return offset if abs(offset) <= ONSET_LIMIT_SECONDS else 0.0


def mouth_for(audio_path: Path | str | None, audio: Any, rate: int) -> dict[str, Any] | None:
    """The mouth data for one piece, when a face is shown (None otherwise): the shape timeline when the voice wrote
    phoneme timing that matches this audio, else the loudness envelope of the samples."""
    if not face_on():
        return None
    try:
        samples = len(audio)
    except TypeError:
        return None
    length = samples / max(1, int(rate))
    if audio_path is not None:
        try:
            timing = json.loads(timing_path(audio_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            timing = None
        if isinstance(timing, dict) and timing.get("phonemes"):
            made = int(timing.get("samples") or 0) / max(1, int(timing.get("rate") or rate))
            if abs(made - length) <= TIMING_TOLERANCE_SECONDS:
                early = _onset_offset(timing["phonemes"], audio, rate)
                phonemes = [[p, float(a) - early, float(b) - early] for p, a, b in timing["phonemes"]]
                shapes = mouth_shapes.timeline(phonemes, length)
                if shapes:
                    words = [
                        [str(w), max(0, round((float(a) - early) * 1000)), round((float(b) - early) * 1000)]
                        for w, a, b in timing.get("words") or []
                    ]
                    return {"mouth": "visemes", "visemes": shapes, "words": words, "onset_ms": round(early * 1000)}
    try:
        values = mouth_shapes.envelope(audio, rate)
    except Exception:
        logger.debug("Could not measure a piece's loudness for the face", exc_info=True)
        return {"mouth": "none"}
    return {"mouth": "envelope", "envelope": values, "envelope_rate": mouth_shapes.ENVELOPE_RATE}
