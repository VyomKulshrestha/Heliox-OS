"""Process boundary for optional local TTS model inference, and its lifecycle.

The worker (``python -m pilot.system.local_tts_worker --engine kokoro_tts --serve``) keeps the model runtime out of
the daemon: importing Kokoro pulls in torch and transformers, about 1.6-1.9 GB resident plus about 840 MB of GPU
memory on CUDA. It answers one JSON line per request on stdout, followed by the raw samples the line announces
("bytes") for speech.

``LocalTTSWorkerManager`` is the daemon side. Measured on the owner's machine (scripts/measure_kokoro_startup.py), a
cold worker needs about 20-25 s before its first word: 0.25 s process start, 3.3 s importing torch, 12-13 s importing
Kokoro (transformers walks thousands of files), 2.7 s loading weights and 0.9 s for the first synthesis. A loaded
worker says a short sentence in 0.1 s and a long one in 0.25 s on the GPU. So the manager:

* keeps the worker loaded while a conversation is plausible: until ``keep_loaded_seconds`` (default 300) pass with
  nobody speaking to Heliox and Heliox saying nothing, then releases it;
* starts it ahead of need (``prewarm_local_voice``) as soon as the person is heard;
* speaks a long reply sentence by sentence, synthesising the next sentence while the current one plays; a piece's
  samples come back on the worker's pipe, already trimmed, and play from memory (nothing is written to disk before
  the person hears it);
* plays a voice that streams (STREAM_ENGINES: Pocket TTS) from its first chunk, the rest appended as it is made;
* keeps one voice per reply (REPLY_GAP_SECONDS): a reply that began in the system voice ends in it;
* keeps the worker when speech is cut off mid-synthesis (the worker finishes that request unheard);
* after the worker fails to start or crashes, reports it once and lets the caller use the system voice at once,
  retrying the local voice in the background after ``RETRY_AFTER_SECONDS``;
* runs the voice on the CPU when CUDA_VISIBLE_DEVICES hides every card, when the GPU arbiter says the card cannot
  be used, and for the rest of the session after the voice hit a GPU error there (the same voice, said once).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pilot.system import speech_trace
from pilot.timeouts import bounded

logger = logging.getLogger("pilot.system.local_tts_worker")

LOCAL_ENGINES = frozenset({"kokoro_tts", "pocket_tts"})
# Engines whose worker streams a sentence chunk by chunk ("stream" op): playback starts on the first chunk. Pocket's
# first chunk came 0.074 to 0.136 s after the request on one development laptop's processor, against 0.40 to 1.68 s
# for Kokoro there making a whole piece. Kokoro gives its audio once per line of text, so it stays on whole pieces.
STREAM_ENGINES = frozenset({"pocket_tts"})
PREPARED_LIMIT = 6  # sentences synthesised ahead and not yet spoken
DEVICES = ("auto", "cpu", "cuda")
MAX_TEXT_CHARACTERS = 20_000
DEFAULT_KEEP_LOADED_SECONDS = 300.0
MEMORY_RECHECK_SECONDS = 60.0  # an idle voice kept by memory: how often the memory is read again
MAX_KEEP_LOADED_SECONDS = 4 * 3600.0
RETRY_AFTER_SECONDS = 120.0
LOAD_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 120.0
SHUTDOWN_WAIT_SECONDS = 2.0  # shutdown(): the longest it waits for the manager's cancelled tasks to end
LOADING_MESSAGE = "Getting my voice ready…"
# The longest a reply waits for Heliox's own voice to finish loading; past it the system voice speaks the reply. A
# reply waits only for a load expected to end within it (expected_load_left: how long loads of this voice have taken
# on this computer, less how long this one has run), so a reply that comes with no load under way, or early in one,
# goes to the system voice at once while the load goes on behind it. A cold load takes 15 to 25 s (one development
# laptop: 15.4 s reloading on the card, 19.9 s on the processor, 27 s from a cold start), so it is never waited for.
LOAD_GRACE_SECONDS = 0.5
LOAD_WAIT_MARGIN_SECONDS = 0.15  # added to the time a load is expected to take still: loads vary by about this much
# A load of this voice with nothing recorded yet is taken to last this long (the middle of 15 to 25 s): with no record
# of this computer, only a load that has run about that long is waited for.
DEFAULT_LOAD_SECONDS = 20.0
LOAD_RECORDS = 10  # the latest loads kept per voice and device; their median is the expected length
VOICE_HOST_RAM_BYTES = 2 * 1024**3  # Kokoro's worker holds 1.6-1.9 GB once loaded
ON_CPU_MESSAGE = (
    "My voice hit a graphics card error, so it runs on the processor for the rest of this session. "
    "It may take a moment longer to start speaking."
)
# What a model runtime says when the card, its driver or its libraries fail (cuDNN, cuBLAS, a device-side assert,
# out of memory on the card). Matched on the worker's error text.
_GPU_ERROR = re.compile(r"cuda|cudnn|cublas|nvrtc|nccl|device-side|\bgpu\b|hip error|out of memory", re.IGNORECASE)


def cuda_hidden(environ: Any = None) -> bool:
    """CUDA_VISIBLE_DEVICES hides every card: set and empty, or only negative indexes ("-1"). Read from the
    daemon's own environment: on Windows an empty value is dropped from the environment a child process gets, so
    the worker would otherwise see every card."""
    value = (os.environ if environ is None else environ).get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return False
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return not parts or all(part.startswith("-") for part in parts)


def is_gpu_error(reason: str) -> bool:
    return bool(_GPU_ERROR.search(reason or ""))


# ── Worker side ──────────────────────────────────────────────────────


def _engine_module(engine: str) -> Any:
    if engine == "kokoro_tts":
        from pilot.system import kokoro_tts

        return kokoro_tts
    if engine == "pocket_tts":
        from pilot.system import pocket_tts

        return pocket_tts
    raise ValueError(f"Unsupported local TTS engine: {engine}")  # pragma: no cover - argparse enforces the choices


# Threads the voice model computes on. Measured 13 November on the owner's CPU (16 cores, 24 threads), Kokoro on the
# six sentences of a short reply, 1 to 15 words: 4 threads 449 to 1645 ms; torch's default (16) 507 to 1741 ms; all 24
# 605 to 1929 ms; 2 threads 654 to 2627 ms. More than four only adds contention, and leaves the recognisers less.
VOICE_THREADS = 4
_threads_set = False


def _use_voice_threads(threads: int = VOICE_THREADS) -> None:
    """Give the voice model VOICE_THREADS threads (fewer on a smaller machine), once, in the worker."""
    global _threads_set
    if _threads_set:
        return
    _threads_set = True
    try:
        import torch
    except Exception:
        return
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or threads)))


def _synthesize(engine: str, text: str, voice: str, output_file: str, timing_file: str = "") -> None:
    """Make one piece on the worker's own thread. Through asyncio.run, each request started a new event loop and ran
    the model on a new executor thread: 85 to 130 ms more per sentence than the same generation called directly
    (measured 13 November on the owner's CPU, the six sentences of a short reply).

    ``timing_file``: also write when each phoneme is said there, when the engine knows (a face is shown)."""
    _use_voice_threads()
    module = _engine_module(engine)
    timed = getattr(module, "synthesize_with_timing_here", None) if timing_file else None
    if timed is not None:
        timed(text, voice, output_file, timing_file)
    else:
        module.synthesize_to_file_here(text, voice, output_file)


def _make_samples(
    engine: str, text: str, voice: str, keep_lead: float, keep_tail: float | None, timing: bool
) -> tuple[Any, int, dict[str, Any] | None, dict[str, float]]:
    """Make one piece in memory on the worker's own thread, with its quiet lead and tail cut (as trim_silence does to a
    file): float32 mono samples, their rate, the phoneme timing counted from the first sample kept (``timing``, when
    the engine knows it), and the seconds each step took."""
    import numpy as np

    from pilot.system import speech_notices

    _use_voice_threads()
    module = _engine_module(engine)
    started = time.perf_counter()
    audio, rate, found = module.generate_here(text, voice, timing=timing)
    made = time.perf_counter()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    start, stop = silence_edges(audio, int(rate), keep_lead, keep_tail)
    audio = np.ascontiguousarray(audio[start:stop])
    if found is not None:
        found = speech_notices.shifted_timing(found, start, stop)
    return audio, int(rate), found, {"synth_s": made - started, "trim_s": time.perf_counter() - made}


def _request_text(request: dict[str, Any]) -> str:
    """The text a request asks to be spoken: given inline ("text"), or in its file. Opening a just-written file took
    5 to 17 ms on one development laptop (a virus scanner reads it first); the daemon sends both, so a worker of either
    age understands it."""
    text = request.get("text")
    if isinstance(text, str):
        return text
    return Path(request["text_file"]).read_text(encoding="utf-8")


def _send(response: dict[str, Any], payload: bytes = b"") -> None:
    """One answer on stdout: a JSON line, then ``payload`` (the raw bytes its "bytes" field announces)."""
    out = sys.stdout
    out.flush()
    data = (json.dumps(response) + "\n").encode("utf-8") + payload
    buffer = getattr(out, "buffer", None)
    if buffer is None:  # a text-only stream (a test's): no payload can go there
        out.write(data.decode("utf-8", errors="replace"))
        out.flush()
        return
    buffer.write(data)
    buffer.flush()


def _load(engine: str, voice: str) -> None:
    """Load the model and voice, and run one throwaway synthesis so the first real sentence is warm."""
    _use_voice_threads()
    module = _engine_module(engine)
    if hasattr(module, "prepare"):
        module.prepare(voice)
    else:
        asyncio.run(module.warmup(voice))


def _take_gpu_lease(device: str) -> tuple[str, Any]:
    """The daemon let this worker use the GPU: hold Heliox's GPU lease for the worker's life, or use the CPU."""
    from pilot.system.gpu_arbiter import GpuLease

    lease = GpuLease()
    try:
        if lease.try_acquire("voice"):
            return device, lease
    except OSError:
        pass
    return "cpu", None


def _own_stdin(stdin: Any) -> Any:
    """The worker's request pipe, moved off the process's standard input before a thread blocks reading it.

    On Windows a read waiting on a pipe blocks every other call on that pipe, and a DLL loaded meanwhile asks the
    standard input for its type while it starts (loading numpy's core, and so torch, waited for ever while the
    request reader waited for the next line). So the pipe is read through a private copy, and the standard input
    becomes the null device. Elsewhere, and for a stdin that is not a file, it is used as it is."""
    if os.name != "nt":
        return stdin
    try:
        fd = stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return stdin  # not a real pipe (a test's)
    try:
        import ctypes
        import msvcrt

        private = os.dup(fd)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, fd)
        os.close(null)
        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(fd))  # STD_INPUT_HANDLE
        return open(private, encoding=getattr(stdin, "encoding", None) or "utf-8", closefd=True)
    except Exception:
        logger.debug("Could not move the worker's requests off its standard input", exc_info=True)
        return stdin


class _Requests:
    """The worker's stdin, read on a thread of its own so that a stop reaches a stream while it is being made: the
    requests in their order, and each ``{"op": "stop", "id": n}`` sets stream n's stop at once (a stop that comes
    before its stream has started stops it before its first chunk)."""

    def __init__(self, stdin: Any) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._stops: dict[int, threading.Event] = {}
        self._early: set[int] = set()  # stops for streams not started yet
        self._finished = 0  # the highest stream id that has ended: later stops for it are ignored
        threading.Thread(target=self._read, args=(stdin,), name="heliox-voice-requests", daemon=True).start()

    def _read(self, stdin: Any) -> None:
        try:
            for line in stdin:
                try:
                    request = json.loads(line)
                except ValueError:
                    request = None
                if isinstance(request, dict) and request.get("op") == "stop":
                    with contextlib.suppress(TypeError, ValueError):
                        self._stop(int(request.get("id") or 0))
                    continue
                self._queue.put(line)
        finally:
            self._queue.put(None)

    def next(self) -> str | None:
        """The next request, or None when the daemon closed the pipe."""
        return self._queue.get()

    def _stop(self, stream_id: int) -> None:
        with self._lock:
            event = self._stops.get(stream_id)
            if event is not None:
                event.set()
            elif stream_id > self._finished:
                self._early.add(stream_id)

    def stop_event(self, stream_id: int) -> threading.Event:
        with self._lock:
            event = threading.Event()
            if stream_id in self._early:
                self._early.discard(stream_id)
                event.set()
            self._stops[stream_id] = event
            return event

    def done(self, stream_id: int) -> None:
        with self._lock:
            self._stops.pop(stream_id, None)
            self._finished = max(self._finished, stream_id)
            self._early = {early for early in self._early if early > self._finished}


def _stream(engine: str, request: dict[str, Any], requests: _Requests) -> None:
    """Speak one piece as a stream: an "audio" frame (a JSON line, then its float32 samples) for each chunk as the
    engine makes it, then one "end" frame. A stop for this stream ends it before the next chunk is made."""
    import numpy as np

    stream_id = int(request.get("id") or 0)
    stop = requests.stop_event(stream_id)
    started = time.perf_counter()
    chunks = samples = 0
    rate = 0
    first_s: float | None = None
    error = ""
    try:
        text = _request_text(request)
        if not text or len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text must contain between 1 and 20,000 characters")
        module = _engine_module(engine)
        stream_here = getattr(module, "stream_here", None)
        if stream_here is None:
            raise ValueError(f"{engine} does not stream")
        _use_voice_threads()
        rate, made = stream_here(text, str(request.get("voice") or ""), stop)
        for chunk in made:
            data = np.ascontiguousarray(np.asarray(chunk, dtype=np.float32).reshape(-1))
            payload = data.tobytes()
            _send({"frame": "audio", "id": stream_id, "seq": chunks, "rate": int(rate), "bytes": len(payload)}, payload)
            if first_s is None:
                first_s = time.perf_counter() - started
            chunks += 1
            samples += len(data)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        requests.done(stream_id)
    end: dict[str, Any] = {"frame": "end", "id": stream_id, "chunks": chunks, "cut": stop.is_set(),
                           "audio_s": round(samples / rate, 4) if rate else 0.0,
                           "made_s": round(time.perf_counter() - started, 4),
                           "first_s": None if first_s is None else round(first_s, 4)}  # fmt: skip
    if error:
        end["error"] = error
    _send(end)


def _serve(engine: str, device: str = "auto", gpu_lease: bool = False) -> int:
    lease = None
    if device != "cpu" and cuda_hidden():
        device = "cpu"  # every card hidden: on POSIX the worker sees the empty value itself
    if gpu_lease and device != "cpu":
        device, lease = _take_gpu_lease(device)
    if engine == "kokoro_tts":
        from pilot.system import kokoro_tts

        kokoro_tts.configure(device=device)
    requests = _Requests(_own_stdin(sys.stdin))
    while True:
        line = requests.next()
        if line is None:
            break
        payload = b""
        try:
            request = json.loads(line)
            if request.get("op") == "stream":
                _stream(engine, request, requests)
                continue
            if request.get("op") == "load":
                _load(engine, str(request.get("voice") or ""))
                _send({"status": "ok", "op": "load"})
                continue
            started = time.perf_counter()
            text = _request_text(request)
            if not text or len(text) > MAX_TEXT_CHARACTERS:
                raise ValueError("Speech text must contain between 1 and 20,000 characters")
            read = time.perf_counter()
            took = {"read_s": read - started}
            if request.get("reply") == "samples":
                # Speech: the samples go back on this pipe, trimmed, with the phoneme timing in the answer. Nothing is
                # written to disk (the file, its trim and the daemon reading it back cost 25 to 45 ms a piece).
                tail = request.get("keep_tail")
                audio, rate, timing, steps = _make_samples(
                    engine,
                    text,
                    str(request["voice"]),
                    float(request.get("keep_lead", KEEP_LEAD_SECONDS)),
                    None if tail is None else float(tail),
                    bool(request.get("timing") or request.get("timing_file")),
                )
                payload = audio.tobytes()
                took.update(steps, total_s=time.perf_counter() - started)
                response = {"status": "ok", "rate": rate, "samples": len(audio), "format": "f32le",
                            "bytes": len(payload), "timing": timing}  # fmt: skip
            else:
                _synthesize(
                    engine,
                    text,
                    str(request["voice"]),
                    str(request["output_file"]),
                    str(request.get("timing_file") or ""),
                )
                took["total_s"] = time.perf_counter() - started
                took.update(getattr(_engine_module(engine), "last_took", None) or {})
                response = {"status": "ok"}
            response["took"] = {k: round(v, 5) for k, v in took.items()}
        except Exception as exc:
            payload = b""
            response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        _send(response, payload)
    if lease is not None:
        lease.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve bounded Heliox local-TTS synthesis")
    parser.add_argument("--engine", choices=sorted(LOCAL_ENGINES), required=True)
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--gpu-lease", action="store_true", help="hold Heliox's GPU lease while on the GPU")
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--parent-pid", type=int, default=0, help="end when this process (the daemon) is gone")
    args = parser.parse_args(argv)
    if args.parent_pid > 0:
        # A killed daemon closes this worker's stdin, but a worker busy importing torch and Kokoro, loading weights or
        # synthesising reads that only when it is done, holding memory (and the card) meanwhile. The same watch as
        # Gnomon's worker: Windows waits on the parent's process handle; macOS and Linux see the parent id change.
        from pilot.intelligence.gnomon_worker import watch_parent

        watch_parent(args.parent_pid, lambda: os._exit(0))
    return _serve(args.engine, args.device, args.gpu_lease)


# ── Daemon side ──────────────────────────────────────────────────────


class LocalTTSUnavailable(RuntimeError):
    """The local voice cannot speak right now; the failure has already been logged once. ``unspoken``: the part of the
    reply it had not said when it stopped partway; "" means the whole reply."""

    already_logged = True
    unspoken: str = ""


# Which voice spoke a reply, and by which branch of the rule (data for the bench and the speech result; nothing is
# shown for it). Voices: "local" (Heliox's own voice, synthesised), "store" (made audio played from SpeechStore),
# "system" (the system voice). Branches: "ready" (the local voice was loaded), "loaded_within_grace",
# "store" (every piece kept), "waited_no_system_voice", "file" (written to a file), "system_still_loading" (the local
# voice was loading past the grace), "system_local_unavailable" (it cannot load or failed), "system_chosen" (the
# person's engine is the system voice), "system_same_reply" (the reply began in the system voice: it ends in it).
REPLY_VOICES = ("local", "store", "system")
REPLY_BRANCHES = (
    "ready",
    "loaded_within_grace",
    "store",
    "waited_no_system_voice",
    "file",
    "system_still_loading",
    "system_local_unavailable",
    "system_chosen",
    "system_same_reply",
)


@dataclass(frozen=True)
class ReplyVoice:
    sequence: int  # increases with every reply recorded (record_reply_voice)
    voice: str  # one of REPLY_VOICES
    branch: str  # one of REPLY_BRANCHES
    reason: str = ""  # for the system voice: why, in plain words
    at: float = 0.0  # time.monotonic() when the voice was chosen


class LocalVoiceNotReady(LocalTTSUnavailable):
    """The local voice is still loading when a reply is ready to be spoken: this reply is spoken with the system voice,
    and the local voice takes over from the next one. Not a failure: nothing is marked failed.

    ``unspoken``: the part of the reply the local voice has not said, when it stopped partway (its worker moved from
    the graphics card to the processor after a fault and is loading again); "" means the whole reply. ``branch``: why,
    for record_reply_voice."""

    unspoken: str = ""
    branch: str = "system_still_loading"


# One voice per reply. A reply is often several speak() calls (the conversation layer says each sentence as it comes,
# after a hum or a holding phrase), so the voice the first of them used is kept for the rest: the person never hears
# the voice change inside a reply, and Heliox's own voice, once ready, takes over at the next reply. A reply ends when
# the person is heard again (prewarm) or when Heliox has said nothing for REPLY_GAP_SECONDS: inside a reply the gaps
# are the model's (a hum ends about 1.3 s before the answer; the layer stops waiting for a model after about 4.5 s).
REPLY_GAP_SECONDS = 6.0


class _MovedToCpu(LocalTTSUnavailable):
    """The voice hit a GPU error; the worker was closed and the next one starts on the CPU. Retried at once."""


_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No)|(?:\b[A-Za-z]\.)*\b[A-Za-z])\.$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+|\n+")


def split_for_speech(text: str, *, clauses: bool = True) -> list[str]:
    """Sentences to synthesise one at a time, so the first one can play while the rest are made. ``clauses``: the
    first sentence is cut at a pause (clause_pieces), to hide a whole-piece engine's wait; a streamed engine
    (STREAM_ENGINES) starts sounding after its first chunk, so its sentences stay whole."""
    pieces: list[str] = []
    position = 0
    for match in _SENTENCE_BREAK.finditer(text):
        piece = text[position : match.end()].strip()
        if piece and _ABBREVIATION.search(piece.rstrip("\"'”’)]")):
            continue  # "Dr. Smith" is one sentence
        if piece:
            pieces.append(piece)
        position = match.end()
    tail = text[position:].strip()
    if tail:
        pieces.append(tail)
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < 3:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    if merged and clauses:
        merged[:1] = clause_pieces(merged[0])  # the first sound comes from the first few words
    return merged or ([text] if text.strip() else [])


# The first sound comes from the first clause of the first sentence: a head of 2 to FIRST_CLAUSE_MAX words, cut
# where the sentence pauses anyway (after a comma, semicolon, colon or dash; else before a joining word from the
# fourth word on), is synthesised and played while the rest is made. Kokoro on a CPU takes 0.38 to 0.45 s per second
# of speech (measured 12 November on the owner's machine, 15 to 40 % load from other work): a 20-word sentence
# 2.4 s before its first sound, a 5-word head 0.8 s. Each later piece has at most GROWTH times the words of the one
# before, so it is ready before that one has finished playing (at 0.45 s per second of speech, twice the words take
# 0.9 of the time the piece before plays). A sentence with no such pause is synthesised whole, as before.
FIRST_CLAUSE_MAX = 9
SHORT_SENTENCE_WORDS = 11  # nothing to gain: the whole sentence is about as quick as a head
GROWTH = 2
_JOINING = frozenset({"and", "but", "so", "because", "then", "which", "while", "or", "when", "if", "although"})
_DASHES = ("-", "–", "—")


def _pause_after(words: list[str], index: int) -> str:
    """ "mark" when words[index - 1] ends with a pause mark, "join" when words[index] is a joining word, else ""."""
    if words[index - 1].endswith((",", ";", ":", *_DASHES[1:])) or words[index] in _DASHES:
        return "mark"
    return "join" if words[index].casefold() in _JOINING else ""


def _cut(words: list[str], low: int, high: int, *, first: bool) -> int:
    """Where to cut ``words`` into a piece of ``low`` to ``high`` words (leaving at least three), or 0: a pause mark
    before a joining word; the first place for the opening clause (sooner), the last for later pieces (fewer seams)."""
    places = [(i, _pause_after(words, i)) for i in range(low, min(high, len(words) - 3) + 1)]
    marked = [i for i, kind in places if kind == "mark"]
    ranked = marked or [i for i, kind in places if kind == "join" and (first is False or i >= 4)]
    if not ranked:
        return 0
    return ranked[0] if first else ranked[-1]


def clause_pieces(sentence: str) -> list[str]:
    """``sentence`` as the pieces to synthesise one after another: one piece when it is short or has no natural
    pause where a head could end."""
    words = sentence.split()
    if len(words) <= SHORT_SENTENCE_WORDS:
        return [sentence]
    cut = _cut(words, 2, FIRST_CLAUSE_MAX, first=True)
    if not cut:
        return [sentence]
    pieces = [" ".join(words[:cut])]
    rest = words[cut:]
    limit = GROWTH * cut
    while len(rest) > limit + 2:
        # The longest piece that is ready in time; with no pause that early, the first pause after it.
        cut = _cut(rest, 3, limit, first=False) or _cut(rest, limit + 1, len(rest), first=True)
        if not cut:
            break  # no pause left: the rest as one piece
        pieces.append(" ".join(rest[:cut]))
        rest = rest[cut:]
        limit = GROWTH * cut
    pieces.append(" ".join(rest))
    return pieces


def ends_sentence(piece: str) -> bool:
    return piece.rstrip("\"')]”’").endswith((".", "!", "?", "…"))


HEAD_CACHE_ENTRIES = 40  # at most; each a second or two of 24 kHz audio, so about 3 MB in all
HEAD_CACHE_WORDS = 6  # a head longer than this is too particular to say twice
HEAD_CACHE_AFTER = 2  # said this many times: its audio is kept; from the next time it plays at once
HEAD_CACHE_VERSION = 1
# Making fixed sentences ahead rests this many times each piece's synthesis time before the next: about a sixth of four
# threads' time while the machine is idle, and nothing while Heliox is speaking or the person is talking to it.
MAKE_AHEAD_REST_FACTOR = 5.0
# No piece is made ahead until Heliox has been silent and unused this long: a conversation (a request, every reply, a
# cut, a follow-up) keeps it waiting, so a reply never finds the worker busy with a piece made for later. On CI's
# control scenarios a piece started one second after a cut, as the "Okay." for that cut arrived, and the reply waited
# for it (0.45 to 0.95 s later to first sound on f6a1f8e).
MAKE_AHEAD_IDLE_SECONDS = 30.0
# Only pieces this short are made ahead, so one already being made when the first request after a quiet stretch
# arrives ends soon: 8 words took 0.75 s on a 16-core laptop CPU (13 November), and the request's own speech and the
# end of its turn take longer than that. Longer fixed pieces are synthesised when they are said, as before.
MAKE_AHEAD_MAX_WORDS = 8
# The fixed sentences made ahead (pilot.system.fixed_speech) may hold at most this much audio on disk; past it, no more
# are made (none is dropped to make room: each was made once, when the machine was idle). About 220 pieces of 1 to 3 s
# of 24 kHz float audio, about 40 MB in all, fit.
FIXED_SPEECH_BYTES = 64 * 1024 * 1024


def on_battery() -> bool:
    """The computer runs on its battery (psutil's sensors_battery, which reads it on Windows, macOS and Linux). False
    with no battery (a desktop), when it is plugged in, or when the state cannot be read."""
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:
        return False
    return bool(battery is not None and battery.power_plugged is False)


class SpeechStore:
    """Audio Heliox's voice has already made, kept so a piece is played instead of synthesised: the one store of
    ready-made speech.

    Two kinds of entry, one key: the piece's exact text and everything that makes its sound (engine, voice, speed, the
    trim settings, this version). A piece plays from the store only when the piece about to be spoken is exactly that
    text, made the same way: it never adds, drops or changes a word of a reply.
    - The fixed sentences Heliox says word for word (``pilot.system.fixed_speech``), made ahead while the machine is
      idle (``LocalTTSWorkerManager.make_ahead``), bounded by FIXED_SPEECH_BYTES and never evicted.
    - Heads of a reply this Heliox said twice (six words or fewer), copied from audio already made for a reply,
      bounded to HEAD_CACHE_ENTRIES (the least recently used go first).
    Kept beside Heliox's other caches (``~/.cache/heliox/voice_heads``, which removal deletes on every system) and
    emptied of every entry made another way when the voice or its settings change (``keep_only``)."""

    def __init__(self, folder: Path | None = None) -> None:
        if folder is None:
            from pilot.config import CACHE_DIR

            folder = CACHE_DIR / "voice_heads"
        self.folder = folder
        self._index: dict[str, dict[str, Any]] | None = None

    @staticmethod
    def signature(engine: str, voice: str, speed: float = 1.0) -> str:
        return json.dumps([HEAD_CACHE_VERSION, engine, voice, speed, KEEP_LEAD_SECONDS, KEEP_CLAUSE_SECONDS])

    @staticmethod
    def key(text: str, signature: str) -> str:
        import hashlib

        return hashlib.sha256(json.dumps([text, signature]).encode("utf-8")).hexdigest()[:32]

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            try:
                self._index = json.loads((self.folder / "index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._index = {}
        return self._index

    def _save(self) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "index.json").write_text(json.dumps(self._load(), indent=1), encoding="utf-8")
        except OSError:
            logger.debug("Could not save the voice's head cache index", exc_info=True)

    def lookup(self, text: str, signature: str) -> Path | None:
        """The kept audio for exactly ``text`` made this way, or None."""
        entry = self._load().get(self.key(text, signature))
        if not entry or entry.get("text") != text or entry.get("signature") != signature:
            return None
        path = self.folder / f"{self.key(text, signature)}.wav"
        if not path.is_file():
            return None
        entry["last"] = time.time()
        return path

    @staticmethod
    def _audio_bytes(audio: Any) -> int:
        """About how many bytes ``audio`` takes as a kept WAV (16-bit samples), or 0 when there is none."""
        if isinstance(audio, Path):
            return audio.stat().st_size if audio.is_file() else 0
        if getattr(audio, "audio", None) is not None:
            return 2 * len(audio.audio) + 44 if len(audio.audio) else 0  # a piece with no sound is never kept
        path = getattr(audio, "path", None)
        return path.stat().st_size if isinstance(path, Path) and path.is_file() else 0

    @staticmethod
    def _write(audio: Any, target: Path) -> None:
        """Keep ``audio`` (a WAV file, or a Piece with samples in memory) as the WAV ``target``, with the phoneme timing
        that came with it beside it (speech_notices.timing_path, already shifted for the trim), so a kept piece gives
        the face the same mouth timeline as a freshly made one. Audio without timing keeps none (the face then moves by
        its loudness, as before)."""
        import shutil

        from pilot.system import speech_notices

        source = audio if isinstance(audio, Path) else getattr(audio, "path", None)
        samples = None if isinstance(audio, Path) else getattr(audio, "audio", None)
        timing = None if isinstance(audio, Path) else getattr(audio, "timing", None)
        kept_timing = speech_notices.timing_path(target)
        if samples is None:
            shutil.copyfile(source, target)
            if timing is None and source is not None and speech_notices.timing_path(source).is_file():
                shutil.copyfile(speech_notices.timing_path(source), kept_timing)
                return
        else:
            import soundfile

            soundfile.write(str(target), samples, int(audio.rate))
        if timing:
            kept_timing.write_text(json.dumps(timing, ensure_ascii=False), encoding="utf-8")
        else:
            kept_timing.unlink(missing_ok=True)

    def said(self, text: str, signature: str, audio: Any) -> bool:
        """``text`` was just spoken from ``audio`` (made for a reply: a WAV, or a Piece with its samples). Counts it,
        and keeps a copy of the audio once it has been said HEAD_CACHE_AFTER times. True when it was kept now."""
        if len(text.split()) > HEAD_CACHE_WORDS or not self._audio_bytes(audio):
            return False
        index = self._load()
        key = self.key(text, signature)
        entry = index.setdefault(key, {"text": text, "signature": signature, "count": 0, "kept": False})
        entry["count"] += 1
        entry["last"] = time.time()
        kept = False
        if entry["count"] >= HEAD_CACHE_AFTER and not entry["kept"]:
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                self._write(audio, self.folder / f"{key}.wav")
                entry["kept"] = kept = True
            except Exception:
                logger.debug("Could not keep a head's audio", exc_info=True)
        self._prune()
        self._save()
        return kept

    def has(self, text: str, signature: str) -> bool:
        """Whether audio for exactly ``text`` made this way is kept (without counting it as used)."""
        entry = self._load().get(self.key(text, signature))
        return bool(
            entry
            and entry.get("kept")
            and entry.get("text") == text
            and entry.get("signature") == signature
            and (self.folder / f"{self.key(text, signature)}.wav").is_file()
        )

    def fixed_bytes(self) -> int:
        total = 0
        for key, entry in self._load().items():
            if entry.get("fixed"):
                with contextlib.suppress(OSError):
                    total += (self.folder / f"{key}.wav").stat().st_size
        return total

    def add_fixed(self, text: str, signature: str, audio: Any) -> bool:
        """Keep ``audio`` (a WAV, or a Piece with its samples), made ahead for the fixed piece ``text``. False when the
        store is full or it cannot be kept."""
        size = self._audio_bytes(audio)
        if not size or self.fixed_bytes() + size > FIXED_SPEECH_BYTES:
            return False
        key = self.key(text, signature)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._write(audio, self.folder / f"{key}.wav")
        except Exception:
            logger.debug("Could not keep a fixed sentence's audio", exc_info=True)
            return False
        self._load()[key] = {"text": text, "signature": signature, "count": 0, "kept": True, "fixed": True,
                             "last": time.time()}  # fmt: skip
        self._save()
        return True

    def keep_only(self, signature: str) -> None:
        """Drop every entry made another way (the voice or its settings changed): they would never be played."""
        index = self._load()
        for key in [k for k, e in index.items() if e.get("signature") != signature]:
            self._drop(key)
        self._save()

    def _drop(self, key: str) -> None:
        self._load().pop(key, None)
        with contextlib.suppress(OSError):
            (self.folder / f"{key}.wav").unlink()
        with contextlib.suppress(OSError):
            (self.folder / f"{key}.wav.timing.json").unlink()

    def _prune(self) -> None:
        index = self._load()
        kept = sorted((e.get("last", 0.0), k) for k, e in index.items() if e.get("kept") and not e.get("fixed"))
        for _last, key in kept[: max(0, len(kept) - HEAD_CACHE_ENTRIES)]:
            self._drop(key)
        counted = sorted((e.get("last", 0.0), k) for k, e in index.items() if not e.get("kept"))
        for _last, key in counted[: max(0, len(counted) - 4 * HEAD_CACHE_ENTRIES)]:
            self._drop(key)


# Kokoro pads every synthesis with 0.32 to 0.40 s of silence before the first word and 0.45 to 0.55 s after the
# last (measured 12 November). Played as made, the first sound came that much after the audio began, and a clause
# seam had about 0.8 s of silence where one synthesis pauses 0.3 to 0.5 s at a comma. So every piece keeps
# KEEP_LEAD_SECONDS before its first sound, and a piece that ends mid-sentence keeps KEEP_CLAUSE_SECONDS after its
# last (with the output line's hand-over, about 0.1 s, the seam is a comma's pause). A piece that ends a sentence
# keeps its own pause.
KEEP_LEAD_SECONDS = 0.03
KEEP_CLAUSE_SECONDS = 0.25
SILENCE_LEVEL = 0.002


def silence_edges(audio: Any, rate: int, keep_lead: float, keep_tail: float | None) -> tuple[int, int]:
    """The sample range to keep: from ``keep_lead`` before the first sound to ``keep_tail`` after the last (the
    whole end when None). The whole buffer when it holds no sound."""
    import numpy as np

    level = np.abs(np.asarray(audio, dtype=np.float32))
    if level.ndim > 1:
        level = level.max(axis=1)
    loud = np.flatnonzero(level > SILENCE_LEVEL)
    if loud.size == 0:
        return 0, len(level)
    start = max(0, int(loud[0]) - int(keep_lead * rate))
    stop = len(level) if keep_tail is None else min(len(level), int(loud[-1]) + 1 + int(keep_tail * rate))
    return start, stop


def trim_silence(path: Path, keep_lead: float, keep_tail: float | None) -> tuple[int, int] | None:
    """Cut a synthesised WAV's padding in place (see KEEP_LEAD_SECONDS). Without soundfile it is left as made.
    Returns the sample range kept of the audio as made (None when it was not read)."""
    try:
        import soundfile
    except ImportError:
        return None
    info = soundfile.info(str(path))
    audio, rate = soundfile.read(str(path), dtype="float32")
    start, stop = silence_edges(audio, rate, keep_lead, keep_tail)
    if start == 0 and stop == len(audio):
        return start, stop
    soundfile.write(str(path), audio[start:stop], rate, subtype=info.subtype)
    return start, stop


def _default_spawn(engine: str, device: str, gpu_lease: bool = False) -> Any:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [
        sys.executable,
        "-m",
        "pilot.system.local_tts_worker",
        "--device",
        device,
        *(["--gpu-lease"] if gpu_lease else []),
        "--parent-pid",
        str(os.getpid()),
        "--engine",
        engine,
        "--serve",
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _read_exactly(stream: Any, size: int) -> bytes:
    parts: list[bytes] = []
    left = size
    while left > 0:
        part = stream.read(left)
        if not part:
            raise EOFError(f"the worker's answer ended {left} bytes short")
        parts.append(part)
        left -= len(part)
    return b"".join(parts)


_stdin_lock = threading.Lock()  # a stop line (any thread) never lands inside a request line


def _write_line(process: Any, request: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("Local TTS worker streams are unavailable")
    with _stdin_lock:
        process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        process.stdin.flush()


def _exchange(process: Any, request: dict[str, Any]) -> tuple[bytes, bytes]:
    """Send one request; its answer line, and the bytes that follow it when the line announces them ("bytes")."""
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Local TTS worker streams are unavailable")
    _write_line(process, request)
    line = process.stdout.readline()
    size = 0
    if line and b'"bytes"' in line:
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            size = int(json.loads(line).get("bytes") or 0)
    return line, (_read_exactly(process.stdout, size) if size > 0 else b"")


def _close_process(worker: Any) -> None:
    try:
        if worker.poll() is None:
            if worker.stdin is not None:
                with contextlib.suppress(BrokenPipeError, OSError):
                    worker.stdin.close()
            try:
                worker.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    finally:
        if worker.stdin is not None and not worker.stdin.closed:
            with contextlib.suppress(BrokenPipeError, OSError):
                worker.stdin.close()
        if worker.stdout is not None:
            worker.stdout.close()


class _Spawning:
    """One start of the worker process, on a thread of its own. Cancelling the task that waits for it does not stop
    the thread, so a worker that starts after its caller gave up (shutdown() during the start) would be left running
    with nobody to close it. abandon() makes sure it is closed: by the starting thread when the start returns later,
    or here when it has returned already."""

    def __init__(self, spawn: Callable[..., Any], args: tuple[Any, ...]) -> None:
        self._spawn = spawn
        self._args = args
        self._lock = threading.Lock()
        self._abandoned = False
        self._process: Any = None

    def run(self) -> Any:
        process = self._spawn(*self._args)
        with self._lock:
            if not self._abandoned:
                self._process = process
                return process
        _close_process(process)
        return None

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            process, self._process = self._process, None
        if process is not None:
            threading.Thread(target=_close_process, args=(process,), name="heliox-voice-close", daemon=True).start()


StatusCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

# Who goes first on the worker's pipe when several requests wait for it (lower first, in arrival order within one):
# a piece a reply waits to play now (and the voice's load), then the next sentence of a reply made while this one
# plays (prepare), then fixed sentences made ahead for the store (make_ahead). The worker makes one piece at a time
# and cannot drop one it has started, so this decides only who goes next.
PIPE_NOW = 0
PIPE_NEXT = 1
PIPE_AHEAD = 2
_PIPE_KINDS = {PIPE_NOW: "reply", PIPE_NEXT: "next", PIPE_AHEAD: "ahead"}


class _Pipe:
    """The worker's pipe: one request at a time. A request waiting for it is ranked (PIPE_NOW, PIPE_NEXT, PIPE_AHEAD,
    or a call that says the rank when the pipe frees up: a sentence made ahead becomes PIPE_NOW once a reply waits for
    it), and the lowest goes next. A plain lock served whoever asked first: on 24 November the next sentence of a
    reply, asked for 18 ms before the sentence being spoken, held the pipe for 1.15 s while that sentence waited."""

    def __init__(self) -> None:
        self._held = False
        self._order = 0
        self._waiting: list[tuple[Callable[[], int], int, asyncio.Future[None]]] = []

    def locked(self) -> bool:
        return self._held

    async def acquire(self, rank: int | Callable[[], int] = PIPE_NOW) -> None:
        if not self._held and not self._waiting:
            self._held = True
            return
        ranked = rank if callable(rank) else (lambda: rank)
        self._order += 1
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (ranked, self._order, future)
        self._waiting.append(entry)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # it had been given the pipe as it was cancelled: the next one takes it
            elif entry in self._waiting:
                self._waiting.remove(entry)
            raise

    def release(self) -> None:
        while self._waiting:
            entry = min(self._waiting, key=lambda item: (_rank_of(item[0]), item[1]))
            self._waiting.remove(entry)
            if not entry[2].done():
                entry[2].set_result(None)  # the pipe stays held: it is that request's now
                return
        self._held = False


def _rank_of(rank: Callable[[], int]) -> int:
    try:
        return int(rank())
    except Exception:
        return PIPE_NOW


@dataclass
class _Next:
    """A sentence prepare() was asked to make while the reply's current sentence is spoken: made once the speech that
    follows the request has all its own pieces (``after``: how many speeches had started when it was asked for), or
    at once when a reply is waiting for it (``wanted``)."""

    after: int
    wanted: bool = False


@dataclass
class Piece:
    """One piece of speech ready to play: its samples in memory (made for this reply, already trimmed), or a WAV on
    disk (``path``: a piece from the store, or one a worker wrote because it could not send samples)."""

    text: str
    audio: Any = None  # float32 mono samples; None: read them from ``path``
    rate: int = 0
    timing: dict[str, Any] | None = None  # when each phoneme is said, counted from the first sample (a face is shown)
    path: Path | None = None
    trimmed: bool = True  # False: a file as the worker wrote it, its quiet lead and tail not cut yet
    source: str = "made"  # "made", "ahead" or "store"


def _piece_from_answer(text: str, answer: dict[str, Any]) -> Piece | None:
    """The samples an answer carried, or None when it carried none (a worker that wrote the file instead)."""
    payload = answer.get("_payload") or b""
    if not payload or answer.get("format", "f32le") != "f32le":
        return None
    import numpy as np

    audio = np.frombuffer(payload, dtype="<f4")
    timing = answer.get("timing")
    return Piece(text, audio=audio, rate=int(answer.get("rate") or 24000), timing=timing if timing else None)


class StreamedPiece:
    """One sentence a streaming worker is making (STREAM_ENGINES): its chunks as they arrive, until its end frame.

    The manager's reader thread adds the chunks; the player takes them as they come (``take``) and plays from the
    first; ``stop()`` asks the worker to stop making it (a cut), which it does before its next chunk. Once it has
    ended, ``audio`` is the whole piece (for the store) and ``error`` says why it ended early ("" when it did not)."""

    def __init__(self, text: str, stream_id: int = 0, stopper: Callable[[int], None] | None = None) -> None:
        self.text = text
        self.id = stream_id
        self.rate = 0
        self.source = "made"
        self.path: Path | None = None
        self.timing: dict[str, Any] | None = None
        self.received = 0  # samples arrived so far
        self.complete = False  # the end frame arrived (or the stream broke off)
        self.cut = False  # the worker stopped it on a stop
        self.error = ""
        self.end: dict[str, Any] = {}
        self._chunks: list[Any] = []
        self._cond = threading.Condition()
        self._stopper = stopper
        self._stop_sent = False
        self._joined: Any = None

    def add(self, chunk: Any, rate: int) -> None:
        with self._cond:
            self._chunks.append(chunk)
            self.received += len(chunk)
            self.rate = int(rate) or self.rate
            self._cond.notify_all()

    def finish(self, end: dict[str, Any]) -> None:
        with self._cond:
            if self.complete:
                return
            self.end = dict(end)
            self.cut = bool(end.get("cut"))
            self.error = str(end.get("error") or "")
            self.complete = True
            self._cond.notify_all()

    def take(self, start: int, timeout: float) -> tuple[list[Any], bool]:
        """The chunks from index ``start`` on (waiting at most ``timeout`` s for one), and whether the stream ended."""
        with self._cond:
            if len(self._chunks) <= start and not self.complete and timeout > 0:
                self._cond.wait(timeout)
            return self._chunks[start:], self.complete

    def stop(self) -> None:
        """Ask the worker to stop making this piece (at most once, never after its end)."""
        with self._cond:
            if self.complete or self._stop_sent or self._stopper is None:
                return
            self._stop_sent = True
        try:
            self._stopper(self.id)
        except Exception:
            logger.debug("Could not ask the voice worker to stop a stream", exc_info=True)

    @property
    def audio(self) -> Any:
        """Every sample arrived so far, as one float32 array."""
        import numpy as np

        with self._cond:
            chunks = list(self._chunks)
        if self._joined is None or len(self._joined) != self.received:
            self._joined = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return self._joined


def _stream_exchange(
    process: Any,
    request: dict[str, Any],
    piece: StreamedPiece,
    on_first: Callable[[], None],
    on_end: Callable[[], None],
) -> None:
    """On a thread of its own: send a stream request and read its frames into ``piece`` until the end frame (or the
    worker going away). ``on_first`` when the first chunk (or the end) has arrived, ``on_end`` after the end."""
    import numpy as np

    first = False
    try:
        _write_line(process, request)
        while True:
            line = process.stdout.readline()
            if not line:
                piece.finish({"error": "the worker exited"})
                return
            head = json.loads(line)
            if head.get("frame") == "audio":
                data = _read_exactly(process.stdout, int(head.get("bytes") or 0))
                piece.add(np.frombuffer(data, dtype="<f4"), int(head.get("rate") or 0))
                if not first:
                    first = True
                    speech_trace.emit("stream first chunk", id=piece.id, at=time.monotonic())
                    on_first()
            elif head.get("frame") == "end":
                piece.finish(head)
                speech_trace.emit("stream end", at=time.monotonic(), **{k: v for k, v in head.items() if k != "frame"})
                return
            else:
                piece.finish({"error": str(head.get("error") or "an answer the stream did not expect")})
                return
    except Exception as exc:
        piece.finish({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if not first:
            on_first()
        on_end()


PlayFile = Callable[[Any], Awaitable[None]]  # takes a Piece, a StreamedPiece (or a Path), returns once it has played

# What of a sentence the person heard when Heliox was cut off. A word counts only if it had finished playing this
# long before the voice stopped: from the person's first sound to silence takes 0.45 to 0.65 s (the barge-in
# decision, then an 8 ms fade and the device's buffered block), and what Heliox believes it said must never be more
# than the person heard, so the longer end is taken.
HEARD_MARGIN_SECONDS = 0.65
# The slowest pace a voice says a sentence at, in seconds per letter (spaces counted): English speech runs at about
# 14 to 16 letters a second, so 10 a second is longer than any sentence takes. Used for a streamed piece whose length
# is not known yet (SpeechProgress.heard_words).
SLOW_SECONDS_PER_LETTER = 0.1


def playback_samples(playback: Any) -> int:
    """How many samples a playback has to play so far: a streamed one's ``received``, a whole one's ``audio``."""
    received = getattr(playback, "received", None)
    return int(received) if received is not None else len(playback.audio)


@dataclass
class SpeechProgress:
    """One speak() of the local voice: its pieces, how long each finished one played, and the one playing now
    (its playback object, set by the player through ``speech_progress``)."""

    text: str
    pieces: list[str]
    sequence: int = 0
    durations: list[float] = field(default_factory=list)  # seconds of each piece that played to the end
    playing: Any = None  # the playback of the piece after them: .position (samples), .sample_rate, .audio
    finished: bool = False

    def played(self) -> tuple[float, float]:
        """(seconds played, seconds of audio made so far) across the pieces."""
        played = total = sum(self.durations)
        current = self.playing
        if current is not None and len(self.durations) < len(self.pieces):
            rate = max(1, int(getattr(current, "sample_rate", 1)))
            made = playback_samples(current)
            played += min(int(getattr(current, "position", 0)), made) / rate
            total += made / rate
        return played, total

    def heard_words(self, margin: float = HEARD_MARGIN_SECONDS) -> int:
        """How many of the sentence's words had finished playing ``margin`` seconds before the cut. Within a piece a
        word's end is placed in proportion to the letters before it (the voice's pace is about even inside a
        clause); a piece that played to the end counts whole."""
        if self.finished:
            return sum(len(piece.split()) for piece in self.pieces)
        played, _total = self.played()
        left = played - margin
        words = 0
        lengths = [*self.durations]
        current = self.playing
        if current is not None and len(lengths) < len(self.pieces):
            seconds = playback_samples(current) / max(1, int(getattr(current, "sample_rate", 1)))
            if getattr(current, "complete", True) is False:
                # A streamed piece still arriving: its length is not known yet. Placing its words over the audio
                # received so far would put them too early (more heard than was); placing them over the longest the
                # sentence can take puts them late, so what Heliox believes it said is never more than was heard.
                letters = sum(len(word) + 1 for word in self.pieces[len(lengths)].split())
                seconds = max(seconds, letters * SLOW_SECONDS_PER_LETTER)
            lengths.append(seconds)
        for piece, seconds in zip(self.pieces, lengths, strict=False):
            piece_words = piece.split()
            if left >= seconds:
                words += len(piece_words)
                left -= seconds
                continue
            weights = [len(word) + 1 for word in piece_words]
            total = sum(weights) or 1
            reached = 0.0
            for weight in weights:
                reached += seconds * weight / total
                if reached > left:
                    break
                words += 1
            break
        return max(0, words)


# The speak() in progress, for the player to report its playback (set by LocalTTSWorkerManager.speak).
speech_progress: ContextVar[SpeechProgress | None] = ContextVar("speech_progress", default=None)


class LocalTTSWorkerManager:
    """One local-TTS worker process: when it starts, how long it stays, and how requests reach it."""

    def __init__(
        self,
        *,
        spawn: Callable[[str, str], Any] = _default_spawn,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        keep_loaded_seconds: float = DEFAULT_KEEP_LOADED_SECONDS,
        device: str = "auto",
    ) -> None:
        self._spawn = spawn
        self._clock = clock
        self._sleep = sleep
        self.keep_loaded_seconds = keep_loaded_seconds
        self.device = device
        self.on_status: StatusCallback | None = None
        self.process: Any = None
        self.engine: str | None = None
        self.process_device: str | None = None
        # Where the running worker was actually started: the configured device, or the CPU when the GPU arbiter
        # reported the card hot, busy or switched off (``choose_device``).
        self.effective_device: str | None = None
        self.choose_device: Callable[[str], tuple[str, bool]] | None = None
        self.on_gpu_fault: Callable[[str, str], Any] | None = None
        # Where CUDA_VISIBLE_DEVICES is read (cuda_hidden); None: the daemon's own environment. Tests give a made-up
        # machine its own environment here and keep every card hidden from the test process itself.
        self.environ: Any = None
        # Why the voice runs on the CPU for the rest of the session ("" while the GPU may still be used).
        self.cpu_for_session = ""
        # Where the running worker's voice runs and why, in plain words, for the model health status.
        self.device_reason = ""
        # The heads this Heliox says most, kept as audio (SpeechStore); None: nothing kept (tests, a written file).
        self.head_cache: SpeechStore | None = None
        # The last speak()'s pieces and playback, for what was heard of it when it is cut off (SpeechProgress).
        self.last_progress: SpeechProgress | None = None
        self.last_reply: ReplyVoice | None = None  # which voice spoke the last reply (record_reply_voice)
        self._spoken_sequence = 0
        self.gpu_lease = False  # the running worker holds Heliox's GPU lease (the arbiter said the card is usable)
        # How many times another component has found the card held (GpuArbiter.wants), and the count when this
        # worker took the lease: a voice on the card is kept while idle only until the count grows (_kept_by_memory).
        self.lease_wants: Callable[[], int] | None = None
        self._wants_at_lease = 0
        # Free system memory now, or None: the worker is not started when less than VOICE_HOST_RAM_BYTES is free
        # (on an 8 GB machine the model's 1.6-1.9 GB can push everything into swap); the system voice speaks.
        self.free_memory: Callable[[], int | None] | None = None
        # The small-machine rule (pilot.system.speech_budget): a sentence when this computer should speak with the
        # system voice by itself, "" otherwise. Checked before every start.
        self.light_reason: Callable[[], str] | None = None
        # Why the local voice is not in use now ("" while it is, or before it was ever tried): the reason in the
        # manager's words, for ready.speaks and the voice bar (speech_budget.unavailable_sentence).
        self.unavailable_reason = ""
        self.loaded = False
        self._spawn_lock = asyncio.Lock()
        self._io_lock = _Pipe()
        self._load_task: asyncio.Future[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._busy = 0
        # The voice's loads and releases with their time.monotonic(): ("load_started" | "load_ended" | "load_failed" |
        # "released", detail, time). Data for the bench; nothing reads it to decide.
        self.reload_log: deque[tuple[str, str, float]] = deque(maxlen=50)
        # Every synthesis request with time.monotonic() stamps: (kind, words, requested, started, done). kind is
        # "reply" (a piece a reply waits for), "next" (the next sentence, made while one plays: prepare) or "ahead"
        # (made ahead for the store); started is when this request got the worker's pipe (after any request before
        # it). Data for the bench: it shows a reply waiting behind a piece made ahead.
        self.synth_log: deque[tuple[str, int, float, float, float]] = deque(maxlen=200)
        self._request_started_at = 0.0  # the last request's start on the pipe (stream() reads its own)
        # Speeches started (speak(), not a file) and the latest one whose pieces are all made (or that ended): a
        # sentence prepared ahead waits for the speech after it to be made first (_Next).
        self._speeches = 0
        self._speeches_made = 0
        self._turn_waiters: list[asyncio.Future[None]] = []
        self._next_state: dict[tuple[str, str, str], _Next] = {}
        self._stream_ids = 0  # numbers the streams sent to the worker (their stop names them)
        # Whether an idle voice may stay loaded past keep_loaded_seconds (voice.tts_release = "memory": the computer
        # has the memory to spare); None: released after keep_loaded_seconds, by the clock alone.
        self.keep_while: Callable[[], bool] | None = None
        self._announcing: asyncio.Future[None] | None = None  # a load started for a reply (_ready_within_grace)
        # The voice of the reply being spoken ("local" or "system"; "" before it is decided) and when Heliox last
        # finished speaking in any voice (note_spoken): see REPLY_GAP_SECONDS.
        self._reply_voice = ""
        self._spoke_until = clock()
        # How long each completed load took, [engine, "card" | "cpu", seconds], oldest first (loads.json in the
        # store's folder when there is a store, so a restart still knows this computer); and when the load under way
        # started (the manager's clock).
        self.load_times: list[list[Any]] = []
        self._load_times_read = False
        self._loading_since = clock()
        self._last_activity = clock()
        self._failed_until: float | None = None
        self._background: set[asyncio.Future[Any]] = set()
        # Sentences synthesised ahead of their turn (prepare()), keyed by (engine, voice, sentence).
        self._prepared: dict[tuple[str, str, str], asyncio.Task[Piece]] = {}
        self._prepared_dir: Path | None = None

    # Settings and activity -------------------------------------------------

    def configure(self, *, keep_loaded_seconds: float | None = None, device: str | None = None) -> None:
        if keep_loaded_seconds is not None:
            try:
                seconds = float(keep_loaded_seconds)
            except (TypeError, ValueError):
                seconds = DEFAULT_KEEP_LOADED_SECONDS
            if seconds != seconds or seconds < 0:  # NaN or negative
                seconds = DEFAULT_KEEP_LOADED_SECONDS
            self.keep_loaded_seconds = min(seconds, MAX_KEEP_LOADED_SECONDS)
        if device is not None:
            self.device = device if device in DEVICES else "auto"
        if self._idle_task is not None and not self._idle_task.done():
            # Re-arm with the new length rather than finishing a sleep sized for the old one.
            self._idle_task.cancel()
            self._idle_task = None
            self._start_idle_watch()

    def note_activity(self) -> None:
        """The person spoke to Heliox, or Heliox spoke: a conversation is plausible, keep the voice loaded."""
        self._last_activity = self._clock()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def failing(self) -> bool:
        return self._failed_until is not None

    def begin_reply(self) -> None:
        """A new reply starts (the person was heard): its voice is decided afresh when it is first spoken."""
        self._reply_voice = ""

    def note_spoken(self) -> None:
        """Heliox has just finished saying something, in any voice (the system voice's speech is timed by the caller)."""
        self._spoke_until = self._clock()

    def reply_voice(self) -> str:
        """The voice the reply being spoken keeps ("local", "system", or "" when a new reply decides afresh)."""
        if self._reply_voice and self._busy <= 1 and self._clock() - self._spoke_until > REPLY_GAP_SECONDS:
            self._reply_voice = ""  # Heliox has been quiet long enough: the next speech is a new reply
        return self._reply_voice

    def speaks_locally_now(self, engine: str) -> bool:
        """Whether speech started now would be in Heliox's own voice without waiting: loaded, not failed, and not in a
        reply that began in the system voice."""
        loaded = self.loaded and self.running and self.engine == engine and self._failed_until is None
        return loaded and self.reply_voice() != "system"

    def prewarm(self, engine: str, voice: str) -> None:
        """Start the worker now, without waiting, so it is ready by the time there is something to say. The person is
        talking to Heliox: what is said next is a new reply (begin_reply)."""
        if engine not in LOCAL_ENGINES:
            return
        self.begin_reply()
        self.note_activity()
        if self._failed_until is not None and self._clock() < self._failed_until:
            return
        if self.running and self.engine == engine:
            return
        self._start_prewarm(engine, voice)

    def _start_prewarm(self, engine: str, voice: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._prewarm(engine, voice))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _prewarm(self, engine: str, voice: str) -> None:
        with contextlib.suppress(LocalTTSUnavailable, asyncio.CancelledError):
            await self.ensure_ready(engine, voice)

    # Worker process --------------------------------------------------------

    async def ensure_ready(self, engine: str, voice: str) -> None:
        """Start the worker and load the model; returns when it can synthesise."""
        await self._ensure_process(engine, voice, preload=True)
        if self._load_task is not None:
            await asyncio.shield(self._load_task)

    async def _ensure_process(self, engine: str, voice: str, *, preload: bool) -> Any:
        async with self._spawn_lock:
            if self.running and self.engine == engine:
                return self.process
            if self.process is not None:
                await self._close(self.process)
            blocked = self._start_blocked()
            if blocked:
                self._mark_failed(engine, blocked)
                raise LocalTTSUnavailable(f"{engine}: {blocked}")
            device, gpu_lease = self.device, False
            if engine == "pocket_tts":
                device = "cpu"  # Pocket runs on the processor: it never holds the card's lease another part may need
            elif device != "cpu" and (self.cpu_for_session or cuda_hidden(self.environ)):
                device = "cpu"  # a GPU error earlier this session, or every card hidden
            elif self.choose_device is not None:
                try:
                    device, gpu_lease = await asyncio.to_thread(self.choose_device, self.device)
                except Exception:
                    logger.debug("GPU check for the local voice failed; using the configured device", exc_info=True)
            if gpu_lease and engine == "kokoro_tts":
                from pilot.system import gpu_paths

                if gpu_paths.torch_build() == "cpu":
                    # A card the arbiter would give the voice, and a PyTorch that cannot use it: not installed.
                    gpu_paths.note_not_installed("voice", "a CUDA build of PyTorch")
                    device, gpu_lease = "cpu", False
            self.device_reason = (
                "On the processor: this voice is made to run there." if engine == "pocket_tts" else
                self._describe(device, gpu_lease)
            )  # fmt: skip
            if preload:
                self._loading_since = self._clock()
            spawning = _Spawning(self._spawn, (engine, device, True) if gpu_lease else (engine, device))
            try:
                process = await asyncio.to_thread(spawning.run)
            except asyncio.CancelledError:
                spawning.abandon()  # shutdown() while the worker starts: it is closed as soon as it has started
                raise
            except Exception as exc:
                self._mark_failed(engine, f"could not start: {exc}")
                raise LocalTTSUnavailable(f"{engine} worker could not start: {exc}") from exc
            self.process = process
            self.engine = engine
            self.process_device = self.device
            self.effective_device = device
            self.gpu_lease = gpu_lease
            self._wants_at_lease = self._wants_now()
            self.loaded = False
            self._load_task = None
            self.note_activity()
            self._start_idle_watch()
            if preload:
                self.reload_log.append(("load_started", engine, time.monotonic()))
                self._load_task = asyncio.ensure_future(self._load(process, engine, voice))
                self._background.add(self._load_task)
                self._load_task.add_done_callback(self._background.discard)
            return process

    def _start_blocked(self) -> str:
        """Why the worker may not start now ("" when it may): the small-machine rule, or too little free memory."""
        light = self.light_reason() if self.light_reason is not None else ""
        if light:
            return light
        free = self.free_memory() if self.free_memory is not None else None
        if free is not None and free < VOICE_HOST_RAM_BYTES:
            return (
                f"not enough free memory for the local voice ({free / 1024**3:.1f} GB free, "
                f"{VOICE_HOST_RAM_BYTES / 1024**3:.0f} GB needed)"
            )
        return ""

    async def _load(self, process: Any, engine: str, voice: str) -> None:
        started = self._clock()
        try:
            try:
                await self._request(process, engine, {"op": "load", "voice": voice}, timeout=LOAD_TIMEOUT_SECONDS)
            except _MovedToCpu:
                process = await self._ensure_process(engine, voice, preload=False)
                await self._request(process, engine, {"op": "load", "voice": voice}, timeout=LOAD_TIMEOUT_SECONDS)
        except BaseException as error:
            self.reload_log.append(("load_failed", type(error).__name__, time.monotonic()))
            raise
        self.reload_log.append(("load_ended", engine, time.monotonic()))
        self._note_load_time(engine, self._clock() - self._loading_since)
        logger.info("Local voice %s loaded in %.1f s", engine, self._clock() - started)

    def _start_idle_watch(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_task = loop.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        while self.process is not None:
            remaining = self._last_activity + self.keep_loaded_seconds - self._clock()
            if remaining <= 0 and self.keep_while is not None and self._kept_by_memory():
                await self._sleep(MEMORY_RECHECK_SECONDS)  # idle, and the computer has the memory: stays loaded
                continue
            if remaining <= 0 and self._busy == 0 and not self._io_lock.locked():
                logger.info(
                    "Released the local voice after %.0f s without conversation; it reloads when the person speaks",
                    self.keep_loaded_seconds,
                )
                self._idle_task = None
                await self._release_now("idle")
                return
            await self._sleep(max(remaining, 1.0))

    async def _release_now(self, reason: str) -> None:
        """What every release runs: the worker closed, and the release logged (reload_log)."""
        process = self.process
        if process is None:
            return
        await self._close(process)
        self.reload_log.append(("released", reason, time.monotonic()))

    async def release(self, reason: str = "asked", *, force: bool = False) -> bool:
        """Release the local voice now, as the idle release does. Never while a reply is being spoken or made.
        Without ``force``, only when the idle release would (idle for keep_loaded_seconds, and not kept by memory);
        with it, whatever the keep rule says. True when it was released."""
        if self.process is None or self._busy or self._io_lock.locked():
            return False
        if not force:
            idle = self._clock() - self._last_activity >= self.keep_loaded_seconds
            if not idle or (self.keep_while is not None and self._kept_by_memory()):
                return False
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None
        logger.info("Released the local voice (%s)", reason)
        await self._release_now(reason)
        return True

    def _wants_now(self) -> int:
        try:
            return int(self.lease_wants()) if self.lease_wants is not None else 0
        except Exception:
            return 0

    def _kept_by_memory(self) -> bool:
        """Whether the idle voice stays loaded past keep_loaded_seconds: while the computer has the memory
        (keep_while), on the CPU and on the card alike. On the card, also only until another component (Gnomon,
        Whisper, screen text reading) has found the card held by the voice since it took the lease: the lease is one
        process at a time and is not taken from its holder, so the idle voice gives it up then.

        Measured on one development laptop (19 November): released by the clock after 300 s, the next reply waited for
        a 15 s reload and was spoken by the system voice meanwhile. The worker process itself (the launcher's child)
        held 1.86 to 1.90 GB resident on the card, 0.8 GB of graphics memory besides, against 1.54 to 1.57 GB on the
        CPU: about 0.3 GB more of the computer's memory. So the floor that releases an idle voice on the CPU
        (speech_budget.RELEASE_BELOW_FREE_BYTES, 1 GB free) is the floor here too: below it, releasing gives other
        programs about 1.9 GB back."""
        if self.gpu_lease and self._wants_now() > self._wants_at_lease:
            logger.info("Another component asked for the graphics card; the idle voice gives it up")
            return False
        try:
            return bool(self.keep_while())  # type: ignore[misc]
        except Exception:
            logger.debug("Could not read the memory for keeping the voice; releasing by the clock", exc_info=True)
            return False

    async def _close(self, process: Any) -> None:
        if process is self.process:
            self.process = None
            self.engine = None
            self.loaded = False
            self._load_task = None
        await asyncio.to_thread(_close_process, process)

    async def shutdown(self) -> None:
        """Stop the worker and every task the manager started: a load under way, the announcement waiting for it,
        a pipe exchange, a prewarm, sentences made ahead and the idle watch. They are cancelled and waited for
        (at most SHUTDOWN_WAIT_SECONDS) before the worker is closed, so a load the closing cuts short is not taken
        for a failure of the voice or of the graphics card, and none is left pending when the event loop closes."""
        current = asyncio.current_task()
        idle = self._idle_task
        self._idle_task = None
        started = [*self._background, *self._prepared.values()]
        if idle is not None:
            started.append(idle)
        self._prepared.clear()
        left = [task for task in started if task is not current and not task.done()]
        for task in left:
            task.cancel()
        if left:
            await asyncio.wait(left, timeout=SHUTDOWN_WAIT_SECONDS)
            for task in left:
                if task.done() and not task.cancelled():
                    task.exception()  # retrieved: an error of a cancelled load is not reported again at exit
        if self.process is not None:
            await self._close(self.process)

    def _mark_failed(self, engine: str, reason: str) -> None:
        self.unavailable_reason = reason
        if self._failed_until is None:
            logger.warning(
                "Local voice %s is unavailable (%s); speaking with the system voice until it recovers",
                engine,
                reason[-300:],
            )
        self._failed_until = self._clock() + RETRY_AFTER_SECONDS

    def _describe(self, device: str, gpu_lease: bool) -> str:
        from pilot.system import gpu_paths

        missing = gpu_paths.not_installed("voice")
        if device != "cpu":
            return "On the graphics card." if gpu_lease else ""
        if self.device == "cpu":
            return "On the processor, as chosen in Settings."
        if missing:
            return gpu_paths.on_processor_because_missing(missing)
        if self.cpu_for_session:
            return "On the processor for this session: the graphics card failed for the voice."
        if cuda_hidden(self.environ):
            return "On the processor: CUDA_VISIBLE_DEVICES hides every graphics card."
        return "On the processor: the graphics card was hot, busy or switched off when it started."

    async def _fail(self, process: Any, engine: str, reason: str) -> LocalTTSUnavailable:
        on_gpu = process is self.process and self.effective_device not in {None, "cpu"}
        from pilot.system import gpu_paths

        missing = gpu_paths.missing_library(reason) if on_gpu else ""
        if missing:
            # The worker on the card found a library or runtime not installed: the same voice goes on on the
            # processor for the session, and the card is not blamed (no fault for the arbiter, no message).
            gpu_paths.note_not_installed("voice", missing)
            self.cpu_for_session = reason[-300:]
            await self._close(process)
            return _MovedToCpu(f"{engine} moved to the CPU: its GPU path is not installed ({missing})")
        if on_gpu and self.on_gpu_fault is not None:
            # A voice worker on the GPU that died or hung counts toward switching the GPU off for the session.
            with contextlib.suppress(Exception):
                self.on_gpu_fault("voice", reason)
        # On the card for certain (the arbiter leased it, or "cuda" was asked for), a crash or hang is the card's;
        # on "auto" without a lease only an error that names the card says the worker was on it.
        on_card = self.effective_device == "cuda" or self.gpu_lease
        died = reason == "the worker exited" or reason.startswith("no answer")
        if on_gpu and (is_gpu_error(reason) or (on_card and died)):
            # The card failed, not the voice: the same voice goes on on the CPU, for the rest of the session.
            first = not self.cpu_for_session
            self.cpu_for_session = reason[-300:]
            if first:
                logger.warning("Local voice %s hit a GPU error (%s); it runs on the CPU for this session", engine,
                               reason[-300:])  # fmt: skip
                await self._notify("voice_on_cpu", {"message": ON_CPU_MESSAGE})
            await self._close(process)
            return _MovedToCpu(f"{engine} moved to the CPU after a GPU error: {reason}")
        self._mark_failed(engine, reason)
        if process is self.process or process.poll() is None:
            await self._close(process)
        return LocalTTSUnavailable(f"{engine} worker failed: {reason}")

    async def _request(
        self,
        process: Any,
        engine: str,
        request: dict[str, Any],
        *,
        timeout: float,
        rank: int | Callable[[], int] = PIPE_NOW,
    ) -> dict[str, Any]:
        """One request/response on the pipe: the worker's answer, with the bytes that came after it (the samples) as
        "_payload", and when this request got the pipe as "_started" (time.monotonic()). ``rank``: who goes first
        when several wait for the pipe (_Pipe). Cancelling the caller lets the worker finish it unheard."""
        await self._io_lock.acquire(rank)
        started = self._request_started_at = time.monotonic()
        exchange = asyncio.ensure_future(asyncio.to_thread(_exchange, process, request))
        exchange.add_done_callback(lambda _done: self._io_lock.release())
        self._background.add(exchange)
        exchange.add_done_callback(self._background.discard)
        try:
            line, payload = await bounded(asyncio.shield(exchange), timeout)
        except TimeoutError:
            raise await self._fail(process, engine, f"no answer within {timeout:.0f} s") from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise await self._fail(process, engine, f"{type(exc).__name__}: {exc}") from exc
        if not line:
            raise await self._fail(process, engine, "the worker exited")
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            raise await self._fail(process, engine, "invalid response") from None
        if response.get("status") != "ok":
            raise await self._fail(process, engine, str(response.get("error") or "unknown synthesis failure"))
        if not self.loaded or self._failed_until is not None:
            if self._failed_until is not None:
                logger.info("Local voice %s recovered", engine)
            self._failed_until = None
            self.unavailable_reason = ""
            self.loaded = True
        response["_payload"] = payload
        response["_started"] = started
        return response

    # Speaking ----------------------------------------------------------------

    async def synthesize(
        self,
        engine: str,
        text: str,
        voice: str,
        output_file: Path,
        request_dir: Path,
        *,
        wait_after_gpu_fault: bool = True,
        samples: bool = False,
        keep_tail: float | None = None,
        rank: int | Callable[[], int] = PIPE_NOW,
        kind: str = "reply",
    ) -> Piece | None:
        """Make ``text`` into ``output_file``, or with ``samples`` into memory: the worker sends the samples back on
        its pipe, trimmed (KEEP_LEAD_SECONDS before the first sound, ``keep_tail`` after the last, the whole end when
        None), and the Piece holds them; ``output_file`` is then written only by a worker that cannot send samples
        (the Piece names it, untrimmed). After a GPU fault the voice moves to the CPU: with ``wait_after_gpu_fault``
        (a file someone asked for) this piece waits for the CPU worker; without it (speech) the CPU worker loads in
        the background and LocalVoiceNotReady says so at once, so no sentence waits the 5 to 12 s of a load (the
        system voice speaks it meanwhile)."""
        if engine not in LOCAL_ENGINES:
            raise ValueError(f"Unsupported local TTS engine: {engine}")
        if not text or len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text must contain between 1 and 20,000 characters")
        requested = time.monotonic()
        process = await self._usable_process(engine, voice)
        process_ready = time.monotonic()
        request_path = request_dir / f"{output_file.stem}-{id(output_file):x}.txt"
        request_path.write_text(text, encoding="utf-8")  # 0.4 ms (measured): for a worker that reads the file
        text_written = time.monotonic()
        request: dict[str, Any] = {
            "text_file": str(request_path), "text": text, "voice": voice, "output_file": str(output_file)
        }  # fmt: skip
        if samples:
            # The phoneme timing always comes with samples (a few hundred bytes, made with the audio): a piece kept in
            # the store then has it for the face later, whether or not a face is shown now.
            request.update(reply="samples", keep_lead=KEEP_LEAD_SECONDS, keep_tail=keep_tail, timing=True)
        from pilot.system import speech_notices

        if speech_notices.face_on():
            # A face is shown: the worker also gives when each phoneme is said (in its answer, or beside the file).
            request["timing_file"] = str(speech_notices.timing_path(output_file))
        try:
            answer = await self._request(
                process, engine, request, timeout=LOAD_TIMEOUT_SECONDS if not self.loaded else REQUEST_TIMEOUT_SECONDS,
                rank=rank,
            ) or {}  # fmt: skip
        except _MovedToCpu:
            if not wait_after_gpu_fault:
                self._start_prewarm(engine, voice)  # the same voice on the CPU, for the next sentence
                logger.info(
                    "Local voice moved to the CPU after a GPU fault; this sentence is spoken by the system voice"
                )
                raise LocalVoiceNotReady(f"{engine}: loading on the CPU after a GPU fault") from None
            # The same sentence, once, from the same voice started on the CPU.
            process = await self._usable_process(engine, voice)
            answer = await self._request(process, engine, request, timeout=LOAD_TIMEOUT_SECONDS, rank=rank) or {}
        piece = _piece_from_answer(text, answer) if samples else None
        if piece is None and (not output_file.is_file() or output_file.stat().st_size == 0):
            raise await self._fail(process, engine, "the worker produced no audio")
        done = time.monotonic()
        # This request's own start on the pipe: the manager's last start may already be the next request's.
        started = float(answer.get("_started") or self._request_started_at)
        self.synth_log.append((kind, len(text.split()), requested, started, done))
        speech_trace.emit(
            "piece synthesized",
            words=len(text.split()),
            kind=kind,
            requested=requested,
            worker_started=started,
            done=done,
            took=dict(answer.get("took") or {}),
            process_ready=process_ready,
            text_written=text_written,
            in_memory=piece is not None,
        )
        if samples and piece is None:
            return Piece(text, path=output_file, trimmed=False)
        return piece

    async def _usable_process(self, engine: str, voice: str) -> Any:
        if self._failed_until is not None:
            # After a failure the person hears the system voice at once; the local one is retried in the background.
            if self._clock() >= self._failed_until and not (self.running and self.engine == engine):
                self._failed_until = self._clock() + RETRY_AFTER_SECONDS
                self._start_prewarm(engine, voice)
            if not (self.running and self.engine == engine and self.loaded):
                raise LocalTTSUnavailable(f"{engine} is unavailable")
        process = await self._ensure_process(engine, voice, preload=False)
        if self._load_task is not None and not self._load_task.done():
            await asyncio.shield(self._load_task)
        if not self.running:
            raise LocalTTSUnavailable(f"{engine} worker is not running")
        return self.process

    async def stream(self, engine: str, text: str, voice: str, request_dir: Path) -> StreamedPiece:
        """Start streaming ``text`` (a STREAM_ENGINES worker) and return once its first chunk has arrived, the chunks
        after it arriving on a thread of their own. The pipe is the stream's until its end frame: the next request
        waits for it, at most one chunk after a stop. Cancelled while it waits for the first chunk: the stream is
        stopped."""
        if not text or len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text must contain between 1 and 20,000 characters")
        requested = time.monotonic()
        process = await self._usable_process(engine, voice)
        self._stream_ids += 1
        stream_id = self._stream_ids
        request_path = request_dir / f"stream-{stream_id}.txt"
        request_path.write_text(text, encoding="utf-8")

        def write_stop(which: int) -> None:
            try:
                _write_line(process, {"op": "stop", "id": which})
            except Exception:
                logger.debug("Could not send the voice worker a stop", exc_info=True)

        def send_stop(which: int) -> None:
            # Off the event loop: a pipe the worker is slow to read must not hold the loop.
            threading.Thread(target=write_stop, args=(which,), name="heliox-voice-stop", daemon=True).start()

        piece = StreamedPiece(text, stream_id, send_stop)
        loop = asyncio.get_running_loop()
        first: asyncio.Future[None] = loop.create_future()
        ended: asyncio.Future[None] = loop.create_future()

        def settle(future: asyncio.Future[None]) -> None:
            with contextlib.suppress(RuntimeError):  # the loop has closed
                loop.call_soon_threadsafe(lambda: future.done() or future.set_result(None))

        await self._io_lock.acquire()
        ended.add_done_callback(lambda _done: self._io_lock.release())
        self._background.add(ended)
        ended.add_done_callback(self._background.discard)
        self._request_started_at = time.monotonic()
        request = {"op": "stream", "id": stream_id, "text_file": str(request_path), "text": text, "voice": voice}
        threading.Thread(
            target=_stream_exchange,
            args=(process, request, piece, lambda: settle(first), lambda: settle(ended)),
            name="heliox-voice-stream",
            daemon=True,
        ).start()
        try:
            await bounded(asyncio.shield(first), REQUEST_TIMEOUT_SECONDS)
        except TimeoutError:
            piece.stop()
            raise await self._fail(process, engine, f"no answer within {REQUEST_TIMEOUT_SECONDS:.0f} s") from None
        except asyncio.CancelledError:
            piece.stop()  # cut before the first sound: the worker stops it before its next chunk
            raise
        if piece.received == 0 and piece.error:
            raise await self._fail(process, engine, piece.error)
        if self._failed_until is not None:
            logger.info("Local voice %s recovered", engine)
            self._failed_until = None
            self.unavailable_reason = ""
        speech_trace.emit(
            "stream started", words=len(text.split()), requested=requested,
            worker_started=self._request_started_at, first_chunk=time.monotonic(),
        )  # fmt: skip
        return piece

    async def speak(
        self,
        engine: str,
        text: str,
        voice: str,
        output_file: str | None,
        play: PlayFile,
        *,
        wait_for_load: bool = False,
    ) -> None:
        """Say ``text`` with the local voice, or write it to ``output_file``. A voice that is not loaded is waited for
        only when its load is nearly done (_ready_within_grace; LocalVoiceNotReady otherwise, and the system voice
        speaks), unless ``wait_for_load``: then the reply waits for the load, as when this computer has no system voice
        to speak it meanwhile. One voice per reply (_voice_for_reply).

        Playback starts after the first sentence is synthesised; later sentences are made while earlier ones play.
        Cancelling stops at once: ``play`` stops the audio, and a sentence still being made is finished unheard.
        """
        if engine not in LOCAL_ENGINES:
            raise ValueError(f"Unsupported local TTS engine: {engine}")
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text exceeds the 20,000 character limit")
        self._busy += 1
        self.note_activity()
        speech = 0
        if not output_file:
            self._speeches += 1
            speech = self._speeches
        try:
            sentences = [] if output_file else split_for_speech(text, clauses=engine not in STREAM_ENGINES)
            if output_file:
                await self._announce_loading(engine, voice)  # a file: waited for
                record_reply_voice("local", "file")
            elif wait_for_load:
                # No system voice on this computer: Heliox's own voice is the only one, so there is no switch to avoid.
                if self._all_kept(engine, voice, sentences):
                    record_reply_voice("store", "store")  # played as kept, without waiting for the load
                else:
                    await self._announce_loading(engine, voice)
                    record_reply_voice("local", "waited_no_system_voice")
                self._reply_voice = "local"
            else:
                await self._voice_for_reply(engine, voice, sentences)
            with tempfile.TemporaryDirectory(prefix="heliox-tts-", ignore_cleanup_errors=True) as temp_dir:
                folder = Path(temp_dir)
                if output_file:
                    await self.synthesize(engine, text, voice, Path(output_file), folder)
                    return
                if not sentences:
                    return
                self._spoken_sequence += 1
                progress = SpeechProgress(text, sentences, self._spoken_sequence)
                self.last_progress = progress  # read when the speech is cut off (heard_words)
                token = speech_progress.set(progress)
                try:
                    await self._speak_sentences(engine, sentences, voice, folder, play, speech=speech)
                    progress.finished = True
                except LocalTTSUnavailable:
                    self._reply_voice = "system"  # stopped partway (a fault): the system voice says the rest of it
                    raise
                finally:
                    speech_progress.reset(token)
        finally:
            self._busy -= 1
            self.note_activity()
            if not output_file:
                self._speech_made(speech)  # ended (said, cut, failed or handed to the system voice)
                self.note_spoken()

    def _speech_made(self, speech: int) -> None:
        """Every piece of speech number ``speech`` is made (or it ended): sentences prepared before it may be made."""
        if speech <= self._speeches_made:
            return
        self._speeches_made = speech
        waiters, self._turn_waiters = self._turn_waiters, []
        for waiter in waiters:
            if not waiter.done():
                with contextlib.suppress(RuntimeError):  # its loop has closed
                    waiter.set_result(None)

    def _turn_moved(self) -> None:
        waiters, self._turn_waiters = self._turn_waiters, []
        for waiter in waiters:
            if not waiter.done():
                with contextlib.suppress(RuntimeError):
                    waiter.set_result(None)

    async def _next_turn(self, state: _Next) -> None:
        """Return when a sentence prepared ahead may take the worker: the speech after it has all its own pieces (or
        has ended), or a reply is waiting for this sentence now."""
        while not (state.wanted or self._speeches_made > state.after):
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._turn_waiters.append(waiter)
            try:
                await waiter
            finally:
                if waiter in self._turn_waiters:
                    self._turn_waiters.remove(waiter)

    async def _voice_for_reply(self, engine: str, voice: str, sentences: list[str]) -> None:
        """Decide (or keep) the voice of the reply this speech belongs to, and return when it is Heliox's own voice.
        Raises LocalVoiceNotReady (or LocalTTSUnavailable for a voice that failed) when it is the system voice:

        * the reply began in the system voice: it ends in it, even when Heliox's voice has loaded meanwhile, and even
          for sentences the store could play (the load goes on behind it, for the next reply);
        * Heliox's voice is loaded: it speaks, from the store where every piece is kept;
        * it is not loaded: the system voice speaks this whole reply (_ready_within_grace decides how long a load
          that is nearly done is waited for), and the reply keeps it."""
        ready = self.loaded and self.running and self.engine == engine
        if self.reply_voice() == "system":
            if not ready and self._failed_until is None:
                self._load_behind(engine, voice)
            same = LocalVoiceNotReady(f"{engine}: this reply began in the system voice")
            same.branch = "system_same_reply"
            raise same
        if ready:
            self._reply_voice = "local"
            if self._all_kept(engine, voice, sentences):
                record_reply_voice("store", "store")
            else:
                record_reply_voice("local", "ready")
            return
        self._reply_voice = "system"  # decided now; kept unless the load ends within the wait just below
        if self._failed_until is not None:
            await self._usable_process(engine, voice)  # failed: retried behind (raises LocalTTSUnavailable)
        await self._ready_within_grace(engine, voice)
        if not (self.loaded and self.running and self.engine == engine):
            raise LocalTTSUnavailable(f"{engine} is unavailable")
        self._reply_voice = "local"
        record_reply_voice("local", "loaded_within_grace")

    def _loading_now(self) -> bool:
        """A load of the local voice is under way: its worker is starting, or started and loading."""
        return bool(
            self._spawn_lock.locked()
            or (self._announcing is not None and not self._announcing.done())
            or (self._load_task is not None and not self._load_task.done())
        )

    def _load_behind(self, engine: str, voice: str) -> asyncio.Future[None]:
        """Start loading the local voice when no load is under way; never waits. The load (announced) under way."""
        loading = self._announcing
        if loading is not None and not loading.done():
            return loading
        if not self._loading_now():
            self._loading_since = self._clock()  # a new load starts now
        loading = asyncio.ensure_future(self._announce_loading(engine, voice))
        self._announcing = loading
        self._background.add(loading)
        loading.add_done_callback(self._background.discard)
        loading.add_done_callback(lambda done: done.cancelled() or done.exception())
        return loading

    def _read_load_times(self) -> None:
        if self._load_times_read or self.head_cache is None:
            return
        self._load_times_read = True
        try:
            kept = json.loads((self.head_cache.folder / "loads.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(kept, list):
            ok = [r for r in kept if isinstance(r, list) and len(r) == 3 and isinstance(r[2], (int, float))]
            self.load_times = ok + self.load_times

    def _note_load_time(self, engine: str, seconds: float) -> None:
        """Keep how long a load just took (the latest LOAD_RECORDS per voice and device), and save them."""
        self._read_load_times()
        where = "cpu" if self.effective_device == "cpu" else "card"
        self.load_times.append([engine, where, round(max(0.0, seconds), 2)])
        mine = [r for r in self.load_times if r[0] == engine and r[1] == where]
        for old in mine[: max(0, len(mine) - LOAD_RECORDS)]:
            self.load_times.remove(old)
        if self.head_cache is not None:
            try:
                self.head_cache.folder.mkdir(parents=True, exist_ok=True)
                (self.head_cache.folder / "loads.json").write_text(json.dumps(self.load_times), encoding="utf-8")
            except OSError:
                logger.debug("Could not save how long the voice took to load", exc_info=True)

    def expected_load_left(self, engine: str) -> float:
        """How long the load under way (or one starting now) is expected to take still: the median of this voice's
        recorded loads on the device it is starting on (any device before the worker has started), less how long it
        has run; DEFAULT_LOAD_SECONDS with nothing recorded."""
        self._read_load_times()
        where = None if self.process is None else ("cpu" if self.effective_device == "cpu" else "card")
        seconds = [r[2] for r in self.load_times if r[0] == engine and (where is None or r[1] == where)]
        if not seconds and where is not None:
            seconds = [r[2] for r in self.load_times if r[0] == engine]
        import statistics

        expected = statistics.median(seconds) if seconds else DEFAULT_LOAD_SECONDS
        ran = self._clock() - self._loading_since if self._loading_now() else 0.0
        return max(0.0, expected - ran)

    def _all_kept(self, engine: str, voice: str, pieces: list[str]) -> bool:
        """Every piece of the reply is in the store of made audio: it plays without the worker, loaded or not."""
        store = self.head_cache
        if store is None or not pieces:
            return False
        signature = SpeechStore.signature(engine, voice)
        return all(store.has(piece, signature) for piece in pieces)

    async def _ready_within_grace(self, engine: str, voice: str) -> None:
        """Return when the local voice can speak this reply: at once when it is loaded; else the load is started (when
        none is under way) and left running, and the reply waits for it only when it is expected to end within
        LOAD_GRACE_SECONDS (expected_load_left, plus LOAD_WAIT_MARGIN_SECONDS). Otherwise, or still loading after
        that wait: LocalVoiceNotReady at once, and the system voice speaks this reply whole. A voice that cannot load
        (too little memory, a small computer, a failure) raises LocalTTSUnavailable as before."""
        if self.loaded and self.running and self.engine == engine:
            return
        if self._failed_until is not None:
            return  # the caller's synthesis falls back as before, without a new announcement
        started = not self._loading_now()
        if started and not (self.running and self.engine == engine):
            blocked = self._start_blocked()
            if blocked:
                # This computer should not load it now: said once, in plain words, and the system voice speaks.
                from pilot.system.speech_budget import unavailable_sentence

                self._mark_failed(engine, blocked)
                await self._notify("voice_unavailable", {"message": unavailable_sentence(self.unavailable_reason)})
                raise LocalTTSUnavailable(f"{engine}: {blocked}")
        loading = self._load_behind(engine, voice)
        if started:
            # The load's first step runs now: "Getting my voice ready" goes out as the system voice speaks.
            await asyncio.sleep(0)
        left = self.expected_load_left(engine)
        wait = left + LOAD_WAIT_MARGIN_SECONDS
        if wait > LOAD_GRACE_SECONDS:
            logger.info(
                "Heliox's own voice is %s (about %.0f s to go); this reply is spoken with the system voice",
                "starting to load" if started else "still loading",
                left,
            )
            raise LocalVoiceNotReady(f"{engine}: still loading")
        try:
            await bounded(asyncio.shield(loading), timeout=wait)
        except TimeoutError:
            logger.info("The local voice is still loading; this reply is spoken with the system voice")
            raise LocalVoiceNotReady(f"{engine}: still loading") from None

    async def _announce_loading(self, engine: str, voice: str) -> None:
        if self.loaded and self.running and self.engine == engine:
            return
        if self._failed_until is not None:
            return
        started = self._clock()
        await self._notify("voice_loading", {"message": LOADING_MESSAGE})
        try:
            await self._ensure_process(engine, voice, preload=True)
            if self._load_task is not None:
                await asyncio.shield(self._load_task)
        except LocalTTSUnavailable:
            from pilot.system.speech_budget import unavailable_sentence

            # Said once per failure (a failed voice is not announced again until it recovers): why, in plain words.
            await self._notify("voice_unavailable", {"message": unavailable_sentence(self.unavailable_reason)})
            raise
        logger.info("Waited %.1f s for the local voice to load before speaking", self._clock() - started)
        await self._notify("voice_ready", {})

    async def _speak_sentences(
        self,
        engine: str,
        sentences: list[str],
        voice: str,
        folder: Path,
        play: PlayFile,
        *,
        speech: int = 0,
    ) -> None:
        heads = self.head_cache
        signature = SpeechStore.signature(engine, voice)
        cached_head = heads.lookup(sentences[0], signature) if heads is not None and sentences else None

        sources: dict[int, str] = {}

        def make(index: int) -> asyncio.Future[Piece]:
            kept = cached_head if index == 0 else (heads.lookup(sentences[index], signature) if heads else None)
            if kept is not None:
                sources[index] = "store"
                done: asyncio.Future[Piece] = asyncio.get_running_loop().create_future()
                done.set_result(Piece(sentences[index], path=kept, source="store"))  # exactly this text, made this way
                return done
            key = (engine, voice, sentences[index])
            ahead = self._prepared.pop(key, None)
            state = self._next_state.pop(key, None)
            if ahead is not None and not ahead.cancelled():
                sources[index] = "ahead"
                if state is not None and not state.wanted:
                    state.wanted = True  # this reply waits for it now: it goes first on the pipe
                    self._turn_moved()
                return ahead  # synthesised while the sentence before it played
            sources[index] = "made"
            if engine in STREAM_ENGINES:
                # Streamed: ready on its first chunk. The one after the sentence playing starts as soon as that one's
                # end frame frees the pipe, so its first chunk is in before this one has finished playing.
                return asyncio.ensure_future(self.stream(engine, sentences[index], voice, folder))
            target = folder / f"sentence-{index}.wav"
            return asyncio.ensure_future(self._make(engine, sentences[index], voice, target, folder))

        upcoming: asyncio.Future[Any] | None = make(0)
        playing: Any = None
        try:
            for index in range(len(sentences)):
                assert upcoming is not None
                try:
                    piece = await upcoming
                except LocalTTSUnavailable as not_ready:
                    # What was said stays said: the system voice takes over from this sentence on.
                    not_ready.unspoken = " ".join(sentences[index:]) if index else ""
                    raise
                playing = piece
                words = len(sentences[index].split())
                speech_trace.emit(
                    "piece ready", index=index, words=words, source=sources.get(index, ""), at=time.monotonic()
                )
                upcoming = make(index + 1) if index + 1 < len(sentences) else None
                if upcoming is None and speech:
                    self._speech_made(speech)  # every piece of this speech is made: what was prepared may go now
                speech_trace.emit("piece plays", index=index, words=words, at=time.monotonic())
                await play(piece)
                progress = speech_progress.get()
                if progress is not None and progress.playing is not None:
                    current = progress.playing
                    progress.durations.append(playback_samples(current) / max(1, int(current.sample_rate)))
                    progress.playing = None
                self.note_activity()
                if isinstance(piece, StreamedPiece) and piece.error:
                    # The worker failed partway through the sentence: what arrived was played, the rest of the reply
                    # goes to the system voice from the next sentence on (a partial sentence is not said twice).
                    failure = await self._fail(self.process, engine, piece.error) if self.process else None
                    unavailable = failure or LocalTTSUnavailable(f"{engine}: {piece.error}")
                    unavailable.unspoken = " ".join(sentences[index + 1 :])
                    if not unavailable.unspoken:
                        return
                    raise unavailable
                if index == 0 and heads is not None and cached_head is None:
                    # Counted, and its audio kept once said HEAD_CACHE_AFTER times: written after it was heard.
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(heads.said, sentences[0], signature, piece)
                path = piece if isinstance(piece, Path) else getattr(piece, "path", None)
                if path is not None and self._prepared_dir is not None and path.parent == self._prepared_dir:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    with contextlib.suppress(OSError):
                        path.with_name(path.name + ".timing.json").unlink(missing_ok=True)
        finally:
            if isinstance(playing, StreamedPiece):
                playing.stop()  # cut or failed mid-sentence: the worker stops making it (nothing once it has ended)
            if upcoming is not None and not upcoming.done():
                upcoming.cancel()
                with contextlib.suppress(BaseException):
                    await upcoming
            elif upcoming is not None and not upcoming.cancelled() and upcoming.exception() is None:
                ahead = upcoming.result()
                if isinstance(ahead, StreamedPiece):
                    ahead.stop()  # the next sentence had started streaming: never heard, so not made on

    def prepare(self, engine: str, text: str, voice: str) -> None:
        """Start synthesising ``text`` now, so that a later speak() of it plays at once. Never blocks.

        For a reply streamed sentence by sentence: prepare the next sentence while the current one plays, and
        there is no gap between them. Unused preparations are dropped (the oldest after PREPARED_LIMIT).

        The conversation asks for the next sentence just before it hands over the current one. So a sentence prepared
        here is not made until the speech that follows this call has all its own pieces (or has ended), and when a
        reply waits for it, it goes first: the current sentence never waits behind the next one (_Next, _Pipe).
        """
        if engine not in LOCAL_ENGINES or engine in STREAM_ENGINES or self._failed_until is not None:
            return  # a streamed engine starts sounding within a chunk: nothing made ahead to hold the pipe
        for sentence in split_for_speech(text):
            key = (engine, voice, sentence)
            if key in self._prepared:
                continue
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            if self._prepared_dir is None:
                self._prepared_dir = Path(tempfile.mkdtemp(prefix="heliox-tts-ahead-"))
            target = self._prepared_dir / f"ahead-{len(self._prepared)}-{id(key):x}.wav"
            state = _Next(after=self._speeches)
            task = loop.create_task(self._make_next(engine, sentence, voice, target, self._prepared_dir, state))
            task.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._prepared[key] = task
            self._next_state[key] = state
            while len(self._prepared) > PREPARED_LIMIT:
                oldest = next(iter(self._prepared))
                self._prepared.pop(oldest).cancel()
                self._next_state.pop(oldest, None)

    async def _make_next(self, engine: str, text: str, voice: str, target: Path, folder: Path, state: _Next) -> Piece:
        """A sentence prepared ahead (prepare): made once its turn comes (_next_turn), behind any piece a reply waits
        for unless a reply waits for this one."""
        await self._next_turn(state)
        return await self._make(
            engine, text, voice, target, folder, rank=lambda: PIPE_NOW if state.wanted else PIPE_NEXT, kind="next"
        )

    async def make_ahead(
        self,
        engine: str,
        voice: str,
        pieces: Sequence[str],
        *,
        quiet: Callable[[], bool] = lambda: True,
        battery: Callable[[], bool] = on_battery,
        rest_factor: float = MAKE_AHEAD_REST_FACTOR,
        poll_seconds: float = 1.0,
    ) -> int:
        """Make the audio of the fixed ``pieces`` the store lacks (of MAKE_AHEAD_MAX_WORDS words or fewer), in their
        order, one at a time, only while the voice is loaded, nothing has used it for MAKE_AHEAD_IDLE_SECONDS (no
        reply, no request, no sign of use: note_activity), ``quiet()`` (no conversation open and the room quiet) and
        the computer is not on ``battery()`` power (it waits, and goes on once plugged in), resting ``rest_factor``
        times each piece's synthesis time before the next. A piece already being made when a reply arrives is
        finished first (the worker cannot drop it): at most one short piece's synthesis. Stops when the store is full
        or the voice changes. Returns how many pieces were made."""
        store = self.head_cache
        if store is None or engine not in LOCAL_ENGINES:
            return 0
        signature = SpeechStore.signature(engine, voice)
        made = 0
        with tempfile.TemporaryDirectory(prefix="heliox-tts-fixed-", ignore_cleanup_errors=True) as temp_dir:
            folder = Path(temp_dir)
            for index, text in enumerate(pieces):
                if store.has(text, signature) or len(text.split()) > MAKE_AHEAD_MAX_WORDS:
                    continue
                while not (
                    self.running
                    and self.loaded
                    and self.engine == engine
                    and self._busy == 0
                    and self._clock() - self._last_activity >= MAKE_AHEAD_IDLE_SECONDS
                    and quiet()
                    and not battery()
                ):
                    if self.head_cache is not store or self._failed_until is not None:
                        return made
                    await self._sleep(poll_seconds)
                started = self._clock()
                try:
                    piece = await self._make(
                        engine, text, voice, folder / f"fixed-{index}.wav", folder, rank=PIPE_AHEAD, kind="ahead"
                    )
                except LocalTTSUnavailable:
                    return made
                if not store.add_fixed(text, signature, piece):
                    return made  # full
                made += 1
                if piece.path is not None:
                    with contextlib.suppress(OSError):
                        piece.path.unlink()
                await self._sleep(max(poll_seconds, rest_factor * (self._clock() - started)))
        logger.info("Made %d fixed sentences of the local voice ahead", made)
        return made

    async def _make(
        self,
        engine: str,
        text: str,
        voice: str,
        target: Path,
        folder: Path,
        *,
        rank: int | Callable[[], int] = PIPE_NOW,
        kind: str = "reply",
    ) -> Piece:
        """One piece made for speech, in memory and trimmed by the worker. A worker that wrote the file instead (one
        that cannot send samples) has it trimmed here, as before. ``rank`` and ``kind``: see _Pipe and synth_log."""
        keep_tail = None if ends_sentence(text) else KEEP_CLAUSE_SECONDS
        piece = await self.synthesize(
            engine, text, voice, target, folder, wait_after_gpu_fault=False, samples=True, keep_tail=keep_tail,
            rank=rank, kind=kind,
        )  # fmt: skip
        assert piece is not None
        if not piece.trimmed:
            try:
                kept = await asyncio.to_thread(trim_silence, target, KEEP_LEAD_SECONDS, keep_tail)
            except Exception:
                kept = None
                logger.debug("Could not trim the silence around a synthesised piece; playing it as made", exc_info=True)
            if kept is not None:
                from pilot.system import speech_notices

                # The phoneme times count from the first sample made: from the first one kept after the trim.
                await asyncio.to_thread(speech_notices.shift_timing, target, kept[0], kept[1])
            piece.trimmed = True
        speech_trace.emit("piece trimmed", words=len(text.split()), done=time.monotonic())
        return piece

    async def _notify(self, status: str, data: dict[str, Any]) -> None:
        callback = self.on_status
        if callback is None:
            return
        try:
            await callback(status, data)
        except Exception:
            logger.debug("Local voice status callback failed", exc_info=True)


_manager = LocalTTSWorkerManager()


def manager() -> LocalTTSWorkerManager:
    return _manager


_reply_sequence = 0


def record_reply_voice(voice: str, branch: str, reason: str = "") -> ReplyVoice:
    """Note which voice speaks the reply being started, and by which branch (REPLY_VOICES, REPLY_BRANCHES): kept as
    manager().last_reply, read by the speech coordinator for the reply's SpeechOutcome. Data only."""
    global _reply_sequence
    _reply_sequence += 1
    record = ReplyVoice(_reply_sequence, voice, branch, reason, time.monotonic())
    _manager.last_reply = record
    return record


def reply_sequence() -> int:
    return _reply_sequence


def configure_local_voice(voice_config: Any, on_status: StatusCallback | None = None) -> None:
    """Apply ``[voice]`` settings: how long the voice stays loaded and on which device it runs.

    A worker started for another engine or device is released; the next speech loads the new one.
    """
    _manager.configure(
        keep_loaded_seconds=getattr(voice_config, "tts_keep_loaded_seconds", DEFAULT_KEEP_LOADED_SECONDS),
        device=getattr(voice_config, "tts_device", "auto"),
    )
    if on_status is not None:
        _manager.on_status = on_status
    # One GPU user at a time: the voice starts on the CPU when the card is hot, busy or switched off.
    from pilot.system import gpu_arbiter

    _manager.choose_device = gpu_arbiter.voice_device
    _manager.on_gpu_fault = gpu_arbiter.arbiter().record_fault
    _manager.lease_wants = gpu_arbiter.arbiter().wants
    _manager.free_memory = lambda: gpu_arbiter.probe_host_memory()[0]
    # Released when idle only when the computer needs the memory, unless the person chose the clock (voice.tts_release).
    from pilot.system import speech_budget

    if str(getattr(voice_config, "tts_release", "memory")) == "time":
        _manager.keep_while = None
    else:
        _manager.keep_while = lambda: speech_budget.keep_voice_loaded(
            speech_budget.free_memory(), speech_budget.total_memory()
        )

    light = bool(getattr(voice_config, "light_on_small_memory", True))
    _manager.light_reason = lambda: speech_budget.light_voice_reason(speech_budget.total_memory(), light)
    engine = str(getattr(voice_config, "tts_engine", ""))
    if engine in LOCAL_ENGINES:
        # The heads this voice says most, ready to play; entries made with another voice or settings are dropped.
        if _manager.head_cache is None:
            _manager.head_cache = SpeechStore()
        with contextlib.suppress(Exception):
            _manager.head_cache.keep_only(SpeechStore.signature(engine, str(getattr(voice_config, "tts_voice", ""))))
    if _manager.running and (_manager.engine != engine or _manager.process_device != _manager.device):
        with contextlib.suppress(RuntimeError):
            task = asyncio.get_running_loop().create_task(_manager.shutdown())
            _manager._background.add(task)
            task.add_done_callback(_manager._background.discard)


def local_voice_state(voice_config: Any) -> dict[str, bool]:
    """``voice_loaded``: Heliox can speak now without loading its voice. ``voice_loading``: a load is under way.

    The system voice (``os_native``) needs no load. A local voice that failed counts as not loaded: Heliox would
    speak with the system voice until it recovers.
    """
    engine = str(getattr(voice_config, "tts_engine", ""))
    if engine not in LOCAL_ENGINES:
        return {"voice_loaded": True, "voice_loading": False}
    worker = _manager
    same = worker.running and worker.engine == engine
    loaded = bool(same and worker.loaded and worker._failed_until is None)
    loading = bool(same and not loaded and worker._load_task is not None and not worker._load_task.done())
    return {"voice_loaded": loaded, "voice_loading": loading}


def made_ahead(voice_config: Any, text: str) -> bool:
    """Whether saying ``text`` now plays audio made ahead in Heliox's own voice: every piece of it (as the worker is
    given it) is in the store, made with the configured local voice, and that voice speaks now (loaded, and the reply
    under way did not begin in the system voice: speaks_locally_now). False for the system voice, with no store,
    while Heliox's voice is not loaded, or when a piece has not been made yet. Reads the store's index (kept in
    memory after the first read); never synthesises."""
    engine, voice = str(getattr(voice_config, "tts_engine", "")), str(getattr(voice_config, "tts_voice", ""))
    store = _manager.head_cache
    if engine not in LOCAL_ENGINES or store is None:
        return False
    if not _manager.speaks_locally_now(engine):
        # Heliox's own voice is not loaded, or this reply began in the system voice: what is said now is said in the
        # system voice, so audio made ahead in Heliox's voice is not "ready" (no hum in one voice before an answer in
        # another).
        return False
    pieces = split_for_speech(text, clauses=engine not in STREAM_ENGINES)
    signature = SpeechStore.signature(engine, voice)
    return bool(pieces) and all(store.has(piece, signature) for piece in pieces)


def prepare_local_speech(voice_config: Any, text: str) -> None:
    """Synthesise ``text`` ahead of time with the configured local voice (no-op for the system voice), so the
    speak() that follows plays it without waiting. Call it for the next sentence while the current one plays."""
    _manager.prepare(str(getattr(voice_config, "tts_engine", "")), text, str(getattr(voice_config, "tts_voice", "")))


def make_fixed_speech_ahead(voice_config: Any, quiet: Callable[[], bool] = lambda: True) -> asyncio.Task[int] | None:
    """Start making the configured local voice's audio for every fixed sentence (pilot.system.fixed_speech) in the
    background, while the machine is idle; once per engine and voice. None when there is nothing to start."""
    global _making_ahead
    engine, voice = str(getattr(voice_config, "tts_engine", "")), str(getattr(voice_config, "tts_voice", ""))
    if engine not in LOCAL_ENGINES or _manager.head_cache is None:
        return None
    current = _making_ahead
    if current is not None and current[0] == (engine, voice) and not current[1].done():
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    async def run() -> int:
        from pilot.system.fixed_speech import gather

        pieces = await asyncio.to_thread(gather)  # reads the source files: off the event loop
        return await _manager.make_ahead(engine, voice, pieces, quiet=quiet)

    task = loop.create_task(run())
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
    _making_ahead = ((engine, voice), task)
    return task


_making_ahead: tuple[tuple[str, str], asyncio.Task[int]] | None = None


def prewarm_local_voice(voice_config: Any) -> None:
    """The person is talking to Heliox: load the configured local voice now. Safe to call often; never blocks."""
    _manager.prewarm(str(getattr(voice_config, "tts_engine", "")), str(getattr(voice_config, "tts_voice", "")))


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
