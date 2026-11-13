"""Process boundary for optional local TTS model inference, and its lifecycle.

The worker (``python -m pilot.system.local_tts_worker --engine kokoro_tts --serve``) keeps the model runtime out of
the daemon: importing Kokoro pulls in torch and transformers, about 1.6-1.9 GB resident plus about 840 MB of GPU
memory on CUDA. It answers one JSON line per request on stdout.

``LocalTTSWorkerManager`` is the daemon side. Measured on the owner's machine (scripts/measure_kokoro_startup.py), a
cold worker needs about 20-25 s before its first word: 0.25 s process start, 3.3 s importing torch, 12-13 s importing
Kokoro (transformers walks thousands of files), 2.7 s loading weights and 0.9 s for the first synthesis. A loaded
worker says a short sentence in 0.1 s and a long one in 0.25 s on the GPU. So the manager:

* keeps the worker loaded while a conversation is plausible: until ``keep_loaded_seconds`` (default 300) pass with
  nobody speaking to Heliox and Heliox saying nothing, then releases it;
* starts it ahead of need (``prewarm_local_voice``) as soon as the person is heard;
* speaks a long reply sentence by sentence, synthesising the next sentence while the current one plays;
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
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pilot.timeouts import bounded

logger = logging.getLogger("pilot.system.local_tts_worker")

LOCAL_ENGINES = frozenset({"kokoro_tts", "pocket_tts"})
PREPARED_LIMIT = 6  # sentences synthesised ahead and not yet spoken
DEVICES = ("auto", "cpu", "cuda")
MAX_TEXT_CHARACTERS = 20_000
DEFAULT_KEEP_LOADED_SECONDS = 300.0
MAX_KEEP_LOADED_SECONDS = 4 * 3600.0
RETRY_AFTER_SECONDS = 120.0
LOAD_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 120.0
LOADING_MESSAGE = "Getting my voice ready…"
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


def _synthesize(engine: str, text: str, voice: str, output_file: str) -> None:
    """Make one piece on the worker's own thread. Through asyncio.run, each request started a new event loop and ran
    the model on a new executor thread: 85 to 130 ms more per sentence than the same generation called directly
    (measured 13 November on the owner's CPU, the six sentences of a short reply)."""
    _use_voice_threads()
    _engine_module(engine).synthesize_to_file_here(text, voice, output_file)


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


def _serve(engine: str, device: str = "auto", gpu_lease: bool = False) -> int:
    lease = None
    if device != "cpu" and cuda_hidden():
        device = "cpu"  # every card hidden: on POSIX the worker sees the empty value itself
    if gpu_lease and device != "cpu":
        device, lease = _take_gpu_lease(device)
    if engine == "kokoro_tts":
        from pilot.system import kokoro_tts

        kokoro_tts.configure(device=device)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "load":
                _load(engine, str(request.get("voice") or ""))
                print(json.dumps({"status": "ok", "op": "load"}), flush=True)
                continue
            text = Path(request["text_file"]).read_text(encoding="utf-8")
            if not text or len(text) > MAX_TEXT_CHARACTERS:
                raise ValueError("Speech text must contain between 1 and 20,000 characters")
            _synthesize(engine, text, str(request["voice"]), str(request["output_file"]))
            response = {"status": "ok"}
        except Exception as exc:
            response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)
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
    """The local voice cannot speak right now; the failure has already been logged once."""

    already_logged = True


class _MovedToCpu(LocalTTSUnavailable):
    """The voice hit a GPU error; the worker was closed and the next one starts on the CPU. Retried at once."""


_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No)|(?:\b[A-Za-z]\.)*\b[A-Za-z])\.$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+|\n+")


def split_for_speech(text: str) -> list[str]:
    """Sentences to synthesise one at a time, so the first one can play while the rest are made."""
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
    if merged:
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


class HeadCache:
    """The heads this person's Heliox says most, kept as audio so the next one plays without being synthesised.

    Keyed by the head's exact text and everything that makes its sound (engine, voice, speed, the trim settings, this
    version), so a head plays from the cache only when the piece about to be spoken is exactly that text, made the
    same way: it never adds, drops or changes a word of a reply. Filled only from audio Heliox has already made for a
    reply (never synthesised for the cache), bounded to HEAD_CACHE_ENTRIES (the least recently used go first), kept
    beside Heliox's other caches (``~/.cache/heliox/voice_heads``, which removal deletes on every system), and
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

    def said(self, text: str, signature: str, audio: Path) -> bool:
        """``text`` was just spoken from ``audio`` (made for a reply). Counts it, and keeps a copy of the audio once it
        has been said HEAD_CACHE_AFTER times. True when it was kept now."""
        if len(text.split()) > HEAD_CACHE_WORDS or not audio.is_file():
            return False
        index = self._load()
        key = self.key(text, signature)
        entry = index.setdefault(key, {"text": text, "signature": signature, "count": 0, "kept": False})
        entry["count"] += 1
        entry["last"] = time.time()
        kept = False
        if entry["count"] >= HEAD_CACHE_AFTER and not entry["kept"]:
            try:
                import shutil

                self.folder.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(audio, self.folder / f"{key}.wav")
                entry["kept"] = kept = True
            except OSError:
                logger.debug("Could not keep a head's audio", exc_info=True)
        self._prune()
        self._save()
        return kept

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

    def _prune(self) -> None:
        index = self._load()
        kept = sorted((e.get("last", 0.0), k) for k, e in index.items() if e.get("kept"))
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


def trim_silence(path: Path, keep_lead: float, keep_tail: float | None) -> None:
    """Cut a synthesised WAV's padding in place (see KEEP_LEAD_SECONDS). Without soundfile it is left as made."""
    try:
        import soundfile
    except ImportError:
        return
    info = soundfile.info(str(path))
    audio, rate = soundfile.read(str(path), dtype="float32")
    start, stop = silence_edges(audio, rate, keep_lead, keep_tail)
    if start == 0 and stop == len(audio):
        return
    soundfile.write(str(path), audio[start:stop], rate, subtype=info.subtype)


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


def _exchange(process: Any, request: dict[str, str]) -> bytes:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Local TTS worker streams are unavailable")
    process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
    process.stdin.flush()
    return process.stdout.readline()


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


StatusCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
PlayFile = Callable[[Path], Awaitable[None]]

# What of a sentence the person heard when Heliox was cut off. A word counts only if it had finished playing this
# long before the voice stopped: from the person's first sound to silence takes 0.45 to 0.65 s (the barge-in
# decision, then an 8 ms fade and the device's buffered block), and what Heliox believes it said must never be more
# than the person heard, so the longer end is taken.
HEARD_MARGIN_SECONDS = 0.65


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
            played += min(int(getattr(current, "position", 0)), len(current.audio)) / rate
            total += len(current.audio) / rate
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
            lengths.append(len(current.audio) / max(1, int(getattr(current, "sample_rate", 1))))
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
        # Why the voice runs on the CPU for the rest of the session ("" while the GPU may still be used).
        self.cpu_for_session = ""
        # Where the running worker's voice runs and why, in plain words, for the model health status.
        self.device_reason = ""
        # The heads this Heliox says most, kept as audio (HeadCache); None: nothing kept (tests, a written file).
        self.head_cache: HeadCache | None = None
        # The last speak()'s pieces and playback, for what was heard of it when it is cut off (SpeechProgress).
        self.last_progress: SpeechProgress | None = None
        self._spoken_sequence = 0
        self.gpu_lease = False  # the running worker holds Heliox's GPU lease (the arbiter said the card is usable)
        # Free system memory now, or None: the worker is not started when less than VOICE_HOST_RAM_BYTES is free
        # (on an 8 GB machine the model's 1.6-1.9 GB can push everything into swap); the system voice speaks.
        self.free_memory: Callable[[], int | None] | None = None
        self.loaded = False
        self._spawn_lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._load_task: asyncio.Future[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._busy = 0
        self._last_activity = clock()
        self._failed_until: float | None = None
        self._background: set[asyncio.Future[Any]] = set()
        # Sentences synthesised ahead of their turn (prepare()), keyed by (engine, voice, sentence).
        self._prepared: dict[tuple[str, str, str], asyncio.Task[Path]] = {}
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

    def prewarm(self, engine: str, voice: str) -> None:
        """Start the worker now, without waiting, so it is ready by the time there is something to say."""
        if engine not in LOCAL_ENGINES:
            return
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
            free = self.free_memory() if self.free_memory is not None else None
            if free is not None and free < VOICE_HOST_RAM_BYTES:
                reason = (
                    f"not enough free memory for the local voice ({free / 1024**3:.1f} GB free, "
                    f"{VOICE_HOST_RAM_BYTES / 1024**3:.0f} GB needed)"
                )
                self._mark_failed(engine, reason)
                raise LocalTTSUnavailable(f"{engine}: {reason}")
            device, gpu_lease = self.device, False
            if device != "cpu" and (self.cpu_for_session or cuda_hidden()):
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
            self.device_reason = self._describe(device, gpu_lease)
            try:
                if gpu_lease:
                    process = await asyncio.to_thread(self._spawn, engine, device, True)
                else:
                    process = await asyncio.to_thread(self._spawn, engine, device)
            except Exception as exc:
                self._mark_failed(engine, f"could not start: {exc}")
                raise LocalTTSUnavailable(f"{engine} worker could not start: {exc}") from exc
            self.process = process
            self.engine = engine
            self.process_device = self.device
            self.effective_device = device
            self.gpu_lease = gpu_lease
            self.loaded = False
            self._load_task = None
            self.note_activity()
            self._start_idle_watch()
            if preload:
                self._load_task = asyncio.ensure_future(self._load(process, engine, voice))
                self._background.add(self._load_task)
                self._load_task.add_done_callback(self._background.discard)
            return process

    async def _load(self, process: Any, engine: str, voice: str) -> None:
        started = self._clock()
        try:
            await self._request(process, engine, {"op": "load", "voice": voice}, timeout=LOAD_TIMEOUT_SECONDS)
        except _MovedToCpu:
            process = await self._ensure_process(engine, voice, preload=False)
            await self._request(process, engine, {"op": "load", "voice": voice}, timeout=LOAD_TIMEOUT_SECONDS)
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
            if remaining <= 0 and self._busy == 0 and not self._io_lock.locked():
                process = self.process
                logger.info(
                    "Released the local voice after %.0f s without conversation; it reloads when the person speaks",
                    self.keep_loaded_seconds,
                )
                self._idle_task = None
                await self._close(process)
                return
            await self._sleep(max(remaining, 1.0))

    async def _close(self, process: Any) -> None:
        if process is self.process:
            self.process = None
            self.engine = None
            self.loaded = False
            self._load_task = None
        await asyncio.to_thread(_close_process, process)

    async def shutdown(self) -> None:
        idle = self._idle_task
        self._idle_task = None
        if idle is not None and idle is not asyncio.current_task() and not idle.done():
            idle.cancel()
        if self.process is not None:
            await self._close(self.process)

    def _mark_failed(self, engine: str, reason: str) -> None:
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
        if cuda_hidden():
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

    async def _request(self, process: Any, engine: str, request: dict[str, str], *, timeout: float) -> None:
        """One request/response on the pipe. Cancelling the caller lets the worker finish it unheard."""
        await self._io_lock.acquire()
        exchange = asyncio.ensure_future(asyncio.to_thread(_exchange, process, request))
        exchange.add_done_callback(lambda _done: self._io_lock.release())
        self._background.add(exchange)
        exchange.add_done_callback(self._background.discard)
        try:
            line = await bounded(asyncio.shield(exchange), timeout)
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
            self.loaded = True

    # Speaking ----------------------------------------------------------------

    async def synthesize(self, engine: str, text: str, voice: str, output_file: Path, request_dir: Path) -> None:
        if engine not in LOCAL_ENGINES:
            raise ValueError(f"Unsupported local TTS engine: {engine}")
        if not text or len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text must contain between 1 and 20,000 characters")
        process = await self._usable_process(engine, voice)
        request_path = request_dir / f"{output_file.stem}-{id(output_file):x}.txt"
        request_path.write_text(text, encoding="utf-8")
        request = {"text_file": str(request_path), "voice": voice, "output_file": str(output_file)}
        try:
            await self._request(
                process, engine, request, timeout=LOAD_TIMEOUT_SECONDS if not self.loaded else REQUEST_TIMEOUT_SECONDS
            )
        except _MovedToCpu:
            # The same sentence, once, from the same voice started on the CPU.
            process = await self._usable_process(engine, voice)
            await self._request(process, engine, request, timeout=LOAD_TIMEOUT_SECONDS)
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise await self._fail(process, engine, "the worker produced no audio")

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

    async def speak(
        self,
        engine: str,
        text: str,
        voice: str,
        output_file: str | None,
        play: PlayFile,
    ) -> None:
        """Say ``text`` with the local voice, or write it to ``output_file``.

        Playback starts after the first sentence is synthesised; later sentences are made while earlier ones play.
        Cancelling stops at once: ``play`` stops the audio, and a sentence still being made is finished unheard.
        """
        if engine not in LOCAL_ENGINES:
            raise ValueError(f"Unsupported local TTS engine: {engine}")
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError("Speech text exceeds the 20,000 character limit")
        self._busy += 1
        self.note_activity()
        try:
            await self._announce_loading(engine, voice)
            with tempfile.TemporaryDirectory(prefix="heliox-tts-", ignore_cleanup_errors=True) as temp_dir:
                folder = Path(temp_dir)
                if output_file:
                    await self.synthesize(engine, text, voice, Path(output_file), folder)
                    return
                sentences = split_for_speech(text)
                if not sentences:
                    return
                self._spoken_sequence += 1
                progress = SpeechProgress(text, sentences, self._spoken_sequence)
                self.last_progress = progress  # read when the speech is cut off (heard_words)
                token = speech_progress.set(progress)
                try:
                    await self._speak_sentences(engine, sentences, voice, folder, play)
                    progress.finished = True
                finally:
                    speech_progress.reset(token)
        finally:
            self._busy -= 1
            self.note_activity()

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
            await self._notify("voice_unavailable", {"message": "Speaking with the system voice"})
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
    ) -> None:
        heads = self.head_cache
        signature = HeadCache.signature(engine, voice)
        cached_head = heads.lookup(sentences[0], signature) if heads is not None and sentences else None

        def make(index: int) -> asyncio.Future[Path]:
            if index == 0 and cached_head is not None:
                done: asyncio.Future[Path] = asyncio.get_running_loop().create_future()
                done.set_result(cached_head)  # exactly this text, made this way: played as kept
                return done
            ahead = self._prepared.pop((engine, voice, sentences[index]), None)
            if ahead is not None and not ahead.cancelled():
                return ahead  # synthesised while the sentence before it played
            target = folder / f"sentence-{index}.wav"
            return asyncio.ensure_future(self._make(engine, sentences[index], voice, target, folder))

        upcoming: asyncio.Future[Path] | None = make(0)
        try:
            for index in range(len(sentences)):
                assert upcoming is not None
                path = await upcoming
                upcoming = make(index + 1) if index + 1 < len(sentences) else None
                await play(path)
                progress = speech_progress.get()
                if progress is not None and progress.playing is not None:
                    current = progress.playing
                    progress.durations.append(len(current.audio) / max(1, int(current.sample_rate)))
                    progress.playing = None
                self.note_activity()
                if index == 0 and heads is not None and cached_head is None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(heads.said, sentences[0], signature, path)
                if self._prepared_dir is not None and path.parent == self._prepared_dir:
                    with contextlib.suppress(OSError):
                        path.unlink()
        finally:
            if upcoming is not None and not upcoming.done():
                upcoming.cancel()
                with contextlib.suppress(BaseException):
                    await upcoming

    def prepare(self, engine: str, text: str, voice: str) -> None:
        """Start synthesising ``text`` now, so that a later speak() of it plays at once. Never blocks.

        For a reply streamed sentence by sentence: prepare the next sentence while the current one plays, and
        there is no gap between them. Unused preparations are dropped (the oldest after PREPARED_LIMIT).
        """
        if engine not in LOCAL_ENGINES or self._failed_until is not None:
            return
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
            task = loop.create_task(self._make(engine, sentence, voice, target, self._prepared_dir))
            task.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._prepared[key] = task
            while len(self._prepared) > PREPARED_LIMIT:
                oldest = next(iter(self._prepared))
                self._prepared.pop(oldest).cancel()

    async def _make(self, engine: str, text: str, voice: str, target: Path, folder: Path) -> Path:
        await self.synthesize(engine, text, voice, target, folder)
        keep_tail = None if ends_sentence(text) else KEEP_CLAUSE_SECONDS
        try:
            await asyncio.to_thread(trim_silence, target, KEEP_LEAD_SECONDS, keep_tail)
        except Exception:
            logger.debug("Could not trim the silence around a synthesised piece; playing it as made", exc_info=True)
        return target

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
    _manager.free_memory = lambda: gpu_arbiter.probe_host_memory()[0]
    engine = str(getattr(voice_config, "tts_engine", ""))
    if engine in LOCAL_ENGINES:
        # The heads this voice says most, ready to play; entries made with another voice or settings are dropped.
        if _manager.head_cache is None:
            _manager.head_cache = HeadCache()
        with contextlib.suppress(Exception):
            _manager.head_cache.keep_only(HeadCache.signature(engine, str(getattr(voice_config, "tts_voice", ""))))
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


def prepare_local_speech(voice_config: Any, text: str) -> None:
    """Synthesise ``text`` ahead of time with the configured local voice (no-op for the system voice), so the
    speak() that follows plays it without waiting. Call it for the next sentence while the current one plays."""
    _manager.prepare(str(getattr(voice_config, "tts_engine", "")), text, str(getattr(voice_config, "tts_voice", "")))


def prewarm_local_voice(voice_config: Any) -> None:
    """The person is talking to Heliox: load the configured local voice now. Safe to call often; never blocks."""
    _manager.prewarm(str(getattr(voice_config, "tts_engine", "")), str(getattr(voice_config, "tts_voice", "")))


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
