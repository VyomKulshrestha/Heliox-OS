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
  retrying the local voice in the background after ``RETRY_AFTER_SECONDS``.
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
from pathlib import Path
from typing import Any

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

# ── Worker side ──────────────────────────────────────────────────────


def _engine_module(engine: str) -> Any:
    if engine == "kokoro_tts":
        from pilot.system import kokoro_tts

        return kokoro_tts
    if engine == "pocket_tts":
        from pilot.system import pocket_tts

        return pocket_tts
    raise ValueError(f"Unsupported local TTS engine: {engine}")  # pragma: no cover - argparse enforces the choices


async def _synthesize(engine: str, text: str, voice: str, output_file: str) -> None:
    await _engine_module(engine).synthesize_to_file(text, voice, output_file)


def _load(engine: str, voice: str) -> None:
    """Load the model and voice, and run one throwaway synthesis so the first real sentence is warm."""
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
            asyncio.run(_synthesize(engine, text, str(request["voice"]), str(request["output_file"])))
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
    args = parser.parse_args(argv)
    return _serve(args.engine, args.device, args.gpu_lease)


# ── Daemon side ──────────────────────────────────────────────────────


class LocalTTSUnavailable(RuntimeError):
    """The local voice cannot speak right now; the failure has already been logged once."""

    already_logged = True


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
    return merged or ([text] if text.strip() else [])


def _default_spawn(engine: str, device: str, gpu_lease: bool = False) -> Any:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [
        sys.executable,
        "-m",
        "pilot.system.local_tts_worker",
        "--device",
        device,
        *(["--gpu-lease"] if gpu_lease else []),
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
            if self.choose_device is not None:
                try:
                    device, gpu_lease = await asyncio.to_thread(self.choose_device, self.device)
                except Exception:
                    logger.debug("GPU check for the local voice failed; using the configured device", exc_info=True)
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

    async def _fail(self, process: Any, engine: str, reason: str) -> LocalTTSUnavailable:
        self._mark_failed(engine, reason)
        if self.on_gpu_fault is not None and process is self.process and self.effective_device not in {None, "cpu"}:
            # A voice worker on the GPU that died or hung counts toward switching the GPU off for the session.
            with contextlib.suppress(Exception):
                self.on_gpu_fault("voice", reason)
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
            line = await asyncio.wait_for(asyncio.shield(exchange), timeout)
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
        await self._request(
            process,
            engine,
            {"text_file": str(request_path), "voice": voice, "output_file": str(output_file)},
            timeout=LOAD_TIMEOUT_SECONDS if not self.loaded else REQUEST_TIMEOUT_SECONDS,
        )
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
                await self._speak_sentences(engine, sentences, voice, folder, play)
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
        def make(index: int) -> asyncio.Future[Path]:
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
                self.note_activity()
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
