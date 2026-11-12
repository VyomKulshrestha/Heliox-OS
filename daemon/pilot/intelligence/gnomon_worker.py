"""Gnomon in its own process, so a GPU fault costs a review and never the daemon.

On 10 November 2026 at 11:57 the daemon was loading Gnomon in-process when bitsandbytes hit "an illegal memory
access" on the overheating RTX 4060; its CUDA error handler ended the whole daemon. CUDA errors are not exceptions
Python can always catch (bitsandbytes calls ``exit``, a GPU reset leaves a "sticky" context that fails every later
call), so the only safe place for the model is a process that may die.

The worker (``python -m pilot.intelligence.gnomon_worker --serve``) loads one typed-decision engine and answers one
JSON line per request on its own copy of stdout; everything libraries print goes to stderr, which the daemon keeps
in ``gnomon_worker.log``. Before loading it takes the GPU lease (``pilot.system.gpu_arbiter``) and keeps it while
the model is loaded; the operating system releases it when the worker exits.

``GnomonWorkerEngine`` is the daemon side. It has the engine interface ``GnomonReviewer`` already uses (``load``,
``score``, ``score_questions``, ``unload``), so the reviewer's warm-up, time limits, resource retry and fail-closed
answers are unchanged. It adds supervision: a worker that exits, hangs past its limit, answers garbage or reports a
CUDA driver error is killed and reported as a ``GnomonWorkerFault``; the reviewer then answers "not reviewed" and
restarts the worker after a back-off. Faults are counted by the GPU arbiter, which switches the GPU off for the
session after repeated faults in a short time (default 3 in 10 minutes).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger("pilot.intelligence.gnomon_worker")

GIB = 1024**3
# The loaders refuse to start with less than 4 GiB of free CUDA memory; the arbiter adds its headroom to this.
GNOMON_VRAM_BYTES = int(3.5 * GIB)
LOAD_TIMEOUT_SECONDS = 240.0  # import torch + transformers, read the weights, first kernels: ~30-60 s here
SCORE_TIMEOUT_SECONDS = 15.0  # warm answers take 0.2-0.6 s; this long means the worker is stuck
START_TIMEOUT_SECONDS = 60.0
RESTART_BASE_SECONDS = 5.0
RESTART_MAX_SECONDS = 300.0
GPU_WAIT_RETRY_SECONDS = 60.0  # hot or busy card: look again after this long
STDERR_TAIL_CHARS = 600
# Same rule as typed_decision.require_available_host_memory, which the worker still applies itself. It is an absolute
# amount (the worker's own peak while importing torch and reading the weights), so it is the same on an 8 GB and a
# 32 GB machine: on 8 GB it is usually what stops Gnomon, on 32 GB it rarely matters.
HOST_RAM_NEED_BYTES = int(2.5 * GIB)
NO_CUDA_REASON = (
    "Not supported on this system: Gnomon's 4-bit model needs an NVIDIA graphics card with CUDA, and none was found."
)
PARENT_POLL_SECONDS = 1.0


# ── Worker side ──────────────────────────────────────────────────────────────


def build_engine(spec: dict[str, Any]) -> Any:
    """The same engine the daemon used to build in-process (``PilotServer._typed_decision_model``)."""
    if spec.get("scoring_mode") == "native_next_token":
        from pilot.intelligence.native_typed_decision import LocalNextTokenTypedDecisionModel

        return LocalNextTokenTypedDecisionModel(
            str(spec.get("model_id") or ""),
            max_length=int(spec.get("max_length") or 2048),
            temperature=float(spec.get("temperature") or 1.0),
        )
    from pilot.intelligence.typed_decision import LocalTypedDecisionModel

    return LocalTypedDecisionModel(str(spec.get("artifact_dir") or ""), str(spec.get("report_path") or ""))


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def serve(
    requests: IO[str],
    replies: IO[str],
    *,
    engine_factory: Callable[[dict[str, Any]], Any] = build_engine,
    lease_factory: Callable[[], Any] | None = None,
) -> int:
    """Answer requests until stdin closes. Every request gets exactly one reply line with the same ``id``."""
    if lease_factory is None:
        from pilot.system.gpu_arbiter import GpuLease

        lease_factory = GpuLease
    engine: Any = None
    lease: Any = None

    def reply(payload: dict[str, Any]) -> None:
        replies.write(json.dumps(payload) + "\n")
        replies.flush()

    for line in requests:
        if not line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            op = request.get("op")
            if op == "ping":
                reply({"id": request_id, "status": "ok", "pid": os.getpid(), "rss_bytes": _rss_bytes()})
            elif op == "load":
                if lease is None:
                    candidate = lease_factory()
                    if not candidate.try_acquire("gnomon"):
                        reply({"id": request_id, "status": "error", "code": "gpu_busy",
                               "error": "Another Heliox process is using the graphics card."})  # fmt: skip
                        continue
                    lease = candidate
                started = time.monotonic()
                engine = engine_factory(dict(request.get("spec") or {}))
                engine.load()
                reply({"id": request_id, "status": "ok", "load_seconds": round(time.monotonic() - started, 2),
                       "rss_bytes": _rss_bytes()})  # fmt: skip
            elif op == "score":
                if engine is None:
                    raise RuntimeError("Gnomon is not loaded in this worker")
                result = engine.score(request["state"], request["kind"], request["question"], request["options"])
                reply({"id": request_id, "status": "ok", "result": result})
            elif op == "score_questions":
                if engine is None:
                    raise RuntimeError("Gnomon is not loaded in this worker")
                reply({"id": request_id, "status": "ok", "result": engine.score_questions(request["state"],
                       request["questions"])})  # fmt: skip
            else:
                raise ValueError(f"unknown request {op!r}")
        except Exception as error:
            from pilot.intelligence.gnomon_reviewer import is_driver_error

            text = str(error)[:500] or type(error).__name__
            reply({"id": request_id, "status": "error", "error": text, "type": type(error).__name__,
                   "driver": is_driver_error(error)})  # fmt: skip
    if lease is not None:
        lease.release()
    return 0


def watch_parent(
    parent_pid: int, on_gone: Callable[[], None], *, poll: float = PARENT_POLL_SECONDS
) -> threading.Thread:
    """End the worker when the daemon is gone, even mid-load, so a crashed daemon leaves no worker holding the GPU.

    Normally the daemon's exit closes the worker's stdin and ``serve`` returns; this covers a worker busy in a long
    load or a stuck CUDA call. Windows: wait on a handle to the parent process (immune to process-id reuse). macOS
    and Linux: when the parent dies the worker is re-parented, so its parent id changes (checked every second); for
    a process that is not our parent, signal 0 says whether it still exists.
    """
    initial_parent = os.getppid()

    def windows_wait() -> None:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE
        if not handle:
            return  # already gone
        try:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        finally:
            kernel32.CloseHandle(handle)

    def posix_alive() -> bool:
        if parent_pid == initial_parent:
            return os.getppid() == parent_pid
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def run() -> None:
        if sys.platform == "win32":
            windows_wait()
        else:
            while posix_alive():
                time.sleep(poll)
        on_gone()

    thread = threading.Thread(target=run, name="gnomon-parent-watch", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Gnomon reviews in a separate process")
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    if args.parent_pid > 0:
        watch_parent(args.parent_pid, lambda: os._exit(0))
    # The protocol gets its own copy of stdout; anything a library prints to stdout goes to stderr instead.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return serve(sys.stdin, protocol)


# ── Daemon side ──────────────────────────────────────────────────────────────


class GnomonWorkerFault(RuntimeError):
    """The worker process died, hung, answered garbage or reported a CUDA driver error. It has been killed."""

    worker_fault = True

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind  # "start" | "exit" | "hang" | "garbage" | "driver"


class GpuUnavailableError(RuntimeError):
    """The arbiter said no: the card is hot, busy, unreadable or switched off for the session."""

    gpu_unavailable = True

    def __init__(self, code: str, message: str, retry_seconds: float | None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_seconds = retry_seconds  # None: not for the rest of this session


def worker_log_path() -> Path:
    from pilot.config import DATA_DIR

    return DATA_DIR / "logs" / "gnomon_worker.log"


def worker_command(parent_pid: int) -> list[str]:
    """``python -m pilot.intelligence.gnomon_worker``: the daemon's own interpreter, in a source checkout and in the
    installed app (a virtual environment with the daemon package installed) alike."""
    return [sys.executable, "-m", "pilot.intelligence.gnomon_worker", "--serve", "--parent-pid", str(parent_pid)]


def worker_environment(gpu_index: int | None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The daemon's environment, with the package importable from wherever it lives and, on a machine with several
    NVIDIA cards, only the card the arbiter chose visible to the worker."""
    env = dict(os.environ if base is None else base)
    package_root = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_root + (os.pathsep + existing if existing else "")
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    return env


WORKER_LOG_MAX_BYTES = 5 * 1024 * 1024


def rotate_worker_log(path: Path, max_bytes: int = WORKER_LOG_MAX_BYTES) -> None:
    """Before each start: a log past 5 MB becomes gnomon_worker.log.1 (replacing the older one), so the worker's log
    holds at most 10 MB however often it restarts."""
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def _default_spawn(gpu_index: int | None = None) -> Any:
    """A fresh interpreter (fork+exec on macOS and Linux, CreateProcess on Windows): nothing of the daemon's threads,
    locks or CUDA state is inherited, unlike multiprocessing's fork. Pipes work the same on all three systems."""
    if getattr(sys, "frozen", False):
        raise RuntimeError("a frozen build has no Python interpreter to run the worker")
    log_path = worker_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotate_worker_log(log_path)
    stderr = open(log_path, "ab")  # noqa: SIM115 - the child owns the handle for its lifetime
    stderr.write(f"\n--- Gnomon worker started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
    stderr.flush()
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True  # a Ctrl+C in the daemon's terminal is the daemon's to handle
    try:
        return subprocess.Popen(
            worker_command(os.getpid()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=worker_environment(gpu_index),
            close_fds=True,
            **options,
        )
    finally:
        stderr.close()


def _stderr_tail() -> str:
    """The last words of the worker's log, e.g. "Error an illegal memory access was encountered at line 81"."""
    try:
        with open(worker_log_path(), "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 4000))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("--- Gnomon")]
    return " | ".join(lines[-3:])[-STDERR_TAIL_CHARS:]


class GnomonWorkerEngine:
    """The typed-decision engine interface, answered by a supervised worker process.

    Methods are blocking and thread-safe (the reviewer calls them through ``asyncio.to_thread``); one request is on
    the pipe at a time.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        spawn: Callable[[int | None], Any] = _default_spawn,
        arbiter: Any = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        stderr_tail: Callable[[], str] = _stderr_tail,
        load_timeout: float = LOAD_TIMEOUT_SECONDS,
        score_timeout: float = SCORE_TIMEOUT_SECONDS,
        start_timeout: float = START_TIMEOUT_SECONDS,
        need_bytes: int = GNOMON_VRAM_BYTES,
    ) -> None:
        self.spec = dict(spec)
        self._spawn = spawn
        if arbiter is None:
            from pilot.system.gpu_arbiter import arbiter as shared_arbiter

            arbiter = shared_arbiter()
        self._arbiter = arbiter
        self._clock = clock
        self._wall = wall
        self._stderr_tail = stderr_tail
        self._load_timeout = load_timeout
        self._score_timeout = score_timeout
        self._start_timeout = start_timeout
        self._need_bytes = need_bytes
        self._lock = threading.RLock()
        self._process: Any = None
        self._replies: queue.Queue[bytes] | None = None
        self._next_id = 0
        self._loaded = False
        self._recent_faults: list[float] = []
        self.last_fault: dict[str, Any] | None = None
        # A fault whose caller had already given up (the reviewer's own time limit ran out first); status() picks
        # it up through note_exit().
        self._unreported: GnomonWorkerFault | None = None
        self.starts = 0
        self.start_seconds: float | None = None  # process start to first answer
        self.load_seconds: float | None = None
        self.worker_rss_bytes: int | None = None
        self.worker_pid: int | None = None

    # Status --------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded and self.alive

    @property
    def alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def gpu_disabled(self) -> bool:
        return bool(self._arbiter.disabled)

    @property
    def gpu_disabled_reason(self) -> str:
        return str(self._arbiter.disabled_reason)

    def note_exit(self) -> GnomonWorkerFault | None:
        """A loaded worker that is no longer running (killed, crashed while idle): record the fault once.

        Called from the event loop by status(), so it never waits for a request in progress.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._loaded and self._process is not None and not self.alive:
                self._fault("exit", self._exit_message())
            fault, self._unreported = self._unreported, None
            return fault
        finally:
            self._lock.release()

    def take_fault(self) -> None:
        """The caller has handled the fault it was raised; status() must not report it a second time."""
        self._unreported = None

    def restart_delay(self) -> float:
        """Back-off before the next start: 5 s after one fault, doubling per recent fault, at most 5 minutes."""
        now = self._clock()
        window = float(getattr(self._arbiter, "fault_window_seconds", 600.0))
        recent = [at for at in self._recent_faults if now - at <= window]
        if not recent:
            return 0.0
        return min(RESTART_BASE_SECONDS * 2 ** (len(recent) - 1), RESTART_MAX_SECONDS)

    def worker_status(self) -> dict[str, Any]:
        process = self._process
        return {
            "process": "separate",
            "pid": (self.worker_pid or getattr(process, "pid", None)) if self.alive else None,
            "running": self.alive,
            "device": "cuda" if self.loaded else None,
            "starts": self.starts,
            "start_seconds": self.start_seconds,
            "load_seconds": self.load_seconds,
            "rss_mb": None if self.worker_rss_bytes is None else round(self.worker_rss_bytes / 1024**2),
            "last_fault": self.last_fault,
        }

    # Engine interface ----------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if self._arbiter.disabled:
                raise GpuUnavailableError("disabled", self._arbiter.disabled_reason, None)
            # Our own previous worker may still hold the lease while it exits; it is closed first.
            self._close()
            self._unreported = None
            from pilot.system.gpu_arbiter import Needs

            decision = self._arbiter.check(
                "gnomon", Needs(gpu_bytes=self._need_bytes, host_bytes=HOST_RAM_NEED_BYTES, requires_cuda=True)
            )
            # Memory: the reviewer's resource retry already waits for it; same messages as the loader's own guards.
            if decision.code == "low_host_memory":
                raise RuntimeError(
                    f"At least 2.5 GiB of available system RAM is required before local model loading ({decision.reason})"
                )
            if decision.code == "low_memory":
                raise RuntimeError(
                    f"At least 4 GiB of free CUDA memory is required for local model scoring ({decision.reason})"
                )
            if decision.code == "unsupported":
                raise GpuUnavailableError("unsupported", decision.reason, None)
            if decision.blocked:
                raise GpuUnavailableError(decision.code, decision.reason, GPU_WAIT_RETRY_SECONDS)
            # "ok" on an NVIDIA card, or "no_gpu" (no nvidia-smi: cannot tell, so the loader finds out).
            self._start(decision.gpu_index)
            reply = self._request({"op": "load", "spec": self.spec}, self._load_timeout)
            if reply.get("status") != "ok":
                self._close()
                error = str(reply.get("error") or "Gnomon could not load")
                if reply.get("code") == "gpu_busy":
                    raise GpuUnavailableError("busy", error, GPU_WAIT_RETRY_SECONDS)
                if reply.get("driver"):
                    raise self._fault("driver", error)
                if "cuda is required" in error.lower():
                    raise GpuUnavailableError("unsupported", NO_CUDA_REASON, None)
                raise RuntimeError(error)
            self.load_seconds = reply.get("load_seconds")
            self.worker_rss_bytes = reply.get("rss_bytes")
            self._loaded = True

    def score(self, state: Any, kind: str, question: str, options: list[str]) -> dict[str, Any]:
        return self._answer({"op": "score", "state": state, "kind": kind, "question": question, "options": options})

    def score_questions(self, state: Any, questions: Any) -> dict[str, Any]:
        return self._answer({"op": "score_questions", "state": state, "questions": questions})

    def unload(self) -> None:
        """Ending the process frees its memory, its CUDA context and the GPU lease in one step."""
        with self._lock:
            self._close()

    # Internals -----------------------------------------------------------------

    def _answer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self.loaded:
                if self._loaded and not self.alive:  # killed or crashed while idle
                    raise self._fault("exit", self._exit_message())
                if self._unreported is not None:  # faulted after its caller stopped waiting
                    raise self._unreported
                raise RuntimeError("Gnomon is not loaded")
            reply = self._request(request, self._score_timeout)
            if reply.get("status") != "ok":
                if reply.get("driver"):
                    raise self._fault("driver", str(reply.get("error") or "CUDA driver error"))
                raise RuntimeError(str(reply.get("error") or "Gnomon could not answer"))
            result = reply.get("result")
            if not isinstance(result, dict):
                raise self._fault("garbage", "The review model's process answered without a result.")
            return result

    def _start(self, gpu_index: int | None = None) -> None:
        started = self._clock()
        try:
            process = self._spawn(gpu_index)
        except Exception as error:
            raise self._fault("start", f"The review model's process could not start: {error}") from error
        self._process = process
        self._replies = queue.Queue()
        self.starts += 1
        reader = threading.Thread(
            target=_read_lines, args=(process.stdout, self._replies), name="gnomon-worker-reader", daemon=True
        )
        reader.start()
        reply = self._request({"op": "ping"}, self._start_timeout)
        if reply.get("status") != "ok":
            raise self._fault("start", "The review model's process did not start correctly.")
        self.start_seconds = round(self._clock() - started, 2)
        self.worker_rss_bytes = reply.get("rss_bytes")
        self.worker_pid = reply.get("pid")  # the interpreter itself (behind a venv launcher on Windows)

    def _request(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        process, replies = self._process, self._replies
        if process is None or replies is None:
            raise self._fault("exit", "The review model's process is not running.")
        self._next_id += 1
        request_id = self._next_id
        try:
            process.stdin.write((json.dumps({**request, "id": request_id}) + "\n").encode("utf-8"))
            process.stdin.flush()
        except (OSError, ValueError, AttributeError) as error:
            raise self._fault("exit", self._exit_message(f"could not send the request ({error})")) from error
        try:
            line = replies.get(timeout=timeout)
        except queue.Empty:
            raise self._fault("hang", f"The review model's process did not answer within {timeout:.0f} s.") from None
        if not line:
            raise self._fault("exit", self._exit_message())
        try:
            reply = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise self._fault("garbage", f"The review model's process answered garbage: {line[:80]!r}") from None
        if not isinstance(reply, dict) or reply.get("id") != request_id:
            raise self._fault("garbage", "The review model's process answered a different question.")
        return reply

    def _exit_message(self, detail: str = "") -> str:
        process = self._process
        code = None
        if process is not None:
            try:  # its stdout closes a moment before the exit code is there
                code = process.wait(timeout=2.0)
            except Exception:
                code = process.poll()
        message = "The review model's process stopped"
        if code is not None:
            message += f" (exit code {code})"
        if detail:
            message += f": {detail}"
        tail = self._stderr_tail()
        return f"{message}. Last output: {tail}" if tail else f"{message}."

    def _fault(self, kind: str, message: str) -> GnomonWorkerFault:
        """Kill the worker, count the fault with the arbiter, and return the exception to raise."""
        self._close()
        now = self._clock()
        self._recent_faults = [at for at in self._recent_faults if now - at <= 3600.0] + [now]
        self.last_fault = {
            "kind": kind,
            "message": message[:400],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._wall())),
        }
        logger.warning("Gnomon worker fault (%s): %s", kind, message[:400])
        if kind != "start":  # a process that never ran tells nothing about the card
            self._arbiter.record_fault("gnomon", f"{kind}: {message}")
        fault = GnomonWorkerFault(kind, message)
        self._unreported = fault
        return fault

    def _close(self) -> None:
        process, self._process = self._process, None
        self._replies = None
        self._loaded = False
        if process is None:
            return
        with contextlib.suppress(Exception):
            if process.poll() is None:
                _kill_tree(process)
            process.wait(timeout=5.0)
        for stream in (getattr(process, "stdin", None), getattr(process, "stdout", None)):
            with contextlib.suppress(Exception):
                if stream is not None:
                    stream.close()


def _kill_tree(process: Any) -> None:
    """Kill the worker and anything it started. On Windows a virtual environment's python.exe is a launcher that
    runs the real interpreter as its child; killing only the launcher would leave the model running, holding the
    pipes and the GPU."""
    try:
        import psutil

        children = psutil.Process(process.pid).children(recursive=True) if isinstance(process, subprocess.Popen) else []
    except Exception:
        children = []
    for child in children:
        with contextlib.suppress(Exception):
            child.kill()
    process.kill()


def _read_lines(stream: Any, replies: queue.Queue[bytes]) -> None:
    """Hand each reply line to the waiting request; an empty line means the worker's stdout closed."""
    try:
        for line in iter(stream.readline, b""):
            replies.put(line)
    except (OSError, ValueError):
        pass
    replies.put(b"")


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
