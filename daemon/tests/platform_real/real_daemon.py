"""A real daemon for the fault checks on CI runners: start ``python -m pilot.server``, talk to it as the window does,
stop it cleanly. Used by storage_faults_check.py, keychain_locked_check.py and soak_check.py.

The daemon's folders follow the environment (HOME, or the XDG_* variables), so each check sets them before it calls
``start`` and before it imports ``pilot.config``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PORT = 8785
CLOSED_PORT = 9  # discard: nothing listens there on a runner, so a model call is refused at once and costs nothing


def write_config(extra: str = "") -> None:
    """A finished first run whose model is a local Ollama address that refuses at once: no model, no network, no cost."""
    from pilot.config import CONFIG_FILE

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        'first_run_complete = true\n\n[model]\nprovider = "ollama"\n'
        f'ollama_base_url = "http://127.0.0.1:{CLOSED_PORT}"\n{extra}',
        encoding="utf-8",
    )


_output: list[Path] = []  # where each daemon this process started wrote its stdout and stderr


def start(env: dict[str, str] | None = None) -> subprocess.Popen:
    """``python -m pilot.server``; its stdout and stderr go to a file (``output_tail``), so a daemon that ends before
    it writes its own log (or before it listens) still says why."""
    import tempfile

    folder = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    path = folder / f"heliox-daemon-output-{os.getpid()}-{len(_output)}.log"
    _output.append(path)
    with open(path, "wb") as sink:  # the child keeps its own copy of the handle
        return subprocess.Popen(
            [sys.executable, "-m", "pilot.server"],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )


def output_tail(lines: int = 40) -> list[str]:
    """The last lines the most recently started daemon printed (a traceback before its log exists, for one)."""
    if not _output:
        return []
    try:
        text = _output[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.rstrip()[-240:] for line in text[-lines:]]


class Session:
    """One authenticated connection; ``call`` skips notifications and returns the reply's result."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self._next = 100

    async def call(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self._next += 1
        request_id = self._next
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}, "id": request_id}))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no reply to {method} in {timeout:g} s")
            message = json.loads(await asyncio.wait_for(self.ws.recv(), remaining))
            if message.get("id") == request_id:
                if "error" in message:
                    return {"rpc_error": message["error"]}
                result = message.get("result")
                return result if isinstance(result, dict) else {"value": result}

    async def ask(self, text: str, timeout: float = 120.0) -> str:
        """A typed request, as the window sends it; the sentence Heliox answered with."""
        reply = await self.call("execute", {"input": text}, timeout)
        error = reply.get("rpc_error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(reply.get("explanation") or reply.get("message") or reply)

    async def timed(self, method: str, params: dict | None = None, timeout: float = 60.0) -> tuple[dict, float]:
        began = time.monotonic()
        reply = await self.call(method, params, timeout)
        return reply, time.monotonic() - began

    async def wait_ready(self, limit: float) -> tuple[bool, dict]:
        deadline = time.monotonic() + limit
        status: dict = {}
        while time.monotonic() < deadline:
            status = await self.call("startup_status", timeout=30)
            if status.get("state") == "ready":
                return True, status
            await asyncio.sleep(1)
        return False, status

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:
            pass


async def connect(limit: float = 240.0) -> Session:
    """Wait for the token file and the listening daemon, then authenticate with the token on disk."""
    import websockets

    from pilot.config import RUNTIME_DIR

    token_file = RUNTIME_DIR / "auth_token"
    deadline = time.monotonic() + limit
    last = ""
    while time.monotonic() < deadline:
        if token_file.exists():
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{PORT}", open_timeout=5, max_size=None)
                session = Session(ws)
                token = token_file.read_text(encoding="utf-8").strip()
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": token}, "id": 1}))
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    if message.get("id") == 1:
                        break
                if "error" in message:
                    await session.close()
                    last = str(message["error"])
                else:
                    return session
            except Exception as error:
                last = f"{type(error).__name__}: {error}"
        await asyncio.sleep(1)
    raise TimeoutError(f"the daemon did not answer in {limit:g} s ({last or 'no token file'})")


def serving_process(popen: subprocess.Popen, mark: str = "pilot.server") -> Any:
    """The interpreter that serves, for a process a check started with ``sys.executable``. On Windows a virtual
    environment's python.exe is a launcher and the interpreter is its child (same arguments): killing or freezing the
    launcher alone leaves the daemon serving. Elsewhere, and with a plain interpreter, it is the Popen's own process."""
    import psutil

    root = psutil.Process(popen.pid)
    for child in root.children():
        try:
            if child.name().lower().startswith("python") and any(mark in part for part in child.cmdline()):
                return child
        except psutil.Error:
            continue
    return root


def kill_serving(popen: subprocess.Popen, mark: str = "pilot.server") -> None:
    """A hard kill of the serving interpreter only (SIGKILL; TerminateProcess on Windows), as a crash ends it: the
    workers it started stay for their own watch to end. The launcher, if any, then ends with its child."""
    import psutil

    try:
        serving_process(popen, mark).kill()
    except psutil.Error:
        pass
    popen.wait(30)


def end_tree(popen: subprocess.Popen) -> None:
    """Clean-up: the process and everything under it, launcher, interpreter and workers alike."""
    import psutil

    try:
        root = psutil.Process(popen.pid)
        processes = [*root.children(recursive=True), root]
    except psutil.Error:
        processes = []
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    try:
        popen.wait(30)
    except subprocess.TimeoutExpired:
        pass


async def stop(daemon: subprocess.Popen, session: Session | None, limit: float = 40.0) -> None:
    """Ask for the clean stop (``daemon_stop``, the same as SIGTERM, on every system), then make sure it is gone."""
    if daemon.poll() is None and session is not None:
        try:
            await session.call("daemon_stop", {"reason": "check finished"}, timeout=15)
        except Exception:
            pass
    if session is not None:
        await session.close()
    deadline = time.monotonic() + limit
    while daemon.poll() is None and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    if daemon.poll() is None:
        end_tree(daemon)


def log_lines(marks: tuple[str, ...], last: int = 20) -> list[str]:
    from pilot.config import LOG_FILE

    try:
        text = Path(LOG_FILE).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.strip()[-240:] for line in text if any(mark in line for mark in marks)][-last:]
