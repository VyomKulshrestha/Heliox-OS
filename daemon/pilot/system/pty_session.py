"""Persistent PTY session management for the Terminal Agent.

Wraps ptyprocess to maintain a live bash shell across multiple commands,
preserving environment variables, working directory, and shell state.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from typing import ClassVar

logger = logging.getLogger("pilot.system.pty_session")

_SENTINEL = "PILOT_PTY_DONE"
_INIT_MARKER = "PILOT_INIT_READY"

# One compound command so only ONE line is echoed by the terminal before
# stty -echo takes effect.  Both the sentinel and the init marker are
# assembled at runtime from two shell variables so neither string appears
# as a contiguous literal in the echoed command line — preventing false-
# positive matches inside _read_until before the actual output arrives.
#
# Echoed line will contain:  _P1=PILOT_; _P2=PTY_DONE  and  _A=PILOT_IN; _B=IT_READY
# — neither "PILOT_PTY_DONE" nor "PILOT_INIT_READY" appears there.
_INIT_CMD = (
    "stty -echo; "
    '_P1=PILOT_; _P2=PTY_DONE; export PS1="${_P1}${_P2} "; '
    "export VIRTUAL_ENV_DISABLE_PROMPT=1; "
    '_A=PILOT_IN; _B=IT_READY; echo "${_A}${_B}"\n'
)


class PtySession:
    """A single persistent bash PTY session."""

    def __init__(self) -> None:
        import asyncio

        self._proc: object = None  # ptyprocess.PtyProcess
        self._lock = asyncio.Lock()
        self._alive = False

    # ------------------------------------------------------------------
    # Blocking helpers — run in a thread via run_in_executor
    # ------------------------------------------------------------------

    def _read_until(self, marker: str, timeout: float) -> tuple[str, bool]:
        """Read PTY output until marker appears or timeout expires.

        Uses select() so reads never block longer than 0.1 s at a time,
        making the overall deadline reliable without busy-looping.

        Returns (accumulated_output, found_marker).
        """
        buf = b""
        marker_b = marker.encode()
        deadline = time.monotonic() + timeout
        fd = self._proc.fd  # type: ignore[attr-defined]

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            rlist, _, _ = select.select([fd], [], [], min(remaining, 0.1))
            if not rlist:
                continue
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                self._alive = False
                break
            if chunk:
                buf += chunk
                if marker_b in buf:
                    return buf.decode("utf-8", errors="replace"), True

        return buf.decode("utf-8", errors="replace"), False

    def _spawn(self) -> None:
        import ptyprocess  # type: ignore[import-untyped]

        env = {
            **os.environ,
            "TERM": "dumb",                      # no ANSI escape sequences
            "BASH_ENV": "/dev/null",              # no ENV file
            "VIRTUAL_ENV_DISABLE_PROMPT": "1",   # venv must not prepend to PS1
        }

        self._proc = ptyprocess.PtyProcess.spawn(
            ["bash", "--norc", "--noprofile"],
            env=env,
        )
        self._alive = True

        self._proc.write(_INIT_CMD.encode())  # type: ignore[attr-defined]

        # _INIT_MARKER appears in the actual echo output but NOT in the echoed
        # command line (it's hidden behind $_M), so this won't match too early.
        raw, found = self._read_until(_INIT_MARKER, timeout=10)
        if not found:
            raise RuntimeError("PTY session init timed out")

        # Drain the sentinel prompt that follows PILOT_INIT_READY
        if _SENTINEL not in raw:
            self._read_until(_SENTINEL, timeout=5)

    def _run_command(self, command: str, timeout: float) -> str:
        self._proc.write((command + "\n").encode())  # type: ignore[attr-defined]
        raw, found = self._read_until(_SENTINEL, timeout=timeout)

        if not found:
            # Interrupt and attempt to recover the session
            try:
                self._proc.write(b"\x03\n")  # Ctrl+C
                self._read_until(_SENTINEL, timeout=3)
            except Exception:
                self._alive = False
            return f"ERROR: command timed out after {timeout:.0f}s"

        idx = raw.rfind(_SENTINEL)
        output = raw[:idx].rstrip() if idx != -1 else raw.rstrip()
        return output or "(no output)"

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def exec(self, command: str, timeout: int = 30) -> str:
        """Send a command to the shell and return its output."""
        import asyncio

        loop = asyncio.get_event_loop()

        async with self._lock:
            if not self._alive or self._proc is None or not self._proc.isalive():  # type: ignore[attr-defined]
                await loop.run_in_executor(None, self._spawn)

            def _run() -> str:
                return self._run_command(command, float(timeout))

            # run_in_executor timeout is generous — _run_command self-limits via select
            return await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=timeout + 10,
            )

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.write(b"exit\n")  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                self._proc.terminate(force=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._proc = None
            self._alive = False

    def is_alive(self) -> bool:
        return (
            self._alive
            and self._proc is not None
            and self._proc.isalive()  # type: ignore[attr-defined]
        )


class PtySessionManager:
    """Singleton registry of named PTY sessions."""

    _sessions: ClassVar[dict[str, PtySession]] = {}

    @classmethod
    def get_session(cls, session_id: str = "default") -> PtySession:
        if sys.platform == "win32":
            raise RuntimeError(
                "PTY sessions are not supported on Windows; use SHELL_COMMAND instead"
            )
        if session_id not in cls._sessions or not cls._sessions[session_id].is_alive():
            logger.info("Creating new PTY session: %s", session_id)
            cls._sessions[session_id] = PtySession()
        return cls._sessions[session_id]

    @classmethod
    def close_session(cls, session_id: str) -> None:
        session = cls._sessions.pop(session_id, None)
        if session:
            session.close()
            logger.info("Closed PTY session: %s", session_id)

    @classmethod
    def close_all(cls) -> None:
        for session_id, session in list(cls._sessions.items()):
            session.close()
            logger.info("Closed PTY session: %s", session_id)
        cls._sessions.clear()
