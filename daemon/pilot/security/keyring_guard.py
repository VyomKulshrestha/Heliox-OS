"""Every call into the system credential store, bounded, so a slow or blocked keychain never stops Heliox.

On macOS a keychain write from a process whose home has no login keychain (or whose keychain is locked) puts up a
dialog and the call waits for it, for ever if nobody sees it. Before this module, Heliox made such calls directly on
its event loop: the daemon froze right after it started listening, and nothing answered. Now:

- each get, set or delete runs in a daemon thread of its own and is waited for at most ``TIMEOUT_SECONDS``;
- a call that does not answer in time raises ``KeyringStalled`` (a ``KeyringError``, so existing handling of a
  failing keyring applies), and for ``COOLDOWN_SECONDS`` after that every call fails at once instead of piling up
  threads behind the stuck one;
- the first stall is said once in the log, and ``status()`` gives the sentence for the health status;
- the stuck thread is a daemon thread: it never holds up a stop.

Async callers (``KeyVault``) run these in a worker thread as well, so the event loop never waits on the keychain at
all. Synchronous callers (the recording and study key stores) wait at most ``TIMEOUT_SECONDS`` once, then fail fast.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("pilot.security.keyring_guard")

TIMEOUT_SECONDS = 3.0
COOLDOWN_SECONDS = 300.0
STALLED_SENTENCE = (
    "The system credential store (Keychain, Credential Manager or Secret Service) did not answer within {seconds:g} s, "
    "so Heliox carried on without it: saved API keys and other secrets cannot be read or saved until it does. If a "
    "keychain dialog is waiting, answer it or unlock the keychain, then restart Heliox."
)

T = TypeVar("T")

_lock = threading.Lock()
_stalled_until = 0.0
_said = False
_last_reason = ""
_clock: Callable[[], float] = time.monotonic
_loop_calls = 0


def _on_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _caller() -> str:
    import traceback

    for frame in reversed(traceback.extract_stack()[:-3]):
        if "keyring_guard" not in frame.filename:
            return f"{frame.filename}:{frame.lineno} {frame.name}"
    return "unknown"


def loop_calls() -> int:
    """How many credential-store calls ran on an event loop's own thread (tests assert none)."""
    return _loop_calls


def _keyring_errors() -> type[Exception]:
    try:
        from keyring.errors import KeyringError

        return KeyringError
    except Exception:  # keyring missing: a plain error will do
        return RuntimeError


class KeyringStalled(_keyring_errors()):  # type: ignore[misc]
    """The credential store did not answer in time (or recently did not); the caller carries on without it."""


def status() -> dict[str, Any]:
    """For the health status: whether the credential store has stalled, and what the person can do."""
    with _lock:
        stalled = _clock() < _stalled_until
        return {"state": "stalled" if stalled else "ok", "message": _last_reason if stalled else ""}


def reset() -> None:
    """Forget a stall (tests, and a restart of Heliox)."""
    global _stalled_until, _said, _last_reason, _loop_calls
    with _lock:
        _stalled_until, _said, _last_reason, _loop_calls = 0.0, False, "", 0


def call(function: Callable[..., T], *args: Any, timeout: float | None = None) -> T:
    """Run one credential-store call in a daemon thread, waiting at most ``timeout`` (default TIMEOUT_SECONDS)."""
    global _stalled_until, _said, _last_reason, _loop_calls
    limit = TIMEOUT_SECONDS if timeout is None else timeout
    if _on_event_loop():
        # Every caller should come through a worker thread; say which one did not, once per place.
        _loop_calls += 1
        logger.warning("A credential-store call ran on the event loop (%s)", _caller(), stack_info=_loop_calls == 1)
    with _lock:
        if _clock() < _stalled_until:
            raise KeyringStalled(_last_reason)
    result: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            result["value"] = function(*args)
        except BaseException as error:  # handed to the caller below
            result["error"] = error
        finally:
            done.set()

    threading.Thread(target=run, name="keyring-call", daemon=True).start()
    if not done.wait(limit):
        with _lock:
            _stalled_until = _clock() + COOLDOWN_SECONDS
            _last_reason = STALLED_SENTENCE.format(seconds=limit)
            first = not _said
            _said = True
        if first:
            logger.warning("%s", _last_reason)
        raise KeyringStalled(_last_reason)
    if "error" in result:
        raise result["error"]
    return result["value"]


def get_password(service: str, account: str, *, timeout: float | None = None) -> str | None:
    import keyring

    return call(keyring.get_password, service, account, timeout=timeout)


def set_password(service: str, account: str, secret: str, *, timeout: float | None = None) -> None:
    import keyring

    call(keyring.set_password, service, account, secret, timeout=timeout)


def delete_password(service: str, account: str, *, timeout: float | None = None) -> None:
    import keyring

    call(keyring.delete_password, service, account, timeout=timeout)
