"""Whether the person is in a call, for Heliox Live.

When the voice listener runs it already keeps ``in_call`` current (``pilot.system.call_detection``, every 2 s), and
that is the answer. Someone can use Live with voice off; then this polls the same detection on a worker thread at
most every ``EVERY_SECONDS`` and answers from the last result, so a Live tick never waits on it. The person's
``voice.quiet_in_calls`` setting turns all of it off.

What Live does in a call (``pilot.live.engine``): it says nothing, shows urgent cards only, and keeps the other
catches until the call ends; those still true then are said together, a few seconds after it ends.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("pilot.live.calls")

EVERY_SECONDS = 5.0


def _detect() -> str:
    from pilot.system import call_detection

    try:
        # Heliox is not listening here (the listener would answer itself), which older macOS needs to know.
        return str(call_detection.current_call(heliox_listening=False) or "")
    except TypeError:  # an older detection without the argument
        return str(call_detection.current_call() or "")


class CallWatch:
    def __init__(
        self,
        *,
        listener: Callable[[], Any] = lambda: None,
        enabled: Callable[[], bool] = lambda: True,
        detect: Callable[[], str] = _detect,
        clock: Callable[[], float] = time.monotonic,
        every: float = EVERY_SECONDS,
    ) -> None:
        self._listener = listener
        self._enabled = enabled
        self._detect = detect
        self._clock = clock
        self._every = every
        self._cached = ""
        self._checked_at = float("-inf")
        self._pending: Any = None

    def __call__(self) -> str:
        """The call app, or "" when not in a call (or when it cannot be told here)."""
        try:
            if not self._enabled():
                return ""
        except Exception:
            return ""
        listener = self._listener()
        if listener is not None and getattr(listener, "is_running", False):
            return str(getattr(listener, "in_call", "") or "")
        self._refresh()
        return self._cached

    def _refresh(self) -> None:
        now = self._clock()
        if now - self._checked_at < self._every or (self._pending is not None and not self._pending.done()):
            return
        self._checked_at = now
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        future = loop.run_in_executor(None, self._detect)

        def done(fut: Any) -> None:
            if fut.cancelled():
                return
            error = fut.exception()
            if error is not None:
                logger.debug("Call detection failed: %s", error)
                return
            self._cached = str(fut.result() or "")

        future.add_done_callback(done)
        self._pending = future


__all__ = ["CallWatch"]
