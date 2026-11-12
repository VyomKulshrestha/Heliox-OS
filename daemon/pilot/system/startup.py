"""Start-up readiness: what the daemon is doing while it starts, and holding requests until it can answer them.

On a slow or loaded machine start-up takes minutes (importing the model stack, opening the stores, loading speech
models). The window used to see nothing but a refused connection ("Reconnecting...") all that time. Now the daemon
listens first, answers ``ping``, ``health``, ``ready`` and ``startup_status`` at once, and tells every window which
stage it is in ("starting: opening Heliox's records"). Anything else the person asks while it starts waits until
start-up is done and is then answered as usual; nothing is refused with an error that only means "not ready yet".
If start-up fails, the waiting requests get that failure in plain words.

``StartupProgress`` is pure bookkeeping with an injected clock; the server reports stages to it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("pilot.system.startup")

# How long a request waits for start-up before it is answered with the reason start-up has not finished.
WAIT_FOR_READY_SECONDS = 600.0


class StartupProgress:
    """starting -> (waiting_for_storage ->) ready, or failed. Stages are said in plain words."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self.state = "starting"
        self.stage = "starting"
        self.reason = ""
        self.stages: list[tuple[str, float]] = []  # (words, seconds since start) for the log and the bench
        self._ready = asyncio.Event()
        self.on_change: Callable[[dict[str, Any]], None] | None = None

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"

    def begin(self, words: str) -> None:
        """A new stage starts, e.g. "opening Heliox's records"."""
        elapsed = self._clock() - self._started
        self.stages.append((words, elapsed))
        self.stage = words
        logger.info("Start-up: %s (%.1f s since start)", words, elapsed)
        self._changed()

    def waiting_for_storage(self, reason: str) -> None:
        self.state = "waiting_for_storage"
        self.reason = reason
        self.stage = "waiting until the data folder can be written"
        self._changed()

    def ready(self) -> None:
        self.state = "ready"
        self.stage = ""
        self.reason = ""
        self.stages.append(("ready", self._clock() - self._started))
        logger.info("Start-up finished in %.1f s", self._clock() - self._started)
        self._event().set()
        self._changed()

    def failed(self, reason: str) -> None:
        self.state = "failed"
        self.reason = reason
        self._event().set()
        self._changed()

    async def wait_ready(self, timeout: float = WAIT_FOR_READY_SECONDS) -> str:
        """Wait until start-up has finished. Returns "" when ready, else the plain reason it is not."""
        if self.state == "ready":
            return ""
        if self.state != "failed":
            try:
                await asyncio.wait_for(self._event().wait(), timeout=timeout)
            except TimeoutError:
                return f"Heliox is still starting ({self.stage}). Try again in a moment."
        if self.state == "failed":
            return f"Heliox could not finish starting: {self.reason}"
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "stage": self.stage,
            "reason": self.reason,
            "stages_done": [words for words, _ in self.stages],
        }

    def _event(self) -> asyncio.Event:
        return self._ready

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change(self.snapshot())
            except Exception:
                logger.debug("Could not announce the start-up stage", exc_info=True)
