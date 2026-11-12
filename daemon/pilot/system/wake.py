"""Noticing that the computer slept, on every system.

A laptop lid closed for an hour looks different to each clock. On Linux (CLOCK_MONOTONIC) and macOS
(mach_absolute_time), ``time.monotonic()`` stops while asleep and the wall clock jumps ahead. On Windows,
``time.monotonic()`` keeps counting through sleep, so both clocks jump together. In every case, though, a loop that
meant to run every few seconds did not run for an hour. ``WakeDetector`` notices both signs: the two clocks parting
by more than the threshold, or either one jumping far past the loop's interval. A wall clock set back (a manual
change, a time zone fix) is reported separately and is not a wake.

What depends on it (pilot.server._on_wake): any window that let speech count as addressed without the wake phrase is
closed, because on Linux and macOS a window kept in monotonic time would otherwise still be open an hour later;
reminders are checked at once, so those that came due during sleep go out together right away.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger("pilot.system.wake")

CHECK_EVERY_SECONDS = 5.0
# A gap this much longer than the check interval means the process was suspended (sleep, hibernation).
WAKE_THRESHOLD_SECONDS = 60.0


@dataclass(frozen=True)
class ClockEvent:
    kind: str  # "wake" or "wall_clock_set_back"
    seconds: float  # how long it slept, or how far the wall clock went back


class WakeDetector:
    """Compares the wall clock and the monotonic clock between checks. Pure: both clocks are injected."""

    def __init__(
        self,
        *,
        wall: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        interval: float = CHECK_EVERY_SECONDS,
        threshold: float = WAKE_THRESHOLD_SECONDS,
    ) -> None:
        self._wall = wall
        self._monotonic = monotonic
        self._interval = interval
        self._threshold = threshold
        self._last_wall = wall()
        self._last_monotonic = monotonic()

    def check(self) -> ClockEvent | None:
        wall, mono = self._wall(), self._monotonic()
        wall_step, mono_step = wall - self._last_wall, mono - self._last_monotonic
        self._last_wall, self._last_monotonic = wall, mono
        if wall_step < -self._threshold:
            return ClockEvent("wall_clock_set_back", -wall_step)
        gap = max(wall_step, mono_step) - self._interval
        if gap > self._threshold or wall_step - mono_step > self._threshold:
            return ClockEvent("wake", max(gap, wall_step - mono_step))
        return None


async def watch(
    on_event: Callable[[ClockEvent], Awaitable[None]],
    *,
    detector: WakeDetector | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Check every few seconds until cancelled; hand each wake or clock change to ``on_event``."""
    detector = detector or WakeDetector()
    while True:
        await sleep(CHECK_EVERY_SECONDS)
        event = detector.check()
        if event is None:
            continue
        logger.info("Clock event: %s (%.0f s)", event.kind, event.seconds)
        try:
            await on_event(event)
        except Exception:
            logger.warning("Handling a %s failed", event.kind, exc_info=True)
