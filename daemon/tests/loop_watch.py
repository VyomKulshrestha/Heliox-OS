"""Which steps hold the event loop, and for how long: asyncio's debug mode, collected per test.

Off by default. With ``HELIOX_LOOP_WATCH=1`` and ``-p loop_watch`` (the tests folder on PYTHONPATH), every event loop
runs in debug mode, each callback or task step that holds the loop longer than ``HELIOX_LOOP_WATCH_MS`` (default 100)
is recorded with the test that was running, and the run ends with one line per step, longest first:

    LOOP-WATCH 0.412 s  tests/test_x.py::test_y  <Task pending name='...' coro=<PilotServer.start() running at ...>>

asyncio names the line where the task paused next; the step that held the loop is the code just before it. The
lines are for reading, not a pass or fail: CI runs this as a non-blocking step. ``watch()`` gives a test the same
collector to assert on.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

ENV = "HELIOX_LOOP_WATCH"
THRESHOLD_ENV = "HELIOX_LOOP_WATCH_MS"
_TOOK = re.compile(r" took ([0-9.]+) seconds")


@dataclass
class SlowStep:
    seconds: float
    test: str
    step: str
    at: float  # wall clock, from the log record


class Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[SlowStep] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        found = _TOOK.search(message)
        if found:
            test = os.environ.get("PYTEST_CURRENT_TEST", "").split(" (")[0]
            self.steps.append(SlowStep(float(found.group(1)), test, message[:400], record.created))


def threshold_seconds() -> float:
    return float(os.environ.get(THRESHOLD_ENV, "100")) / 1000.0


def watch(loop, threshold: float | None = None) -> Collector:
    """Debug mode on this loop, and a collector of the steps that held it longer than ``threshold`` seconds."""
    loop.set_debug(True)
    loop.slow_callback_duration = threshold_seconds() if threshold is None else threshold
    collector = Collector()
    logger = logging.getLogger("asyncio")
    logger.addHandler(collector)
    logger.setLevel(min(logger.level or logging.WARNING, logging.WARNING))
    return collector


def unwatch(collector: Collector) -> None:
    logging.getLogger("asyncio").removeHandler(collector)


# ── as a pytest plugin (-p loop_watch) ──

_session: Collector | None = None


def pytest_configure(config) -> None:
    global _session
    if os.environ.get(ENV) != "1":
        return
    os.environ["PYTHONASYNCIODEBUG"] = "1"  # every loop created from now on runs in debug mode
    import asyncio.base_events

    original = asyncio.base_events.BaseEventLoop.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.slow_callback_duration = threshold_seconds()

    asyncio.base_events.BaseEventLoop.__init__ = init
    _session = Collector()
    logger = logging.getLogger("asyncio")
    logger.addHandler(_session)
    logger.setLevel(logging.WARNING)


def pytest_terminal_summary(terminalreporter) -> None:
    if _session is None:
        return
    steps = sorted(_session.steps, key=lambda step: step.seconds, reverse=True)
    terminalreporter.write_line(
        f"LOOP-WATCH {len(steps)} steps held the event loop longer than {threshold_seconds() * 1000:.0f} ms"
    )
    for step in steps:
        terminalreporter.write_line(f"LOOP-WATCH {step.seconds:.3f} s  {step.test}  {step.step}")
