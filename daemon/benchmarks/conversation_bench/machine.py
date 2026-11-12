"""Is the machine quiet enough to time anything? (psutil when present; otherwise unknown, treated as busy.)

Timings taken while other programs hold the CPU measure the other programs. A run checks the load first; timing
lines measured on a busy machine are printed as owed, never as numbers. Lines that do not depend on time (found or
not, split or not, addressed or not, word error rate) stay valid under load.
"""

from __future__ import annotations

import time
from collections.abc import Callable

BUSY_PERCENT = 50.0
TIMING_LINES = frozenset({"1.1a", "1.1b", "1.1c", "1.1d", "1.2b", "4.1"})


def cpu_percent(seconds: float = 5.0) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.cpu_percent(interval=seconds))


def is_busy(percent: float | None, threshold: float = BUSY_PERCENT) -> bool:
    """Unknown load counts as busy: a timing nobody can vouch for is not quoted."""
    return percent is None or percent >= threshold


def wait_for_quiet(
    max_wait_s: float,
    *,
    sample: Callable[[], float | None] = cpu_percent,
    sleep: Callable[[float], None] = time.sleep,
    threshold: float = BUSY_PERCENT,
) -> tuple[bool, float | None]:
    """Wait up to ``max_wait_s`` for the load to drop under ``threshold``; returns (quiet, last load)."""
    waited = 0.0
    while True:
        load = sample()
        if not is_busy(load, threshold):
            return True, load
        if waited >= max_wait_s:
            return False, load
        sleep(30.0)
        waited += 30.0
