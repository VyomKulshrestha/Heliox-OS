"""Learned Risk Gate — Layer 1: Observation.

Captures a snapshot of real OS state via psutil — process count, disk
usage, memory usage — the same kind of telemetry Ferrum-OS's world model
reads through its own kernel syscalls (see cognitive/world_model/
observation.rs), but through real cross-platform APIs since Heliox runs
on the user's actual OS, not a purpose-built kernel.

Pure data capture, no risk judgment here — see risk_safety.py for the
hardcoded, auditable rules that actually score a predicted outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("pilot.security.risk_observation")

# Normalizes proc_count into roughly [0, 1] the same way Ferrum's encoder.rs
# divides by NOMINAL_PROC_CAPACITY — a nominal "typical load" ceiling, not a
# hard limit. Approximate; revisit if real usage shows most snapshots
# clustering at the boundary instead of spread across the range.
NOMINAL_PROC_CAPACITY = 300.0


@dataclass(frozen=True)
class OsSnapshot:
    """A point-in-time real OS state reading."""

    proc_count: int
    disk_usage_fraction: float  # 0-1, of the volume containing `disk_path`
    memory_usage_fraction: float  # 0-1
    disk_path: str

    @property
    def proc_count_normalized(self) -> float:
        return min(1.0, self.proc_count / NOMINAL_PROC_CAPACITY)


def capture_os_snapshot(disk_path: str | None = None) -> OsSnapshot:
    """Captures a real OsSnapshot via psutil. `disk_path` scopes the disk
    reading to the volume a proposed action actually targets (e.g. a
    sandbox temp dir during training data collection, or a file action's
    own target path in production) — defaults to the OS drive root if not
    given.

    Never raises: psutil calls that fail (e.g. an unmounted/invalid path)
    fall back to 0.0 fractions rather than propagating, since observation
    failures shouldn't be able to crash whatever's asking for a risk
    prediction — see RiskGate's own graceful-degradation philosophy.
    """
    import os

    import psutil

    path = disk_path or os.path.abspath(os.sep)

    try:
        proc_count = len(psutil.pids())
    except Exception:
        logger.debug("Failed to read process count", exc_info=True)
        proc_count = 0

    try:
        # psutil's own `.percent` field is rounded to 1 decimal place --
        # coarse enough that a single small file write (a few hundred KB
        # against a multi-hundred-GB drive) always rounds away to an
        # exact-zero delta. Computing the fraction from the raw used/total
        # byte counts instead keeps full float precision, so
        # collect_risk_training_data.py's real file-write/delete samples
        # actually show a measurable (if tiny) delta.
        usage = psutil.disk_usage(path)
        disk_usage_fraction = usage.used / usage.total if usage.total else 0.0
    except Exception:
        logger.debug("Failed to read disk usage for %s", path, exc_info=True)
        disk_usage_fraction = 0.0

    try:
        memory_usage_fraction = psutil.virtual_memory().percent / 100.0
    except Exception:
        logger.debug("Failed to read memory usage", exc_info=True)
        memory_usage_fraction = 0.0

    return OsSnapshot(
        proc_count=proc_count,
        disk_usage_fraction=disk_usage_fraction,
        memory_usage_fraction=memory_usage_fraction,
        disk_path=path,
    )
