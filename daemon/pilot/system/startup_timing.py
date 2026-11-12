"""When start-up reached each point, measured by the daemon itself, and what was loaded by then.

Points: ``listening`` (the port accepts), ``ready`` (requests are answered), ``learner_loaded`` and ``all_started``
(every part that starts after ready has started). For each: seconds since the process was created (from the
operating system, so the interpreter's own start and the imports before ``start()`` are included), how many modules
were loaded, and which of the heavy numerical libraries were among them.

Every point is written to the log as one line ("Start-up timing ready: 9.1 s after process start, 488 modules, heavy:
none"). With ``HELIOX_STARTUP_TIMINGS=1`` they are also in ``startup_status`` under ``timings``, which the first-run CI
job reads; that field is off otherwise. The CI job fails its check if any heavy library is loaded at ``ready``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("pilot.system.startup_timing")

# Libraries that must load after ready, never before: each is hundreds of modules (see benchmarks/startup_stages.py).
HEAVY_LIBRARIES = ("scipy", "sklearn", "pandas", "river", "torch")
ENV_FLAG = "HELIOX_STARTUP_TIMINGS"


def _process_created() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return None


def heavy_loaded(modules: dict[str, Any] | None = None) -> list[str]:
    """The heavy libraries that are imported (a ``None`` entry, as tests use to block one, does not count)."""
    loaded = sys.modules if modules is None else modules
    return [name for name in HEAVY_LIBRARIES if loaded.get(name) is not None]


def packages(loaded: dict[str, Any]) -> dict[str, int]:
    """Modules loaded, counted by top-level package (Heliox's own by its subpackage, pilot.system and so on)."""
    counts: dict[str, int] = {}
    for name, module in loaded.items():
        if module is None:
            continue
        parts = name.split(".")
        key = ".".join(parts[:2]) if parts[0] == "pilot" else parts[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


class StartupTimings:
    def __init__(
        self,
        *,
        wall: Callable[[], float] = time.time,
        created: float | None = None,
        modules: Callable[[], dict[str, Any]] = lambda: sys.modules,
    ) -> None:
        self._wall = wall
        self._modules = modules
        self._created = created if created is not None else _process_created()
        self._fallback_start = wall()  # when the OS cannot say: from when timing began
        self.points: dict[str, dict[str, Any]] = {}

    def mark(self, name: str) -> dict[str, Any]:
        start = self._created if self._created is not None else self._fallback_start
        loaded = self._modules()
        point = {
            "seconds": round(self._wall() - start, 2),
            "since": "process start" if self._created is not None else "start()",
            "modules": len(loaded),
            "heavy": heavy_loaded(loaded),
        }
        if os.environ.get(ENV_FLAG) == "1":
            point["packages"] = packages(loaded)  # what the modules are, for the first-run job's comparison
        self.points[name] = point
        logger.info(
            "Start-up timing %s: %.1f s after %s, %d modules, heavy: %s",
            name,
            point["seconds"],
            point["since"],
            point["modules"],
            ", ".join(point["heavy"]) or "none",
        )
        return point

    def since_start(self, monotonic_at: float) -> float:
        """A time.monotonic() reading as seconds after the same start the points use."""
        start = self._created if self._created is not None else self._fallback_start
        return round(self._wall() - (time.monotonic() - monotonic_at) - start, 2)

    def snapshot(self) -> dict[str, dict[str, Any]] | None:
        """For startup_status: only with HELIOX_STARTUP_TIMINGS=1."""
        return dict(self.points) if os.environ.get(ENV_FLAG) == "1" else None
