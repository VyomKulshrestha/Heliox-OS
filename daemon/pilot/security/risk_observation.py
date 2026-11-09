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
import os
import shutil
import stat as stat_module
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

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


# ── Size observation for the disk lookahead ──
#
# The transition model has no feature for how many bytes an action writes,
# so on its own it predicts the same disk change for a 5-byte file and a
# 79 GB folder. These helpers read file-system METADATA ONLY (directory
# listings and stat sizes, never file contents) so the gate can predict the
# real disk change where the size is knowable before running.

# One budget is shared by every scan in a plan, so a plan of twenty folder
# copies still spends at most this long walking directories in total. The
# daemon runs evaluate_plan() in a worker thread after planning, which
# already takes several seconds, so 1.5 s is affordable. A caller that
# still runs it on an event loop thread gets the short budget instead.
# Measured on a cold Windows cache: about 7,000 entries per second, so the
# entry cap only binds on warm caches; the walk keeps no per-entry data
# (only the queue of unvisited folders), so memory stays flat.
PLAN_SCAN_BUDGET_SECONDS = 1.5
ON_LOOP_SCAN_BUDGET_SECONDS = 0.1
TREE_SCAN_MAX_ENTRIES = 200_000


@dataclass(frozen=True)
class TreeSize:
    """Bytes under a path, from a bounded metadata walk.

    `complete` is False when the walk stopped at its time or entry limit,
    skipped a symlink/junction, or met an unreadable entry. In that case
    `bytes` is only a lower bound ("at least this much, still counting")
    and must never be treated as the size.
    """

    bytes: int
    entries: int
    complete: bool
    stopped_early: bool = False
    files: int = 0


@dataclass(frozen=True)
class VolumeUsage:
    """Used and total bytes of the volume that holds a path."""

    key: str
    root: str
    used: int
    total: int


# A Windows reparse point whose tag has this bit set is a "name surrogate":
# it points at another place in the namespace (symlinks, junctions, volume
# mount points). Cloud-file and dedup placeholders do not set it and hold
# their data in place, so walking into them stays inside the tree.
_REPARSE_TAG_NAME_SURROGATE = 0x20000000


def _is_link_stat(st: os.stat_result) -> bool:
    """True when a stat taken WITHOUT following links describes a symlink,
    junction or other redirecting reparse point. Works the same on every
    Python version: it reads st_file_attributes and st_reparse_tag, which
    Windows has provided since 3.8, rather than DirEntry.is_junction()
    (3.12+). A reparse point with no readable tag counts as a link, so the
    walk under-counts instead of leaving the tree."""
    if stat_module.S_ISLNK(st.st_mode):
        return True
    attributes = getattr(st, "st_file_attributes", 0)
    if not attributes & getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return False
    tag = getattr(st, "st_reparse_tag", 0)
    return not tag or bool(tag & _REPARSE_TAG_NAME_SURROGATE)


def _is_link(entry: os.DirEntry[str]) -> bool:
    if entry.is_symlink():
        return True
    return _is_link_stat(entry.stat(follow_symlinks=False))


def measure_tree_size(
    path: str,
    *,
    deadline: float,
    max_entries: int = TREE_SCAN_MAX_ENTRIES,
    scandir: Callable[[str], Any] = os.scandir,
    clock: Callable[[], float] = time.perf_counter,
) -> TreeSize | None:
    """Sum file sizes under `path` without reading any file contents.

    Stops at `deadline` (a `clock()` value) or after `max_entries` entries
    and then reports a lower bound. Never follows symlinks or junctions,
    including when `path` itself is one. Returns None when the path does
    not exist or cannot be inspected at all. Never raises.
    """
    try:
        root = os.path.abspath(os.path.expanduser(path))
        st = os.lstat(root)
        if _is_link_stat(st):
            # The copy would follow the link to somewhere outside this
            # tree; we do not, so nothing about its size is known.
            return TreeSize(bytes=0, entries=0, complete=False)
        if not stat_module.S_ISDIR(st.st_mode):
            return TreeSize(bytes=int(st.st_size), entries=1, complete=True, files=1)

        total = 0
        entries = 0
        files = 0
        complete = True
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with scandir(directory) as iterator:
                    for entry in iterator:
                        entries += 1
                        if entries > max_entries or clock() >= deadline:
                            return TreeSize(
                                bytes=total, entries=entries - 1, complete=False, stopped_early=True, files=files
                            )
                        try:
                            if _is_link(entry):
                                complete = False
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(entry.path)
                            else:
                                total += int(entry.stat(follow_symlinks=False).st_size)
                                files += 1
                        except OSError:
                            complete = False
            except OSError:
                complete = False
        return TreeSize(bytes=total, entries=entries, complete=complete, files=files)
    except Exception:
        logger.debug("Size scan failed for %s", path, exc_info=True)
        return None


def _is_network_path(path: str) -> bool:
    if path.startswith(("\\\\", "//")):
        return True
    if sys.platform != "win32":
        return False
    drive = os.path.splitdrive(path)[0]
    if not drive or not drive.endswith(":"):
        return False
    try:
        import ctypes

        drive_remote = 4
        return int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")) == drive_remote
    except Exception:
        return True


def volume_of(path: str) -> VolumeUsage | None:
    """Usage of the local volume holding `path` (or its nearest existing
    parent, for a destination that does not exist yet). Returns None for
    network paths, which could block, and on any error. Never raises."""
    try:
        current = os.path.abspath(os.path.expanduser(path))
        if _is_network_path(current):
            return None
        for _ in range(256):
            if os.path.exists(current):
                break
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent
        else:
            return None
        device = os.stat(current).st_dev
        mount = current
        for _ in range(256):
            if os.path.ismount(mount):
                break
            parent = os.path.dirname(mount)
            if parent == mount:
                break
            mount = parent
        usage = shutil.disk_usage(current)
        if usage.total <= 0:
            return None
        return VolumeUsage(key=str(device), root=mount, used=int(usage.used), total=int(usage.total))
    except Exception:
        logger.debug("Volume lookup failed for %s", path, exc_info=True)
        return None
