"""Archive the older part of the audit log (audit.jsonl), only when the person asks.

Heliox never prunes its audit log on its own. When the person chooses "archive older than <date>":

1. Lines dated before that day are copied, unchanged and in order, to a dated archive beside the live file
   (``audit-2026-11-12.jsonl``; ``-2``, ``-3`` when that name is taken). Lines without a readable date stay in the
   live file: nothing is moved on a guess.
2. Before anything is written, the free space is checked: the archive and the new live file must both fit, with room
   to spare. If they do not, the archive is refused with a plain sentence and nothing changes.
3. The archive is fsynced, read back, and its line count and SHA-256 compared with what was meant to be written.
4. Only then is the new live file (the remaining lines) written beside the old one, fsynced, and renamed over it.
   The live file is never truncated in place, so a crash at any point leaves either the old live file or the new one,
   and the archive.
5. One line in ``audit.archives.jsonl`` records the cut: when, the date, the archive's name, its line count and
   SHA-256, and how many lines stayed live.

The caller holds the audit logger's lock for the whole run, so no line is appended between the read and the rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "audit.archives.jsonl"
# Room to leave on the disk beyond the two files, so archiving never fills it.
SPARE_BYTES = 64 * 1024 * 1024


class ArchiveRefused(Exception):
    """A plain sentence saying why nothing was archived."""


def _line_date(line: bytes) -> date | None:
    try:
        stamp = json.loads(line).get("timestamp")
        return datetime.fromisoformat(str(stamp)).astimezone(UTC).date()
    except (ValueError, TypeError, AttributeError):
        return None


def _free_name(folder: Path, today: date) -> Path:
    candidate = folder / f"audit-{today.isoformat()}.jsonl"
    number = 2
    while candidate.exists():
        candidate = folder / f"audit-{today.isoformat()}-{number}.jsonl"
        number += 1
    return candidate


def _write_synced(path: Path, data: bytes, mode: str) -> None:
    with open(path, mode) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def archive_older_than(
    live: Path,
    older_than: date,
    *,
    today: date | None = None,
    free_bytes: Callable[[Path], int] = lambda folder: shutil.disk_usage(folder).free,
) -> dict[str, Any]:
    """Move audit lines dated before ``older_than`` to a dated archive. Raises ArchiveRefused with the reason."""
    today = today or datetime.now(UTC).date()
    if older_than > today:
        raise ArchiveRefused("Choose a day that is not in the future.")
    if not live.exists():
        raise ArchiveRefused("There is no audit log yet, so there is nothing to archive.")
    raw = live.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines[-1] += b"\n"  # a last line cut short by a crash is kept, now ended
    is_old = [(day := _line_date(line)) is not None and day < older_than for line in lines]
    old = [line for line, moved in zip(lines, is_old, strict=True) if moved]
    keep = [line for line, moved in zip(lines, is_old, strict=True) if not moved]
    if not old:
        raise ArchiveRefused(f"The audit log has no entries from before {older_than.isoformat()}.")
    archive_data, keep_data = b"".join(old), b"".join(keep)

    needed = len(archive_data) + len(keep_data) + SPARE_BYTES
    free = free_bytes(live.parent)
    if free < needed:
        raise ArchiveRefused(
            f"There is not enough free space to archive the audit log: it needs {needed // 2**20} MB and "
            f"{free // 2**20} MB is free. Nothing was changed."
        )

    archive = _free_name(live.parent, today)
    expected_sha = hashlib.sha256(archive_data).hexdigest()
    try:
        _write_synced(archive, archive_data, "xb")
        written = archive.read_bytes()
        if hashlib.sha256(written).hexdigest() != expected_sha or len(written.splitlines()) != len(old):
            raise ArchiveRefused("The archive did not read back as written, so the live audit log was left as it was.")
    except BaseException:
        archive.unlink(missing_ok=True)
        raise

    staged = live.with_name(live.name + ".new")
    _write_synced(staged, keep_data, "wb")
    os.replace(staged, live)
    record = {
        "cut_at": datetime.now(UTC).isoformat(),
        "older_than": older_than.isoformat(),
        "archive": archive.name,
        "lines": len(old),
        "sha256": expected_sha,
        "live_lines_kept": len(keep),
    }
    _write_synced(live.parent / MANIFEST_NAME, (json.dumps(record) + "\n").encode("utf-8"), "ab")
    return record
