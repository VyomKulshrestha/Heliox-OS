"""How big Heliox's data folder and its unbounded stores are, and one plain warning when the folder passes a size.

The audit log, the durable task tables, the checkpoints and the experience ledger have no size bound: they are records
Heliox never prunes on its own (docs/RELIABILITY.md, "How much each store can grow"). Instead the person sees each
one's size where storage health is shown, and is told once when the whole data folder passes
``storage.data_folder_warn_gb``. The warning is said again only after the folder has gone back under the limit (for
example after archiving the audit log) and then passed it again.

Sizes are read from the file system only; nothing here opens a database.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.data_size")

GB = 1024**3

# (name shown to the person, files in the data folder that make up that store; SQLite's -wal and -shm included)
STORES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Audit log", ("audit.jsonl",)),
    ("Audit log archives", ("audit-*.jsonl",)),
    ("Durable tasks", ("durable_tasks.db*",)),
    ("Checkpoints", ("workflow_checkpoints.db*",)),
    ("Experience ledger", ("experience.db*",)),
)


def folder_bytes(folder: Path) -> int:
    """Total size of every file under ``folder``; files that vanish or cannot be read meanwhile count as 0."""
    total = 0
    for root, _dirs, files in os.walk(folder, onerror=lambda _error: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name), follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def store_bytes(folder: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        for path in folder.glob(pattern):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def sizes(folder: Path) -> dict[str, Any]:
    """Each named store's bytes and the whole folder's bytes."""
    return {
        "folder": str(folder),
        "total_bytes": folder_bytes(folder) if folder.exists() else 0,
        "stores": [{"name": name, "bytes": store_bytes(folder, patterns)} for name, patterns in STORES],
    }


def readable(size: int) -> str:
    if size >= GB:
        return f"{size / GB:.1f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.0f} MB"
    return f"{size / 1024:.0f} KB"


def warning_sentence(snapshot: dict[str, Any], limit_gb: float) -> str:
    biggest = max(snapshot["stores"], key=lambda store: store["bytes"], default=None)
    detail = f" ({biggest['name'].lower()} {readable(biggest['bytes'])})" if biggest and biggest["bytes"] else ""
    return (
        f"Heliox's data folder is {readable(snapshot['total_bytes'])}{detail}, over the {limit_gb:g} GB you set. "
        "Nothing is deleted on its own; you can archive the older audit log in Settings, or raise the limit there."
    )


@dataclass
class SizeWatch:
    """Says the warning once per crossing of the limit."""

    limit_gb: Callable[[], float]
    over: bool = False

    def check(self, snapshot: dict[str, Any]) -> str | None:
        limit = float(self.limit_gb())
        is_over = snapshot["total_bytes"] > limit * GB
        crossed = is_over and not self.over
        self.over = is_over
        return warning_sentence(snapshot, limit) if crossed else None
