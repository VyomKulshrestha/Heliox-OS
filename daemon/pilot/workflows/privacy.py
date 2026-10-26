"""What Heliox keeps about you, how big it is, and how to clear it, forget it or take it with you.

The inventory covers the personal stores under ``DATA_DIR``: the activity timeline, task
receipts, undo copies, notes, the to-do list, reminders, voice shortcuts, setups, usage
nudges, focus distractions and quiet hours, plus the clipboard history held in memory. Each can be cleared on its own. "Forget my activity from the last hour" removes
just that stretch of the timeline.

A backup is a zip of those stores written to the Downloads folder. It never contains
Heliox's configuration, its daemon token, API keys or signing keys, and undo copies are
left out. The activity timeline is included only when asked. Restoring reads only the
known file names from a backup in the Downloads folder, so a crafted zip cannot write
anywhere else.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

BACKUP_PREFIX = "heliox-backup-"
BACKUP_FORMAT = "heliox-backup/1"
MAX_BACKUP_BYTES = 200 * 1024 * 1024

# key -> (label, path relative to DATA_DIR, what it holds)
STORES: dict[str, tuple[str, str, str]] = {
    "timeline": ("Activity timeline", "timeline", "Which app and window title were in front, and when."),
    "receipts": ("Task receipts", "receipts", "Signed records of what Heliox ran; contents only as hashes."),
    "undo": ("Undo copies", "undo", "Copies of files Heliox changed, kept so tasks can be undone."),
    "notes": ("Notes", "notes.json", "Your quick notes."),
    "snippets": ("Snippets", "snippets.json", "Named texts you saved to copy again."),
    "todos": ("To-do list", "todos.json", "Things on your list, done or not."),
    "reminders": ("Reminders", "reminders.json", "Reminders and timers that have not fired yet."),
    "app_reminders": ("App reminders", "app_reminders.json", "Reminders that wait for an app to open."),
    "shortcuts": ("Voice shortcuts", "voice_shortcuts.json", "Phrases and the commands they stand for."),
    "workspaces": ("Setups", "workspaces.json", "Names of the apps in each saved setup."),
    "nudges": ("Usage nudges", "nudges.json", "Daily limits for apps or sites."),
    "distractions": ("Focus distractions", "focus_distractions.json", "Apps or sites to mention during focus."),
    "quiet_hours": ("Quiet hours", "quiet_hours.json", "When focus starts by itself."),
    # Kept in memory only: nothing on disk, never in a backup, gone after a restart.
    "clipboard": ("Clipboard history", "", "Text you copied recently, in memory only."),
}
# Files a backup may contain, by archive name; nothing else is ever restored.
BACKUP_FILES = (
    "notes.json",
    "snippets.json",
    "todos.json",
    "reminders.json",
    "app_reminders.json",
    "voice_shortcuts.json",
    "workspaces.json",
    "nudges.json",
    "focus_distractions.json",
    "quiet_hours.json",
)
FORGET_REQUEST = re.compile(
    r"^(?:please )?(?:forget|delete|erase|remove) (?:my |the )?(?:activity|timeline|what i did|what i was doing)"
    r"(?: from| for| in)? (?P<when>.+)$"
)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def human_size(size: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def inventory(data_dir: Path, counts: dict[str, int | None]) -> list[dict[str, Any]]:
    """Each personal store with its size on disk and, where known, how many items it holds."""
    rows = []
    for key, (label, relative, about) in STORES.items():
        path = Path(data_dir) / relative
        size = _size(path) if relative and path.exists() else 0
        rows.append(
            {
                "key": key,
                "label": label,
                "about": about,
                "bytes": size,
                "size": human_size(size),
                "items": counts.get(key),
                "memory_only": not relative,
            }
        )
    return rows


def forget_window(text: str, now: datetime) -> tuple[float, float, str] | None:
    """(start, end, label) from "forget my activity from the last hour" and similar; None if not a forget request."""
    from pilot.workflows.activity_timeline import parse_when

    normalized = " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
    asked = FORGET_REQUEST.match(normalized)
    if not asked:
        return None
    when = asked.group("when")
    recent = re.fullmatch(
        r"(?:the )?(?:last|past) (?:(?P<amount>\d+|an|a|one|two|three|half an) )?(?P<unit>minutes?|hours?)", when
    )
    if recent:
        words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
        amount_text = recent.group("amount") or "1"
        if amount_text == "half an":
            minutes = 30
        else:
            amount = int(amount_text) if amount_text.isdigit() else words[amount_text]
            minutes = amount * (60 if recent.group("unit").startswith("hour") else 1)
        label = f"the last {minutes} minutes" if minutes < 60 or minutes % 60 else f"the last {minutes // 60} hour(s)"
        return (now - timedelta(minutes=minutes)).timestamp(), now.timestamp(), label
    start, end, label = parse_when(when, now)
    return start, end, label


def create_backup(data_dir: Path, target_dir: Path, *, include_timeline: bool = False) -> Path:
    """Zip the personal stores into ``target_dir``. Never includes configuration, tokens or keys."""
    data_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{BACKUP_PREFIX}{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("FORMAT", BACKUP_FORMAT)
        for name in BACKUP_FILES:
            source = data_dir / name
            if source.is_file():
                archive.write(source, name)
        receipts = data_dir / "receipts"
        if receipts.is_dir():
            for receipt in sorted(receipts.glob("*.json")):
                archive.write(receipt, f"receipts/{receipt.name}")
        timeline = data_dir / "timeline" / "timeline.db"
        if include_timeline and timeline.is_file():
            archive.write(timeline, "timeline/timeline.db")
    return path


def list_backups(target_dir: Path) -> list[dict[str, Any]]:
    if not target_dir.is_dir():
        return []
    rows = []
    for path in sorted(target_dir.glob(f"{BACKUP_PREFIX}*.zip"), reverse=True):
        stat = path.stat()
        rows.append({"name": path.name, "bytes": stat.st_size, "size": human_size(stat.st_size), "at": stat.st_mtime})
    return rows


def restore_backup(data_dir: Path, target_dir: Path, name: str) -> list[str]:
    """Restore the known stores from a backup in ``target_dir``. Returns what was restored."""
    if not re.fullmatch(rf"{re.escape(BACKUP_PREFIX)}[\w-]+\.zip", name):
        raise ValueError("Choose one of the Heliox backups in your Downloads folder")
    path = Path(target_dir) / name
    if not path.is_file() or path.stat().st_size > MAX_BACKUP_BYTES:
        raise ValueError("That backup is missing or too large")
    data_dir = Path(data_dir)
    restored: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "FORMAT" not in names or archive.read("FORMAT").decode("utf-8", "replace") != BACKUP_FORMAT:
            raise ValueError("This is not a Heliox backup")
        for name_in_zip in BACKUP_FILES:
            if name_in_zip in names:
                _write_atomic(data_dir / name_in_zip, archive.read(name_in_zip))
                restored.append(name_in_zip)
        receipt_names = [n for n in names if re.fullmatch(r"receipts/[\w.-]+\.json", n)]
        if receipt_names:
            (data_dir / "receipts").mkdir(parents=True, exist_ok=True)
            for receipt_name in receipt_names:
                target = data_dir / "receipts" / Path(receipt_name).name
                if not target.exists():  # never overwrite a receipt this computer already holds
                    _write_atomic(target, archive.read(receipt_name))
            restored.append(f"{len(receipt_names)} receipts")
        if "timeline/timeline.db" in names:
            restored.append("timeline (left out: restore it by hand while Heliox is stopped)")
    return restored


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{int(time.time() * 1000)}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def clear_store(data_dir: Path, key: str) -> None:
    """Delete one JSON store or folder (the timeline and undo copies are cleared through their own objects)."""
    if key not in STORES:
        raise ValueError(f"Unknown store: {key}")
    relative = STORES[key][1]
    if not relative:  # kept in memory only; there is nothing on disk to delete
        return
    path = Path(data_dir) / relative
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()
