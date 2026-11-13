"""What a run wrote, to show it stayed inside its own root.

Two sources. In this process, Python's audit hook sees every file opened for writing, every folder made and every
file created, moved or removed through Python, and every SQLite database opened. Another process (the voice worker)
is not seen by the hook, so folders it could write are also compared by modification time: anything newer than the
start of the run. Only names and times are read, never a file's contents.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

_PATH_EVENTS = {
    "os.mkdir": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.truncate": (0,),
    "os.symlink": (1,),
    "os.link": (1,),
    "shutil.copyfile": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _writes_on_open(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _sqlite_path(database: object) -> str | None:
    text = os.fsdecode(database) if isinstance(database, (str, bytes, os.PathLike)) else ""
    if not text or text == ":memory:" or "immutable=1" in text or "mode=memory" in text:
        return None
    return text[5:].split("?", 1)[0] if text.startswith("file:") else text


class Writes:
    """The paths this process wrote while ``on``. An audit hook cannot be removed, so ``stop`` only switches it off."""

    def __init__(self) -> None:
        self.paths: set[str] = set()
        self.on = False
        self._installed = False

    def install(self) -> Writes:
        if not self._installed:
            sys.addaudithook(self._hook)
            self._installed = True
        self.on = True
        return self

    def stop(self) -> None:
        self.on = False

    def _hook(self, event: str, args: tuple) -> None:
        if not self.on:
            return
        try:
            if event == "open":
                path, mode, flags = (tuple(args) + (None, None, None))[:3]
                if path is not None and not isinstance(path, int) and _writes_on_open(mode, flags):
                    self._add(path)
            elif event == "sqlite3.connect":
                found = _sqlite_path(args[0]) if args else None
                if found:
                    self._add(found)
            elif event in _PATH_EVENTS:
                for index in _PATH_EVENTS[event]:
                    if index < len(args) and isinstance(args[index], (str, bytes, os.PathLike)):
                        self._add(args[index])
        except Exception:  # an audit hook must never break the program it watches
            return

    def _add(self, path: object) -> None:
        self.paths.add(os.path.abspath(os.fsdecode(path)))  # type: ignore[arg-type]


def newer_files(folders: Iterable[Path], since: float) -> list[str]:
    """Files under ``folders`` whose modification time is after ``since`` (names and times only)."""
    found = []
    for folder in folders:
        if not Path(folder).exists():
            continue
        for top, _dirs, files in os.walk(folder):
            for name in files:
                path = os.path.join(top, name)
                try:
                    if os.stat(path).st_mtime > since:
                        found.append(path)
                except OSError:
                    continue
    return sorted(found)


def inside(path: str, root: Path) -> bool:
    inner = os.path.normcase(os.path.realpath(path))
    outer = os.path.normcase(os.path.realpath(root))
    return inner == outer or inner.startswith(outer.rstrip(os.sep) + os.sep)


def outside(paths: Iterable[str], root: Path) -> list[str]:
    return sorted(p for p in paths if not inside(p, root))
