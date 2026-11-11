"""Apps as people name them: "Windows Terminal", not "WindowsTerminal"; "Word", not "WINWORD".

``display_name`` is pure: a small table for process names that are not what people say, then the program's own
description when the system has a real one, then the process name unchanged. Unknown names are never split or
re-cased by rule, since that would mangle names that are already right.

The description comes from the program itself on each system: the file description on Windows, the bundle's
display name (``CFBundleDisplayName``, else ``CFBundleName``) on macOS, and the desktop entry's ``Name`` on Linux.
The lookups are pure functions over a path or a file's text, tested on every system.
"""

from __future__ import annotations

import functools
import logging
import os
import plistlib
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

logger = logging.getLogger("pilot.system.app_names")

# Process name (casefolded, without .exe) -> what people call it.
KNOWN = {
    "windowsterminal": "Windows Terminal",
    "openconsole": "Windows Terminal",
    "conhost": "Console",
    "code": "VS Code",
    "code - insiders": "VS Code Insiders",
    "msedge": "Edge",
    "chrome": "Chrome",
    "firefox": "Firefox",
    "brave": "Brave",
    "outlook": "Outlook",
    "olk": "Outlook",  # the new Outlook
    "winword": "Word",
    "excel": "Excel",
    "powerpnt": "PowerPoint",
    "onenote": "OneNote",
    "msteams": "Teams",
    "ms-teams": "Teams",
    "powershell": "PowerShell",
    "pwsh": "PowerShell",
    "cmd": "Command Prompt",
    "explorer": "File Explorer",
    "notepad": "Notepad",
    "wordpad": "WordPad",
    "mspaint": "Paint",
    "claude": "Claude",
    "applicationframehost": "",  # a host for Store apps: its own name says nothing, so the description decides
}

# Descriptions that name the plumbing rather than the program.
_NOT_A_NAME = ("host", "helper", "launcher", "runtime", "service", "broker", "processor", "stub", "wrapper")


def _bare(process: str) -> str:
    name = process.strip()
    return name[:-4] if name.casefold().endswith(".exe") else name


def _real_description(description: str) -> str:
    text = " ".join(description.split())
    if not text or len(text) > 40:
        return ""
    if any(word in text.casefold().split() for word in _NOT_A_NAME):
        return ""
    return text


def display_name(process: str, description: str = "") -> str:
    """How a person names the app behind ``process`` (a process name such as "WindowsTerminal" or "code.exe").

    ``description`` is what Windows reports for the program (its file description), when known.
    """
    bare = _bare(process)
    known = KNOWN.get(bare.casefold())
    if known:
        return known
    return _real_description(description) or bare


def bundle_display_name(exe: str, read: Callable[[Path], bytes] = Path.read_bytes) -> str:
    """The name a macOS app bundle gives itself, from the path of its executable ("" when not in a bundle)."""
    parts = Path(exe).parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].endswith(".app"):
            try:
                info = plistlib.loads(read(Path(*parts[: index + 1]) / "Contents" / "Info.plist"))
            except Exception:
                return ""
            return str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or "").strip()
    return ""


def parse_desktop_entry(text: str) -> dict[str, str]:
    """The keys of a desktop entry's [Desktop Entry] group (unlocalised values only)."""
    keys: dict[str, str] = {}
    group = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]
            continue
        if group == "Desktop Entry" and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            if "[" not in key:
                keys.setdefault(key, value.strip())
    return keys


def _exec_name(command: str) -> str:
    """The program an Exec line starts, skipping ``env VAR=value`` prefixes."""
    words = command.split()
    if words and Path(words[0].strip('"')).name == "env":
        words = [word for word in words[1:] if "=" not in word]
    return Path(words[0].strip('"')).name if words else ""


def desktop_entry_name(process: str, entries: Iterable[tuple[str, str]]) -> str:
    """The ``Name`` of the desktop entry that launches ``process``, from (file stem, file text) pairs.

    An entry matches by its file name (whole, or its last dotted part: "org.gnome.Nautilus" for "nautilus"), by
    the program its ``Exec`` starts, or by ``StartupWMClass``. A visible entry wins over a hidden one.
    """
    wanted = _bare(process).casefold()
    if not wanted:
        return ""
    hidden = ""
    for stem, text in entries:
        keys = parse_desktop_entry(text)
        if keys.get("Type", "Application") != "Application" or not keys.get("Name"):
            continue
        names = {
            stem.casefold(),
            stem.rsplit(".", 1)[-1].casefold(),
            _exec_name(keys.get("Exec", "")).casefold(),
            keys.get("StartupWMClass", "").casefold(),
        }
        if wanted in names:
            if keys.get("NoDisplay", "").lower() != "true" and keys.get("Hidden", "").lower() != "true":
                return keys["Name"]
            hidden = hidden or keys["Name"]
    return hidden


def _desktop_dirs() -> list[Path]:
    home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    data = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    dirs = [home, *(Path(d) for d in data.split(":") if d)]
    dirs += [Path("/var/lib/flatpak/exports/share"), home / "flatpak" / "exports" / "share"]
    folders = [d / "applications" for d in dirs]
    return [*folders, Path("/var/lib/snapd/desktop/applications")]


@functools.lru_cache(maxsize=1)
def _desktop_entries() -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for folder in _desktop_dirs():
        try:
            files = sorted(folder.glob("*.desktop"))
        except OSError:
            continue
        for path in files[:2000]:
            try:
                found.append((path.stem, path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return tuple(found)


def _process_exe(process: str) -> str:
    import psutil

    wanted = _bare(process).casefold()
    for proc in psutil.process_iter(["name", "exe"]):
        name = proc.info.get("name") or ""
        if _bare(name).casefold() == wanted and proc.info.get("exe"):
            return str(proc.info["exe"])
    return ""


@functools.lru_cache(maxsize=128)
def describe(process: str) -> str:
    """What the system calls the program behind a running process, or "".

    Windows: the file description. macOS: the app bundle's display name. Linux: the desktop entry's name.
    """
    if not process:
        return ""
    try:
        if os.name == "nt":
            import win32api

            exe = _process_exe(process)
            if not exe:
                return ""
            (lang, page), *_ = win32api.GetFileVersionInfo(exe, "\\VarFileInfo\\Translation")
            return str(win32api.GetFileVersionInfo(exe, f"\\StringFileInfo\\{lang:04x}{page:04x}\\FileDescription"))
        if sys.platform == "darwin":
            exe = _process_exe(process)
            return bundle_display_name(exe) if exe else ""
        if sys.platform.startswith("linux"):
            return desktop_entry_name(process, _desktop_entries())
    except Exception:
        logger.debug("No description for %s", process, exc_info=True)
    return ""


def app_label(process: str) -> str:
    """``display_name`` with the description looked up on this computer."""
    known = KNOWN.get(_bare(process).casefold())
    if known:
        return known
    return display_name(process, describe(process))
