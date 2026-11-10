"""Apps as people name them: "Windows Terminal", not "WindowsTerminal"; "Word", not "WINWORD".

``display_name`` is pure: a small table for process names that are not what people say, then the program's own
description when Windows has a real one, then the process name unchanged. Unknown names are never split or
re-cased by rule, since that would mangle names that are already right.
"""

from __future__ import annotations

import functools
import logging
import os

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


@functools.lru_cache(maxsize=128)
def describe(process: str) -> str:
    """The file description Windows reports for a running process with this name, or "" (always "" off Windows)."""
    if os.name != "nt" or not process:
        return ""
    try:
        import psutil
        import win32api

        wanted = _bare(process).casefold()
        for proc in psutil.process_iter(["name", "exe"]):
            name = proc.info.get("name") or ""
            if _bare(name).casefold() != wanted or not proc.info.get("exe"):
                continue
            exe = proc.info["exe"]
            (lang, page), *_ = win32api.GetFileVersionInfo(exe, "\\VarFileInfo\\Translation")
            return str(win32api.GetFileVersionInfo(exe, f"\\StringFileInfo\\{lang:04x}{page:04x}\\FileDescription"))
    except Exception:
        logger.debug("No description for %s", process, exc_info=True)
    return ""


def app_label(process: str) -> str:
    """``display_name`` with the description looked up on this computer."""
    known = KNOWN.get(_bare(process).casefold())
    if known:
        return known
    return display_name(process, describe(process))
