"""Workspaces: "save this as my coding setup", then "open my coding setup".

Saving records which apps have windows open right now, by name only (no window
titles, no file names). Opening a workspace does not launch anything itself: it
becomes an ordinary request ("open Visual Studio Code, Slack and Google Chrome") that
goes through the normal planning, gateway and approvals.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

MAX_WORKSPACES = 30
MAX_APPS = 15
# Desktop furniture and Heliox itself are never part of a workspace.
IGNORED_APPS = frozenset(
    {
        "applicationframehost",
        "dock",
        "explorer",
        "finder",
        "heliox",
        "heliox os",
        "lockapp",
        "pilot",
        "searchhost",
        "shellexperiencehost",
        "startmenuexperiencehost",
        "systemsettings",
        "systemuiserver",
        "textinputhost",
        "windowserver",
    }
)

SAVE_REQUEST = re.compile(
    r"^(?:please )?(?:save|remember) (?:this|these apps|what i have open|my (?:current )?(?:setup|apps|windows|workspace))"
    r" as (?:my |the )?(?P<name>.+?)(?: setup| workspace)?$"
)
OPEN_REQUEST = re.compile(
    r"^(?:please )?(?:open|restore|load|start|set up) (?:my |the )?(?P<name>.+?) (?:setup|workspace)$"
)
LIST_REQUEST = re.compile(r"^(?:what|which) (?:setups|workspaces) do i have$|^(?:list|show) my (?:setups|workspaces)$")
DELETE_REQUEST = re.compile(r"^(?:please )?(?:delete|forget|remove) (?:my |the )?(?P<name>.+?) (?:setup|workspace)$")


def normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.casefold()).split())


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


async def open_apps() -> list[str]:
    """Friendly names of apps that have a window open, in a stable order, without duplicates."""
    names: list[str] = []
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _err = await run_powershell(
            "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Select-Object ProcessName, Description | ConvertTo-Json -Compress"
        )
        if code == 0 and out.strip():
            data = json.loads(out)
            for item in data if isinstance(data, list) else [data]:
                process = str(item.get("ProcessName") or "")
                description = str(item.get("Description") or "").strip()
                if normalize(process) not in IGNORED_APPS:
                    names.append(description or process)
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, _err = await run_command(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of every process whose background only is false',
            ]
        )
        if code == 0:
            names = [name.strip() for name in out.split(",") if name.strip()]
    else:
        code, out, _err = await run_command(["wmctrl", "-lx"])
        if code == 0:
            for line in out.splitlines():
                parts = line.split(None, 4)
                if len(parts) >= 3 and "." in parts[2]:
                    names.append(parts[2].split(".", 1)[1])
    seen: set[str] = set()
    apps = []
    for name in names:
        key = normalize(name)
        if key and key not in seen and key not in IGNORED_APPS:
            seen.add(key)
            apps.append(name)
    return apps[:MAX_APPS]


class WorkspaceStore:
    """Named sets of apps kept in a small JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("name")] if isinstance(data, list) else []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def save(self, name: str, apps: list[str]) -> dict[str, Any]:
        name = normalize(name)
        if not 1 <= len(name) <= 40:
            raise ValueError("A workspace name must be 1 to 40 characters")
        if not apps:
            raise ValueError("No app windows are open to save")
        items = [item for item in self.list() if item["name"] != name]
        if len(items) >= MAX_WORKSPACES:
            raise ValueError(f"You can keep up to {MAX_WORKSPACES} workspaces")
        entry = {"name": name, "apps": list(apps)[:MAX_APPS], "saved_at": time.time()}
        items.append(entry)
        self._save(items)
        return entry

    def get(self, name: str) -> dict[str, Any] | None:
        name = normalize(name)
        return next((item for item in self.list() if item["name"] == name), None)

    def delete(self, name: str) -> bool:
        name = normalize(name)
        items = self.list()
        kept = [item for item in items if item["name"] != name]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    def open_request(self, text: str) -> tuple[str, str] | None:
        """("open A, B and C", note) for "open my coding setup", or None."""
        asked = OPEN_REQUEST.match(normalize(text))
        workspace = self.get(asked.group("name")) if asked else None
        if workspace is None:
            return None
        command = f"open {_join(workspace['apps'])}"
        return command, f'Workspace "{workspace["name"]}": {command}'

    def describe(self, entry: dict[str, Any]) -> str:
        return f"{entry['name']} ({_join(entry['apps'])})"
