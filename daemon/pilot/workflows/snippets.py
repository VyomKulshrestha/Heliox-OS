"""Saved snippets: "save that as my signature", "copy my signature", "read what I copied".

A snippet is a short named text the person reuses: an address, a signature, a meeting
link. Saving takes what is on the clipboard now; copying puts it back on the clipboard,
ready to paste. Nothing is typed into other apps. Text that looks like a password, key,
token or card number is refused (that belongs in a password manager). Snippets are kept
in ``DATA_DIR/snippets.json``.

"Read what I copied" reads the clipboard back, unless it looks like a secret.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pilot.workflows.clipboard_history import looks_secret

MAX_SNIPPETS = 50
MAX_TEXT = 5000
READ_ALOUD_LIMIT = 1200

SAVE_REQUEST = re.compile(
    r"^(?:please )?save (?:that|this|it|what i (?:just )?copied|the clipboard|my clipboard) as (?:my |a |the )?"
    r"(?P<name>.+?)(?: snippet)?$"
)
COPY_REQUEST = re.compile(r"^(?:please )?(?:copy|paste|give me|get) (?:me )?(?:my |the )?(?P<name>.+?)(?: snippet)?$")
LIST_REQUEST = re.compile(r"^(?:what|which) snippets do i have$|^(?:list|show) my snippets$|^what have i saved$")
DELETE_REQUEST = re.compile(r"^(?:please )?(?:delete|remove|forget) (?:my |the )?(?P<name>.+?) snippet$")
READ_REQUEST = re.compile(
    r"^(?:please )?read (?:me )?(?:what i (?:just )?copied|my clipboard|the clipboard|what s on (?:my |the )?clipboard)"
    r"(?: to me| aloud| out loud)?$"
)
_NOT_SNIPPETS = re.compile(r"\b(?:setup|workspace|layout|routine|shortcut)$")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _preview(text: str, limit: int = 50) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class SnippetStore:
    """Named texts in a small JSON file."""

    def __init__(
        self,
        path: Path,
        *,
        read: Callable[[], Awaitable[str]] | None = None,
        write: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self._path = Path(path)
        self._read = read
        self._write = write

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("name") and item.get("text")]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self, name: str) -> dict[str, Any] | None:
        wanted = _normalize(name)
        return next((item for item in self.list() if _normalize(item["name"]) == wanted), None)

    def save(self, name: str, text: str) -> dict[str, Any]:
        name = " ".join(_normalize(name).split())
        if not 1 <= len(name) <= 40:
            raise ValueError("A snippet name must be 1 to 40 characters")
        if not text.strip() or len(text) > MAX_TEXT:
            raise ValueError(f"A snippet must be 1 to {MAX_TEXT} characters")
        if looks_secret(text):
            raise ValueError("That looks like a password or key, so I won't save it. Keep it in your password manager.")
        items = [item for item in self.list() if _normalize(item["name"]) != name]
        if len(items) >= MAX_SNIPPETS:
            raise ValueError(f"You can keep up to {MAX_SNIPPETS} snippets")
        entry = {"name": name, "text": text, "created_at": time.time()}
        items.append(entry)
        self._save(items)
        return entry

    def delete(self, name: str) -> bool:
        items = self.list()
        kept = [item for item in items if _normalize(item["name"]) != _normalize(name)]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    async def copy(self, name: str) -> dict[str, Any] | None:
        item = self.get(name)
        if item is None or self._write is None:
            return None
        await self._write(item["text"])
        return item

    async def answer_async(self, text: str) -> str | None:
        """Save, copy, list or delete snippets, or read the clipboard; None when it is none of those."""
        normalized = _normalize(text)
        if READ_REQUEST.match(normalized):
            if self._read is None:
                return None
            current = await self._read()
            if not current.strip() or current == "(clipboard is empty)":
                return "The clipboard is empty."
            if looks_secret(current):
                return "That looks like a password or key, so I won't read it aloud."
            flat = " ".join(current.split())
            if len(flat) > READ_ALOUD_LIMIT:
                return f"You copied: {flat[:READ_ALOUD_LIMIT]}… That's the first part; it goes on."
            return f"You copied: {flat}"
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return 'You have no snippets. Copy some text, then say "save that as my signature".'
            return (
                "Your snippets: " + "; ".join(f"{item['name']} ({_preview(item['text'], 30)})" for item in items) + "."
            )
        delete = DELETE_REQUEST.match(normalized)
        if delete:
            name = delete.group("name")
            return f"Deleted your {name} snippet." if self.delete(name) else f"You have no snippet called {name}."
        save = SAVE_REQUEST.match(normalized)
        if save and not _NOT_SNIPPETS.search(save.group("name")):
            if self._read is None:
                return None
            current = await self._read()
            if not current.strip() or current == "(clipboard is empty)":
                return "The clipboard is empty. Copy the text first, then say it again."
            try:
                entry = self.save(save.group("name"), current)
            except ValueError as exc:
                return str(exc)
            return (
                f'Saved "{_preview(entry["text"])}" as your {entry["name"]}. Say "copy my {entry["name"]}" to use it.'
            )
        copy = COPY_REQUEST.match(normalized)
        if copy and self.get(copy.group("name")) is not None:  # "copy my files to D:" is a task for the planner
            item = await self.copy(copy.group("name"))
            if item is None:
                return "I couldn't put that on the clipboard."
            return f"Your {item['name']} is on the clipboard. Press Ctrl+V to paste it."
        return None
