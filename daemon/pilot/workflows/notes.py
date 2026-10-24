"""Quick notes: "note that the plumber comes Tuesday", "read my notes", "what did I note about the bank".

Notes are short texts kept in ``DATA_DIR/notes.json`` on this computer. Taking, reading,
searching and deleting the last note are handled in code; no model sees them unless
the person asks a task to use them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_NOTES = 500
MAX_TEXT = 500

TAKE_REQUEST = re.compile(
    r"^(?:please )?(?:take a note|make a note|add a note|jot down|note down|note)(?: that)?[:,]? (?P<text>.+)$",
    re.IGNORECASE,
)
READ_REQUEST = re.compile(r"^(?:read|show|list) (?:me )?my notes$|^what are my notes$")
SEARCH_REQUEST = re.compile(r"^what did i (?:note|write down) (?:about|on) (?P<what>.+)$")
DELETE_LAST_REQUEST = re.compile(r"^(?:delete|remove|forget) (?:my |the )?last note$")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _when(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%a %d %b")


class NoteStore:
    """Short notes in a small JSON file, newest last."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("text")] if isinstance(data, list) else []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, text: str) -> dict[str, Any]:
        text = " ".join(str(text).split())
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"A note must be 1 to {MAX_TEXT} characters")
        items = self.list()
        if len(items) >= MAX_NOTES:
            raise ValueError(f"You can keep up to {MAX_NOTES} notes; delete some first")
        entry = {"id": uuid.uuid4().hex[:10], "text": text, "created_at": time.time()}
        items.append(entry)
        self._save(items)
        return entry

    def delete(self, note_id: str) -> bool:
        items = self.list()
        kept = [item for item in items if item["id"] != note_id]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    def search(self, query: str) -> list[dict[str, Any]]:
        words = [word for word in _normalize(query).split() if len(word) > 1]
        if not words:
            return []
        return [item for item in reversed(self.list()) if all(word in _normalize(item["text"]) for word in words)]

    def answer(self, text: str) -> str | None:
        """Take, read, search or delete notes from a sentence; None when it is not about notes."""
        normalized = _normalize(text)
        if READ_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return 'You have no notes. Say "note that …" to add one.'
            latest = list(reversed(items))[:5]
            listed = "; ".join(f"{_when(item['created_at'])}: {item['text']}" for item in latest)
            more = f" And {len(items) - 5} older." if len(items) > 5 else ""
            return f"Your latest notes: {listed}.{more}"
        search = SEARCH_REQUEST.match(normalized)
        if search:
            hits = self.search(search.group("what"))
            if not hits:
                return f"You have no note about {search.group('what')}."
            return "; ".join(f"{_when(item['created_at'])}: {item['text']}" for item in hits[:3]) + "."
        if DELETE_LAST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return "You have no notes to delete."
            self.delete(items[-1]["id"])
            return f'Deleted your last note: "{items[-1]["text"]}".'
        take = TAKE_REQUEST.match(text.strip())
        if take:
            try:
                self.add(take.group("text"))
            except ValueError as exc:
                return str(exc)
            return "Noted."
        return None

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
