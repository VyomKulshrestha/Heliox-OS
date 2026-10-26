"""A to-do list: "add call the bank to my list", "what's on my list", "mark call the bank done".

Items are kept in ``DATA_DIR/todos.json`` on this computer. Adding, reading, finishing and
removing items are handled in code; no model sees the list unless the person asks a
task to use it. Open items show in the Today tab and in the daily brief.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

MAX_ITEMS = 300
MAX_TEXT = 200

_LIST = r"(?:my |the )?(?:to do |todo |todos |task )?list"
ADD_REQUEST = re.compile(r"^(?:please )?(?:add|put) (?P<text>.+?) (?:to|on|onto) " + _LIST + "$")
READ_REQUEST = re.compile(
    r"^(?:what s|what is|whats) (?:left )?on " + _LIST + "$"
    r"|^(?:read|show|list) (?:me )?my (?:to do |todo )?(?:list|to dos|todos)$"
    r"|^what do i (?:still )?(?:have|need) to do$"
)
DONE_REQUEST = re.compile(
    r"^(?:please )?(?:mark (?P<a>.+?) (?:as )?(?:done|complete|completed|finished)"
    r"|(?:check|cross|tick) off (?P<b>.+?))(?: on| from| off)?(?: " + _LIST + ")?$"
)
REMOVE_REQUEST = re.compile(r"^(?:please )?(?:remove|delete|take) (?P<text>.+?) (?:from|off) " + _LIST + "$")
CLEAR_DONE_REQUEST = re.compile(
    r"^(?:please )?clear (?:the |my )?(?:done|finished|completed|checked) (?:items|tasks|to dos|todos)"
    r"(?: from " + _LIST + ")?$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


class TodoStore:
    """To-do items in a small JSON file, oldest first."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("text")] if isinstance(data, list) else []

    def open_items(self) -> list[dict[str, Any]]:
        return [item for item in self.list() if not item.get("done")]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, text: str) -> dict[str, Any]:
        text = " ".join(str(text).split())
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"A to-do must be 1 to {MAX_TEXT} characters")
        items = self.list()
        if any(not item.get("done") and _normalize(item["text"]) == _normalize(text) for item in items):
            raise ValueError(f'"{text}" is already on your list')
        if len(items) >= MAX_ITEMS:
            raise ValueError(f"You can keep up to {MAX_ITEMS} to-dos; clear finished ones first")
        entry = {"id": uuid.uuid4().hex[:10], "text": text, "created_at": time.time(), "done": False, "done_at": None}
        items.append(entry)
        self._save(items)
        return entry

    def set_done(self, item_id: str, done: bool = True) -> dict[str, Any] | None:
        items = self.list()
        for item in items:
            if item["id"] == item_id:
                item["done"] = bool(done)
                item["done_at"] = time.time() if done else None
                self._save(items)
                return item
        return None

    def delete(self, item_id: str) -> bool:
        items = self.list()
        kept = [item for item in items if item["id"] != item_id]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    def clear_done(self) -> int:
        items = self.list()
        kept = [item for item in items if not item.get("done")]
        if len(kept) != len(items):
            self._save(kept)
        return len(items) - len(kept)

    def find(self, query: str, *, done: bool | None = False) -> list[dict[str, Any]]:
        """Items matching ``query``: an exact match alone, otherwise every item containing all its words."""
        wanted = _normalize(query)
        words = [word for word in wanted.split() if word not in ("the", "a", "an", "my")]
        candidates = [item for item in self.list() if done is None or bool(item.get("done")) == done]
        exact = [item for item in candidates if _normalize(item["text"]) == wanted]
        if exact or not words:
            return exact
        return [item for item in candidates if all(word in _normalize(item["text"]).split() for word in words)]

    def _pick(self, query: str, *, done: bool | None) -> dict[str, Any] | str:
        hits = self.find(query, done=done)
        if not hits:
            return f'There is no "{query}" on your list.'
        if len(hits) > 1:
            names = "; ".join(item["text"] for item in hits[:3])
            return f"Which one: {names}?"
        return hits[0]

    def answer(self, text: str) -> str | None:
        """Add, read, finish or remove to-dos from a sentence; None when it is not about the list."""
        normalized = _normalize(text)
        if READ_REQUEST.match(normalized):
            items = self.open_items()
            if not items:
                return 'Your list is empty. Say "add call the bank to my list" to start one.'
            listed = "; ".join(item["text"] for item in items[:7])
            more = f" And {len(items) - 7} more." if len(items) > 7 else ""
            return f"You have {len(items)} thing{'' if len(items) == 1 else 's'} to do: {listed}.{more}"
        if CLEAR_DONE_REQUEST.match(normalized):
            removed = self.clear_done()
            if not removed:
                return "There are no finished items to clear."
            return f"Cleared {removed} finished item{'' if removed == 1 else 's'}."
        done = DONE_REQUEST.match(normalized)
        if done:
            query = (done.group("a") or done.group("b")).strip()
            # "Mark this email as done" is a task for the planner unless the list is named or an item matches.
            if not normalized.endswith("list") and not self.find(query, done=False):
                return None
            picked = self._pick(query, done=False)
            if isinstance(picked, str):
                return picked
            self.set_done(picked["id"])
            left = len(self.open_items())
            return (
                f'Done: "{picked["text"]}". {left} left.' if left else f'Done: "{picked["text"]}". Your list is clear.'
            )
        remove = REMOVE_REQUEST.match(normalized)
        if remove:
            picked = self._pick(remove.group("text").strip(), done=None)
            if isinstance(picked, str):
                return picked
            self.delete(picked["id"])
            return f'Removed "{picked["text"]}" from your list.'
        add = ADD_REQUEST.match(normalized)
        if add:
            # Keep the person's own wording and capitals; the normalized match only locates it.
            raw = re.match(r"^(?:please )?(?:add|put) (?P<text>.+?) (?:to|on|onto) ", text.strip(), re.IGNORECASE)
            item_text = raw.group("text") if raw else add.group("text")
            try:
                entry = self.add(item_text)
            except ValueError as exc:
                return str(exc)
            count = len(self.open_items())
            return f'Added "{entry["text"]}". You have {count} thing{"" if count == 1 else "s"} to do.'
        return None

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
