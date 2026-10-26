"""Focus distractions: "during focus, tell me if I open YouTube".

The person names apps or sites (words to look for in the app name or window title). While
a focus session runs, Heliox looks at which window is in front every ten seconds and,
the first time one of those words shows up in a session, gives one gentle heads-up with
the time left. It never blocks, closes or records anything: the window it sees is
compared with the list and then forgotten. The words are kept in
``DATA_DIR/focus_distractions.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.focus_guard")

CHECK_EVERY_SECONDS = 10.0
MAX_WORDS = 20

ADD_REQUEST = re.compile(
    r"^(?:please )?(?:(?:during|while i m|while i am|when i m|when i am|while i) (?:focus|focusing|in focus)"
    r"(?: mode)?,? )?(?:tell|warn|remind|alert|nudge) me (?:if|when) i (?:open|go to|start|use|switch to) "
    r"(?P<what>.+?)(?: (?:during|while i m|while i am|while i|when i m) (?:focus|focusing|in focus)(?: mode)?)?$"
)
LIST_REQUEST = re.compile(
    r"^(?:what are|list|show) my (?:focus )?distractions$|^what do you warn me about(?: during focus)?$"
)
REMOVE_REQUEST = re.compile(
    r"^(?:please )?(?:stop warning me about (?P<a>.+?)(?: during focus)?"
    r"|remove (?P<b>.+?) from my (?:focus )?distractions)$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.]", " ", text.casefold()).split()).strip(" .")


def _foreground() -> tuple[str, str]:
    from pilot.agents.screen_vision import _get_active_window

    return _get_active_window()


class FocusGuard:
    """Distraction words in a small JSON file, checked against the front window during focus."""

    def __init__(
        self,
        path: Path,
        *,
        focus_status: Callable[[], dict[str, Any]] = lambda: {"active": False},
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        foreground: Callable[[], tuple[str, str]] = _foreground,
    ) -> None:
        self._path = Path(path)
        self._focus_status = focus_status
        self._deliver = deliver
        self._foreground = foreground
        self._warned: set[str] = set()
        self._session: float | None = None
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [str(word) for word in data if isinstance(word, str) and word.strip()] if isinstance(data, list) else []

    def _save(self, words: list[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(words, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, word: str) -> str:
        word = " ".join(re.sub(r"^(?:the |my )", "", str(word).strip()).split())
        if not 2 <= len(word) <= 40:
            raise ValueError("A distraction must be 2 to 40 characters, such as YouTube or reddit")
        words = [item for item in self.list() if item.casefold() != word.casefold()]
        if len(words) >= MAX_WORDS:
            raise ValueError(f"You can keep up to {MAX_WORDS} distractions")
        words.append(word)
        self._save(words)
        return word

    def remove(self, word: str) -> bool:
        words = self.list()
        kept = [item for item in words if item.casefold() != word.strip().casefold()]
        if len(kept) == len(words):
            return False
        self._save(kept)
        return True

    async def check(self) -> dict[str, Any] | None:
        """Look at the front window once; give a heads-up the first time a word shows up this session."""
        status = self._focus_status()
        if not status.get("active"):
            self._session = None
            self._warned = set()
            return None
        if status.get("started_at") != self._session:
            self._session = status.get("started_at")
            self._warned = set()
        words = [word for word in self.list() if word.casefold() not in self._warned]
        if not words:
            return None
        app, title = await asyncio.to_thread(self._foreground)
        seen = f"{app} {title}".casefold()
        word = next((word for word in words if word.casefold() in seen), None)
        if word is None:
            return None
        self._warned.add(word.casefold())
        minutes = max(1, round(int(status.get("remaining_seconds") or 0) / 60))
        event = {
            "word": word,
            "app": app,
            "text": f"Heads-up: {word} is open, and you're focusing for another {minutes} minute{'' if minutes == 1 else 's'}.",
        }
        if self._deliver is not None:
            await self._deliver(event)
        return event

    async def _loop(self) -> None:
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Focus distraction check failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def answer(self, text: str) -> str | None:
        """Add, list or remove distractions from a sentence; None when it is not about them."""
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            words = self.list()
            if not words:
                return 'You have no focus distractions. Say "during focus, tell me if I open YouTube" to add one.'
            return "During focus I'll give you a heads-up about " + ", ".join(words) + "."
        remove = REMOVE_REQUEST.match(normalized)
        if remove:
            word = (remove.group("a") or remove.group("b")).strip()
            return f"I'll stop warning you about {word}." if self.remove(word) else f"{word} isn't on your list."
        added = ADD_REQUEST.match(normalized)
        if not added:
            return None
        # Only a focus request: "tell me if I open YouTube" without focus is left to nudges and the planner.
        if not re.search(r"\bfocus", normalized):
            return None
        try:
            word = self.add(added.group("what"))
        except ValueError as exc:
            return str(exc)
        return f"Okay. During focus I'll give you one heads-up if {word} is in front. Nothing gets blocked."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
