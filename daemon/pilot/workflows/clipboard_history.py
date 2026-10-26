"""Clipboard history: "what did I copy earlier?", "copy the second one back".

Off by default (``[clipboard] history``). When on, Heliox checks the clipboard every two
seconds and remembers the last 25 pieces of text you copied, for up to eight hours. The
history lives in memory only: it is never written to disk, never included in backups,
and a restart or turning the setting off forgets it. Nothing is kept while a password
manager or a private browsing window is in front (the activity timeline's excluded
apps), and text that looks like a password, key, token or card number is never kept.

Putting an earlier item back on the clipboard is the person's own request about their
own text, so it happens right away, like pressing Copy.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("pilot.workflows.clipboard_history")

CHECK_EVERY_SECONDS = 2.0
MAX_ITEMS = 25
MAX_TEXT = 10_000
KEEP_SECONDS = 8 * 3600

LIST_REQUEST = re.compile(
    r"^what (?:did|have) i (?:just )?cop(?:y|ied)(?: earlier| before| recently| today| lately)?$"
    r"|^(?:show|read|list) (?:me )?my clipboard(?: history)?$|^what s (?:in )?my clipboard history$"
)
COPY_BACK_REQUEST = re.compile(
    r"^(?:please )?(?:copy|put) (?:back )?(?:the )?(?P<which>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|"
    r"\d|previous|last|other|link|url|email|email address)(?: one| thing| item| link| i copied| from before)?"
    r"(?: back)?(?: on (?:the|my) clipboard| to (?:the|my) clipboard| again)?$"
)
COPY_BACK_MARKER = re.compile(r"\b(?:back|again|i copied|from before)\b")
CLEAR_REQUEST = re.compile(
    r"^(?:please )?(?:clear|delete|erase) (?:my |the )?clipboard history$|^forget what i copied$"
)

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_ORDINALS |= {"fifth": 5, "5th": 5, "previous": 2, "last": 2, "other": 2}
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*(?:PRIVATE KEY|CERTIFICATE)-----"),
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[0-9A-Za-z]{10,}"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghu|ghr)_[0-9A-Za-z]{20,}|\bgithub_pat_[0-9A-Za-z_]{20,}"),
    re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\beyJ[0-9A-Za-z_-]{8,}\.[0-9A-Za-z_-]{8,}\.[0-9A-Za-z_-]{8,}"),
    re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_ -]?key|token)\s*[:=]\s*\S+"),
)
_URL = re.compile(r"^https?://\S+$")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|~/|\\\\)")


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _entropy(text: str) -> float:
    counts = Counter(text)
    return -sum(count / len(text) * math.log2(count / len(text)) for count in counts.values())


def looks_secret(text: str) -> bool:
    """True for text that looks like a password, key, token or card number."""
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return True
    compact = re.sub(r"[\s-]", "", text)
    if compact.isdigit() and 13 <= len(compact) <= 19 and _luhn(compact):
        return True
    token = text.strip()
    if any(char.isspace() for char in token) or not 8 <= len(token) <= 200:
        return False
    if _URL.match(token) or _EMAIL.match(token) or _PATH.match(token):
        return False
    classes = sum(
        (
            any(char.islower() for char in token),
            any(char.isupper() for char in token),
            any(char.isdigit() for char in token),
            any(not char.isalnum() for char in token),
        )
    )
    return classes >= 3 and _entropy(token) >= 3.0


def _preview(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _foreground() -> tuple[str, str]:
    from pilot.agents.screen_vision import _get_active_window

    return _get_active_window()


def change_detector() -> Callable[[], bool] | None:
    """On Windows, a cheap "did the clipboard change?" check, so it is only opened when it did.

    Opening the clipboard every two seconds can briefly block other apps from using it.
    Elsewhere there is no cheap check and the clipboard is read each time.
    """
    import sys

    if sys.platform != "win32":
        return None
    try:
        import ctypes

        sequence = ctypes.windll.user32.GetClipboardSequenceNumber  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    last = [None]

    def changed() -> bool:
        current = sequence()
        if current == last[0]:
            return False
        last[0] = current
        return True

    return changed


class ClipboardHistory:
    """The last few things copied, in memory only, while the setting is on."""

    def __init__(
        self,
        *,
        enabled: Callable[[], bool],
        excluded: Callable[[str, str], bool] = lambda app, title: False,
        read: Callable[[], Awaitable[str]] | None = None,
        write: Callable[[str], Awaitable[Any]] | None = None,
        foreground: Callable[[], tuple[str, str]] = _foreground,
        clock: Callable[[], float] = time.time,
        changed: Callable[[], bool] | None = None,
    ) -> None:
        self._enabled = enabled
        self._changed = changed
        self._excluded = excluded
        self._read = read
        self._write = write
        self._foreground = foreground
        self._clock = clock
        self._items: list[dict[str, Any]] = []  # newest first
        self._last_seen: str | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled())

    def list(self) -> list[dict[str, Any]]:
        cutoff = self._clock() - KEEP_SECONDS
        self._items = [item for item in self._items if item["at"] >= cutoff]
        return [dict(item) for item in self._items]

    def clear(self) -> int:
        removed = len(self._items)
        self._items = []
        return removed

    def remember(self, text: str, app: str = "", title: str = "") -> bool:
        """Keep ``text`` as the newest item, unless it is empty, too long, private or secret."""
        if not self.enabled or not text.strip() or len(text) > MAX_TEXT:
            return False
        if self._excluded(app, title) or looks_secret(text):
            return False
        self._items = [item for item in self._items if item["text"] != text]
        self._items.insert(0, {"id": uuid.uuid4().hex[:10], "text": text, "at": self._clock(), "app": app})
        del self._items[MAX_ITEMS:]
        return True

    async def check(self) -> bool:
        """Read the clipboard once and remember it if it changed. Returns whether something was kept."""
        if not self.enabled:
            if self._items:
                self.clear()
            self._last_seen = None
            return False
        if self._read is None:
            return False
        if self._changed is not None and not self._changed():
            return False
        try:
            text = await self._read()
        except Exception:
            logger.debug("Clipboard read failed", exc_info=True)
            return False
        if text == "(clipboard is empty)" or text == self._last_seen:
            return False
        self._last_seen = text
        try:
            app, title = await asyncio.to_thread(self._foreground)
        except Exception:
            app, title = "", ""
        return self.remember(text, app, title)

    async def _loop(self) -> None:
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Clipboard history check failed", exc_info=True)
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

    def pick(self, which: str) -> dict[str, Any] | None:
        items = self.list()
        if which in ("link", "url"):
            return next((item for item in items if _URL.match(item["text"].strip())), None)
        if which in ("email", "email address"):
            return next((item for item in items if _EMAIL.match(item["text"].strip())), None)
        index = int(which) if which.isdigit() else _ORDINALS.get(which, 0)
        return items[index - 1] if 1 <= index <= len(items) else None

    async def copy_back(self, item_id: str) -> dict[str, Any] | None:
        """Put an earlier item back on the clipboard and make it the newest."""
        item = next((item for item in self._items if item["id"] == item_id), None)
        if item is None or self._write is None:
            return None
        await self._write(item["text"])
        self._last_seen = item["text"]
        self._items = [item] + [other for other in self._items if other["id"] != item_id]
        item["at"] = self._clock()
        return dict(item)

    async def answer_async(self, text: str) -> str | None:
        """List, copy back or clear clipboard history from a sentence; None when it is not about it."""
        normalized = _normalize(text)
        listing = LIST_REQUEST.match(normalized)
        copy = COPY_BACK_REQUEST.match(normalized)
        if copy and not COPY_BACK_MARKER.search(normalized):
            copy = None  # "copy the link" on its own is a browser task for the planner
        clear = CLEAR_REQUEST.match(normalized)
        if not (listing or copy or clear):
            return None
        if clear:
            removed = self.clear()
            return (
                f"Forgot {removed} copied item{'' if removed == 1 else 's'}."
                if removed
                else "There was nothing to forget."
            )
        if not self.enabled:
            return 'Clipboard history is off. Say "turn on clipboard history" or turn it on in Settings.'
        items = self.list()
        if listing:
            if not items:
                return "I haven't seen you copy anything since clipboard history was turned on."
            listed = "; ".join(f"{number}. {_preview(item['text'])}" for number, item in enumerate(items[:5], 1))
            return f"You copied: {listed}."
        item = self.pick(copy.group("which"))
        if item is None:
            return "I can't find that in what you copied."
        if await self.copy_back(item["id"]) is None:
            return "I couldn't put that back on the clipboard."
        return f'Copied "{_preview(item["text"], 40)}" again. It\'s ready to paste.'
