"""App reminders: "remind me to submit my timesheet when I open Outlook".

An app reminder waits for an app or site (a word to look for in the front app name or
window title) to come to the front, then gives its reminder: once, or every time with
"every time I open Slack, remind me to set my status". Heliox looks at which window is
in front every ten seconds and compares it with the waiting reminders; it records
nothing. A reminder does not fire for a window that was already in front when it was
set: the app has to be opened, or switched to, again. Reminders are kept in
``DATA_DIR/app_reminders.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.workflows.app_reminders")

CHECK_EVERY_SECONDS = 10.0
MAX_REMINDERS = 50
MAX_TEXT = 300

_OPEN = r"(?:open|start|launch|switch to|go to|use|get to)"
SET_REQUEST = re.compile(
    rf"^(?:please )?remind me (?:to |about |that )?(?P<what>.+?) (?:when|next time|the next time|once) i {_OPEN} "
    rf"(?P<app>[\w .+-]+?)$"
    rf"|^(?:please )?(?:when|next time|the next time|once) i {_OPEN} (?P<app2>[\w .+-]+?),? "
    rf"remind me (?:to |about |that )?(?P<what2>.+)$"
    rf"|^(?:please )?(?:every time|whenever|each time) i {_OPEN} (?P<app3>[\w .+-]+?),? "
    rf"remind me (?:to |about |that )?(?P<what3>.+)$"
    rf"|^(?:please )?remind me (?:to |about |that )?(?P<what4>.+?) (?:every time|whenever|each time) i {_OPEN} "
    rf"(?P<app4>[\w .+-]+?)$"
    rf"|^(?:please )?remind me (?:when|next time|the next time) i {_OPEN} (?P<app5>[\w .+-]+?) (?:to|about|that) (?P<what5>.+)$"
)
LIST_REQUEST = re.compile(r"^(?:what|which) app reminders do i have$|^(?:list|show) my app reminders$")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.+-]", " ", text.casefold()).split()).strip(" .")


def _foreground() -> tuple[str, str]:
    from pilot.agents.screen_vision import _get_active_window

    return _get_active_window()


def in_front(app_word: str, app: str, title: str) -> bool:
    return app_word.casefold() in f"{app} {title}".casefold()


class AppReminders:
    """Reminders that wait for an app to come to the front."""

    def __init__(
        self,
        path: Path,
        *,
        deliver: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        foreground: Callable[[], tuple[str, str]] = _foreground,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._deliver = deliver
        self._foreground = foreground
        self._clock = clock
        self._front: dict[str, bool] = {}  # reminder id -> its app was in front at the last check
        self._task: asyncio.Task[None] | None = None

    def list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict) and item.get("app") and item.get("text")]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(
        self, app: str, text: str, *, every_time: bool = False, front: tuple[str, str] | None = None
    ) -> dict[str, Any]:
        app = " ".join(re.sub(r"^(?:the |my )", "", str(app).strip()).split())
        if app.endswith(" app"):
            app = app[: -len(" app")]
        text = " ".join(str(text).split())
        if not 2 <= len(app) <= 40:
            raise ValueError("Name the app or site, such as Outlook or YouTube")
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError(f"A reminder must be 1 to {MAX_TEXT} characters")
        items = self.list()
        if len(items) >= MAX_REMINDERS:
            raise ValueError(f"You can keep up to {MAX_REMINDERS} app reminders")
        entry = {
            "id": uuid.uuid4().hex[:10],
            "app": app,
            "text": text,
            "every_time": bool(every_time),
            "created_at": self._clock(),
        }
        items.append(entry)
        self._save(items)
        if front is not None:
            self._front[entry["id"]] = in_front(app, *front)
        return entry

    def delete(self, reminder_id: str) -> bool:
        items = self.list()
        kept = [item for item in items if item["id"] != reminder_id]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    async def check(self) -> int:
        """Give every reminder whose app just came to the front. Returns how many went out."""
        items = self.list()
        if not items:
            return 0
        app, title = await asyncio.to_thread(self._foreground)
        fired = []
        for item in items:
            now_front = in_front(item["app"], app, title)
            was_front = self._front.get(item["id"], now_front)  # unknown (after a restart): wait for the next switch
            self._front[item["id"]] = now_front
            if now_front and not was_front:
                fired.append(item)
        if not fired:
            return 0
        once = {item["id"] for item in fired if not item.get("every_time")}
        if once:
            self._save([item for item in items if item["id"] not in once])
        for item in fired:
            if self._deliver is not None:
                await self._deliver({"id": item["id"], "text": item["text"], "app": item["app"]})
        return len(fired)

    async def _loop(self) -> None:
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("App reminder check failed", exc_info=True)
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
        """Set or list app reminders from a sentence; None when it is not one."""
        normalized = _normalize(text)
        if LIST_REQUEST.match(normalized):
            items = self.list()
            if not items:
                return "You have no app reminders."
            listed = "; ".join(
                f"{'every time' if item.get('every_time') else 'when'} you open {item['app']}: {item['text']}"
                for item in items[:5]
            )
            return f"Your app reminders: {listed}."
        asked = SET_REQUEST.match(normalized)
        if not asked:
            return None
        groups = asked.groupdict()
        app = next(groups[key] for key in ("app", "app2", "app3", "app4", "app5") if groups.get(key))
        what = next(groups[key] for key in ("what", "what2", "what3", "what4", "what5") if groups.get(key))
        every_time = bool(groups.get("app3") or groups.get("app4"))
        try:
            front = self._foreground()
        except Exception:
            front = None
        try:
            entry = self.add(app, what, every_time=every_time, front=front)
        except ValueError as exc:
            return str(exc)
        when = "every time you open" if every_time else "next time you open"
        return f"Okay, {when} {entry['app']}, I'll remind you to {entry['text']}."

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
