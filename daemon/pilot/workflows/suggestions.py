"""Proactive suggestions: one timely, gentle nudge from what Heliox already keeps.

"What should I do now?", "any suggestions?" and "what should I work on?" compute a single
suggestion in code from your goals, habits, to-do list and focus state: a goal that has
gone quiet, a habit streak you haven't kept today, open to-dos late in the day, or a
goal's next step. Nothing is sent to a model, and Heliox never acts on a suggestion; it
only offers one, and every suggestion is something you can say back to it.

With ``[suggestions] enabled`` on (off by default), a background check offers one at most
once every few hours, held during a focus session, and never more than once for the same
thing in a day. It is a quieter cousin of usage nudges and the co-adaptation study: it
changes nothing on its own.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

logger = logging.getLogger("pilot.workflows.suggestions")

CHECK_EVERY_SECONDS = 1800.0  # look every 30 minutes
MIN_GAP_SECONDS = 3 * 3600  # at most one proactive suggestion every few hours

ASK_REQUEST = re.compile(
    r"^(?:please )?(?:what should i (?:do|work on|focus on)(?: now| next| today)?"
    r"|what(?:'s| is) next|any suggestions|got any suggestions|what do you suggest"
    r"|give me a suggestion|help me (?:get started|focus))$"
)
TOGGLE_REQUEST = re.compile(
    r"^(?:please )?turn (?P<state>on|off) (?:proactive )?suggestions$"
    r"|^(?:please )?(?:start|stop) suggesting things$"
)


def _part_of_day(now: datetime) -> str:
    if now.hour < 12:
        return "morning"
    if now.hour < 17:
        return "afternoon"
    return "evening"


class Suggester:
    """Picks one suggestion from injected readers of the local stores. Pure and testable."""

    def __init__(
        self,
        *,
        goals: Callable[[], list[dict[str, Any]]] = list,
        stale_goals: Callable[[], list[dict[str, Any]]] = list,
        habits: Callable[[], list[dict[str, Any]]] = list,
        open_todos: Callable[[], list[dict[str, Any]]] = list,
        focus_active: Callable[[], bool] = lambda: False,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._goals = goals
        self._stale_goals = stale_goals
        self._habits = habits
        self._open_todos = open_todos
        self._focus_active = focus_active
        self._now = now

    def suggest(self) -> dict[str, Any] | None:
        """The single best suggestion right now, or None when nothing is worth raising."""
        now = self._now()
        part = _part_of_day(now)

        # 1. A goal that has gone quiet, with a next step: the clearest "pick this back up".
        for goal in self._stale_goals():
            if goal.get("next_step"):
                return {
                    "kind": "stale_goal",
                    "key": f"stale:{goal['id']}",
                    "text": f'Your goal "{goal["title"]}" has been quiet for a while. Its next step is {goal["next_step"]}.',
                    "say": f"remind me to {goal['next_step']}",
                }

        # 2. A habit streak not kept today, once the day is underway.
        if part != "morning":
            for habit in self._habits():
                if habit.get("streak", 0) >= 2 and not habit.get("today"):
                    return {
                        "kind": "habit",
                        "key": f"habit:{habit['name']}",
                        "text": f"You're on a {habit['streak']}-day streak for {habit['name']} and haven't logged it today.",
                        "say": f"log {habit['name']}",
                    }

        # 3. Open to-dos late in the day.
        todos = self._open_todos()
        if part == "evening" and todos:
            count = len(todos)
            return {
                "kind": "todos",
                "key": f"todos:{now.date().isoformat()}",
                "text": f'You still have {count} thing{"" if count == 1 else "s"} on your list; the first is "{todos[0]["text"]}".',
                "say": "what's on my list",
            }

        # 4. Any active goal's next step, as a gentle default.
        for goal in self._goals():
            if goal.get("status") == "active" and goal.get("next_step"):
                return {
                    "kind": "goal_step",
                    "key": f"step:{goal['id']}:{goal['next_step']}",
                    "text": f'On "{goal["title"]}", your next step is {goal["next_step"]}.',
                    "say": None,
                }
        return None

    def spoken(self) -> str:
        """The on-demand answer: a suggestion, or a calm "nothing pressing"."""
        pick = self.suggest()
        if pick is None:
            return "Nothing pressing right now. You could start a goal, or tell me what you're working on."
        text = pick["text"]
        if pick.get("say"):
            text += f' If you like, say "{pick["say"]}".'
        return text

    def answer(self, text: str) -> str | None:
        return self.spoken() if ASK_REQUEST.match(" ".join(re.sub(r"[^\w\s']", " ", text.casefold()).split())) else None

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)


class SuggestionLoop:
    """Offers one suggestion now and then, only when turned on. Suggestion-only; never acts."""

    def __init__(
        self,
        suggester: Suggester,
        *,
        enabled: Callable[[], bool],
        deliver: Callable[[dict[str, Any]], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: datetime.now().date().isoformat(),
    ) -> None:
        self._suggester = suggester
        self._enabled = enabled
        self._deliver = deliver
        self._clock = clock
        self._today = today
        self._last_at = 0.0
        self._seen: dict[str, str] = {}  # suggestion key -> day last offered
        self._task: asyncio.Task[None] | None = None

    async def tick(self) -> bool:
        """Offer one suggestion if due and not recently offered. Returns whether one went out."""
        if not self._enabled():
            return False
        now = self._clock()
        if self._last_at and now - self._last_at < MIN_GAP_SECONDS:
            return False
        pick = self._suggester.suggest()
        if pick is None:
            return False
        today = self._today()
        if self._seen.get(pick["key"]) == today:
            return False  # already raised this exact thing today
        self._seen = {key: day for key, day in self._seen.items() if day == today}
        self._seen[pick["key"]] = today
        self._last_at = now
        await self._deliver(pick)
        return True

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Proactive suggestion check failed", exc_info=True)
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
