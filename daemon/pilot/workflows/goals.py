"""Goals: long-lived threads Heliox keeps and helps you move forward.

"Start a goal: launch the podcast", "add to my podcast goal: recorded episode one",
"what's my next step on the podcast", "set the next step on the podcast to book a guest",
"what are my goals", "I finished the podcast goal". Each goal is a title, a running list
of your own updates, an optional next step and a status, kept in ``DATA_DIR/goals.json``
on this computer. Heliox surfaces your next steps in the morning briefing and the Today
tab; it only tracks and reminds, and acts on a goal only through the usual approved
requests. A goal with no update for a while is flagged so you can pick it back up.
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

MAX_GOALS = 40
MAX_TITLE = 80
MAX_TEXT = 300
MAX_UPDATES = 200
STALE_DAYS = 10

# Patterns run on punctuation-stripped text (no colon, no apostrophe), so the word
# "goal"/"project" is the separator and runtime matches the reservation check exactly.
_GOAL = r"(?:the |my )?(?P<name>.+?)(?: goal| project)?"
_CREATE = r"(?:start|add|create|set|make|track|new|begin) (?:a |an |me a )?(?:new )?(?:goal|project)(?: to| of| for| called| named)?"
CREATE_REQUEST = re.compile(
    rf"^(?:please )?{_CREATE} (?P<title>.+)$"
    r"|^(?:please )?(?:my (?:goal|project) is|i want my (?:goal|project) to be) (?P<title2>.+)$"
)
_UPDATE_LEAD = r"(?:add to|update|log progress on|log|note on|progress on|made progress on)"
UPDATE_REQUEST = re.compile(
    rf"^(?:please )?{_UPDATE_LEAD} (?:the |my )?(?P<name>.+?) (?:goal|project) (?P<text>.+)$"
    r"|^(?:please )?(?:on|for) (?:the |my )?(?P<name2>.+?) (?:goal|project) (?P<text2>.+)$"
)
NEXT_SET_REQUEST = re.compile(
    rf"^(?:please )?(?:set|make|change) (?:the |my )?next step (?:on|for|of) {_GOAL} (?:to|as) (?P<step>.+)$"
    r"|^(?:please )?next step (?:on|for) (?:the |my )?(?P<name2>.+?)(?: goal| project)? is (?P<step2>.+)$"
)
NEXT_GET_REQUEST = re.compile(
    rf"^(?:please )?what(?: s| is) (?:my |the )?next step (?:on|for|with|of) {_GOAL}$"
    r"|^(?:please )?(?:where am i (?:with|on)|how(?: s| is)) " + _GOAL.replace("name>", "name2>") + r"(?: going)?$"
)
LIST_REQUEST = re.compile(
    r"^(?:what are|list|show) (?:my )?(?:goals|projects)$|^what(?: s| is| are) (?:on )?my (?:goals|projects)"
    r"(?: list)?$|^(?:how are my|check (?:in on )?my) (?:goals|projects)$"
)
DONE_REQUEST = re.compile(
    rf"^(?:please )?(?:i(?: ve| have)? )?(?:finished|completed|done with|achieved) {_GOAL}$"
    r"|^(?:please )?(?:mark|close|complete) "
    + _GOAL.replace("name>", "name2>")
    + r"(?: as)?(?: done| complete| finished)$"
)
STATUS_REQUEST = re.compile(rf"^(?:please )?(?:pause|put on hold|hold|snooze|resume|reactivate|reopen) {_GOAL}$")
DELETE_REQUEST = re.compile(
    r"^(?:please )?(?:drop|delete|remove|abandon|give up on|forget) (?:the |my )?(?P<name>.+?)(?: goal| project)$"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _clean(text: str, limit: int = MAX_TEXT) -> str:
    return " ".join(str(text).split())[:limit].strip()


def _ago(seconds: float) -> str:
    days = int(seconds // 86400)
    if days <= 0:
        hours = int(seconds // 3600)
        return "today" if hours < 12 else "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return f"{days // 7} weeks ago"


class GoalStore:
    """Goals in a small JSON file, newest activity first."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return (
            [g for g in data if isinstance(g, dict) and g.get("id") and g.get("title")]
            if isinstance(data, list)
            else []
        )

    def _save(self, goals: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(goals, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def list(self) -> list[dict[str, Any]]:
        order = {"active": 0, "paused": 1, "done": 2}
        return sorted(self._load(), key=lambda g: (order.get(g.get("status"), 3), -g.get("updated_at", 0)))

    def active(self) -> list[dict[str, Any]]:
        return [g for g in self.list() if g.get("status") == "active"]

    def find(self, name: str, *, include_done: bool = False) -> dict[str, Any] | None:
        words = [w for w in _normalize(name).split() if w not in ("the", "my", "a", "goal", "project", "to")]
        if not words:
            return None
        best, best_score = None, 0
        for goal in self._load():
            if not include_done and goal.get("status") == "done":
                continue
            title = _normalize(goal["title"])
            score = sum(1 for w in words if w in title.split())
            if score > best_score or (score == best_score and score and goal.get("status") == "active"):
                best, best_score = goal, score
        return best if best_score else None

    def create(self, title: str) -> dict[str, Any]:
        title = _clean(title, MAX_TITLE)
        if not 2 <= len(title) <= MAX_TITLE:
            raise ValueError("A goal needs a title of 2 to 80 characters")
        goals = self._load()
        if any(_normalize(g["title"]) == _normalize(title) and g.get("status") != "done" for g in goals):
            raise ValueError(f'You already have a goal for "{title}"')
        if len([g for g in goals if g.get("status") != "done"]) >= MAX_GOALS:
            raise ValueError(f"You can keep up to {MAX_GOALS} open goals; finish or drop some first")
        now = time.time()
        goal = {
            "id": uuid.uuid4().hex[:10],
            "title": title,
            "status": "active",
            "next_step": "",
            "updates": [],
            "created_at": now,
            "updated_at": now,
        }
        goals.append(goal)
        self._save(goals)
        return goal

    def _mutate(self, goal_id: str, change) -> dict[str, Any] | None:
        goals = self._load()
        for goal in goals:
            if goal["id"] == goal_id:
                change(goal)
                goal["updated_at"] = time.time()
                self._save(goals)
                return goal
        return None

    def add_update(self, goal_id: str, text: str) -> dict[str, Any] | None:
        text = _clean(text)
        if not text:
            return None

        def change(goal: dict[str, Any]) -> None:
            goal["updates"] = (goal.get("updates") or [])[-(MAX_UPDATES - 1) :]
            goal["updates"].append({"at": time.time(), "text": text})

        return self._mutate(goal_id, change)

    def set_next(self, goal_id: str, step: str) -> dict[str, Any] | None:
        step = _clean(step)
        return self._mutate(goal_id, lambda g: g.__setitem__("next_step", step))

    def set_status(self, goal_id: str, status: str) -> dict[str, Any] | None:
        return self._mutate(goal_id, lambda g: g.__setitem__("status", status))

    def delete(self, goal_id: str) -> bool:
        goals = self._load()
        kept = [g for g in goals if g["id"] != goal_id]
        if len(kept) == len(goals):
            return False
        self._save(kept)
        return True

    def stale(self, days: int = STALE_DAYS) -> list[dict[str, Any]]:
        cutoff = time.time() - days * 86400
        return [g for g in self.active() if g.get("updated_at", 0) < cutoff]

    def next_steps(self) -> list[dict[str, Any]]:
        """Active goals that have a next step, for the morning briefing."""
        return [{"title": g["title"], "next_step": g["next_step"]} for g in self.active() if g.get("next_step")]

    def _describe(self, goal: dict[str, Any]) -> str:
        parts = [goal["title"]]
        if goal.get("status") != "active":
            parts[0] += f" ({goal['status']})"
        if goal.get("next_step"):
            parts.append(f"next: {goal['next_step']}")
        updates = goal.get("updates") or []
        if updates:
            parts.append(f"last update {_ago(time.time() - updates[-1]['at'])}: {updates[-1]['text']}")
        return "; ".join(parts)

    def answer(self, text: str) -> str | None:
        normalized = _normalize(text)
        raw = text.strip()

        if LIST_REQUEST.match(normalized):
            goals = [g for g in self.list() if g.get("status") != "done"]
            if not goals:
                return 'You have no goals yet. Say "start a goal: ..." to add one.'
            lines = []
            for goal in goals[:6]:
                step = f" (next: {goal['next_step']})" if goal.get("next_step") else ""
                tag = "" if goal["status"] == "active" else f" [{goal['status']}]"
                lines.append(f"{goal['title']}{tag}{step}")
            return f"You have {len(goals)} goal{'' if len(goals) == 1 else 's'}: " + "; ".join(lines) + "."

        create = CREATE_REQUEST.match(normalized)
        if create:
            m = re.match(
                r"(?i)^(?:please )?(?:(?:start|add|create|set|make|track|new|begin) (?:a |an |me a )?(?:new )?"
                r"(?:goal|project)(?: to| of| for| called| named)?|(?:my (?:goal|project) is|i want my (?:goal|project) to be))"
                r"[:,]? (.+)$",
                raw,
            )
            title = m.group(1) if m else (create.group("title") or create.group("title2"))
            try:
                goal = self.create(title)
            except ValueError as exc:
                return str(exc)
            return f'Started the goal "{goal["title"]}". Tell me updates, or set its next step.'

        step_set = NEXT_SET_REQUEST.match(normalized)
        if step_set:
            name = step_set.group("name") or step_set.group("name2")
            goal = self.find(name)
            if goal is None:
                return f"I don't have a goal like {name.strip()}."
            m = re.search(r"(?i)\b(?:to|as|is)[:,]? (.+)$", raw)
            step = m.group(1) if m else (step_set.group("step") or step_set.group("step2"))
            self.set_next(goal["id"], step)
            return f'Next step on "{goal["title"]}": {_clean(step)}.'

        step_get = NEXT_GET_REQUEST.match(normalized)
        if step_get:
            name = step_get.group("name") or step_get.group("name2")
            goal = self.find(name)
            if goal is None:
                return None  # "how is the weather" is not a goal question
            if goal.get("next_step"):
                return f'Your next step on "{goal["title"]}" is: {goal["next_step"]}.'
            updates = goal.get("updates") or []
            if updates:
                return f'No next step set on "{goal["title"]}" yet. Last update {_ago(time.time() - updates[-1]["at"])}: {updates[-1]["text"]}.'
            return f'No next step set on "{goal["title"]}" yet, and no updates logged. Tell me what you have in mind.'

        update = UPDATE_REQUEST.match(normalized)
        if update:
            name = update.group("name") or update.group("name2")
            goal = self.find(name)
            if goal is None:
                return None  # let the planner handle "update chrome" etc.
            m = re.search(r"(?i)(?:goal|project)[:,]? (.+)$", raw)
            body = m.group(1) if m else (update.group("text") or update.group("text2"))
            self.add_update(goal["id"], body)
            return f'Logged on "{goal["title"]}": {_clean(body)}.'

        done = DONE_REQUEST.match(normalized)
        if done:
            name = done.group("name") or done.group("name2")
            goal = self.find(name)
            if goal is None:
                return None
            self.set_status(goal["id"], "done")
            return f'Nice work. I\'ve marked "{goal["title"]}" done.'

        status = STATUS_REQUEST.match(normalized)
        if status:
            goal = self.find(status.group("name"), include_done=True)
            if goal is None:
                return None
            new_status = "paused" if re.match(r"(?:pause|put on hold|hold|snooze)", normalized) else "active"
            self.set_status(goal["id"], new_status)
            return f'Okay, "{goal["title"]}" is {new_status}.'

        delete = DELETE_REQUEST.match(normalized)
        if delete:
            goal = self.find(delete.group("name"), include_done=True)
            if goal is None:
                return None
            self.delete(goal["id"])
            return f'Dropped the goal "{goal["title"]}".'
        return None

    async def answer_async(self, text: str) -> str | None:
        return await asyncio.to_thread(self.answer, text)
