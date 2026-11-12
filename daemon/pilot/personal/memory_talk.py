"""Telling Heliox what to remember and forget, and asking what it knows, in the words people use.

The profile (``pilot.workflows.user_profile``) already takes "remember that I ...", "my sister is Ana", "I prefer
...", "forget that I ..." and "what do you know about me". Measured against ordinary phrasing, these slipped past it
and went to the planner as tasks (a model call and a plan for a sentence that only needed storing):

* "remember that the stand-up is at ten on Mondays", "remember my car is a blue Corolla", "note that Ana is vegan":
  anything not starting with "I";
* "what did I tell you about my car?", "what do you know about Ana?", "do you remember anything about the stand-up?";
* "forget what I just told you", "forget that last one";
* "forget that the stand-up is at ten" when nothing matched: the planner was asked to forget something.

This handler takes those, after the profile's own handler, and answers in one sentence that says exactly what was
stored, found or removed. Storing goes through ``UserProfile.add`` (so keys and passwords are refused and the entry
is mirrored into long-term memory); forgetting goes through ``UserProfile`` too, which retracts the mirrored fact.
English phrasing only; another language falls through to the conversation layer, which can still answer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pilot.personal.context import words
from pilot.personal.passing import Note, never_kept, passing_note

logger = logging.getLogger("pilot.personal.memory_talk")

_NORM = re.compile(r"[^\w\s'-]")

REMEMBER = re.compile(
    r"^(?:please )?(?:remember|keep in mind|note|make a note|bear in mind)(?: that)? (?P<text>.+)$", re.IGNORECASE
)
RECALL = re.compile(
    r"^(?:please )?(?:what did i (?:tell|say to) you about|what have i told you about|what do you know about"
    r"|what do you remember about|do you remember (?:anything )?about|tell me what you know about"
    r"|what did i say about) (?P<what>.+?)\??$",
    re.IGNORECASE,
)
FORGET_LAST = re.compile(
    r"^(?:please )?forget (?:what i just (?:said|told you)|that last (?:one|thing)|the last thing i (?:said|told you))$"
)
FORGET = re.compile(r"^(?:please )?forget (?:that |about )?(?P<what>.+?)$")
_SELF = frozenset({"me", "myself", "me please"})
_TOO_VAGUE = frozenset({"it", "that", "this", "them", "him", "her", "everything", "anything", "something"})
# "remember to buy milk" is a reminder, and "remember when ..." is a question: neither is a fact to keep.
_NOT_A_FACT = re.compile(r"^(?:to|when|where|how|what|why|who|if|whether|the time|me)\b", re.IGNORECASE)
# A fact states something: "remember that thing we talked about" has no verb that states anything.
_STATES = re.compile(
    r"\b(?:is|are|was|were|has|have|had|\w+'s|starts?|ends?|lives?|works?|likes?|loves?|hates?|needs?|wants?"
    r"|prefers?|goes|takes?|uses?|costs?|opens?|closes?|meets?|runs?|can|can't|cannot|will|won't|should|must"
    r"|doesn't|don't|isn't|aren't|wasn't|weren't|never|always|usually)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.casefold()).split())


def _clean(text: str) -> str:
    return " ".join(text.split()).strip().rstrip(".!")


def _second_person(text: str) -> str:
    for first, second in (("I'm ", "you're "), ("I am ", "you are "), ("I've ", "you've "), ("I ", "you ")):
        if text.startswith(first):
            return second + text[len(first) :]
    return re.sub(r"\bmy\b", "your", text)


class MemoryTalk:
    """Phrase handler over a ``UserProfile`` and the person's goals. ``answer`` returns None when not a phrase."""

    def __init__(
        self,
        *,
        profile: Callable[[], Any],
        goals: Callable[[], list[dict[str, Any]]] = list,
        now: Callable[[], datetime] = datetime.now,
        notes_enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        self._profile = profile
        self._goals = goals
        self._now = now
        self._notes_enabled = notes_enabled  # memory.passing_notes

    async def answer(self, text: str) -> str | None:
        profile = self._profile()
        if profile is None:
            return None
        raw = " ".join(str(text).split()).strip()
        normalized = _normalize(raw)
        if not normalized:
            return None
        prune = getattr(profile, "prune_lapsed", None)
        if prune is not None:
            try:
                await prune()  # notes said in passing whose day is over leave the profile and long-term memory
            except Exception:
                logger.warning("Could not remove lapsed notes", exc_info=True)

        if FORGET_LAST.match(normalized):
            removed = await profile.forget_last()
            return f'Okay, I\'ve forgotten "{removed}".' if removed else "You haven't told me anything to forget yet."

        recall = RECALL.match(raw)
        if recall:
            what = _clean(recall.group("what"))
            if _normalize(what) in _SELF:
                return None  # "what do you know about me" is the profile's own reply
            return self.recall(what)

        remember = REMEMBER.match(raw)
        if remember:
            fact = _clean(remember.group("text"))
            if (
                not fact
                or _NOT_A_FACT.match(fact)
                or _normalize(fact) in _TOO_VAGUE
                or len(fact.split()) < 3
                or not _STATES.search(fact)
            ):
                return None
            if re.match(r"(?i)^i\b", fact):
                return None  # the profile's own phrase, which it already declined
            try:
                entry = await profile.add("about", fact)
            except ValueError as error:
                return str(error)
            return f'Remembered: "{entry["text"]}".'

        forget = FORGET.match(normalized)
        if forget:
            what = forget.group("what").strip()
            if what in _TOO_VAGUE or not words(what):
                return None
            removed = await profile.forget_matching(what)
            if removed:
                return f'Okay, I\'ve forgotten "{removed}".'
            return f"I don't have anything stored about {what}, so there was nothing to forget."

        if not self._notes_on():
            return None
        note = passing_note(raw, self._now())
        if note is not None:
            why = never_kept(raw)
            if why:
                return f"I haven't kept that, because {why}. I don't keep notes like that from things said in passing."
            return await self._keep(profile, note)
        return None

    def _notes_on(self) -> bool:
        try:
            return bool(self._notes_enabled())
        except Exception:
            return True

    async def _keep(self, profile: Any, note: Note) -> str | None:
        """Keep a note said in passing, replacing an older note about the same thing."""
        for item in list(profile.get().get("about") or []):
            if item.get("note_thing") == note.thing:
                await profile.delete(item["id"])
        try:
            entry = await profile.add("about", note.text, note_thing=note.thing, expires=f"{note.expires:.0f}")
        except ValueError as error:
            return str(error)
        said = entry["text"].split(" (said on ", 1)[0]
        until = f"until {note.until}" if note.until else "for a week"
        return f'Noted: "{said}". I\'ll keep that {until}; say "forget about my {note.thing}" to drop it sooner.'

    def recall(self, what: str) -> str:
        """Everything stored that mentions ``what``, read back in the second person."""
        profile = self._profile()
        asked = words(what)
        if not asked:
            return f"You haven't told me anything about {what}."
        found: list[str] = []
        data = profile.get() if profile is not None else {}
        for key in ("about", "people", "prefers"):
            for item in data.get(key) or []:
                body = f"{item.get('text', '')} {item.get('name', '')} {item.get('relation', '')}"
                if asked <= words(body) or (len(asked) > 1 and len(asked & words(body)) >= 2):
                    text = str(item.get("text", ""))
                    if key == "people":
                        text = f"{item.get('name', '')} is your {item.get('relation', '')}"
                    elif key == "prefers":
                        text = f"you {text}"
                    found.append(_second_person(text))
        try:
            goals = list(self._goals() or [])
        except Exception:
            goals = []
        for goal in goals:
            if asked & words(f"{goal.get('title', '')} {goal.get('next_step', '')}"):
                found.append(f'you have a goal "{goal.get("title", "")}"')
        if not found:
            return f"You haven't told me anything about {what}."
        listed = "; ".join(dict.fromkeys(found[:4]))
        return f"You told me {listed}." if len(found) <= 4 else f"You told me {listed}, and a few more things."


__all__ = ["FORGET", "FORGET_LAST", "RECALL", "REMEMBER", "MemoryTalk"]
