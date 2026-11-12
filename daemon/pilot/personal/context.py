"""The few things Heliox knows about the person that bear on what they just said, for the fast model.

The conversation layer already sends up to eight profile lines with every turn, in the order they were stored. With
a profile of any size that is the oldest eight: preferences and the answer style (stored after the facts) are the
first to fall off, and "book dinner near home" never reaches the model with "I live in Pune". This module adds one
context line per turn (``ContextSources.extra``) with what fits *this* utterance:

* the answer style and stated preferences, always (they apply to every reply);
* people named, or whose relation is named ("my sister" finds Ana);
* facts about the person, active goals and habits that share a word with the utterance.

Retrieval is word overlap with a light stem, in code, over the local profile, goals and habits files. No model call,
no embeddings, no network, a fraction of a millisecond on any machine. Lines the layer already sends (the first
``PROFILE_LINES`` of the profile) are not repeated, and the line stays under ``MAX_CHARS`` so it fits the layer's
small context budget. What it holds goes to whichever conversation model is configured, exactly like the profile
lines (the privacy table in docs/INTERACTION_MODEL.md).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("pilot.personal.context")

PROFILE_LINES = 8  # what pilot.conversation.context already sends (kept in step with its PROFILE_LINES)
MAX_CHARS = 360
MAX_ITEMS = 5
MAX_PREFERENCES = 3
NOTE_TTL_SECONDS = 30.0  # the utterance noted for the turn about to be answered

_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "about",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "we",
        "our",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "just",
        "please",
        "hey",
        "heliox",
        "okay",
        "ok",
        "so",
        "not",
        "no",
        "yes",
        "some",
        "any",
        "all",
        "get",
        "got",
        "go",
        "going",
        "make",
        "let",
        "tell",
        "say",
        "said",
        "know",
        "want",
        "need",
        "like",
        "think",
        "one",
        "up",
        "out",
        "now",
        "then",
        "than",
        "too",
        "very",
        "really",
    ]
)


def stem(word: str) -> str:
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def words(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in _STOP and len(w) > 1}


def relevant_lines(
    text: str,
    *,
    profile: dict[str, Any] | None,
    goals: list[dict[str, Any]] | None = None,
    habits: list[dict[str, Any]] | None = None,
    already: list[str] | None = None,
) -> list[str]:
    """The lines worth adding for this utterance, most useful first. Pure: tested on every system."""
    profile = profile or {}
    asked = words(text)
    folded = text.casefold()
    sent = {line.casefold() for line in already or []}
    out: list[tuple[float, str]] = []

    def add(score: float, line: str) -> None:
        if line.casefold() not in sent and all(line != existing for _, existing in out):
            out.append((score, line))

    for field_name, value in (profile.get("style") or {}).items():
        add(10.0, f"Answer style, {field_name}: {value}")
    for item in (profile.get("prefers") or [])[-MAX_PREFERENCES:]:
        add(9.0, f"Preference: {item.get('text', '')}")
    for item in profile.get("people") or []:
        name, relation = str(item.get("name") or ""), str(item.get("relation") or "")
        named = name and re.search(rf"\b{re.escape(name.casefold())}\b", folded)
        related = relation and re.search(rf"\bmy {re.escape(relation.casefold())}\b", folded)
        if named or related:
            add(8.0, f"Person they know: {item.get('text') or f'{relation} {name}'}")
    for item in profile.get("about") or []:
        overlap = asked & words(str(item.get("text", "")))
        if overlap:
            add(2.0 + len(overlap), f"About them: {item.get('text', '')}")
    for goal in goals or []:
        if goal.get("status", "active") != "active":
            continue
        body = f"{goal.get('title', '')} {goal.get('next_step', '')}"
        overlap = asked & words(body)
        if overlap or "goal" in asked:
            step = f", next step: {goal['next_step']}" if goal.get("next_step") else ""
            add(1.5 + len(overlap), f"Their goal: {goal.get('title', '')}{step}")
    for habit in habits or []:
        overlap = asked & words(str(habit.get("name", "")))
        if overlap or "habit" in asked or "streak" in asked:
            add(1.0 + len(overlap), f"Habit they track: {habit.get('name', '')}, {habit.get('streak', 0)}-day streak")
    out.sort(key=lambda pair: -pair[0])
    return [line for _, line in out[:MAX_ITEMS]]


def as_context(lines: list[str], limit: int = MAX_CHARS) -> str:
    if not lines:
        return ""
    block = "Also relevant about the person:"
    for line in lines:
        piece = " " + line.rstrip(".") + "."
        if len(block) + len(piece) > limit:
            break
        block += piece
    return block if block.endswith(".") else ""


class PersonalContext:
    """Holds the utterance being answered and gives the layer its relevant line (``extra()``)."""

    def __init__(
        self,
        *,
        profile: Callable[[], Any],
        goals: Callable[[], list[dict[str, Any]]] = list,
        habits: Callable[[], list[dict[str, Any]]] = list,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._profile = profile
        self._goals = goals
        self._habits = habits
        self._clock = clock
        self._noted = ""
        self._noted_at = float("-inf")

    def note(self, text: str) -> None:
        """The utterance about to be answered, for a layer whose extra callables are not given the words."""
        self._noted = str(text or "")
        self._noted_at = self._clock()

    def lines_for(self, text: str) -> list[str]:
        profile_obj = self._profile()
        if profile_obj is None:
            return []
        try:
            data = profile_obj.get()
            already = _as_sent(list(profile_obj.context_lines()))[:PROFILE_LINES]
        except Exception:
            return []
        try:
            goals = list(self._goals() or [])
        except Exception:
            logger.warning("Could not read goals for the conversation context", exc_info=True)
            goals = []
        try:
            habits = list(self._habits() or [])
        except Exception:
            logger.warning("Could not read habits for the conversation context", exc_info=True)
            habits = []
        return relevant_lines(text, profile=data, goals=goals, habits=habits, already=already)

    def extra(self, text: str = "") -> str:
        """The line for the words being answered: given by the layer when it passes them, else the noted ones."""
        if not text:
            if not self._noted or self._clock() - self._noted_at > NOTE_TTL_SECONDS:
                return ""
            text = self._noted
        return as_context(self.lines_for(text))


def _as_sent(lines: list[str]) -> list[str]:
    """The profile lines in the order the conversation layer sends them (it may put style and preferences first)."""
    try:
        from pilot.conversation.adapters import prioritise_profile
    except ImportError:
        return lines
    return list(prioritise_profile(lines))


__all__ = ["PersonalContext", "as_context", "relevant_lines", "stem", "words"]
