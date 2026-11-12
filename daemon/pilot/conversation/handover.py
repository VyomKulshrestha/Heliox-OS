"""Handing a request to the planner while keeping the person company.

What the person hears: at once, a short sentence about what they asked ("Opening Notepad."), built in code from
what the router parsed, or the fast model's own sentence when the request needed the model to understand it. If
planning takes longer than about two seconds, one sentence of progress taken from the interaction state ("I'm
checking the plan for risks."), and another only when the state moves on. If an approval is needed, it says so
and points to the dialog; voice cannot approve. The result is spoken by the existing voice path.

The target of a hand-over is a choice. Today it is always the full planner, with its reviewers; a quick road for
plain requests can be added as another ``HandoverTarget`` that takes the same ``HandoverIntent``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pilot.conversation.router import ACTION_VERBS, Route

PROGRESS_AFTER_SECONDS = 2.0
PROGRESS_SPACING_SECONDS = 6.0
MAX_PROGRESS_LINES = 3
POLL_SECONDS = 0.2
SIMPLE_OBJECT_WORDS = 6


class HandoverTarget(StrEnum):
    FULL_PLANNER = "full_planner"  # plan, reviewers, approval, executor, verifier: unchanged authority
    QUICK = "quick"  # one allow-listed action, no planner call; same validator, world model, executor, logs


@dataclass(frozen=True, slots=True)
class HandoverIntent:
    """What the person wants done, structured so a planner can take it without re-reading the conversation."""

    verb: str
    object: str
    words: str  # what the person said, as recognised
    request: str  # the request written out on its own; the person's words when nothing needed resolving
    source: str  # "router" (parsed in code) or "model" (the fast model understood it)
    say: str = ""  # the fast model's own acknowledgement, when it wrote one


def choose_target(intent: HandoverIntent, *, quick: bool = False) -> HandoverTarget:
    """The quick road when a one-action plan on its allow-list was built for this intent, else the full planner."""
    return HandoverTarget.QUICK if quick else HandoverTarget.FULL_PLANNER


_PRONOUNS = {
    "my": "your",
    "me": "you",
    "i": "you",
    "i'm": "you're",
    "mine": "yours",
    "myself": "yourself",
    "our": "our",
    "your": "my",
    "you": "me",
}
_LEADING_FILLER = re.compile(r"^(?:up|on|the app|app|program|application)\s+", re.IGNORECASE)
_TRAILING_POLITE = re.compile(r"\s+(?:please|for me|now|right now|thanks|thank you)$", re.IGNORECASE)


def _speakable_object(target: str) -> str:
    words = [_PRONOUNS.get(word, word) for word in target.split()]
    text = _TRAILING_POLITE.sub("", " ".join(words))
    text = _TRAILING_POLITE.sub("", text)
    text = _LEADING_FILLER.sub("", text) if not text.startswith(("up ", "on ")) or len(text.split()) > 1 else text
    return text.strip()


_SPOKEN_UPPER = frozenset({"vs", "pdf", "ui", "os", "ide", "vlc", "obs", "aws", "gcp"})


def _title_app(name: str) -> str:
    """ "notepad" -> "Notepad", "vs code" -> "VS Code", leaving "your launch folder" alone."""
    try:
        from pilot.system.app_names import KNOWN
    except Exception:  # pragma: no cover - the module is pure, this is belt and braces
        KNOWN = {}
    known = KNOWN.get(name.replace(" ", "").casefold()) or KNOWN.get(name.casefold())
    if known:
        return known
    if len(name.split()) <= 2 and not name.startswith(("your ", "the ", "a ", "an ", "my ", "this ", "that ")):
        return " ".join(word.upper() if word in _SPOKEN_UPPER else word.capitalize() for word in name.split())
    return name


def acknowledgement(intent: HandoverIntent, *, last: str = "") -> str:
    """A short sentence about what was asked, never a claim that it is done."""
    if intent.say and (intent.source == "model" or not intent.object):
        sentence = intent.say
    else:
        sentence = _built(intent)
    if sentence.casefold() == last.casefold():
        sentence = "Okay, " + sentence[:1].lower() + sentence[1:]
    return sentence


MEDIA_ACKS = {
    "playpause": "Okay.",
    "nexttrack": "Next track.",
    "prevtrack": "Previous track.",
    "volumeup": "Turning it up.",
    "volumedown": "Turning it down.",
}


def _built(intent: HandoverIntent) -> str:
    verb = intent.verb.casefold()
    target = _speakable_object(intent.object)
    words = target.split()
    if verb == "remind":
        return "Setting that reminder."
    if verb == "set" and words[:2] in (["a", "timer"], ["timer"]):
        return f"Setting {target}."
    if verb == "check" and intent.source == "router":
        return "I'll check."
    if verb == "media":
        return MEDIA_ACKS.get(intent.object, "Okay.")
    if verb == "set" and words[:1] == ["volume"]:
        return f"Setting the {target}."
    if not target and verb in ACTION_VERBS and " " not in ACTION_VERBS[verb]:
        return f"{ACTION_VERBS[verb]}."  # "Muting."
    if not target or len(words) > SIMPLE_OBJECT_WORDS:
        gerund = ACTION_VERBS.get(verb, "")
        return f"{gerund} that now." if gerund and " " not in gerund else "I'll get that started."
    gerund = ACTION_VERBS.get(verb)
    if gerund is None:
        return f"I'll {verb} {target}."
    if verb in {"open", "launch", "start", "close", "quit", "exit", "visit", "browse"}:
        target = _title_app(target)
    if verb in {"search", "look"} and words[0] not in {"for", "up", "at", "in", "through"}:
        return f"{gerund} for {target}."
    return f"{gerund} {target}."


def intent_from_route(route: Route, words: str) -> HandoverIntent:
    return HandoverIntent(verb=route.verb, object=route.target, words=words, request=words, source="router")


# What the interaction state's phase means, said aloud. Planning and reviewing are the long ones.
def progress_sentence(phase: str, message: str) -> str:
    text = message.casefold()
    if phase == "awaiting_approval":
        return ""
    if "risk" in text or "review" in text or "double-check" in text or "gnomon" in text:
        return "I'm checking the plan for risks."
    if phase in {"planning", "understanding"} or "plan" in text:
        return "I'm working out the steps."
    if phase == "acting":
        return "Doing it now."
    if phase == "verifying":
        return "Checking that it worked."
    if phase == "correcting":
        return "Changing the plan."
    return "Still working on it."


APPROVAL_SENTENCE = "This one needs your approval. It's on the screen; press Approve if you want it to go ahead."

Speak = Callable[[str, str, str], Awaitable[str]]


async def keep_company(
    *,
    acknowledgement_text: str,
    speak: Speak,
    status: Callable[[], dict[str, Any]],
    on_said: Callable[[str], None] = lambda _text: None,
    clock: Callable[[], float] = time.monotonic,
    progress_after: float = PROGRESS_AFTER_SECONDS,
    spacing: float = PROGRESS_SPACING_SECONDS,
    poll: float = POLL_SECONDS,
) -> None:
    """Runs from the hand-over until the caller cancels it (the planner returned)."""
    started = clock()
    key = f"conv-ack:{started}"
    if acknowledgement_text:
        outcome = await speak(acknowledgement_text, "task_narration", key)
        if outcome in ("spoken", "interrupted"):
            on_said(acknowledgement_text)
    said_at = clock()
    lines = 0
    last_line = acknowledgement_text
    approval_said = False
    while True:
        await asyncio.sleep(poll)
        state = status() or {}
        phase = str(state.get("phase") or "")
        now = clock()
        if phase == "awaiting_approval" and not approval_said:
            approval_said = True
            outcome = await speak(APPROVAL_SENTENCE, "approval_risk", f"conv-approval:{started}")
            if outcome in ("spoken", "interrupted"):
                on_said(APPROVAL_SENTENCE)
            said_at = clock()
            continue
        if lines >= MAX_PROGRESS_LINES or now - started < progress_after:
            continue
        sentence = progress_sentence(phase, str(state.get("message") or ""))
        # The first progress line once planning runs long; later ones only when the work has moved on.
        if not sentence or sentence == last_line or (lines and now - said_at < spacing):
            continue
        lines += 1
        outcome = await speak(sentence, "task_narration", f"conv-progress:{started}:{lines}")
        if outcome in ("spoken", "interrupted"):
            on_said(sentence)
        last_line = sentence
        said_at = clock()
