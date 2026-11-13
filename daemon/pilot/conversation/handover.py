"""Handing a request to the planner while keeping the person company.

What the person hears: at once, a short sentence about what Heliox will do ("I'll open your launch folder."), built
in code from what the router parsed or the fast model understood, never that it is already doing it; nothing first
when the plan will need an approval (pilot.conversation.action_talk). If
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

from pilot.conversation.action_talk import asked_sentence, said_again
from pilot.conversation.router import Route

PROGRESS_AFTER_SECONDS = 2.0
PROGRESS_SPACING_SECONDS = 9.0  # a new line every 8 to 10 seconds while the work goes on: never a long silence
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
    """The first sentence for a request handed to the planner, from the action's asked-for state
    (pilot.conversation.action_talk): what Heliox will do, never that it is doing or has done it, and "" when the
    plan will need an approval first (the approval sentence is then the first thing said)."""
    verb = intent.verb.casefold()
    target = _speakable_object(intent.object)
    words = target.split()
    if verb in {"open", "launch", "start", "close", "quit", "exit", "visit", "browse"} and target:
        target = _title_app(target)
    if verb == "set" and words[:1] == ["volume"]:
        target = f"the {target}"
    if verb == "media":
        sentence = "Okay."
    else:
        sentence = asked_sentence(verb, target, intent.say)
    if sentence and sentence.casefold() == last.casefold():
        sentence = said_again(sentence)
    return sentence


MEDIA_ACKS = {
    "playpause": "Okay.",
    "nexttrack": "Next track.",
    "prevtrack": "Previous track.",
    "volumeup": "Turning it up.",
    "volumedown": "Turning it down.",
}


def intent_from_route(route: Route, words: str) -> HandoverIntent:
    return HandoverIntent(verb=route.verb, object=route.target, words=words, request=words, source="router")


# What the interaction state's phase means, said aloud. Planning and reviewing are the long ones.
# Said when the work has not moved on since the last line: true, specific, and never the same twice running.
STILL = {
    "planning": ("I'm still working out the steps.", "Still planning; this one takes a while.", "Still on the plan."),
    "reviewing": ("I'm still checking the plan for risks.", "Still checking the plan."),
    "acting": ("Still doing it.", "Still working through the steps."),
    "verifying": ("Still checking that it worked.", "Still checking the result."),
    "": ("Still working on it.", "I'm still on it."),
}
LOCAL_SLOW = "The model on this computer is slow the first time. I'm still planning."
_STEP = re.compile(r"\bstep (\d+) of (\d+)\b", re.IGNORECASE)


def _kind(phase: str, message: str) -> str:
    text = message.casefold()
    if "risk" in text or "review" in text or "double-check" in text or "gnomon" in text:
        return "reviewing"
    if phase in {"planning", "understanding"} or "plan" in text:
        return "planning"
    if phase in {"acting", "verifying"}:
        return phase
    return ""


def still_sentence(phase: str, message: str, last: str) -> str:
    """A line for work that has not moved on: which step it is on when the state says so, else a "still" line
    for its kind, never ``last``."""
    step = _STEP.search(message)
    if step and f"On step {step.group(1)} of {step.group(2)}." != last:
        return f"On step {step.group(1)} of {step.group(2)}."
    options = [line for line in STILL[_kind(phase, message)] if line != last]
    return options[0] if options else ""


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
# The person changed a request while its dialog was open: that dialog was closed (never approved), and then a new one.
CHANGING = "Okay, I've closed that one and I'm changing it."
APPROVAL_CHANGED = "The changed one is on the screen; press Approve if you want it."

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
    user_speaking: Callable[[], bool] = lambda: False,
    local_slow_once: Callable[[], bool] = lambda: False,
    carried: tuple[str, float] | None = None,
) -> None:
    """Runs from the hand-over until the caller cancels it (the planner returned).

    The acknowledgement at once; after ``progress_after`` a line saying what is happening; then, as long as the
    work goes on, a new line every ``spacing`` seconds: what changed when the state moved on, which step it is on,
    or a "still" line for the same work, never the same sentence twice running. Nothing is said while the person is
    talking. ``local_slow_once``: True once per session when the plan comes from a model on this computer, so the
    first long wait says why."""
    started = clock()
    key = f"conv-ack:{started}"
    if acknowledgement_text:
        outcome = await speak(acknowledgement_text, "task_narration", key)
        if outcome in ("spoken", "interrupted"):
            on_said(acknowledgement_text)
    said_at = clock()
    lines = 0
    last_line = acknowledgement_text
    if carried is not None:
        # The sentence's first half, withdrawn a moment ago, already said this line: it counts as said here, so the
        # joined request does not say it again and the next line comes on the same cadence.
        last_line, said_at, lines = carried[0], carried[1], 1
    last_state = ("", "")
    # The first half's approval sentence stands for the joined request: the person was just told to look at the
    # dialog, and the joined plan's dialog takes its place.
    approval_said = carried is not None and carried[0] == APPROVAL_SENTENCE
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
        if now - started < progress_after or (lines and now - said_at < spacing) or phase == "awaiting_approval":
            continue
        if user_speaking():
            continue  # never over the person; the line waits for the next poll
        message = str(state.get("message") or "")
        moved_on = (phase, message) != last_state
        sentence = progress_sentence(phase, message) if moved_on else ""
        if lines and not moved_on and _kind(phase, message) == "planning" and local_slow_once():
            sentence = LOCAL_SLOW
        if not sentence or sentence == last_line:
            sentence = still_sentence(phase, message, last_line)
        if not sentence:
            continue
        last_state = (phase, message)
        lines += 1
        outcome = await speak(sentence, "task_narration", f"conv-progress:{started}:{lines}")
        if outcome in ("spoken", "interrupted"):
            on_said(sentence)
        last_line = sentence
        said_at = clock()
