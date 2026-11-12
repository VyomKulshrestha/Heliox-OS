"""Getting to know the person: at most three questions, each one skippable, then it stops.

On a new install Heliox knows nothing about the person, and everything personal (offers, the relevant memory lines,
the way it talks) starts from that empty profile. This is the first conversation that fills it, offered once and
never pushed:

1. "What should I call you?"
2. "What do you mainly use this computer for?"
3. "Last one: short answers or more detail, and should I speak up when I notice something, or stay quiet unless
   you ask?"

Any answer can be "skip"; "stop", "later" or "that's enough" ends it at once. When it ends Heliox reads back every
line it stored, and each one can be removed (by voice, "forget ...", or on the card in the app).

Everything here is parsed in code, on this computer: no model is asked, so it works the same with a Claude key,
another provider, a local model or no model at all. What it stores goes into the profile (``UserProfile``, mirrored
into long-term memory as explicit facts), and from then on those lines go with later conversation turns to whichever
model is configured, exactly like any profile line (docs/INTERACTION_MODEL.md). The answers themselves never reach a
model: they are taken by Heliox's local handlers before the conversation layer.

"Stay quiet unless you ask" changes two settings, and says so: offers go off, and Heliox Live (if it is on) moves to
its quiet level, where it speaks only about risky things. English phrasing only; another language can use the form.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

IDLE_SECONDS = 300.0  # a session left unanswered this long is over

QUESTIONS = (
    "What should I call you?",
    "What do you mainly use this computer for?",
    "Last one: short answers or more detail, and should I speak up when I notice something, or stay quiet unless "
    "you ask?",
)
OPENING = "Three quick questions, and you can skip any of them by saying skip. "

START = re.compile(
    r"^(?:please )?(?:(?:let'?s )?get to know (?:me|each other)|ask me (?:some |a few )?questions(?: about me)?"
    r"|set up my profile|tell you about (?:me|myself)|let me tell you about myself)$"
)
SKIP = re.compile(
    r"^(?:skip|skip it|skip that|skip this one|pass|next|no|nope|none|nothing|rather not|i'?d rather not(?: say)?"
    r"|not telling|i don'?t want to say|no comment)$"
)
STOP = re.compile(
    r"^(?:stop|that'?s enough|enough|later|not now|cancel|never mind|let'?s do this later|done|i'?m done|no more)$"
)
ELSEWHERE = re.compile(
    r"^(?:open|close|start|launch|play|pause|set|remind|search|find|show|turn|send|delete|move|copy|undo|stop the"
    r"|what(?:'s| is| are| time)|how|when|where|why|who|can you|could you|would you)\b"
)
_NAME_LEAD = re.compile(
    r"^(?:you can call me|just call me|call me|my name is|my name'?s|i'?m called|i am called|i'?m|i am|it'?s|its"
    r"|just)\s+",
    re.IGNORECASE,
)
_USE_LEAD = re.compile(
    r"^(?:i\s+)?(?:mostly|mainly|usually|generally)?\s*(?:use it|use this computer|use the computer|it'?s|it is)?\s*"
    r"(?:mostly|mainly)?\s*(?:for\s+)?",
    re.IGNORECASE,
)
SHORT = re.compile(r"\b(?:short|shorter|brief|quick|concise|to the point|keep it short)\b")
DETAILED = re.compile(r"\b(?:detail|detailed|details|longer|thorough|explain|in depth|more)\b")
SPEAK_UP = re.compile(
    r"\b(?:speak up|tell me|point (?:it|things) out|let me know|chime in|interrupt|say something|mention it)\b"
)
QUIET = re.compile(
    r"\b(?:quiet|don'?t interrupt|do not interrupt|only when i ask|unless i ask|leave me alone|silent|keep quiet"
    r"|stay out)\b"
)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.casefold()).split())


def parse_name(text: str) -> str:
    """A name from "call me Sam", "Sam", "it's Dr Rao please". "" when it does not look like a name."""
    raw = " ".join(text.split()).strip(" .!?,")
    raw = _NAME_LEAD.sub("", raw).strip(" .!?,")
    raw = re.sub(r"(?i)[\s,]+(?:please|thanks|thank you)$", "", raw).strip(" .!?,")
    words = raw.split()
    if not 1 <= len(words) <= 3 or not all(re.fullmatch(r"[\w.'-]+", w) for w in words):
        return ""
    if any(w.casefold() in ("the", "a", "and", "for", "to", "of", "it", "is") for w in words):
        return ""
    return raw


def parse_use(text: str) -> str:
    raw = " ".join(text.split()).strip(" .!?,")
    rest = _USE_LEAD.sub("", raw, count=1).strip(" .!?,")
    return rest if 2 <= len(rest) <= 160 else ""


def parse_preferences(text: str) -> tuple[str, str]:
    """(length, speaking): length "short", "detailed" or ""; speaking "speak_up", "quiet" or ""."""
    norm = _norm(text)
    length = "short" if SHORT.search(norm) else "detailed" if DETAILED.search(norm) else ""
    speaking = "quiet" if QUIET.search(norm) else "speak_up" if SPEAK_UP.search(norm) else ""
    return length, speaking


from pilot.workflows.user_profile import QUIET_PREF, SPEAK_UP_PREF  # noqa: E402  (one place for both)


@dataclass
class Stored:
    id: str  # a UserProfile.list() row id: "call_me", "style:length" or an entry id
    text: str  # read back to the person, in the second person


@dataclass
class Session:
    step: int = 0
    at: float = 0.0
    stored: list[Stored] = field(default_factory=list)
    retried: bool = False


class GettingToKnow:
    """The three questions, by voice or typed chat, and the same thing as a form."""

    def __init__(
        self,
        *,
        profile: Callable[[], Any],
        set_quiet: Callable[[bool], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._profile = profile
        self._set_quiet = set_quiet
        self._clock = clock
        self.session: Session | None = None

    @property
    def active(self) -> bool:
        if self.session is not None and self._clock() - self.session.at > IDLE_SECONDS:
            self.session = None
        return self.session is not None

    def start(self) -> str:
        self.session = Session(at=self._clock())
        return OPENING + QUESTIONS[0]

    def current_question(self) -> str:
        """The question waiting for an answer, or "" when no session is open."""
        if not self.active or self.session is None:
            return ""
        return QUESTIONS[min(self.session.step, len(QUESTIONS) - 1)]

    def stop(self) -> str:
        return self._finish(stopped=True)

    async def answer(self, text: str) -> str | None:
        """The person's reply while a session is open; a phrase that starts one; otherwise None."""
        normalized = _norm(text)
        if not self.active:
            return self.start() if START.match(normalized) else None
        session = self.session
        assert session is not None
        session.at = self._clock()
        if STOP.match(normalized):
            return self._finish(stopped=True)
        if ELSEWHERE.match(normalized):
            # A request or a question, not an answer ("open my mail", "what's the time"): the questions end
            # quietly and the request goes where it would have gone.
            self.session = None
            return None
        if not SKIP.match(normalized):
            problem = await self._take(session.step, text)
            if problem and not session.retried:
                session.retried = True
                return problem
        return self._next()

    async def _take(self, step: int, text: str) -> str:
        """Store the answer to question ``step``. Returns a question to ask again, or ""."""
        profile = self._profile()
        session = self.session
        assert session is not None and profile is not None
        if step == 0:
            name = parse_name(text)
            if not name:
                return "Sorry, which name should I use? Or say skip."
            try:
                saved = await profile.set_name(name)
            except ValueError as error:
                return f"{error}. Which name should I use? Or say skip."
            session.stored.append(Stored("call_me", f"I call you {saved}"))
        elif step == 1:
            use = parse_use(text)
            if not use:
                return "Sorry, what do you mostly use it for? Or say skip."
            try:
                entry = await profile.add("about", f"I mainly use the computer for {use}")
            except ValueError as error:
                return f"{error} Or say skip."
            session.stored.append(Stored(entry["id"], f"you mainly use the computer for {use}"))
        elif step == 2:
            length, speaking = parse_preferences(text)
            if not length and not speaking:
                return "Sorry, I didn't catch that. Short answers or more detail? Or say skip."
            await self._apply(length, speaking)
        return ""

    async def _apply(self, length: str, speaking: str) -> None:
        profile = self._profile()
        session = self.session
        assert session is not None
        if length:
            await profile.set_style("length", length)
            session.stored.append(
                Stored("style:length", "you want short answers" if length == "short" else "you want more detail")
            )
        if speaking:
            text = QUIET_PREF if speaking == "quiet" else SPEAK_UP_PREF
            entry = await profile.add("prefers", text)
            session.stored.append(
                Stored(
                    entry["id"],
                    "I stay quiet unless you ask: offers are off, and Live speaks only about risky things"
                    if speaking == "quiet"
                    else "I speak up when I notice something",
                )
            )
            if self._set_quiet is not None:
                await self._set_quiet(speaking == "quiet")

    def _next(self) -> str:
        session = self.session
        assert session is not None
        session.step += 1
        session.retried = False
        if session.step >= len(QUESTIONS):
            return self._finish()
        return QUESTIONS[session.step]

    def _finish(self, *, stopped: bool = False) -> str:
        session, self.session = self.session, None
        stored = session.stored if session is not None else []
        if not stored:
            return (
                "Okay, I didn't keep anything. You can tell me about yourself any time."
                if not stopped
                else "Okay, we can do this another time."
            )
        lines = "; ".join(item.text for item in stored)
        return (
            f"Thanks. Here's what I kept: {lines}. To change any of it, say forget and what it was, or remove it in "
            "Settings, under About you."
        )

    def last_stored(self) -> list[Stored]:
        return list(self.session.stored) if self.session is not None else []

    async def save_form(self, *, name: str = "", use: str = "", length: str = "", speaking: str = "") -> dict[str, Any]:
        """The same three answers from the app's form. Empty fields are skipped. Returns what was stored."""
        self.session = Session(at=self._clock())
        problems: list[str] = []
        if name.strip():
            problems.append(await self._take(0, name))
        if use.strip():
            problems.append(await self._take(1, f"for {use}"))
        if length in ("short", "detailed") or speaking in ("speak_up", "quiet"):
            await self._apply(
                length if length in ("short", "detailed") else "", speaking if speaking in ("speak_up", "quiet") else ""
            )
        stored = [item.__dict__ for item in self.session.stored]
        reply = self._finish()
        return {"stored": stored, "reply": reply, "problems": [p for p in problems if p]}


__all__ = ["QUESTIONS", "GettingToKnow", "parse_name", "parse_preferences", "parse_use"]
