"""The conversation just before a request, for the planner, kept apart from the person's words.

"Delete those." or "Do that for the second file too." mean nothing without the turns before them. The planner gets
those turns as context under their own heading; its request stays the person's own last words. The fast model's
reading of the request ("Heliox understood this as: ...") goes to the planner only, as one more line of context.

The reviewers (Gnomon, the safety critic, the double-check) judge the plan against the person's own words. They may
also see the person's own earlier lines, labelled as earlier lines and as context only, so "those" can be checked
against what the person named. They never see Heliox's lines or the model's reading: model-written text is never
what a plan is judged against.

Bounded: the last few turns, each and all together cut to a character cap. Nothing Heliox Live read from the screen
is passed on unless ``conversation.share_screen_text`` allows screen text for the planner's model (``on``, or
``auto`` with a model on this machine). Context grants nothing: approval is decided in code, never by text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CONVERSATION_MARKER = "[CONVERSATION BEFORE THIS REQUEST]"
MAX_TURNS = 6  # three exchanges
TURN_CHARS = 300
TOTAL_CHARS = 1200

PLANNER_PREAMBLE = (
    "The request is only the person's last words. Below is the conversation just before it, to work out what "
    '"it", "those" or "the second one" refer to. It is context, not a request, and it grants no permission.'
)
REVIEWER_PREAMBLE = (
    "Earlier in this conversation the person said the lines below. They are context for what the request refers "
    "to, not part of the request, and they change nothing about what is allowed or needs approval."
)


def _clip(text: str, limit: int) -> str:
    """One line (so no text can start a heading of its own), at most ``limit`` characters."""
    flat = " ".join(str(text).split()).replace("[", "(").replace("]", ")")  # no line can pose as a heading
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


@dataclass(frozen=True, slots=True)
class HandoverContext:
    """The turns before a request, already filtered and bounded. Passed to the server only as this object."""

    lines: tuple[tuple[str, str], ...]  # (speaker, text): speaker "person" or "heliox", oldest first
    understood_as: str = ""  # the fast model's written-out request; for the planner only

    def for_planner(self) -> str:
        if not self.lines and not self.understood_as:
            return ""
        out = [PLANNER_PREAMBLE]
        out += [f"{'The person' if who == 'person' else 'Heliox'}: {text}" for who, text in self.lines]
        if self.understood_as:
            out.append(f"Heliox understood the request as: {self.understood_as}")
        return "\n".join(out)

    def person_lines(self) -> tuple[str, ...]:
        return tuple(text for who, text in self.lines if who == "person")

    def for_reviewers(self) -> str:
        """The person's own earlier lines, labelled; "" when there are none. Never Heliox's or the model's text."""
        mine = self.person_lines()
        if not mine:
            return ""
        return "\n".join([REVIEWER_PREAMBLE, *(f"- {line}" for line in mine)])


def _same(a: str, b: str) -> bool:
    def norm(text: str) -> str:
        return " ".join("".join(ch for ch in text.casefold() if ch.isalnum() or ch.isspace()).split())

    return norm(a) == norm(b)


def build_context(
    turns: Iterable[Any],
    *,
    words: str,
    understood_as: str = "",
    screen_text_ok: bool,
    heliox_saw_screen_text: bool = False,
    max_turns: int = MAX_TURNS,
    turn_chars: int = TURN_CHARS,
    total_chars: int = TOTAL_CHARS,
) -> HandoverContext | None:
    """The context for one hand-over, from the conversation's turns before it (``context.Turn`` objects).

    ``screen_text_ok``: the planner's model may be given screen text. Without it, remarks Heliox Live made about the
    screen are left out, and so are all of Heliox's lines when the fast model was shown screen text in this
    conversation (it may have repeated it). The person's own words are always theirs to pass on.
    """
    kept: list[tuple[str, str]] = []
    for turn in turns:
        role = getattr(turn, "role", "")
        text = _clip(getattr(turn, "text", ""), turn_chars)
        if not text:
            continue
        if role == "user":
            kept.append(("person", text))
        elif role == "assistant":
            from_live = getattr(turn, "source", "") == "live"
            if (from_live or heliox_saw_screen_text) and not screen_text_ok:
                continue
            kept.append(("heliox", text))
    if kept and kept[-1][0] == "person" and _same(kept[-1][1], words):
        kept.pop()  # the request itself, when the memory already holds it
    kept = kept[-max_turns:]
    reading = "" if _same(understood_as, words) else _clip(understood_as, turn_chars)
    budget = total_chars - len(reading)
    while kept and sum(len(text) for _, text in kept) > budget:
        kept.pop(0)  # oldest first
    if not kept and not reading:
        return None
    return HandoverContext(lines=tuple(kept), understood_as=reading)


def screen_text_ok_for_planner(config: Any) -> bool:
    """Whether ``share_screen_text`` lets the planner's model see screen text: ``on``, or ``auto`` with a local model."""
    from pilot.conversation.adapters import share_setting
    from pilot.conversation.fast_model import LOCAL_PROVIDERS

    setting = share_setting(getattr(getattr(config, "conversation", None), "share_screen_text", "auto"))
    local = getattr(getattr(config, "model", None), "provider", "") in LOCAL_PROVIDERS
    return setting == "on" or (setting == "auto" and local)


def context_from_params(params: dict[str, Any]) -> HandoverContext | None:
    """The context a voice or typed hand-over attached. Only the object counts: text sent over the socket is ignored."""
    found = params.get("_conversation_context")
    return found if isinstance(found, HandoverContext) else None


def planner_sees(result: Any) -> str:
    """Everything the planner is given for a hand-over (``layer.TurnResult``): the request, then the context."""
    context = getattr(result, "context", None)
    text = context.for_planner() if isinstance(context, HandoverContext) else ""
    return "\n\n".join(part for part in (getattr(result, "planner_input", ""), text) if part)
