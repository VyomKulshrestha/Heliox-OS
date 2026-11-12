"""The plain-speech rubric a script can check on anything Heliox says aloud (pure).

Checks: length (a sentence or two), no markdown or symbols read out, no list, the facts the context requires,
no claim of an action that did not happen, and not the same stock phrase as the previous reply. A model-judged
rubric (sensible, in context) is a separate, opt-in slot because it costs money; see ``judge_prompt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_WORDS = 40
_SYMBOLS = re.compile(r"[*#_`|<>\[\]{}~^\\]|https?://|\.{4,}|->|=>")
_LIST = re.compile(r"(^|\n)\s*(?:[-•]|\d+[.)])\s+")
_ACTION_CLAIMS = re.compile(
    r"\b(i(?:'ve| have)? (?:opened|sent|deleted|moved|created|saved|scheduled|set|booked|emailed|closed|"
    r"turned (?:on|off)|started|launched|installed|removed|renamed|written|wrote)|done[.!]|it'?s done)\b",
    re.IGNORECASE,
)


@dataclass
class RubricResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def stock_key(text: str) -> str:
    """The opening of a reply, the part that repeats when Heliox falls back on a stock phrase."""
    return " ".join(w.casefold() for w in _words(text)[:4])


def check(
    reply: str,
    *,
    required_facts: tuple[str, ...] = (),
    action_happened: bool = False,
    previous: str = "",
    max_words: int = MAX_WORDS,
) -> RubricResult:
    failures: list[str] = []
    words = _words(reply)
    if not words:
        failures.append("empty")
    if len(words) > max_words:
        failures.append(f"too long ({len(words)} words)")
    if _SYMBOLS.search(reply):
        failures.append("markdown or symbols")
    if _LIST.search(reply) or reply.count(";") >= 2:
        failures.append("reads out a list")
    low = reply.casefold()
    missing = [fact for fact in required_facts if fact.casefold() not in low]
    if missing:
        failures.append("missing facts: " + ", ".join(missing))
    if not action_happened and _ACTION_CLAIMS.search(reply):
        failures.append("claims an action")
    if previous and stock_key(previous) and stock_key(previous) == stock_key(reply) and len(words) > 2:
        failures.append("same stock phrase as the previous reply")
    return RubricResult(not failures, failures)


def repeats_running(replies: list[str]) -> int:
    """How many replies open with the same stock phrase as the reply before them."""
    count = 0
    for before, after in zip(replies, replies[1:], strict=False):
        if stock_key(before) and stock_key(before) == stock_key(after) and len(_words(after)) > 2:
            count += 1
    return count


def judge_prompt(turns: list[dict[str, str]], reply: str, context: str) -> str:
    """The opt-in model-judged rubric's prompt (not sent anywhere by this module)."""
    history = "\n".join(f"{t['who']}: {t['text']}" for t in turns)
    return (
        "You are grading one spoken reply from a desktop assistant. Answer with JSON "
        '{"sensible": 0-2, "in_context": 0-2, "direct": 0-2, "why": "..."}.\n'
        f"What is on screen and known:\n{context}\n\nConversation so far:\n{history}\n\nReply spoken aloud:\n{reply}"
    )
