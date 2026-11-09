"""Earlier turns of a long conversation, so a detail from twenty turns back is not lost.

The planner sends the model only the last few turns of a session (6, or 10 in full mode). Everything before that
used to be dropped. In a long working session that loses exactly the things people set up first: "today's report is
for the Porto warehouse", "use the blue template".

This keeps them without a model call or a summary that might drift: from the older turns it takes the first two
(where a session usually sets its subject) and the ones that share the most words with the new request, puts them
back in order, shortens each, and stops at a token budget. The planner adds the result to its memory context as
"Earlier in this conversation".
"""

from __future__ import annotations

from typing import Any

from pilot.memory.assembler import memory_terms
from pilot.memory.sliding_window import get_token_count

FIRST_TURNS = 2
MAX_TURN_CHARS = 220
EARLIER_TURNS_LOOKED_AT = 40
_STOP = frozenset(
    [
        "this",
        "that",
        "with",
        "from",
        "have",
        "what",
        "when",
        "where",
        "which",
        "will",
        "would",
        "could",
        "should",
        "about",
        "into",
        "your",
        "their",
        "there",
        "them",
        "they",
        "then",
        "than",
        "were",
        "been",
        "being",
        "make",
        "made",
        "please",
        "thanks",
        "just",
        "also",
        "some",
        "more",
        "most",
        "only",
        "over",
    ]
)


def _terms(text: str) -> frozenset[str]:
    return frozenset(term for term in memory_terms(text) if len(term) >= 4 and term not in _STOP)


def _line(entry: dict[str, Any]) -> str:
    asked = " ".join(str(entry.get("user_input", "")).split())
    answered = " ".join(str(entry.get("explanation", "")).split())
    line = f"- They asked: {asked}"
    if answered:
        line += f" Heliox: {answered}"
    return line if len(line) <= MAX_TURN_CHARS else line[: MAX_TURN_CHARS - 1] + "…"


def earlier_turns(older: list[dict[str, Any]], query: str, max_tokens: int) -> str:
    """``older`` is oldest first: the session's turns before the ones the planner already sends."""
    if not older or max_tokens <= 0:
        return ""
    wanted = _terms(query)
    scored = []
    for position, entry in enumerate(older):
        overlap = len(wanted & _terms(f"{entry.get('user_input', '')} {entry.get('explanation', '')}"))
        opening = position < FIRST_TURNS
        if overlap or opening:
            scored.append((overlap + (1 if opening else 0), -position, position))
    chosen: list[int] = []
    header = "Earlier in this conversation:"
    for _, _, position in sorted(scored, reverse=True):
        trial = sorted([*chosen, position])
        text = "\n".join([header, *(_line(older[i]) for i in trial)])
        if get_token_count(text) > max_tokens:
            continue
        chosen = trial
    if not chosen:
        return ""
    return "\n".join([header, *(_line(older[i]) for i in chosen)])


__all__ = ["EARLIER_TURNS_LOOKED_AT", "earlier_turns"]
