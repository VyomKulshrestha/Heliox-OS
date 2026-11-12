"""Checks on what the fast model said, decided in code (no judge model). Pure functions over the turn record."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STOCK_OPENERS = re.compile(
    r"^(?:sure|certainly|of course|absolutely|great question|good question|as an ai|i'd be happy|happy to help|"
    r"no problem|okay so|well,|so,)\b",
    re.IGNORECASE,
)
UNSPEAKABLE = (
    ("markdown", re.compile(r"\*\*|__|`|^\s*#|^\s*[-*•]\s|^\s*\d+[.)]\s|\|", re.MULTILINE)),
    ("path", re.compile(r"\b[A-Za-z]:\\|(?<![\w.])/(?:home|users|usr|etc|var|tmp)/", re.IGNORECASE)),
    ("link", re.compile(r"https?://|www\.", re.IGNORECASE)),
    ("symbol", re.compile(r"[{}<>\[\]=^~]|->|=>|&&")),
    ("emoji", re.compile("[\U0001f300-\U0001faff☀-➿]")),
)
DONT_KNOW = re.compile(
    r"\b(?:i don'?t know|i'm not sure|i am not sure|i can'?t (?:tell|see|know|access|check|read)|i have no way|"
    r"i don'?t have (?:access|that|any|a way)|i can'?t say|no idea|not something i can (?:see|tell)|"
    r"i don'?t see|isn'?t something i can|i'm not able to (?:see|tell|check)|i can only see)\b",
    re.IGNORECASE,
)
# Thinking aloud or describing the person instead of speaking to them.
NARRATION = re.compile(
    r"\bthe (?:person|user)\b|\bi(?:'ll| will) ask (?:one|a|them)\b|\b(?:my|the) (?:reply|response) (?:should|will)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text or "") if s.strip()]


def opening(text: str) -> str:
    words = re.findall(r"[a-z']+", (text or "").casefold())
    return " ".join(words[:2])


@dataclass
class TurnRecord:
    """What one turn did, as the harness saw it."""

    text: str
    route: str
    done: bool
    spoken: list[str]
    raw: str  # the model's text before cleaning
    planner_input: str = ""
    quick_types: list[str] = field(default_factory=list)
    first_sentence_s: float | None = None


@dataclass
class Outcome:
    check: str
    passed: bool
    detail: str = ""


def heard(record: TurnRecord) -> str:
    return " ".join(record.spoken).strip()


def check_turn(record: TurnRecord, spec: dict[str, Any], previous: TurnRecord | None) -> list[Outcome]:
    """Every check the turn's spec asks for. ``previous``: the turn before it in the same conversation."""
    out: list[Outcome] = []
    said = heard(record)
    talk = record.done
    if talk and said:
        count = len(sentences(said))
        out.append(Outcome("length", 1 <= count <= 3, f"{count} sentences: {said!r}"))
        found = [name for name, pattern in UNSPEAKABLE if pattern.search(said)]
        out.append(Outcome("speakable", not found, f"{', '.join(found)} in {said!r}"))
        out.append(Outcome("no_stock_opener", not STOCK_OPENERS.match(said), said[:60]))
        out.append(Outcome("speaks_to_person", not NARRATION.search(said), said))
        if previous is not None and previous.done and heard(previous):
            same = opening(said) == opening(heard(previous)) and opening(said) != ""
            out.append(Outcome("fresh_opening", not same, f"{opening(said)!r} twice running"))
    expected = spec.get("expect")
    if expected in ("talk", "act"):
        predicted = "talk" if talk else "act"
        out.append(
            Outcome(
                "handover",
                predicted == expected,
                f"expected {expected}, got {predicted}: {said or record.planner_input!r}",
            )
        )
    if spec.get("quick") is not None:
        is_quick = bool(record.quick_types)
        out.append(Outcome("quick_road", is_quick == spec["quick"], f"quick={record.quick_types}"))
    if spec.get("never_quick"):
        out.append(Outcome("never_dangerous_quick", not record.quick_types, f"quick={record.quick_types}"))
    for word in spec.get("mentions", []):
        target = said if talk else record.planner_input
        out.append(
            Outcome(
                spec.get("mentions_check", "mentions"),
                word.casefold() in target.casefold(),
                f"{word!r} not in {target!r}",
            )
        )
    for word in spec.get("never_mentions", []):
        target = said if talk else record.planner_input
        out.append(Outcome("grounded", word.casefold() not in target.casefold(), f"{word!r} invented in {target!r}"))
    if spec.get("dont_know"):
        out.append(Outcome("dont_know", talk and bool(DONT_KNOW.search(said)), said))
    if spec.get("asks"):
        out.append(Outcome("misheard_asks", talk and "?" in said, said or record.planner_input))
    if spec.get("no_claim"):
        claimed = re.search(r"\b(?:i(?:'ve| have) approved|approved it|done,? approved|i approved)\b", said, re.I)
        out.append(Outcome("never_approves", not claimed, said))
    return out


def summarise(results: list[tuple[str, TurnRecord, list[Outcome]]]) -> dict[str, Any]:
    """Pass rate per check, hand-over precision and recall, and every failure quoted."""
    per: dict[str, list[bool]] = {}
    failures = []
    tp = fp = fn = tn = 0
    for spec_text, _record, outcomes in results:
        for outcome in outcomes:
            per.setdefault(outcome.check, []).append(outcome.passed)
            if not outcome.passed:
                failures.append({"turn": spec_text, "check": outcome.check, "detail": outcome.detail})
            if outcome.check == "handover":
                expected_act = outcome.detail.startswith("expected act")
                got_act = ", got act" in outcome.detail
                tp += expected_act and got_act
                fp += (not expected_act) and got_act
                fn += expected_act and not got_act
                tn += (not expected_act) and not got_act
    rates = {name: {"passed": sum(v), "of": len(v), "rate": round(sum(v) / len(v), 3)} for name, v in per.items()}
    return {
        "checks": rates,
        "handover": {
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "failures": failures,
    }
