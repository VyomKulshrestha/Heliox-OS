"""The scorecard lines from the program brief and their targets (pure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["max", "min", "zero"]


@dataclass(frozen=True)
class Line:
    """One measurable scorecard line.

    ``statistic`` names how scenario values are combined (median, max, rate, count, share). ``target`` is the
    limit; ``direction`` says whether the value must stay at or under it ("max"), reach it ("min"), or be 0.
    """

    id: str
    brief: str
    statistic: str
    target: float
    direction: Direction
    unit: str = ""


LINES: tuple[Line, ...] = (
    Line("1.1a", "Voice starts after the last word, answered locally (median)", "median", 0.8, "max", "s"),
    Line("1.1b", "Voice starts after the last word when a model answers (median)", "median", 1.5, "max", "s"),
    Line("1.1c", "Voice starts after the last word when a model answers (worst)", "max", 2.5, "max", "s"),
    Line("1.1d", "Longest silent gap with nothing said while working", "max", 2.5, "max", "s"),
    Line("1.2a", "Times Heliox started speaking over the person", "count", 0, "zero"),
    Line("1.2b", "Voice stops after the person speaks over it (worst)", "max", 0.4, "max", "s"),
    Line("1.2c", "Listens after being talked over (the interruption was heard)", "share", 1.0, "min"),
    Line("1.2d", "Voice not stopped by a cough, typing, silence or its own echo", "share", 1.0, "min"),
    Line("1.3", "Owner's sentences split at a mid-sentence pause", "rate", 0.05, "max"),
    Line("1.4a", "Spoken replies passing the plain-speech rubric", "share", 1.0, "min"),
    Line("1.4b", "Same stock phrase said twice running", "count", 0, "zero"),
    Line("2.1a", "Word error rate on the owner's sentences", "wer", 0.08, "max"),
    Line("2.1b", "Wake phrase found", "share", 0.95, "min"),
    Line("2.1c", "No wake phrase needed while a conversation is open", "share", 1.0, "min"),
    Line("2.2", "Speech not addressed to Heliox: acted on, spoken or stored", "count", 0, "zero"),
    Line("3.1", "Spoken slips raised as a card", "share", 1.0, "min"),
    Line("4.1", "Simple spoken request under way after the last word (worst)", "max", 3.0, "max", "s"),
    Line("4.2", "Control words handled (stop, wait, no, yes refused, correction, undo)", "share", 1.0, "min"),
    Line("Q", "Conversation quality rubric (fast spoken layer)", "share", 1.0, "min"),
)

BY_ID = {line.id: line for line in LINES}


def passes(line: Line, value: float | None) -> bool | None:
    """Whether ``value`` meets ``line``'s target; None when nothing was measured."""
    if value is None:
        return None
    if line.direction == "zero":
        return value == 0
    if line.direction == "max":
        return value <= line.target
    return value >= line.target
