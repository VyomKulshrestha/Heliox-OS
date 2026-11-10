"""What the person asked for a task, in their own words, kept apart from the prompts Heliox builds from it.

A live correction ("Skip the backup.") makes Heliox plan again from a prompt that wraps the request in its own
headings and instructions ("[LIVE USER CORRECTION] ... Revise the plan to honor the live correction."). That prompt
is for the planner only. Everything a person reads or that is kept about the task (Activity log, receipts, the Today
view, the daily brief, plan history, checkpoints, the task's memory entry) uses ``TaskRequest.display()``: the
request, then each correction as "Then: ...".

Entries written before this split carry the planner prompt. ``readable_request`` renders those for display by
cutting at the first of Heliox's own markers and keeping the corrections it finds; stored history is not rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Headings Heliox puts into planner prompts. None of them is ever the person's own words.
LIVE_CORRECTION_MARKER = "[LIVE USER CORRECTION]"
OBSERVED_OUTCOMES_MARKER = "[ALREADY OBSERVED OUTCOMES]"
COMPANION_REVIEW_MARKER = "[INDEPENDENT COMPANION REVIEW]"
ATTACHED_CONTEXT_MARKER = "\n\nAttached Context:\n"
_MARKERS = (LIVE_CORRECTION_MARKER, OBSERVED_OUTCOMES_MARKER, COMPANION_REVIEW_MARKER)
_ADDITIONAL = re.compile(r"\n\s*Additional correction:\s*")


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    return text if not text or text[-1] in ".!?" else text + "."


def split_corrections(text: str) -> list[str]:
    """One live correction message may carry several ("first\\nAdditional correction: second")."""
    return [" ".join(part.split()) for part in _ADDITIONAL.split(str(text).strip()) if part.strip()]


@dataclass(frozen=True)
class TaskRequest:
    """The person's request and the corrections they made while it ran, in order."""

    words: str
    corrections: tuple[str, ...] = field(default=())
    # What had already run when the latest correction arrived ("file_copy on C:\a: completed"). Heliox's own
    # observation, never shown as the request; the reviewers need it so a step that already ran is not "missing".
    completed_before: tuple[str, ...] = field(default=())

    def with_corrections(self, correction_text: str, completed: list[str] | tuple[str, ...] = ()) -> TaskRequest:
        return TaskRequest(self.words, (*self.corrections, *split_corrections(correction_text)), tuple(completed))

    def display(self) -> str:
        """For people and records: "<request> Then: <correction>." for each correction."""
        parts = [" ".join(self.words.split())]
        parts += [f"Then: {_sentence(correction)}" for correction in self.corrections]
        return " ".join(part for part in parts if part)

    def for_reviewers(self) -> str:
        """For the safety critic and the companion: the request, what the person then said, and what had already
        run, in plain sentences without the planner's headings or instructions."""
        lines = [self.words.strip()]
        lines += [f"The person then said: {_sentence(correction)}" for correction in self.corrections]
        if self.corrections:
            done = "; ".join(self.completed_before) if self.completed_before else "nothing"
            lines.append(f"Already done before the correction: {done}.")
        return "\n".join(line for line in lines if line)


def readable_request(stored: str) -> str:
    """Render a stored request for display. Text from before the split (a planner prompt) is cut at the first of
    Heliox's markers, and the corrections it holds are kept as "Then: ..."; anything else is returned unchanged."""
    text = str(stored or "")
    cut = min((text.find(marker) for marker in _MARKERS if marker in text), default=-1)
    if cut < 0:
        return text
    corrections: list[str] = []
    for section in text.split(LIVE_CORRECTION_MARKER)[1:]:
        body = section
        for marker in (*_MARKERS, "\n\nRevise the plan"):
            index = body.find(marker)
            if index >= 0:
                body = body[:index]
        corrections += split_corrections(body)
    return TaskRequest(text[:cut].strip(), tuple(corrections)).display()
