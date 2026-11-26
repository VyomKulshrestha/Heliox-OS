"""Say it when a read returns only part of what it found.

A read that keeps the first N rows, entries or characters must tell both the next step and the person. The next step
reads the output text, so a line goes into the text; the person reads the reply and the task receipt, so the same
sentence goes into ``ActionResult.notice`` through ``OutputWithNotice``.
"""

from __future__ import annotations

from pilot.actions import OutputWithNotice

# Every cut line starts with this, so a reader (and a verifier) can tell a cut that was said from one that was not.
CUT_MARK = "[Cut: "


def cut_sentence(shown: int, total: int | None, noun: str) -> str:
    """'Only the first 50 of 120 matching elements are shown.' A total of None means 'more than shown'."""
    if total is None:
        return f"Only the first {shown:,} {noun} are shown; there are more."
    return f"Only the first {shown:,} of {total:,} {noun} are shown."


def cut_line(shown: int, total: int | None, noun: str) -> str:
    return f"{CUT_MARK}{cut_sentence(shown, total, noun)}]"


def with_cut(text: str, shown: int, total: int | None, noun: str) -> str:
    """``text`` with the cut said at its end and in the notice; ``text`` unchanged when nothing was cut."""
    if total is not None and total <= shown:
        return text
    line = cut_line(shown, total, noun)
    joined = f"{text}\n{line}" if text else line
    return OutputWithNotice(joined, cut_sentence(shown, total, noun))


def with_notes(text: str, notes: list[str]) -> str:
    """``text`` with each note appended as a cut line and joined into the notice; unchanged without notes."""
    notes = [note for note in notes if note]
    if not notes:
        return text
    lines = "\n".join(f"{CUT_MARK}{note}]" for note in notes)
    return OutputWithNotice(f"{text}\n{lines}" if text else lines, " ".join(notes))


def was_cut(text: str) -> bool:
    return CUT_MARK in (text or "")
