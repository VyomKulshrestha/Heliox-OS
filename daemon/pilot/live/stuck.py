"""Noticing when someone is stuck, from what the live layer already sees, and offering one concrete kind of help.

A companion that only corrects slips misses the moments people most want help: rewriting the same sentence for the
fourth time, flicking between two windows to copy something across, running a command that fails the same way
again. Each leaves a plain trace in what Live already reads every 0.6 s (the focused text and which window it is
in), so no new sensor, keystroke hook or camera is needed:

- **Rewriting.** In the same text field, three or more times within two minutes, they delete at least 15
  characters, and the text has not grown by more than 40 characters since the first time. They are circling.
- **Back and forth.** Eight or more switches between the same two windows within a minute.
- **The same error again.** One error line from the error detector appears three or more times in a terminal.

Each sign raises at most one low-urgency card per window every 15 minutes, never spoken, with one offer the person
can accept by saying "fix it" or clicking: a clearer way to say the sentence, the two windows side by side, or a
look at the error. The offer goes to the planner through the normal approval path; nothing happens on its own.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

from pilot.live.detectors import LOW, Correction, TextContext, error_on_screen
from pilot.system.app_names import app_label

KIND = "stuck"
DELETED_CHARS = 15
REWRITES = 3
REWRITE_WINDOW_SECONDS = 120.0
MAX_GROWTH_CHARS = 40
SWITCHES = 8
SWITCH_WINDOW_SECONDS = 60.0
ERROR_REPEATS = 3
QUIET_SECONDS = 900.0  # one offer per sign per window in 15 minutes
MAX_QUOTE = 200


@dataclass
class _Field:
    peak: int = 0  # the longest the text got since the last deletion
    last: int = 0
    counted: bool = False  # this deletion has been counted
    rewrites: deque[tuple[float, int]] = field(default_factory=deque)  # (when, length at the first deletion)


def _app_name(window: str) -> str:
    return app_label(window.split("|", 1)[0]) or "that window"


def _last_sentence(text: str) -> str:
    tail = " ".join(text.split())[-MAX_QUOTE:]
    parts = re.split(r"(?<=[.!?])\s+", tail)
    return (parts[-1] if parts[-1].strip() else tail).strip()


class StuckWatcher:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._fields: dict[str, _Field] = {}
        self._switches: deque[tuple[float, str, str]] = deque()
        self._window = ""
        self._offered: dict[tuple[str, str], float] = {}

    def observe(
        self,
        text: str,
        window: str,
        ctx: TextContext,
        now: float | None = None,
        *,
        writing: bool = True,
        field: str = "",
    ) -> list[Correction]:
        """One look: the focused text and its window. Returns any offers that are due.

        ``writing`` is False when focus is on page text, a label or a button: the window still counts for the
        back-and-forth sign, but there is no text of the person's to look at. ``field`` tells two text fields
        in the same window apart, so moving between them is not read as deleting text.
        """
        now = self._clock() if now is None else now
        found: list[Correction] = []
        if self._window and window != self._window:
            self._switches.append((now, self._window, window))
            found += self._back_and_forth(now, window)
        self._window = window
        if not writing:
            return found
        if ctx.is_terminal:
            found += self._repeated_error(text, window, ctx, now)
        elif not ctx.is_code_editor:
            found += self._rewriting(text, window, now, field)
        return found

    # ── the three signs ──

    def _rewriting(self, text: str, window: str, now: float, field: str = "") -> list[Correction]:
        # The field, not the title: many editors put the text itself or an unsaved marker in the title.
        place = f"{window.split('|', 1)[0]}|#{field}" if field else f"{window}|"
        state = self._fields.setdefault(place, _Field(peak=len(text), last=len(text)))
        length = len(text)
        while state.rewrites and now - state.rewrites[0][0] > REWRITE_WINDOW_SECONDS:
            state.rewrites.popleft()
        growing = length > state.last
        state.last = length
        if growing:
            if state.counted:  # typing again after a deletion: the next deletion is a new one
                state.counted, state.peak = False, length
            state.peak = max(state.peak, length)
            return []
        if state.counted or state.peak - length < DELETED_CHARS:
            return []  # still the same deletion (it spans several reads), or too small to count
        state.counted = True
        state.rewrites.append((now, state.peak))
        if len(state.rewrites) < REWRITES or length - state.rewrites[0][1] > MAX_GROWTH_CHARS:
            return []
        sentence = _last_sentence(text)
        if len(sentence) < 8 or not self._due("rewrite", place, now):
            return []
        state.rewrites.clear()
        return [
            Correction(
                kind=KIND,
                key=f"{KIND}:rewrite:{place}:{int(now)}",
                title="Want another way to say it?",
                message="You've rewritten this a few times.",
                evidence=sentence[-80:],
                fix=f"Suggest a clearer way to write this: {sentence}",
                fix_kind="ask",
                urgency=LOW,
                confidence=0.6,
            )
        ]

    def _back_and_forth(self, now: float, window: str) -> list[Correction]:
        while self._switches and now - self._switches[0][0] > SWITCH_WINDOW_SECONDS:
            self._switches.popleft()
        pair = frozenset((self._switches[-1][1], self._switches[-1][2]))
        between = [s for s in self._switches if frozenset((s[1], s[2])) == pair]
        if len(pair) != 2 or len(between) < SWITCHES:
            return []
        first, second = sorted(pair)
        if not self._due("switching", "|".join((first, second)), now):
            return []
        self._switches.clear()
        a, b = _app_name(first), _app_name(second)
        if a == b:
            a, b = first.split("|", 1)[-1][:40] or a, second.split("|", 1)[-1][:40] or b
        return [
            Correction(
                kind=KIND,
                key=f"{KIND}:switching:{first}:{second}:{int(now)}",
                title=f"Want {a} and {b} side by side?",
                message="You keep switching between them.",
                evidence=f"{a} and {b}",
                fix=f"Put the {a} window and the {b} window side by side",
                fix_kind="ask",
                urgency=LOW,
                confidence=0.6,
            )
        ]

    def _repeated_error(self, text: str, window: str, ctx: TextContext, now: float) -> list[Correction]:
        for error in error_on_screen(text, ctx):
            line = error.evidence.strip()
            if not line or text.count(line) < ERROR_REPEATS:
                continue
            if not self._due("error", f"{window}|{line}", now):
                continue
            return [
                Correction(
                    kind=KIND,
                    key=f"{KIND}:error:{line.casefold()}",
                    title="That error keeps coming back",
                    message=f"It has come up {text.count(line)} times.",
                    evidence=line[:120],
                    detail=error.message,
                    fix=f"Help me fix this error: {line[:160]}",
                    fix_kind="ask",
                    urgency=LOW,
                    confidence=0.7,
                )
            ]
        return []

    def _due(self, sign: str, where: str, now: float) -> bool:
        last = self._offered.get((sign, where))
        if last is not None and now - last < QUIET_SECONDS:
            return False
        self._offered[(sign, where)] = now
        return True

    def forget_window(self, window: str) -> None:
        for key in [key for key in self._fields if key.startswith(f"{window}|")]:
            self._fields.pop(key, None)


__all__ = ["KIND", "StuckWatcher"]
