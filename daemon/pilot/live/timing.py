"""When this person is done: a personal timing model for the live layer.

A fixed "wait 1.2 s after typing stops" is wrong for most people. A fast typist who pauses for half a second
between thoughts waits longer than they need to; someone who stops for two or three seconds in the middle of a
sentence to find the next word gets interrupted mid-thought. Interaction models learn this from data at a scale
no laptop can train; this module learns it from the one person using it, on the CPU, in a few numbers.

Two signals, both local and both stored as plain numbers (never text):

1. **Rhythm.** Every pause that ends with the person typing on in the same sentence is a pause they were not
   done at. The 90th percentile of those, plus a margin, is how long a pause has to last before it is probably a
   real break. With few pauses the level's default dominates; after about 20 the person's own rhythm does.
2. **Feedback.** A card the person dismisses or ignores, after they resumed typing within 1.5 s of it appearing,
   was an interruption: the waits grow by 15%. A card they act on was well timed: the waits shrink by 2%.

At a sentence boundary (the text ends in ``.``, ``!``, ``?`` or a new line) the person has finished a thought,
so the shorter of the level default and the learned mid-sentence pause applies. Urgent catches ignore all of
this: a risky command still lands at the first 0.6 s pause, before Enter.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pilot.live.timing")

MIN_PAUSE_SECONDS = 0.9  # shorter gaps are ordinary typing
MAX_PAUSE_SECONDS = 20.0  # longer means they stepped away, not that they were thinking
MAX_PAUSES = 400
PRIOR_WEIGHT = 20.0  # pauses it takes for the person's rhythm to outweigh the level default
QUANTILE = 0.9
MARGIN = 1.2
FLOOR_SECONDS = 0.7
CEILING_SECONDS = 6.0
CUT_OFF_SECONDS = 1.5  # typing resumed this soon after a card that then went unused: it interrupted them
GROW_ON_CUT_OFF = 1.15
SHRINK_ON_WELL_TIMED = 0.98
MIN_SCALE, MAX_SCALE = 0.8, 2.0
SAVE_EVERY = 5

GOOD_VERDICTS = frozenset({"helpful", "fixed", "used_fix"})
BAD_VERDICTS = frozenset({"dismissed", "ignored", "expired", "mute"})


def at_boundary(text: str) -> bool:
    """True when the text ends a sentence or a line, so a pause here is a natural break."""
    stripped = text.rstrip(" \t")
    return not stripped or stripped[-1] in ".!?\n"


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class PersonalTiming:
    path: Path | None = None
    pauses: list[float] = field(default_factory=list)  # mid-sentence pauses that ended with more typing
    scale: float = 1.0
    cut_offs: int = 0
    well_timed: int = 0
    _unsaved: int = 0

    def __post_init__(self) -> None:
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                pauses = [float(p) for p in raw.get("pauses", [])]
                self.pauses = [p for p in pauses if MIN_PAUSE_SECONDS <= p <= MAX_PAUSE_SECONDS][-MAX_PAUSES:]
                self.scale = min(MAX_SCALE, max(MIN_SCALE, float(raw.get("scale", 1.0))))
                self.cut_offs = int(raw.get("cut_offs", 0))
                self.well_timed = int(raw.get("well_timed", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                logger.debug("Ignoring an unreadable live timing file", exc_info=True)

    # ── learning ──

    def observe_pause(self, seconds: float, mid_sentence: bool) -> None:
        """The person stopped for ``seconds`` and then typed on in the same window."""
        if not mid_sentence or not MIN_PAUSE_SECONDS <= seconds <= MAX_PAUSE_SECONDS:
            return
        self.pauses.append(round(seconds, 2))
        del self.pauses[:-MAX_PAUSES]
        self._unsaved += 1
        if self._unsaved >= SAVE_EVERY:
            self._save()

    def record_outcome(self, verdict: str, resumed_after: float | None) -> str:
        """How a delivered card went. Returns "well_timed", "cut_off" or "" (says nothing about timing)."""
        if verdict in GOOD_VERDICTS:
            self.well_timed += 1
            self.scale = max(MIN_SCALE, self.scale * SHRINK_ON_WELL_TIMED)
            self._save()
            return "well_timed"
        if verdict in BAD_VERDICTS and resumed_after is not None and resumed_after < CUT_OFF_SECONDS:
            self.cut_offs += 1
            self.scale = min(MAX_SCALE, self.scale * GROW_ON_CUT_OFF)
            self._save()
            return "cut_off"
        return ""

    # ── the waits ──

    def mid_sentence_pause(self, default: float) -> float:
        n = len(self.pauses)
        learned = quantile(self.pauses, QUANTILE) * MARGIN if n else default
        weight = n / (n + PRIOR_WEIGHT)
        return self._clamp((weight * learned + (1 - weight) * default) * self.scale)

    def boundary_pause(self, default: float) -> float:
        return self._clamp(min(default * self.scale, self.mid_sentence_pause(default)))

    def pause_for(self, text: str, default: float) -> float:
        return self.boundary_pause(default) if at_boundary(text) else self.mid_sentence_pause(default)

    @staticmethod
    def _clamp(seconds: float) -> float:
        return round(min(CEILING_SECONDS, max(FLOOR_SECONDS, seconds)), 2)

    # ── state ──

    def snapshot(self, default: float) -> dict[str, float | int]:
        return {
            "mid_sentence_seconds": self.mid_sentence_pause(default),
            "boundary_seconds": self.boundary_pause(default),
            "default_seconds": default,
            "learned_from": len(self.pauses),
            "cut_offs": self.cut_offs,
            "well_timed": self.well_timed,
        }

    def forget(self) -> None:
        """Back to the level defaults, in memory; the caller deletes the file."""
        self.pauses = []
        self.scale = 1.0
        self.cut_offs = 0
        self.well_timed = 0
        self._unsaved = 0

    def reset(self) -> None:
        self.forget()
        self._save()

    def flush(self) -> None:
        if self._unsaved:
            self._save()

    def _save(self) -> None:
        self._unsaved = 0
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "pauses": self.pauses,
                "scale": round(self.scale, 4),
                "cut_offs": self.cut_offs,
                "well_timed": self.well_timed,
            }
            self.path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.debug("Could not save live timing", exc_info=True)


__all__ = ["CUT_OFF_SECONDS", "PersonalTiming", "at_boundary", "quantile"]
