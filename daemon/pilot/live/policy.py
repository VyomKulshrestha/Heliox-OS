"""When the live layer speaks: the per-micro-turn decision for each pending correction.

A correction can wait, show (a card in the app and an OS notification), speak (voice plus the card) or be
dropped. The rules follow how a considerate person interrupts: urgent things at the first pause, everything
else at a natural pause, never while the person is talking, rarely while they are focused, and less often for
the kinds of correction they keep waving away.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pilot.live.detectors import LOW, NORMAL, URGENT, Correction

logger = logging.getLogger("pilot.live.policy")

WAIT, SHOW, SPEAK, DROP = "wait", "show", "speak", "drop"
# The card now, silently; the voice may follow at the longer pause, if the card is still unanswered and the person
# has not gone on (LiveCompanion._voice_later). A slip that is the last thing typed: the card and the voice have
# separate clocks, so a message sent inside the voice's longer wait still got its card.
SHOW_FIRST = "show_first"


@dataclass(frozen=True)
class Level:
    speak: frozenset[str]
    show: frozenset[str]
    pause_seconds: float  # how long typing must stop before a non-urgent correction
    spoken_gap_seconds: float  # minimum time between two spoken non-urgent corrections
    shown_gap_seconds: float  # minimum time between two silent cards


LEVELS = {
    "quiet": Level(frozenset({URGENT}), frozenset({URGENT, NORMAL}), 2.0, 600.0, 30.0),
    "balanced": Level(frozenset({URGENT, NORMAL}), frozenset({URGENT, NORMAL, LOW}), 1.2, 120.0, 8.0),
    "proactive": Level(frozenset({URGENT, NORMAL}), frozenset({URGENT, NORMAL, LOW}), 0.7, 45.0, 4.0),
}
DEFAULT_LEVEL = "balanced"

URGENT_PAUSE_SECONDS = 0.6  # long enough to not cut into a word, short enough to land before Enter
# A spoken sentence reaches Live only after the listener has decided the person's turn is over, so it is already
# the pause: no extra wait (it was 0.3 s, plus up to one 0.2 s tick).
SPEECH_SETTLE_SECONDS = 0.0
TRAILING_PAUSE_FACTOR = 2.5  # the mistake is the last thing typed: they may be mid-number or mid-word
TRAILING_MIN_PAUSE_SECONDS = 3.0
# At most this much longer than the person's own pause: a learned pause is already long enough to cover a half-typed
# number, and stretching a 5 s pause by the factor kept a finished line waiting 15 s.
TRAILING_MAX_EXTRA_SECONDS = 3.0
MIN_RELEVANCE_TO_SPEAK = 0.5
MIN_RELEVANCE_TO_SHOW = 0.2
MAX_VISIBLE_CARDS = 3

# Errors are worth a moment to read before being explained; the rest only need the person to pause.
MIN_DWELL_SECONDS = {"error_on_screen": 5.0, "missing_attachment": 2.0}


@dataclass
class Activity:
    """What the person and Heliox are doing at this micro-turn."""

    now: float
    last_text_change: float
    user_speaking: bool = False
    heliox_speaking: bool = False
    focus_active: bool = False
    visible_cards: int = 0
    last_spoken_at: float = float("-inf")
    last_shown_at: float = float("-inf")
    # The pause this person needs before a non-urgent correction (pilot.live.timing); None uses the level's.
    personal_pause: float | None = None
    # The same without week one's patience: what a slip that is the last thing typed stretches from, as it already
    # waits longer on its own (TRAILING_*). None: ``personal_pause``.
    trailing_pause: float | None = None
    # A conversation turn holds the floor (pilot.live.floor): Live may show a card but not start speaking, and a
    # non-urgent catch about typed text waits until the turn is over.
    floor_held: bool = False
    # In a call (pilot.live.calls), or one ended a moment ago: nothing is said, only urgent cards are shown.
    in_call: bool = False
    # The person has moved to another field or window since the slip: a card only, never the voice.
    elsewhere: bool = False


@dataclass
class Pending:
    correction: Correction
    source: str
    first_seen: float
    last_seen: float
    window: str = ""
    delivered: bool = False
    id: str = ""
    mode: str = ""
    delivered_at: float = 0.0
    # Set when a detection pass over the same window no longer finds the evidence.
    missing_since: float | None = None
    # Seconds from delivery to the next keystroke in the same window: short and unused means it interrupted.
    resumed_after: float | None = None
    # Noticed or kept while the person was in a call: said after it, if it still applies.
    held_in_call: bool = False
    # When it was noticed, as a pilot.live.sequence number: orders it against a conversation turn.
    noted_seq: int = 0
    # How long the person had stopped when it was delivered, and the wait Live required (pilot.live.timing).
    paused_for: float | None = None
    needed_pause: float | None = None
    # Shown silently first (SHOW_FIRST): the pause after which the voice may follow, and whether it did.
    voice_after: float | None = None
    voiced: bool = False
    # The message holding the slip was sent (its field emptied or replaced) before the catch: a card only, saying
    # what was sent (LiveCompanion._note_sent).
    sent: bool = False


def decide(
    item: Pending,
    act: Activity,
    level: Level,
    relevance: float,
    muted: frozenset[str],
    quiet: frozenset[str] = frozenset(),
) -> str:
    """``quiet``: kinds the person told "not now" on two different days within a week (LiveCompanion.quiet_kinds):
    shown as a card, never spoken, unless urgent."""
    correction = item.correction
    urgent = correction.urgency == URGENT
    if correction.kind in muted:
        return DROP
    if correction.urgency not in level.show:
        return DROP
    if not urgent and relevance < MIN_RELEVANCE_TO_SHOW:
        return DROP
    if act.in_call and not urgent:
        return WAIT  # never a voice into someone's call, and only urgent cards: the rest waits for the call to end
    if (act.user_speaking and not act.in_call) or item.missing_since is not None:
        # Never over the person's voice (in a call they talk to others, and a silent card does not interrupt
        # that), and never about text that is no longer there.
        return WAIT
    if item.source == "speech":
        # The person just spoke to Heliox, so answering is the expected next turn: no rate limits or focus hold.
        if act.now - item.last_seen < SPEECH_SETTLE_SECONDS:
            return WAIT
        if act.heliox_speaking:
            return SHOW
        speak = urgent or (
            correction.urgency in level.speak and relevance >= MIN_RELEVANCE_TO_SPEAK and correction.kind not in quiet
        )
        return SPEAK if speak else SHOW
    needed_pause, voice_pause = pauses(item, act, level)
    if act.now - act.last_text_change < needed_pause:
        return WAIT
    if act.now - item.first_seen < MIN_DWELL_SECONDS.get(correction.kind, 0.0):
        return WAIT
    if urgent:
        return SHOW if act.heliox_speaking or item.sent else SPEAK
    if act.floor_held:
        return WAIT
    if item.sent or act.elsewhere:
        # Sent before the catch, or the person has moved on: a card that says so, never the voice.
        return SHOW if act.visible_cards < MAX_VISIBLE_CARDS and not act.focus_active else WAIT
    if act.focus_active or act.visible_cards >= MAX_VISIBLE_CARDS:
        return WAIT
    speakable = (
        correction.urgency in level.speak
        and relevance >= MIN_RELEVANCE_TO_SPEAK
        and correction.kind not in quiet
        and not act.heliox_speaking
        and act.now - act.last_spoken_at >= level.spoken_gap_seconds
    )
    if speakable:
        # The last thing typed: the card at the ordinary pause, the voice only after the longer one.
        if act.now - act.last_text_change < voice_pause:
            item.voice_after = voice_pause
            return SHOW_FIRST
        return SPEAK
    if act.now - act.last_shown_at >= level.shown_gap_seconds:
        return SHOW
    return WAIT


def pauses(item: Pending, act: Activity, level: Level) -> tuple[float, float]:
    """(the pause before the card, the pause before the voice). The same for most catches; for a slip that is the
    last thing typed (they may be mid-number or mid-word) the voice waits longer: TRAILING_PAUSE_FACTOR times the
    person's pause, at most TRAILING_MAX_EXTRA_SECONDS more, never under TRAILING_MIN_PAUSE_SECONDS. The card keeps
    the ordinary pause without week one's patience (``trailing_pause``), silently, unless the slip ends in a digit
    (a number may still be being typed): then the card waits as long as the voice."""
    correction = item.correction
    urgent = correction.urgency == URGENT
    needed = URGENT_PAUSE_SECONDS if urgent else level.pause_seconds
    if not urgent and act.personal_pause is not None:
        needed = act.personal_pause
    if not correction.trailing or urgent:
        return needed, needed
    base = act.trailing_pause if act.trailing_pause is not None else needed
    stretched = max(min(base * TRAILING_PAUSE_FACTOR, base + TRAILING_MAX_EXTRA_SECONDS), TRAILING_MIN_PAUSE_SECONDS)
    if correction.evidence.rstrip()[-1:].isdigit():
        return stretched, stretched  # "12 x 15 = 17": they may be typing 170 still, so the card waits too
    return base, stretched


@dataclass
class Relevance:
    """How welcome each kind of correction has been, as a Beta(helpful, unwelcome) mean.

    Starts at 2/3: corrections are assumed useful until the person says otherwise. A correction the person acts
    on (thanks, uses the fix, or fixes the text after being told) raises it; dismissing lowers it; one that is
    ignored until it goes away lowers it a little.
    """

    path: Path | None = None
    counts: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.counts = {
                    k: [float(v[0]), float(v[1])] for k, v in raw.items() if isinstance(v, list) and len(v) == 2
                }
            except (OSError, ValueError, TypeError):
                logger.debug("Ignoring an unreadable live relevance file", exc_info=True)

    def score(self, kind: str) -> float:
        helpful, unwelcome = self.counts.get(kind, [2.0, 1.0])
        return helpful / (helpful + unwelcome)

    def update(self, kind: str, verdict: str) -> None:
        helpful, unwelcome = self.counts.get(kind, [2.0, 1.0])
        if verdict in ("helpful", "fixed", "used_fix"):
            helpful += 1.0
        elif verdict == "dismissed":
            unwelcome += 1.0
        elif verdict == "ignored":
            unwelcome += 0.25
        else:
            return
        self.counts[kind] = [helpful, unwelcome]
        self._save()

    def reset(self) -> None:
        self.counts = {}
        self._save()

    def snapshot(self) -> dict[str, float]:
        return {kind: round(self.score(kind), 2) for kind in self.counts}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.counts), encoding="utf-8")
        except OSError:
            logger.debug("Could not save live relevance", exc_info=True)


__all__ = [
    "DEFAULT_LEVEL",
    "DROP",
    "LEVELS",
    "LOW",
    "NORMAL",
    "SHOW",
    "SPEAK",
    "URGENT",
    "WAIT",
    "Activity",
    "Level",
    "Pending",
    "Relevance",
    "decide",
]
