"""On-device wake-word calibration — continual-learning loop for voice recognition.

Personalizes wake-word matching to a user's accent/microphone by learning a
small set of "confirmed variant" transcripts that Whisper reliably produces
for this specific user's speech, without retraining Whisper or storing any
raw audio. This is deliberately narrow: `ContinuousVoiceListener`'s fixed
exact-substring match (`voice.py`) stays the fast path and is never touched;
this module only adds a fallback tried after that fast path misses.

Signal (implicit, no new "was this right?" UI): if a transcript nearly
matches a wake word (small edit distance) but doesn't exact-match, and a
real exact-match wake-word hit follows shortly after in the same listening
session, that's read as "the near-miss was probably a failed wake attempt
for this accent/mic, and the user just repeated themselves." A near-miss
seen this way several times (PROMOTION_THRESHOLD) is promoted to a trusted
variant. A single occurrence proves nothing and is not promoted.

Storage: a small JSON file under ~/.cache/heliox/voice_calibration/ (mirrors
pilot.cognitive.biometric_loop's storage location pattern) — only wake-word
text variants, counts, and timestamps are stored. No audio, no general
conversation transcripts, nothing transmitted anywhere.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("pilot.system.voice_calibration")


def levenshtein(a: str, b: str) -> int:
    """Plain iterative Levenshtein edit distance (two-row DP).

    Hand-rolled rather than a new dependency — this is the only place in the
    codebase that needs edit distance, and pulling in a package for one
    ~15-line function would be unwarranted.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,  # deletion
                curr_row[j - 1] + 1,  # insertion
                prev_row[j - 1] + cost,  # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


@dataclass
class WakeWordVariant:
    text: str
    confirmed_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_confirmed: float = field(default_factory=time.time)


class VoiceCalibrationStore:
    """Persists learned wake-word variants to a local JSON file."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or (Path.home() / ".cache" / "heliox" / "voice_calibration")
        self._path = self._data_dir / "wake_variants.json"

    def load(self) -> dict[str, WakeWordVariant]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {text: WakeWordVariant(**fields) for text, fields in raw.items()}
        except Exception:
            logger.warning("Failed to load voice calibration — starting fresh", exc_info=True)
            return {}

    def save(self, variants: dict[str, WakeWordVariant]) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {text: asdict(v) for text, v in variants.items()}
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save voice calibration (non-critical)", exc_info=True)

    def reset(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to reset voice calibration", exc_info=True)


class WakeWordCalibrator:
    """Learns accent/mic-specific near-miss wake-word variants.

    Wired into `ContinuousVoiceListener._listen_loop()` as a fallback tried
    only after the fixed exact-substring wake-word check misses — the common
    case (exact match) is entirely untouched by this class.
    """

    NEAR_MISS_MAX_DISTANCE = 2
    PROMOTION_THRESHOLD = 5
    # How many _listen_loop iterations a near-miss stays "pending", waiting
    # to see if a real wake-word hit follows shortly after.
    CONFIRM_WINDOW_ITERATIONS = 2

    def __init__(self, base_wake_words: list[str], store: VoiceCalibrationStore | None = None) -> None:
        self._base_wake_words = base_wake_words
        self._store = store or VoiceCalibrationStore()
        self._variants: dict[str, WakeWordVariant] = self._store.load()
        # candidate text -> iterations remaining before it expires unconfirmed
        self._pending: dict[str, int] = {}

    def check_near_miss(self, transcript_lower: str) -> str | None:
        """Wake words are always spoken first, so compare each base wake word
        only against a LEADING WORD-COUNT-MATCHED window of the transcript —
        e.g. a 2-word wake word is compared against the transcript's first 2
        words, not a fixed character count, so trailing command text never
        bleeds into the comparison. Returns the best near-miss candidate
        within NEAR_MISS_MAX_DISTANCE, or None.
        """
        words = transcript_lower.split()
        best_candidate: str | None = None
        best_distance = self.NEAR_MISS_MAX_DISTANCE + 1

        for wake in self._base_wake_words:
            wake_word_count = len(wake.split())
            candidate_words = words[:wake_word_count]
            if not candidate_words:
                continue
            candidate = " ".join(candidate_words)

            distance = levenshtein(candidate, wake)
            if distance == 0:
                # Exact match belongs to the fast path in voice.py, not here.
                continue
            if distance <= self.NEAR_MISS_MAX_DISTANCE and distance < best_distance:
                best_distance = distance
                best_candidate = candidate

        return best_candidate

    def record_pending(self, candidate: str) -> None:
        """Mark `candidate` as awaiting confirmation for CONFIRM_WINDOW_ITERATIONS."""
        self._pending[candidate] = self.CONFIRM_WINDOW_ITERATIONS

    def tick(self) -> None:
        """Call once per `_listen_loop` iteration to expire stale pending candidates."""
        expired = []
        for candidate, remaining in self._pending.items():
            if remaining <= 1:
                expired.append(candidate)
            else:
                self._pending[candidate] = remaining - 1
        for candidate in expired:
            del self._pending[candidate]

    def confirm_pending_if_followed_by_hit(self) -> None:
        """Called right after a REAL exact-match wake-word hit. Any near-miss
        still pending is the implicit "this was a failed wake attempt for
        this accent, and the user just repeated themselves" signal — bump
        its confirmed count and persist.
        """
        if not self._pending:
            return

        now = time.time()
        for candidate in list(self._pending.keys()):
            variant = self._variants.get(candidate)
            if variant is None:
                variant = WakeWordVariant(text=candidate, confirmed_count=0, first_seen=now, last_confirmed=now)
                self._variants[candidate] = variant
            variant.confirmed_count += 1
            variant.last_confirmed = now

        self._pending.clear()
        self._store.save(self._variants)

    def match_promoted_variant(self, transcript_lower: str) -> str | None:
        """Returns a learned variant substring if the transcript contains one
        that has crossed PROMOTION_THRESHOLD confirmed occurrences, else None.
        """
        for text, variant in self._variants.items():
            if variant.confirmed_count >= self.PROMOTION_THRESHOLD and text in transcript_lower:
                return text
        return None

    def get_variants(self) -> list[WakeWordVariant]:
        """All learned variants (confirmed or still pending promotion) — for
        the Settings transparency view."""
        return list(self._variants.values())
