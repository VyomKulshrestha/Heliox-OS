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

Taught spellings (``teach``): the person says the wake phrase a few times on purpose (Settings, Voice); a
spelling their recogniser wrote at least twice is kept as theirs, at most MAX_TAUGHT of them, shown in Settings and
removable. A spelling made only of ordinary words ("hey alex", "hello", "here it is") is refused: it would wake
Heliox in ordinary talk. This is the path a new person reaches on day one; the implicit learning below needs
five near-misses each followed by an exact hit, which a voice the recogniser never writes as "Heliox" never gives.

Storage: a small JSON file under ~/.cache/heliox/voice_calibration/ (mirrors
pilot.cognitive.biometric_loop's storage location pattern) — only wake-word
text variants, counts, and timestamps are stored. No audio, no general
conversation transcripts, nothing transmitted anywhere.
"""

from __future__ import annotations

import json
import logging
import re
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
    taught: bool = False  # said on purpose in the teach step (not learned from near-misses)


MAX_TAUGHT = 5
TEACH_MIN_REPEATS = 2  # a spelling counts when the recogniser wrote it this many times in one teach session
LEADS = ("hey", "hay", "hi", "okay", "ok", "hello", "here", "dear", "a", "eh", "oh")
# Words that cannot be the name: a taught spelling made only of these (after a greeting) would wake Heliox in
# ordinary talk ("Hey Alex" to a colleague, "Hello there", "Here it is"). Common English words and first names
# recognisers produce for a two-syllable name; a spelling with any other word is accepted.
_ORDINARY_TEXT = """
    a about after again all also am an and any are as at away back be because been before being but by can come
    could day did do does done down each even every for from get go going good got had has have he hello help
    her here hey hi him his how i if in into is it its just know let like look lot made make many may me more
    most much must my need new no not now of off oh ok okay old on one only or other our out over own people
    please put right said say see she should so some still such take than thank thanks that the their them then
    there these they thing think this those through time to too two up us very want was way we well were what
    when where which who why will with work would yeah yes yet you your here's there's it's that's what's let's
    yo hiya ho hm hmm uh um ah eh lex alex alexa alice alexis ali ella ellie eliot elliot eli elias felix helix
    helen helena hillary holly leo lee leon lisa lucy luke max mike nick olivia oscar rex rick rio sam sean
    sophie theo tom tony zoe hell hollow halo hilly hills hello's hero heroes heal healing heals helium oil
"""
ORDINARY = frozenset(_ORDINARY_TEXT.split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.casefold())


def teach_candidate(text: str) -> str:
    """The spelling to learn from one try: the recogniser's words for the wake phrase alone, lower case, at most
    three words ("Hey, Helios!" -> "hey helios")."""
    return " ".join(_words(text)[:3])


_COMMON: frozenset[str] | None = None


def common_words() -> frozenset[str]:
    """Ordinary words: ORDINARY plus every whole word in the installed Whisper model's vocabulary (about 24,000:
    the words Whisper writes as one piece are the frequent ones; "Heliox", "Helios" and "Helix" are not among
    them). Read from the model on this computer; nothing is downloaded. Without Whisper, ORDINARY alone."""
    global _COMMON
    if _COMMON is None:
        words = set(ORDINARY)
        try:
            words |= _whisper_whole_words()
        except Exception:
            logger.debug("Whisper vocabulary not readable; the short list guards the taught spellings", exc_info=True)
        _COMMON = frozenset(words)
    return _COMMON


def _whisper_whole_words() -> set[str]:
    from pilot.system.model_presence import FASTER_WHISPER_REPOS, faster_whisper_source, hf_cached

    for size in ("base", "small", "tiny", "medium"):
        source = faster_whisper_source(size)
        if source is None:
            continue
        folder = Path(source)
        path = folder / "tokenizer.json" if folder.is_dir() else None
        if path is None or not path.is_file():
            cached = hf_cached(FASTER_WHISPER_REPOS[size], "tokenizer.json")
            path = Path(cached) if cached else None
        if path is not None and path.is_file():
            vocab = json.loads(path.read_text(encoding="utf-8"))["model"]["vocab"]
            return {t[1:].lower() for t in vocab if t.startswith("\u0120") and t[1:].isalpha()}
    return set()


def refusal(candidate: str) -> str:
    """Why this spelling cannot be kept, or "" when it can. Kept: a greeting at most, then a word that is not
    ordinary and is within four letters of "heliox" ("hey helios", "hey hilux"). Refused: ordinary words ("hey
    alex", "here it is", "find the email"), too short ("hey lux"), or nothing like the name ("type powershell")."""
    words = _words(candidate)
    if not words:
        return "Nothing was heard."
    if any(w == "heliox" for w in words):
        return "That is already the name."
    core = words[1:] if words[0] in LEADS and len(words) > 1 else words
    common = common_words()
    if all(w in common for w in core):
        return f"“{candidate}” is made of ordinary words; it would wake Heliox in everyday talk."
    if sum(len(w) for w in core) < 4:
        return f"“{candidate}” is too short to tell apart from other speech."
    if levenshtein("".join(core), "heliox") > 4:
        return f"“{candidate}” does not sound like the name."
    return ""


def teach(tries: list[str]) -> tuple[list[str], dict[str, str]]:
    """From the recogniser's words for each try, the spellings to keep (written at least TEACH_MIN_REPEATS times
    and not refused) and the ones refused with the reason. Tries that already read as "Heliox" need nothing."""
    counts: dict[str, int] = {}
    for text in tries:
        candidate = teach_candidate(text)
        if candidate:
            counts[candidate] = counts.get(candidate, 0) + 1
    keep: list[str] = []
    refused: dict[str, str] = {}
    for candidate, count in sorted(counts.items(), key=lambda item: -item[1]):
        why = refusal(candidate)
        if why == "That is already the name.":
            continue
        if why:
            refused[candidate] = why
        elif count < TEACH_MIN_REPEATS:
            refused[candidate] = f"“{candidate}” was heard only once."
        else:
            keep.append(candidate)
    return keep, refused


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
        words = _words(transcript_lower)  # "Hey Heliocs, open" is "hey heliocs": the comma is not a letter apart
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

    def add_taught(self, spellings: list[str]) -> list[str]:
        """Keep spellings from the teach step as the person's own, at most MAX_TAUGHT (the oldest taught ones go
        first). Returns the taught spellings now kept."""
        now = time.time()
        for text in spellings:
            if refusal(text):
                continue
            variant = self._variants.get(text) or WakeWordVariant(text=text, first_seen=now)
            variant.taught = True
            variant.confirmed_count = max(variant.confirmed_count, self.PROMOTION_THRESHOLD)
            variant.last_confirmed = now
            self._variants[text] = variant
        taught = sorted((v for v in self._variants.values() if v.taught), key=lambda v: v.last_confirmed)
        for old in taught[: max(0, len(taught) - MAX_TAUGHT)]:
            del self._variants[old.text]
        self._store.save(self._variants)
        return [v.text for v in self._variants.values() if v.taught]

    def remove(self, text: str) -> bool:
        """Remove one spelling (taught or learned); it no longer wakes Heliox."""
        if self._variants.pop(text, None) is None:
            return False
        self._store.save(self._variants)
        return True

    def promoted(self) -> list[str]:
        """Spellings that count as the name: taught ones and learned ones past PROMOTION_THRESHOLD, longest first."""
        texts = [t for t, v in self._variants.items() if v.taught or v.confirmed_count >= self.PROMOTION_THRESHOLD]
        return sorted(texts, key=len, reverse=True)

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

    def reset(self) -> None:
        """Clears all learned variants and pending candidates, and deletes
        the on-device store — the "reset learned wake words" Settings
        action. Safe to call on a calibrator backing a currently-running
        listener; takes effect immediately."""
        self._variants = {}
        self._pending = {}
        self._store.reset()
