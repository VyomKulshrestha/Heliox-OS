"""The live layer: a micro-turn clock that perceives, notices and decides when to speak.

Every ``TICK_SECONDS`` the companion looks at the moment: what the person is typing (the focused field, read
locally), whether they are talking, whether Heliox is talking, whether a focus session is on. The detectors find
specific mistakes; the policy decides, per correction, whether to wait, show a card, speak, or let it go. When
the person fixes something they were told about, the card goes away by itself and that counts as a thank-you.

It never acts on its own. "Fix it" hands a request to the planner through the normal approval path, the same
as if the person had typed it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pilot.live import focused_text, spoken
from pilot.live.commitments import KIND as COMMITMENT
from pilot.live.commitments import find_commitments
from pilot.live.detectors import (
    KIND_LABELS,
    KINDS,
    MODEL_KINDS,
    NORMAL,
    SPEECH_KINDS,
    URGENT,
    Correction,
    TextContext,
    detect,
)
from pilot.live.policy import (
    DEFAULT_LEVEL,
    DROP,
    LEVELS,
    SHOW,
    SPEAK,
    Activity,
    Pending,
    Relevance,
    decide,
)
from pilot.live.stuck import KIND as STUCK
from pilot.live.stuck import StuckWatcher
from pilot.live.timing import PersonalTiming, at_boundary
from pilot.system.app_names import app_label

logger = logging.getLogger("pilot.live.engine")

TICK_SECONDS = 0.2
PERCEIVE_EVERY_SECONDS = 0.6
REDETECT_EVERY_SECONDS = 2.0
STALE_AFTER_SECONDS = 1.6  # evidence gone this long from the same window means the person fixed it
SPEECH_TTL_SECONDS = 10.0
PENDING_TTL_SECONDS = 90.0
VISIBLE_TTL_SECONDS = 600.0  # a card left alone this long is withdrawn and counts as mildly unwelcome
RECENT_LIMIT = 20
WINDOW_CHECK_SECONDS = 1.0  # how often cards look for the window they were typed in
REPLY_WINDOW_SECONDS = 90.0  # how long "why?", "thanks" or "fix it" refer to the last correction
OCR_KINDS = ("error_on_screen",)  # a whole-screen OCR is too noisy for anything finer
# Offers that stay up when their text leaves the field: a promise matters most once the message is sent.
STICKY_KINDS = frozenset({STUCK, COMMITMENT})
GROUNDED_PAUSE_SECONDS = 1.0  # typing stopped this long: worth asking the local model about the last sentence
GROUNDED_EVERY_SECONDS = 5.0  # and at most this often, so a laptop CPU is not kept busy
# Kinds about the person's own words. Text that was already in a field when it got focus (a quoted email, a
# document someone else wrote, terminal history) does not raise them unless the person's edits add the slip again.
OWN_WORDS_KINDS = frozenset(
    {
        "weekday_date",
        "arithmetic",
        "profile_fact",
        "missing_attachment",
        "repeated_word",
        "community_catch",
        "personal_fact",
        COMMITMENT,
    }
)
BASELINE_LIMIT = 64  # fields whose starting text is remembered, most recent first
REPLACED_MIN_CHARS = 8  # see replaced(): the least text that has to go at once for a field to start over
# Supervision notices: only keyboard-hook matches (what the person just typed) become cards, each rule at most once
# an hour. An OCR match can be anywhere on screen and has no evidence the person could check.
NOTICE_SOURCES = frozenset({"keystroke"})
NOTICE_QUIET_SECONDS = 3600.0
# "Not now" holds that kind of catch back this long (urgent risks still show). "Don't mention these" mutes it.
SNOOZE_SECONDS = 1800.0
# After a call ends, this long before the held catches are said: people say goodbye and breathe first.
AFTER_CALL_SECONDS = 5.0
# A speech outcome that means the remark was not heard; it must not count against the spoken rate limit.
UNHEARD = frozenset({"suppressed", "held", "dropped", "error", "cancelled", "superseded"})


@dataclass(frozen=True)
class Observation:
    """One look at what the person is working on. ``text`` never leaves this process."""

    text: str
    app: str
    title: str
    source: str  # "focused_field" or "screen_ocr"
    # False when focus is on page text, a label or a button: the person is reading, not writing, and the text is
    # not checked. The reader sends no text at all in that case.
    editable: bool = True
    field: str = ""  # which field in the window (UI Automation runtime id), "" when unknown

    @property
    def window(self) -> str:
        return f"{self.app}|{self.title}"

    @property
    def field_key(self) -> str:
        """Which text field this is, stable while the person types in it.

        The UI Automation runtime id names the field; the window title does not, because many editors put the
        text itself or an unsaved marker in the title (Windows 11 Notepad titles an untitled tab with its first
        line, VS Code adds a dot on the first keystroke). Without a runtime id the title is all there is.
        """
        return f"{self.app}|#{self.field}" if self.field else f"{self.window}|"


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def replaced(old: str, new: str) -> bool:
    """The field's text was swapped for other text rather than edited: a sent message, another tab, a new file.

    Typing, deleting a word or retyping one keeps most of the old text around the change; here more than half of
    it, and at least ``REPLACED_MIN_CHARS`` characters, went between two looks.
    """
    if old == new:
        return False
    limit = min(len(old), len(new))
    prefix = 0
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and old[-1 - suffix] == new[-1 - suffix]:
        suffix += 1
    removed = len(old) - prefix - suffix
    return removed >= REPLACED_MIN_CHARS and removed * 2 > len(old)


def say(title: str, message: str, source: str = "") -> str:
    """A card read out as one reply: its title states the fact and its body adds what they wrote or what to do."""
    if source == "speech":
        message = message.replace("You wrote ", "You said ", 1)
    if not title:
        return message
    return f"{title}{'' if title[-1] in '.?!' else '.'} {message}".strip()


_COMMON_FIRST = frozenset(
    [
        "that",
        "this",
        "these",
        "those",
        "your",
        "the",
        "it",
        "its",
        "there",
        "a",
        "an",
        "port",
        "you",
        "what",
        "one",
        "two",
        "some",
        "no",
        "nothing",
    ]
)
MAX_SPOKEN_QUOTE = 120


def _lower_first(text: str) -> str:
    first = text.split(" ", 1)[0]
    return text[0].lower() + text[1:] if first.casefold() in _COMMON_FIRST else text


def because(entry: dict[str, Any]) -> str:
    """Why Live spoke up, as one or two spoken sentences built from what it noticed: where it was, the person's own
    words it saw or heard, and what was wrong with them. Nothing is invented; every part comes from the card."""
    title = " ".join(str(entry.get("title") or "").split()).rstrip(".!")
    message = " ".join(str(entry.get("message") or "").split())
    detail = " ".join(str(entry.get("detail") or "").split())
    evidence = " ".join(str(entry.get("evidence") or "").split())
    if len(evidence) > MAX_SPOKEN_QUOTE:
        evidence = evidence[: MAX_SPOKEN_QUOTE - 3].rstrip() + "..."
    source, kind, app = entry.get("source", ""), entry.get("kind", ""), entry.get("app", "")
    where = f"In {app}, " if app and source not in ("speech", "notice") else ""

    def join(*parts: str) -> str:
        out = [p if p[-1] in ".?!" else p + "." for p in (part.strip() for part in parts) if p]
        return " ".join(out)

    if source == "notice" or not evidence:
        label = KIND_LABELS.get(kind, "something risky").lower()
        lead = (
            f"I spoke up because what you just typed matched my check for {label}"
            if source == "notice"
            else "I spoke up because of this"
        )
        return join(lead, title, message, detail)
    if title.endswith("?"):  # an offer: a promise without a reminder, or signs of being stuck
        reason = message.rstrip(".")
        if kind == COMMITMENT:
            return join(f"I offered because {_lower_first(reason)}, and I didn't find a reminder for it")
        return join(f'{where}I offered because {_lower_first(reason)}: "{evidence}"', detail)
    if source == "speech":
        verb = "you said"
    elif kind == "error_on_screen":
        verb = "the error was"
    else:
        verb = "you wrote"
    lead = f"{where}{verb}" if where else verb[0].upper() + verb[1:]
    extra = "" if message.startswith(("You wrote", "You said")) else message
    return join(f'{lead} "{evidence}", and {_lower_first(title)}', extra, detail)


@dataclass(frozen=True)
class LiveReply:
    reply: str
    delegate: str = ""  # a request to hand to the planner, when the person says "fix it"


# Supervision's rule names (pilot.security.risk_patterns) as something worth saying out loud.
_RISK_NOTICES = {
    "destructive_shell_command": (
        "That command could wipe a lot",
        "There's no undo.",
    ),
    "credential_exposure": (
        "That looks like a password or key",
        "What you just typed looks like a password or key. Make sure it isn't going somewhere other people can read.",
    ),
    "destructive_sql": ("That SQL deletes data for good", "Make sure you have a backup first."),
    "unauthorized_admin_command": (
        "That changes admin rights",
        "Check it's the account you mean.",
    ),
}

_THANKS = re.compile(
    r"^(?:thanks|thank you|thanks heliox|got it|good catch|nice catch|nice one|okay thanks|ok thanks"
    r"|oh thanks|good spot|well spotted)$"
)
_DISMISS = re.compile(
    r"^(?:not now|not right now|later|maybe later|ignore (?:it|that)|i know|i know that|that'?s fine|it'?s fine"
    r"|never mind|leave it|not now thanks|no thanks)$"
)
_MUTE = re.compile(
    r"^(?:please )?(?:don'?t|do not|never) (?:mention|point out|tell me about|bring up|flag) "
    r"(?:these|those|this|that|it|them|that kind of thing|things like that)(?: again| any more| anymore)?"
    r"(?: please)?$"
    r"|^(?:please )?stop (?:mentioning|pointing out|flagging) (?:these|those|this|that|them)(?: please)?$"
)
_WHY = re.compile(
    r"^(?:why|why is that|why not|how come|what do you mean|what did you mean|explain|explain that|what'?s wrong"
    r"|what'?s wrong with (?:it|that)|why did you say (?:that|it)|why did you mention (?:that|it)"
    r"|why are you telling me (?:that|this)|what was that|what was that about|what did you notice"
    r"|what are you talking about|what do you mean by that)$"
)
_FIX = re.compile(r"^(?:fix it|fix that|do it|go ahead|go ahead and fix it|please fix it)$")


class LiveCompanion:
    def __init__(
        self,
        *,
        settings: Callable[[], Any],
        observe: Callable[[], Awaitable[Observation | None]],
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
        speak: Callable[[str, bool], Awaitable[None]],
        user_speaking: Callable[[], bool] = lambda: False,
        heliox_speaking: Callable[[], bool] = lambda: False,
        focus_active: Callable[[], bool] = lambda: False,
        people: Callable[[], dict[str, str]] = dict,
        mute_kind: Callable[[str], Awaitable[None]] | None = None,
        on_disable: Callable[[], Awaitable[None]] | None = None,
        relevance_path: Path | None = None,
        timing_path: Path | None = None,
        grounded: Any = None,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
        wallclock: Callable[[], datetime] = datetime.now,
        reminders: Callable[[], list[dict[str, Any]]] = list,
        window_exists: Callable[[int], bool] = focused_text.window_exists,
        floor: Any = None,
        in_call: Callable[[], str] = lambda: "",
    ) -> None:
        self._settings = settings
        self._observe = observe
        self._emit = emit
        self._speak = speak
        self._user_speaking = user_speaking
        self._heliox_speaking = heliox_speaking
        self._focus_active = focus_active
        self._people = people
        self._mute_kind = mute_kind
        self._on_disable = on_disable
        self._was_on = False
        self._clock = clock
        self._today = today
        self._wallclock = wallclock
        self._reminders = reminders
        self._window_exists = window_exists
        self.relevance = Relevance(relevance_path)
        self.timing = PersonalTiming(timing_path)
        self.stuck = StuckWatcher(clock)
        self.grounded = grounded  # pilot.live.grounded.GroundedChecker, or None
        self._grounded_task: asyncio.Task[None] | None = None
        self._grounded_hash = ""
        self._last_grounded_at = float("-inf")

        self._task: asyncio.Task[None] | None = None
        self._paused = False
        self._pending: dict[str, Pending] = {}
        self._visible: dict[str, Pending] = {}  # delivered and still on screen, by correction id
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT_LIMIT)
        self._last_obs: Observation | None = None
        self._last_hash = ""
        self._last_text = ""
        self._last_text_change = float("-inf")
        self._last_perceive = float("-inf")
        self._last_detect = float("-inf")
        self._last_window_check = float("-inf")
        self._last_spoken_at = float("-inf")
        self._last_shown_at = float("-inf")
        self._state: dict[str, Any] = {}
        # Each field's text when it was first seen, or after it was emptied (see OWN_WORDS_KINDS).
        self._baselines: OrderedDict[str, str] = OrderedDict()
        self._field_text: OrderedDict[str, str] = OrderedDict()  # each field's text at the last look
        self._writing_seen = False  # a writable field has been seen since Live was turned on
        self._notices: dict[str, float] = {}  # supervision rule -> when it last raised a card
        # pilot.live.floor.Floor shared with the conversation: Live does not start speaking while a turn holds it.
        self.floor = floor
        self._in_call = in_call  # the call app, or "" (pilot.live.calls.CallWatch)
        self._call = ""
        self._call_ended_at = float("-inf")
        self._needed_pause: float | None = None  # the wait the last decision required
        self._snoozed: dict[str, float] = {}  # kind -> until when "not now" holds it back

    # ── lifecycle ──

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run(), name="heliox-live")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._withdraw_all("off")
        self._pending.clear()
        self.timing.flush()
        if self._grounded_task is not None:
            self._grounded_task.cancel()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Live tick failed", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)

    # ── one micro-turn ──

    def _config(self) -> Any:
        return self._settings()

    def _personal_timing(self) -> bool:
        return bool(getattr(self._config(), "personal_timing", True))

    def _kinds(self) -> tuple[str, ...]:
        muted = set(getattr(self._config(), "muted_kinds", []) or [])
        return tuple(kind for kind in KINDS if kind not in muted)

    def _context(self, obs: Observation | None, source: str) -> TextContext:
        return TextContext(
            today=self._today(),
            hour=self._wallclock().hour,
            app=obs.app if obs else "",
            title=obs.title if obs else "",
            source=source,
            people=self._people(),
        )

    async def tick(self) -> None:
        cfg = self._config()
        now = self._clock()
        if not getattr(cfg, "enabled", False) or self._paused:
            if self._was_on:
                # Turned off or paused: take the cards down and let the text reader go.
                self._was_on = False
                await self._withdraw_all("off")
                self._pending.clear()
                self._last_obs = None
                self._baselines.clear()
                self._field_text.clear()
                self._writing_seen = False
                if self._on_disable:
                    with contextlib.suppress(Exception):
                        await self._on_disable()
            await self._publish_state(now)
            return
        self._was_on = True

        if now - self._last_perceive >= PERCEIVE_EVERY_SECONDS:
            self._last_perceive = now
            await self._perceive(now)
            self._maybe_ground(now)

        self._watch_call(now)
        self._expire(now)
        await self._deliver_due(now)
        await self._publish_state(now)

    async def _perceive(self, now: float) -> None:
        try:
            obs = await asyncio.wait_for(self._observe(), timeout=2.0)
        except (TimeoutError, Exception):
            obs = None
        if obs is None:
            self._last_obs = None
            return
        if not obs.editable:
            # Focus is on page text, a label or a button: the person is reading. Only the window counts (for the
            # back-and-forth sign); nothing is checked, and cards about a field elsewhere stay as they are.
            if STUCK in self._kinds():
                ctx = self._context(obs, obs.source)
                for offer in self.stuck.observe("", obs.window, ctx, now, writing=False):
                    self._note(offer, obs.source, obs.field_key, now)
            self._last_obs = obs
            return
        digest = hashlib.sha1(obs.text.encode("utf-8", errors="ignore")).hexdigest()
        last = self._last_obs
        same_field = last is not None and last.editable and obs.field_key == last.field_key
        changed = digest != self._last_hash or not same_field
        if changed:
            if same_field:
                self._typed_on(obs.field_key, now)
            self._last_hash = digest
            self._last_text_change = now
            self._last_text = obs.text
        self._last_obs = obs
        self._remember_baseline(obs)
        if obs.source == "focused_field" and STUCK in self._kinds():
            ctx = self._context(obs, obs.source)
            for offer in self.stuck.observe(obs.text, obs.window, ctx, now, field=obs.field):
                self._note(offer, obs.source, obs.field_key, now)
        if changed or now - self._last_detect >= REDETECT_EVERY_SECONDS:
            self._last_detect = now
            kinds = self._kinds()
            if obs.source == "screen_ocr":
                kinds = tuple(kind for kind in kinds if kind in OCR_KINDS)
            found: set[str] = set()
            for correction in detect(obs.text, self._context(obs, obs.source), kinds=kinds):
                found.add(correction.key)
                if not self._pre_existing(correction, obs):
                    self._note(correction, obs.source, obs.field_key, now)
            if COMMITMENT in kinds and obs.source == "focused_field":
                try:
                    promises = find_commitments(
                        obs.text, self._context(obs, obs.source), self._wallclock(), self._reminders()
                    )
                except Exception:
                    logger.debug("Commitment check failed", exc_info=True)
                    promises = []
                for promise in promises:
                    if not self._pre_existing(promise, obs):
                        self._note(promise, obs.source, obs.field_key, now)
            for item in [*self._pending.values(), *self._visible.values()]:
                if item.source == "speech" or item.window != obs.field_key or item.correction.kind in STICKY_KINDS:
                    continue  # an offer is not about text that can be fixed away
                if item.correction.kind in MODEL_KINDS and item.correction.evidence in obs.text:
                    found.add(item.correction.key)
                if item.correction.key in found:
                    item.missing_since = None
                elif item.missing_since is None:
                    item.missing_since = now

    def _remember_baseline(self, obs: Observation) -> None:
        """Keep the text a field held when focus moved into it. A field that was emptied, or whose text was
        swapped for other text (a sent message, another tab, another file), starts over from what it holds now.

        The first field seen after Live is turned on has no baseline: someone who turns Live on mid-draft wants
        that draft checked. Every field focus moves to after that keeps what it already held as its baseline.
        """
        if obs.source != "focused_field":
            return
        key = obs.field_key
        last = self._field_text.get(key)
        if key not in self._baselines or not obs.text.strip() or (last is not None and replaced(last, obs.text)):
            self._baselines[key] = obs.text if self._writing_seen else ""
        self._writing_seen = True
        self._field_text[key] = obs.text
        for store in (self._baselines, self._field_text):
            store.move_to_end(key)
            while len(store) > BASELINE_LIMIT:
                store.popitem(last=False)

    def _pre_existing(self, correction: Correction, obs: Observation) -> bool:
        """The slip was already in the field before the person started editing it, so it is not theirs.

        Only for kinds about the person's own words, and never for a correction already waiting or on screen.
        Counting matters: a doubled word that was there once and that the person types again is theirs.
        """
        if correction.kind not in OWN_WORDS_KINDS or obs.source != "focused_field":
            return False
        if correction.key in self._pending or any(i.correction.key == correction.key for i in self._visible.values()):
            return False
        evidence = _norm(correction.evidence)
        baseline = _norm(self._baselines.get(obs.field_key, ""))
        if not evidence or not baseline or evidence not in baseline:
            return False
        return baseline.count(evidence) >= _norm(obs.text).count(evidence)

    def _typed_on(self, place: str, now: float) -> None:
        """The person typed again in the same field: learn how long they paused, and whether a card cut in."""
        if self._personal_timing():
            self.timing.observe_pause(now - self._last_text_change, mid_sentence=not at_boundary(self._last_text))
        for item in self._visible.values():
            if item.window == place and item.resumed_after is None:
                item.resumed_after = now - item.delivered_at

    def _outcome(self, item: Pending | None, verdict: str) -> None:
        # A catch held through a call was timed by the call, not by the person's typing: it teaches nothing.
        if item is not None and item.source != "speech" and not item.held_in_call and self._personal_timing():
            self.timing.record_outcome(
                verdict, item.resumed_after, paused_for=item.paused_for, needed=item.needed_pause
            )

    def _note(self, correction: Correction, source: str, window: str, now: float) -> None:
        if any(item.correction.key == correction.key for item in self._visible.values()):
            for item in self._visible.values():
                if item.correction.key == correction.key:
                    item.last_seen = now
            return
        if any(entry.get("key") == correction.key for entry in self._recent):
            return  # already told once; a correction is never repeated for the same text
        existing = self._pending.get(correction.key)
        if existing:
            existing.last_seen = now
            # Keep the latest reading: a slip first seen at the end of the text stops trailing once they type on.
            existing.correction = correction
            return
        self._pending[correction.key] = Pending(correction, source, now, now, window=window)

    def offer_speech(self, text: str) -> None:
        """Check a sentence the person said. Spoken corrections land as soon as they stop talking."""
        cfg = self._config()
        if not getattr(cfg, "enabled", False) or self._paused or not getattr(cfg, "listen_to_speech", True):
            return
        now = self._clock()
        kinds = tuple(kind for kind in self._kinds() if kind in SPEECH_KINDS)
        ctx = self._context(None, "speech")
        for correction in detect(text, ctx, kinds=kinds):
            self._note(correction, "speech", "speech", now)
        if COMMITMENT in self._kinds():
            # "I'll send Priya the deck by Friday", said to Heliox or while talking with it: offer the reminder.
            try:
                promises = find_commitments(text, ctx, self._wallclock(), self._reminders())
            except Exception:
                logger.debug("Spoken commitment check failed", exc_info=True)
                promises = []
            for promise in promises:
                self._note(promise, "speech", "speech", now)
        if self._grounding() and "personal_fact" in self._kinds():
            # What they asked for, against their calendar and lists, while the planner gets going.
            asyncio.ensure_future(self._ground(text, "speech", "speech"))
        if any(item.source == "speech" for item in self._pending.values()):
            # Say it now rather than at the next tick: the person has just finished speaking.
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(self._deliver_due(self._clock()))

    # ── corrections a local model finds in the person's own data (pilot.live.grounded) ──

    def _grounding(self) -> bool:
        return self.grounded is not None and bool(getattr(self._config(), "grounded_checks", False))

    def _maybe_ground(self, now: float) -> None:
        obs = self._last_obs
        if not self._grounding() or obs is None or obs.source != "focused_field" or not obs.editable:
            return
        if self._user_speaking():
            return
        if self._grounded_task is not None and not self._grounded_task.done():
            return
        if (
            now - self._last_text_change < GROUNDED_PAUSE_SECONDS
            or now - self._last_grounded_at < GROUNDED_EVERY_SECONDS
        ):
            return
        if self._last_hash == self._grounded_hash:
            return  # already checked this text
        self._grounded_hash = self._last_hash
        self._last_grounded_at = now
        self._grounded_task = asyncio.create_task(self._ground(obs.text, obs.source, obs.field_key))

    async def _ground(self, text: str, source: str, window: str) -> None:
        if source == "speech":
            # The local model reads dates and numbers as written ("the twelfth of November" is "12 November").
            text = spoken.to_written(text)
        try:
            found = await self.grounded.check(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Grounded check failed", exc_info=True)
            return
        obs = self._last_obs
        for correction in found:
            if source != "speech":
                # The answer took a while: only raise it if the words it quotes are still on screen.
                if obs is None or obs.field_key != window or correction.evidence not in obs.text:
                    continue
                if self._pre_existing(correction, obs):
                    continue
            self._note(correction, source, window, self._clock())

    def _gone(self, item: Pending, now: float) -> bool:
        return item.missing_since is not None and now - item.missing_since > STALE_AFTER_SECONDS

    async def notice_risk(self, pattern: str, source: str) -> None:
        """A risk supervision caught, by rule name only (the matched text never reaches Live).

        Only keyboard-hook matches become cards, since they are about what the person just typed. A screen OCR
        match can come from anywhere on screen (a web page, a chat transcript, one of Live's own cards) and has
        no evidence the person could check; the focused-field detectors already catch keys and risky commands
        in what the person writes, with the evidence. A notice is a normal card, so the level's rate limits,
        focus sessions, relevance and mutes apply, and each rule raises at most one card an hour. When a field
        card of the same kind is already waiting, on screen or recent, that card covers it.
        """
        if source not in NOTICE_SOURCES:
            return
        kind = "secret_in_text" if pattern == "credential_exposure" else "risky_command"
        now = self._clock()
        last = self._notices.get(pattern)
        if last is not None and now - last < NOTICE_QUIET_SECONDS:
            return
        if any(
            item.correction.kind == kind and item.source == "focused_field"
            for item in [*self._pending.values(), *self._visible.values()]
        ) or any(
            entry["kind"] == kind and entry["source"] == "focused_field" and now - entry["at"] < NOTICE_QUIET_SECONDS
            for entry in self._recent
        ):
            return
        self._notices[pattern] = now
        title, message = _RISK_NOTICES.get(pattern, ("That looks risky", "Heads up, what you just typed looks risky."))
        correction = Correction(
            kind=kind,
            key=f"notice:{pattern}",
            title=title,
            message=message,
            evidence="",
            urgency=NORMAL,
        )
        self._note(correction, "notice", "notice", now)

    def _closed(self, item: Pending) -> bool:
        """The window the slip was typed in has been closed: there is nothing left for the person to fix."""
        if item.source != "focused_field" or "|#" not in item.window:
            return False
        handle = focused_text.window_handle(item.window.split("|#", 1)[1])
        return handle is not None and not self._window_exists(handle)

    def _watch_call(self, now: float) -> None:
        try:
            call = str(self._in_call() or "")
        except Exception:
            call = ""
        if self._call and not call:
            self._call_ended_at = now
            logger.info("Live: the call ended; held catches are said if they still apply")
        self._call = call
        if call:
            for item in self._pending.values():
                item.held_in_call = True

    def in_call(self, now: float | None = None) -> bool:
        """In a call, or one ended a moment ago."""
        now = self._clock() if now is None else now
        return bool(self._call) or now - self._call_ended_at < AFTER_CALL_SECONDS

    def _expire(self, now: float) -> None:
        if now - self._last_window_check >= WINDOW_CHECK_SECONDS:
            self._last_window_check = now
            for key, item in list(self._pending.items()):
                if self._closed(item):
                    del self._pending[key]
            for correction_id, item in list(self._visible.items()):
                if self._closed(item):
                    # Neither fixed nor ignored: the document or message went away, so relevance learns nothing.
                    self._visible.pop(correction_id, None)
                    asyncio.ensure_future(
                        self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": "closed"})
                    )
        for key, item in list(self._pending.items()):
            ttl = SPEECH_TTL_SECONDS if item.source == "speech" else PENDING_TTL_SECONDS
            # A catch held through a call keeps its time from when the call ended: a long call does not expire it.
            since = max(item.first_seen, self._call_ended_at) if item.held_in_call else item.first_seen
            if self._call and item.held_in_call and item.source != "speech":
                since = now
            if self._gone(item, now) or now - since > ttl:
                del self._pending[key]
        for correction_id, item in list(self._visible.items()):
            if self._gone(item, now):
                # The evidence left the same window it was found in: they fixed it after being told.
                self.relevance.update(item.correction.kind, "fixed")
                reason = "fixed"
            elif now - item.delivered_at > VISIBLE_TTL_SECONDS:
                self.relevance.update(item.correction.kind, "ignored")
                reason = "expired"
            else:
                continue
            self._outcome(item, reason)
            self._visible.pop(correction_id, None)
            asyncio.ensure_future(self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": reason}))

    def _verdicts(self, now: float, spoken_at: float, shown_at: float) -> list[tuple[str, Pending, str]]:
        cfg = self._config()
        level = LEVELS.get(getattr(cfg, "level", DEFAULT_LEVEL), LEVELS[DEFAULT_LEVEL])
        muted = frozenset(getattr(cfg, "muted_kinds", []) or [])
        snoozed = self.snoozed_kinds(now)
        personal_pause = (
            self.timing.pause_for(self._last_text, level.pause_seconds) if self._personal_timing() else None
        )
        self._needed_pause = personal_pause if personal_pause is not None else level.pause_seconds
        order = {URGENT: 0}
        verdicts = []
        audio = self._heliox_speaking()
        calling = self.in_call(now)
        for key, item in sorted(self._pending.items(), key=lambda kv: order.get(kv[1].correction.urgency, 1)):
            if item.correction.kind in snoozed and item.correction.urgency != URGENT:
                verdicts.append((key, item, DROP))  # "not now" for this kind: an urgent risk still gets through
                continue
            held = not self.may_speak(item)
            act = Activity(
                now=now,
                last_text_change=self._last_text_change,
                user_speaking=self._user_speaking(),
                heliox_speaking=audio or held or calling,  # in a call an urgent risk is a card, never a voice
                floor_held=held,
                in_call=calling,
                focus_active=self._focus_active(),
                visible_cards=len(self._visible) + sum(v in (SHOW, SPEAK) for _, _, v in verdicts),
                # Everything due at this pause goes out together as one interruption, so the rate limits
                # compare against the turn before this one.
                last_spoken_at=spoken_at,
                last_shown_at=shown_at,
                personal_pause=personal_pause,
            )
            verdicts.append((key, item, decide(item, act, level, self.relevance.score(item.correction.kind), muted)))
        return verdicts

    async def _deliver_due(self, now: float) -> None:
        spoken_at, shown_at = self._last_spoken_at, self._last_shown_at
        verdicts = self._verdicts(now, spoken_at, shown_at)
        typed = any(v in (SHOW, SPEAK) and item.source not in ("speech", "notice") for _, item, v in verdicts)
        if typed and self._last_perceive < now:
            # Look again before speaking up: the text is read every 0.6 s, so they may have started typing since.
            self._last_perceive = now
            await self._perceive(now)
            verdicts = self._verdicts(now, spoken_at, shown_at)
        said: list[str] = []
        urgent_said = False
        after_call = False
        for key, item, verdict in verdicts:
            if verdict == DROP:
                del self._pending[key]
            elif verdict in (SHOW, SPEAK):
                del self._pending[key]
                await self._deliver(item, spoken=verdict == SPEAK, now=now)
                if verdict == SPEAK:
                    after_call = after_call or item.held_in_call
                    said.append(say(item.correction.title, item.correction.message, item.source))
                    # An answer to what the person just said is part of the conversation: it goes out on the
                    # correction channel, so Heliox's own "Opening it now." cannot talk over it.
                    urgent_said = urgent_said or item.correction.urgency == URGENT or item.source == "speech"
        if said:
            # Several catches in one turn are said as one reply, not as a queue of interruptions.
            self._last_spoken_at = now
            text = " ".join(said)
            if after_call:
                text = "While you were on the call: " + _lower_first(text)
            asyncio.ensure_future(self._say_aloud(text, urgent_said, now, spoken_at))

    async def _say_aloud(self, text: str, urgent: bool, now: float, before: float) -> None:
        """Speak, and when the remark was not heard (a call, a focus session, held by the floor) give the spoken
        rate limit back, so the next real remark is not held up by one nobody heard."""
        try:
            outcome = await self._speak(text, urgent)
        except Exception:
            outcome = "error"
        status = str(getattr(outcome, "status", outcome) or "")
        if status in UNHEARD and self._last_spoken_at == now:
            self._last_spoken_at = before
            for entry in self._recent:
                if entry.get("at") == now and entry.get("spoken"):
                    entry["spoken"] = False

    async def _deliver(self, item: Pending, *, spoken: bool, now: float) -> None:
        correction = item.correction
        item.id = uuid.uuid4().hex[:10]
        item.delivered = True
        item.delivered_at = now
        item.paused_for = now - self._last_text_change
        item.needed_pause = self._needed_pause
        item.mode = SPEAK if spoken else SHOW
        if item.source != "speech":
            self._visible[item.id] = item
        payload = {
            "id": item.id,
            "kind": correction.kind,
            "title": correction.title,
            "message": correction.message,
            "detail": correction.detail,
            "evidence": correction.evidence,
            "fix": correction.fix,
            "fix_kind": correction.fix_kind,
            "urgency": correction.urgency,
            "spoken": spoken,
            "source": item.source,
            "app": "" if item.source == "notice" else app_label(item.window.split("|", 1)[0]),
        }
        self._recent.append({**payload, "key": correction.key, "at": now})
        self._last_shown_at = now
        await self._emit("live_interjection", payload)
        if spoken:
            self._last_spoken_at = now

    async def _withdraw_all(self, reason: str) -> None:
        for correction_id in list(self._visible):
            self._visible.pop(correction_id, None)
            with contextlib.suppress(Exception):
                await self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": reason})

    # ── sharing the floor with the conversation (pilot.live.floor) ──

    def may_speak(self, item: Pending | None = None) -> bool:
        """Whether Live may start speaking now: never while a conversation turn holds the floor, except a catch
        about the words that opened the turn, before the turn has said anything."""
        if self.floor is None:
            return True
        about = item is not None and item.source == "speech" and self.floor.about_this_turn(item.first_seen)
        return bool(self.floor.live_may_speak(about_this_turn=about))

    def snoozed_kinds(self, now: float | None = None) -> frozenset[str]:
        now = self._clock() if now is None else now
        self._snoozed = {kind: until for kind, until in self._snoozed.items() if until > now}
        return frozenset(self._snoozed)

    def snooze(self, kind: str, seconds: float = SNOOZE_SECONDS) -> None:
        self._snoozed[kind] = self._clock() + seconds

    def current_observation(self) -> Observation | None:
        """Live's latest look at the focused window (already through Live's privacy gates), or None when Live is
        off, paused or sees nothing. For the conversation layer's screen context."""
        cfg = self._config()
        if not getattr(cfg, "enabled", False) or self._paused:
            return None
        return self._last_obs

    def last_spoken_entries(self) -> list[dict[str, Any]]:
        """The catches said together in Live's latest spoken remark (several catches in one turn are one remark)."""
        return [
            {k: v for k, v in entry.items() if k != "key"}
            for entry in self._recent
            if entry.get("spoken") and entry.get("at") == self._last_spoken_at
        ]

    def last_raised(self) -> dict[str, Any] | None:
        """The latest card or remark Live raised (within the reply window), without its internal fields."""
        entry = self._latest(self._clock())
        if entry is None:
            return None
        return {k: v for k, v in entry.items() if k != "key"}

    # ── the person's response ──

    def _latest(self, now: float) -> dict[str, Any] | None:
        if not self._recent:
            return None
        entry = self._recent[-1]
        return entry if now - entry["at"] <= REPLY_WINDOW_SECONDS else None

    def _find(self, correction_id: str) -> dict[str, Any] | None:
        return next((entry for entry in self._recent if entry["id"] == correction_id), None)

    async def feedback(self, correction_id: str, verdict: str) -> dict[str, Any]:
        entry = self._find(correction_id)
        if entry is None:
            return {"status": "error", "message": "That correction is no longer available."}
        if verdict == "mute":
            self.relevance.update(entry["kind"], "dismissed")
            if self._mute_kind:
                await self._mute_kind(entry["kind"])
        else:
            self.relevance.update(entry["kind"], verdict)
        self._outcome(self._visible.get(correction_id), verdict)
        if correction_id in self._visible:
            self._visible.pop(correction_id, None)
            await self._emit("live_interjection_withdrawn", {"id": correction_id, "reason": verdict})
        return {"status": "ok", "kind": entry["kind"], "relevance": round(self.relevance.score(entry["kind"]), 2)}

    def explain(self, correction_id: str = "") -> str:
        entry = self._find(correction_id) if correction_id else self._latest(self._clock())
        if entry is None:
            if correction_id:
                return "That correction is no longer available."
            return "Nothing to explain: Live hasn't raised anything in the last minute and a half."
        parts = [say(entry["title"], entry["message"], entry.get("source", ""))]
        if entry.get("detail"):
            parts.append(entry["detail"])
        if entry.get("fix") and entry.get("fix_kind") in ("replace", "command"):
            parts.append(f"The fix: {entry['fix']}.")
        return " ".join(parts)

    def why_spoken(self, correction_id: str = "") -> str:
        """Why Live spoke, said from what it noticed: where, the words it saw or heard, and what was wrong."""
        entry = self._find(correction_id) if correction_id else self._latest(self._clock())
        if entry is None:
            return self.explain(correction_id)
        return because(entry)

    async def handle_reply(self, normalized: str) -> LiveReply | None:
        """ "Thanks", "not now", "why?", "don't mention these" and "fix it", when they follow a recent correction."""
        entry = self._latest(self._clock())
        if entry is None:
            return None
        if _WHY.match(normalized):
            return LiveReply(because(entry))
        if _THANKS.match(normalized):
            await self.feedback(entry["id"], "helpful")
            return LiveReply("Glad it helped.")
        if _MUTE.match(normalized):
            await self.feedback(entry["id"], "mute")
            label = KIND_LABELS.get(entry["kind"], "that kind of thing").lower()
            return LiveReply(
                f"Okay, I won't mention these again: {label}. You can turn them back on in Settings, under Live."
            )
        if _DISMISS.match(normalized):
            await self.feedback(entry["id"], "dismissed")
            if entry.get("urgency") != URGENT:
                self.snooze(entry["kind"])
                return LiveReply("Okay. I won't bring that kind of thing up for the next half hour.")
            return LiveReply("Okay, leaving it.")
        if _FIX.match(normalized):
            if entry.get("fix_kind") == "ask" and entry.get("fix"):
                await self.feedback(entry["id"], "used_fix")
                return LiveReply("On it.", delegate=entry["fix"])
            if entry.get("fix"):
                await self.feedback(entry["id"], "used_fix")
                return LiveReply(f"I can't edit your text for you, so here's the fix to use: {entry['fix']}.")
            return LiveReply("There's no automatic fix for that one.")
        return None

    # ── status ──

    def _user_state(self, now: float) -> str:
        if self._user_speaking():
            return "speaking"
        if self._last_obs is None:
            return "away"
        return "typing" if now - self._last_text_change < 1.2 else "paused"

    def status(self) -> dict[str, Any]:
        cfg = self._config()
        now = self._clock()
        return {
            "status": "ok",
            "enabled": bool(getattr(cfg, "enabled", False)),
            "paused": self._paused,
            "running": self.running,
            "level": getattr(cfg, "level", DEFAULT_LEVEL),
            "watching": app_label(self._last_obs.app) if self._last_obs else "",
            "source": self._last_obs.source if self._last_obs else "",
            "user_state": self._user_state(now),
            "visible": [entry for entry in self._recent if entry["id"] in self._visible],
            "recent": [{k: v for k, v in entry.items() if k not in ("key", "at")} for entry in list(self._recent)[-8:]],
            "relevance": self.relevance.snapshot(),
            "muted_kinds": list(getattr(cfg, "muted_kinds", []) or []),
            "personal_timing": self._personal_timing(),
            "grounded": {"enabled": self._grounding(), **(self.grounded.stats() if self.grounded else {})},
            "timing": self.timing.snapshot(LEVELS.get(getattr(cfg, "level", ""), LEVELS[DEFAULT_LEVEL]).pause_seconds),
        }

    def forget(self, what: str) -> None:
        """Drop learned state from memory after its file was deleted (Your data)."""
        if what == "live":
            self.relevance.counts = {}
        elif what == "live_timing":
            self.timing.forget()

    async def _publish_state(self, now: float) -> None:
        cfg = self._config()
        state = {
            "enabled": bool(getattr(cfg, "enabled", False)),
            "paused": self._paused,
            "watching": app_label(self._last_obs.app) if self._last_obs and getattr(cfg, "enabled", False) else "",
            "user_state": self._user_state(now) if getattr(cfg, "enabled", False) else "off",
        }
        if state != self._state:
            self._state = state
            with contextlib.suppress(Exception):
                await self._emit("live_state", state)
