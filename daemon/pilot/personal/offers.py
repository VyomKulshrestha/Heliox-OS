"""Offers: Heliox suggests something the person usually does, at the moment they usually do it, and asks.

Live catches slips; this is the other half of "watch your task, suggest you": an offer built from what Heliox
already keeps about the person, computed on this machine by rules, said once, and easy to refuse for good.

Sources, all local:

* **Usual apps** (only when the person turned the activity timeline on): an app they first brought to the front
  within 20 minutes of this time of day on at least 3 of their last 5 working days of the same type (weekdays or
  weekend days), and not yet today. "You usually open Outlook about now. Want me to open it?"
* **Habits, goals and to-dos**, from ``pilot.workflows.suggestions.Suggester``: a streak not kept today, a goal gone
  quiet with a next step, open to-dos in the evening. Taken only when the older background suggestions
  (``suggestions.enabled``) are off, so nothing is offered twice.

Manners:

* At most ``MAX_PER_HOUR`` offer an hour and ``MAX_PER_DAY`` a day, at least ``MIN_GAP_SECONDS`` apart, the same
  thing at most once a day, and only at a quiet moment (the server decides: nobody talking, no conversation turn,
  not typing, no focus session).
* "Why did you offer that?" gets the evidence ("On 4 of your last 5 weekdays you first opened Outlook between 8:50
  and 9:30, and you haven't opened it today."). "No thanks" drops it for the day. "Don't offer that again" silences
  that offer for good (kept in ``offers.json``). "Stop offering things" turns offers off in Settings.
* "Yes" sends the request down the normal road (the conversation layer and the planner, with every approval rule);
  voice never approves a plan.

Default (``suggestions.offers``): ``auto``. Offers are on when everything they use is on this machine (always, here:
no source calls a model) and the person has told Heliox something about themselves (a non-empty profile), so a
first run with an empty profile never starts suggesting things out of nowhere. ``on`` and ``off`` override.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.live.sequence import next_seq
from pilot.turn_changes import note_change

logger = logging.getLogger("pilot.personal.offers")

MAX_PER_HOUR = 1
MAX_PER_DAY = 3
MIN_GAP_SECONDS = 1200.0
CHECK_EVERY_SECONDS = 60.0
ANSWER_WINDOW_SECONDS = 120.0  # "yes" or "no" refers to the offer this long
EXPLAIN_WINDOW_SECONDS = 600.0  # "why did you offer that" and "don't offer that" this long
USUAL_WINDOW_MINUTES = 20
USUAL_NEED = 3
USUAL_OF_LAST = 5
HISTORY_DAYS = 21
MODES = ("auto", "on", "off")
# Never offered as "usual": the shell, Heliox itself, and things that are always in front anyway.
_NEVER_USUAL = frozenset(
    {"explorer.exe", "finder", "dock", "systemsettings.exe", "searchhost.exe", "shellexperiencehost.exe",
     "heliox", "heliox-os", "heliox-os.exe", "gnome-shell", "plasmashell", "lockapp.exe", "unknown", ""}
)  # fmt: skip

_YES = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|yes please|go ahead|do it|sounds good|why not|go for it|sure thing"
    r"|okay do it|yes do it)$"
)
_NO = re.compile(r"^(?:no|nope|no thanks|no thank you|not now|not today|maybe later|later|nah)$")
_WHY = re.compile(
    r"^(?:why|how come|why do you ask|why did you (?:offer|suggest|ask|mention) (?:that|it|this)"
    r"|why are you (?:offering|suggesting|asking) (?:that|this)|what made you (?:offer|suggest) that)$"
)
_SILENCE = re.compile(
    r"^(?:please )?(?:don(?:'| )?t|do not|never) (?:offer|suggest|ask) (?:me )?(?:that|this|it)(?: again| any more| anymore)?$"
    r"|^(?:please )?stop (?:offering|suggesting|asking) (?:me )?(?:that|this)$"
)
# Heliox Live's words for "never this kind again", said after an offer: every offer of that kind goes quiet.
_SILENCE_KIND = re.compile(
    r"^(?:please )?(?:don(?:'| )?t|do not|never) (?:mention|offer|suggest|bring up) (?:these|those|things like that"
    r"|that kind of thing)(?: again| any more| anymore)?(?: please)?$"
    r"|^(?:please )?stop (?:mentioning|offering|suggesting) (?:these|those)(?: please)?$"
)
_SILENCE_ALL = re.compile(
    r"^(?:please )?(?:stop (?:offering|suggesting) (?:me )?(?:things|stuff|anything)"
    r"|(?:don(?:'| )?t|do not) (?:offer|suggest) (?:me )?anything(?: else)?|no more (?:offers|suggestions)"
    r"|turn off offers)$"
)


@dataclass(frozen=True)
class Offer:
    key: str  # what "don't offer that" silences: "usual_app:outlook.exe", "habit:reading", ...
    kind: str  # "usual_app", "habit", "stale_goal", "todos"
    text: str  # said and shown
    request: str  # what "yes" sends down the normal road
    why: str  # the evidence, in a sentence


# The one offer made while the profile is still empty: three questions to get to know the person
# (pilot.personal.getting_to_know). Every other offer waits until the profile has something in it.
INTRO = Offer(
    key="get_to_know",
    kind="get_to_know",
    text="Would you like to tell me a little about yourself? Three short questions, and you can skip any of them.",
    request="get to know me",
    why=(
        "I don't know anything about you yet, so I can't suggest anything useful or talk the way you like. "
        "It stays on this computer; nothing you answer is sent to a language model."
    ),
)


@dataclass(frozen=True)
class OfferReply:
    reply: str
    request: str = ""  # hand this to the normal road (the person said yes)
    silence_all: bool = False


@dataclass
class _Made:
    offer: Offer
    at: float
    day: str
    answered: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    seq: int = field(default_factory=next_seq)  # order against Live's cards, never by clock


def _day_type(day: datetime) -> str:
    return "weekend" if day.weekday() >= 5 else "weekday"


def _clock_text(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60}:{minutes % 60:02d}"


def usual_apps(
    spans: list[dict[str, Any]],
    now: datetime,
    *,
    label: Callable[[str], str] = lambda app: app,
    window: int = USUAL_WINDOW_MINUTES,
    need: int = USUAL_NEED,
    of_last: int = USUAL_OF_LAST,
) -> list[Offer]:
    """Apps first opened about now on most recent days of the same type, and not yet today. Pure: local time."""
    first_seen: dict[str, dict[str, float]] = {}  # day -> app -> first start (local timestamp)
    for span in spans:
        app = str(span.get("app") or "").strip()
        if app.casefold() in _NEVER_USUAL:
            continue
        start = datetime.fromtimestamp(float(span["start"]))
        day = start.date().isoformat()
        apps = first_seen.setdefault(day, {})
        apps[app] = min(apps.get(app, float("inf")), float(span["start"]))
    today = now.date().isoformat()
    kind = _day_type(now)
    past = sorted(
        (day for day in first_seen if day < today and _day_type(datetime.fromisoformat(day)) == kind), reverse=True
    )[:of_last]
    if len(past) < of_last:
        return []  # not enough history of this kind of day to call anything usual
    seen_today = set(first_seen.get(today, {}))
    minute_now = now.hour * 60 + now.minute
    offers: list[tuple[int, Offer]] = []
    candidates = {app for day in past for app in first_seen[day]} - seen_today
    for app in sorted(candidates):
        hits = []
        for day in past:
            started = first_seen[day].get(app)
            if started is None:
                continue
            at = datetime.fromtimestamp(started)
            minute = at.hour * 60 + at.minute
            if abs(minute - minute_now) <= window:
                hits.append(minute)
        if len(hits) < need:
            continue
        name = label(app) or app
        days = "weekdays" if kind == "weekday" else "weekend days"
        why = (
            f"On {len(hits)} of your last {len(past)} {days} you first opened {name} between "
            f"{_clock_text(min(hits))} and {_clock_text(max(hits))}, and you haven't opened it yet today."
        )
        offers.append(
            (
                len(hits),
                Offer(
                    key=f"usual_app:{app.casefold()}",
                    kind="usual_app",
                    text=f"You usually open {name} about now. Want me to open it?",
                    request=f"open {name}",
                    why=why,
                ),
            )
        )
    offers.sort(key=lambda pair: -pair[0])
    return [offer for _, offer in offers]


_KIND_WORDS = {
    "usual_app": "opening the apps you usually open",
    "habit": "logging your habits",
    "todos": "reading your to-do list",
    "stale_goal": "reminders for quiet goals",
    "get_to_know": "questions to get to know you",
}


def _label(offer: Offer) -> str:
    """What was offered, in a few words, for the list of silenced offers."""
    if offer.kind == "usual_app":
        return f"Opening {offer.request.removeprefix('open ').strip()} at the usual time"
    if offer.kind == "habit":
        return f"Logging {offer.request.removeprefix('log ').strip()}"
    if offer.kind == "todos":
        return "Reading your to-do list in the evening"
    if offer.kind == "stale_goal":
        return f"A reminder to {offer.request.split(' to ', 1)[-1]}"
    if offer.kind == "get_to_know":
        return "Questions to get to know you"
    return offer.text[:80]


def from_suggestion(pick: dict[str, Any] | None) -> Offer | None:
    """An offer from a ``Suggester`` pick that can be acted on, with the reason it was picked."""
    if not pick or not pick.get("say"):
        return None
    kind, text, say = str(pick.get("kind", "")), str(pick.get("text", "")), str(pick["say"])
    if kind == "habit":
        name = say.removeprefix("log ").strip()
        return Offer(
            key=f"habit:{name.casefold()}",
            kind=kind,
            text=f"{text} Want me to log it?",
            request=say,
            why=f"You've logged {name} every day lately and not yet today, and the day is well under way.",
        )
    if kind == "todos":
        return Offer(
            key="todos",
            kind=kind,
            text=f"{text} Want me to read the list?",
            request=say,
            why="It's evening and your to-do list still has open items.",
        )
    if kind == "stale_goal":
        step = say.removeprefix("remind me to ").strip()
        return Offer(
            key=str(pick.get("key") or f"stale_goal:{step}"),
            kind=kind,
            text=f"{text} Want a reminder for it in an hour?",
            request=f"remind me in an hour to {step}",
            why="That goal has had no update for a while, and it has a next step written down.",
        )
    return None


@dataclass
class Offers:
    """Picks the next offer, keeps the caps, and answers the person's replies."""

    mode: Callable[[], str]
    profile_empty: Callable[[], bool]
    suggestion: Callable[[], dict[str, Any] | None] = lambda: None
    legacy_suggestions_on: Callable[[], bool] = lambda: False
    spans: Callable[[float, float], list[dict[str, Any]]] | None = None
    label: Callable[[str], str] = lambda app: app
    now: Callable[[], datetime] = datetime.now
    clock: Callable[[], float] = time.monotonic
    state_path: Path | None = None
    made: list[_Made] = field(default_factory=list)
    declined: dict[str, str] = field(default_factory=dict)  # key -> day
    # key -> what was offered, in words, so Settings can list it and undo it
    silenced: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_path and self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8")).get("silenced", {})
                if isinstance(raw, list):  # the first version kept keys only
                    raw = {key: key for key in raw}
                self.silenced = {str(k): str(v) for k, v in raw.items() if isinstance(k, str)}
            except (OSError, ValueError, AttributeError):
                logger.debug("Ignoring an unreadable offers file", exc_info=True)

    # ── on or off ──

    def active(self) -> bool:
        mode = str(self.mode() or "auto").casefold()
        if mode == "off":
            return False
        if mode == "on":
            return True
        try:
            return not self.profile_empty()
        except Exception:
            return False

    # ── choosing ──

    def candidates(self) -> list[Offer]:
        now = self.now()
        found: list[Offer] = []
        if self.spans is not None:
            start = (now - timedelta(days=HISTORY_DAYS)).timestamp()
            try:
                found += usual_apps(self.spans(start, now.timestamp()), now, label=self.label)
            except Exception:  # one source failing never stops the others
                logger.warning("Usual-app offers failed; the other offer sources still ran", exc_info=True)
        if not self.legacy_suggestions_on():
            try:
                offer = from_suggestion(self.suggestion())
            except Exception:
                logger.warning("Suggestion offers failed; the other offer sources still ran", exc_info=True)
                offer = None
            if offer is not None:
                found.append(offer)
        return found

    def _today(self) -> str:
        return self.now().date().isoformat()

    def room(self) -> bool:
        """Under the hourly and daily caps and the minimum gap."""
        now, today = self.clock(), self._today()
        self.made = [m for m in self.made if m.day == today or now - m.at < 3600]
        if any(now - m.at < MIN_GAP_SECONDS for m in self.made):
            return False
        if sum(now - m.at < 3600 for m in self.made) >= MAX_PER_HOUR:
            return False
        return sum(m.day == today for m in self.made) < MAX_PER_DAY

    def _intro_due(self) -> bool:
        if (
            str(self.mode() or "auto").casefold() == "off"
            or INTRO.key in self.silenced
            or "kind:get_to_know" in self.silenced
        ):
            return False
        try:
            empty = bool(self.profile_empty())
        except Exception:
            return False
        return empty and not any(m.offer.key == INTRO.key for m in self.made)

    def next_offer(self) -> Offer | None:
        if not self.room():
            return None
        if self._intro_due():
            return INTRO  # once a run, until the person answers, says "don't offer this" or tells Heliox something
        if not self.active():
            return None
        today = self._today()
        offered_today = {m.offer.key for m in self.made if m.day == today}
        for offer in self.candidates():
            if (
                offer.key in self.silenced
                or f"kind:{offer.kind}" in self.silenced
                or offer.key in offered_today
                or self.declined.get(offer.key) == today
            ):
                continue
            return offer
        return None

    def mark_offered(self, offer: Offer) -> str:
        made = _Made(offer, self.clock(), self._today())
        self.made.append(made)
        return made.id

    # ── the person's answer ──

    def latest(self, within: float = EXPLAIN_WINDOW_SECONDS) -> _Made | None:
        if not self.made:
            return None
        made = self.made[-1]
        return made if self.clock() - made.at <= within else None

    def _answered(self, made: _Made | None = None) -> None:
        (made or self.made[-1]).answered = True

    def silence(self, offer: Offer) -> None:
        self.silenced[offer.key] = _label(offer)
        self._save()

    def silence_kind(self, kind: str) -> None:
        self.silenced[f"kind:{kind}"] = f"All offers like this: {_KIND_WORDS.get(kind, kind)}"
        self._save()

    def unsilence(self, key: str) -> bool:
        if self.silenced.pop(key, None) is None:
            return False
        self._save()
        return True

    def answer(self, offer_id: str, answer: str) -> OfferReply | None:
        """A button on the offer's card: "yes", "not_today", "silence" or "why". None when the offer is unknown."""
        made = next((m for m in reversed(self.made) if m.id == offer_id), None)
        if made is None:
            return None
        offer = made.offer
        if answer == "why":
            return OfferReply(offer.why)
        if answer == "silence":
            self.silence(offer)
            self._answered(made)
            return OfferReply("Okay, I won't offer that again.")
        if made.answered:
            return OfferReply("That one was already answered.")
        self._answered(made)
        if answer == "yes":
            return OfferReply("Okay.", request=offer.request)
        if answer == "not_today":
            self.declined[offer.key] = made.day
            return OfferReply("Okay, not today.")
        return None

    def status(self) -> dict[str, Any]:
        mode = str(self.mode() or "auto").casefold()
        try:
            empty = bool(self.profile_empty())
        except Exception:
            empty = True
        return {
            "mode": mode if mode in MODES else "auto",
            "active": self.active(),
            "waiting_for_profile": mode == "auto" and empty,
            "silenced": [
                {"key": key, "label": label} for key, label in sorted(self.silenced.items()) if key != INTRO.key
            ],
            # The getting-to-know offer has its own switch in Settings rather than a line in the silenced list.
            "intro": INTRO.key not in self.silenced,
            "per_hour": MAX_PER_HOUR,
            "per_day": MAX_PER_DAY,
        }

    def last_offered_seq(self) -> int | None:
        return self.made[-1].seq if self.made else None

    def reply(self, normalized: str) -> OfferReply | None:
        """The person's answer to the latest offer, or None when it is not one."""
        if _SILENCE_ALL.match(normalized):
            return OfferReply(
                "Okay, no more offers. You can turn them back on in Settings, under Suggestions.", silence_all=True
            )
        made = self.latest()
        if made is None:
            return None
        offer = made.offer
        if _WHY.match(normalized):
            return OfferReply(offer.why)
        if _SILENCE.match(normalized):
            self.silence(offer)
            self._answered(made)
            return OfferReply("Okay, I won't offer that again.")
        if _SILENCE_KIND.match(normalized):
            self.silence_kind(offer.kind)
            self._answered(made)
            return OfferReply(f"Okay, no more offers like that: {_KIND_WORDS.get(offer.kind, 'offers of that kind')}.")
        if made.answered or self.clock() - made.at > ANSWER_WINDOW_SECONDS:
            return None
        if _YES.match(normalized):
            self._answered()  # a second "yes" is not this offer any more
            return OfferReply("Okay.", request=offer.request)
        if _NO.match(normalized):
            self.declined[offer.key] = made.day
            self._answered()
            return OfferReply("Okay, not today.")
        return None

    def forget(self) -> None:
        """Your data: drop what was silenced and declined."""
        self.silenced.clear()
        self.declined.clear()
        self._save()

    def _save(self) -> None:
        note_change("silenced that offer")  # a voice turn that did this cannot be withdrawn
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"silenced": dict(sorted(self.silenced.items()))}), encoding="utf-8")
        except OSError:
            logger.debug("Could not save offers", exc_info=True)


class OfferLoop:
    """Looks for an offer every minute and makes it at a quiet moment."""

    def __init__(
        self,
        offers: Offers,
        *,
        quiet_moment: Callable[[], bool],
        deliver: Callable[[Offer, str], Awaitable[None]],
    ) -> None:
        self._offers = offers
        self._quiet = quiet_moment
        self._deliver = deliver
        self._task: Any = None

    async def tick(self) -> Offer | None:
        if str(self._offers.mode() or "auto").casefold() == "off":
            return None  # next_offer decides the rest: the one intro while the profile is empty, then the others
        try:
            if not self._quiet():
                return None
        except Exception:
            return None
        offer = self._offers.next_offer()
        if offer is None:
            return None
        offer_id = self._offers.mark_offered(offer)
        await self._deliver(offer, offer_id)
        return offer

    async def _loop(self) -> None:
        import asyncio

        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Offer check failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        import asyncio

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="heliox-offers")

    async def stop(self) -> None:
        import asyncio
        import contextlib

        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


__all__ = ["INTRO", "MODES", "Offer", "OfferLoop", "OfferReply", "Offers", "from_suggestion", "usual_apps"]
