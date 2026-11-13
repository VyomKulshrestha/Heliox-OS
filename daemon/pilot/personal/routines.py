"""The person's own routines, learned from what they ask Heliox for, on this computer only.

Kept: for each request the person addressed to Heliox and Heliox took (typed, or said after the wake phrase), the
kind of request and its target as the router has them ("open" + "outlook"), or one of a few requests Heliox answers in
code ("day", "wrap_up", "list"), with the weekday, the hour and the day. Not kept: the words as said, the audio, the
reply; nothing overheard (it never reaches a handler) and nothing Live reads. Kept for 60 days, then dropped.

A routine is the same request in the same hour band (the hour, or the one either side) on at least ``NEED`` of the
last ``OF_LAST`` days of the same kind (weekdays, or weekend days) on which it could have happened, and on at least
``CONCENTRATION`` of all the times it was asked: a request made often at any hour is not a routine. From it comes at
most one offer a day ("You often ask me to open outlook around nine. Want me to do it now?"), with "why" answered
("You've asked for this around nine on three days."). Nothing from it is put into a model's prompt.

The person sees and ends it: "what do you know about my routines" lists what was learned, "forget my routines"
empties it, "stop learning my routines" turns it off (``memory.learn_routines``) and keeps nothing more. It is in
"what do you know about me", in the backup and in Your data, and removal of Heliox's data removes it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pilot.turn_changes import note_change

logger = logging.getLogger("pilot.personal.routines")

KEEP_DAYS = 60
NEED = 4  # days in the band (simulated sparse use: 3 gives 0.32 false routines a month, 4 gives 0.04)
OF_LAST = 10  # ... out of the last this many days of the same kind
CONCENTRATION = 0.7  # and at least this share of all the times it was asked
MAX_ENTRIES = 5000

# Requests Heliox answers in code that a routine may be made of, and what "yes" says to repeat them.
_CODE_KINDS = (
    ("day", re.compile(r"^(?:good morning|morning|(?:start|begin) my day|what(?: s| is) (?:on )?(?:for )?today"
                       r"|what do i have (?:on )?today|what(?: s| is| does) my day look(?:ing)?(?: like)?)$"),
     "what's my day look like", "hear your day"),
    ("wrap_up", re.compile(r"^(?:good night|goodnight|(?:wrap up|end|finish|close out) (?:my |the )?day)$"),
     "wrap up my day", "wrap up the day"),
    ("list", re.compile(r"^(?:what s|what is) on my (?:to do |todo )?list$|^read (?:me )?my (?:to do |todo )?list$"),
     "what's on my list", "hear your list"),
)  # fmt: skip
# Actions worth offering again; anything else (delete, send, pay, move ...) is never learned.
_FILLER_END = re.compile(r"(?:\s+(?:now|please|for me|again|right now|thanks|thank you))+$")
_LEARNED_VERBS = frozenset({"open", "launch", "start", "play", "show", "find", "check"})

ROUTINES_REQUEST = re.compile(
    r"^(?:what do you know about|what have you learned about|what are|tell me) my routines?$"
    r"|^what routines? (?:have you learned|do you know)$"
)
FORGET_ROUTINES = re.compile(r"^(?:please )?(?:forget|clear|delete) my routines?$")
STOP_LEARNING = re.compile(r"^(?:please )?(?:stop|don t|do not) (?:learning|learn|tracking|track) my routines?$")
START_LEARNING = re.compile(r"^(?:please )?(?:you can )?(?:start|resume) learning my routines?(?: again)?$")
PATTERNS = (ROUTINES_REQUEST, FORGET_ROUTINES, STOP_LEARNING, START_LEARNING)

_HOUR_WORDS = ("midnight", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
               "noon")  # fmt: skip
_COUNT_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).casefold()).split())


def _hour_words(hour: int) -> str:
    if hour == 0:
        return "midnight"
    if hour == 12:
        return "noon"
    return _HOUR_WORDS[hour % 12]


def what_to_keep(text: str, *, planned: bool) -> tuple[str, str] | None:
    """(kind, target) for a request that may be learned, or None. A planned request keeps the router's verb and target
    ("open", "outlook") for the verbs worth offering again; a code answer keeps its kind ("day")."""
    said = _normalize(text)
    if not said:
        return None
    if not planned:
        for kind, pattern, _request, _offer in _CODE_KINDS:
            if pattern.match(said):
                return kind, ""
        return None
    from pilot.conversation.router import route

    found = route(said)
    verb, target = str(getattr(found, "verb", "") or ""), _normalize(getattr(found, "target", "") or "")
    target = _FILLER_END.sub("", target).strip()  # "outlook now" and "outlook please" are one routine
    if verb not in _LEARNED_VERBS or not target or len(target) > 60:
        return None
    return verb, target


def _request_for(kind: str, target: str) -> tuple[str, str]:
    """What "yes" sends, and the words for the offer."""
    for code_kind, _pattern, request, words in _CODE_KINDS:
        if kind == code_kind:
            return request, words
    return f"{kind} {target}", f"{kind} {target}"


class RoutineRecord:
    """The kept requests, in a small JSON file, and what they add up to."""

    def __init__(
        self,
        path: Path,
        *,
        enabled: Callable[[], bool] = lambda: True,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._path = Path(path)
        self._enabled = enabled
        self._now = now

    # ── the store ──

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items = data.get("requests", []) if isinstance(data, dict) else []
        oldest = (self._now().date() - timedelta(days=KEEP_DAYS)).isoformat()
        return [
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("day"), str)
            and item["day"] > oldest
            and isinstance(item.get("hour"), int)
            and isinstance(item.get("kind"), str)
        ]

    def _save(self, items: list[dict[str, Any]]) -> None:
        note_change("noted your routine")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f".{self._path.name}.{int(time.time() * 1000)}.tmp")
        tmp.write_text(json.dumps({"requests": items[-MAX_ENTRIES:]}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def list(self) -> list[dict[str, Any]]:
        return self._load()

    def purge(self) -> int:
        """Drop what is older than KEEP_DAYS from the file. Returns how many were dropped."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8")).get("requests", [])
        except (OSError, ValueError, AttributeError):
            return 0
        kept = self._load()
        if len(kept) != len(raw):
            self._save(kept)
        return len(raw) - len(kept)

    def note(self, text: str, *, planned: bool) -> bool:
        """Keep one request the person addressed to Heliox and Heliox took. False when it is not kept (learning off,
        or not a request a routine is made of)."""
        if not self._learning():
            return False
        kept = what_to_keep(text, planned=planned)
        if kept is None:
            return False
        now = self._now()
        items = self._load()  # what is too old goes now
        items.append({"kind": kept[0], "target": kept[1], "day": now.date().isoformat(), "hour": now.hour})
        self._save(items)
        return True

    def forget(self) -> int:
        items = self._load()
        if self._path.exists():
            self._save([])
        return len(items)

    def _learning(self) -> bool:
        try:
            return bool(self._enabled())
        except Exception:
            return True

    # ── what it adds up to ──

    def routines(self) -> list[dict[str, Any]]:
        """Each learned routine: kind, target, hour (the usual one), days (how many of the last OF_LAST days of the
        same kind it was asked in the band). Best first."""
        today = self._now().date()
        weekend_today = today.weekday() >= 5
        days_of_kind = [
            (today - timedelta(days=n)).isoformat()
            for n in range(1, 40)
            if ((today - timedelta(days=n)).weekday() >= 5) == weekend_today
        ][:OF_LAST]
        window = set(days_of_kind)
        by_key: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for item in self._load():
            if item["day"] in window:
                by_key.setdefault((item["kind"], item.get("target", "")), []).append((item["day"], item["hour"]))
        found = []
        for (kind, target), seen in by_key.items():
            best = None
            for hour in sorted({h for _d, h in seen}):
                days = {d for d, h in seen if abs(h - hour) <= 1}
                # Tied to a time of day: most of the times it was asked at all, it was asked in this band.
                in_band = sum(1 for _d, h in seen if abs(h - hour) <= 1)
                if len(days) < NEED or in_band < CONCENTRATION * len(seen):
                    continue
                if best is None or len(days) > best[1]:
                    best = (hour, len(days))
            if best is not None:
                found.append({"kind": kind, "target": target, "hour": best[0], "days": best[1]})
        return sorted(found, key=lambda r: (-r["days"], r["hour"]))

    def asked_today(self, kind: str, target: str) -> bool:
        today = self._now().date().isoformat()
        return any(i["day"] == today and i["kind"] == kind and i.get("target", "") == target for i in self._load())

    def offers(self, now: datetime) -> list[Any]:
        """At most one offer: the best routine whose hour this is and that was not asked for yet today."""
        from pilot.personal.offers import Offer

        if not self._learning():
            return []
        for routine in self.routines():
            if routine["hour"] != now.hour or self.asked_today(routine["kind"], routine["target"]):
                continue
            request, words = _request_for(routine["kind"], routine["target"])
            when = _hour_words(routine["hour"])
            days = _COUNT_WORDS.get(routine["days"], str(routine["days"]))
            return [
                Offer(
                    key=f"learned:{routine['kind']}:{routine['target']}",
                    kind="learned_routine",
                    text=f"You often ask me to {words} around {when}. Want me to do it now?",
                    request=request,
                    why=f"You've asked for this around {when} on {days} days.",
                )
            ]
        return []

    def describe(self) -> str:
        """The routines in plain sentences, for "what do you know about my routines" and "about me"."""
        if not self._learning():
            return "I'm not learning your routines: you turned that off."
        found = self.routines()
        if not found:
            return "I haven't noticed a routine yet."
        named = [
            f"you ask me to {_request_for(r['kind'], r['target'])[1]} around {_hour_words(r['hour'])}"
            for r in found[:4]
        ]
        more = f", and {len(found) - 4} more" if len(found) > 4 else ""
        return f"From what you ask me: {'; '.join(named)}{more}."

    # ── by voice ──

    async def answer(self, text: str, *, stop: Callable[[], Any] | None = None) -> str | None:
        said = _normalize(text)
        if ROUTINES_REQUEST.match(said):
            return self.describe()
        if FORGET_ROUTINES.match(said):
            count = self.forget()
            return f"Done, I've forgotten {count} request{'' if count == 1 else 's'} I'd noted for your routines."
        if STOP_LEARNING.match(said):
            count = self.forget()
            if stop is not None:
                await stop()
            return (
                f"Okay, I've stopped learning your routines and dropped the {count} I'd noted. You can turn it back on "
                "in Settings."
            )
        if START_LEARNING.match(said):
            return "You can turn that back on in Settings, under Things said in passing."
        return None


def entry_count(path: Path) -> int:
    """How many requests are kept (Your data)."""
    return len(RoutineRecord(path).list())


__all__ = ["KEEP_DAYS", "NEED", "OF_LAST", "PATTERNS", "RoutineRecord", "what_to_keep"]
