"""About you: a user-owned profile Heliox learns from what you tell it, and uses.

"Call me Vyom", "remember that I'm a founder", "my manager is Sarah", "I prefer short
answers", "what do you know about me", "forget that I'm a founder". The profile is kept
in ``DATA_DIR/user_profile.json`` on this computer: how to address you, short facts about
you, the people you name, your stated preferences, and how you like to be spoken to. It
is shown and editable in Settings and listed in Your data.

Only what you say in these phrases is stored, as your own words (never inferred silently
from your activity). What is never kept, whatever the route (``pilot.personal.never_kept``): anything that looks
like a password, key or code, health, money, and other people's private details. ``add`` refuses it and says why,
and anything like that stored before is left out when the profile is read and removed by ``sweep``.
Each entry is also mirrored into Heliox's long-term memory as an explicit-user fact, so
the planner already has it; clearing or removing it here retracts the mirrored fact too.
The profile personalizes greetings, the daily brief and tone. It never grants Heliox any
authority or changes what it may do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pilot.personal.never_kept import KINDS as NEVER_KEPT_KINDS
from pilot.personal.never_kept import never_kept
from pilot.turn_changes import note_change
from pilot.workflows.clipboard_history import looks_secret

# "the wifi password is hunter2": the value itself may not look like a key, but the sentence says it is one.
NAMES_A_SECRET = re.compile(
    r"\b(?:password|passcode|passphrase|pin(?: code| number)?|api key|secret key|access key|private key|token"
    r"|security code|cvv|cvc|recovery code|seed phrase)s?\b\s*(?:is|was|=|:)\s*\S",
    re.IGNORECASE,
)

logger = logging.getLogger("pilot.workflows.user_profile")

MAX_ENTRIES = 60  # per list
MAX_TEXT = 200
MAX_NAME = 40

# Relations that make "my X is Y" / "Y is my X" clearly a person, not a task or a question.
RELATIONS = (
    "manager", "boss", "supervisor", "lead", "mentor", "assistant", "colleague", "coworker", "co-worker",
    "teammate", "partner", "co-founder", "cofounder", "client", "customer", "wife", "husband", "spouse",
    "girlfriend", "boyfriend", "fiance", "fiancee", "mum", "mom", "mother", "dad", "father", "parent",
    "son", "daughter", "kid", "child", "brother", "sister", "sibling", "friend", "best friend", "roommate",
    "flatmate", "housemate", "neighbour", "neighbor", "doctor", "dentist", "therapist", "landlord", "agent",
    "accountant", "lawyer", "trainer", "coach", "nanny", "babysitter", "pet", "dog", "cat",
)  # fmt: skip
_REL = "|".join(sorted((re.escape(r) for r in RELATIONS), key=len, reverse=True))

NAME_REQUEST = re.compile(
    r"^(?:please )?(?:call me|my name(?:'s| is)|i'm called|i am called) (?P<name>[\w .'-]{1,40})$"
)
ABOUT_REQUEST = re.compile(
    r"^(?:please )?(?:remember|keep in mind|know) (?:that )?i(?:'m| am| have|'ve| work| live| like| love| was| do)?"
    r" (?P<text>.+)$"
)
PERSON_REQUEST = re.compile(
    rf"^(?:please )?(?:remember (?:that )?)?my (?:new |current )?(?P<rel>{_REL})(?:'s name)? is (?:now )?"
    rf"(?P<name>[\w .'-]{{1,40}}?)(?: now| these days| at the moment)?$"
    rf"|^(?:please )?(?:remember (?:that )?)?(?P<name2>[\w .'-]{{1,40}}) is (?:now )?my (?:new )?(?P<rel2>{_REL})"
    rf"(?: now)?$"
)
PREF_REQUEST = re.compile(
    r"^(?:please )?(?:remember (?:that )?)?i(?:'d| would)? (?P<verb>prefer|like my|rather|usually|always|normally) (?P<text>.+)$"
)
STYLE_REQUEST = re.compile(
    r"^(?:please )?(?:be |keep (?:it |your answers )?|talk to me |speak to me |answer )?"
    r"(?P<style>brief|concise|short|shorter|detailed|thorough|more detailed|casual|informal|formal|professional)"
    r"(?: with me| please| from now on)?$"
)
READ_REQUEST = re.compile(
    r"^(?:what do you know about me|what(?:'s| is) in my profile|what have you remembered about me"
    r"|what do you remember about me|tell me about myself|show my profile|read my profile)$"
)
FORGET_REQUEST = re.compile(
    r"^(?:please )?(?:forget|delete|remove) (?:that )?(?:my )?(?P<what>.+?)(?: from my profile)?$"
    r"|^(?:please )?that(?:'s| is) (?:not right|wrong|incorrect)(?:,? i(?:'m| am)? (?P<correct>.+))?$"
)
# A correction said in front of a phrase: "No, call me Sam", "Actually, I live in Pune", "Sorry, I mean ...".
# "No more interruptions" is a request, not a correction.
CORRECTION = re.compile(
    r"^(?:no(?![ ](?:more|longer)[ ])|nope|actually|sorry|oops|wait|i mean|correction)[,.!:]*\s+(?:i mean\s+)?(?P<rest>.+)$"
)
# "not Alex, Sam" / "not Alex, call me Sam" / "it's Sam" (right after a name was set or corrected)
NOT_THAT_NAME = re.compile(r"^not (?P<old>[\w'-]+),? (?:it'?s |call me |i'?m )?(?P<new>[\w .'-]{1,40})$")
ITS_NAME = re.compile(r"^(?:it'?s|i'?m|it is) (?P<new>[a-z][\w'-]{0,39})$")
NAME_QUESTION = re.compile(r"^(?:what(?:'s| is) my name|what do you call me|do you know my name|who am i)$")
# How the person wants Heliox to behave between questions: one preference, kept in the profile (QUIET_PREF).
QUIET_REQUEST = re.compile(
    r"^(?:please )?(?:stay|be|keep) quiet(?: unless i ask(?: you)?| until i ask(?: you)?(?: something)?| please| for now)?$"
    r"|^(?:please )?(?:don'?t|do not|never) (?:interrupt me|speak up|chime in|talk to me)(?: unless i ask(?: you)?)?$"
    r"|^(?:please )?(?:stop|quit) interrupting me(?: please)?$|^(?:please )?no more interruptions$"
    r"|^(?:please )?only (?:speak|talk)(?: up)? when i ask(?: you)?$"
)
SPEAK_REQUEST = re.compile(
    r"^(?:you can |feel free to |please )?(?:speak up|chime in|talk|interrupt me)(?: again| when you notice something"
    r"| whenever you like)?$|^(?:you can )?stop being quiet$|^you don'?t have to stay quiet(?: any ?more)?$"
)
# The end of the working day: "I stop work at six", "my working day ends at 18:00".
WORK_END = re.compile(
    r"\b(?:i (?:stop|finish|end|leave|knock off) work(?:ing)?|my (?:working day|work ?day) (?:ends|finishes|is over))"
    r" (?:at |by |around )?(?P<time>[\w:.]+(?: (?:am|pm|p m|a m))?)",
    re.IGNORECASE,
)
WORK_END_REQUEST = re.compile(
    r"^(?:i (?:stop|finish|end|leave|knock off) work(?:ing)?|my (?:working day|work ?day) (?:ends|finishes|is over))"
    r" (?:at |by |around )?[\w:.]+(?: (?:am|pm))?(?: (?:every day|on weekdays|most days))?$"
)
QUIET_PREF = "prefer that Heliox stays quiet unless I ask"
SPEAK_UP_PREF = "prefer that Heliox speaks up when it notices something"
_HOUR_WORDS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}  # fmt: skip


def parse_work_end(text: str) -> int | None:
    """Minutes after midnight for "I stop work at six" (18:00), "at 5:30 pm", "at 17:30"; None when unclear. A bare
    hour from 1 to 11 is the afternoon or evening: nobody ends work at six in the morning by default."""
    found = WORK_END.search(text)
    if not found:
        return None
    raw = found.group("time").casefold().replace(" ", "").replace(".", ":")
    meridiem = "pm" if raw.endswith(("pm", "p")) else "am" if raw.endswith(("am", "a")) else ""
    raw = raw.removesuffix("pm").removesuffix("am")
    hour_text, _, minute_text = raw.partition(":")
    hour = int(hour_text) if hour_text.isdigit() else _HOUR_WORDS.get(hour_text)
    minute = int(minute_text) if minute_text.isdigit() else 0
    if hour is None or not 0 <= minute <= 59:
        return None
    if meridiem == "pm" and hour < 12 or not meridiem and 1 <= hour <= 11:
        hour += 12
    return hour * 60 + minute if 0 <= hour <= 23 else None


CLEAR_REQUEST = re.compile(
    r"^(?:please )?(?:forget everything about me|clear my profile|delete my profile|forget who i am)$"
)

_STYLE_MAP = {
    "brief": ("length", "short"), "concise": ("length", "short"), "short": ("length", "short"),
    "shorter": ("length", "short"), "detailed": ("length", "detailed"), "thorough": ("length", "detailed"),
    "more detailed": ("length", "detailed"), "casual": ("formality", "casual"), "informal": ("formality", "casual"),
    "formal": ("formality", "formal"), "professional": ("formality", "formal"),
}  # fmt: skip
_LISTS = ("about", "people", "prefers")

# Things with one current value: a newer statement replaces the older one ("I live in Pune now" after "I live in
# Mumbai"). Everything else is kept side by side.
_SLOTS = (
    ("home", re.compile(r"\b(?:i live in|i'm living in|i am living in|i moved to|i've moved to|i'm based in|"
                        r"i am based in|my home is in|i'm from|i am from)\b", re.IGNORECASE)),
    ("work", re.compile(r"\b(?:i work (?:at|for|as)|i'm working (?:at|for)|i am working (?:at|for)|my job is|"
                        r"i'm employed (?:at|by))\b", re.IGNORECASE)),
    ("work_end", WORK_END),
)  # fmt: skip


def _slot(kind: str, text: str, extra: dict[str, str]) -> str:
    if kind == "people":
        from pilot.live.detectors import SINGULAR_RELATIONS

        relation = str(extra.get("relation", "")).casefold()
        return f"person:{relation}" if relation in SINGULAR_RELATIONS else ""
    if kind == "about":
        return next((name for name, pattern in _SLOTS if pattern.search(text)), "")
    return ""


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", text.casefold()).split())


def _clean(text: str, limit: int = MAX_TEXT) -> str:
    return " ".join(str(text).split())[:limit].strip()


def _titlecase_name(name: str) -> str:
    name = _clean(name, MAX_NAME)
    return " ".join(word if (word.isupper() and len(word) > 1) else word.capitalize() for word in name.split())


_LEAD = re.compile(r"^(?:please )?(?:remember|keep in mind|know)(?: that)? ", re.IGNORECASE)
_PRONOUNS = {"i", "it", "that", "this", "he", "she", "they", "we", "you", "me", "my", "the"}


def _strip_lead(raw: str) -> str:
    """The person's own words after a "remember that ..." opener, with their casing kept."""
    return _LEAD.sub("", raw.strip(), count=1).strip()


def _never(key: str, item: dict[str, Any]) -> str:
    """Why a stored entry may not be kept, or "". For a person, their name only: "Dr Mehta (dentist)" says who
    someone is to the person, which is what the people list is for; health or money about them never gets this far."""
    text = str(item.get("name", "")) if key == "people" else str(item.get("text", ""))
    return never_kept(text)


def _lapsed(item: dict[str, Any], now: float) -> bool:
    try:
        return "expires" in item and float(item["expires"]) <= now
    except (TypeError, ValueError):
        return False


def _you(text: str) -> str:
    """Turn a first-person "about" fact into second person for reading back: "I'm X" -> "you're X"."""
    for first, second in (("I'm ", "you're "), ("I've ", "you've "), ("I ", "you "), ("my ", "your ")):
        if text.startswith(first):
            return second + text[len(first) :]
    return text


class UserProfile:
    """The person's own profile in a small JSON file, mirrored into long-term memory."""

    def __init__(
        self,
        path: Path,
        *,
        remember: Callable[[str, str], Awaitable[str | None]] | None = None,
        forget: Callable[[str], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        # remember(predicate, value) mirrors an explicit-user fact and returns its id;
        # forget(fact_id) retracts it. Ids are kept in the profile's "_mirror" map.
        self._remember = remember
        self._forget = forget
        self._clock = clock  # for notes said in passing, which lapse (an "expires" field on the entry)
        self._dropped: list[str] = []  # kinds of entries the sweep removed, said once in the next "about me" answer

    async def _mirror_add(self, profile: dict[str, Any], key: str, predicate: str, value: str) -> None:
        if self._remember is None:
            return
        try:
            fact_id = await self._remember(predicate, value)
        except Exception:
            fact_id = None
        if fact_id:
            profile.setdefault("_mirror", {})[key] = fact_id

    async def _mirror_forget(self, profile: dict[str, Any], key: str) -> None:
        fact_id = (profile.get("_mirror") or {}).pop(key, None)
        if fact_id and self._forget is not None:
            try:
                await self._forget(fact_id)
            except Exception:
                pass

    # ------------------------------------------------------------------ storage

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        profile: dict[str, Any] = {"call_me": "", "style": {}, "_mirror": {}}
        if isinstance(data.get("_mirror"), dict):
            profile["_mirror"] = {str(k): str(v) for k, v in data["_mirror"].items() if isinstance(v, str)}
        if isinstance(data.get("call_me"), str):
            profile["call_me"] = data["call_me"]
        if isinstance(data.get("call_me_at"), (int, float)):
            profile["call_me_at"] = float(data["call_me_at"])
        if isinstance(data.get("style"), dict):
            profile["style"] = {
                k: v for k, v in data["style"].items() if k in ("length", "formality") and isinstance(v, str)
            }
        now = self._clock()
        for key in _LISTS:
            items = data.get(key)
            profile[key] = (
                [
                    item
                    for item in items
                    if isinstance(item, dict) and item.get("id") and not _lapsed(item, now) and not _never(key, item)
                ]
                if isinstance(items, list)
                else []
            )
        return profile

    async def sweep(self) -> int:
        """Remove what is never kept (``pilot.personal.never_kept``) from the file and from long-term memory: entries
        stored before the rule covered every route. Reading already leaves them out. Returns how many."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if not isinstance(data, dict):
            return 0
        found = [
            (item["id"], _never(key, item))
            for key in _LISTS
            for item in (data.get(key) or [])
            if isinstance(item, dict) and item.get("id") and _never(key, item)
        ]
        if not found:
            return 0
        profile = self._load()  # already without them
        profile["_mirror"] = {str(k): str(v) for k, v in (data.get("_mirror") or {}).items()}
        for entry_id, _why in found:
            await self._mirror_forget(profile, entry_id)
        self._save(profile)
        self._dropped += [NEVER_KEPT_KINDS.get(why, "something never kept") for _id, why in found]
        logger.info("Profile: removed %d stored entries Heliox never keeps (%s)", len(found), ", ".join(self._dropped))
        return len(found)

    async def prune_lapsed(self) -> int:
        """Remove notes said in passing whose time is over, here and in long-term memory. Returns how many."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if not isinstance(data, dict):
            return 0
        now = self._clock()
        lapsed = [
            item["id"]
            for key in _LISTS
            for item in (data.get(key) or [])
            if isinstance(item, dict) and item.get("id") and _lapsed(item, now)
        ]
        if not lapsed:
            return 0
        profile = self._load()  # already without the lapsed entries
        profile["_mirror"] = {str(k): str(v) for k, v in (data.get("_mirror") or {}).items()}
        for entry_id in lapsed:
            await self._mirror_forget(profile, entry_id)
        self._save(profile)
        return len(lapsed)

    def _save(self, profile: dict[str, Any]) -> None:
        note_change("remembered that")  # a voice turn that did this cannot be withdrawn
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(profile, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self) -> dict[str, Any]:
        return self._load()

    def is_empty(self) -> bool:
        profile = self._load()
        return not profile["call_me"] and not profile["style"] and not any(profile[key] for key in _LISTS)

    def name(self) -> str:
        return self._load()["call_me"]

    def style(self) -> dict[str, str]:
        return self._load()["style"]

    # ------------------------------------------------------------------ writing

    async def set_name(self, name: str) -> str:
        name = _titlecase_name(name)
        if not 1 <= len(name) <= MAX_NAME:
            raise ValueError("Tell me a name of 1 to 40 characters")
        profile = self._load()
        await self._mirror_forget(profile, "call_me")
        profile["call_me"] = name
        profile["call_me_at"] = self._clock()
        await self._mirror_add(profile, "call_me", "profile:call_me", name)
        self._save(profile)
        return name

    async def set_style(self, field: str, value: str) -> None:
        profile = self._load()
        await self._mirror_forget(profile, f"style:{field}")
        profile["style"][field] = value
        await self._mirror_add(profile, f"style:{field}", f"profile:style:{field}", value)
        self._save(profile)

    async def remove_item(self, item_id: str) -> bool:
        """Remove one row of ``list()`` by its id: "call_me", "style:<field>" or a list entry's id."""
        profile = self._load()
        if item_id == "call_me":
            if not profile["call_me"]:
                return False
            await self._mirror_forget(profile, "call_me")
            profile["call_me"] = ""
            self._save(profile)
            return True
        if item_id.startswith("style:"):
            field = item_id.split(":", 1)[1]
            if field not in profile["style"]:
                return False
            await self._mirror_forget(profile, item_id)
            del profile["style"][field]
            self._save(profile)
            return True
        return await self.delete(item_id)

    async def add(self, kind: str, text: str, **extra: str) -> dict[str, Any]:
        if kind not in _LISTS:
            raise ValueError("Unknown profile list")
        text = _clean(text)
        if not 2 <= len(text) <= MAX_TEXT:
            raise ValueError("That is too short or too long to remember")
        if looks_secret(text) or NAMES_A_SECRET.search(text) or any(looks_secret(str(v)) for v in extra.values()):
            raise ValueError("That looks like a password or key, so I won't keep it. Use your password manager.")
        why = _never(kind, {"text": text, **extra})
        if why:
            raise ValueError(f"I won't keep that, because {why}. Heliox never keeps notes like that.")
        profile = self._load()
        items = profile[kind]
        norm = _normalize(text)
        slot = _slot(kind, text, extra)
        replaced = [item for item in items if slot and item.get("slot") == slot]
        for item in replaced:
            await self._mirror_forget(profile, item["id"])  # the old value is no longer true
        items[:] = [item for item in items if _normalize(item.get("text", "")) != norm and item not in replaced]
        if len(items) >= MAX_ENTRIES:
            raise ValueError(f"I can keep up to {MAX_ENTRIES} of those; clear some first")
        # Strictly later than every entry already kept. The Windows clock moves in steps of about 15 ms, so two
        # things said within one step tied, and "forget what I just told you" could remove the earlier one.
        newest = max((float(item.get("at", 0.0)) for key in _LISTS for item in profile[key]), default=0.0)
        entry = {"id": uuid.uuid4().hex[:10], "text": text, "at": max(self._clock(), newest + 1e-6), **extra}
        if slot:
            entry["slot"] = slot
        if slot == "work_end":
            minute = parse_work_end(text)
            if minute is not None:
                entry["minute"] = minute  # read by pilot.personal.wiring.quiet_now: no offers after it
        items.append(entry)
        # One memory label per slot, or per entry: a shared label made each new entry supersede every other one.
        await self._mirror_add(profile, entry["id"], f"profile:{kind}:{slot or entry['id']}", text)
        self._save(profile)
        return entry

    async def add_person(self, name: str, relation: str) -> dict[str, Any]:
        name, relation = _titlecase_name(name), _clean(relation, MAX_NAME).casefold()
        return await self.add("people", f"{name} ({relation})", name=name, relation=relation)

    async def delete(self, entry_id: str) -> bool:
        profile = self._load()
        for key in _LISTS:
            kept = [item for item in profile[key] if item["id"] != entry_id]
            if len(kept) != len(profile[key]):
                profile[key] = kept
                await self._mirror_forget(profile, entry_id)
                self._save(profile)
                return True
        return False

    async def clear(self) -> int:
        profile = self._load()
        removed = sum(len(profile[key]) for key in _LISTS) + (1 if profile["call_me"] else 0) + len(profile["style"])
        for fact_id in list((profile.get("_mirror") or {}).values()):
            if self._forget is not None:
                try:
                    await self._forget(fact_id)
                except Exception:
                    pass
        self._save({"call_me": "", "style": {}, "_mirror": {}, **{key: [] for key in _LISTS}})
        return removed

    def quiet(self) -> bool:
        """The person asked Heliox to stay quiet unless they ask (said on its own, or while getting to know them)."""
        return any(item.get("text") == QUIET_PREF for item in self._load()["prefers"])

    def work_end_minute(self) -> int | None:
        """When the person said their working day ends, in minutes after midnight, or None."""
        for item in self._load()["about"]:
            if item.get("slot") == "work_end" and isinstance(item.get("minute"), int):
                return int(item["minute"])
        return None

    async def set_quiet(self, quiet: bool) -> None:
        """Keep one of the two preferences about speaking up, replacing the other."""
        profile = self._load()
        for item in list(profile["prefers"]):
            if item.get("text") in (QUIET_PREF, SPEAK_UP_PREF):
                await self.delete(item["id"])
        if quiet:
            await self.add("prefers", QUIET_PREF)

    def people_by_relation(self) -> dict[str, str]:
        """Relation -> name, for relations the person named exactly one person for (the live layer checks these)."""
        names: dict[str, set[str]] = {}
        for item in self._load()["people"]:
            relation, name = str(item.get("relation") or ""), str(item.get("name") or "")
            if relation and name:
                names.setdefault(relation, set()).add(name)
        return {relation: next(iter(found)) for relation, found in names.items() if len(found) == 1}

    def context_lines(self) -> list[str]:
        """Everything the person told Heliox about themselves, one line each, for every planner prompt."""
        profile = self._load()
        lines = []
        if profile["call_me"]:
            lines.append(f"Call them {profile['call_me']}.")
        lines += [f"About them: {item['text']}" for item in profile["about"]]
        lines += [f"Person they know: {item['text']}" for item in profile["people"]]
        lines += [f"Preference: {item['text']}" for item in profile["prefers"]]
        lines += [f"Answer style, {field}: {value}" for field, value in profile["style"].items()]
        return lines

    def list(self) -> list[dict[str, Any]]:
        """A flat view for Your data: one row per stored thing."""
        profile = self._load()
        rows: list[dict[str, Any]] = []
        if profile["call_me"]:
            rows.append({"id": "call_me", "kind": "name", "text": f"Call me {profile['call_me']}"})
        for field, value in profile["style"].items():
            rows.append({"id": f"style:{field}", "kind": "style", "text": f"{field}: {value}"})
        labels = {"about": "about", "people": "person", "prefers": "preference"}
        for key in _LISTS:
            for item in profile[key]:
                rows.append({"id": item["id"], "kind": labels[key], "text": item["text"]})
        return rows

    # ------------------------------------------------------------------ voice and chat

    def _dropped_sentence(self) -> str:
        """Once: what the sweep removed, by kind, never repeating the words."""
        if not self._dropped:
            return ""
        kinds = sorted(set(self._dropped))
        count = len(self._dropped)
        self._dropped = []
        things = "one thing" if count == 1 else f"{count} things"
        return (
            f"I've stopped keeping {things} you told me before, about {', '.join(kinds)}: Heliox never keeps "
            "notes like that."
        )

    def describe(self) -> str:
        dropped = self._dropped_sentence()
        if dropped:
            return f"{dropped} {self._describe()}"
        return self._describe()

    def _describe(self) -> str:
        profile = self._load()
        if self.is_empty():
            return (
                'I don\'t know anything about you yet. Tell me things like "call me Vyom", '
                '"remember that I\'m a founder" or "I prefer short answers".'
            )
        parts = []
        if profile["call_me"]:
            parts.append(f"I call you {profile['call_me']}")
        about = [_you(item["text"]) for item in profile["about"]]
        if about:
            parts.append("I know that " + "; ".join(about[:5]))
        people = [item["text"] for item in profile["people"]]
        if people:
            parts.append("people you've mentioned: " + ", ".join(people[:5]))
        prefers = [item["text"] for item in profile["prefers"]]
        if prefers:
            parts.append("you " + "; ".join(prefers[:5]))
        if profile["style"]:
            parts.append("and I keep my answers " + ", ".join(profile["style"].values()))
        return ". ".join(part[0].upper() + part[1:] for part in parts) + "."

    async def answer_async(self, text: str) -> str | None:
        """Handle a profile phrase; None when it is not one."""
        raw = text.strip()
        normalized = _normalize(text)
        try:
            await self.sweep()  # what is never kept leaves the file and long-term memory the next time it is used
        except Exception:
            logger.warning("Could not remove stored entries Heliox never keeps", exc_info=True)
        corrected = CORRECTION.match(normalized)
        if corrected:
            # "No, call me Sam": the phrase after the correction, which replaces the old fact wherever it is used.
            normalized = corrected.group("rest").strip(" .!")
            raw = re.sub(
                r"(?i)^\W*(?:no|nope|actually|sorry|oops|wait|i mean|correction)[,.!:]*\s+(?:i mean\s+)?", "", raw
            )
        if READ_REQUEST.match(normalized):
            return self.describe()
        if NAME_QUESTION.match(normalized):
            current = self.name()
            return f"I call you {current}." if current else 'You haven\'t told me your name yet. Say "call me" and it.'
        renamed = await self._rename(normalized, corrected=bool(corrected))
        if renamed:
            return renamed
        if QUIET_REQUEST.match(normalized):
            await self.set_quiet(True)
            # Under the spoken cap (pilot.conversation.layer.SPOKEN_WORD_CAP), so it is said whole.
            return 'Okay, I\'ll stay quiet unless you ask: no offers, and Live only flags risky things. "You can speak up again" undoes it.'
        if SPEAK_REQUEST.match(normalized) and self.quiet():
            await self.set_quiet(False)
            return "Okay, I'll speak up again when I notice something, and offers are back on."
        if WORK_END_REQUEST.match(normalized):
            minute = parse_work_end(normalized)
            if minute is None:
                return None
            try:
                await self.add("about", _clean(_strip_lead(raw).rstrip(".!")))
            except ValueError as exc:
                return str(exc)
            return (
                f"Noted: you stop work at {minute // 60:02d}:{minute % 60:02d}. After that I'll keep offers and "
                "suggestions to myself until the morning."
            )
        if CLEAR_REQUEST.match(normalized):
            removed = await self.clear()
            return "Done, I've forgotten what I knew about you." if removed else "I didn't have anything about you."

        name = NAME_REQUEST.match(normalized)
        if name:
            had = self.name()
            try:
                saved = await self.set_name(name.group("name"))
            except ValueError as exc:
                return str(exc)
            if corrected and had and had != saved:
                return f"Sorry about that, {saved}. I'll call you {saved} from now on."
            return f"Nice to meet you, {saved}. I'll call you that."

        person = PERSON_REQUEST.match(normalized)
        if person:
            rel = person.group("rel") or person.group("rel2")
            who = person.group("name") or person.group("name2")
            if _normalize(who) in _PRONOUNS:
                return None  # "it is my dog" is not a person worth saving
            try:
                entry = await self.add_person(who, rel)
            except ValueError as exc:
                return str(exc)
            return f"Got it, I'll remember your {entry['relation']} {entry['name']}."

        pref = PREF_REQUEST.match(normalized)
        if pref:
            # Keep the person's own wording and capitals; the normalized match only locates it.
            m = re.match(
                r"(?i)^i(?:'d| would)? (prefer|like my|rather|usually|always|normally) (.+)$", _strip_lead(raw)
            )
            verb = pref.group("verb")
            spoken = "would rather" if verb == "rather" else verb
            body = f"{m.group(1)} {m.group(2)}" if m else f"{pref.group('verb')} {pref.group('text')}"
            body = re.sub(r"(?i)^rather\b", "would rather", body)
            try:
                entry = await self.add("prefers", _clean(body))
            except ValueError as exc:
                return str(exc)
            return (
                f"Noted: you {entry['text']}."
                if entry["text"].split()[0].lower() in ("prefer", "like", "would", "usually", "always", "normally")
                else f"Noted: you {spoken} {entry['text']}."
            )

        style = STYLE_REQUEST.match(normalized)
        if style:
            field, value = _STYLE_MAP[style.group("style")]
            await self.set_style(field, value)
            if field == "length":
                return "Okay, I'll keep my answers short." if value == "short" else "Okay, I'll give more detail."
            return f"Okay, I'll keep a {value} tone."

        about = ABOUT_REQUEST.match(normalized)
        if about:
            statement = _strip_lead(raw)  # starts at "I ..." with the person's own words
            if not re.match(r"(?i)^i\b", statement):
                statement = f"I {about.group('text')}"
            try:
                entry = await self.add("about", _clean(statement))
            except ValueError as exc:
                return str(exc)
            return f'Remembered: "{entry["text"]}".'

        forget = FORGET_REQUEST.match(normalized)
        if forget and (forget.group("what") or forget.group("correct")):
            what = (forget.group("what") or "").strip()
            correct = (forget.group("correct") or "").strip()
            if what in ("everything about me", "who i am", "my profile"):
                removed = await self.clear()
                return "Done, I've forgotten what I knew about you." if removed else "I didn't have anything about you."
            removed = await self.forget_matching(what) if what else ""
            if correct:
                raw_correct = re.search(r"(?i)\bi (?:'m |am )?(.+)$", raw)
                correct = raw_correct.group(1).strip() if raw_correct else correct
                if not removed:
                    await self._forget_last()  # "that's wrong, I ..." replaces the thing just said
                entry = await self.add("about", f"I {correct}")
                return f'Fixed. I\'ll remember instead: "{entry["text"]}".'
            if removed:
                # Say exactly what went, so a wrong match is heard at once.
                return f'Okay, I\'ve forgotten "{removed}".'
            return None  # not in the profile: let other handlers or the planner take it
        return None

    async def _rename(self, normalized: str, *, corrected: bool) -> str:
        """'not Alex, Sam', and "it's Sam" right after a name was set or corrected: the name replaced everywhere."""
        current = self.name()
        if not current:
            return ""
        new = ""
        not_that = NOT_THAT_NAME.match(normalized)
        if not_that and _normalize(not_that.group("old")) == _normalize(current):
            new = not_that.group("new")
        its = ITS_NAME.match(normalized)
        recent = self._clock() - float(self._load().get("call_me_at", 0.0) or 0.0) < 600
        if its and (corrected or recent):
            new = its.group("new")
        if not new or _normalize(new) == _normalize(current):
            return ""
        try:
            saved = await self.set_name(new)
        except ValueError:
            return ""
        return f"Sorry about that, {saved}. I'll call you {saved} from now on."

    async def forget_last(self) -> str:
        """Remove the most recently added list entry (what the person just told me). Returns its text, or ""."""
        profile = self._load()
        latest = max(
            (item for key in _LISTS for item in profile[key]),
            key=lambda item: item.get("at", 0),
            default=None,
        )
        if latest is None or not await self.delete(latest["id"]):
            return ""
        return str(latest.get("text", ""))

    async def _forget_last(self) -> bool:
        return bool(await self.forget_last())

    async def forget_matching(self, what: str) -> str:
        """Remove the first entry whose words include every word of ``what``. Returns what was removed, or ""."""
        profile = self._load()
        norm = _normalize(what)
        if norm and profile["call_me"] and norm in _normalize(profile["call_me"]):
            name = profile["call_me"]
            await self._mirror_forget(profile, "call_me")
            profile["call_me"] = ""
            self._save(profile)
            return f"call me {name}"
        for key in _LISTS:
            for item in profile[key]:
                haystack = _normalize(item["text"] + " " + item.get("name", ""))
                if norm and all(word in haystack for word in norm.split()):
                    return str(item["text"]) if await self.delete(item["id"]) else ""
        return ""
