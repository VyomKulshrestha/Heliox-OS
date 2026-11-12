"""A provisional turn: the fast model starts on the words the listener has when the person has probably stopped, while
the end of the turn and the accurate recognition pass are still being decided (conv_interfaces.md, section c).

Nothing from a provisional answer is spoken, shown, stored or acted on until the final words for the same turn arrive
and are the same words. Then the final turn takes over the running answer as if it had asked for it itself; otherwise
the call is closed and the turn starts as it always did.

Only for an utterance already known to be addressed to Heliox before its end (the wake phrase on the same proof the
final rule accepts, or the microphone button), only on the spoken fast-model road, and within limits on calls and on
money, because a dropped start still costs about what an answer costs.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.conversation")

# How the utterance was known to be addressed before its end. Nothing else ever starts a provisional call: a follow-up
# without the wake phrase, a continuation, a doubtful follow-up, a cut-off, anything the addressing rule has not taken.
ADDRESSED = frozenset({"wake", "button"})
MIN_WORDS = 3
FINAL_WITHIN_SECONDS = 8.0  # no final words by then: dropped
DROPS_PER_MINUTE = 6
DROP_USD_PER_HOUR = 0.05  # what dropped provisional calls may cost in an hour, whatever the model
DROP_SHARE = 0.10  # ... and at most this share of the session's other conversation calls
DROP_SHARE_AFTER_CALLS = 10  # once the session has made this many calls
BUDGET_NEAR = 0.90  # off for the session once the month's spend reaches this share of the monthly limit
# "auto" turns provisional answers on for a cloud model only when the measured adoption rate (the streaming words at
# the last pause equal to the final words, normalised as below) on the recorded corpus reaches AUTO_ADOPTION_RATE.
# Not measured yet: "auto" is off until the listener's measurement is in (docs/INTERACTION_MODEL.md).
MEASURED_ADOPTION_RATE: float | None = None
AUTO_ADOPTION_RATE = 0.80
TAG = "conversation.provisional"  # the budget tracker's agent id for every provisional call

# The turn id the voice dispatch running in this context carries (set by the server as the dispatch starts).
dispatch_turn: contextvars.ContextVar[str] = contextvars.ContextVar("heliox_provisional_dispatch_turn", default="")

FILLERS = frozenset({"um", "umm", "uh", "uhh", "er", "erm", "hmm", "mm", "mmm"})
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _numbers_as_digits(words: list[str]) -> list[str]:
    """ "twenty two" and "22" read the same; "one hundred and five" is 105. Anything else is left as it was."""
    out: list[str] = []
    spoken: list[bool] = []  # out[k] is a number said in words
    i = 0
    while i < len(words):
        if words[i] not in _UNITS and words[i] not in _TENS:
            out.append(words[i])
            spoken.append(False)
            i += 1
            continue
        total, current, last, j = 0, 0, "", i
        while j < len(words):
            w = words[j]
            kind = (
                ("unit" if _UNITS[w] < 10 else "teen")
                if w in _UNITS
                else "tens"
                if w in _TENS
                else "scale"
                if w in _SCALES
                else "and"
                if w == "and"
                else ""
            )
            if (
                kind == "unit"
                and last in ("", "hundred", "scale", "and", "tens")
                or (kind == "teen" and last in ("", "hundred", "scale", "and"))
            ):
                current += _UNITS[w]
            elif kind == "tens" and last in ("", "hundred", "scale", "and"):
                current += _TENS[w]
            elif kind == "scale" and w == "hundred" and last in ("unit", "teen"):
                current *= 100
                kind = "hundred"
            elif kind == "scale" and w != "hundred" and last in ("unit", "teen", "tens", "hundred"):
                total += current * _SCALES[w]
                current = 0
            elif (
                kind == "and"
                and last in ("hundred", "scale")
                and j + 1 < len(words)
                and (words[j + 1] in _UNITS or words[j + 1] in _TENS)
            ):
                pass  # "one hundred and five"
            else:
                break  # "two three" is two numbers, never five
            last = kind
            j += 1
        number = total + current
        if spoken and spoken[-1] and out[-1].isdigit() and 10 <= int(out[-1]) <= 99 and 10 <= number <= 99:
            out[-1] += str(number)  # a year said in pairs: "twenty twenty six" is 2026
        else:
            out.append(str(number))
            spoken.append(True)
        i = j
    return out


def normalise(text: str) -> str:
    """Case folded, punctuation dropped, fillers removed, numbers as digits, single spaces. Nothing looser: answering a
    question the person did not ask is worse than answering a second later."""
    text = text.casefold().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", text)  # 1,000 -> 1000
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in FILLERS]
    return " ".join(_numbers_as_digits(words))


def same_words(provisional: str, final: str) -> bool:
    return bool(normalise(final)) and normalise(provisional) == normalise(final)


def call_cost(provider: str, model: str, record: Any) -> float:
    """The budget tracker's own estimate of one call's cost (0 for a model on this computer)."""
    try:
        from pilot.models import budget_tracker as bt

        input_tokens = int(getattr(record, "input_tokens", 0) or 0)
        output_tokens = int(getattr(record, "output_tokens", 0) or 0)
        if provider == "claude":
            return float(bt.claude_cost_usd(model, input_tokens, output_tokens))
        return float(bt._estimate_cost(provider, input_tokens, output_tokens))
    except Exception:
        return 0.0


@dataclass
class Provisional:
    """One provisional answer: the model's text so far, held."""

    turn_id: str
    words: str
    addressed: str
    started: float
    model: Any
    before: list[Any]  # the conversation when it started: adopted only if it is still the same
    pieces: list[str] = field(default_factory=list)
    finished: bool = False
    error: BaseException | None = None
    adopted: bool = False
    dropped: str = ""  # why it was dropped ("" while it is not)
    messages: list[dict[str, str]] | None = None
    screen: Any = None
    first_text_at: float | None = None
    task: asyncio.Task[None] | None = None
    record: Any = None  # the model's CallRecord for this call, once known
    _changed: asyncio.Event = field(default_factory=asyncio.Event)
    _calls_before: int = 0

    def replay(self) -> AsyncIterator[str]:
        """The held text and then the rest as it comes, for the turn that adopts it. Closing it ends the call."""
        return _Replay(self)


class _Replay:
    def __init__(self, provisional: Provisional) -> None:
        self._p = provisional
        self._at = 0

    def __aiter__(self) -> _Replay:
        return self

    async def __anext__(self) -> str:
        p = self._p
        while True:
            if self._at < len(p.pieces):
                piece = p.pieces[self._at]
                self._at += 1
                return piece
            if p.finished:
                if p.error is not None:
                    raise p.error
                raise StopAsyncIteration
            p._changed.clear()
            await p._changed.wait()

    async def aclose(self) -> None:
        task = self._p.task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class ProvisionalAnswers:
    """The provisional answers of one conversation layer, and the limits on them."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._live: dict[str, Provisional] = {}
        self._drops: deque[float] = deque()  # when each dropped call was dropped (the per-minute cap)
        self._hour = -1
        self._hour_dropped_usd = 0.0
        self._hour_closed = False  # a money cap was reached in this hour: none start until the next one
        self._session_dropped_usd = 0.0
        self._dropped_records: set[int] = set()
        self._off_for_session = ""  # why provisional answers are off until the daemon restarts
        self.started = 0
        self.adopted = 0
        self.dropped = 0

    # ── the limits ──

    def _roll_hour(self) -> None:
        hour = int(self._clock() // 3600)
        if hour != self._hour:
            self._hour, self._hour_dropped_usd, self._hour_closed = hour, 0.0, False

    def refusal(self, model: Any) -> str:
        """Why a provisional call may not start now ("" when it may)."""
        if self._off_for_session:
            return self._off_for_session
        now = self._clock()
        while self._drops and now - self._drops[0] > 60.0:
            self._drops.popleft()
        if len(self._drops) >= DROPS_PER_MINUTE:
            return "drops_per_minute"
        budget = getattr(model, "budget", None)
        limit = float(getattr(budget, "_monthly_limit", 0.0) or 0.0)
        spent = float(getattr(budget, "_monthly_cost", 0.0) or 0.0)
        if limit > 0 and spent >= BUDGET_NEAR * limit:
            self._off_for_session = "budget_near"
            logger.info("Provisional answers are off for this session: the month's spend is near its limit")
            return self._off_for_session
        self._roll_hour()
        if self._hour_closed:
            return "money_this_hour"
        if self._hour_dropped_usd >= DROP_USD_PER_HOUR:
            self._hour_closed = True
            return "money_this_hour"
        calls = list(getattr(model, "calls", []) or [])
        if len(calls) >= DROP_SHARE_AFTER_CALLS:
            provider, name = str(getattr(model, "provider", "")), str(getattr(model, "name", ""))
            other = sum(call_cost(provider, name, r) for r in calls if id(r) not in self._dropped_records)
            if self._session_dropped_usd > DROP_SHARE * other:
                self._hour_closed = True
                return "money_share"
        return ""

    def _count_drop(self, provisional: Provisional) -> None:
        """A dropped call: counted for the per-minute cap now, and its cost once the call has ended."""
        self.dropped += 1
        self._drops.append(self._clock())

        def costed(_task: Any = None) -> None:
            record = provisional.record or _find_record(provisional)
            if record is None:
                return
            self._dropped_records.add(id(record))
            cost = call_cost(str(getattr(provisional.model, "provider", "")), str(provisional.model.name), record)
            self._roll_hour()
            self._hour_dropped_usd += cost
            self._session_dropped_usd += cost
            if self._hour_dropped_usd >= DROP_USD_PER_HOUR:
                self._hour_closed = True

        task = provisional.task
        if task is not None and not task.done():
            task.add_done_callback(costed)
        else:
            costed()

    # ── one provisional answer ──

    def get(self, turn_id: str) -> Provisional | None:
        return self._live.get(turn_id)

    def add(self, provisional: Provisional) -> None:
        self.drop(provisional.turn_id, "replaced")
        self._live[provisional.turn_id] = provisional
        self.started += 1

    def drop(self, turn_id: str, reason: str) -> bool:
        provisional = self._live.pop(turn_id, None)
        if provisional is None or provisional.adopted:
            return False
        provisional.dropped = reason or "dropped"
        task = provisional.task
        if task is not None and not task.done():
            task.cancel()  # closes the stream: the reply stops costing output tokens
        self._count_drop(provisional)
        logger.info("Provisional answer for %s dropped (%s)", turn_id, provisional.dropped)
        return True

    def expire(self, provisional: Provisional) -> None:
        """No final words within FINAL_WITHIN_SECONDS of the start: dropped (a later start for the same turn is not)."""
        if self._live.get(provisional.turn_id) is provisional:
            self.drop(provisional.turn_id, "no_final_words")

    def drop_others(self, turn_id: str) -> None:
        for other in [key for key in self._live if key != turn_id]:
            self.drop(other, "another_turn")

    def claim(self, turn_id: str) -> Provisional | None:
        """The provisional answer for this turn, taken out of the registry (adopted or dropped by the caller)."""
        return self._live.pop(turn_id, None)

    def finish(self, turn_id: str) -> None:
        """The dispatch for this turn has ended: anything still held for it was not used."""
        self.drop(turn_id, "not_used")

    def note_adopted(self, provisional: Provisional) -> None:
        provisional.adopted = True
        self.adopted += 1

    def not_adopted(self, provisional: Provisional, reason: str) -> None:
        """Claimed by its turn, which could not use it (other words, another road)."""
        if provisional.adopted:
            return
        self._live[provisional.turn_id] = provisional
        self.drop(provisional.turn_id, reason)


def _find_record(provisional: Provisional) -> Any:
    calls = list(getattr(provisional.model, "calls", []) or [])
    for record in calls[provisional._calls_before :]:
        if getattr(record, "started", 0.0) >= provisional.started - 1e-6:
            provisional.record = record
            return record
    return None


async def pump(provisional: Provisional, system: str, build: Callable[[], Any]) -> None:
    """Ask the model and hold what it says. ``build`` gives (messages, screen) as the final turn would build them."""
    from pilot.models.budget_tracker import current_agent_id

    current_agent_id.set(TAG)  # this task's own context: every provisional call is counted on its own
    p = provisional
    try:
        p.messages, p.screen = await build()
        p._calls_before = len(getattr(p.model, "calls", []) or [])
        stream = p.model.stream(system, p.messages)
        try:
            async for piece in stream:
                if p.first_text_at is None:
                    p.first_text_at = time.monotonic()
                p.pieces.append(piece)
                p._changed.set()
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
    except asyncio.CancelledError:
        p.error = asyncio.CancelledError()
        raise
    except BaseException as error:  # FastModelUnavailable and the rest: the adopting turn handles it as its own
        p.error = error
    finally:
        p.finished = True
        _find_record(p)
        p._changed.set()
