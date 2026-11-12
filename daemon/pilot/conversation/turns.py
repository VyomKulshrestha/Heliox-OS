"""A sentence split at a pause: withdrawing the first half's dispatch (conv_interfaces.md, section b).

The listener ends a turn at a pause ("Hey Heliox, email the invoice" ... "to Priya") and dispatches the first half;
when it hears the rest and decides it was one sentence, it asks the server to withdraw the first dispatch and take
the joined sentence instead. The listener decides when a continuation is one; this module decides what can still be
withdrawn, from what the server recorded about the turn, never from its wording:

* nothing sent to the executor yet (an answer being written or spoken, a planner call, an approval waiting, a quick
  plan not yet run): withdrawn; the reply stops, the call is cancelled, a waiting approval is denied quietly, and
  the joined sentence is dispatched in its place. If it leads to the same action, the acknowledgement is not said
  again.
* an action sent to the executor, or anything a handler that answers in code recorded as changed
  (pilot.turn_changes: a reminder, a note, a setting, a habit, the clipboard, a focus session ...): too
  late; Heliox says what was done, and the listener takes the second half as a new turn.
* no such turn, or it began more than WITHDRAW_SECONDS ago: unknown; the second half is a new turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pilot.system.voice import voice_turn_addressed, voice_turn_id  # set per dispatch by the listener
from pilot.turn_changes import turn_changes

WITHDRAW_SECONDS = 6.0
KEEP_TURNS = 20
_POLITE = {"please", "thanks", "now", "then", "for", "me", "thank", "you"}


_DONE = {
    "open_application": "opened {target}",
    "close_application": "closed {target}",
    "volume_mute": "muted the sound",
    "volume_set": "set the volume",
    "keyboard_press": "pressed {target}",
    "file_delete": "deleted {target}",
    "file_move": "moved {target}",
    "file_write": "saved {target}",
    "send_email": "sent the email",
}


@dataclass(frozen=True)
class Withdrawal:
    status: str  # "withdrawn" | "too_late" | "unknown"
    said: str = ""  # too_late: what Heliox says about what already ran
    ran: tuple[str, ...] = ()


@dataclass
class TurnState:
    turn_id: str
    task: asyncio.Task[Any] | None
    started: float
    approvals_before: frozenset[str] = frozenset()
    ran: list[tuple[str, str]] = field(default_factory=list)  # (action type, target) sent to the executor
    kept: list[str] = field(default_factory=list)  # what code-answered handlers changed ("set the reminder")
    signature: str = ""  # what the turn leads to: the quick plan's action and target, or the hand-over's verb/object
    acknowledged: bool = False  # an acknowledgement for this sentence was started ("I'm on it"), here or earlier
    addressed: str = ""  # how the first half was addressed; the joined sentence keeps it
    verb: str = ""  # the request's verb ("open"), so a too-late "and Spotify" can be read as "open Spotify"


def signature_of(turn: Any) -> str:
    """What a conversation turn leads to, for "is the joined sentence the same action?"."""
    quick = getattr(turn, "quick", None)
    plan = getattr(quick, "plan", None)
    actions = getattr(plan, "actions", None) or []
    if actions:
        action = actions[0]
        kind = getattr(getattr(action, "action_type", None), "value", "")
        return f"quick:{kind}:{str(getattr(action, 'target', '') or '').casefold()}"
    intent = getattr(turn, "intent", None)
    if intent is not None:
        words = re.sub(r"[^\w\s]", " ", str(intent.object).casefold()).split()
        while words and words[-1] in _POLITE:
            words.pop()  # "the screenshots please" is the same request as "the screenshots"
        return f"intent:{intent.verb}:{' '.join(words)}"
    return ""


def said_about(ran: list[tuple[str, str]], kept: list[str] | None = None) -> str:
    """ "I've already opened Notepad." from what was sent to the executor and what code-answered handlers changed."""
    parts = list((kept or [])[:2])
    for kind, target in ran[:2]:
        phrase = _DONE.get(kind)
        if phrase is None:
            continue
        name = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or "it"
        parts.append(phrase.format(target=name.title() if kind.endswith("application") else name))
    if not parts:
        return "I've already done the first part."
    return "I've already " + " and ".join(parts) + "."


def carried_verb(rest: str, verb: str) -> str:
    """ "and Spotify" after a too-late "open Notepad" -> "open Spotify"; anything else is left as said."""
    words = rest.strip().split()
    if not verb or len(words) < 2 or words[0].casefold() not in {"and", "also", "plus"}:
        return rest
    return f"{verb} {' '.join(words[1:])}"


class TurnLog:
    """The server's recent voice turns, by id, with what each one has done so far."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._turns: OrderedDict[str, TurnState] = OrderedDict()
        self.carried: tuple[str, float] = ("", float("-inf"))  # the verb a too-late first half leaves to the rest

    def begin(self, approvals: Any = ()) -> TurnState:
        turn_id = voice_turn_id.get() or uuid.uuid4().hex[:12]
        voice_turn_id.set(turn_id)
        state = TurnState(turn_id, asyncio.current_task(), self._clock(), frozenset(approvals))
        state.addressed = voice_turn_addressed.get()
        turn_changes.set(state.kept)  # every change this turn's code makes is recorded here (note_change)
        self._turns[turn_id] = state
        while len(self._turns) > KEEP_TURNS:
            self._turns.popitem(last=False)
        return state

    def current(self) -> TurnState | None:
        return self._turns.get(voice_turn_id.get())

    def note_executing(self, plan: Any) -> None:
        """The executor is about to run ``plan`` for the current voice turn: from now on it cannot be withdrawn."""
        state = self.current()
        if state is None:
            return
        for action in getattr(plan, "actions", None) or []:
            kind = getattr(getattr(action, "action_type", None), "value", "")
            state.ran.append((kind, str(getattr(action, "target", "") or "")))

    def rest_of_split(self, text: str) -> str:
        """The second half the listener dispatches after "too_late", with the first half's verb when it needs it."""
        verb, at = self.carried
        self.carried = ("", float("-inf"))
        if self._clock() - at > WITHDRAW_SECONDS:
            return text
        return carried_verb(text, verb)

    def get(self, turn_id: str, gap: float | None = None) -> TurnState | None:
        """The recent turn, or None. ``gap``: seconds on the listener's capture clock from the end of the first half
        to the start of the words that continue it; when given, the limit is counted on it, so a slow recogniser on
        a busy machine does not turn one sentence into two actions. Without it, the time since the dispatch."""
        state = self._turns.get(turn_id)
        if state is None:
            return None
        waited = gap if gap is not None else self._clock() - state.started
        return None if waited > WITHDRAW_SECONDS else state

    async def withdraw(
        self,
        turn_id: str,
        *,
        deny_quietly: Callable[[str], Any],
        pending: Callable[[], Any],
        gap: float | None = None,
    ) -> Withdrawal:
        """Stop the turn if nothing irreversible has run. The caller dispatches the joined sentence after
        "withdrawn", and the second half alone after "too_late" or "unknown"."""
        state = self.get(turn_id, gap)
        if state is None:
            return Withdrawal("unknown")
        if state.ran or state.kept:
            self.carried = (state.verb, self._clock())
            return Withdrawal("too_late", said_about(state.ran, state.kept), tuple(kind for kind, _ in state.ran))
        task = state.task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(task), 2.0)
        for plan_id in set(pending()) - state.approvals_before:
            with contextlib.suppress(Exception):
                await deny_quietly(plan_id)
        self._turns.pop(turn_id, None)
        return Withdrawal("withdrawn")
