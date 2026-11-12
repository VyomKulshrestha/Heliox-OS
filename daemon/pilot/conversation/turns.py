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

from pilot.system.voice import (  # set per dispatch by the listener
    CONTINUATION_SECONDS,
    voice_turn_addressed,
    voice_turn_id,
)
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
    result: Any = None  # the conversation layer's TurnResult for this turn, when the layer took it
    withdrawn: bool = False  # replaced by the joined sentence: nothing more is said for it
    addressed: str = ""  # how the first half was addressed; the joined sentence keeps it
    verb: str = ""  # the request's verb ("open"), so a too-late "and Spotify" can be read as "open Spotify"
    # What a withdrawn first half must not leave behind: the conversation and what Heliox last said as they were
    # when it began, the request it showed on screen and its interaction, and the words Live kept from it.
    memory_before: Any = None
    spoken_before: tuple[Any, ...] | None = None
    shown: str = ""
    interaction_id: str = ""
    live_text: str = ""


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


def said_about(ran: list[tuple[str, str]], kept: list[str] | None = None, *, undo: bool = False) -> str:
    """ "I've already opened Notepad." from what was sent to the executor and what code-answered handlers changed."""
    parts = list((kept or [])[:2])
    for kind, target in ran[:2]:
        phrase = _DONE.get(kind)
        if phrase is None:
            continue
        name = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or "it"
        parts.append(phrase.format(target=name.title() if kind.endswith("application") else name))
    said = "I've already done the first part." if not parts else "I've already " + " and ".join(parts) + "."
    if undo and any(kind in _undoable_kinds() for kind, _target in ran):
        said += " Say undo to take it back."  # the undo journal kept what it needs to put it back
    return said


def _undoable_kinds() -> frozenset[str]:
    try:
        from pilot.workflows.undo_journal import UNDOABLE_TYPES

        return frozenset(kind.value for kind in UNDOABLE_TYPES)
    except Exception:
        return frozenset()


def carried_verb(rest: str, verb: str) -> str:
    """ "and Spotify" after a too-late "open Notepad" -> "open Spotify"; anything else is left as said."""
    words = rest.strip().split()
    if not verb or len(words) < 2 or words[0].casefold() not in {"and", "also", "plus"}:
        return rest
    return f"{verb} {' '.join(words[1:])}"


SETTLE_POLL_SECONDS = 0.05
SETTLE_BOUND_SECONDS = 10.0  # the longest a first half waits for the person to finish what they are saying


def changes_something(actions: list[Any]) -> bool:
    """Any action outside READ_ONLY_ACTIONS: opening and reading things change nothing."""
    from pilot.actions import READ_ONLY_ACTIONS

    return any(getattr(action, "action_type", None) not in READ_ONLY_ACTIONS for action in actions)


def must_wait_out_window(actions: list[Any]) -> bool:
    """A plan the risk classification calls destructive or not undoable: an action that would need an approval if
    the learner had not waived it (``requires_confirmation``), or one no snapshot can reverse (``is_irreversible``).
    Such a plan waits out the continuation window even in silence; any other change runs at once."""
    for action in actions:
        try:
            if action.requires_confirmation or action.is_irreversible:
                return True
        except Exception:
            return True  # not classifiable: treated as the careful case
    return False


def continuation_possible(listener: Any, turn_id: str, *, window: bool) -> bool:
    """The person may still be going on with turn ``turn_id`` (read from the listener, never from the words).

    True while the listener has decided that what followed continues it (the withdrawal is on its way), or while it
    is hearing or recognising speech that began within CONTINUATION_SECONDS of that turn's end. With ``window``, also
    while that window is open and nothing has begun yet. Only the listener's last turn can be continued."""
    if listener is None or not turn_id:
        return False
    pending = getattr(listener, "_continuation_of", None)
    if isinstance(pending, tuple) and len(pending) > 1 and pending[1] == turn_id:
        return True
    last = getattr(listener, "_last_turn", None)
    if not isinstance(last, tuple) or len(last) < 3 or last[2] != turn_id or not isinstance(last[0], float):
        return False
    ended = last[0]
    recorder = getattr(listener, "_recorder", None)
    started = getattr(recorder, "speech_started_at", None)
    hearing = bool(getattr(listener, "user_is_speaking", False) or getattr(listener, "_busy", False))
    if hearing and isinstance(started, float) and 0.0 <= started - ended <= CONTINUATION_SECONDS:
        return True
    if not window:
        return False
    now = getattr(recorder, "_frame_captured_at", None)  # the capture clock, which the turn's end is counted on
    if not isinstance(now, float):
        now = time.monotonic()
    return now - ended <= CONTINUATION_SECONDS


async def settle(listener: Any, turn_id: str, *, window: bool) -> float:
    """Wait while the person may be going on with this turn; returns the seconds waited. A continuation withdraws
    the turn, which cancels this wait, so what follows it never runs. Bounded by SETTLE_BOUND_SECONDS."""
    started = time.monotonic()
    while continuation_possible(listener, turn_id, window=window):
        if time.monotonic() - started >= SETTLE_BOUND_SECONDS:
            break
        await asyncio.sleep(SETTLE_POLL_SECONDS)
    return time.monotonic() - started


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
        undo: bool = False,
    ) -> Withdrawal:
        """Stop the turn if nothing irreversible has run. The caller dispatches the joined sentence after
        "withdrawn", and the second half alone after "too_late" or "unknown"."""
        state = self.get(turn_id, gap)
        if state is None:
            return Withdrawal("unknown")
        if state.ran or state.kept:
            self.carried = (state.verb, self._clock())
            said = said_about(state.ran, state.kept, undo=undo)
            return Withdrawal("too_late", said, tuple(kind for kind, _ in state.ran))
        # Marked before the cancellation lands: a planner call that ends in between says nothing ("All set."), and
        # its company says no more lines.
        state.withdrawn = True
        if state.result is not None:
            state.result.withdrawn = True
        task = state.task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            # Wait a moment for it to end, without wait_for (which can lose a cancellation before Python 3.12) and
            # without being cancelled ourselves through it.
            await asyncio.wait({task}, timeout=2.0)
        for plan_id in set(pending()) - state.approvals_before:
            with contextlib.suppress(Exception):
                await deny_quietly(plan_id)
        self._turns.pop(turn_id, None)
        return Withdrawal("withdrawn")
