"""Who may speak now: the person, the conversation Heliox is having with them, or Heliox Live.

Heliox has two voices that start on their own initiative: the conversation (an answer to what the person just said,
streamed sentence by sentence, or the planner's acknowledgement, progress and result) and Live (a slip, a risk or an
offer it noticed). Before this module each spoke when its own checks passed. Live looked only at whether audio was
playing at that instant, so it could start in the second between two streamed sentences, or while the model was
still writing the first one, and its correction channel outranks an answer, so a Live remark could cut a reply off.

The rule, in the order it is applied:

1. **The person first.** While the person is talking, nobody starts. (Live's policy and the listener already wait.)
2. **A conversation turn holds the floor** from the moment an addressed utterance reaches dispatch until Heliox has
   finished answering it (the streamed reply spoken, or the planner's result spoken), and for ``GRACE_SECONDS``
   after, so the person can come back to it. During that time Live does not start speaking. Its card still appears
   at once (an urgent risk is never held back), and a non-urgent catch keeps waiting and may be said afterwards.
3. **One exception: a catch about the words that opened the turn** ("remind me on Tuesday the twelfth" when the
   twelfth is a Thursday) is the first thing a person sitting next to you would say. It may be spoken, but only
   before the turn has spoken anything. It goes first and the reply waits behind it in the speech queue. Once the
   turn has started speaking, it becomes a card.
4. **The reply is never cut by Live.** Rules 2 and 3 mean Live never asks to speak while a reply is being said. A
   reply that starts while a Live remark is playing waits for it (the speech queue plays the remark to the end).

Nothing here touches audio; it only answers "may Live start speaking now?". Pure and clock-injected, so the rule is
tested the same way on every system.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

GRACE_SECONDS = 3.0  # after Heliox has answered, the person gets this long to come back before Live speaks up


@dataclass
class _Turn:
    opened_at: float
    spoke: bool = False


class Floor:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, grace: float = GRACE_SECONDS) -> None:
        self._clock = clock
        self._grace = grace
        self._turns: dict[int, _Turn] = {}
        self._next = 0
        self._closed_at = float("-inf")

    # ── the conversation's side ──

    def open_turn(self) -> int:
        """An addressed utterance reached dispatch. Returns a token for ``close_turn``."""
        self._next += 1
        self._turns[self._next] = _Turn(self._clock())
        return self._next

    def close_turn(self, token: int) -> None:
        if self._turns.pop(token, None) is not None and not self._turns:
            self._closed_at = self._clock()

    def heliox_spoke(self) -> None:
        """Heliox's conversation voice (not Live) asked to speak: open turns have now started their reply."""
        for turn in self._turns.values():
            turn.spoke = True

    # ── Live's side ──

    @property
    def turn_open(self) -> bool:
        return bool(self._turns)

    def opened_at(self) -> float | None:
        """When the earliest open turn began, or None."""
        return min((turn.opened_at for turn in self._turns.values()), default=None)

    def held(self) -> bool:
        """A turn is open, or one closed less than the grace period ago."""
        return self.turn_open or self._clock() - self._closed_at < self._grace

    def live_may_speak(self, *, about_this_turn: bool = False) -> bool:
        """Whether Live may start speaking now (rules 2 and 3). A card is always allowed."""
        if not self.turn_open:
            return self._clock() - self._closed_at >= self._grace
        if not about_this_turn:
            return False
        return not any(turn.spoke for turn in self._turns.values())

    def about_this_turn(self, noted_at: float) -> bool:
        """A catch noted at ``noted_at`` is about the words that opened the current turn."""
        opened = self.opened_at()
        return opened is not None and noted_at >= opened

    def status(self) -> dict[str, object]:
        return {
            "turn_open": self.turn_open,
            "replying": any(turn.spoke for turn in self._turns.values()),
            "held": self.held(),
        }


__all__ = ["GRACE_SECONDS", "Floor"]
