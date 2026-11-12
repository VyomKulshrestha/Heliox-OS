"""One counter for "which came first" between Live, the conversation floor and offers.

Comparing clock stamps to order events fails on Windows, where ``time.time()`` and ``time.monotonic()`` move in
steps of about 15 ms: two events in one step get the same stamp and the tie picks the wrong one. Anything that must
know which of two things happened later takes a number from here instead. Clocks stay for durations.
"""

from __future__ import annotations

import itertools

_counter = itertools.count(1)


def next_seq() -> int:
    """A number larger than every one handed out before in this process."""
    return next(_counter)


__all__ = ["next_seq"]
