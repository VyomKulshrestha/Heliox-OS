"""A bounded wait that never loses a cancellation: Heliox's replacement for asyncio.wait_for.

Before Python 3.12, asyncio.wait_for runs what it waits for as a task of its own, and a cancellation that arrives as
that task finishes can be lost: the caller that was told to stop carries on. On Windows CI with Python 3.11 that left
a stopped task waiting on an approval for 300 s (54ab435). Heliox installs on Python 3.11 to 3.13, so this is a
person's case, not only a test's.

``bounded`` is wait_for as Python 3.12 and later implement it, on every supported version: the awaitable is awaited
in the caller's own task under ``asyncio.timeout``, so a cancellation is raised exactly where it arrives. What it
keeps from wait_for:

- on the timeout, what is awaited is cancelled (a coroutine where it stands; a task or future, including one from
  asyncio.shield or asyncio.wrap_future, through the caller's cancellation, as wait_for did) and TimeoutError is
  raised (asyncio.TimeoutError is the same class);
- ``timeout=None`` waits without a bound;
- ``timeout`` of zero or less: an awaitable that is already done gives its result; otherwise it is cancelled, waited
  for, and TimeoutError is raised (unless it finished with a result meanwhile), as in Python 3.12's wait_for.

One difference from 3.11's wait_for, which is 3.12's too: a coroutine runs in the caller's task and context, so a
context variable it sets is seen by the caller afterwards.

The tree has no asyncio.wait_for left; tests/test_no_wait_for.py keeps it that way.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def bounded(awaitable: Awaitable[T], timeout: float | None) -> T:
    """Await ``awaitable`` for at most ``timeout`` seconds (None: no bound); TimeoutError when it runs out."""
    if timeout is not None and timeout <= 0:
        future = asyncio.ensure_future(awaitable)
        if future.done():
            return future.result()
        future.cancel()
        await asyncio.wait({future})  # its clean-up runs before the caller hears of the timeout
        try:
            return future.result()
        except asyncio.CancelledError as cancelled:
            raise TimeoutError from cancelled
    async with asyncio.timeout(timeout):
        return await awaitable
