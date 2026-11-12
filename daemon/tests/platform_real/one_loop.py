"""One event loop for a whole check script, so subprocesses outlive no loop.

``asyncio.run`` closes its loop at the end of each call. A program the check launched, or a subprocess still
being reaped, then has its transport collected after that loop is closed, and Python prints "Event loop is
closed" from the transport's ``__del__`` (CI on 008f1a0, in the openbox and macOS OCR steps). The daemon runs one
loop for its whole life, so this is a fault of the checks only. They call ``run(coroutine)`` instead, on one loop
that stays open until the interpreter exits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop.run_until_complete(coroutine)
