"""Where a spoken turn's time goes: a list of (name, monotonic time) stamps, filled as the turn goes.

A voice dispatch starts a trace (``begin``); the server and the conversation layer ``mark`` the points below as the
turn reaches them, and the layer keeps the list on its TurnResult (``result.trace``) and logs it at debug as times
only. Never words: the names are fixed, and ``data`` carries numbers, flags and handler names.

The points, in order (agreed with the conversation bench, benchmarks/conversation_bench/voice_latency.py):

    words received            _voice_command_dispatch starts
    settled                   the waits for a continuation are over (waited=seconds)
    code handlers done        the handlers Heliox answers in code were all asked (handlers_ms, slowest, slowest_ms)
    route decided             the conversation layer has routed the turn (route=answer|action|control|...)
    context ready             the screen was read and the messages built (screen_ms)
    request sent              the model call is made (warm=True when a connection was likely open)
    model first text          the first streamed piece arrived
    first sentence complete   the splitter gave the first whole sentence
    sentence to voice         a sentence is handed to the voice (hold=True for the holding phrase)

``listener``, None by default, is called synchronously at each point as ``listener(name, at=<monotonic>, **data)``;
the bench sets it to its own log. Nothing here changes what a turn does.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger("pilot.conversation.trace")

Trace = list[tuple[str, float]]

listener: Callable[..., Any] | None = None
_current: ContextVar[Trace | None] = ContextVar("conversation_trace", default=None)


def begin() -> Trace:
    """A new trace for the turn this task is running (and the tasks it starts)."""
    found: Trace = []
    _current.set(found)
    return found


def current() -> Trace | None:
    return _current.get()


def mark(name: str, **data: Any) -> None:
    """Stamp ``name`` now on the current turn's trace, and tell the listener. Never raises."""
    at = time.monotonic()
    found = _current.get()
    if found is not None:
        found.append((name, at))
    hook = listener
    if hook is not None:
        try:
            hook(name, at=at, **data)
        except Exception:
            logger.debug("The trace listener failed", exc_info=True)


def mark_once(name: str, **data: Any) -> None:
    """``mark``, unless the current trace already has ``name``."""
    found = _current.get()
    if found is not None and any(stamp == name for stamp, _at in found):
        return
    mark(name, **data)


def summary(found: Trace | None) -> str:
    """The stamps as milliseconds after the first one: "words received +0, settled +12, ...". Times only."""
    if not found:
        return ""
    start = found[0][1]
    return ", ".join(f"{name} +{(at - start) * 1000:.0f}" for name, at in found)
