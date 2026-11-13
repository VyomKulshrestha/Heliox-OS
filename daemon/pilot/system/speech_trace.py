"""Stamps that show where a spoken turn's time goes, for the Conversation Bench and anyone else who asks.

The listener and the voice emit them; nothing here changes what either does. ``sink`` is a callable(event, data) set
by whoever wants the stamps (benchmarks.conversation_bench sets it); None (the default) makes every emit a no-op. It
is called from the event loop and from worker threads (never from PortAudio's audio callback), so it must be
thread-safe and return at once. Times in ``data`` are time.monotonic() seconds.

Events:
    "candidate end"            {at, captured_at, audio_s}: 200 ms (or 100 ms) of quiet; the early final pass may start
    "candidate cancel"         {at}: the person carried on after a candidate end
    "speculation held back"    {at, recognisers}: no early pass, another recogniser was running
    "final pass started"       {id, speculative, at, audio_s}
    "final pass done"          {id, speculative, started, done, audio_s, failed}
    "speculative pass kept"    {id, at}: the turn ended at that pass's candidate; its words are the turn's
    "speculative pass dropped" {id, at, running}: the person went on (or the listener stopped)
    "piece synthesized"        {words, kind, requested, worker_started, done}: the voice worker made a piece
    "piece trimmed"            {words, done}
    "piece ready"              {index, words, source, at}: source "made", "ahead" or "store"
    "piece plays"              {index, words, at}: handed to playback
    "output line open"         {started, opened, latency_setting, latency_s, rate, device}
    "playback start"           {at, rate, seconds}
    "first sample"             {handed, first_block_at, dac_delay_s}: the first block of a piece went to the device;
                               dac_delay_s is PortAudio's own estimate of when it reaches the speaker (None if unknown)
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

sink: Callable[[str, dict[str, Any]], None] | None = None
_ids = itertools.count(1)


def emit(event: str, **data: Any) -> None:
    callback = sink
    if callback is None:
        return
    try:
        callback(event, data)
    except Exception:
        logger.debug("Speech trace sink failed for %s", event, exc_info=True)


def next_id() -> int:
    return next(_ids)


def now() -> float:
    return time.monotonic()
