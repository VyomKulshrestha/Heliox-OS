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
    "opening read started"     {at, waited_s, audio_s}: Whisper reads the opening for the wake phrase (waited_s: how
                               long it waited for another recogniser to finish)
    "opening read done"        {at, confirms, reread}: confirms: the phrase with a word after it; reread: read twice
    "wake check started"       {at, beside}: Whisper on the whole utterance; beside: Parakeet had not returned yet
    "early request started"    {turn, at, n, words}: the talk model asked on a pass's words before the turn settled
                               (n: the how-manyth of this utterance)
    "early request first text" {turn, at}
    "early request kept"       {turn, at, started, first_text_at}: the settled words were its words (0: no text yet)
    "early request dropped"    {turn, at, reason}: carried_on, words_changed, not_addressed, other_words, ...
    "turn end reported"        {turn, at}: the listener told the server the turn ended on addressed words
    "hum decided"              {turn, at, path, hum, reason}: path "turn end" or "hand-over"; hum "" when none
    "hum to the voice"         {turn, at}: a hum decided at the turn end handed to the voice
    "speak started"            {entered, rate_checked, config_read, words}: voice.speak's work before the voice is
                               chosen: the spoken text and the cognitive-load rate check, then reading the settings
    "piece synthesized"        {words, kind, requested, worker_started, done, took, process_ready, text_written,
                               in_memory, sent_at, worker_got, worker_began, worker_synth_s, worker_trim_s,
                               worker_total_s, worker_sent, answer_line_at, answer_read_at, resumed_at,
                               payload_bytes}: the voice worker made a piece. kind: "reply", "next" (the next sentence,
                               made while one plays) or "ahead" (for the store); worker_started: when this request got
                               the pipe. took: the worker's own seconds per step (read_s, synth_s, trim_s, total_s;
                               write_s and timing_s for a file); in_memory: the samples came back on the pipe (False: a
                               file the daemon reads and trims). The rest, plain numbers for a sink that keeps only
                               numbers: request written (sent_at), read by the worker (worker_got), taken by its loop
                               (worker_began), its synthesis and trim seconds, its answer written (worker_sent), the
                               answer line and the samples read here, the event loop back on it (resumed_at); None
                               when a worker does not say
    "piece trimmed"            {words, done}
    "piece ready"              {index, words, source, at}: source "made", "ahead" or "store"
    "piece plays"              {index, words, at}: handed to playback
    "piece read"               {asked, done, in_memory}: the player has the samples (read from a file when not)
    "stream first chunk"       {id, at}: a streamed piece's first chunk arrived from the worker (reader thread)
    "stream started"           {words, requested, worker_started, first_chunk}: the stream is handed to playback
    "stream end"               {at, id, chunks, cut, audio_s, made_s, first_s, error?}: the worker's end frame
    "stream played"            {at, underruns, samples, position, cut}: a streamed piece left the line; underruns:
                               how often the next chunk was late and the line played silence meanwhile
    "output line open"         {started, opened, latency_setting, latency_s, rate, device}
    "playback start"           {at, rate, seconds, streamed?}: seconds is None for a streamed piece
    "first sample"             {handed, first_block_at, dac_delay_s, fitted, attached}: the first block of a piece
                               went to the device; dac_delay_s is PortAudio's own estimate of when it reaches the
                               speaker (None if unknown); fitted: brought to the line's rate; attached: on the line
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
