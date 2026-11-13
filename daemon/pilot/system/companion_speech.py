"""One priority-controlled audio channel for every Heliox speech producer."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable


class SpeechChannel(StrEnum):
    EMERGENCY_STOP = "emergency_stop"
    USER_SPEECH = "user_speech"
    APPROVAL_RISK = "approval_risk"
    LIVE_CORRECTION = "live_correction"
    TASK_FAILURE = "task_failure"
    FINAL_ANSWER = "final_answer"
    TASK_NARRATION = "task_narration"
    PROACTIVE_SUGGESTION = "proactive_suggestion"
    BACKGROUND_INSIGHT = "background_insight"


SPEECH_PRIORITY: dict[SpeechChannel, int] = {
    SpeechChannel.EMERGENCY_STOP: 700,
    SpeechChannel.USER_SPEECH: 600,
    SpeechChannel.APPROVAL_RISK: 500,
    SpeechChannel.LIVE_CORRECTION: 450,
    SpeechChannel.TASK_FAILURE: 400,
    SpeechChannel.FINAL_ANSWER: 350,
    SpeechChannel.TASK_NARRATION: 300,
    SpeechChannel.PROACTIVE_SUGGESTION: 200,
    SpeechChannel.BACKGROUND_INSIGHT: 100,
}


@dataclass(frozen=True, slots=True)
class SpeechOutcome:
    status: str
    channel: SpeechChannel
    message: str
    # When the speech was cut off (interrupted, stopped or superseded) and the local voice spoke it: the words the
    # person heard (whole words that had finished playing HEARD_MARGIN_SECONDS before the voice stopped, the start
    # of the text as given), the seconds played, and the seconds of audio made so far. None when unknown: a system
    # voice (SAPI, say, espeak-ng) is a process killed at the cut and reports nothing.
    heard_text: str | None = None
    played_seconds: float | None = None
    duration_seconds: float | None = None
    # Which voice spoke it ("local", "store" or "system"), by which branch of the voice rule, and for the system voice
    # why (pilot.system.local_tts_worker.record_reply_voice). None when no voice was chosen for it. Data only.
    voice: str | None = None
    voice_branch: str | None = None
    voice_reason: str | None = None


def heard_part(text: str, spoken_words: int, spoken_total: int) -> str:
    """The start of ``text`` covering ``spoken_words`` of the ``spoken_total`` words the voice was given (the voice
    gets the text after pronunciation fixes, which can change the word count; then in proportion, rounded down)."""
    words = text.split()
    if spoken_total <= 0 or spoken_words <= 0:
        return ""
    count = spoken_words if len(words) == spoken_total else (spoken_words * len(words)) // spoken_total
    return " ".join(words[: min(count, len(words))])


@dataclass(slots=True)
class _SpeechRequest:
    request_id: int
    channel: SpeechChannel
    priority: int
    text: str
    key: str
    recorder: Any
    future: asyncio.Future[SpeechOutcome]


Speaker = Callable[[str, Any], Awaitable[bool]]


class CompanionSpeechCoordinator:
    """Serialize speech, preempt by priority, and suppress duplicate output."""

    def __init__(
        self,
        *,
        speaker: Speaker | None = None,
        dedupe_window_seconds: float = 2.5,
        max_queue: int = 16,
    ) -> None:
        self._speaker = speaker or self._default_speaker
        self._dedupe_window_seconds = max(0.0, dedupe_window_seconds)
        self._max_queue = max(1, max_queue)
        self._queue: asyncio.PriorityQueue[tuple[int, int, _SpeechRequest]] = asyncio.PriorityQueue()
        self._lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._current: _SpeechRequest | None = None
        self._current_speech: asyncio.Task[bool] | None = None
        self._cancel_reasons: dict[int, str] = {}
        self._pending_keys: set[str] = set()
        self._recent: dict[str, float] = {}
        self._sequence = 0
        self._preemptions = 0
        self._duplicates = 0
        self._spoken = 0
        self._closing = False

    async def speak(
        self,
        text: str,
        *,
        channel: SpeechChannel | str = SpeechChannel.FINAL_ANSWER,
        dedupe_key: str = "",
        recorder: Any = None,
    ) -> SpeechOutcome:
        clean_text = " ".join(str(text).split())
        resolved_channel = SpeechChannel(channel)
        if not clean_text:
            return SpeechOutcome("error", resolved_channel, "Speech text is empty")
        now = time.monotonic()
        key = dedupe_key.strip() or (f"{resolved_channel.value}:{clean_text.casefold()}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SpeechOutcome] = loop.create_future()

        async with self._lock:
            self._prune_recent(now)
            if key in self._pending_keys or key in self._recent:
                self._duplicates += 1
                return SpeechOutcome(
                    "duplicate",
                    resolved_channel,
                    "Duplicate speech suppressed",
                )
            self._recent[key] = now
            self._pending_keys.add(key)
            self._sequence += 1
            request = _SpeechRequest(
                request_id=self._sequence,
                channel=resolved_channel,
                priority=SPEECH_PRIORITY[resolved_channel],
                text=clean_text,
                key=key,
                recorder=recorder,
                future=future,
            )
            if (
                self._current is not None
                and request.priority > self._current.priority
                and self._current_speech is not None
                and not self._current_speech.done()
            ):
                self._cancel_reasons[self._current.request_id] = "superseded"
                self._current_speech.cancel()
                self._preemptions += 1
            if self._queue.qsize() >= self._max_queue:
                self._pending_keys.discard(key)
                return SpeechOutcome(
                    "dropped",
                    resolved_channel,
                    "Speech queue is full",
                )
            await self._queue.put((-request.priority, request.request_id, request))
            self._ensure_worker()

        try:
            return await future
        except asyncio.CancelledError:
            # The caller no longer wants this said (a quick answer replaced "Opening it now."): drop it from the
            # queue, or stop it if it is already playing, so it is not heard after the answer.
            future.cancel()
            if self._current is request and self._current_speech is not None and not self._current_speech.done():
                self._cancel_reasons[request.request_id] = "cancelled"
                self._current_speech.cancel()
            raise

    async def stop_all(self) -> int:
        """Cancel current speech and resolve every queued caller."""
        cancelled = 0
        async with self._lock:
            if self._current is not None and self._current_speech is not None and not self._current_speech.done():
                self._cancel_reasons[self._current.request_id] = "cancelled"
                self._current_speech.cancel()
                cancelled += 1
            while not self._queue.empty():
                _priority, _sequence, request = self._queue.get_nowait()
                self._pending_keys.discard(request.key)
                if not request.future.done():
                    request.future.set_result(
                        SpeechOutcome(
                            "cancelled",
                            request.channel,
                            "Speech stopped",
                        )
                    )
                self._queue.task_done()
                cancelled += 1
        return cancelled

    def status(self) -> dict[str, Any]:
        return {
            "active": self._current is not None,
            "active_channel": (self._current.channel.value if self._current is not None else ""),
            "queued": self._queue.qsize(),
            "spoken": self._spoken,
            "preemptions": self._preemptions,
            "duplicates_suppressed": self._duplicates,
        }

    async def close(self) -> None:
        self._closing = True
        await self.stop_all()
        worker = self._worker
        if worker is None:
            return
        if self._current is None:
            worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._closing = False
            self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                _priority, _sequence, request = await self._queue.get()
            except asyncio.CancelledError:
                raise
            if request.future.done():  # withdrawn by its caller while it waited
                self._pending_keys.discard(request.key)
                self._queue.task_done()
                async with self._lock:
                    if self._closing or self._queue.empty():
                        self._worker = None
                        return
                continue
            self._current = request
            sequence_before = self._local_sequence()
            reply_before = self._reply_sequence()
            self._current_speech = asyncio.create_task(self._speaker(request.text, request.recorder))
            outcome = SpeechOutcome(
                "cancelled",
                request.channel,
                "Speech coordinator stopped",
            )
            try:
                interrupted = await self._current_speech
                status = "interrupted" if interrupted else "spoken"
                if not interrupted:
                    self._spoken += 1
                outcome = SpeechOutcome(
                    status,
                    request.channel,
                    ("Speech interrupted by user" if interrupted else "Speech completed"),
                )
            except asyncio.CancelledError:
                status = self._cancel_reasons.pop(
                    request.request_id,
                    "cancelled",
                )
                outcome = SpeechOutcome(
                    status,
                    request.channel,
                    "Speech superseded" if status == "superseded" else "Speech stopped",
                )
            except Exception as error:
                from pilot.system.voice import SpeechUnheard

                # "no_voice": the voice failed or there is none, so nothing was heard. Not "spoken": no follow-up
                # window, no Live reply window, and the reply is shown as text (PilotServer._speak_companion_text).
                outcome = SpeechOutcome(
                    "no_voice" if isinstance(error, SpeechUnheard) else "error",
                    request.channel,
                    str(error),
                )
            finally:
                if outcome.status in {"interrupted", "stopped", "superseded", "cancelled"}:
                    outcome = self._with_heard(outcome, request.text, sequence_before)
                outcome = self._with_voice(outcome, reply_before)
                self._pending_keys.discard(request.key)
                if not request.future.done():
                    request.future.set_result(outcome)
                self._queue.task_done()
                self._current = None
                self._current_speech = None
            async with self._lock:
                if self._closing or self._queue.empty():
                    self._worker = None
                    return

    @staticmethod
    def _local_sequence() -> int:
        from pilot.system.local_tts_worker import manager

        progress = manager().last_progress
        return progress.sequence if progress is not None else 0

    @staticmethod
    def _reply_sequence() -> int:
        try:
            from pilot.system.local_tts_worker import reply_sequence

            return reply_sequence()
        except Exception:
            return 0

    @staticmethod
    def _with_voice(outcome: SpeechOutcome, reply_before: int) -> SpeechOutcome:
        """``outcome`` with the voice that spoke it, when one was chosen for this request (the last recorded after it
        was taken)."""
        from dataclasses import replace

        from pilot.system.local_tts_worker import manager

        try:
            record = getattr(manager(), "last_reply", None)
        except Exception:
            return outcome  # data only: never in the way of the outcome
        if record is None or record.sequence <= reply_before:
            return outcome
        return replace(outcome, voice=record.voice, voice_branch=record.branch, voice_reason=record.reason or None)

    def _with_heard(self, outcome: SpeechOutcome, text: str, sequence_before: int) -> SpeechOutcome:
        """``outcome`` with what the person heard, when the local voice spoke this request (a speak() began after
        it was taken); unchanged when a system voice spoke it."""
        from dataclasses import replace

        from pilot.system.local_tts_worker import manager

        progress = manager().last_progress
        if progress is None or progress.sequence <= sequence_before:
            return outcome
        spoken_total = sum(len(piece.split()) for piece in progress.pieces)
        played, total = progress.played()
        heard = heard_part(text, progress.heard_words(), spoken_total)
        self._last_heard = (text, heard)
        return replace(outcome, heard_text=heard, played_seconds=round(played, 2), duration_seconds=round(total, 2))

    def heard_of(self, text: str) -> str | None:
        """What the person heard of ``text`` when it was last cut off, or None when unknown (a system voice spoke
        it, or the last cut was another sentence)."""
        last = getattr(self, "_last_heard", None)
        if last is None or " ".join(last[0].split()) != " ".join((text or "").split()):
            return None
        return last[1]

    def _prune_recent(self, now: float) -> None:
        for key, observed_at in tuple(self._recent.items()):
            if now - observed_at > self._dedupe_window_seconds:
                self._recent.pop(key, None)

    @staticmethod
    async def _default_speaker(text: str, recorder: Any) -> bool:
        from pilot.system.voice import speak, speak_interruptible

        if recorder is not None:
            return await speak_interruptible(text, recorder=recorder)
        from pilot.system.voice import _raise_if_unheard

        _raise_if_unheard(await speak(text))
        return False
