"""The spoken turn: route it, answer it in a streamed voice, or hand it over and keep the person company.

States of one turn::

    routed --(conversation / unclear)--> answering --> speaking 1..n --> done
       |                                     |  \\--(ACT line)--> handed_over
       |                                     \\--(model unavailable)--> fallback sentence --> done
       |--(action)--> handed_over --> (planner runs, the layer speaks ack, progress, approval) --> finished
       \\--(control word that no server handler took)--> "Okay." --> done
    any state --(the person talks, or a new utterance arrives)--> cut

Cut means: generation stops (the stream is closed, usage up to that point is recorded), queued sentences are
dropped, and the conversation remembers only the sentences that were spoken, the interrupted one marked as cut.

The layer serves typed chat later through the same ``take`` with a sink that sends sentences to the chat view
instead of the speaker; nothing in it is voice-specific except the channel names it passes to the sink.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pilot.conversation.context import ContextSources, ConversationMemory, build_messages
from pilot.conversation.fast_model import FastModel, FastModelUnavailable
from pilot.conversation.handover import (
    HandoverIntent,
    HandoverTarget,
    acknowledgement,
    choose_target,
    intent_from_route,
    keep_company,
)
from pilot.conversation.router import Route, TurnKind, route
from pilot.conversation.style import (
    SYSTEM_PROMPT,
    SentenceSplitter,
    claims_an_action,
    clean_for_speech,
    looks_like_act,
    parse_act,
)

logger = logging.getLogger("pilot.conversation")

# A sink says one sentence and reports what happened: "spoken", "interrupted" (talked over), "cancelled",
# "superseded", "duplicate", "suppressed" (in a call, focus) or "error".
Sink = Callable[[str, str, str], Awaitable[str]]
HEARD = frozenset({"spoken"})
STOPPED = frozenset({"interrupted", "cancelled", "superseded"})
MAX_SENTENCES = 5

SETUP_HINT = (
    "In Settings, under Model, you can run a local model with Ollama, which keeps everything on this computer, "
    "or add a cloud key."
)
FALLBACK = {
    "offline": "I can't reach the language model right now, so I can only do what I handle myself, like "
    "reminders, to-dos, timers and undo.",
    "no_key": "There's no working key for the language model, so I can only do what I handle myself. You can "
    "add one in Settings, under Model.",
    "budget": "This month's model budget is used up, so I can only do what I handle myself until you raise it "
    "in Settings.",
    "busy": "The language model is busy right now. Try me again in a moment.",
    "refused": "I can't help with that one.",
    "error": "Something went wrong reaching the language model. I can still do what I handle myself.",
    "none": "Talking things through needs a language model, and none is set up. " + SETUP_HINT,
    "local_down": "Talking things through needs a language model, and the local one isn't running. "
    "Start Ollama, or choose another model in Settings, under Model.",
    "local_missing": "The local model named in Settings isn't installed. Pull it in Ollama, or choose another "
    "one in Settings, under Model.",
}
# Said instead once the reason has been explained in this session: no lecture on every turn.
FALLBACK_AGAIN = {
    "offline": "Still offline, so I can't answer that one.",
    "busy": "The model is still busy. Try again in a moment.",
    "refused": "I can't help with that one.",
}
FALLBACK_AGAIN_DEFAULT = "I can't answer that without a language model."
UNCLEAR_OFFLINE = "Sorry, I didn't catch that. Could you say it again?"
HOLD_AFTER_SECONDS = 2.0  # no first sentence yet: a short holding phrase, so the wait is not silent
HOLD_PHRASES = ("One moment.", "Let me think.", "Give me a second.")


@dataclass
class TurnResult:
    """What the server needs back from ``take``."""

    done: bool  # the layer answered; nothing goes to the planner
    planner_input: str = ""
    intent: HandoverIntent | None = None
    target: HandoverTarget = HandoverTarget.FULL_PLANNER
    route: Route | None = None
    timings: dict[str, float] = field(default_factory=dict)
    spoken: list[str] = field(default_factory=list)
    _layer: Any = None
    _company: asyncio.Task[None] | None = None

    def keep_company(self) -> asyncio.Task[None]:
        """Start the acknowledgement and progress speech; the server cancels it when the planner returns."""
        assert self._layer is not None and self.intent is not None
        self._company = self._layer._start_company(self)
        return self._company

    def finish(self, status: str, result: str) -> None:
        """The planner's outcome, for the conversation: the next question can refer to it."""
        if self._layer is not None:
            self._layer._finished(self, status, result)


class ConversationLayer:
    def __init__(
        self,
        *,
        sink: Sink,
        model: Callable[[], FastModel | None],
        sources: ContextSources | None = None,
        enabled: Callable[[], bool] = lambda: True,
        user_speaking: Callable[[], bool] = lambda: False,
        task_status: Callable[[], dict[str, Any]] = dict,
        clock: Callable[[], float] = time.monotonic,
        context_chars: int = 1400,
        hold_after: float = HOLD_AFTER_SECONDS,
    ) -> None:
        self._sink = sink
        self._model = model
        self.sources = sources or ContextSources()
        self._enabled = enabled
        self._user_speaking = user_speaking
        self._task_status = task_status
        self._clock = clock
        self._context_chars = context_chars
        self.memory = ConversationMemory(clock=clock)
        self._active: asyncio.Task[Any] | None = None
        self._explained: set[str] = set()
        self._background: set[asyncio.Task[Any]] = set()
        self._last_hold = ""
        self._hold_after = hold_after
        self._last_ack = ""
        self.last_result = ""
        self.turns: list[TurnResult] = []  # recent turns, for measurement and the debug log

    # ── the server's hooks ──

    def interrupt(self) -> bool:
        """A new utterance arrived or the person talked over Heliox: stop generating and speaking now."""
        task = self._active
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def prewarm(self) -> None:
        """The person started talking to Heliox: have the fast model ready (built, its SDK imported) before the turn
        ends. Cheap after the first call."""
        if not self._enabled():
            return
        try:
            model = self._model()
        except Exception:
            logger.debug("Could not prepare the fast model", exc_info=True)
            return
        warm = getattr(model, "warm", None)
        if callable(warm):
            try:
                task = asyncio.get_running_loop().create_task(warm())
            except RuntimeError:
                return
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    def clear(self) -> None:
        """The "new conversation" call: earlier turns no longer frame the next question."""
        self.interrupt()
        self.memory.clear()
        self.last_result = ""

    async def take(self, text: str, *, source: str = "voice") -> TurnResult | None:
        """Handle one addressed utterance that no control or local handler took. None: the layer is off."""
        if not self._enabled():
            return None
        started = self._clock()
        decided = route(text, conversation_open=self.memory.open)
        result = TurnResult(done=False, planner_input=text, route=decided, _layer=self)
        result.timings["routed_ms"] = (self._clock() - started) * 1000
        self.turns = [*self.turns[-19:], result]
        logger.info("Conversation turn routed as %s (%s)", decided.kind.value, decided.reason)

        if decided.kind in (TurnKind.ACTION, TurnKind.LOCAL):
            # LOCAL here means the server's own handlers declined it; the planner gets it as before.
            intent = (
                intent_from_route(decided, text)
                if decided.kind is TurnKind.ACTION
                else HandoverIntent(verb="", object="", words=text, request=text, source="router")
            )
            self.memory.add_user(text)
            return self._hand_over(result, intent)
        if decided.kind is TurnKind.CONTROL:
            self.memory.add_user(text)
            await self._say_one(result, "Okay.", started)
            result.done = True
            return result

        task = asyncio.ensure_future(self._answer(text, decided, result, started))
        self._active = task
        try:
            return await task
        except asyncio.CancelledError:
            if task.cancelled() and not _current_task_cancelling():
                # Cut off by the next utterance: the turn is over, nothing goes to the planner.
                result.done = True
                return result
            raise
        finally:
            if self._active is task:
                self._active = None

    # ── answering in words ──

    async def _answer(self, text: str, decided: Route, result: TurnResult, started: float) -> TurnResult:
        model = None
        try:
            model = self._model()
        except Exception:
            logger.warning("The fast model could not be built", exc_info=True)
        if model is None:
            self.memory.add_user(text)
            await self._fallback(result, "none", decided, started)
            return result
        screen = None
        if self.sources.screen_async is not None:
            try:
                screen = await asyncio.wait_for(self.sources.screen_async(bool(getattr(model, "local", False))), 0.5)
            except Exception:
                screen = None  # not available here, or too slow to wait for
        else:
            screen = self.sources.screen()
        messages = build_messages(text, self.sources, self.memory, budget=self._context_chars, screen=screen)
        self.memory.add_user(text)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        act_text: list[str] = []
        failure: list[str] = []

        async def produce() -> None:
            splitter = SentenceSplitter()
            mode: bool | None = None  # True: an ACT line, False: words to speak, None: too early to tell
            head = ""
            count = 0
            stream = model.stream(SYSTEM_PROMPT, messages)
            try:
                async for fragment in stream:
                    if "first_token_ms" not in result.timings:
                        result.timings["first_token_ms"] = (self._clock() - started) * 1000
                    if mode is None:
                        head += fragment
                        mode = looks_like_act(head)
                        if mode is None:
                            continue
                        fragment, head = head, ""
                    if mode:
                        act_text.append(fragment)
                        continue
                    for sentence in splitter.feed(fragment):
                        count += 1
                        await queue.put(sentence)
                    if count >= MAX_SENTENCES:
                        break
                if mode is None and head:
                    if looks_like_act(head + " "):
                        act_text.append(head)
                        mode = True
                    else:
                        splitter.feed(head)
                if not mode:
                    for sentence in splitter.flush():
                        await queue.put(sentence)
            except FastModelUnavailable as error:
                failure.append(error.reason)
            finally:
                # Closing the stream ends the request now (a cut-off reply stops costing tokens) and records usage.
                with contextlib.suppress(Exception):
                    await stream.aclose()
                await queue.put(None)

        producer = asyncio.ensure_future(produce())
        said: list[str] = []
        current = ""
        try:
            first = True
            held = False
            while True:
                if first and not held:
                    try:
                        sentence = await asyncio.wait_for(queue.get(), self._hold_after)
                    except TimeoutError:
                        held = True
                        # Said alongside the wait: when the answer is ready it follows (and outranks) the phrase.
                        hold = asyncio.ensure_future(self._hold(result, started))
                        self._background.add(hold)
                        hold.add_done_callback(self._background.discard)
                        continue
                else:
                    sentence = await queue.get()
                if sentence is None:
                    break
                clean = clean_for_speech(sentence, first=first)
                if not clean or claims_an_action(clean):
                    if clean:
                        logger.info("Dropped a sentence claiming an action the fast model cannot take: %r", clean)
                    continue
                if self._user_speaking():
                    # The person started talking while the next sentence was being written: stop before it.
                    self._remember_said(said, None)
                    result.done = True
                    return result
                if first:
                    result.timings["first_sentence_ms"] = (self._clock() - started) * 1000
                first = False
                current = clean
                outcome = await self._sink(clean, "final_answer", f"conv:{id(result)}:{len(said)}")
                current = ""
                if outcome in STOPPED:
                    self._remember_said(said, clean)
                    result.done = True
                    return result
                if outcome in HEARD or outcome == "duplicate":
                    said.append(clean)
                    result.spoken.append(clean)
            await producer
        except asyncio.CancelledError:
            # A new utterance arrived: what was said is remembered, the sentence being spoken as cut off.
            self._remember_said(said, current or None)
            result.done = True
            raise
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer
        if said:
            self.memory.add_heliox(" ".join(said))

        if act_text:
            act = parse_act("".join(act_text))
            if act is not None:
                intent = HandoverIntent(
                    verb=act.verb, object=act.object, words=text, request=act.request, source="model", say=act.say
                )
                return self._hand_over(result, intent)
            failure.append("unparsed_act")
        if failure and not said:
            await self._fallback(result, failure[0], decided, started)
            return result
        result.done = True
        result.timings["done_ms"] = (self._clock() - started) * 1000
        return result

    def _remember_said(self, said: list[str], cut: str | None) -> None:
        if cut:
            self.memory.add_heliox(" ".join([*said, cut]), cut_off=True)
        elif said:
            self.memory.add_heliox(" ".join(said), cut_off=True)

    async def _say_one(
        self, result: TurnResult, sentence: str, started: float, *, channel: str = "final_answer"
    ) -> str:
        result.timings.setdefault("first_sentence_ms", (self._clock() - started) * 1000)
        outcome = await self._sink(sentence, channel, f"conv-one:{id(result)}")
        if outcome in HEARD or outcome in STOPPED:
            self.memory.add_heliox(sentence, cut_off=outcome in STOPPED)
            result.spoken.append(sentence)
        return outcome

    async def _fallback(self, result: TurnResult, reason: str, decided: Route, started: float) -> None:
        """No model: say why in one sentence, once per session; later turns get a short line."""
        logger.info("Fast model unavailable (%s); answering from local rules", reason)
        if reason == "unparsed_act" or (decided.kind is TurnKind.UNCLEAR and reason in self._explained):
            sentence = UNCLEAR_OFFLINE
        elif reason in self._explained:
            sentence = FALLBACK_AGAIN.get(reason, FALLBACK_AGAIN_DEFAULT)
        else:
            sentence = FALLBACK.get(reason, FALLBACK["error"])
            if reason not in ("busy", "refused", "error"):
                self._explained.add(reason)
        await self._say_one(result, sentence, started)
        result.done = True

    async def _hold(self, result: TurnResult, started: float) -> None:
        index = (HOLD_PHRASES.index(self._last_hold) + 1) % len(HOLD_PHRASES) if self._last_hold in HOLD_PHRASES else 0
        phrase = HOLD_PHRASES[index]
        self._last_hold = phrase
        result.timings.setdefault("hold_ms", (self._clock() - started) * 1000)
        await self._sink(phrase, "task_narration", f"conv-hold:{id(result)}")

    # ── handing over ──

    def _hand_over(self, result: TurnResult, intent: HandoverIntent) -> TurnResult:
        result.intent = intent
        result.target = choose_target(intent)
        result.planner_input = intent.request or intent.words
        result.done = False
        logger.info(
            "Handing over to the %s: verb=%r object=%r (%s)",
            result.target.value,
            intent.verb,
            intent.object,
            intent.source,
        )
        return result

    def _start_company(self, result: TurnResult) -> asyncio.Task[None]:
        assert result.intent is not None
        ack = acknowledgement(result.intent, last=self._last_ack) if result.intent.verb or result.intent.say else ""
        if not ack:
            ack = "Working on it."
        self._last_ack = ack

        def said(sentence: str) -> None:
            result.spoken.append(sentence)

        return asyncio.ensure_future(
            keep_company(acknowledgement_text=ack, speak=self._sink, status=self._task_status, on_said=said)
        )

    def _finished(self, result: TurnResult, status: str, text: str) -> None:
        spoken = " ".join(result.spoken)
        summary = " ".join(str(text).split())[:240]
        self.last_result = f"{status}: {summary}" if summary else status
        self.memory.add_heliox(f"{spoken} {summary}".strip())


def _current_task_cancelling() -> bool:
    """Whether the caller itself is being cancelled (as opposed to the inner answer task)."""
    task = asyncio.current_task()
    if task is None:
        return False
    cancelling = getattr(task, "cancelling", None)  # 3.11+
    return bool(cancelling()) if callable(cancelling) else False
