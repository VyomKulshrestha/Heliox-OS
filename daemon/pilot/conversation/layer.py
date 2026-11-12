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
import re
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
from pilot.conversation.quick import (
    QuickPlan,
    from_model,
    from_words,
    quick_acknowledgement,
    quick_result_sentence,
    valid_search_url,
)
from pilot.conversation.router import Route, TurnKind, normalize, route
from pilot.conversation.style import (
    SYSTEM_PROMPT,
    TYPED_SYSTEM_PROMPT,
    SentenceSplitter,
    TextChunker,
    claims_an_action,
    clean_for_speech,
    drop_action_claims,
    looks_like_act,
    parse_act,
    promised_action,
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
    "credit": "Your {provider} account is out of credit. Add credit there, or choose another model in Settings, "
    "under Model.",
    "slow": "The language model isn't answering right now, so I can only do what I handle myself for the moment.",
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
    "credit": "The model account is still out of credit.",
    "slow": "The language model still isn't answering.",
    "refused": "I can't help with that one.",
}
FALLBACK_AGAIN_DEFAULT = "I can't answer that without a language model."
AGAIN_ALTERNATE = "Not without the language model, sorry."
UNCLEAR_OFFLINE = "Sorry, I didn't catch that. Could you say it again?"
HOLD_AFTER_SECONDS = 2.0  # no first sentence yet: a short holding phrase, so the wait is not silent
HOLD_PHRASES = ("One moment.", "Let me think.", "Give me a second.")
LOCAL_LOADING = "Loading the local model, one moment."
SPOKEN_WORD_CAP = 30  # about 12 seconds of speech: a sentence that would go past it is not started
OFFER_SECONDS = 60.0  # how long "Do you want me to ...?" waits for a yes
YES = frozenset(
    {
        "yes", "yeah", "yep", "sure", "ok", "okay", "please", "yes please", "please do", "do it", "go ahead",
        "yes do it", "yes go ahead", "sure go ahead", "go for it", "yes thanks", "yeah please", "sure please",
    }
)  # fmt: skip
_QUESTION_WORDS = re.compile(r"^(?:how|what|why|when|where|which|who|whats|what's|hows|how's)\b")
_ASKS_TO_ACT = re.compile(r"^(?:can|could|would|will) you\b|^please\b")
LOST_MID_REPLY = "I lost the connection there."
# No text from a cloud model by then: it is treated as not answering. A local model may be loading into memory.
FIRST_TEXT_DEADLINE = 4.5  # the holding phrase at 2 s, then at most 2.5 s more of quiet
LOCAL_FIRST_TEXT_DEADLINE = 90.0
PROVIDER_NAMES = {
    "claude": "Anthropic",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "gemini": "Google",
    "meta": "Meta",
}


@dataclass
class TurnResult:
    """What the server needs back from ``take``."""

    done: bool  # the layer answered; nothing goes to the planner
    planner_input: str = ""
    intent: HandoverIntent | None = None
    target: HandoverTarget = HandoverTarget.FULL_PLANNER
    source: str = "voice"  # "voice" or "typed"
    _sink: Any = None  # the sink for this turn when it is not the layer's own (typed chat)
    quick: QuickPlan | None = None  # the one-action plan for the quick road, when there is one
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

    def execute_params(self) -> dict[str, Any]:
        """What the server adds to the planner call: the quick plan, used once and only as an object."""
        return {"_quick_plan": self.quick.plan} if self.quick is not None else {}

    def spoken_result(self, result: str, status: str = "success") -> str:
        """The quick road's result in a sentence ("" leaves the server's own wording): a state read in words, a
        done action briefly, a failure with what failed."""
        if self.quick is None:
            return ""
        return quick_result_sentence(self.quick, result, status)

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
        first_text_deadline: float = FIRST_TEXT_DEADLINE,
        prepare: Callable[[str], None] | None = None,
        search_url: Callable[[], str] = lambda: "",
        turn_ended: Callable[[], float | None] = lambda: None,
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
        self._offer: tuple[float, HandoverIntent, Any] | None = None  # (when, what, its quick plan) awaiting a yes
        self._last_fallback = ""
        self._background: set[asyncio.Task[Any]] = set()
        self._last_hold = ""
        self._hold_after = hold_after
        self._first_text_deadline = first_text_deadline
        self._said_loading = False
        # Synthesise the next sentence while this one plays (one ahead only), and when the person's last word ended.
        self._prepare = prepare
        self._search_url = search_url
        self._turn_ended = turn_ended
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

    async def take(self, text: str, *, source: str = "voice", sink: Sink | None = None) -> TurnResult | None:
        """Handle one addressed utterance that no control or local handler took. None: the layer is off."""
        if not self._enabled():
            return None
        started = self._clock()
        decided = route(text, conversation_open=self.memory.open)
        result = TurnResult(done=False, planner_input=text, route=decided, _layer=self, source=source, _sink=sink)
        result.timings["routed_ms"] = (self._clock() - started) * 1000
        try:
            ended = self._turn_ended()
        except Exception:
            ended = None
        if ended is not None:
            # On the same monotonic clock: how long recognition and dispatch took before this layer was asked.
            result.timings["last_word_to_take_ms"] = max(0.0, (started - ended) * 1000)
        self.turns = [*self.turns[-19:], result]
        logger.info("Conversation turn routed as %s (%s)", decided.kind.value, decided.reason)

        offer, self._offer = self._offer, None
        if offer is not None and normalize(text) in YES and self._clock() - offer[0] <= OFFER_SECONDS:
            # The person's yes to "Do you want me to ...?": their request, taken through the normal road with all
            # its reviews and approvals. It is never an approval of a pending plan: with one pending, the server
            # refuses a spoken yes before this layer is asked.
            self.memory.add_user(text)
            return self._hand_over(result, offer[1], quick=offer[2])
        if decided.kind is TurnKind.ACTION:
            self.memory.add_user(text)
            return self._hand_over(result, intent_from_route(decided, text))
        # LOCAL here means the server's own handlers had no answer ("why did you say that?" with no card to
        # explain): it is talk, and the fast model answers from the conversation, as for any question.
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
        typed = result.source == "typed"
        sink = result._sink or self._sink
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
        promised: list[str] = []

        async def produce() -> None:
            splitter: Any = TextChunker() if typed else SentenceSplitter()
            mode: bool | None = None  # True: an ACT line, False: words to speak, None: too early to tell
            head = ""
            count = 0
            stream = model.stream(TYPED_SYSTEM_PROMPT if typed else SYSTEM_PROMPT, messages)
            # A cloud model that has said nothing by the deadline is not answering; a local one may be loading.
            deadline = LOCAL_FIRST_TEXT_DEADLINE if getattr(model, "local", False) else self._first_text_deadline
            limit_sentences = 10_000 if typed else MAX_SENTENCES
            try:
                async with asyncio.timeout(deadline) as limit:
                    async for fragment in stream:
                        if limit.when() is not None:
                            limit.reschedule(None)  # text arrived: no deadline for the rest of the reply
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
                        if count >= limit_sentences:
                            break
                if mode is None and head:
                    if looks_like_act(head + " "):
                        act_text.append(head)
                        mode = True
                    else:
                        for sentence in splitter.feed(head):
                            await queue.put(sentence)
                if not mode:
                    for sentence in splitter.flush():
                        await queue.put(sentence)
            except FastModelUnavailable as error:
                failure.append(error.reason)
            except TimeoutError:
                failure.append("slow")
                hold = getattr(model, "hold", None)
                if callable(hold):
                    hold(30.0, "slow")
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
            capped = False
            held = False
            while True:
                if first and not held and not typed:
                    try:
                        sentence = await asyncio.wait_for(queue.get(), self._hold_after)
                    except TimeoutError:
                        held = True
                        # Said alongside the wait: when the answer is ready it follows (and outranks) the phrase.
                        hold = asyncio.ensure_future(
                            self._hold(result, started, local=bool(getattr(model, "local", False)))
                        )
                        self._background.add(hold)
                        hold.add_done_callback(self._background.discard)
                        continue
                else:
                    sentence = await queue.get()
                if sentence is None:
                    break
                # Spoken text is cleaned for the ear; typed text keeps its markdown for the chat view.
                clean = drop_action_claims(sentence) if typed else clean_for_speech(sentence, first=first)
                verb = promised_action(clean)
                if verb:
                    # "I'll open Notepad" with no ACT line would leave a promise and no action: not said here;
                    # the request is handed over below.
                    promised.append(verb)
                    continue
                if not clean or claims_an_action(clean):
                    if clean:
                        logger.info("Dropped a sentence claiming an action the fast model cannot take: %r", clean)
                    continue
                words_so_far = sum(len(part.split()) for part in said)
                if not typed and said and words_so_far + len(clean.split()) > SPOKEN_WORD_CAP:
                    # Long enough to listen to: the reply ends at the last whole sentence (never after a clause
                    # cut at a comma: those are joined to their sentence before this point).
                    capped = True
                    break
                if not typed and self._user_speaking():
                    # The person started talking while the next sentence was being written: stop before it.
                    self._remember_said(said, None)
                    result.done = True
                    return result
                if first:
                    result.timings["first_sentence_ms"] = (self._clock() - started) * 1000
                first = False
                current = clean
                if not typed:
                    self._prepare_next(queue)
                outcome = await sink(clean, "final_answer", f"conv:{id(result)}:{len(said)}")
                current = ""
                if outcome in STOPPED:
                    self._remember_said(said, clean)
                    result.done = True
                    return result
                if outcome in HEARD or outcome == "duplicate":
                    said.append(clean)
                    result.spoken.append(clean)
            if not capped:
                await producer  # (when capped, the finally below closes the stream, so it stops costing tokens)
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
                quick = from_model(act.action, words=text) if act.action is not None else None
                if self._only_asked(text):
                    # "How do I open the registry editor?" asks how, not for it to be done: offer first.
                    await self._offer_to(result, intent, quick, started)
                    return result
                return self._hand_over(result, intent, quick=quick)
            failure.append("unparsed_act")
        elif promised and not failure:
            # The model said in words that it would act but wrote no ACT line. The person is not left with a
            # promise and nothing done: what they said goes to the full planner, which reviews it as always.
            intent = HandoverIntent(verb=promised[0], object="", words=text, request=text, source="model")
            if decided.kind is TurnKind.UNCLEAR or self._only_asked(text):
                await self._offer_to(result, intent, None, started)
                return result
            if said:
                self.memory.add_heliox(" ".join(said))
            return self._hand_over(result, intent)
        if failure and not said:
            await self._fallback(result, failure[0], decided, started, provider=getattr(model, "provider", ""))
            return result
        if failure and said:
            # The reply stopped part-way: say so rather than trailing off.
            await self._say_one(result, LOST_MID_REPLY, started)
        result.done = True
        result.timings["done_ms"] = (self._clock() - started) * 1000
        return result

    def _prepare_next(self, queue: asyncio.Queue[str | None]) -> None:
        """Hand the sentence after this one to the speech engine to synthesise while this one plays."""
        if self._prepare is None:
            return
        waiting = getattr(queue, "_queue", None)  # asyncio.Queue's own deque; read only
        nxt = waiting[0] if waiting else None
        if not nxt:
            return
        clean = clean_for_speech(nxt)
        if clean and not claims_an_action(clean):
            try:
                self._prepare(clean)
            except Exception:
                logger.debug("Preparing the next sentence failed", exc_info=True)

    @staticmethod
    def _only_asked(text: str) -> bool:
        """A how/what/why question, not a request phrased as one ("could you open ...?")."""
        words = normalize(text)
        return bool(_QUESTION_WORDS.match(words)) and not _ASKS_TO_ACT.match(words)

    async def _offer_to(self, result: TurnResult, intent: HandoverIntent, quick: Any, started: float) -> None:
        """Ask "Do you want me to ...?" and keep the request for a yes in the next minute."""
        request = intent.request.strip().rstrip(".?!")
        sentence = f"Do you want me to {request[:1].lower()}{request[1:]}?" if request else "Do you want me to do that?"
        await self._say_one(result, clean_for_speech(sentence), started)
        self._offer = (self._clock(), intent, quick)
        result.done = True

    def _remember_said(self, said: list[str], cut: str | None) -> None:
        if cut:
            self.memory.add_heliox(" ".join([*said, cut]), cut_off=True)
        elif said:
            self.memory.add_heliox(" ".join(said), cut_off=True)

    async def _say_one(
        self, result: TurnResult, sentence: str, started: float, *, channel: str = "final_answer"
    ) -> str:
        result.timings.setdefault("first_sentence_ms", (self._clock() - started) * 1000)
        outcome = await (result._sink or self._sink)(sentence, channel, f"conv-one:{id(result)}")
        if outcome in HEARD or outcome in STOPPED:
            self.memory.add_heliox(sentence, cut_off=outcome in STOPPED)
            result.spoken.append(sentence)
        return outcome

    async def _fallback(
        self, result: TurnResult, reason: str, decided: Route, started: float, *, provider: str = ""
    ) -> None:
        """No model: say why in one sentence, once per session; later turns get a short line."""
        logger.info("Fast model unavailable (%s); answering from local rules", reason)
        if reason == "unparsed_act" or (decided.kind is TurnKind.UNCLEAR and reason in self._explained):
            sentence = UNCLEAR_OFFLINE
        elif reason in self._explained:
            sentence = FALLBACK_AGAIN.get(reason, FALLBACK_AGAIN_DEFAULT)
            if sentence == self._last_fallback:
                # Never the same line twice running: the short form, then the plain one, in turn.
                sentence = AGAIN_ALTERNATE if sentence != AGAIN_ALTERNATE else FALLBACK_AGAIN_DEFAULT
        else:
            sentence = FALLBACK.get(reason, FALLBACK["error"]).replace(
                "{provider}", PROVIDER_NAMES.get(provider, "model provider's")
            )
            if reason not in ("busy", "refused", "error"):
                self._explained.add(reason)
        self._last_fallback = sentence
        await self._say_one(result, sentence, started)
        result.done = True

    async def _hold(self, result: TurnResult, started: float, *, local: bool = False) -> None:
        if local and not self._said_loading:
            # A local model loads into memory on its first call after a while: say what the wait is, once.
            self._said_loading = True
            result.timings.setdefault("hold_ms", (self._clock() - started) * 1000)
            await self._sink(LOCAL_LOADING, "task_narration", f"conv-hold:{id(result)}")
            return
        await self._hold_phrase(result, started)

    async def _hold_phrase(self, result: TurnResult, started: float) -> None:
        index = (HOLD_PHRASES.index(self._last_hold) + 1) % len(HOLD_PHRASES) if self._last_hold in HOLD_PHRASES else 0
        phrase = HOLD_PHRASES[index]
        self._last_hold = phrase
        result.timings.setdefault("hold_ms", (self._clock() - started) * 1000)
        await self._sink(phrase, "task_narration", f"conv-hold:{id(result)}")

    # ── handing over ──

    def _hand_over(self, result: TurnResult, intent: HandoverIntent, *, quick: QuickPlan | None = None) -> TurnResult:
        if quick is None and intent.source == "router" and intent.verb:
            try:
                quick = from_words(
                    normalize(intent.words), intent.verb, intent.object, search_url=valid_search_url(self._search_url())
                )
            except Exception:
                logger.debug("No quick plan for %r", intent.words, exc_info=True)
                quick = None
        result.intent = intent
        result.quick = quick
        result.target = choose_target(intent, quick=quick is not None)
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

    def acknowledgement_for(self, result: TurnResult) -> str:
        """What the person hears first for a hand-over (the same sentence keep_company says)."""
        if result.intent is None:
            return ""
        ack = (quick_acknowledgement(result.quick, result.intent.words) if result.quick is not None else "") or (
            acknowledgement(result.intent, last=self._last_ack) if result.intent.verb or result.intent.say else ""
        )
        return ack or "Working on it."

    def _start_company(self, result: TurnResult) -> asyncio.Task[None]:
        assert result.intent is not None
        ack = (quick_acknowledgement(result.quick, result.intent.words) if result.quick is not None else "") or (
            acknowledgement(result.intent, last=self._last_ack) if result.intent.verb or result.intent.say else ""
        )
        if ack == self._last_ack and ack:
            ack = "Okay, " + ack[:1].lower() + ack[1:]
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
