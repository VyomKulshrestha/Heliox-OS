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

from pilot.conversation import provisional as prov
from pilot.conversation.action_talk import needs_approval_first, result_sentence, said_again
from pilot.conversation.context import SAVE_OR_CHANGE, ContextSources, ConversationMemory, build_messages
from pilot.conversation.fast_model import FastModel, FastModelUnavailable
from pilot.conversation.handover import (
    HandoverIntent,
    HandoverTarget,
    acknowledgement,
    choose_target,
    intent_from_route,
    keep_company,
)
from pilot.conversation.planner_context import HandoverContext, build_context
from pilot.conversation.promises import act_start_held, declined, hedges, late_act_at, promised_action
from pilot.conversation.quick import (
    QuickPlan,
    from_model,
    from_words,
    quick_acknowledgement,
    quick_result_sentence,
    valid_search_url,
)
from pilot.conversation.router import Route, TurnKind, misheard_command, normalize, route
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
)

logger = logging.getLogger("pilot.conversation")

# A sink says one sentence and reports what happened: "spoken", "interrupted" (talked over), "cancelled",
# "superseded", "duplicate", "suppressed" (in a call, focus) or "error".
Sink = Callable[[str, str, str], Awaitable[str]]
HEARD = frozenset({"spoken"})
STOPPED = frozenset({"interrupted", "cancelled", "superseded"})
MAX_SENTENCES = 5

SETUP_HINT = "You can add a local model or a cloud key in Settings, under Model."
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
    "slow": "The language model is slow right now, so I stopped waiting. Ask me again.",
    "refused": "I can't help with that one.",
    "error": "Something went wrong reaching the language model. I can still do what I handle myself.",
    "not_api": "I'm getting a web page instead of the language model. The network may want you to sign in first.",
    "empty": "The language model sent back nothing that time. Ask me again?",
    "none": "That needs a language model, and none is set up. " + SETUP_HINT,  # under 25 words, said once
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
    "slow": "The language model is still slow. Ask me again in a moment.",
    "refused": "I can't help with that one.",
}
FALLBACK_AGAIN_DEFAULT = "I can't answer that without a language model."
AGAIN_ALTERNATE = "Not without the language model, sorry."
# A model that is only slow is there: its second plain line does not say it is missing.
AGAIN_ALTERNATES = {"slow": "Still slow, sorry. Try again in a few seconds."}
UNCLEAR_OFFLINE = "Sorry, I didn't catch that. Could you say it again?"
HOLD_AFTER_SECONDS = 2.0  # no first sentence yet: a short holding phrase, so the wait is not silent
HOLD_PHRASES = ("One moment.", "Let me think.", "Give me a second.")
LOCAL_LOADING = "Loading the local model, one moment."
SPOKEN_WORD_CAP = (
    25  # about 10 seconds of speech: a sentence that would go past it is not started (the prompt asks for 20)
)
# The spoken cap when the person set an answer length (the context tells the model the same).
PERSONAL_WORD_CAPS = {"short": 15, "detailed": 60}
OFFER_SECONDS = 60.0  # how long "Do you want me to ...?" waits for a yes
YES = frozenset(
    {
        "yes", "yeah", "yep", "sure", "ok", "okay", "please", "yes please", "please do", "do it", "go ahead",
        "yes do it", "yes go ahead", "sure go ahead", "go for it", "yes thanks", "yeah please", "sure please",
    }
)  # fmt: skip
_QUESTION_WORDS = re.compile(r"^(?:how|what|why|when|where|which|who|whats|what's|hows|how's)\b")
_ASKS_TO_ACT = re.compile(r"^(?:can|could|would|will) you\b|^please\b")
DOUBTFUL = "doubtful_follow_up"
# Whether Heliox kept or changed something is never the model's to say (three paid runs: it claimed saves that
# did not happen and denied one that did). A request to keep something or behave differently that no handler
# took gets one fixed, true sentence, by kind; "did you get that?" right after is answered from the record.
NOT_KEPT = (
    (
        re.compile(r"\b(?:call me|my name is|my name's|i'm called)\b", re.I),
        'I didn\'t change your name. Say "call me" and then the name.',
    ),
    (
        re.compile(r"\b(?:quiet|interrupt|interrupting|speak up|suggest|suggesting|offers?)\b", re.I),
        'I didn\'t change when I speak up. Say "stay quiet unless I ask", or "you can speak up again".',
    ),
    (
        re.compile(r"\b(?:short|shorter|brief|detail|detailed|longer)\b", re.I),
        'I didn\'t change how long my answers are. Say "keep it short" or "be detailed".',
    ),
    (
        re.compile(r"\b(?:remember|keep in mind|note|forget)\b", re.I),
        'I didn\'t keep that. Say "remember that" and then the fact.',
    ),
)
NOT_CHANGED = 'I didn\'t change anything. Say "remember that" and the fact to keep it, or use Settings.'
DID_YOU = re.compile(
    r"^(?:so |and |ok |okay |wait |hey heliox,? )?(?:did you (?:get|save|keep|store|catch|change|do|note) "
    r"(?:that|it|this|my name)(?: right| down)?|what did you (?:just )?do|did that work|is that (?:saved|done|kept)"
    r"|have you (?:saved|got|kept) (?:that|it))$"
)
DONE_SECONDS = 60.0  # how long "did you get that?" is about the turn before
# Words that start a stop, a denial or an undo (the listener's own rule): never answered early.
_CONTROL_START = re.compile(
    r"^(?:stop|wait|hold on|no|nope|cancel|never ?mind|undo|skip|don'?t|do not|not (?:that|now)|actually|instead|"
    r"go ahead|continue|that'?s (?:it|enough|all)|enough)\b"
)
# "Thanks" on its own (after normalize): the whole sentence, so "thanks, now open Notepad" is never taken here.
THANKS = re.compile(
    r"^(?:(?:ok(?:ay)?|great|perfect|cool|nice|lovely|brilliant|awesome) )?"
    r"(?:thanks|thank you|thank you so much|thanks a lot|thanks so much|many thanks|cheers|ta)"
    r"(?: (?:heliox|very much|again|mate))?$"
)


def ends_with_question(sentence: str) -> bool:
    return sentence.rstrip().rstrip("\"')]").endswith("?")


# The rest of a reply the person talked over, asked for: said from what was written, for two minutes.
WHAT_WERE_YOU_SAYING = re.compile(
    r"^(?:sorry,? )?(?:what were you saying|you were saying|go on|carry on|keep going|continue|"
    r"finish what you were saying|what were you going to say|sorry,? go on|go ahead,? finish)$"
)
UNSAID_SECONDS = 120.0
# "Are you there?": the person checks Heliox is listening.
ARE_YOU_THERE = re.compile(
    r"^(?:are you (?:still )?there|you there|are you (?:still )?listening|can you hear me|are you awake|"
    r"hello,? are you there|you still there)$"
)
HERE = "I'm here."
# The person says Heliox's last reply was wrong.
CORRECTED = re.compile(
    r"^(?:no,? )?(?:that'?s|that is|you'?re|you are) (?:wrong|not right|incorrect|not true|not it)\b"
    r"|^(?:no,? )?(?:wrong|that'?s wrong|not quite)\b|^no,? it'?s not\b|^no,? it isn'?t\b"
)


def _waiting(queue: asyncio.Queue[str | None]) -> list[str]:
    """The sentences already written and not yet said when the person talked over the reply."""
    waiting: list[str] = []
    while not queue.empty():
        item = queue.get_nowait()
        if item:
            waiting.append(item.strip())
    return waiting


# "Okay." on its own: the person acknowledges. YES (an offer's yes) is checked before this, with an offer open.
BARE_ACK = re.compile(
    r"^(?:ok(?:ay)?|okay then|alright|all right|right|got it|gotcha|cool|fine|good|great|nice|perfect|"
    r"i see|makes sense|fair enough|understood|noted|mm+|hmm+|mhm|uh huh)$"
)
# After a statement that asks for nothing: Heliox talking about what it can't check ("I can't confirm it's still
# open, though."), and a sentence offering to do something.
HEDGE = re.compile(
    r"^(?:but |though )?i (?:can't|cannot|can not|couldn't|haven't|have not|don't|do not) "
    r"(?:confirm|check|tell|see|verify|know|be sure)\b"
)
OFFER = re.compile(
    r"^(?:want me to|do you want me to|would you like me to|should i|shall i|i can|i could|if you want i can|"
    r"if you like i can)(?!'t|n't| not)\b"
)
# What a statement hears when every sentence the model wrote was dropped (an offer the rules did not allow): a sign
# it was heard, never silence after the wake phrase.
HEARD_YOU = "Right."
_OFFER_LEAD = re.compile(
    r"^(?:want me to|do you want me to|would you like me to|should i|shall i|i can|i could|if you want,? i can|"
    r"if you like,? i can)\s+",
    re.IGNORECASE,
)


def offered_request(sentence: str) -> str:
    """ "Want me to list the biggest folders?" -> "List the biggest folders": what a yes would ask for."""
    rest = _OFFER_LEAD.sub("", sentence.strip()).rstrip(" ?.!")
    return rest[:1].upper() + rest[1:]


RECENT_REPLIES = 3  # a sentence Heliox said in its last three replies is not said again unprompted
SENTENCE = re.compile(r"(?<=[.!?])\s+")


THANKS_REPLIES = ("You're welcome.", "Any time.", "Sure.", "Glad to help.")


def _safe_call(source: Any, default: Any) -> Any:
    try:
        return source() if source is not None else default
    except Exception:
        return default


CARRY_SECONDS = 10.0  # a joined sentence does not repeat a company line its first half said this recently
ASK_WHAT = "Which ones do you mean?"
ASK_MORE = "Could you tell me a bit more?"
SORRY = "Sorry?"
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
    understood: bool = True  # False: the turn was not understood and opens nothing (the listener is told)
    withdrawn: bool = False  # replaced by the joined sentence: nothing more is said for this turn
    # The conversation before a hand-over, for the planner (and the person's own lines for the reviewers).
    context: HandoverContext | None = None
    _before: list[Any] = field(default_factory=list)
    _sink: Any = None  # the sink for this turn when it is not the layer's own (typed chat)
    quick: QuickPlan | None = None  # the one-action plan for the quick road, when there is one
    route: Route | None = None
    timings: dict[str, float] = field(default_factory=dict)
    spoken: list[str] = field(default_factory=list)
    _layer: Any = None
    _company: asyncio.Task[None] | None = None

    def keep_company(self, *, acknowledge: bool = True) -> asyncio.Task[None]:
        """Start the acknowledgement and progress speech; the server cancels it when the planner returns.

        ``acknowledge=False``: the acknowledgement was already said for this request (the first half of a sentence
        split at a pause); progress and the approval pointer are said as usual."""
        assert self._layer is not None and self.intent is not None
        self._company = self._layer._start_company(self, acknowledge=acknowledge)
        return self._company

    def execute_params(self) -> dict[str, Any]:
        """What the server adds to the planner call, each only as an object (never text a client could send): the
        quick plan, used once, and the conversation before the request."""
        params: dict[str, Any] = {}
        if self.quick is not None:
            params["_quick_plan"] = self.quick.plan
        if self.context is not None:
            params["_conversation_context"] = self.context
        return params

    def spoken_result(self, result: str, status: str = "success") -> str:
        """The result in a sentence ("" leaves the server's own wording): on the quick road a state read in words, a
        done action briefly, a failure with what failed; on the planner's, from its end state (action_talk)."""
        if self.quick is None:
            return result_sentence(status, result)
        return quick_result_sentence(self.quick, result, status)

    def finish(self, status: str, result: str, said: str = "") -> None:
        """The planner's outcome, for the conversation: the next question can refer to it. ``said``: the words the
        person heard for it, which are what the conversation keeps (the raw result is the task's own record)."""
        if self._layer is not None:
            self._layer._finished(self, status, result, said)


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
        addressed: Callable[[], str] = lambda: "",
        not_understood: Callable[[], None] = lambda: None,
        search_url: Callable[[], str] = lambda: "",
        turn_ended: Callable[[], float | None] = lambda: None,
        screen_text_ok: Callable[[], bool] = lambda: False,
        provisional_setting: Callable[[], str] = lambda: "off",
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
        # How the turn reached Heliox ("wake", "button", "follow_up", "doubtful_follow_up"; "" unknown), and how to
        # tell the listener a turn was not understood so its follow-up window is not extended.
        self._addressed = addressed
        self._not_understood = not_understood
        self._sorry_said = False
        self._said_local_slow = False  # "the model on this computer is slow the first time" said once
        self._company_said: tuple[str, float] | None = None  # the last company line said, and when
        self._last_thanks = ""  # the last reply to a plain thanks: the next one is different
        self._unsaid: tuple[str, float] | None = None  # the rest of a reply the person talked over, and when
        self._thanks_said = 0
        self._search_url = search_url
        # Whether the planner's model may be given screen text (conversation.share_screen_text), and whether the
        # fast model was shown some in this conversation (then Heliox's lines may repeat it).
        self._screen_text_ok = screen_text_ok
        self._saw_screen_text = False
        self._turn_ended = turn_ended
        self._last_ack = ""
        # The last turn Heliox answered in code: (its answer, what it changed, when). None: the last turn was not.
        self._last_done: tuple[str, tuple[str, ...], float] | None = None
        self.last_result = ""
        self.turns: list[TurnResult] = []  # recent turns, for measurement and the debug log
        # Answers started on the listener's words before the turn was confirmed (pilot.conversation.provisional).
        self._provisional_setting = provisional_setting
        self.provisional = prov.ProvisionalAnswers(clock=clock)

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

    # ── provisional answers (conv_interfaces.md, section c) ──

    def provisional_on(self, model: Any) -> bool:
        """The setting: "on" for every model; "auto" for a cloud model only when the measured adoption rate earns
        it (a model on this computer would share the cores the accurate recognition pass needs); "off" never."""
        try:
            setting = str(self._provisional_setting() or "auto").strip().casefold()
        except Exception:
            setting = "off"
        if setting == "on":
            return True
        if setting != "auto" or getattr(model, "local", False):
            return False
        rate = prov.MEASURED_ADOPTION_RATE
        return rate is not None and rate >= prov.AUTO_ADOPTION_RATE

    def begin_provisional(self, turn_id: str, words: str, addressed: str) -> bool:
        """Start the fast model on the words the listener has now, holding everything it says until the final words
        for ``turn_id`` arrive. True if it started. Only for an utterance known to be addressed to Heliox before its
        end, and only when the words would be answered by the fast model in speech."""
        if addressed not in prov.ADDRESSED or not turn_id or not self._enabled():
            return False
        if len(prov.normalise(words).split()) < prov.MIN_WORDS:
            return False
        fixed = misheard_command(words, addressed=addressed)
        text = fixed if fixed is not None else words
        decided = route(text, conversation_open=self.memory.open)
        if decided.kind in (TurnKind.ACTION, TurnKind.CONTROL, TurnKind.UNCLEAR) or _CONTROL_START.match(
            normalize(text)
        ):
            return False  # a request to act, a control word or unclear words: those roads never start early
        offer = self._offer
        if offer is not None and normalize(text) in YES:
            return False
        try:
            model = self._model()
        except Exception:
            return False
        if model is None or not getattr(model, "streams", False) or not self.provisional_on(model):
            return False
        refused = self.provisional.refusal(model)
        if refused:
            logger.info("No provisional answer for %s (%s)", turn_id, refused)
            return False
        held = prov.Provisional(
            turn_id=turn_id,
            words=words,
            addressed=addressed,
            started=self._clock(),
            model=model,
            before=self.memory.recent(),
        )

        async def build() -> tuple[list[dict[str, str]], Any]:
            screen = await self._read_screen(model)
            return build_messages(text, self.sources, self.memory, budget=self._context_chars, screen=screen), screen

        self.provisional.add(held)
        held.task = asyncio.ensure_future(prov.pump(held, SYSTEM_PROMPT, build))
        self._background.add(held.task)
        held.task.add_done_callback(self._background.discard)
        try:
            asyncio.get_running_loop().call_later(prov.FINAL_WITHIN_SECONDS, self.provisional.expire, held)
        except RuntimeError:
            pass
        logger.info("Provisional answer started for %s", turn_id)
        return True

    def cancel_provisional(self, turn_id: str, reason: str) -> None:
        """The listener: the person carried on, the addressing rule did not take the utterance, or it was dropped."""
        self.provisional.drop(turn_id, reason or "dropped")

    def provisional_dispatch(self, turn_id: str) -> None:
        """A voice dispatch starts: every provisional answer for another turn is dropped (a stop, a new request), and
        this one waits for the layer's turn to adopt it."""
        self.provisional.drop_others(turn_id)
        prov.dispatch_turn.set(turn_id)

    def provisional_done(self, turn_id: str) -> None:
        """The voice dispatch for ``turn_id`` has ended: a provisional answer it did not adopt is dropped."""
        self.provisional.finish(turn_id)

    async def _read_screen(self, model: Any) -> Any:
        """What the screen shows, for the fast model's context, as _answer reads it."""
        if self.sources.screen_async is not None:
            try:
                async with asyncio.timeout(0.5):
                    return await self.sources.screen_async(bool(getattr(model, "local", False)))
            except Exception:
                return None
        return self.sources.screen()

    def _adoptable(self, text: str, decided: Route, source: str) -> prov.Provisional | None:
        """The provisional answer this turn can take over: the same turn, the same words, the conversation as it was,
        and the spoken fast-model road. Anything else drops it, and the turn asks the model itself."""
        turn_id = prov.dispatch_turn.get() if source == "voice" else ""
        held = self.provisional.claim(turn_id) if turn_id else None
        if held is None:
            return None
        if held.dropped:
            return None
        if not prov.same_words(held.words, text):
            self.provisional.not_adopted(held, "other_words")
            return None
        if decided.kind in (TurnKind.ACTION, TurnKind.CONTROL) or held.messages is None and held.finished:
            self.provisional.not_adopted(held, "other_road")
            return None
        if [(t.role, t.text, t.at) for t in self.memory.recent()] != [(t.role, t.text, t.at) for t in held.before]:
            self.provisional.not_adopted(held, "conversation_moved")
            return None
        return held

    def clear(self) -> None:
        """The "new conversation" call: earlier turns no longer frame the next question."""
        self.interrupt()
        self.memory.clear()
        self._sorry_said = False
        self._saw_screen_text = False
        self.last_result = ""

    async def take(self, text: str, *, source: str = "voice", sink: Sink | None = None) -> TurnResult | None:
        """Handle one addressed utterance that no control or local handler took. None: the layer is off."""
        if not self._enabled():
            return None
        started = self._clock()
        if source == "voice":
            fixed = misheard_command(text, addressed=self._how_addressed())
            if fixed is not None:
                logger.info("Read %r, said right after the wake phrase, as the command %r", text, fixed)
                text = fixed
        decided = route(text, conversation_open=self.memory.open)
        result = TurnResult(done=False, planner_input=text, route=decided, _layer=self, source=source, _sink=sink)
        if not self.memory.open:
            self._saw_screen_text = False
        result._before = self.memory.recent()  # the turns before this one, for the planner's context
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
            result._before = self.memory.recent()  # with Heliox's offer and this yes; the request is the person's own
            return self._hand_over(result, offer[1], quick=offer[2])
        if decided.kind is not TurnKind.CONTROL and source == "voice" and self._how_addressed() == DOUBTFUL:
            # A follow-up the listener doubted was meant for Heliox: probably speech that was not addressed to it.
            # It is never sent to a model (overheard speech never reaches one), never acted on, not kept as
            # conversation, opens nothing, and at most one short "Sorry?" is said per conversation.
            result.understood = False
            result.done = True
            if not self._sorry_said:
                self._sorry_said = True
                outcome = await self._sink(SORRY, "final_answer", f"conv-sorry:{id(result)}")
                if outcome in HEARD:
                    result.spoken.append(SORRY)
            try:
                self._not_understood()
            except Exception:
                logger.debug("Could not tell the listener the turn was not understood", exc_info=True)
            return result
        done, self._last_done = self._last_done, None
        settled = self._kept_or_changed(text, done)
        if settled is not None:
            # Answered in code, true by construction: the model is not asked whether anything was kept.
            self.memory.add_user(text)
            await self._say_one(result, settled, started)  # remembered there, as said
            if done is None:
                self._last_done = (settled, (), self._clock())  # "did you get that?" next: no
            result.done = True
            return result
        if ARE_YOU_THERE.match(normalize(text)):
            # A check-in, often after a long quiet: one short line from code, and nothing about the old topic.
            self.memory.add_user(text)
            await self._say_one(result, HERE, started)
            result.done = True
            return result
        if CORRECTED.match(normalize(text)) and self.memory.last_heliox():
            # "No, that's wrong": the last reply is marked wrong for the model (it gives the right answer, and the
            # wrong one is not given again later); the turn itself goes on to the model.
            self.memory.mark_last_wrong()
        rest = self._the_rest(text)
        if rest:
            # The rest of the reply the person talked over, said now that they asked for it; no model call.
            self.memory.add_user(text)
            for sentence in SENTENCE.split(rest):
                outcome = await self._say_one(result, clean_for_speech(sentence), started)
                if outcome in STOPPED:
                    break
            result.done = True
            return result
        thanked = self._thanks_reply(text, offer)
        if thanked:
            # A plain thanks with nothing pending: a short reply from code, with no model call and no wait.
            self.memory.add_user(text)
            await self._say_one(result, thanked, started)  # remembered there, as said
            result.done = True
            return result
        if self._nothing_open(offer) and BARE_ACK.match(normalize(text)):
            # "Okay." with nothing open: the person heard the answer and is done. Nothing is said, and the model is
            # not asked: given the conversation, it would answer the last question again.
            self.memory.add_user(text)
            result.done = True
            return result
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

        held = self._adoptable(text, decided, source)
        if held is not None:
            task = asyncio.ensure_future(self._adopt(held, text, decided, result, started))
        else:
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
                async with asyncio.timeout(0.5):  # not wait_for: it can lose a cancellation before 3.12
                    screen = await self.sources.screen_async(bool(getattr(model, "local", False)))
            except Exception:
                screen = None  # not available here, or too slow to wait for
        else:
            screen = self.sources.screen()
        if screen is not None and getattr(screen, "excerpt", ""):
            self._saw_screen_text = True
        messages = build_messages(text, self.sources, self.memory, budget=self._context_chars, screen=screen)
        self.memory.add_user(text)
        return await self._answer_from(text, decided, result, started, model, messages)

    async def _adopt(
        self, held: prov.Provisional, text: str, decided: Route, result: TurnResult, started: float
    ) -> TurnResult:
        """The final words are the provisional ones: the running answer becomes this turn's own, said and kept
        exactly as if this turn had asked for it."""
        self.provisional.note_adopted(held)
        result.timings["provisional_ms"] = max(0.0, (started - held.started) * 1000)
        logger.info("Adopted the provisional answer for %s", held.turn_id)
        while held.messages is None and not held.finished:
            held._changed.clear()
            await held._changed.wait()
        if held.screen is not None and getattr(held.screen, "excerpt", ""):
            self._saw_screen_text = True
        self.memory.add_user(text)
        return await self._answer_from(text, decided, result, started, held.model, held.messages or [], held)

    async def _answer_from(
        self,
        text: str,
        decided: Route,
        result: TurnResult,
        started: float,
        model: Any,
        messages: list[dict[str, str]],
        adopted: prov.Provisional | None = None,
    ) -> TurnResult:
        """Speak the model's answer as it streams (``adopted``: a provisional answer taken over, already running)."""
        typed = result.source == "typed"
        sink = result._sink or self._sink
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        act_text: list[str] = []
        failure: list[str] = []
        promised: list[str] = []
        late_act: list[
            bool
        ] = []  # an ACT line came after words: never spoken, and it hands over only if nothing held back

        async def produce() -> None:
            splitter: Any = TextChunker() if typed else SentenceSplitter()
            mode: bool | None = None  # True: an ACT line, False: words to speak, None: too early to tell
            head = ""
            tail = ""  # the end of the words when it could start an ACT line ("AC"), held until the next piece
            count = 0
            stream = (
                adopted.replay()
                if adopted is not None
                else model.stream(TYPED_SYSTEM_PROMPT if typed else SYSTEM_PROMPT, messages)
            )
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
                            answered = getattr(model, "note_answered", None)
                            if callable(answered):
                                answered()
                        if mode is None:
                            head += fragment
                            mode = looks_like_act(head)
                            if mode is None:
                                continue
                            fragment, head = head, ""
                        if mode:
                            act_text.append(fragment)
                            continue
                        tail += fragment
                        at = late_act_at(tail)
                        if at >= 0:
                            fragment, mode = tail[:at], True
                            act_text.append(tail[at:])
                            late_act.append(True)
                            tail = ""
                        else:
                            held = act_start_held(tail)
                            fragment, tail = (tail[:-held], tail[-held:]) if held else (tail, "")
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
                    for sentence in [*splitter.feed(tail), *splitter.flush()]:
                        await queue.put(sentence)
            except FastModelUnavailable as error:
                failure.append(error.reason)
            except TimeoutError:
                failure.append("slow")
                note_slow = getattr(model, "note_slow", None)
                if callable(note_slow):
                    note_slow()  # one late answer costs this turn; only late answers in a row rest the provider
            finally:
                # Closing the stream ends the request now (a cut-off reply stops costing tokens) and records usage.
                with contextlib.suppress(Exception):
                    await stream.aclose()
                await queue.put(None)

        producer = asyncio.ensure_future(produce())
        statement = decided.reason == "statement"
        repeat_asked = self._asked_again(text)
        dropped = False
        said: list[str] = []
        current = ""
        try:
            first = True
            capped = False
            held = False
            while True:
                if first and not held and not typed:
                    try:
                        # asyncio.timeout, not wait_for: before Python 3.12 an interrupt arriving as the first
                        # sentence is dequeued could be lost, and Heliox would go on talking after being stopped.
                        async with asyncio.timeout(self._hold_after):
                            sentence = await queue.get()
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
                if not typed and statement and said and not ends_with_question(clean):
                    # A statement that asks for nothing gets one sentence, and at most one question after it. The
                    # rest is read (a promise to act still hands the request over) but not said.
                    continue
                if not typed and not self._allowed_in_reply(
                    clean, statement=statement, said=said, repeat_asked=repeat_asked, heard=text
                ):
                    dropped = True
                    continue
                words_so_far = sum(len(part.split()) for part in said)
                if not typed and said and words_so_far + len(clean.split()) > self._word_cap():
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
                    self._remember_said(said, clean, _waiting(queue))
                    result.done = True
                    return result
                if outcome in HEARD or outcome == "duplicate":
                    said.append(clean)
                    result.spoken.append(clean)
                if not typed and ends_with_question(clean):
                    # One question per turn, and it ends the reply: the person answers it, not a second one.
                    capped = True
                    break
            if not capped:
                await producer  # (when capped, the finally below closes the stream, so it stops costing tokens)
        except asyncio.CancelledError:
            # A new utterance arrived: what was heard is remembered, the sentence being spoken as cut off.
            self._remember_said(said, current or None, _waiting(queue))
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
            if act is not None and declined(said, say=act.say):
                # The reply held back ("I can't tell which ones you mean") and wrote an ACT line anyway: the decline
                # wins and nothing is handed over. What it said stands; if it said nothing, its own sentence or a
                # short question is said.
                logger.info("The reply declined and also wrote an ACT line: nothing handed over (%r)", act.request)
                if not said:
                    await self._say_one(result, act.say if hedges(act.say) else ASK_WHAT, started)
                result.done = True
                return result
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
        elif promised and not failure and declined(said):
            # "I can't tell which files you mean, so I won't delete anything yet" with a promise elsewhere in the
            # reply: holding back wins; what was said stands and nothing is handed over.
            logger.info("The reply promised and also held back: nothing handed over")
            if said and said[-1].rstrip()[-1:] not in (".", "?", "!"):
                # The promise that was not said followed a clause cut at a comma: end on a whole question instead
                # of trailing off ("I'm not sure which app you use or when the review starts,").
                await self._say_one(result, ASK_MORE, started)
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
        if not failure and not said and dropped and not typed:
            # Every sentence was held back (an offer the rules did not allow now): a sign it was heard, not silence.
            await self._say_one(result, HEARD_YOU, started)
        result.done = True
        result.timings["done_ms"] = (self._clock() - started) * 1000
        return result

    def _people(self) -> dict[str, str]:
        try:
            return dict(self.sources.people() or {})
        except Exception:
            return {}

    def _thanks_reply(self, text: str, offer: Any) -> str:
        """ "Thanks" said on its own, with nothing pending: a short reply, not the same as the last one. "" when the
        thanks may be part of something open (an approval waiting, a task running, a question or offer of Heliox's
        not yet answered) or the sentence asks for more ("thanks, now open Notepad"): the turn goes on as usual."""
        if not THANKS.match(normalize(text)) or not self._nothing_open(offer):
            return ""
        options = [reply for reply in THANKS_REPLIES if reply != self._last_thanks]
        reply = options[self._thanks_said % len(options)]
        self._thanks_said += 1
        self._last_thanks = reply
        return reply

    def _the_rest(self, text: str) -> str:
        """ "What were you saying?" right after the person talked over a reply: the rest of it, as it was written.
        "" when there is no cut-off reply from the last two minutes (the turn then goes on as usual)."""
        if self._unsaid is None or not WHAT_WERE_YOU_SAYING.match(normalize(text)):
            return ""
        rest, at = self._unsaid
        if self._clock() - at > UNSAID_SECONDS:
            return ""
        self._unsaid = None
        return rest

    def _nothing_open(self, offer: Any) -> bool:
        """No offer of Heliox's, approval, running task or question of Heliox's waiting for the person."""
        if offer is not None:
            return False
        state = _safe_call(self.sources.task_state, {})
        if isinstance(state, dict) and (state.get("approval_waiting") or state.get("running")):
            return False
        return not ends_with_question(self.memory.last_heliox())

    def _allowed_in_reply(
        self, sentence: str, *, statement: bool, said: list[str], repeat_asked: bool, heard: str = ""
    ) -> bool:
        """What code decides about a model's sentence before it is said. Never a sentence Heliox said in its last
        few replies, unless the person asked the same thing again. After a statement that asks for nothing: no
        hedge about what Heliox can't check, and an offer only when the offers' own rules allow one now
        (pilot.personal.wiring.offer_in_reply: on, not the first week, not quiet, in a call or over a cap, not
        declined or silenced); an offer made is then an offer like any other ("yes", "why?", "not today")."""
        words = normalize(sentence)
        if not repeat_asked and words and words in self._recent_heliox_sentences():
            logger.info("Dropped a sentence Heliox said a moment ago: %r", sentence)
            return False
        if statement and HEDGE.match(words):
            logger.info("Dropped a hedge after a statement: %r", sentence)
            return False
        if OFFER.match(words) and " or " not in words:
            # Any reply's offer, not only a statement's: "stop suggesting things" holds for every road. "Do you want
            # me to open a folder, or find something?" asks what the person meant: a question, not an offer.
            request = offered_request(sentence)
            why = f'You said "{" ".join(heard.split())[:120]}".' if heard else ""
            try:
                allowed = bool(self.sources.offer_in_reply(sentence, request, why))
            except Exception:
                logger.warning("The offer check failed; the offer is not made", exc_info=True)
                allowed = False
            if not allowed:
                logger.info("Dropped an offer after a statement (the offers' rules say not now): %r", sentence)
                return False
        return True

    def _recent_heliox_sentences(self) -> set[str]:
        replies = [turn.text for turn in self.memory.recent() if turn.role == "assistant"][-RECENT_REPLIES:]
        return {normalize(part) for reply in replies for part in SENTENCE.split(reply) if normalize(part)}

    def _asked_again(self, text: str) -> bool:
        words = normalize(text)
        earlier = self.memory.recent()
        if earlier and earlier[-1].role == "user" and normalize(earlier[-1].text) == words:
            earlier = earlier[:-1]  # this turn itself, already in the conversation
        return any(turn.role == "user" and normalize(turn.text) == words for turn in earlier)

    def note_code_answer(self, answer: str, changed: tuple[str, ...] | list[str]) -> None:
        """The server answered this turn in code (a handler took the words): what it said, and what it changed."""
        self._last_done = (str(answer or ""), tuple(changed or ()), self._clock())

    def _kept_or_changed(self, text: str, done: tuple[str, tuple[str, ...], float] | None) -> str | None:
        """The fixed sentence for words about keeping or changing something, or None to go on as usual.

        "Did you get that?" right after a turn Heliox answered in code: from what that turn changed. A request to keep
        something or behave differently that reached this layer: no handler took it, so nothing was kept or changed."""
        words = normalize(text)
        if done is not None and self._clock() - done[2] <= DONE_SECONDS and DID_YOU.match(words):
            answer, changed, _at = done
            if changed:
                said = re.sub(r"^(?:okay|ok|done)[,.]\s+", "", answer.strip(), flags=re.IGNORECASE)
                said = said[:1].upper() + said[1:]  # "Yes, I remembered that. You're a founder.", not "Okay, ..."
                return f"Yes, I {' and '.join(changed[:2])}. {said}".strip()
            if answer.startswith("I didn't"):
                return "No, nothing was saved or changed."
            # A turn answered in code that changed nothing ("what time is it?"): the question is not about a save.
        if not SAVE_OR_CHANGE.search(text.strip()):
            return None
        return next((sentence for pattern, sentence in NOT_KEPT if pattern.search(words)), NOT_CHANGED)

    def _word_cap(self) -> int:
        """How long a spoken reply may run: the person's own answer style when they set one ("keep it short",
        "I like detail"), else SPOKEN_WORD_CAP."""
        try:
            length = str(self.sources.answer_length() or "")
        except Exception:
            length = ""
        return PERSONAL_WORD_CAPS.get(length, SPOKEN_WORD_CAP)

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

    def _how_addressed(self) -> str:
        try:
            return str(self._addressed() or "")
        except Exception:
            return ""

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

    def _remember_said(self, said: list[str], cut: str | None, unsaid: list[str] | None = None) -> None:
        """The reply up to where the person talked over it: the sentences said, and of the one being spoken only
        what they heard (the voice's report). The rest is kept a while, for "what were you saying?", and never said
        unasked."""
        heard = ""
        if cut:
            heard = str(_safe_call(lambda: self.sources.heard_of(cut), cut) or "").strip()
            if not cut.startswith(heard):
                heard = cut  # an answer that is not a prefix of the sentence is not trusted
        rest = [cut] if cut and len(heard) < len(cut) else []  # resumed from the start of the cut sentence
        rest += [part for part in (unsaid or []) if part]
        self._unsaid = (" ".join(part for part in rest if part), self._clock()) if any(rest) else None
        if said or heard:
            self.memory.add_heliox(" ".join([*said, heard] if heard else said), cut_off=True)

    async def _say_one(
        self, result: TurnResult, sentence: str, started: float, *, channel: str = "final_answer"
    ) -> str:
        result.timings.setdefault("first_sentence_ms", (self._clock() - started) * 1000)
        outcome = await (result._sink or self._sink)(sentence, channel, f"conv-one:{id(result)}")
        if outcome in STOPPED:
            self._remember_said([], sentence)  # what was heard of it; the rest kept for "what were you saying?"
            result.spoken.append(sentence)
        elif outcome in HEARD:
            self.memory.add_heliox(sentence)
            result.spoken.append(sentence)
        return outcome

    async def _fallback(
        self, result: TurnResult, reason: str, decided: Route, started: float, *, provider: str = ""
    ) -> None:
        """No model: say why in one sentence, once per session; later turns get a short line."""
        logger.info("Fast model unavailable (%s); answering from local rules", reason)
        if self._user_speaking():
            # The person is still talking (the turn was cut at a pause): nothing is said over them, and the reason
            # is kept for the turn they finish.
            result.done = True
            return
        if reason == "unparsed_act" or (decided.kind is TurnKind.UNCLEAR and reason in self._explained):
            sentence = UNCLEAR_OFFLINE
        elif reason in self._explained:
            sentence = FALLBACK_AGAIN.get(reason, FALLBACK_AGAIN_DEFAULT)
            if sentence == self._last_fallback:
                # Never the same line twice running: the short form, then the plain one, in turn.
                other = AGAIN_ALTERNATES.get(reason, AGAIN_ALTERNATE)
                sentence = other if sentence != other else FALLBACK_AGAIN.get(reason, FALLBACK_AGAIN_DEFAULT)
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
        # The planner's request is always the person's own words; the model's reading of them, and the turns before,
        # travel separately as context (never what the reviewers judge the plan against).
        result.planner_input = intent.words or intent.request
        try:
            result.context = build_context(
                result._before,
                words=result.planner_input,
                understood_as=intent.request if intent.source == "model" else "",
                screen_text_ok=bool(self._screen_text_ok()),
                heliox_saw_screen_text=self._saw_screen_text,
                people=self._people(),
            )
        except Exception:
            logger.debug("No conversation context for the planner", exc_info=True)
            result.context = None
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
        return self._first_sentence(result)

    def _first_sentence(self, result: TurnResult) -> str:
        """The quick road's sentence as its one action starts (running), else the asked-for state's: what Heliox
        will do, or "" when the plan will need an approval first (pilot.conversation.action_talk)."""
        assert result.intent is not None
        if result.quick is not None:
            return quick_acknowledgement(result.quick, result.intent.words)
        if not self._has_model():
            return ""  # no model to plan with: promising "I'll ..." would be followed by why it can't
        ack = acknowledgement(result.intent, last=self._last_ack) if result.intent.verb or result.intent.say else ""
        if not ack and needs_approval_first(result.intent.verb):
            return ""
        return ack or "Working on it."

    def _has_model(self) -> bool:
        try:
            return self._model() is not None
        except Exception:
            return False

    def _start_company(self, result: TurnResult, *, acknowledge: bool = True) -> asyncio.Task[None]:
        assert result.intent is not None
        if not acknowledge:
            # The joined half of a split sentence: the first half's acknowledgement and progress line stand.
            return asyncio.ensure_future(
                keep_company(
                    acknowledgement_text="",
                    speak=self._company_speaker(result),
                    status=self._task_status,
                    clock=self._clock,
                    user_speaking=self._user_speaking,
                    local_slow_once=self._local_slow_once,
                    carried=self._recent_company_line(),
                )
            )
        ack = self._first_sentence(result)
        if ack == self._last_ack and ack:
            ack = said_again(ack)
        if ack:
            self._last_ack = ack

        def said(sentence: str) -> None:
            result.spoken.append(sentence)

        return asyncio.ensure_future(
            keep_company(
                acknowledgement_text=ack,
                speak=self._company_speaker(result),
                status=self._task_status,
                clock=self._clock,
                on_said=said,
                user_speaking=self._user_speaking,
                local_slow_once=self._local_slow_once,
            )
        )

    def _company_speaker(self, result: TurnResult) -> Sink:
        """Speech for a hand-over's company: nothing once the turn is withdrawn, and each line remembered so a joined
        sentence does not repeat it."""

        async def speak(text: str, channel: str, key: str) -> str:
            if result.withdrawn:
                return "cancelled"
            outcome = await self._sink(text, channel, key)
            if outcome in HEARD and channel in ("task_narration", "approval_risk"):
                self._company_said = (text, self._clock())
            return outcome

        return speak

    def _recent_company_line(self) -> tuple[str, float] | None:
        """The last acknowledgement or progress line, when it was said a moment ago (CARRY_SECONDS)."""
        said = self._company_said
        if said is None or self._clock() - said[1] > CARRY_SECONDS:
            return None
        return said

    def _local_slow_once(self) -> bool:
        """True the first time in a session a plan from a model on this computer runs long."""
        if self._said_local_slow:
            return False
        try:
            local = bool(getattr(self._model(), "local", False))
        except Exception:
            local = False
        self._said_local_slow = local
        return local

    def _finished(self, result: TurnResult, status: str, text: str, said: str = "") -> None:
        spoken = " ".join(result.spoken)
        summary = " ".join(str(text).split())[:240]
        self.last_result = f"{status}: {summary}" if summary else status
        self.memory.add_heliox(f"{spoken} {said or summary}".strip())


def _current_task_cancelling() -> bool:
    """Whether the caller itself is being cancelled (as opposed to the inner answer task)."""
    task = asyncio.current_task()
    if task is None:
        return False
    cancelling = getattr(task, "cancelling", None)  # 3.11+
    return bool(cancelling()) if callable(cancelling) else False
