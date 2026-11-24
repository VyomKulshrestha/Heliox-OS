"""The turn-taking rules and the secret rules together.

The turn-taking series takes more speech as a turn without the wake phrase (said back right after an answer, said
over Heliox, begun in the gap between two pieces) and keeps a cut reply as far as it was heard. The knowledge series
takes a secret out where the words first exist (ContinuousVoiceListener._final_words) and keeps it out of every
model, log line, notice and stored turn. These tests run the turn-taking roads with a secret in the words, and a
costly request said back without the wake phrase. Fakes, scripted transcripts and clocks the tests control; nothing
waits on real time."""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.context import ContextSources
from pilot.conversation.handover import APPROVAL_SENTENCE
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.withheld import forget_recent, left_out
from pilot.server import PilotServer
from pilot.system import local_tts_worker
from pilot.system.addressing import TALK_TAKEN_SECONDS
from pilot.system.voice import voice_turn_addressed
from tests.test_addressing_said_back import answered, say
from tests.test_conversation_direct_concern import REASON, Model, _company
from tests.test_conversation_secrets import shows
from tests.test_secret_everywhere import _server_with_model
from tests.test_turn_cut_reply_kept import _Voice, coordinator_for
from tests.test_turn_interruption import over, replying
from tests.test_voice_conversation import make_listener

HUNTER = "My password is hunter two, don't tell anyone."
PIN_OVER = "Wait, my PIN is 4471."
PIN_QUESTION_OVER = "Wait, my PIN is 4471, what do I need it for?"


@pytest.fixture(autouse=True)
def _fresh():
    forget_recent()
    yield
    forget_recent()


def _nowhere(value: str, *places: object) -> None:
    """``value`` is in none of ``places``: a PIN only as a whole run of digits (shows), so a log's millisecond offset
    such as "+244718" is not taken for 4471; a word secret as a plain substring, ignoring case."""
    for place in places:
        assert not shows(value, place), place


# ── (1) a secret said back without the wake phrase, within the short while after an answer ──


@pytest.mark.parametrize("after", [0.5, 1.3, TALK_TAKEN_SECONDS - 0.5])
def test_the_said_back_rule_takes_the_secret_sentence_as_the_listener_hands_it_on(after):
    """The addressing rule reads the words after _final_words: the left-out form. It is still a reply in shape."""
    rule, clock = answered()
    words = left_out(HUNTER)
    assert "hunter" not in words
    decision = say(rule, clock, words, after)
    assert decision.act, decision.why
    assert "hunter" not in decision.command


async def test_a_secret_said_back_reaches_no_model_log_notice_or_stored_turn(caplog):
    server, sent = _server_with_model()
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("What's the capital of Australia?")  # answered: the window is open
    listener, run = make_listener([HUNTER], follow_up=30)
    listener._on_command = server._voice_command_dispatch
    statuses: list[tuple[str, dict]] = []

    async def status(name, data):
        statuses.append((name, data))

    listener._on_status = status
    with caplog.at_level(logging.DEBUG):
        _dispatched, overheard = await run()
    assert len(sent) >= 2  # taken as a turn without the wake phrase, and answered
    assert "[left out: a password]" in sent[-1]
    _nowhere("hunter", *sent, caplog.text, statuses, overheard, listener.last_transcript)
    _nowhere("hunter", *(turn.text for turn in layer_for(server).memory.recent()))


async def test_no_early_request_starts_on_a_secret():
    """An early request starts only for the wake phrase or the button, never for a follow-up or speech over Heliox;
    and the layer refuses a secret whichever words reach it."""
    listener, _run = make_listener([], follow_up=30)
    for addressed in ("follow_up", "interruption"):
        assert listener.early_request_words(HUNTER, {"addressed": addressed, "said": HUNTER}) is None
    model = Model()
    layer = ConversationLayer(sink=AsyncMock(return_value="spoken"), model=lambda: model)
    assert not layer.begin_provisional("v1", "hey heliox " + HUNTER, "wake")
    assert not layer.begin_provisional("v2", HUNTER, "follow_up")
    assert model.calls == []


# ── (2) a secret said over Heliox: the interruption's words and the gap's words ──


def test_a_secret_statement_said_over_heliox_stops_it_and_is_not_a_request():
    """Over Heliox only a stop, a question or a request is taken (the interruption rule); a statement holding a
    secret is a stop when short and not taken when long, as any statement is. The words are the left-out form."""
    rule, clock = replying()
    short = over(rule, clock, left_out(PIN_OVER), 1.2)
    assert short.act and short.command == "Stop"
    rule, clock = replying()
    question = over(rule, clock, left_out(PIN_QUESTION_OVER), 2.4)
    # Opening with "Wait" and longer than a cut-off: taken whole, as for any words over Heliox that open that way.
    assert question.act and question.why in {"interruption", "follow-up: control"}
    assert question.command == "Wait, my PIN is [left out: a PIN], what do I need it for"


@pytest.mark.parametrize(("said", "voiced"), [(PIN_OVER, 1.2), (PIN_QUESTION_OVER, 2.4)])
async def test_a_secret_said_over_heliox_goes_nowhere(said, voiced, caplog):
    """Barge-in during a piece and speech begun in the gap between two pieces end the same way: the watcher calls
    note_interruption (BargeInWatcher._wait_for_speech_start) and the words are read by the listener's loop, through
    _final_words, like any other utterance (tests/test_secret_everywhere.py checks it is the only road)."""
    server, sent = _server_with_model()
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("Tell me how the Eiffel Tower was built.")
    asked_before = len(sent)
    server._voice_cut_off_at = time.monotonic() - 0.5  # the barge-in has just stopped Heliox
    listener, run = make_listener([said], suppressed=True, follow_up=30)
    listener._on_command = server._voice_command_dispatch
    listener.note_interruption()
    listener._recorder.last_voiced_seconds = voiced
    statuses: list[tuple[str, dict]] = []

    async def status(name, data):
        statuses.append((name, data))

    listener._on_status = status
    with caplog.at_level(logging.DEBUG):
        _dispatched, overheard = await run()
    _nowhere("4471", *sent, caplog.text, statuses, overheard, listener.last_transcript)
    _nowhere("4471", *(turn.text for turn in layer_for(server).memory.recent()))
    if said == PIN_QUESTION_OVER:
        assert len(sent) > asked_before and "[left out: a PIN]" in sent[-1]  # answered without the value


# ── (3) a costly request said back without the wake phrase ──

THESIS = "Delete my only copy of the thesis for me."


async def test_a_costly_request_said_back_gets_its_reason_before_the_approval():
    rule, clock = answered()
    decision = say(rule, clock, THESIS, 1.0)
    assert decision.act, decision.why  # taken without the wake phrase
    voice_turn_addressed.set("follow_up")  # a quiet room: the follow-up keeps its reach
    model = Model()
    layer = ConversationLayer(
        sink=AsyncMock(return_value="spoken"), model=lambda: model, disagree=lambda: "when_it_matters"
    )
    result = await layer.take(decision.command, source="voice")
    assert not result.done and result.intent.verb == "delete"
    concern = result._concern
    assert concern is not None and await concern == REASON
    said = await _company(concern, approval_at=1.5)
    assert said[:2] == [REASON, APPROVAL_SENTENCE]
    assert not any(line.endswith("?") for line in said)


@pytest.mark.parametrize("words", ["Yes.", "Yes, go ahead.", "Approve it."])
async def test_a_yes_said_back_still_never_approves(words):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._pending_confirms["plan-1"] = MagicMock()
    server._handle_confirm = AsyncMock()
    rule, clock = answered()
    assert say(rule, clock, words, 1.0).act  # said back, no wake phrase
    voice_turn_addressed.set("follow_up")
    await server._voice_command_dispatch(words.strip(" .?!"))
    server._handle_confirm.assert_not_awaited()


# ── (4) a reply cut off by the person keeps only the left-out form ──

FIRST = "Your bank's help line can reset it."
SECOND = "Room 4471 of the branch on the high street handles that, from nine to five on weekdays."


class _Scripted:
    """A talk model that writes one scripted reply per request and keeps every request it was sent."""

    name = provider = "fake"
    streams, local = True, False

    def __init__(self, replies: list[list[str]]) -> None:
        self.replies = list(replies)
        self.sent: list[str] = []

    async def stream(self, system, messages):
        self.sent.append(f"{system}\n{messages}")
        for fragment in self.replies.pop(0) if self.replies else ["Okay."]:
            yield fragment


@pytest.fixture
def voice(monkeypatch):
    found = _Voice()
    monkeypatch.setattr(local_tts_worker, "manager", lambda: found)
    return found


async def test_a_reply_cut_by_words_holding_a_secret_keeps_only_the_left_out_form(voice, caplog):
    """The person's PIN is in the conversation; the reply happens to say the same number and is cut off there by
    "wait, my PIN is 4471". The part kept as heard, the person's interruption and every later request hold only the
    marker."""
    coordinator, reached = coordinator_for(voice, cut=SECOND, played=3.0)
    model = _Scripted([[f"{FIRST} ", f"{SECOND} "], ["Okay, I won't use it."]])

    async def sink(text: str, channel: str, key: str) -> str:
        return (await coordinator.speak(text, channel=channel, dedupe_key=key)).status

    def heard_of(text: str) -> str:
        heard = coordinator.heard_of(text)
        return text if heard is None else heard

    layer = ConversationLayer(sink=sink, model=lambda: model, sources=ContextSources(heard_of=heard_of))
    with caplog.at_level(logging.DEBUG):
        turn = asyncio.ensure_future(layer.take("My PIN is 4471, who can reset it?", source="voice"))
        waiting = asyncio.ensure_future(reached.wait())
        await asyncio.wait({turn, waiting}, return_when=asyncio.FIRST_COMPLETED)
        assert waiting.done(), "the reply never reached its second sentence"
        assert layer.interrupt()  # the person talked over it
        await turn
        await layer.take(PIN_OVER, source="voice")
    kept = [t for t in layer.memory.recent() if t.role == "assistant" and t.cut_off]
    assert len(kept) == 1 and kept[0].text.startswith(FIRST)
    assert "[left out: " in kept[0].text  # the number was heard; it is kept only as the marker
    _nowhere("4471", caplog.text, *model.sent, *(t.text for t in layer.memory.recent()))
