"""Where batch six's series meet: times said in words with the said-back rule, a short "No." with a costly request's
concern and a waiting approval, and a secret said as Heliox ends a piece of its reply. Fakes, scripted transcripts,
the server's own wall clock set by hand and a company that polls without waiting; nothing waits on real time."""

from __future__ import annotations

import asyncio
import logging
import types
from datetime import datetime

import pytest

from pilot.config import PilotConfig
from pilot.conversation import handover, layer_for
from pilot.conversation.handover import APPROVAL_SENTENCE
from pilot.conversation.withheld import forget_recent
from pilot.live.commitments import deadline_of, remind_at
from pilot.server import PendingConfirmation, PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.interaction import InteractionPhase
from pilot.system.voice import voice_turn_addressed
from pilot.workflows.reminders import parse_reminder
from tests.test_addressing_said_back import answered, say
from tests.test_conversation_direct_concern import REASON, Model
from tests.test_conversation_due import _server as _due_server
from tests.test_conversation_secrets import shows
from tests.test_conversation_times_said import _card
from tests.test_secret_everywhere import _server_with_model
from tests.test_turn6_gap import as_heliox_stopped
from tests.test_turn6_short_answers import _Done
from tests.test_voice_conversation import make_listener

MORNING = datetime(2026, 11, 24, 8, 10)
DECISION = "I'm going to watch a series tonight and do the report in the morning"


@pytest.fixture(autouse=True)
def _fresh():
    forget_recent()
    yield
    forget_recent()


# ── (a) Live's evening card: the time it says is the time the reminder is set for ──


def test_an_evening_card_says_the_moment_its_reminder_is_set_for():
    """ "This evening" is 19:00 and the offer is the usual hour before it: the card says "six p m", and the reminder
    its fix sets lands at 18:00, read back by the product's own reminder parser."""
    now = datetime(2026, 11, 24, 11, 21)
    [card] = _card("I'm going to call my mum this evening.", now)
    deadline, has_time = deadline_of("this evening", now.date())
    assert deadline == datetime(2026, 11, 24, 19, 0) and has_time
    moment = remind_at(deadline, has_time, now)
    assert moment == datetime(2026, 11, 24, 18, 0)
    assert "six p m" in card.title
    parsed = parse_reminder(card.fix, now)
    assert parsed is not None and datetime.fromtimestamp(parsed[0]) == moment


# ── (b1) a to-do said with its time in words, then a follow-up without the wake phrase in the morning ──


async def test_a_spoken_to_do_with_its_time_in_words_is_grounds_for_a_morning_follow_up(tmp_path):
    server, sent = _due_server(tmp_path, "claude")
    server._wall_clock = lambda: MORNING
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("add finish the quarterly report by nine tomorrow morning to my to-do list")
    assert not sent  # kept in code: no model, no planner
    assert "nine tomorrow morning" in server._todos.open_items()[0]["text"]
    rule, clock = answered()
    assert say(rule, clock, DECISION, 1.0).act  # said back without the wake phrase, and taken
    voice_turn_addressed.set("follow_up")  # a quiet room: the follow-up keeps its reach
    await server._voice_command_dispatch(DECISION)
    assert 'a to-do about \\"report\\", due tomorrow at 9:00 in the morning' in sent[-1]


# ── (b2) "No." while an approval waits, after a costly request that got its concern ──


@pytest.fixture
def company_without_waiting(monkeypatch):
    """The company's poll (asyncio.sleep(POLL_SECONDS)) yields once instead of waiting real time."""
    real = asyncio

    async def no_wait(_seconds: float = 0.0) -> None:
        await real.sleep(0)

    proxy = types.SimpleNamespace(**{name: getattr(real, name) for name in dir(real) if not name.startswith("__")})
    proxy.sleep = no_wait
    monkeypatch.setattr(handover, "asyncio", proxy)


async def _until(condition, rounds: int = 20000) -> None:
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("never happened")


async def test_no_while_the_approval_waits_ends_the_turn_and_closes_once_and_yes_approves_nothing(
    company_without_waiting,
):
    server = PilotServer(PilotConfig())
    heard: list[str] = []
    answered_with: list[bool] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        heard.append(text)
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def reserved(params, ws):
        """The product's own _handle_execute around it; this part stands for planning to an approval."""
        pending = PendingConfirmation(plan_id="p1", event=asyncio.Event())
        server._pending_confirms["p1"] = pending
        server._active_plan_id = "p1"
        await server._interaction_runtime.transition(InteractionPhase.AWAITING_APPROVAL, message="Waiting")
        await pending.event.wait()
        answered_with.append(pending.confirmed)
        server._pending_confirms.pop("p1", None)
        return {"status": "cancelled", "message": "Cancelled before execution."}

    model = Model()
    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute_reserved = reserved
    server._model_router = object()
    layer_for(server)._model = lambda: model
    voice_turn_addressed.set("wake")
    request = asyncio.ensure_future(server._voice_command_dispatch("delete my only copy of the thesis for me"))
    await _until(lambda: APPROVAL_SENTENCE in heard)
    assert heard.index(REASON) < heard.index(APPROVAL_SENTENCE)  # the concern, then the pointer

    listener, _ = make_listener([])
    listener.task_open = server._voice_task_open  # an approval waits
    for words in ("No.", "Yes."):
        listener._speculation = _Done(words)
        assert listener._short_answer_heard()  # the turn ends as soon as the pass has read it

    voice_turn_addressed.set("follow_up")
    await server._voice_command_dispatch("Yes")
    assert "p1" in server._pending_confirms and not server._pending_confirms["p1"].event.is_set()  # not approved
    await server._voice_command_dispatch("No")
    await request
    assert answered_with == [False]
    closing = [line for line in heard if "won't" in line or "Cancelled" in line or "Nothing ran" in line]
    assert len(closing) == 1, heard
    assert len(model.calls) == 1  # one concern call, and nothing asked of the model by "Yes" or "No"


# ── (b3) a secret said as Heliox ends a piece of its reply ──


async def test_a_question_holding_a_secret_said_as_heliox_stops_is_answered_without_it(caplog):
    server, sent = _server_with_model()
    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("Tell me how the Eiffel Tower was built.")
    asked_before = len(sent)
    # Twelve words take about 3 s to say; at 1.6 s or less over Heliox only a stop or a plain short request is read.
    listener, run = as_heliox_stopped(
        "Wait, my PIN is 4471, what do I need it for?", began_before=0.05, went_on=2.8, voiced=3.0
    )
    listener._on_command = server._voice_command_dispatch
    with caplog.at_level(logging.DEBUG):
        _dispatched, overheard = await run()
    assert len(sent) > asked_before and "[left out: a PIN]" in sent[-1]
    for place in (*sent, caplog.text, *overheard, listener.last_transcript):
        assert not shows("4471", place), place
    assert not any(shows("4471", turn.text) for turn in layer_for(server).memory.recent())


async def test_a_secret_ending_with_heliox_s_voice_is_recorded_without_its_words(caplog):
    listener, run = as_heliox_stopped("My PIN is 4471.", began_before=1.5, went_on=0.2)  # taken for Heliox's echo
    with caplog.at_level(logging.DEBUG):
        dispatched, overheard = await run()
    assert dispatched == []
    assert overheard == ["My PIN is [left out: a PIN]."]  # the record Live gets: the left-out form only
    reasons = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Not taken:")]
    assert len(reasons) == 1 and reasons[0].startswith("Not taken: heard while Heliox was talking")  # no words
    assert not shows("4471", caplog.text) and not shows("4471", listener.last_transcript)
