"""What the conversation eval found on real model output, fixed: Live remarks explained from their evidence, varied
openings, numbers and file names said well, a length cap at a sentence end, "plan" kept as talk, and the fast layer
and the planner agreeing (a promise without ACT, an ACT for a how-to question, and the yes that follows)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.conversation.context import ContextSources, ConversationMemory, build_messages
from pilot.conversation.layer import OFFER_SECONDS, SPOKEN_WORD_CAP, ConversationLayer
from pilot.conversation.router import TurnKind, route
from pilot.conversation.style import SYSTEM_PROMPT, promised_action, speakable_numbers


class Model:
    name, provider, streams, local = "fake", "fake", True, False

    def __init__(self, *pieces):
        self.pieces = pieces
        self.calls = []
        self.closed_early = False

    async def stream(self, system, messages):
        self.calls.append(messages)
        try:
            for piece in self.pieces:
                yield piece
                await asyncio.sleep(0)
        except (GeneratorExit, asyncio.CancelledError):
            self.closed_early = True
            raise


class Replies(Model):
    """One reply per call, in order."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    async def stream(self, system, messages):
        self.calls.append(messages)
        yield self.replies.pop(0) if self.replies else "Okay."


class Sink:
    def __init__(self):
        self.said = []

    async def __call__(self, text, channel, key):
        self.said.append(text)
        return "spoken"


def layer_with(model, clock=None):
    kwargs = {"clock": clock} if clock else {}
    return ConversationLayer(sink=Sink(), model=lambda: model, hold_after=60.0, **kwargs)


# ── Live remarks, openings ──


def test_a_live_remark_is_shown_as_live_speaking_up_and_the_prompt_says_how_to_explain_it():
    memory = ConversationMemory()
    memory.add_heliox("The twelfth of November is a Thursday. You said Monday.", source="live")
    memory.add_user("why?")
    messages = build_messages("why?", ContextSources(), memory)
    joined = " ".join(m["content"] for m in messages)
    assert "Heliox Live spoke up on its own" in joined and "You said Monday" in joined
    assert "Heliox did not make the mistake" in SYSTEM_PROMPT


def test_the_last_opening_is_in_the_context():
    memory = ConversationMemory()
    memory.add_user("what's in my other window?")
    memory.add_heliox("I can't see your other window.")
    final = build_messages("and my last commit?", ContextSources(), memory)[-1]["content"]
    assert "Your last reply began" in final and "I can't" in final and "begin this one differently" in final


# ── numbers, dates, file names ──


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("The file is 16GB.", "The file is 16 gigabytes."),
        ("It took 250 ms.", "It took 250 milliseconds."),
        ("The disk is 79% full.", "The disk is seventy-nine percent full."),
        ("It's backup-2026-11-12.zip, 16GB.", "It's the backup from the twelfth of November, 16 gigabytes."),
        ("It changed on 2026-11-12.", "It changed on the twelfth of November."),
        ("Nothing to change here.", "Nothing to change here."),
    ],
)
def test_numbers_dates_and_file_names_are_said_well(text, said):
    assert speakable_numbers(text) == said


# ── the length cap ──


@pytest.mark.asyncio
async def test_a_long_spoken_reply_stops_at_the_first_sentence_end_past_about_thirty_words():
    sentences = [f"This is sentence number {n} with a few more words in it. " for n in range(8)]
    model = Model(*sentences)
    layer = layer_with(model)
    result = await layer.take("Explain how compilers work")
    said = " ".join(layer._sink.said)
    assert result.done and said.endswith(".")
    assert len(said.split()) <= SPOKEN_WORD_CAP  # the next sentence would have gone past it, so it is not said
    assert model.closed_early  # the rest is never generated


@pytest.mark.asyncio
async def test_the_cap_never_stops_after_a_clause_cut_at_a_comma():
    long_first = "A good commit message has a short summary line under fifty characters, " * 4
    model = Model(long_first + "then a body. ", "Another sentence after it. ")
    layer = layer_with(model)
    await layer.take("What makes a good commit message?")
    assert layer._sink.said[-1].endswith(".")  # never a dangling clause


@pytest.mark.asyncio
async def test_typed_replies_keep_their_length():
    model = Model(*[f"Line {n} of a long typed answer with plenty of words in it.\n" for n in range(8)])
    layer = layer_with(model)
    chunks = []

    async def chat(text, channel, key):
        chunks.append(text)
        return "spoken"

    await layer.take("Explain it in detail", source="typed", sink=chat)
    assert "Line 7" in "".join(chunks)


# ── "plan" is talk ──


def test_planning_the_week_is_talk_not_a_task():
    assert route("I want to plan my week.").kind is TurnKind.CONVERSATION


# ── the fast layer and the planner agreeing ──


def test_promises_are_recognised():
    assert promised_action("I'll open Notepad for you.") == "open"
    assert promised_action("Let me mute that.") == "mute"
    assert promised_action("I can explain how that works.") == ""


@pytest.mark.asyncio
async def test_a_promise_without_act_hands_the_request_to_the_planner_and_is_not_said():
    layer = layer_with(Model("Sure thing. ", "I'll open the calculator for you now. "))
    result = await layer.take("I need the calculator")
    assert not result.done and result.planner_input == "I need the calculator"
    assert not any("open the calculator" in s for s in layer._sink.said)


@pytest.mark.asyncio
async def test_an_act_for_a_how_to_question_asks_first_and_a_yes_takes_it():
    act = (
        'ACT {"verb": "open", "object": "registry editor", "request": "Open the registry editor", "say": "Opening it."}'
    )
    layer = layer_with(Model(act))
    asked = await layer.take("How do I open the registry editor?")
    assert asked.done and layer._sink.said == ["Do you want me to open the registry editor?"]
    yes = await layer.take("Yes, please.")
    # The request stays the person's own words; the offer they said yes to reaches the planner as context.
    assert not yes.done and yes.planner_input == "How do I open the registry editor?"
    planner = yes.context.for_planner()
    assert "Heliox: Do you want me to open the registry editor?" in planner and "The person: Yes, please." in planner
    assert "Heliox understood the request as: Open the registry editor" in planner
    assert "Do you want me" not in yes.context.for_reviewers()


@pytest.mark.asyncio
async def test_an_offer_expires_and_anything_else_closes_it():
    now = [0.0]
    act = 'ACT {"verb": "open", "object": "x", "request": "Open the registry editor", "say": ""}'
    layer = layer_with(Replies([act, "Okay.", act, "Glad to help.", "Okay."]), clock=lambda: now[0])
    await layer.take("How do I open the registry editor?")
    now[0] += OFFER_SECONDS + 1
    late = await layer.take("yes")
    assert late.done  # too late: answered as talk, nothing handed over
    await layer.take("How do I open the registry editor?")
    await layer.take("Thanks, that's all.")
    after = await layer.take("yes")
    assert after.done  # the offer closed when the person said something else


def test_a_request_phrased_as_a_question_is_not_held_back():
    assert not ConversationLayer._only_asked("Could you open the calculator?")
    assert ConversationLayer._only_asked("How do I open the calculator?")


# ── the yes by voice: the person's request, never an approval ──


def _server_with(layer):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._voice_listener = None
    server._broadcast_notification = AsyncMock()
    server._handle_execute = AsyncMock(return_value={"status": "success", "message": "Registry Editor is open."})
    server._speak_voice_response = AsyncMock(return_value=False)
    server._spawn_interaction_speech = MagicMock()
    server._model_router = object()
    server._conversation = layer
    return server


@pytest.mark.asyncio
async def test_yes_after_an_offer_starts_the_request_through_the_normal_road():
    act = 'ACT {"verb": "open", "object": "x", "request": "Open the registry editor", "say": ""}'
    server = _server_with(layer_with(Model(act)))
    await server._voice_command_dispatch("How do I open the registry editor?")
    server._handle_execute.assert_not_awaited()
    await server._voice_command_dispatch("yes")
    server._handle_execute.assert_awaited_once()
    sent = server._handle_execute.await_args.args[0]
    assert sent["input"] == "How do I open the registry editor?"
    assert "Open the registry editor" in sent["_conversation_context"].for_planner()


@pytest.mark.asyncio
async def test_yes_with_a_plan_waiting_for_approval_is_still_refused():
    from pilot.server import PendingConfirmation

    act = 'ACT {"verb": "open", "object": "x", "request": "Open the registry editor", "say": ""}'
    server = _server_with(layer_with(Model(act)))
    await server._voice_command_dispatch("How do I open the registry editor?")
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event())
    server._pending_confirms["plan-1"] = pending
    server._active_plan_id = "plan-1"
    await server._voice_command_dispatch("yes")
    server._handle_execute.assert_not_awaited()
    assert not pending.confirmed and not pending.event.is_set()
    refusal = server._speak_voice_response.await_args.args[0]
    assert "can't approve that by voice" in refusal


def test_a_polite_gerund_names_the_action():
    decided = route("Would you mind muting the sound?")
    assert (decided.kind, decided.verb) == (TurnKind.ACTION, "mute")
    assert route("I was thinking about it").kind is TurnKind.CONVERSATION


# ── how the turn was addressed ──


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Where we store home", "What do you think about that?", "Open the other one"])
async def test_a_doubtful_follow_up_costs_nothing_says_sorry_once_and_opens_nothing(text):
    model = Model("should never be asked")
    told = []
    layer = ConversationLayer(
        sink=Sink(),
        model=lambda: model,
        addressed=lambda: "doubtful_follow_up",
        not_understood=lambda: told.append(1),
    )
    first = await layer.take(text)
    second = await layer.take(text)
    assert first.done and second.done and not first.understood and not second.understood
    assert layer._sink.said == ["Sorry?"]  # once per conversation
    assert model.calls == [] and told == [1, 1]
    assert layer.memory.recent() == []  # not kept as conversation
    layer.clear()
    await layer.take(text)
    assert layer._sink.said == ["Sorry?", "Sorry?"]  # a new conversation may say it once again


@pytest.mark.asyncio
@pytest.mark.parametrize("addressed", ["wake", "button", "follow_up", ""])
async def test_after_a_wake_phrase_or_a_confident_follow_up_an_unclear_turn_is_asked_about(addressed):
    model = Model("Sorry, I didn't catch that. Do you mean your home folder? ")
    layer = ConversationLayer(sink=Sink(), model=lambda: model, addressed=lambda: addressed, hold_after=60.0)
    result = await layer.take("Where we store home")
    assert result.understood and model.calls and layer._sink.said[-1].endswith("?")


@pytest.mark.asyncio
async def test_a_control_word_on_a_doubtful_follow_up_still_works():
    layer = ConversationLayer(sink=Sink(), model=lambda: None, addressed=lambda: "doubtful_follow_up")
    result = await layer.take("cancel")
    assert result.understood and layer._sink.said == ["Okay."]
