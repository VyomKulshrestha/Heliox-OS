"""A slip in something said TO Heliox is asked about before Heliox acts on it (pilot.live.held, LiveCompanion.hold).

The four slips of the conversation bench's scenarios/slips.json, as the recogniser wrote them, are each held with one
spoken question and go on with the person's answer; the must-not-fire cases go straight on."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from pilot.live import spoken
from pilot.live.engine import LiveCompanion, Observation
from pilot.live.held import Turns, hold_for, is_answer, resolve, words_to_number

TODAY = date(2026, 11, 12)  # a Thursday


def build(tmp_path, today=TODAY):
    clock = SimpleNamespace(t=100.0)
    events, said = [], []
    settings = SimpleNamespace(enabled=True, level="balanced", listen_to_speech=True, muted_kinds=[])

    async def observe():
        return Observation(text="", app="x", title="x", source="focused_field", editable=False)

    async def emit(event, payload):
        events.append((event, payload))

    async def speak(message, urgent):
        said.append(message)

    live = LiveCompanion(
        settings=lambda: settings,
        observe=observe,
        emit=emit,
        speak=speak,
        relevance_path=tmp_path / "r.json",
        clock=lambda: clock.t,
        today=lambda: today,
    )
    return live, clock, events, said, settings


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def cards(events):
    return [p for e, p in events if e == "live_interjection"]


# ── the questions (pure) ──


@pytest.mark.parametrize(
    ("said", "question"),
    [
        (
            "put the review in for Monday the 19th at 10.",
            "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?",
        ),
        (
            # as the recogniser wrote it in the conversation bench's audio run (0db01d9): no full stop
            "put the review in for Monday the 19th at 10",
            "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?",
        ),
        (
            "remind me on Tuesday the nineteenth to call the bank",
            "The nineteenth is a Thursday. Tuesday the seventeenth, or Thursday the nineteenth?",
        ),
        (
            "remind me that the invoice is due on the thirty first of november",
            "There's no thirty-first of November. The thirtieth of November, or the first of December?",
        ),
    ],
)
def test_a_slip_in_a_request_gets_one_short_question(said, question):
    held = hold_for(spoken.to_written(said), TODAY, Turns())
    assert held is not None and held.question == question


def test_an_amount_said_a_moment_ago_against_the_one_in_the_request():
    clock = [0.0]
    turns = Turns(clock=lambda: clock[0])
    turns.add(spoken.to_written("the invoice for Acme is 2,400."))
    held = hold_for("email the Acme invoice for 4,200.", TODAY, turns)
    assert held.question == "A moment ago you said the invoice for Acme is 2,400. Is it that, or 4,200?"
    assert resolve(held, "two thousand four hundred").request == "email the Acme invoice for 2,400."
    clock[0] = 601  # a different conversation by now
    assert hold_for("email the Acme invoice for 4,200.", TODAY, turns) is None


@pytest.mark.parametrize(
    "said",
    [
        "put the review in for Thursday the 19th at 10",  # the right weekday
        "it was Monday the 19th of October",  # in the past, and right
        "email the Acme invoice for 2,400",  # the same amount
        "change the Acme invoice to 4,200",  # changed on purpose
        "the invoice for Acme is now 4,200",  # a new statement, not a request
        "the deposit is 400 and email the Acme invoice for 2,400",  # another thing's amount
        "open the review notes",
    ],
)
def test_must_not_ask(said):
    turns = Turns()
    turns.add("the invoice for Acme is 2,400.")
    assert hold_for(spoken.to_written(said), TODAY, turns) is None


@pytest.mark.parametrize(
    ("said", "asked", "question"),
    [
        (
            ["the invoice is 2,400."],
            "email it for 4,200.",
            "A moment ago you said the invoice is 2,400. Is it that, or 4,200?",
        ),
        (
            ["the invoice is 2,400."],
            "send that to Sam for 4,200.",
            "A moment ago you said the invoice is 2,400. Is it that, or 4,200?",
        ),
        (
            ["the invoice is two thousand four hundred."],
            "email it for four thousand two hundred.",
            "A moment ago you said the invoice is 2,400. Is it that, or 4,200?",
        ),
    ],
)
def test_it_or_that_points_back_to_the_last_thing_an_amount_was_stated_about(said, asked, question):
    turns = Turns()
    for text in said:
        turns.add(spoken.to_written(text))
    held = hold_for(spoken.to_written(asked), TODAY, turns)
    assert held is not None and held.question == question
    # The person's pick changes the amount only; "it" stays "it".
    assert resolve(held, "2,400").request == spoken.to_written(asked).replace(held.evidence, "2,400")


@pytest.mark.parametrize(
    ("said", "asked"),
    [
        (["the invoice for Acme is 2,400.", "the hotel is 180."], "book it for 180."),  # two things, two amounts
        (["the invoice is 2,400.", "the deposit is 400."], "pay it for 400."),
        (["the invoice is 2,400."], "pay the deposit of 400."),  # another thing, named
        (["the invoice is 2,400 and the deposit is 400."], "email it for 4,200."),  # not the same number gone wrong
        (["the invoice is 2,400."], "actually make it 4,200."),  # a change said on purpose
        (["the invoice is 2,400."], "change it to 4,200."),
        ([], "email it for 4,200."),  # nothing said before
        (["the invoice is 2,400."], "the total is 4,200 for that."),  # a statement, not a request
    ],
)
def test_pointing_back_must_not_ask(said, asked):
    turns = Turns()
    for text in said:
        turns.add(spoken.to_written(text))
    assert hold_for(spoken.to_written(asked), TODAY, turns) is None


def test_answers():
    held = hold_for("put the review in for Monday the 19th at 10.", TODAY, Turns())
    assert resolve(held, "Monday").request == "put the review in for Monday 16 November at 10."
    assert resolve(held, "the nineteenth please").request == "put the review in for Thursday 19 November at 10."
    assert resolve(held, "the second one").request == "put the review in for Thursday 19 November at 10."
    assert resolve(held, "never mind").kind == "cancel"
    assert resolve(held, "pizza").say == "Sorry, Monday the sixteenth or Thursday the nineteenth?"
    assert resolve(held, "pizza").say == "I've left it. Say it again when you're ready."
    assert words_to_number("four thousand two hundred and fifty") == 4250


# ── the turn, held and answered ──


@pytest.mark.asyncio
async def test_the_turn_waits_for_the_answer_and_goes_on_with_it(tmp_path):
    live, clock, events, said, _ = build(tmp_path)
    first = live.hold("Put the review in for Monday the 19th at 10.")
    assert first.say == "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?"
    assert first.proceed is None
    await settle()
    [card] = cards(events)
    assert card["spoken"] is True and card["title"] == "The nineteenth of November is a Thursday"
    assert said == []  # the question is the turn's answer, said by the hook; Live does not say it again
    answer = live.hold("Thursday")
    assert answer.say == "" and answer.proceed == "Put the review in for Thursday 19 November at 10."
    assert live.hold("what time is it") is None  # nothing held any more


@pytest.mark.asyncio
async def test_an_amount_swap_across_two_turns_is_asked_about(tmp_path):
    live, clock, events, said, _ = build(tmp_path)
    assert live.hold("the invoice for Acme is 2,400.") is None
    clock.t += 15
    held = live.hold("email the Acme invoice for 4,200.")
    assert held.say == "A moment ago you said the invoice for Acme is 2,400. Is it that, or 4,200?"
    assert live.hold("2,400").proceed == "email the Acme invoice for 2,400."


@pytest.mark.asyncio
async def test_a_hold_lapses_and_never_mind_drops_it(tmp_path):
    live, clock, _, _, _ = build(tmp_path)
    live.hold("remind me on Tuesday the nineteenth to call the bank")
    clock.t += 61
    assert live.hold("open my mail") is None  # long after: a new request, not an answer
    live.hold("remind me on Tuesday the nineteenth to call the bank")
    assert live.hold("never mind").say == "Okay, I've left it." and live.hold("open my mail") is None


@pytest.mark.asyncio
async def test_with_live_off_or_not_listening_nothing_is_held(tmp_path):
    live, _, events, _, settings = build(tmp_path)
    settings.listen_to_speech = False
    assert live.hold("put the review in for Monday the 19th at 10.") is None
    settings.listen_to_speech, settings.enabled = True, False
    assert live.hold("put the review in for Monday the 19th at 10.") is None
    settings.enabled, settings.muted_kinds = True, ["weekday_date"]  # turned off in Settings: not asked about
    assert live.hold("put the review in for Monday the 19th at 10.") is None
    await settle()
    assert cards(events) == []


@pytest.mark.asyncio
async def test_the_bench_slips_four_of_four_and_the_colleague_case_silent(tmp_path):
    """scenarios/slips.json as the recogniser wrote it: each slip gets a card before anything acts on it."""
    from pilot.personal.wiring import live_hold

    live, clock, events, _, _ = build(tmp_path, today=date(2026, 11, 12))
    server = SimpleNamespace(_live=live)
    caught = 0
    for turns in (
        ["remind me on Tuesday the 19th to call the bank."],
        ["put the review in for Monday the 19th at 10."],
        ["remind me that the invoice is due on the 31st of November."],
        ["the invoice for Acme is 2,400.", "email the Acme invoice for 4,200."],
    ):
        before = len(cards(events))
        outcomes = []
        for text in turns:
            outcomes.append(await live_hold(server, text))
            clock.t += 10
        await settle()
        held = outcomes[-1]
        if held is not None and held.proceed is None and len(cards(events)) > before:
            caught += 1
        live.drop_held()
    assert caught == 4
    live.offer_speech("the invoice is due on the thirty first of November", addressed=False)  # said to a colleague
    await settle()
    assert len(cards(events)) == 4


# ── a new request while a question is pending is not its answer ──


@pytest.mark.parametrize(
    "said",
    ["Thursday", "the nineteenth please", "the second one", "never mind", "2,400", "two thousand four hundred",
     "oh sorry, Thursday the nineteenth please", "make it Monday", "pizza"],
)  # fmt: skip
def test_answers_are_answers(said):
    assert is_answer(said)


@pytest.mark.parametrize(
    "said",
    ["put the review in for Monday the 19th at 10", "remind me to call the bank tomorrow",
     "Hey Heliox, what's on my calendar", "send the deck to Sam on Thursday the nineteenth please and copy Ana"],
)  # fmt: skip
def test_a_new_request_is_not_an_answer(said):
    assert not is_answer(said)


@pytest.mark.asyncio
async def test_the_bench_order_a_slip_then_another_request_naming_an_option(tmp_path):
    """The audio run (0db01d9): the weekday slip was held, and the next scenario's request, which names "the 19th",
    was taken as the answer, so the report reminder went on and the meeting got no card. Now the new request is its
    own turn: the old question is dropped (nothing acts on it) and the new slip gets its own question."""
    live, clock, events, said, _ = build(tmp_path)
    first = live.hold("remind me on Tuesday the 19th to send the report.")
    assert first.proceed is None
    clock.t += 20  # the next scenario, well inside the 60 s a question waits
    second = live.hold("put the review in for Monday the 19th at 10")
    assert second is not None and second.proceed is None
    assert second.say == "The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?"
    await settle()
    first_card, second_card = cards(events)
    withdrawn = [p for e, p in events if e == "live_interjection_withdrawn"]
    assert withdrawn == [{"id": first_card["id"], "reason": "expired"}]
    assert live.hold("Thursday").proceed == "put the review in for Thursday 19 November at 10"


# ── answering "Is it that, or 4,200?" ──


@pytest.mark.parametrize(
    ("answer", "kind", "goes_on"),
    [
        ("that", "proceed", "email the Acme invoice for 2,400."),
        ("yes", "proceed", "email the Acme invoice for 2,400."),
        ("yes, that one", "proceed", "email the Acme invoice for 2,400."),
        ("no, 4,200", "proceed", "email the Acme invoice for 4,200."),  # the version named wins over the "no"
        ("no, the second", "proceed", "email the Acme invoice for 4,200."),
        ("four thousand two hundred", "proceed", "email the Acme invoice for 4,200."),
        ("no", "cancel", ""),
        ("never mind, 4,200", "cancel", ""),  # words that drop the question drop it
    ],
)
def test_answers_to_the_amount_question(answer, kind, goes_on):
    turns = Turns()
    turns.add("the invoice for Acme is 2,400.")
    held = hold_for("email the Acme invoice for 4,200.", TODAY, turns)
    assert held.question == "A moment ago you said the invoice for Acme is 2,400. Is it that, or 4,200?"
    result = resolve(held, answer)
    assert result.kind == kind and result.request == goes_on


def test_no_question_says_a_value_twice_in_a_row():
    """Once numbers are spoken as words, each question still names each value once and starts no sentence with one."""
    turns = Turns()
    turns.add("the invoice for Acme is 2,400.")
    questions = [
        hold_for("email the Acme invoice for 4,200.", TODAY, turns).question,
        hold_for("put the review in for Monday the 19th at 10", TODAY, Turns()).question,
        hold_for("remind me that the invoice is due on the 31st of November", TODAY, Turns()).question,
    ]
    for question in questions:
        sentences = [part.strip() for part in question.replace("?", ".").split(".") if part.strip()]
        assert all(not sentence[0].isdigit() for sentence in sentences), question
        assert question.count("2,400") <= 1 and question.count("4,200") <= 1, question
    assert resolve(hold_for("put the review in for Monday the 19th at 10", TODAY, Turns()), "no, Thursday").request == (
        "put the review in for Thursday 19 November at 10"
    )
