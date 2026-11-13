"""What Heliox knows about the person reaching what it says and offers: context lines, memory talk, offers.

Everything runs on files in a temporary folder with injected clocks: no model, network, audio or timeline database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from pilot.personal.context import MAX_CHARS, PersonalContext, as_context, relevant_lines
from pilot.personal.memory_talk import MemoryTalk
from pilot.personal.offers import MAX_PER_DAY, MIN_GAP_SECONDS, OfferLoop, Offers, from_suggestion, usual_apps
from pilot.workflows.user_profile import UserProfile


@pytest.fixture
def profile(tmp_path):
    return UserProfile(tmp_path / "user_profile.json")


# ── context lines for one utterance ──


@pytest.mark.asyncio
async def test_relevant_lines_bring_the_person_the_relation_names_and_skip_what_is_already_sent(profile):
    for i in range(12):
        await profile.answer_async(f"remember that I own plant number {i}")
    await profile.answer_async("my sister is Ana")
    await profile.answer_async("remember that I am allergic to peanuts")
    await profile.answer_async("I prefer short answers")
    data = profile.get()
    lines = relevant_lines(
        "What should I cook for my sister on Friday? Nothing with peanuts.",
        profile=data,
        already=list(profile.context_lines())[:8],
    )
    assert "Person they know: Ana (sister)" in lines
    assert "About them: I am allergic to peanuts" in lines
    assert "Preference: prefer short answers" in lines
    assert not any("plant number" in line for line in lines)  # shares no word with the question
    # Already among the lines the layer sends: not repeated.
    again = relevant_lines("plant number 0", profile=data, already=["About them: I own plant number 0"])
    assert "About them: I own plant number 0" not in again and "About them: I own plant number 1" in again


def test_goals_and_habits_are_found_by_their_words_and_the_line_fits_the_budget():
    goals = [
        {"title": "Run a half marathon", "next_step": "buy running shoes", "status": "active"},
        {"title": "Learn the cello", "next_step": "", "status": "done"},
    ]
    habits = [{"name": "reading", "streak": 4}]
    lines = relevant_lines("where can I buy shoes for running", profile={}, goals=goals, habits=habits)
    assert lines == ["Their goal: Run a half marathon, next step: buy running shoes"]
    assert relevant_lines("how is my reading streak", profile={}, habits=habits) == [
        "Habit they track: reading, 4-day streak"
    ]
    long = [f"About them: {'word ' * 30}{i}" for i in range(10)]
    block = as_context(long)
    assert len(block) <= MAX_CHARS and block.startswith("Also relevant about the person:")


@pytest.mark.asyncio
async def test_personal_context_uses_the_words_given_or_the_noted_utterance(profile):
    clock = [0.0]
    await profile.answer_async("my dentist is Dr Rao")
    for i in range(9):  # the dentist is no longer among the eight lines the layer always sends
        await profile.answer_async(f"remember that I own plant number {i}")
    context = PersonalContext(profile=lambda: profile, clock=lambda: clock[0])
    assert context.extra() == ""  # nothing noted yet
    assert "Dr Rao" in context.extra("book my dentist for next week")
    context.note("when does my dentist open")
    assert "Dr Rao" in context.extra()
    clock[0] = 31.0
    assert context.extra() == ""  # a stale note says nothing


# ── remember, recall, forget ──


@pytest.mark.asyncio
async def test_remember_facts_that_do_not_start_with_i_and_refuse_what_is_not_a_fact(profile):
    talk = MemoryTalk(profile=lambda: profile)
    assert await talk.answer("Remember that the stand-up is at ten on Mondays") == (
        'Remembered: "the stand-up is at ten on Mondays".'
    )
    assert await talk.answer("remember my car is a blue Corolla") == "Okay, your car is a blue Corolla."
    for not_a_fact in (
        "remember to buy milk",
        "remember when we went to Goa",
        "remember that",
        "remember that thing we talked about",
        "open the notes app",
    ):
        assert await talk.answer(not_a_fact) is None, not_a_fact
    refused = await talk.answer("remember that the wifi password is hunter2-Xq9!vL3")
    assert refused.startswith("That looks like a password or key")
    assert "hunter2" not in (profile._path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_recall_and_forget_say_exactly_what_was_found_or_removed(profile):
    talk = MemoryTalk(
        profile=lambda: profile, goals=lambda: [{"title": "Sell the old car", "next_step": "", "status": "active"}]
    )
    await talk.answer("remember my car is a blue Corolla")
    await profile.answer_async("my sister is Ana")
    assert await talk.answer("What did I tell you about my car?") == (
        'You told me your car is a blue Corolla; you have a goal "Sell the old car".'
    )
    assert await talk.answer("what do you know about Ana") == "You told me Ana is your sister."
    assert await talk.answer("what do you know about me") is None  # the profile's own reply
    assert await talk.answer("what did I tell you about the garden") == "You haven't told me anything about the garden."
    assert await talk.answer("forget what I just told you") == 'Okay, I\'ve forgotten "Ana (sister)".'
    assert profile.get()["people"] == []
    assert await talk.answer("forget that the boiler is old") == (
        "I don't have anything stored about the boiler is old, so there was nothing to forget."
    )
    assert await talk.answer("forget it") is None  # too vague to claim


@pytest.mark.asyncio
async def test_forgetting_erases_the_words_from_long_term_memory(tmp_path):
    import sqlite3

    from pilot.memory.store import MemoryStore
    from pilot.memory.temporal import ERASED_VALUE, MemoryProvenance, MemoryScope

    db_file = tmp_path / "memory.db"
    memory = MemoryStore(
        checkpoint_interval_seconds=0, pruning_interval_seconds=0, db_file=db_file, semantic_search=False
    )
    await memory.initialize()

    async def remember(predicate, value):
        fact = await memory.remember_fact(
            subject="user",
            predicate=predicate,
            value=value,
            scope=MemoryScope.USER,
            confidence=1.0,
            provenance=MemoryProvenance.EXPLICIT_USER,
        )
        return fact.fact_id

    async def forget(fact_id):
        await memory.erase_fact(fact_id, reason="Removed from your profile")

    profile = UserProfile(tmp_path / "p.json", remember=remember, forget=forget)
    try:
        await profile.answer_async("remember that I am seeing a specialist about my knee")
        assert (await profile.answer_async("forget that I am seeing a specialist")).startswith("Okay, I've forgotten")
    finally:
        await memory.close()
    db = sqlite3.connect(tmp_path / "temporal_memory.db")
    try:
        values = [row[0] for row in db.execute("SELECT value_json FROM temporal_facts")]
        statuses = [row[0] for row in db.execute("SELECT status FROM temporal_facts")]
    finally:
        db.close()
    assert not any("knee" in value for value in values)
    assert ERASED_VALUE in values[0] and statuses == ["retracted"]


@pytest.mark.asyncio
async def test_the_conversational_memory_bench_passes_every_case():
    from pilot.personal import bench

    result = await bench.run()
    assert result["passed"] == result["cases"] >= 7, [row for row in result["rows"] if not row["pass"]]


# ── offers ──


def _spans(days: list[datetime], app: str, hour: int, minute: int) -> list[dict]:
    out = []
    for day in days:
        start = day.replace(hour=hour, minute=minute).timestamp()
        out.append({"start": start, "end": start + 600, "app": app, "title": "private title"})
    return out


def _weekdays_before(now: datetime, count: int) -> list[datetime]:
    days, day = [], now - timedelta(days=1)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days


def test_usual_app_offer_needs_the_habit_on_most_recent_days_and_not_today():
    now = datetime(2026, 11, 12, 9, 5)  # a Thursday
    days = _weekdays_before(now, 5)
    spans = _spans(days[:4], "outlook.exe", 8, 55) + _spans(days, "code.exe", 14, 0)
    [offer] = usual_apps(spans, now, label=lambda app: {"outlook.exe": "Outlook"}.get(app, app))
    assert offer.text == "You usually open Outlook about now. Want me to open it?"
    assert offer.request == "open Outlook"
    assert offer.why == (
        "On 4 of your last 5 weekdays you first opened Outlook between 8:55 and 8:55, and you haven't opened it yet "
        "today."
    )
    assert "private title" not in offer.text + offer.why  # window titles are never said
    # Already opened today: no offer. Two days out of five: no habit. Too little history: nothing is usual.
    today = _spans([now], "outlook.exe", 8, 30)
    assert usual_apps(spans + today, now) == []
    assert usual_apps(_spans(days[:2], "outlook.exe", 8, 55) + _spans(days, "x.exe", 1, 0), now) == []
    assert usual_apps(_spans(days[:3], "outlook.exe", 8, 55), now) == []
    # Weekend days are their own pattern.
    saturday = datetime(2026, 11, 14, 9, 5)
    assert usual_apps(spans, saturday) == []


def test_offers_from_suggestions_carry_their_reason():
    habit = from_suggestion({"kind": "habit", "text": "You're on a 3-day streak for reading.", "say": "log reading"})
    assert habit.request == "log reading" and habit.key == "habit:reading" and "Want me to log it?" in habit.text
    assert from_suggestion({"kind": "goal_step", "text": "x", "say": None}) is None


def _offers(tmp_path, **overrides):
    state = {"t": 10_000.0, "now": datetime(2026, 11, 12, 18, 0)}
    picks = [{"kind": "habit", "key": "habit:reading", "text": "Streak.", "say": "log reading"}]
    kwargs = dict(
        mode=lambda: "auto",
        profile_empty=lambda: False,
        suggestion=lambda: picks[0] if picks else None,
        clock=lambda: state["t"],
        now=lambda: state["now"],
        state_path=tmp_path / "offers.json",
    )
    kwargs.update(overrides)
    return Offers(**kwargs), state, picks


def test_default_is_on_only_with_a_profile_and_never_doubles_the_old_suggestions(tmp_path):
    offers, _, _ = _offers(tmp_path, profile_empty=lambda: True)
    # An empty profile gets only the one getting-to-know offer (pilot.personal.getting_to_know).
    assert not offers.active() and offers.next_offer().key == "get_to_know"
    offers, _, _ = _offers(tmp_path, profile_empty=lambda: True, mode=lambda: "on")
    assert offers.active()
    offers, _, _ = _offers(tmp_path, mode=lambda: "off")
    assert not offers.active()
    offers, _, _ = _offers(tmp_path, legacy_suggestions_on=lambda: True)
    assert offers.next_offer() is None  # the background suggestions already offer this one


def test_caps_gap_and_once_a_day(tmp_path):
    offers, state, picks = _offers(tmp_path)
    first = offers.next_offer()
    offers.mark_offered(first)
    assert offers.next_offer() is None  # the gap
    state["t"] += 3600
    assert offers.next_offer() is None  # the same thing at most once a day
    picks[0] = {"kind": "todos", "key": "todos", "text": "Two open.", "say": "what's on my list"}
    offer = offers.next_offer()
    assert offer is not None and offer.key == "todos"
    for i in range(MAX_PER_DAY):
        offers.mark_offered(offer)
        state["t"] += max(MIN_GAP_SECONDS, 3600) + 1
        picks[0] = {"kind": "todos", "key": f"todos-{i}", "text": "x", "say": "what's on my list"}
    picks[0] = {"kind": "habit", "key": "habit:yoga", "text": "x", "say": "log yoga"}
    assert offers.next_offer() is None  # the daily cap


def test_replies_yes_no_why_and_silence_for_good(tmp_path):
    offers, state, _ = _offers(tmp_path)
    offer = offers.next_offer()
    offers.mark_offered(offer)
    assert offers.reply("why did you offer that").reply == offer.why
    yes = offers.reply("yes please")
    assert yes.request == "log reading" and yes.reply == "Okay."
    assert offers.reply("yes") is None  # answered already
    assert offers.reply("don't offer that again").reply == "Okay, I won't offer that again."
    again, state2, _ = _offers(tmp_path)  # a new run reads the silenced offer back
    state2["t"] += 999_999
    assert again.next_offer() is None
    assert again.reply("stop offering things") is None  # a setting, said any time (pilot.personal.spoken_settings)
    # "No" holds it for the rest of the day.
    offers, state, picks = _offers(tmp_path / "other")
    offers.mark_offered(offers.next_offer())
    assert offers.reply("no thanks").reply == "Okay, not today."
    state["t"] += 99_999
    offers.made.clear()
    assert offers.next_offer() is None
    # Long after the offer, "yes" is not about it.
    offers, state, _ = _offers(tmp_path / "late")
    offers.mark_offered(offers.next_offer())
    state["t"] += 200
    assert offers.reply("yes") is None


@pytest.mark.asyncio
async def test_the_loop_offers_only_at_a_quiet_moment(tmp_path):
    offers, _, _ = _offers(tmp_path)
    quiet = [False]
    deliver = AsyncMock()
    loop = OfferLoop(offers, quiet_moment=lambda: quiet[0], deliver=deliver)
    assert await loop.tick() is None and not deliver.await_count
    quiet[0] = True
    offer = await loop.tick()
    assert offer is not None and deliver.await_args.args == (offer, offers.made[-1].id)


def test_card_buttons_answer_the_offer_they_belong_to_and_silenced_offers_can_be_undone(tmp_path):
    offers, _, _ = _offers(tmp_path)
    offer = offers.next_offer()
    offer_id = offers.mark_offered(offer)
    assert offers.answer(offer_id, "why").reply == offer.why
    assert offers.answer("nope", "yes") is None
    assert offers.answer(offer_id, "yes").request == "log reading"
    assert offers.answer(offer_id, "yes").reply == "That one was already answered."
    assert offers.answer(offer_id, "silence").reply == "Okay, I won't offer that again."
    status = offers.status()
    assert status["silenced"] == [{"key": "habit:reading", "label": "Logging reading"}]
    assert status["mode"] == "auto" and status["active"] and not status["waiting_for_profile"]
    assert offers.unsilence("habit:reading") and not offers.unsilence("habit:reading")
    again, _, _ = _offers(tmp_path)
    assert again.silenced == {}  # undoing it is kept too


def test_the_first_offers_file_with_a_list_of_keys_still_loads(tmp_path):
    (tmp_path / "offers.json").write_text('{"silenced": ["todos"]}', encoding="utf-8")
    offers, _, _ = _offers(tmp_path)
    assert offers.silenced == {"todos": "todos"}


@pytest.mark.asyncio
async def test_the_offers_rpc(tmp_path):
    from unittest.mock import MagicMock

    from pilot.config import PilotConfig
    from pilot.personal.offers import Offer
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._voice_turn = AsyncMock()
    status = await server._handle_offers({"action": "status"}, None)
    assert status["status"] == "ok" and status["mode"] == "auto" and status["waiting_for_profile"]
    assert (await server._handle_offers({"action": "set", "mode": "loud"}, None))["status"] == "error"
    assert (await server._handle_offers({"action": "set", "mode": "on"}, None))["active"] is True
    assert server.config.suggestions.offers == "on"
    offer = Offer("usual_app:outlook.exe", "usual_app", "Want me to open Outlook?", "open Outlook", "Usually.")
    offer_id = server._offers.mark_offered(offer)
    answered = await server._handle_offers({"action": "answer", "id": offer_id, "answer": "yes"}, None)
    assert answered["reply"] == "Okay."
    import asyncio

    await asyncio.sleep(0)
    server._voice_turn.assert_awaited_with("open Outlook")
    server._broadcast_notification.assert_awaited_with("offer_withdrawn", {"id": offer_id, "reason": "yes"})
    silenced = await server._handle_offers({"action": "answer", "id": offer_id, "answer": "silence"}, None)
    assert silenced["silenced"] == [{"key": "usual_app:outlook.exe", "label": "Opening Outlook at the usual time"}]
    undone = await server._handle_offers({"action": "unsilence", "key": "usual_app:outlook.exe"}, None)
    assert undone["silenced"] == []


def test_with_a_frozen_clock_answers_go_to_the_latest_offer(tmp_path):
    offers, state, picks = _offers(tmp_path)
    first = offers.next_offer()
    offers.mark_offered(first)
    offers.made[-1].at -= MIN_GAP_SECONDS  # let a second offer through without moving the clock
    picks[0] = {"kind": "todos", "key": "todos", "text": "Two open.", "say": "what's on my list"}
    offers.room = lambda **_: True
    second = offers.next_offer()
    offers.mark_offered(second)
    assert offers.made[-1].at == offers.made[-2].at + MIN_GAP_SECONDS  # the clock did not move
    assert offers.last_offered_seq() == offers.made[-1].seq > offers.made[-2].seq
    assert offers.reply("yes").request == "what's on my list"
