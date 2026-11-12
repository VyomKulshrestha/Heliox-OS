"""Corrections, quiet unless asked, recall in a person's words, and the end of the working day (personal rules)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from pilot.conversation.layer import SPOKEN_WORD_CAP
from pilot.personal.memory_talk import MemoryTalk
from pilot.personal.wiring import held_for_quiet, quiet_moment, quiet_now
from pilot.workflows.user_profile import QUIET_PREF, UserProfile, parse_work_end


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def profile(tmp_path, clock):
    return UserProfile(tmp_path / "profile.json", clock=clock)


# ── 2. a correction replaces the old fact ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "said",
    ["no, call me Sam", "No call me Sam.", "actually, call me Sam", "sorry, my name is Sam", "not Alex, Sam",
     "not Alex, call me Sam", "it's Sam"],
)  # fmt: skip
async def test_a_correction_replaces_the_name(profile, said):
    assert await profile.answer_async("call me Alex") == "Nice to meet you, Alex. I'll call you that."
    reply = await profile.answer_async(said)
    assert reply == "Sorry about that, Sam. I'll call you Sam from now on."
    assert profile.name() == "Sam" and "Alex" not in " ".join(profile.context_lines())
    assert await profile.answer_async("what's my name") == "I call you Sam."


@pytest.mark.asyncio
async def test_its_a_name_only_right_after_one_was_set(profile, clock):
    await profile.answer_async("call me Alex")
    clock.t += 3600
    assert await profile.answer_async("it's Sam") is None  # an hour later "it's Sam" is not about the name
    assert profile.name() == "Alex"
    assert await profile.answer_async("not Bob, Sam") is None  # Bob was never the name


# ── 3. quiet unless asked ──


@pytest.mark.asyncio
async def test_quiet_unless_asked_said_on_its_own_and_undone(profile):
    reply = await profile.answer_async("stay quiet unless I ask")
    assert reply.startswith("Okay, I'll stay quiet unless you ask: no offers")
    assert len(reply.split()) <= SPOKEN_WORD_CAP  # said whole, never cut off
    assert profile.quiet() and any(QUIET_PREF in line for line in profile.context_lines())
    server = SimpleNamespace(_user_profile=profile, _focus=None)
    assert quiet_now(server) == "asked" and held_for_quiet(server) and not quiet_moment(server)
    assert (await profile.answer_async("you can speak up again")).startswith("Okay, I'll speak up again")
    assert not profile.quiet() and quiet_now(server) == ""
    assert await profile.answer_async("you can speak up again") is None  # nothing to undo: not a phrase now


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["be quiet", "don't interrupt me", "only speak when I ask", "keep quiet please"])
async def test_ways_of_saying_quiet(profile, said):
    assert (await profile.answer_async(said)).startswith("Okay, I'll stay quiet")


@pytest.mark.asyncio
async def test_live_holds_its_non_urgent_catches_while_quiet(tmp_path, profile):
    """Live's focus_active is held_for_quiet: non-urgent catches wait, urgent ones (a risky command) still go."""
    from pilot.live.detectors import Correction
    from pilot.live.policy import LEVELS, SPEAK, WAIT, Activity, Pending, decide

    await profile.answer_async("stay quiet unless I ask")
    act = Activity(now=100.0, last_text_change=0.0, focus_active=held_for_quiet(SimpleNamespace(_user_profile=profile)))
    slip = Pending(
        Correction(kind="weekday_date", key="k", title="t", message="m", evidence="e"), "focused_field", 0, 0
    )
    risky = Pending(Correction(kind="risky_command", key="r", title="t", message="m", evidence="e", urgency="urgent"),
                    "focused_field", 0, 0)  # fmt: skip
    assert decide(slip, act, LEVELS["balanced"], 1.0, frozenset()) == WAIT
    assert decide(risky, act, LEVELS["balanced"], 1.0, frozenset()) == SPEAK


# ── 4. recall in a person's own words ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("told", "asked", "has"),
    [
        ("remember that I stop work at six", "when do I stop work?", "six"),
        ("remember that I live in Pune", "where do I live?", "Pune"),
        ("my manager is Priya", "who is my manager?", "Priya"),
        ("I prefer short emails", "what do I prefer?", "short emails"),
        ("remember that the stand-up is at ten on Mondays", "when is the stand-up?", None),  # not "my": the model's
    ],
)
async def test_recall_in_the_persons_words(profile, told, asked, has):
    talk = MemoryTalk(profile=lambda: profile, now=lambda: datetime(2026, 11, 12, 10))
    assert await profile.answer_async(told) or await talk.answer(told)
    reply = await talk.answer(asked)
    if has is None:
        assert reply is None
    else:
        assert reply and has in reply, reply


@pytest.mark.asyncio
async def test_a_question_about_something_never_told_goes_on(profile):
    talk = MemoryTalk(profile=lambda: profile, now=lambda: datetime(2026, 11, 12, 10))
    assert await talk.answer("when is my meeting?") is None  # the planner's or the model's, not "you told me nothing"


# ── 5. the end of the working day ──


@pytest.mark.parametrize(
    ("said", "minute"),
    [("I stop work at six", 18 * 60), ("I finish work at 5:30 pm", 17 * 60 + 30), ("my working day ends at 17:00", 17 * 60),
     ("I leave work at 4", 16 * 60), ("I stop work at noon-ish", None)],
)  # fmt: skip
def test_parse_work_end(said, minute):
    assert parse_work_end(said) == minute


@pytest.mark.asyncio
async def test_after_work_nothing_is_offered_and_it_says_so_once(profile):
    reply = await profile.answer_async("I stop work at six")
    assert reply.startswith("Noted: you stop work at 18:00.")
    assert profile.work_end_minute() == 18 * 60
    said = []

    async def broadcast(method, params):
        said.append(params["message"])

    server = SimpleNamespace(_user_profile=profile, _focus=None, _broadcast_notification=broadcast)
    assert quiet_now(server, datetime(2026, 11, 12, 17, 59)) == ""
    assert quiet_now(server, datetime(2026, 11, 12, 18, 0)) == "after_work"
    assert quiet_now(server, datetime(2026, 11, 12, 22, 0)) == "after_work"
    assert quiet_now(server, datetime(2026, 11, 13, 2, 0)) == "after_work"  # the same evening
    assert quiet_now(server, datetime(2026, 11, 13, 9, 0)) == ""  # morning: offers again
    import asyncio

    await asyncio.sleep(0)
    assert len(said) == 1 and said[0].startswith("It's after 18:00, when you stop work")
    assert quiet_now(server, datetime(2026, 11, 13, 18, 30)) == "after_work"
    await asyncio.sleep(0)
    assert len(said) == 2  # once a day


@pytest.mark.asyncio
async def test_a_later_time_replaces_the_earlier(profile):
    await profile.answer_async("I stop work at six")
    await profile.answer_async("remember that I finish work at 5 pm")
    assert profile.work_end_minute() == 17 * 60
