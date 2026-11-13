"""The day bench for offers (pilot.personal.offer_day): a simulated week for three made-up people, through the real
offer rules. The numbers are the check: a person who told Heliox nothing gets (almost) nothing."""

from __future__ import annotations

import sys

import pytest

from pilot.personal import offer_day
from pilot.personal.offers import INTRO_MAX_ASKS, MAX_PER_DAY


@pytest.fixture(scope="module")
def week():
    import asyncio

    return asyncio.run(offer_day.run_all())


def test_nobody_known_gets_almost_nothing(week):
    days = week["told Heliox nothing"]
    total = sum(len(day.offers) for day in days)
    assert total <= INTRO_MAX_ASKS  # only the getting-to-know question, at most twice, days apart
    assert all(kind == "get_to_know" for day in days for _t, kind in day.offers)
    asked_on = [i for i, day in enumerate(days) if day.offers]
    assert asked_on == [0, 3]  # Monday, then three days later, then never


def test_the_office_day(week):
    days = week["office, nine to six"]
    assert all(len(day.offers) <= MAX_PER_DAY for day in days)
    for day in days:
        times = dict((kind, t) for t, kind in day.offers)
        assert all(t < "18:00" for t, _k in day.offers)  # nothing after the end of the working day
        if "goal_progress" in times:
            assert "10:30" <= times["goal_progress"] < "10:45"  # right after the podcast request
    kinds = [kind for day in days for _t, kind in day.offers]
    assert kinds.count("goal_progress") <= 5 and "routine" in kinds
    # "don't offer that" to the quiet goal on Thursday: none on Friday.
    assert "stale_goal" not in [kind for _t, kind in days[4].offers]


def test_the_student_day(week):
    days = week["student, no fixed hours"]
    kinds = [kind for day in days for _t, kind in day.offers]
    assert "routine" not in kinds  # no working day told: no working-day offers
    assert all(len(day.offers) <= MAX_PER_DAY for day in days)
    assert "todos" not in [kind for day in days[2:] for _t, kind in day.offers]  # silenced on Tuesday


def test_the_table_prints(week, capsys):
    text = offer_day.table(week)
    assert text.count("\n") == 3 and "told Heliox nothing" in text
    print(text, file=sys.stderr)


def test_no_on_two_days_rests_an_offer_for_a_week(week):
    """The quiet goal is declined on Monday and Tuesday by both people: it is not offered again that week."""
    for name in ("office, nine to six", "student, no fixed hours"):
        days = week[name]
        offered = [i for i, day in enumerate(days) if any(kind == "stale_goal" for _t, kind in day.offers)]
        assert offered == [0, 1], (name, offered)


def test_a_declined_first_pick_does_not_hide_the_rest(week):
    days = week["student, no fixed hours"]
    assert any(kind == "habit" for _t, kind in days[0].offers)  # the habit, though the quiet goal comes first


def test_the_rest_rule_on_its_own(tmp_path):
    from datetime import datetime

    from pilot.personal.offers import Offers

    now = [datetime(2026, 11, 16, 11)]
    offers = Offers(mode=lambda: "on", profile_empty=lambda: False, now=lambda: now[0], state_path=tmp_path / "o.json")
    offers._decline("habit:water", "2026-11-16")
    assert not offers._resting("habit:water")
    offers._decline("habit:water", "2026-11-17")
    now[0] = datetime(2026, 11, 23, 11)
    assert offers._resting("habit:water")  # six days after the last "no"
    now[0] = datetime(2026, 11, 24, 11)
    assert not offers._resting("habit:water")  # a week after it
