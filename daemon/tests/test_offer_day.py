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
    assert text.count("\n") == len(offer_day.PEOPLE) and "told Heliox nothing" in text
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


def test_the_person_who_never_answers_hears_little(week):
    """The office person who never answers: Monday and Tuesday's six offers go unanswered, then nothing for the rest
    of the week (all offers rest); the office person who answers is offered the working day every weekday."""
    days = week["office, never answers"]
    assert [len(day.offers) for day in days] == [3, 3, 0, 0, 0], [day.offers for day in days]
    answered = week["office, nine to six"]
    assert all(any(kind == "routine" for _t, kind in day.offers) for day in answered)


def test_three_weeks_of_nobody_answering(tmp_path):
    """Three weeks of the person who never answers: a handful of offers (45 before the rules about ignored offers;
    the four who answer heard as many over three weeks with and without them, `python -m pilot.personal.offer_day 3`)."""
    import asyncio

    person = next(p for p in offer_day.PEOPLE if p.name == "office, never answers")
    days = asyncio.run(offer_day.run_week(person, tmp_path, 3))
    assert [len(day.offers) for day in days] == [3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [d.offers for d in days]


def test_the_ignore_rule_on_its_own(tmp_path):
    from datetime import datetime

    from pilot.personal.offers import EXPLAIN_WINDOW_SECONDS, Offers, day_moments

    now = [datetime(2026, 11, 16, 9, 0)]
    tick = [0.0]
    path = tmp_path / "o.json"

    def offers_now() -> Offers:
        return Offers(
            mode=lambda: "on",
            profile_empty=lambda: False,
            moments=lambda at: day_moments(at, start=9 * 60, end=18 * 60),
            now=lambda: now[0],
            clock=lambda: tick[0],
            state_path=path,
        )

    offers = offers_now()

    def a_day(day: int, *, reply: str = "") -> bool:
        now[0] = datetime(2026, 11, 16 + day, 9, 0)
        tick[0] = day * 86400.0
        offer = offers.next_offer()
        if offer is None or offer.key != "routine:day_start":
            return False
        offers.mark_offered(offer)
        if reply:
            offers.reply(reply)
        tick[0] += EXPLAIN_WINDOW_SECONDS + 1
        offers.room()  # the next look notes it went unanswered
        return True

    assert a_day(0) and a_day(1) and a_day(2, reply="why")  # "why?" is a reply: the count starts again
    assert a_day(3)  # Thursday
    offers = offers_now()  # a restart keeps the count (dates only, in offers.json)
    assert a_day(4) and a_day(7)  # Friday, then Monday: the third day in a row with no reply
    assert not a_day(8) and not a_day(11)  # rests for a week after that Monday
    assert a_day(14)  # and comes back


def test_offers_go_quiet_when_nobody_answers(tmp_path):
    """Six offers in a row with no reply: none for a week, then one a day until one is answered; any answer, or
    "you can suggest things again", brings the normal cap back. A restart keeps it."""
    from datetime import datetime, timedelta

    from pilot.personal.offers import EXPLAIN_WINDOW_SECONDS, MIN_GAP_SECONDS, Offer, Offers

    now = [datetime(2026, 11, 16, 9, 0)]
    tick = [0.0]
    path = tmp_path / "o.json"
    many = [Offer(key=f"other:{n}", kind="habit", text=f"other {n}", request="", why="") for n in range(30)]

    def offers_now() -> Offers:
        return Offers(mode=lambda: "on", profile_empty=lambda: False, moments=lambda at: many, now=lambda: now[0],
                      clock=lambda: tick[0], state_path=path)  # fmt: skip

    offers = offers_now()

    def a_day(day: int, *, answer: str = "") -> int:
        """Looks every hour from 9:00 to 17:00 on ``day``; returns how many offers were made."""
        made = 0
        for hour in range(9, 18):
            now[0] = datetime(2026, 11, 16, hour) + timedelta(days=day)
            tick[0] = day * 86400.0 + hour * 3600.0
            offer = offers.next_offer()
            if offer is not None:
                offers.mark_offered(offer)
                made += 1
                if answer:
                    offers.reply(answer)
            tick[0] += max(EXPLAIN_WINDOW_SECONDS, MIN_GAP_SECONDS) + 1
            offers.room()
        return made

    assert a_day(0) == 3 and a_day(1) == 3  # six with no reply
    assert a_day(2) == 0 and a_day(6) == 0  # quiet for a week from Tuesday
    offers = offers_now()  # a restart keeps it
    assert a_day(8) == 1 and a_day(9) == 1  # then one a day
    assert a_day(10, answer="no") == 3  # any answer brings the cap back
    assert a_day(11) == 3 and a_day(12) == 3 and a_day(13) == 0  # quiet again after six more with no reply
    offers.welcome_back()  # "you can suggest things again"
    assert a_day(14) == 3


def test_the_timeline_is_offered_once_after_three_days_of_use(week):
    used = week["a week of use, nothing told"]
    offered = [i for i, day in enumerate(used) if any(kind == "timeline" for _t, kind in day.offers)]
    assert offered == [2]  # Wednesday, the third day with a request, and never again
    office = [kind for day in week["office, nine to six"] for _t, kind in day.offers]
    assert office.count("timeline") == 1
    for name in ("student, no fixed hours", "told Heliox nothing"):  # no requests taken: no days of use
        assert not any(kind == "timeline" for day in week[name] for _t, kind in day.offers)
