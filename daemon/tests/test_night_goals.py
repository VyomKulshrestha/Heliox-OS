"""The night summary says, in one clause, which goals the day moved, and says nothing about goals on a day that
moved none (pilot.workflows.day_briefing.goals_moved_today, compose_night)."""

from __future__ import annotations

from datetime import datetime

from pilot.personal.offers import Offers, goal_progress
from pilot.workflows.day_briefing import compose_night, goals_moved_today

NIGHT = datetime(2026, 11, 12, 22, 0)
MIDNIGHT = NIGHT.replace(hour=0).timestamp()
PODCAST = {"id": "g1", "title": "launch the podcast", "status": "active", "updates": []}
THESIS = {"id": "g2", "title": "finish the thesis", "status": "active", "updates": []}


def night(moved=()) -> str:
    return compose_night(NIGHT, brief="", focus_seconds=0, finished_todos=0, open_todos=0, reminders=[],
                         goals_moved=moved)  # fmt: skip


def test_a_day_with_no_progress_says_nothing_about_goals():
    assert night() == "Good night."
    assert goals_moved_today([PODCAST, THESIS], since=MIDNIGHT) == []
    old = {**PODCAST, "updates": [{"at": MIDNIGHT - 3600, "text": "booked a guest"}]}
    assert goals_moved_today([old], since=MIDNIGHT) == []  # yesterday's progress is not today's


def test_progress_noted_today_is_said_in_one_clause():
    noted = {**PODCAST, "updates": [{"at": MIDNIGHT + 3600, "text": "booked a guest"}]}
    moved = goals_moved_today([noted, THESIS], since=MIDNIGHT)
    assert moved == ["launch the podcast"]
    assert night(moved) == 'Good night. You made progress on "launch the podcast" today.'


def test_a_request_that_matched_a_goal_counts():
    assert goals_moved_today([PODCAST, THESIS], since=MIDNIGHT, matched_ids={"g2"}) == ["finish the thesis"]
    assert night(["a", "b"]) == 'Good night. You made progress on "a" and "b" today.'
    assert night(["a", "b", "c"]) == "Good night. You made progress on 3 of your goals today."


def test_the_offers_remember_which_goals_a_request_matched():
    clock = {"t": 0.0}
    offers = Offers(mode=lambda: "on", profile_empty=lambda: False, now=lambda: NIGHT.replace(hour=10),
                    clock=lambda: clock["t"])  # fmt: skip
    offer = goal_progress("open the podcast folder", [PODCAST, THESIS])
    assert offer is not None
    offers.offer_soon(offer)
    clock["t"] += 3600  # the offer was never made: it stopped being timely
    assert offers.goals_matched_today() == {"g1"}
    offers.declined[offer.key] = NIGHT.date().isoformat()  # "no" to noting it: not counted as progress
    assert offers.goals_matched_today() == set()
