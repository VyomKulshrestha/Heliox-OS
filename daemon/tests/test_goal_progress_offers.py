"""A finished request that matches one of the person's goals brings one offer to note it as progress
(pilot.personal.offers.goal_progress, wiring.note_task_done), at the next quiet moment, with "why" from the words."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from pilot.personal.offers import Offers, goal_progress
from pilot.personal.wiring import note_task_done
from pilot.workflows.goals import GoalStore

PODCAST = {"id": "g1", "title": "launch the podcast", "status": "active"}
FIT = {"id": "g2", "title": "run a half marathon", "status": "active"}


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.mark.parametrize(
    ("request_text", "goal_id"),
    [
        ("open the podcast folder", "g1"),
        ("find my podcast notes from last week", "g1"),
        ("play my marathon playlist", "g2"),
        ("open the folder", None),  # no topic word
        ("add to my podcast goal recorded episode one", None),  # already about the goal
        ("send the invoice", None),
        ("find the podcast marathon episode", None),  # two goals: not guessed
    ],
)
def test_which_requests_tie_to_a_goal(request_text, goal_id):
    offer = goal_progress(request_text, [PODCAST, FIT, {"id": "g3", "title": "old podcast", "status": "done"}])
    assert (offer.key.split(":")[1] if offer else None) == goal_id


def test_the_offer_and_its_reason():
    offer = goal_progress("open the podcast folder", [PODCAST])
    assert offer.text == 'That sounds like progress on your goal "launch the podcast". Want me to note it?'
    assert offer.why == 'You asked me to "open the podcast folder", and "podcast" is in your goal "launch the podcast".'
    assert offer.request == "add to my launch the podcast goal open the podcast folder"


def test_yes_notes_it_on_the_goal(tmp_path):
    goals = GoalStore(tmp_path / "goals.json")
    goals.create("launch the podcast")
    offer = goal_progress("open the podcast folder", goals.active())
    reply = goals.answer(offer.request)
    assert reply and "launch the podcast" in reply
    assert "open the podcast folder" in str(goals.find("podcast"))


def _offers(tmp_path, clock, profile_empty=lambda: False):
    return Offers(
        mode=lambda: "auto", profile_empty=profile_empty, clock=clock, now=lambda: datetime(2026, 11, 16, 11),
        state_path=tmp_path / "offers.json",
    )  # fmt: skip


def test_offered_at_the_next_quiet_moment_once_and_only_while_timely(tmp_path):
    clock = Clock()
    goals = GoalStore(tmp_path / "goals.json")
    goals.create("launch the podcast")
    server = SimpleNamespace(_goals=goals, _offers=_offers(tmp_path, clock))
    note_task_done(server, "open the podcast folder")
    offer = server._offers.next_offer()
    assert offer is not None and offer.kind == "goal_progress"
    server._offers.mark_offered(offer)
    clock.t += 3 * 3600
    note_task_done(server, "open the podcast folder")
    assert server._offers.next_offer() is None  # once a day per goal
    server2 = SimpleNamespace(_goals=goals, _offers=_offers(tmp_path / "b", clock))
    note_task_done(server2, "open the podcast folder")
    clock.t += 16 * 60
    assert server2._offers.next_offer() is None  # no quiet moment within 15 minutes: too late to be about it


def test_off_and_silenced(tmp_path):
    clock = Clock()
    goals = GoalStore(tmp_path / "goals.json")
    goals.create("launch the podcast")
    offers = Offers(mode=lambda: "off", profile_empty=lambda: False, clock=clock, state_path=tmp_path / "o.json")
    note_task_done(SimpleNamespace(_goals=goals, _offers=offers), "open the podcast folder")
    assert offers.next_offer() is None
    on = _offers(tmp_path / "c", clock)
    on.silence_kind("goal_progress")
    note_task_done(SimpleNamespace(_goals=goals, _offers=on), "open the podcast folder")
    assert on.next_offer() is None
