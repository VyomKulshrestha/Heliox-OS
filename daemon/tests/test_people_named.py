"""A known person said back by name: "to your manager, Priya", resolved when Heliox speaks, stored as said."""

from __future__ import annotations

import asyncio
from typing import Any

from pilot.personal.people import FRESH_SECONDS, known_people, resolve_people, said_with_people
from pilot.workflows.user_profile import UserProfile


class Clock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def profile_with(tmp_path, *people: tuple[str, str], clock: Clock) -> UserProfile:
    profile = UserProfile(tmp_path / "profile.json", clock=clock)
    for name, relation in people:
        asyncio.run(profile.add_person(name, relation))
    return profile


def test_a_known_person_is_named():
    people = {"manager": "Priya"}
    assert resolve_people("send the deck to my manager", people) == "send the deck to your manager, Priya"
    assert resolve_people("Call your manager.", people) == "Call your manager, Priya."


def test_the_words_stay_when_nothing_is_known_or_the_name_is_there():
    assert resolve_people("send the deck to my manager", {}) == "send the deck to my manager"
    assert resolve_people("tell my manager Priya", {"manager": "Priya"}) == "tell my manager Priya"
    assert resolve_people("my manager's deck", {"manager": "Priya"}) == "my manager's deck"
    assert resolve_people("my managerial notes", {"manager": "Priya"}) == "my managerial notes"


def test_two_people_for_one_role_keep_the_words(tmp_path):
    clock = Clock()
    profile = profile_with(tmp_path, ("Ana", "friend"), ("Ben", "friend"), clock=clock)
    clock.t += 3600
    assert "friend" not in known_people(profile)
    assert said_with_people("call my friend", profile) == "call my friend"


def test_a_person_told_a_moment_ago_keeps_the_words(tmp_path):
    clock = Clock()
    profile = profile_with(tmp_path, ("Priya", "manager"), clock=clock)
    assert said_with_people("send the deck to my manager", profile) == "send the deck to my manager"
    clock.t += FRESH_SECONDS + 1
    assert said_with_people("send the deck to my manager", profile) == "send the deck to your manager, Priya"


def test_no_profile_keeps_the_words():
    assert said_with_people("send the deck to my manager", None) == "send the deck to my manager"


def _server(tmp_path, clock: Clock, profile: UserProfile):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer
    from pilot.workflows.reminders import ReminderStore

    server = PilotServer(PilotConfig())
    shown: list[tuple[str, dict[str, Any]]] = []

    async def broadcast(method, params, **_kwargs):
        shown.append((method, params))
        return 1

    server._broadcast_notification = broadcast
    server._user_profile = profile
    server._reminders = ReminderStore(
        tmp_path / "reminders.json", deliver=server._deliver_reminder, deliver_missed=server._deliver_missed_reminders,
        deliver_held=server._deliver_held_reminders, clock=clock,
    )  # fmt: skip
    return server, shown


def test_the_confirmation_names_the_manager_and_the_store_keeps_the_words(tmp_path):
    from pilot.server import PEOPLE_NAMED

    clock = Clock()
    profile = profile_with(tmp_path, ("Priya", "manager"), clock=clock)
    clock.t += 60
    server, _shown = _server(tmp_path, clock, profile)

    async def ask():
        answer = await server._local_answer("remind me in an hour to send the deck to my manager")
        return answer, PEOPLE_NAMED.get()

    answer, named = asyncio.run(ask())
    assert answer is not None and "your manager, Priya" in answer, answer
    assert named == (answer, named[1]) and "my manager" in named[1], named
    texts = [item["text"] for item in server._reminders.list()]
    assert texts and all("Priya" not in text and "my manager" in text for text in texts), texts


def test_the_reminder_follows_the_profile_as_it_is_when_due(tmp_path):
    clock = Clock()
    profile = profile_with(tmp_path, ("Priya", "manager"), clock=clock)
    server, shown = _server(tmp_path, clock, profile)
    server._reminders.add(clock.t + 3600, "send the deck to my manager")
    clock.t += 1800
    asyncio.run(profile.add_person("Dana", "manager"))  # a new manager before it comes due
    clock.t += 1800 + 1
    asyncio.run(server._deliver_reminder({**server._reminders.list()[0], "late": False}))
    messages = [params.get("message", "") for method, params in shown if method == "reminder"]
    assert messages == ["Reminder: send the deck to your manager, Dana"], messages
