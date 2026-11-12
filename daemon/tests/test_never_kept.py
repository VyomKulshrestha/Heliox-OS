"""What Heliox never keeps about a person, whatever the route (pilot.personal.never_kept): one rule, every store."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from pilot.personal.getting_to_know import QUESTIONS, GettingToKnow
from pilot.personal.memory_talk import MemoryTalk
from pilot.personal.never_kept import never_kept
from pilot.workflows.user_profile import UserProfile


@pytest.fixture
def mirrored():
    return {"added": [], "forgotten": []}


@pytest.fixture
def profile(tmp_path, mirrored):
    async def remember(predicate, value):
        mirrored["added"].append(value)
        return f"fact-{len(mirrored['added'])}"

    async def forget(fact_id):
        mirrored["forgotten"].append(fact_id)

    return UserProfile(tmp_path / "profile.json", remember=remember, forget=forget, clock=lambda: 1000.0)


def stored(profile) -> str:
    return json.dumps(json.loads(profile._path.read_text(encoding="utf-8"))) if profile._path.exists() else ""


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("my salary is 90k", "money"),
        ("I have a dentist appointment on Friday", "health"),
        ("the wifi password is hunter2", "password"),
        ("Ana's address is 5 High Street", "someone else's private details"),
        ("I live in Pune", ""),
        ("the spare key is under the blue pot", ""),  # a bare key or code is ordinary
        ("I write code for a living", ""),
        ("the door code is 4321", "password"),
        ("Priya (manager)", ""),
    ],
)
def test_the_rule(text, why):
    assert (why in never_kept(text)) if why else never_kept(text) == ""


@pytest.mark.asyncio
async def test_the_profiles_own_phrase(profile, mirrored):
    reply = await profile.answer_async("remember that I have a dentist appointment on Friday")
    assert reply == "I won't keep that, because it is about health. Heliox never keeps notes like that."
    assert "dentist" not in stored(profile) and mirrored["added"] == []


@pytest.mark.asyncio
async def test_memory_talk(profile):
    talk = MemoryTalk(profile=lambda: profile, now=lambda: datetime(2026, 11, 12, 10))
    assert (await talk.answer("remember that my salary is 90k")).startswith(
        "I won't keep that, because it is about money"
    )
    assert (await talk.answer("remember that Ana's address is 5 High Street")).startswith(
        "I won't keep that, because it is someone else's private details"
    )
    reply = await talk.answer("my dentist appointment is on Friday")  # said in passing
    assert reply.startswith("I haven't kept that, because it is about health")
    assert "90k" not in stored(profile) and "High Street" not in stored(profile) and "dentist" not in stored(profile)


@pytest.mark.asyncio
async def test_getting_to_know(profile):
    gtk = GettingToKnow(profile=lambda: profile)
    gtk.start()
    assert await gtk.answer("Sam") == QUESTIONS[1]
    reply = await gtk.answer("my therapy notes and my medication schedule")
    assert reply.startswith("I won't keep that, because it is about health")
    assert "therapy" not in stored(profile)


@pytest.mark.asyncio
async def test_people_keep_who_someone_is(profile):
    """A person's role is who they are to the person ("my dentist is Dr Mehta"); health details never get that far."""
    assert await profile.answer_async("my dentist is Dr Mehta") == "Got it, I'll remember your dentist Dr Mehta."


@pytest.mark.asyncio
async def test_what_was_stored_before_is_swept_once_and_said_once(profile, mirrored):
    await profile.add("about", "I live in Pune")
    data = json.loads(profile._path.read_text(encoding="utf-8"))
    data["about"] += [
        {"id": "old0000001", "text": "my salary is 90k", "at": 1.0},
        {"id": "old0000002", "text": "I have a dentist appointment on Friday", "at": 2.0},
    ]
    data["_mirror"].update({"old0000001": "fact-old-1", "old0000002": "fact-old-2"})
    profile._path.write_text(json.dumps(data), encoding="utf-8")
    # Reading already leaves them out, before any sweep.
    assert "90k" not in " ".join(profile.context_lines()) and "dentist" not in profile.describe()
    first = await profile.answer_async("what do you know about me")
    assert first.startswith(
        "I've stopped keeping 2 things you told me before, about health, money: Heliox never keeps notes like that."
    )
    assert "90k" not in first and "dentist" not in first  # what was dropped is never said again
    assert "Pune" in first
    assert "90k" not in stored(profile) and "dentist" not in stored(profile)
    assert set(mirrored["forgotten"]) >= {"fact-old-1", "fact-old-2"}  # gone from long-term memory too
    again = await profile.answer_async("what do you know about me")
    assert not again.startswith("I've stopped keeping")  # said once
