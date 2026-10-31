"""The About you profile: what you tell Heliox, kept locally, mirrored to memory, used."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.day_briefing import compose_morning
from pilot.workflows.user_profile import UserProfile


@pytest.fixture
def mirror():
    store: dict[str, tuple[str, str]] = {}
    counter = {"n": 0}

    async def remember(predicate: str, value: str) -> str:
        counter["n"] += 1
        fact_id = f"f{counter['n']}"
        store[fact_id] = (predicate, value)
        return fact_id

    async def forget(fact_id: str) -> None:
        store.pop(fact_id, None)

    return store, remember, forget


@pytest.fixture
def profile(tmp_path, mirror):
    store, remember, forget = mirror
    return UserProfile(tmp_path / "user_profile.json", remember=remember, forget=forget)


@pytest.mark.asyncio
async def test_learns_name_facts_people_preferences_and_style(profile):
    assert await profile.answer_async("call me Vyom") == "Nice to meet you, Vyom. I'll call you that."
    assert await profile.answer_async("remember that I'm a founder") == 'Remembered: "I\'m a founder".'
    assert await profile.answer_async("remember I work in pharma sales") == 'Remembered: "I work in pharma sales".'
    assert await profile.answer_async("my manager is Sarah") == "Got it, I'll remember your manager Sarah."
    assert await profile.answer_async("Priya is my wife") == "Got it, I'll remember your wife Priya."
    assert await profile.answer_async("I prefer short answers") == "Noted: you prefer short answers."
    assert await profile.answer_async("I'd rather use metric units") == "Noted: you would rather use metric units."
    assert await profile.answer_async("be brief") == "Okay, I'll keep my answers short."
    assert await profile.answer_async("keep it formal") == "Okay, I'll keep a formal tone."

    data = profile.get()
    assert data["call_me"] == "Vyom"
    assert data["style"] == {"length": "short", "formality": "formal"}
    assert [item["text"] for item in data["about"]] == ["I'm a founder", "I work in pharma sales"]
    assert [(p["name"], p["relation"]) for p in data["people"]] == [("Sarah", "manager"), ("Priya", "wife")]

    said = await profile.answer_async("what do you know about me")
    assert said == (
        "I call you Vyom. I know that you're a founder; you work in pharma sales. "
        "People you've mentioned: Sarah (manager), Priya (wife). "
        "You prefer short answers; would rather use metric units. And I keep my answers short, formal."
    )


@pytest.mark.asyncio
async def test_tasks_and_other_commands_pass_through(profile):
    for text in (
        "remember to buy milk",
        "remind me in 10 minutes to stretch",
        "what's my cpu usage",
        "open chrome",
        "note that the plumber comes Tuesday",
        "my cpu is at 100",
        "forget my activity from the last hour",
        "generate a password",
        "save that as my signature",
        "it is my dog",
        "i am tired",
        "log a workout",
    ):
        assert await profile.answer_async(text) is None, text
    assert profile.is_empty()


@pytest.mark.asyncio
async def test_correction_forget_and_clear(profile):
    await profile.answer_async("remember that I'm a founder")
    assert (await profile.answer_async("that's wrong, I run Dovel Lab")).startswith(
        'Fixed. I\'ll remember instead: "I run Dovel Lab"'
    )
    assert [item["text"] for item in profile.get()["about"]] == ["I run Dovel Lab"]

    await profile.answer_async("my doctor is Dr Mehta")
    assert await profile.answer_async("forget my doctor") == "Okay, I've forgotten that."
    assert profile.get()["people"] == []
    # A forget that matches nothing in the profile is left to other handlers.
    assert await profile.answer_async("forget the downloaded file") is None

    assert await profile.answer_async("forget everything about me") == "Done, I've forgotten what I knew about you."
    assert profile.is_empty()


@pytest.mark.asyncio
async def test_secrets_are_refused(profile):
    assert (await profile.answer_async("remember that I ghp_abcdefghijklmnopqrstuvwxyz0123")).startswith(
        "That looks like a password or key"
    )
    assert profile.is_empty()


@pytest.mark.asyncio
async def test_memory_mirror_adds_and_retracts(profile, mirror):
    store, _remember, _forget = mirror
    await profile.answer_async("call me Vyom")
    await profile.answer_async("remember that I'm a founder")
    await profile.answer_async("my manager is Sarah")
    assert len(store) == 3
    assert ("profile:about", "I'm a founder") in store.values()

    founder = next(item["id"] for item in profile.get()["about"])
    await profile.delete(founder)
    assert ("profile:about", "I'm a founder") not in store.values()
    await profile.clear()
    assert store == {}


def test_morning_greeting_uses_the_name():
    now = datetime(2026, 10, 31, 8, 30)
    assert compose_morning(now, reminders=[], todos=[], habits=[], name="Vyom").startswith("Good morning, Vyom.")
    assert compose_morning(now, reminders=[], todos=[], habits=[]).startswith("Good morning. It's")


@pytest.mark.asyncio
async def test_rpcs_routing_and_reservation(tmp_path):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "p.json")
    ws = MagicMock()

    assert (await server._control_reply("call me Vyom")).startswith("Nice to meet you, Vyom")
    added = await server._handle_profile_add({"kind": "about", "text": "I run Dovel Lab"}, ws)
    assert added["profile"]["call_me"] == "Vyom"
    assert any(row["text"] == "I run Dovel Lab" for row in added["entries"])
    assert (await server._handle_profile_add({"kind": "bogus", "text": "x"}, ws))["status"] == "error"
    assert (await server._handle_profile_set_name({"name": "Hardik"}, ws))["profile"]["call_me"] == "Hardik"

    entry_id = next(row["id"] for row in added["entries"] if row["text"] == "I run Dovel Lab")
    assert (await server._handle_profile_delete({"id": entry_id}, ws))["status"] == "ok"
    assert (await server._handle_profile_delete({"id": "missing"}, ws))["status"] == "error"

    got = await server._handle_profile_get({}, ws)
    assert "_mirror" not in got["profile"]
    assert server._phrase_reserved("what do you know about me")
    assert not server._phrase_reserved("open chrome")
