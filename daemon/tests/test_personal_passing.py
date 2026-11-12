"""Things said to Heliox in passing: kept as a note with the day written out, replaced by a newer note about the
same thing, lapsed after their day, forgettable. Injected clocks; no model."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pilot.personal.memory_talk import MemoryTalk
from pilot.personal.passing import passing_note
from pilot.workflows.user_profile import UserProfile

THURSDAY = datetime(2026, 11, 12, 11, 0)


@pytest.mark.parametrize(
    ("said", "kept"),
    [
        ("my flight is on Friday", "My flight is on Friday 13 November (said on Thursday 12 November)"),
        ("I moved the meeting to three", "My meeting is at three (said on Thursday 12 November)"),
        ("I moved the standup to Monday", "My standup is on Monday 16 November (said on Thursday 12 November)"),
        ("the review got moved to 20 November", "My review is on 20 November (said on Thursday 12 November)"),
        (
            "my dentist appointment is tomorrow at four",
            "My dentist appointment is tomorrow (Friday 13 November) at four (said on Thursday 12 November)",
        ),
    ],
)
def test_said_in_passing_is_kept_with_the_day_written_out(said, kept):
    assert passing_note(said, THURSDAY).text == kept


@pytest.mark.parametrize(
    "said",
    [
        "my meeting is at three, remind me before",  # a request rides along: the planner's
        "what time is my flight?",
        "my name is Sam",  # the profile's own phrase
        "my sister is Ana",
        "the build is broken",
    ],
)
def test_not_a_note(said):
    assert passing_note(said, THURSDAY) is None


@pytest.mark.asyncio
async def test_kept_replaced_lapsed_and_forgotten(tmp_path):
    clock = [THURSDAY.timestamp()]
    profile = UserProfile(tmp_path / "p.json", clock=lambda: clock[0])
    talk = MemoryTalk(profile=lambda: profile, now=lambda: datetime.fromtimestamp(clock[0]))
    assert await talk.answer("my meeting is at two") == (
        'Noted: "My meeting is at two". I\'ll keep that for a week; say "forget about my meeting" to drop it sooner.'
    )
    assert (await talk.answer("I moved the meeting to three")).startswith('Noted: "My meeting is at three"')
    assert [i["text"].split(" (")[0] for i in profile.get()["about"]] == ["My meeting is at three"]
    assert (await talk.answer("my flight is on Friday")).endswith(
        'I\'ll keep that until Saturday 14 November; say "forget about my flight" to drop it sooner.'
    )
    clock[0] += timedelta(days=1).total_seconds()  # the next day: still there
    assert any("My flight is on Friday 13 November" in line for line in profile.context_lines())
    clock[0] += timedelta(days=3).total_seconds()  # Monday: the flight's day is over
    assert not any("flight" in line for line in profile.context_lines())
    await talk.answer("anything at all")  # every answer first drops lapsed notes from the file and memory
    assert "flight" not in (tmp_path / "p.json").read_text(encoding="utf-8")
    assert (await talk.answer("forget about my meeting")).startswith("Okay, I've forgotten \"My meeting is at three")
    assert profile.is_empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "why"),
    [
        ("my dentist appointment is tomorrow at four", "it is about health"),
        ("my physio session is on Monday", "it is about health"),
        ("my rent payment is on Friday, 1200 euros", "it is about money"),
        ("my bonus is in March", "it is about money"),
        ("my sister's new address is in Leeds, 4 Elm Road", "it is someone else's private details"),
        ("my door code is at the back of the card", "it looks like a password, key or code"),
        ("my password is at the bottom of the drawer", "it looks like a password, key or code"),
    ],
)
async def test_never_kept_whatever_the_wording(tmp_path, said, why):
    profile = UserProfile(tmp_path / "p.json")
    talk = MemoryTalk(profile=lambda: profile, now=lambda: THURSDAY)
    reply = await talk.answer(said)
    assert reply == f"I haven't kept that, because {why}. I don't keep notes like that from things said in passing."
    assert profile.is_empty()


@pytest.mark.asyncio
async def test_one_switch_turns_passing_notes_off(tmp_path):
    on = [False]
    profile = UserProfile(tmp_path / "p.json")
    talk = MemoryTalk(profile=lambda: profile, now=lambda: THURSDAY, notes_enabled=lambda: on[0])
    assert await talk.answer("my flight is on Friday") is None  # left to the conversation; nothing kept
    assert profile.is_empty()
    assert (await talk.answer("remember that the flight is on Friday")).startswith("Remembered")  # asked for: kept
    on[0] = True
    assert (await talk.answer("my train is on Saturday")).startswith('Noted: "My train is on Saturday')


def test_the_switch_defaults_on_and_loads_from_config():
    from pilot.config import PilotConfig, _merge_config

    assert PilotConfig().memory.passing_notes is True
    assert _merge_config(PilotConfig(), {"memory": {"passing_notes": False}}).memory.passing_notes is False


@pytest.mark.asyncio
async def test_typed_chat_shows_the_same_line(tmp_path):
    """Typed chat has no voice: the reply to a note said in passing is the chat's answer, word for word."""
    from unittest.mock import AsyncMock, MagicMock

    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._user_profile = UserProfile(tmp_path / "p.json")
    server._memory_talk = MemoryTalk(profile=lambda: server._user_profile, now=lambda: THURSDAY)
    ws = MagicMock()
    ws.send = AsyncMock()
    server._broadcast_notification = AsyncMock()
    result = await server._handle_execute({"input": "my flight is on Friday", "session_id": "chat"}, ws)
    assert result["status"] == "success" and result["explanation"].startswith('Noted: "My flight is on Friday 13')
    sent = [call.args[0] for call in ws.send.await_args_list]
    assert any("conversation_response" in str(message) and "Noted:" in str(message) for message in sent)
