"""What Heliox says back about the person, through the server's own handlers (pilot.workflows.user_profile,
pilot.personal.memory_talk).

* An empty profile is answered short. How to tell Heliox things is said to someone who has never told it anything, at
  most once a week, with no one's name in it (it used to be 24 words, every time, with a real person's name).
* What was kept is said back as Heliox's own sentence where the rule that kept it knows the shape: "Okay, you stop
  work at 6 pm.", "Okay, your flight is on Friday the 13th." The person's words are quoted only where it does not,
  and "(said on ...)" only when they asked when they said it.
* The sentence names what was stored, and the turn's change record ("did you get that?") holds "remembered that".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.personal.memory_talk import MemoryTalk
from pilot.server import PilotServer
from pilot.turn_changes import turn_changes
from pilot.workflows.user_profile import HOW_TO_TELL, UserProfile

THURSDAY = datetime(2026, 11, 12, 10, 0)


class Clock:
    def __init__(self) -> None:
        self.at = THURSDAY

    def __call__(self) -> float:
        return self.at.timestamp()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def server(tmp_path, clock):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "profile.json", clock=clock)
    server._memory_talk = MemoryTalk(profile=lambda: server._user_profile, now=lambda: clock.at)
    return server


async def said(server, text: str) -> tuple[str | None, list[str]]:
    """Heliox's reply and what the turn recorded as changed, as a voice turn sees it."""
    changes: list[str] = []
    token = turn_changes.set(changes)
    try:
        return await server._control_reply(text), changes
    finally:
        turn_changes.reset(token)


# ── 2. an empty profile ──


@pytest.mark.asyncio
async def test_a_new_person_hears_how_once_a_week_and_no_name(server, clock, tmp_path):
    reply, changed = await said(server, "what do you know about me")
    assert reply == HOW_TO_TELL and len(reply.split()) <= 16
    assert changed == []  # saying how is not a change the person made
    assert not any(word[:1].isupper() for word in reply.replace('"', " ").split()[1:] if word not in ("Say",))
    assert (await said(server, "what do you know about me"))[0] == "Nothing yet."
    server._user_profile = UserProfile(tmp_path / "profile.json", clock=clock)  # a restart remembers it was said
    assert (await said(server, "what do you know about me"))[0] == "Nothing yet."
    clock.at += timedelta(days=8)
    assert (await said(server, "what do you know about me"))[0] == HOW_TO_TELL


@pytest.mark.asyncio
@pytest.mark.parametrize("forget", ["forget everything about me", "forget what I just told you"])
async def test_after_a_forget_it_is_nothing_at_the_moment(server, forget):
    await said(server, "remember that I'm a founder")
    await said(server, forget)
    assert (await said(server, "what do you know about me"))[0] == "Nothing at the moment."


# ── 3. said back in Heliox's own words ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("told", "reply", "stored"),
    [
        ("remember that I stop work at six", "Okay, you stop work at 6 pm. After that I'll hold offers and "
         "suggestions until morning.", "I stop work at six"),
        ("I finish work at 5:30 pm", "Okay, you stop work at 5:30 pm. After that I'll hold offers and suggestions "
         "until morning.", "I finish work at 5:30 pm"),
        ("remember that I'm a founder", "Okay, you're a founder.", "I'm a founder"),
        ("remember my car is a blue Corolla", "Okay, your car is a blue Corolla.", "my car is a blue Corolla"),
        # not about the person: their words, quoted
        ("remember that the plumber is coming on Thursday", 'Remembered: "the plumber is coming on Thursday".',
         "the plumber is coming on Thursday"),
    ],
)  # fmt: skip
async def test_what_was_kept_is_said_as_heliox_says_it(server, told, reply, stored):
    answer, changed = await said(server, told)
    assert answer == reply
    assert changed == ["remembered that"]  # what "did you get that?" is answered from
    assert any(item["text"] == stored for item in server._user_profile.get()["about"])


@pytest.mark.asyncio
async def test_a_dated_note_is_said_as_a_person_says_it_and_when_only_if_asked(server):
    answer, changed = await said(server, "my flight is on Friday")
    assert answer.startswith("Okay, your flight is on Friday the 13th. I'll keep that until Saturday the 14th")
    assert changed == ["remembered that"]
    stored = server._user_profile.get()["about"][0]["text"]
    assert stored == "My flight is on Friday 13 November (said on Thursday 12 November)"  # what is kept is unchanged
    recalled = (await said(server, "what did I tell you about my flight"))[0]
    assert recalled == "You told me your flight is on Friday the 13th."
    when = (await said(server, "when did I tell you about my flight?"))[0]
    assert when == "You told me your flight is on Friday the 13th (you told me on Thursday 12 November)."


@pytest.mark.asyncio
async def test_forgetting_says_what_went_in_the_same_words(server):
    await said(server, "my flight is on Friday")
    assert (await said(server, "forget about my flight"))[0] == (
        "Okay, I've forgotten that your flight is on Friday 13 November."
    )
