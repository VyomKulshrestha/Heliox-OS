"""The first conversation that fills an empty profile: three questions at most, each skippable, read back at the
end, and the same thing as a form. Files in a temporary folder; no model, no audio."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.personal.getting_to_know import QUESTIONS, GettingToKnow, parse_name, parse_preferences, parse_use
from pilot.personal.offers import INTRO, Offers
from pilot.workflows.user_profile import UserProfile


@pytest.fixture
def profile(tmp_path):
    return UserProfile(tmp_path / "user_profile.json")


def test_answers_are_read_in_the_words_people_use():
    assert parse_name("call me Sam") == "Sam"
    assert parse_name("It's Dr Rao, please") == "Dr Rao"
    assert parse_name("my name is Ana María") == "Ana María"
    assert parse_name("I use it for writing and the odd video call") == ""
    assert parse_use("mostly for writing and video calls") == "writing and video calls"
    assert parse_use("I use it for coding") == "coding"
    assert parse_use("photo editing") == "photo editing"
    assert parse_preferences("short please, and stay quiet unless I ask") == ("short", "quiet")
    assert parse_preferences("more detail, and tell me when you notice something") == ("detailed", "speak_up")
    assert parse_preferences("whatever") == ("", "")


@pytest.mark.asyncio
async def test_three_questions_then_it_stops_and_reads_back_what_it_kept(profile):
    quiet = AsyncMock()
    clock = [0.0]
    gtk = GettingToKnow(profile=lambda: profile, set_quiet=quiet, clock=lambda: clock[0])
    assert await gtk.answer("what's the weather") is None  # not a session, not the phrase
    first = await gtk.answer("get to know me")
    assert first.endswith(QUESTIONS[0]) and "skip" in first
    assert await gtk.answer("call me Sam") == QUESTIONS[1]
    assert await gtk.answer("mostly writing and video calls") == QUESTIONS[2]
    end = await gtk.answer("short answers, and stay quiet unless I ask")
    assert end == (
        "Thanks. Here's what I kept: I call you Sam; you mainly use the computer for writing and video calls; you "
        "want short answers; I stay quiet unless you ask: offers are off, and Live speaks only about risky things. "
        "To change any of it, say forget and what it was, or remove it in Settings, under About you."
    )
    quiet.assert_awaited_once_with(True)
    assert not gtk.active
    assert profile.name() == "Sam" and profile.style() == {"length": "short"}
    assert "Answer style, length: short" in profile.context_lines()
    assert await gtk.answer("thanks") is None  # over: no fourth question


@pytest.mark.asyncio
async def test_every_step_can_be_skipped_and_stop_ends_it_at_once(profile):
    gtk = GettingToKnow(profile=lambda: profile)
    gtk.start()
    assert await gtk.answer("skip") == QUESTIONS[1]
    assert await gtk.answer("I'd rather not") == QUESTIONS[2]
    assert await gtk.answer("pass") == "Okay, I didn't keep anything. You can tell me about yourself any time."
    assert profile.is_empty()
    gtk.start()
    assert await gtk.answer("Sam") == QUESTIONS[1]
    assert (await gtk.answer("not now")).startswith("Thanks. Here's what I kept: I call you Sam.")


@pytest.mark.asyncio
async def test_an_unclear_answer_is_asked_once_more_then_moved_past(profile):
    gtk = GettingToKnow(profile=lambda: profile)
    gtk.start()
    assert await gtk.answer("I mostly use it for writing my thesis on rivers") == (
        "Sorry, which name should I use? Or say skip."
    )
    assert await gtk.answer("hmm I really do not know what to say") == QUESTIONS[1]  # asked once, then on
    assert profile.name() == ""


@pytest.mark.asyncio
async def test_an_abandoned_session_lapses(profile):
    clock = [0.0]
    gtk = GettingToKnow(profile=lambda: profile, clock=lambda: clock[0])
    gtk.start()
    clock[0] = 301.0
    assert await gtk.answer("Sam") is None and profile.is_empty()  # "Sam" an hour later is not an answer


@pytest.mark.asyncio
async def test_the_form_stores_the_same_things_and_each_line_can_be_removed(profile):
    gtk = GettingToKnow(profile=lambda: profile)
    result = await gtk.save_form(name="Priya", use="design work", length="detailed", speaking="speak_up")
    assert [item["text"] for item in result["stored"]] == [
        "I call you Priya",
        "you mainly use the computer for design work",
        "you want more detail",
        "I speak up when I notice something",
    ]
    for item in result["stored"]:
        assert await profile.remove_item(item["id"])
    assert profile.is_empty()
    assert (await gtk.save_form())["stored"] == []
    refused = await gtk.save_form(use="my password is hunter2-Xq9")
    assert refused["stored"] == [] and refused["problems"][0].startswith("That looks like a password or key")


def test_the_intro_is_the_one_offer_an_empty_profile_gets(tmp_path):
    state = {"empty": True}
    offers = Offers(mode=lambda: "auto", profile_empty=lambda: state["empty"], state_path=tmp_path / "o.json")
    assert offers.next_offer() is INTRO
    offers.mark_offered(INTRO)
    assert offers.next_offer() is None  # once a run
    silenced = Offers(mode=lambda: "auto", profile_empty=lambda: True, state_path=tmp_path / "o.json")
    silenced.silence(INTRO)
    assert Offers(mode=lambda: "auto", profile_empty=lambda: True, state_path=tmp_path / "o.json").next_offer() is None
    assert Offers(mode=lambda: "off", profile_empty=lambda: True).next_offer() is None
    state["empty"] = False
    assert Offers(mode=lambda: "auto", profile_empty=lambda: False).next_offer() is None  # no other source here


@pytest.mark.asyncio
async def test_by_voice_or_chat_and_from_the_app(tmp_path):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "p.json")
    assert (await server._control_reply("let's get to know each other")).endswith(QUESTIONS[0])
    assert await server._control_reply("not now") == "Okay, we can do this another time."  # not Live's "not now"
    started = await server._handle_getting_to_know({"action": "start"}, None)
    assert started["question"].endswith(QUESTIONS[0])
    server._broadcast_notification.assert_awaited_with(
        "voice_result",
        {"command": "", "status": "answered", "message": started["question"], "source": "getting_to_know"},
    )
    server._speak_companion_text.assert_not_awaited()  # voice is off: the question is only in the chat
    again = await server._handle_getting_to_know({"action": "start"}, None)
    assert again == {"status": "ok", "question": QUESTIONS[0], "already_active": True}
    assert server._broadcast_notification.await_count == 1  # the second start sent nothing
    status = await server._handle_getting_to_know({"action": "status"}, None)
    assert status == {"status": "ok", "active": True, "profile_empty": True}
    assert await server._control_reply("call me Sam") == QUESTIONS[1]
    assert await server._control_reply("skip") == QUESTIONS[2]
    await server._control_reply("stay quiet unless I ask")
    assert server.config.suggestions.offers == "off" and server.config.live.level == "quiet"
    saved = await server._handle_getting_to_know({"action": "save", "use": "spreadsheets"}, None)
    entry = next(item for item in saved["stored"] if "spreadsheets" in item["text"])
    removed = await server._handle_getting_to_know({"action": "remove", "id": entry["id"]}, None)
    assert removed["status"] == "ok"
    assert (await server._handle_getting_to_know({"action": "remove", "id": entry["id"]}, None))["status"] == "error"


@pytest.mark.asyncio
async def test_a_request_in_the_middle_ends_the_questions_and_goes_where_it_would_have_gone(profile):
    gtk = GettingToKnow(profile=lambda: profile)
    gtk.start()
    assert await gtk.answer("open my mail") is None
    assert not gtk.active and profile.is_empty()
