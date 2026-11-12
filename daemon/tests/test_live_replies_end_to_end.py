"""The person's answers to Live, by voice and by click, and that they last across a daemon restart: "why did you say
that", "thanks", "not now" and "don't mention these"; and "don't mention these" after an offer silences that kind of
offer too. Real server handlers and stores in a temporary folder; no audio."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.live.engine import LiveCompanion, Observation

TEXTS = {
    "weekday": "Lunch on Tuesday, 14 November with Sam",
    "sum": "The total is 12 x 15 = 170 so we order more",
}


def make_server(tmp_path, config=None, text=TEXTS["weekday"]):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    config = config or PilotConfig()
    config.save = MagicMock()
    config.live.enabled = True
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    shown = {"text": text}

    async def observe():
        return Observation(text=shown["text"], app="notepad.exe", title="plan.txt - Notepad", source="focused_field")

    server._live = LiveCompanion(
        settings=lambda: server.config.live,
        observe=observe,
        emit=server._broadcast_notification,
        speak=server._live_speak,
        relevance_path=tmp_path / "live_relevance.json",
        mute_kind=server._live_mute_kind,
        today=lambda: date(2026, 11, 9),
        wallclock=lambda: datetime(2026, 11, 9, 11, 0),
    )
    clock = [100.0]
    server._live._clock = lambda: clock[0]
    return server, shown, clock


async def raise_card(server, clock, seconds=8):
    for _ in range(int(seconds / 0.2)):
        clock[0] += 0.2
        await server._live.tick()
    cards = [c.args[1] for c in server._broadcast_notification.await_args_list if c.args[0] == "live_interjection"]
    return cards[-1] if cards else None


@pytest.mark.asyncio
async def test_by_voice(tmp_path):
    server, _, clock = make_server(tmp_path)
    card = await raise_card(server, clock)
    assert card["kind"] == "weekday_date"
    why = await server._control_reply("why did you say that")
    assert why.startswith('In Notepad, you wrote "Tuesday, 14 November"') and "17 November" in why
    assert await server._control_reply("thanks") == "Glad it helped."
    assert server._live.relevance.score("weekday_date") > 2 / 3
    assert (await server._control_reply("not now")).startswith("Okay. I won't bring that kind of thing up")
    assert "weekday_date" in server._live.snoozed_kinds()
    assert (await server._control_reply("don't mention these again")).startswith("Okay, I won't mention these")
    assert server.config.live.muted_kinds == ["weekday_date"]


@pytest.mark.asyncio
async def test_by_click_and_across_a_restart(tmp_path):
    from pilot.config import PilotConfig

    config = PilotConfig()
    server, shown, clock = make_server(tmp_path, config)
    card = await raise_card(server, clock)
    explained = await server._handle_live_explain({"id": card["id"]}, None)
    assert explained["text"].startswith('In Notepad, you wrote "Tuesday, 14 November"')
    assert (await server._handle_live_feedback({"id": card["id"], "verdict": "helpful"}, None))["status"] == "ok"
    shown["text"] += ". " + TEXTS["sum"]  # typed on in the same field
    card = await raise_card(server, clock)
    assert card["kind"] == "arithmetic"
    await server._handle_live_feedback({"id": card["id"], "verdict": "dismissed"}, None)  # the card's "Not now"
    assert "arithmetic" in server._live.snoozed_kinds()

    # The daemon restarts: same config and data folder, a fresh Live with a fresh clock.
    again, shown_again, clock_again = make_server(tmp_path, config, text=TEXTS["sum"].replace("170", "171"))
    assert "arithmetic" in again._live.snoozed_kinds()  # "not now" still holds
    assert again._live.relevance.score("weekday_date") > 2 / 3  # "thanks" was kept
    assert await raise_card(again, clock_again) is None  # the held-back kind stays quiet
    shown_again["text"] += ". Dinner on Monday, 13 November at eight"
    card = await raise_card(again, clock_again)
    await again._handle_live_feedback({"id": card["id"], "verdict": "mute"}, None)  # "Don't mention these"
    assert config.live.muted_kinds == ["weekday_date"]
    third, shown_third, clock_third = make_server(tmp_path, config, text="Call on Friday, 11 November")
    assert await raise_card(third, clock_third) is None  # muted for good, after another restart


@pytest.mark.asyncio
async def test_dont_mention_these_after_an_offer_silences_that_kind_of_offer(tmp_path):
    from pilot.personal.offers import Offer, Offers

    state = tmp_path / "offers.json"
    offers = Offers(mode=lambda: "on", profile_empty=lambda: False, state_path=state)
    outlook = Offer("usual_app:outlook.exe", "usual_app", "Open Outlook?", "open Outlook", "Usually.")
    offers.mark_offered(outlook)
    reply = offers.reply("don't mention these again")
    assert reply.reply == "Okay, no more offers like that: opening the apps you usually open."
    slack = Offer("usual_app:slack.exe", "usual_app", "Open Slack?", "open Slack", "Usually.")
    after_restart = Offers(mode=lambda: "on", profile_empty=lambda: False, state_path=state)
    after_restart.candidates = lambda: [slack]
    assert after_restart.next_offer() is None  # another app, same kind: silent, after a restart too
    assert {"key": "kind:usual_app", "label": "All offers like this: opening the apps you usually open"} in (
        after_restart.status()["silenced"]
    )
    assert after_restart.unsilence("kind:usual_app")  # and Settings can undo it
    assert after_restart.next_offer() is slack


@pytest.mark.asyncio
async def test_the_spoken_forms_after_the_listener_normalises_them(tmp_path):
    """The listener's normaliser turns "don't" into "don t" and "what's" into "what s": the replies still work."""
    from pilot.personal.offers import Offer
    from pilot.personal.wiring import offers_for

    server, _, clock = make_server(tmp_path)
    await raise_card(server, clock)
    assert (await server._control_reply("what's wrong")).startswith("In Notepad")
    assert (await server._control_reply("that's fine")).startswith("Okay.")
    offers = offers_for(server)
    offers.mode = lambda: "on"
    offers.mark_offered(Offer("habit:reading", "habit", "Log reading?", "log reading", "Daily."))
    assert await server._control_reply("don't offer that again") == "Okay, I won't offer that again."
