"""The one offer to keep an activity timeline (pilot.personal.offers.TIMELINE): after three different days of use,
only while the timeline is off, once ever; a spoken yes asks in the window, because keeping the timeline is a
capture (pilot.security.guard_changes)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.personal import spoken_settings as said_as
from pilot.personal.offers import TIMELINE, Offers
from pilot.security.guard_changes import section_widenings
from pilot.server import PilotServer


def _offers(tmp_path, *, off=True, mode="auto", now=datetime(2026, 11, 18, 11)):
    return Offers(
        mode=lambda: mode, profile_empty=lambda: True, timeline_off=lambda: off, now=lambda: now,
        state_path=tmp_path / "offers.json", clock=lambda: 1_000_000.0,
    )  # fmt: skip


def _used(offers, days=("2026-11-16", "2026-11-17", "2026-11-18")):
    for day in days:
        offers.note_used(day)


def test_not_before_three_days_of_use(tmp_path):
    offers = _offers(tmp_path)
    offers.intro_days = ["x", "y"]  # the getting-to-know question already asked twice
    _used(offers, ("2026-11-16", "2026-11-16", "2026-11-17"))
    assert offers.next_offer() is None
    offers.note_used("2026-11-18")
    assert offers.next_offer() is TIMELINE


def test_once_ever_across_a_restart_whatever_the_answer(tmp_path):
    offers = _offers(tmp_path)
    offers.intro_days = ["x", "y"]
    _used(offers)
    offers.mark_offered(offers.next_offer())
    again = _offers(tmp_path)  # the next run reads offers.json
    again.intro_days = ["x", "y"]
    assert again.timeline_offered and again.next_offer() is None
    assert again.used_days == ["2026-11-16", "2026-11-17", "2026-11-18"]  # dates only, the first three


@pytest.mark.parametrize(("off", "mode"), [(False, "auto"), (True, "off")])
def test_not_when_it_is_on_or_offers_are_off(tmp_path, off, mode):
    offers = _offers(tmp_path, off=off, mode=mode)
    offers.intro_days = ["x", "y"]
    _used(offers)
    assert offers.next_offer() is None


def test_keeping_the_timeline_is_a_capture():
    found = section_widenings("timeline", {"enabled": False}, {"enabled": True})
    assert [w.key for w in found] == ["timeline.enabled"]
    assert section_widenings("timeline", {"enabled": True}, {"enabled": False}) == []  # turning it off is free


@pytest.mark.asyncio
async def test_a_spoken_yes_asks_in_the_window(tmp_path):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    assert TIMELINE.request == "turn on the activity timeline"
    reply = await server._control_reply(TIMELINE.request)  # what "yes" sends down the normal road
    assert reply in (said_as.OPEN_TO_APPROVE, said_as.APPROVE_IN_WINDOW)
    assert server.config.timeline.enabled is False  # nothing turned on by voice
