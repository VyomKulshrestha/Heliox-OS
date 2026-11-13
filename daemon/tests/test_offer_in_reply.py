"""One concrete offer in reply to a complaint or an admission ("I never clean anything up."), under every offers rule
(pilot.personal.wiring.offer_in_reply): off after "stop suggesting things", quiet unless asked, after work, in a
focus session, nothing in the first week but the getting-to-know questions, the caps, and "don't offer that"."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.personal.offers import Offers
from pilot.personal.wiring import offer_in_reply
from pilot.server import PilotServer
from pilot.workflows.user_profile import UserProfile

TIDY = ("tidy_downloads", "Want me to find what's taking up space in Downloads?", "find large files in Downloads",
        "You said you never clean anything up.")  # fmt: skip


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0
        self.now = datetime(2026, 11, 12, 11, 0)

    def __call__(self) -> float:
        return self.t


@pytest.fixture
async def server(tmp_path):
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "profile.json")
    await server._user_profile.answer_async("remember that I'm a founder")  # past the first week's empty profile
    clock = Clock()
    server._offers = Offers(
        mode=lambda: server.config.suggestions.offers,
        profile_empty=server._user_profile.is_empty,
        suggestion=lambda: None,
        clock=clock,
        now=lambda: clock.now,
        state_path=tmp_path / "offers.json",
    )
    server._test_clock = clock
    return server


@pytest.mark.asyncio
async def test_offered_once_and_counted(server):
    first = offer_in_reply(server, *TIDY)
    assert first
    assert server._offers.latest().offer.request == "find large files in Downloads"  # "yes" and "why?" reach it
    assert offer_in_reply(server, *TIDY) == ""  # once: the gap, the hour, and the same offer today


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ["stop suggesting things", "no more offers", "stay quiet unless I ask"])
async def test_nothing_after_the_person_said_so(server, said):
    await server._control_reply(said)
    assert offer_in_reply(server, *TIDY) == ""


@pytest.mark.asyncio
async def test_nothing_in_the_first_week_with_nothing_known(tmp_path, server):
    await server._control_reply("forget everything about me")
    assert offer_in_reply(server, *TIDY) == ""
    server.config.suggestions.offers = "on"  # the person turned offers on themselves
    assert offer_in_reply(server, *TIDY)


@pytest.mark.asyncio
async def test_nothing_after_work_or_in_focus(server):
    await server._control_reply("I stop work at six")
    assert offer_in_reply(server, *TIDY)  # 11:00: the working day
    server._offers.made.clear()
    server._test_clock.now = datetime(2026, 11, 12, 19, 0)
    assert offer_in_reply(server, *TIDY) == ""  # after six
    server._test_clock.now = datetime(2026, 11, 13, 11, 0)
    server._focus = SimpleNamespace(active=True)
    assert offer_in_reply(server, *TIDY) == ""  # a focus session


@pytest.mark.asyncio
async def test_dont_offer_that_silences_it_for_good(server):
    assert offer_in_reply(server, *TIDY)
    assert await server._control_reply("don't offer that again") == "Okay, I won't offer that again."
    server._test_clock.t += 24 * 3600
    server._test_clock.now = datetime(2026, 11, 13, 11, 0)
    assert offer_in_reply(server, *TIDY) == ""
