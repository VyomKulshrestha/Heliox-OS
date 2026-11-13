"""Suggestions, offers and Live by voice, through the server's own handler: the value saved in the person's settings
file and the sentence said, for every phrase family and the negations people use (pilot.personal.spoken_settings).

A request to stop something is never answered by a sentence that describes doing it: on 13 November "stop
suggesting things" was answered "Okay, I'll offer a suggestion now and then" and saved suggestions on.
"""

from __future__ import annotations

import tomllib
from unittest.mock import AsyncMock

import pytest

from pilot import config as config_module
from pilot.config import PilotConfig
from pilot.personal import spoken_settings as said_as
from pilot.personal.wiring import held_for_quiet, offers_for, quiet_now
from pilot.server import PilotServer
from pilot.workflows.routines import normalize_phrase
from pilot.workflows.user_profile import UserProfile

# (phrases, section, the values on disk after, the sentence, the values before)
_SUGGEST_OFF = [
    "stop suggesting things",
    "Stop suggesting things.",
    "please stop suggesting",
    "quit suggesting things",
    "don't suggest things",
    "do not suggest anything",
    "never suggest anything again",
    "no more suggestions",
    "turn off suggestions",
    "turn off proactive suggestions",
    "stop making suggestions",
]
_SUGGEST_ON = [
    "you can suggest things again",
    "suggest things again",
    "start suggesting things",
    "you can start suggesting things again",
    "turn on suggestions",
    "you can make suggestions again",
]
_OFFERS_OFF = ["no more offers", "stop offering things", "quit offering me things", "don't offer anything",
               "never offer me anything again", "turn off offers"]  # fmt: skip
_OFFERS_ON = ["you can offer things again", "turn on offers", "start offering things", "offer things again"]
_LIVE_OFF = ["stop correcting me", "quit correcting me", "don't correct me", "never correct me again",
             "no more corrections", "stop watching me", "stop watching my screen", "turn off the live companion",
             "turn off live", "do not watch my screen"]  # fmt: skip
_LIVE_ON = ["be my live companion", "turn on live", "turn on the live companion", "you can correct me again",
            "start correcting me", "watch over me"]  # fmt: skip
_LIVE_LESS = ["correct me less", "interrupt me less", "fewer corrections", "less interruptions",
              "correct me less often"]  # fmt: skip
_LIVE_MORE = ["correct me more", "you can correct me more often", "more corrections"]

ROWS = (
    [(p, "suggestions", {"enabled": False, "offers": "off"}, said_as.SUGGESTIONS_OFF, {"enabled": True}) for p in _SUGGEST_OFF]
    + [(p, "suggestions", {"enabled": True, "offers": "on"}, said_as.SUGGESTIONS_ON, {"enabled": False, "offers": "off"})
       for p in _SUGGEST_ON]
    + [(p, "suggestions", {"offers": "off"}, said_as.OFFERS_OFF, {"offers": "auto"}) for p in _OFFERS_OFF]
    + [(p, "suggestions", {"offers": "on"}, said_as.OFFERS_ON, {"offers": "off"}) for p in _OFFERS_ON]
    + [(p, "live", {"enabled": False}, said_as.LIVE_OFF, {"enabled": True}) for p in _LIVE_OFF]
    + [(p, "live", {"enabled": True}, said_as.LIVE_ON, {"enabled": False}) for p in _LIVE_ON]
    + [(p, "live", {"level": "quiet"}, said_as.LIVE_LESS, {"enabled": True, "level": "balanced"}) for p in _LIVE_LESS]
    + [(p, "live", {"level": "balanced"}, said_as.LIVE_MORE, {"enabled": True, "level": "quiet"}) for p in _LIVE_MORE]
)  # fmt: skip

# Words that describe doing the thing: never in the answer to a request to stop it.
_DOING = ("now and then", "i'll suggest", "i'll offer", "i'm with you", "i'll point out")


def _server(tmp_path, *, offers_running: bool) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._broadcast_notification = AsyncMock()
    server._user_profile = UserProfile(tmp_path / "profile.json")
    if offers_running:  # as in the daemon, where the offer loop is running (pilot.personal.wiring)
        server._offers = offers_for(server, tmp_path)
    return server


def _on_disk(section: str) -> dict:
    return tomllib.loads(config_module.CONFIG_FILE.read_text(encoding="utf-8")).get(section, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("offers_running", [False, True], ids=["bare", "offers_running"])
@pytest.mark.parametrize(("phrase", "section", "after", "sentence", "before"), ROWS, ids=[r[0] for r in ROWS])
async def test_the_phrase_saves_the_value_and_says_so(
    tmp_path, offers_running, phrase, section, after, sentence, before
):
    server = _server(tmp_path, offers_running=offers_running)
    for key, value in before.items():
        setattr(getattr(server.config, section), key, value)
    reply = await server._control_reply(phrase)
    assert reply == sentence, phrase
    saved = _on_disk(section)
    assert {key: saved.get(key) for key in after} == after, phrase
    assert {key: getattr(getattr(server.config, section), key) for key in after} == after
    assert server._phrase_reserved(normalize_phrase(phrase)), phrase  # no shortcut can take it over
    stopping = after.get("enabled") is False or after.get("offers") == "off" or after.get("level") == "quiet"
    if stopping:
        assert not any(doing in reply.casefold() for doing in _DOING), reply


@pytest.mark.asyncio
async def test_fewer_corrections_while_live_is_off_says_so(tmp_path):
    server = _server(tmp_path, offers_running=False)
    assert await server._control_reply("correct me less") == said_as.LIVE_LESS_WHILE_OFF
    assert _on_disk("live")["level"] == "quiet" and _on_disk("live")["enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    ["stop interrupting me", "quit interrupting me", "don't interrupt me", "never interrupt me", "no more interruptions",
     "stay quiet unless I ask"],
)  # fmt: skip
async def test_interrupting_is_quiet_unless_asked_and_settings_stay(tmp_path, phrase):
    """ "Stop interrupting me" holds offers, suggestions and Live's non-urgent catches; Live still flags a risky
    command. It is the profile's preference, so Settings (and the person's file) are left as they were."""
    server = _server(tmp_path, offers_running=True)
    server.config.live.enabled = True
    server.config.suggestions.enabled = True
    file_before = config_module.CONFIG_FILE.read_bytes() if config_module.CONFIG_FILE.exists() else b""
    reply = await server._control_reply(phrase)
    assert reply.startswith("Okay, I'll stay quiet unless you ask: no offers"), reply
    assert quiet_now(server) == "asked" and held_for_quiet(server)
    after = config_module.CONFIG_FILE.read_bytes() if config_module.CONFIG_FILE.exists() else b""
    assert after == file_before
    assert (await server._control_reply("you can speak up again")).startswith("Okay, I'll speak up again")
    assert quiet_now(server) == ""


@pytest.mark.asyncio
async def test_saying_it_twice_writes_the_file_once(tmp_path):
    server = _server(tmp_path, offers_running=False)
    server.config.suggestions.enabled = True
    await server._control_reply("stop suggesting things")
    first = config_module.CONFIG_FILE.stat().st_mtime_ns
    copies = sorted(config_module.CONFIG_FILE.parent.glob("config.toml.saved-*"))
    assert await server._control_reply("no more suggestions") == said_as.SUGGESTIONS_OFF  # still answered
    assert config_module.CONFIG_FILE.stat().st_mtime_ns == first
    assert sorted(config_module.CONFIG_FILE.parent.glob("config.toml.saved-*")) == copies


@pytest.mark.asyncio
async def test_a_mute_said_twice_writes_once(tmp_path):
    server = _server(tmp_path, offers_running=False)
    calls = []
    real = server._handle_update_config

    async def counting(params, ws):
        calls.append(params)
        return await real(params, ws)

    server._handle_update_config = counting
    await server._live_mute_kind("weekday_date")
    await server._live_mute_kind("weekday_date")
    assert len(calls) == 1 and server.config.live.muted_kinds == ["weekday_date"]
    assert _on_disk("live")["muted_kinds"] == ["weekday_date"]


@pytest.mark.parametrize(
    "phrase",
    ["what do you suggest", "any suggestions", "don t suggest that again", "stop offering me that", "suggest a name",
     "correct this sentence", "stop", "live music near me"],
)  # fmt: skip
def test_what_is_not_a_setting(phrase):
    assert said_as.spoken_setting(normalize_phrase(phrase), PilotConfig()) is None


@pytest.fixture(autouse=True)
def _the_person_approves_setting_changes(monkeypatch):
    """These tests are about checking and saving settings; asking before a widening is test_guard_changes.py's."""
    from pilot.server import PilotServer

    async def approved(self, ws, found):
        return None

    monkeypatch.setattr(PilotServer, "_approve_widenings", approved)

    async def approved_before(self, found):
        return True  # as if this person had approved these captures before on this computer

    monkeypatch.setattr(PilotServer, "_approved_before", approved_before)


@pytest.mark.asyncio
@pytest.mark.parametrize("window_open", [True, False], ids=["window-open", "no-window"])
async def test_turning_live_on_for_the_first_time_asks_in_the_window(tmp_path, monkeypatch, window_open):
    """Voice cannot widen a guard: the first time, the question goes to the window and nothing is saved yet."""

    async def never_before(self, found):
        return False

    async def no_answer_yet(self, ws, found):
        return {"status": "cancelled", "message": "Nothing was changed."}

    monkeypatch.setattr(PilotServer, "_approved_before", never_before)
    monkeypatch.setattr(PilotServer, "_approve_widenings", no_answer_yet)
    server = _server(tmp_path, offers_running=False)
    if window_open:
        server._clients.add(object())

    reply = await server._control_reply("be my live companion")

    assert reply == (said_as.APPROVE_IN_WINDOW if window_open else said_as.OPEN_TO_APPROVE)
    assert server.config.live.enabled is False
    assert _on_disk("live").get("enabled") in (None, False)
    for task in list(server.__dict__.get("_spoken_widening_tasks", set())):
        await task
