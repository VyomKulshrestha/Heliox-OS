"""The companion: the assistant's name and tone, and how they reach the model prompt."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.companion import DEFAULT_NAME, Companion


@pytest.fixture
def companion(tmp_path):
    return Companion(tmp_path / "companion.json")


@pytest.mark.asyncio
async def test_name_and_persona_by_voice(companion):
    assert (await companion.answer_async("what's your name")).startswith("I'm Heliox, your assistant")
    assert await companion.answer_async("your name is Nova") == "Done. I'm Nova from now on."
    assert await companion.answer_async("what's your name") == "I'm Nova, your Heliox assistant."
    assert await companion.answer_async("be more playful") == "Okay, I'll be playful and upbeat."
    assert await companion.answer_async("be strictly professional") == "Okay, I'll be professional and precise."
    assert (await companion.answer_async("what's your personality")).startswith("Right now I keep a professional")
    assert (await companion.answer_async("introduce yourself")).startswith("I'm Nova, your Heliox assistant.")
    assert companion.get() == {"name": "Nova", "persona": "professional"}
    assert (
        await companion.answer_async("go back to normal") == "Back to normal: I'm Heliox again, with a balanced tone."
    )
    assert companion.get() == {"name": DEFAULT_NAME, "persona": "balanced"}


@pytest.mark.asyncio
async def test_unrelated_phrases_pass_through(companion):
    for text in ("be quiet", "be right back", "call mom", "open chrome", "what's the weather", "professional help"):
        assert await companion.answer_async(text) is None, text


def test_system_prompt_line(tmp_path):
    companion = Companion(tmp_path / "c.json")
    assert companion.system_prompt_line("") == ""  # default name and tone, no user name: nothing added
    assert companion.system_prompt_line("Vyom").startswith("Voice and persona")

    import asyncio

    asyncio.run(companion.set_name("Nova"))
    asyncio.run(companion.set_persona("witty"))
    line = companion.system_prompt_line("Vyom")
    assert 'calls you "Nova"' in line and "witty" in line and "speaking with Vyom" in line
    assert "never claim to have done something you did not do" in line


@pytest.mark.asyncio
async def test_on_change_fires(tmp_path):
    hits = {"n": 0}

    async def on_change() -> None:
        hits["n"] += 1

    companion = Companion(tmp_path / "c.json", on_change=on_change)
    await companion.set_name("Nova")
    await companion.set_persona("warm")
    await companion.reset()
    assert hits["n"] == 3


@pytest.mark.asyncio
async def test_rpcs_routing_and_planner_prompt(tmp_path):
    from pilot.agents.planner import Planner
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._companion = Companion(tmp_path / "c.json", on_change=server._companion_apply)
    server._user_profile = MagicMock()
    server._user_profile.name.return_value = "Vyom"
    server._user_profile.answer_async = AsyncMock(return_value=None)
    server._planner = MagicMock(spec=Planner)

    assert (await server._control_reply("call yourself Nova")) == "Done. I'm Nova from now on."
    # Changing the name re-applied the persona to the planner's system prompt.
    assert server._planner.set_persona_context.called
    applied = server._planner.set_persona_context.call_args.args[0]
    assert 'calls you "Nova"' in applied and "speaking with Vyom" in applied

    got = await server._handle_companion_get({}, MagicMock())
    assert got["name"] == "Nova" and "witty" in got["presets"]
    assert (await server._handle_companion_set({"persona": "bogus"}, MagicMock()))["status"] == "error"
    assert (await server._handle_companion_set({"name": "Atlas", "persona": "calm"}, MagicMock()))["name"] == "Atlas"
    assert (await server._handle_companion_reset({}, MagicMock()))["name"] == DEFAULT_NAME
    from pilot.workflows.routines import normalize_phrase

    assert server._phrase_reserved(normalize_phrase("what's your name"))
    assert not server._phrase_reserved(normalize_phrase("open chrome"))


def test_planner_appends_persona_to_system_prompt():
    from unittest.mock import MagicMock

    from pilot.agents.planner import Planner

    planner = Planner(MagicMock(), MagicMock())
    before = planner._system_prompt
    planner.set_persona_context('Voice and persona: speak playfully. The user calls you "Nova".')
    assert "Nova" in planner._system_prompt and planner._system_prompt.startswith(before.split("\n")[0])
    planner.set_persona_context("")
    assert "Nova" not in planner._system_prompt
