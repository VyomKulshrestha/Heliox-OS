"""The Gemini talk default (14 November): gemini-2.5-flash-lite was refused to a key new to it with 404 "no longer
available to new users", though the key's model list still showed it; the successor Google names,
gemini-3.5-flash-lite, refused Heliox's thinkingBudget 0 with 400 "Request contains an invalid argument" and answered
with thinkingLevel "minimal". And when a default is refused, the line said "leave it blank for the default", which is
the model that failed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.conversation.fast_model import FastModelUnavailable, HttpFastModel, build_fast_model, gemini_thinking_off
from pilot.conversation.layer import FALLBACK, ConversationLayer
from pilot.models import catalog


@pytest.mark.parametrize(
    ("name", "thinking"),
    [
        ("gemini-2.5-flash", {"thinkingBudget": 0}),
        ("gemini-2.5-flash-lite", {"thinkingBudget": 0}),
        ("gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
        ("gemini-3.8-flash", {"thinkingLevel": "minimal"}),
        ("gemini-flash-latest", {"thinkingLevel": "minimal"}),
        ("gemini-2.5-pro", None),
        ("gemini-4-carbon", None),
    ],
)
def test_thinking_is_turned_off_the_way_each_generation_takes_it(name, thinking):
    assert gemini_thinking_off(name) == thinking


def test_the_talk_request_for_the_new_default_sends_a_thinking_level_not_a_budget():
    model = HttpFastModel(provider="gemini", name=catalog.TALK_DEFAULTS["gemini"], client_factory=lambda: None)
    _url, headers, body = model.request("s", [{"role": "user", "content": "hi"}], "k")
    assert catalog.TALK_DEFAULTS["gemini"] == "gemini-3.5-flash-lite"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "thinkingBudget" not in str(body)


def test_the_old_default_is_marked_closed_to_new_accounts_with_google_s_successor():
    gone = catalog.withdrawn("gemini", "gemini-2.5-flash-lite")
    assert gone is not None and gone.state == "closed" and gone.successor == "gemini-3.5-flash-lite"
    assert "gemini-2.5-flash-lite" not in catalog.OFFERED["gemini"]
    assert "no longer offered to new Google accounts" in catalog.describe("gemini", "gemini-2.5-flash-lite")


@pytest.mark.parametrize(("setting", "chosen"), [("", False), ("gemini-3.8-flash", True)])
def test_the_model_knows_whether_the_person_chose_it(setting, chosen):
    config = PilotConfig()
    config.model.provider, config.model.cloud_provider = "cloud", "gemini"
    config.conversation.gemini_model = setting
    vault = SimpleNamespace(get_key=lambda provider: None)
    model = build_fast_model(config, vault=vault, router=object(), budget=None)
    assert model.chosen is chosen


class Refused:
    name, provider, streams, local, calls = "gemini-3.5-flash-lite", "gemini", True, False, []

    def __init__(self, chosen: bool) -> None:
        self.chosen = chosen

    async def stream(self, system: str, messages: list) -> AsyncIterator[str]:
        raise FastModelUnavailable("unknown_model", "404")
        yield ""  # pragma: no cover - makes this an async generator


@pytest.mark.parametrize(
    ("chosen", "line"),
    [
        (False, FALLBACK["unknown_default"].replace("{provider}", "Google")),
        (True, FALLBACK["unknown_model"].replace("{provider}", "Google")),
    ],
)
async def test_a_refused_default_says_so_and_never_says_to_leave_it_blank(chosen, line):
    heard: list[str] = []

    async def ear(text: str, channel: str, key: str) -> str:
        heard.append(text)
        return "spoken"

    model = Refused(chosen)
    await ConversationLayer(sink=ear, model=lambda: model, hold_after=60.0).take("Why is my laptop fan so loud?")
    assert heard == [line]
    if not chosen:
        assert "blank" not in heard[0] and "Settings, under Model" in heard[0]
