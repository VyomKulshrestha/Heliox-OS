"""A feed on the person's own network is fetched only when the person approved it, and only from its own host.

Fakes only (tests/fake_outbound.py and the stand-ins of tests/test_rss_agent.py): no feed is fetched for real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import httpx

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.agents.rss_agent import RssAgent
from pilot.config import RSSConfig
from pilot.security import guard_changes
from tests import fake_outbound

FEED = '<?xml version="1.0"?><rss><channel><item><title>Backup done</title></item></channel></rss>'


@dataclass
class _Config:
    rss: RSSConfig = field(default_factory=RSSConfig)

    def save(self) -> None:
        return None


def _agent(feeds: list[str], local_feeds: list[str] | None = None) -> RssAgent:
    router = MagicMock()
    router.generate = AsyncMock(return_value="digest")
    memory = MagicMock()
    memory.set_preference = AsyncMock()
    config = _Config(rss=RSSConfig(enabled=False, feeds=list(feeds), local_feeds=list(local_feeds or [])))
    return RssAgent(router, memory, config, MagicMock())


def _plan(raw_input: str) -> ActionPlan:
    return ActionPlan(actions=[Action(action_type=ActionType.NOTIFY, parameters=EmptyParams())], raw_input=raw_input)


async def test_a_feed_named_in_the_persons_own_request_is_approved_for_its_host(monkeypatch) -> None:
    agent = _agent([])
    seen: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: seen.append(str(r.url)) or httpx.Response(200, text=FEED))

    await agent.handle_task(
        "subscribe http://nas.local/feed.xml", _plan("subscribe to http://nas.local/feed.xml please")
    )
    result = await agent._poll_feeds()

    assert agent._config.rss.local_feeds == ["http://nas.local/feed.xml"]
    assert result["needs_confirmation"] == []
    assert len(seen) == 1


async def test_a_feed_address_the_person_did_not_name_waits_for_confirmation(monkeypatch) -> None:
    agent = _agent([])
    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200, text=FEED))

    results = await agent.handle_task(
        "subscribe http://192.168.1.20/rss", _plan("subscribe to the feed the newsletter mentioned")
    )

    assert agent._config.rss.local_feeds == []
    assert "http://192.168.1.20/rss" in agent._config.rss.feeds
    assert "until you confirm it" in results[0].output


async def test_an_existing_local_feed_is_kept_not_fetched_and_said_once(monkeypatch) -> None:
    agent = _agent(["http://router.lan/feed", "https://news.example/rss"])
    seen: list[str] = []
    fake_outbound.install(
        monkeypatch, handler=lambda r: seen.append(r.headers["host"]) or httpx.Response(200, text=FEED)
    )

    first = await agent._poll_feeds()
    second = await agent._poll_feeds()

    assert seen == ["news.example", "news.example"]
    assert len(first["needs_confirmation"]) == 1 and "router.lan" in first["needs_confirmation"][0]
    assert second["needs_confirmation"] == []
    assert agent._config.rss.feeds == ["http://router.lan/feed", "https://news.example/rss"]


async def test_the_feed_list_marks_a_feed_waiting_for_confirmation() -> None:
    agent = _agent(["http://router.lan/feed", "http://nas.local/ok"], ["http://nas.local/ok"])

    results = await agent.handle_task("list my feeds", _plan("list my feeds"))

    lines = results[0].output.splitlines()
    assert any("router.lan" in line and "waiting for your confirmation" in line for line in lines)
    assert any("nas.local" in line and "waiting" not in line for line in lines)


async def test_an_approved_local_feed_is_not_followed_to_another_host(monkeypatch) -> None:
    agent = _agent(["http://nas.local/feed"], ["http://nas.local/feed"])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["host"])
        return httpx.Response(302, headers={"location": "http://10.0.0.9/admin"})

    fake_outbound.install(monkeypatch, handler=handler, answers={"nas.local": ["192.168.1.5"]})

    result = await agent._poll_feeds()

    assert seen == ["nas.local"]
    assert result["triggered"] is False


def test_confirming_a_local_feed_in_the_window_is_an_approval() -> None:
    now = RSSConfig(feeds=["http://router.lan/feed"])
    found = guard_changes.section_widenings("rss", now, {"local_feeds": ["http://router.lan/feed"]})
    assert [w.key for w in found] == ["rss.local_feeds"]
    assert "router.lan" in found[0].says

    approved = RSSConfig(feeds=["http://router.lan/feed"], local_feeds=["http://router.lan/feed"])
    assert guard_changes.section_widenings("rss", approved, {"local_feeds": ["http://router.lan/feed"]}) == []
