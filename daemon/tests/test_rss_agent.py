"""Unit tests for pilot.agents.rss_agent.RssAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.agents.rss_agent import RssAgent

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <description>First article body.</description>
      <link>https://example.com/1</link>
    </item>
    <item>
      <title>Article Two</title>
      <description>Second article body.</description>
      <link>https://example.com/2</link>
    </item>
  </channel>
</rss>
"""

ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Entry One</title>
    <summary>Atom summary one.</summary>
    <link href="https://example.com/a1"/>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <summary>Atom summary two.</summary>
    <link href="https://example.com/a2"/>
  </entry>
</feed>
"""


@dataclass
class FakeRSSConfig:
    enabled: bool = False
    feeds: list[str] = field(default_factory=list)
    poll_interval_hours: float = 24.0
    max_items_per_feed: int = 10


@dataclass
class FakePilotConfig:
    rss: FakeRSSConfig = field(default_factory=FakeRSSConfig)

    def save(self) -> None:
        pass


def make_agent(
    enabled: bool = False,
    feeds: list[str] | None = None,
    llm_response: str = "Today's news digest.",
) -> tuple[RssAgent, MagicMock, AsyncMock]:
    model_router = MagicMock()
    model_router.generate = AsyncMock(return_value=llm_response)

    memory = MagicMock()
    memory.set_preference = AsyncMock()
    memory.get_preference = AsyncMock(return_value=None)

    config = FakePilotConfig(rss=FakeRSSConfig(enabled=enabled, feeds=feeds or []))

    bg_manager = MagicMock()
    bg_manager.register = MagicMock()
    bg_manager.start = MagicMock()

    agent = RssAgent(model_router, memory, config, bg_manager)
    return agent, bg_manager, memory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_rss_items():
    """_parse_feed correctly extracts items from RSS 2.0 XML."""
    items = RssAgent._parse_feed(RSS_XML, "https://example.com/feed.xml")
    assert len(items) == 2
    assert "Article One" in items[0]
    assert "First article body." in items[0]
    assert "https://example.com/1" in items[0]


def test_parse_atom_items():
    """_parse_feed correctly extracts entries from Atom 1.0 XML."""
    items = RssAgent._parse_feed(ATOM_XML, "https://example.com/atom.xml")
    assert len(items) == 2
    assert "Atom Entry One" in items[0]
    assert "Atom summary one." in items[0]
    assert "https://example.com/a1" in items[0]


def test_no_background_task_when_disabled():
    """No background task is registered when RSS is disabled."""
    _, bg_manager, _ = make_agent(enabled=False, feeds=["https://example.com/feed.xml"])
    bg_manager.register.assert_not_called()


def test_background_task_registered_when_enabled():
    """A background task is registered and started when RSS is enabled with feeds."""
    _, bg_manager, _ = make_agent(enabled=True, feeds=["https://example.com/feed.xml"])
    bg_manager.register.assert_called_once()
    bg_manager.start.assert_called_once_with("rss_feed_poller")


@pytest.mark.asyncio
async def test_poll_feeds_stores_digest():
    """_poll_feeds fetches feed XML, calls LLM, and stores digest via set_preference."""
    agent, _, memory = make_agent(
        enabled=True,
        feeds=["https://example.com/feed.xml"],
        llm_response="Summarized news today.",
    )

    mock_response = MagicMock()
    mock_response.text = RSS_XML
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("pilot.agents.rss_agent.httpx.AsyncClient", return_value=mock_client):
        result = await agent._poll_feeds()

    assert result["triggered"] is True
    assert "Summarized news today." in result["digest"]
    memory.set_preference.assert_called_once()
    key_arg = memory.set_preference.call_args[0][0]
    assert key_arg.startswith("rss_digest_")
