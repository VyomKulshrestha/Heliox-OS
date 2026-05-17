"""RSS/Atom feed polling agent with background news summarization."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import httpx

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.background import BackgroundTask
from pilot.agents.base_agent import AgentCapability, AgentRole, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.background import BackgroundTaskManager
    from pilot.config import PilotConfig
    from pilot.memory.store import MemoryStore
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.rss_agent")

_ATOM_NS = "http://www.w3.org/2005/Atom"
_FETCH_TIMEOUT = 10.0

_SUMMARIZE_PROMPT = """\
You are a news digest assistant. Summarize the following RSS feed items in under 300 words.
Group related stories, highlight the most important developments, and keep the language concise.

Feed items:
{items}

Write the summary as plain text paragraphs, no bullet points or markdown."""


class RssAgent(BaseAgent):
    """Periodically polls subscribed RSS/Atom feeds and stores daily digests in MemoryStore."""

    def __init__(
        self,
        model_router: ModelRouter,
        memory: MemoryStore,
        config: PilotConfig,
        background_manager: BackgroundTaskManager,
    ) -> None:
        super().__init__(role=AgentRole.RSS, model_router=model_router)
        self._memory = memory
        self._config = config
        self._bg = background_manager

        if config.rss.enabled and config.rss.feeds:
            self._register_background_task()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.API_REQUEST,
                description="Poll RSS/Atom feeds and return news summaries",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the RSS AGENT for Heliox OS. "
            "You manage RSS and Atom feed subscriptions, poll them periodically, "
            "and produce concise daily news digests stored in long-term memory."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type == ActionType.API_REQUEST

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        lower = user_input.lower()

        if any(k in lower for k in ["today", "digest", "summary", "headlines"]):
            key = f"rss_digest_{date.today().isoformat()}"
            digest = await self._memory.get_preference(key)
            if digest:
                return [ActionResult(action_type=ActionType.API_REQUEST, success=True, output=digest)]
            return [
                ActionResult(
                    action_type=ActionType.API_REQUEST,
                    success=False,
                    output="No RSS digest available for today. Try 'summarize feeds now' to fetch one.",
                )
            ]

        if "summarize" in lower or "fetch" in lower or "poll" in lower:
            result = await self._poll_feeds()
            digest = result.get("digest", "No items found.")
            return [ActionResult(action_type=ActionType.API_REQUEST, success=True, output=digest)]

        if "subscribe" in lower or "add feed" in lower or "add rss" in lower:
            words = user_input.split()
            urls = [w for w in words if w.startswith("http")]
            if not urls:
                return [
                    ActionResult(
                        action_type=ActionType.API_REQUEST,
                        success=False,
                        output="Please provide a feed URL to subscribe to.",
                    )
                ]
            self._config.rss.feeds.extend(urls)
            if not self._config.rss.enabled:
                self._config.rss.enabled = True
                self._register_background_task()
            self._config.save()
            return [
                ActionResult(
                    action_type=ActionType.API_REQUEST,
                    success=True,
                    output=f"Subscribed to: {', '.join(urls)}",
                )
            ]

        if "list" in lower or "feeds" in lower:
            feeds = self._config.rss.feeds
            if not feeds:
                return [
                    ActionResult(
                        action_type=ActionType.API_REQUEST,
                        success=True,
                        output="No feeds subscribed yet.",
                    )
                ]
            return [
                ActionResult(
                    action_type=ActionType.API_REQUEST,
                    success=True,
                    output="Subscribed feeds:\n" + "\n".join(f"  - {f}" for f in feeds),
                )
            ]

        return [
            ActionResult(
                action_type=ActionType.API_REQUEST,
                success=False,
                output="RSS Agent: unrecognized request. Try 'show today's digest', 'list my feeds', or 'subscribe to <url>'.",
            )
        ]

    # ------------------------------------------------------------------
    # Background task
    # ------------------------------------------------------------------

    def _register_background_task(self) -> None:
        interval = self._config.rss.poll_interval_hours * 3600
        self._bg.register(
            BackgroundTask(
                task_id="rss_feed_poller",
                name="RSS Feed Poller",
                description="Polls subscribed RSS/Atom feeds and stores daily summaries",
                interval_seconds=interval,
                action_fn=self._poll_feeds,
                on_trigger=self._on_new_digest,
            )
        )
        self._bg.start("rss_feed_poller")
        logger.info(
            "RSS Feed Poller registered (%d feeds, %.1fh interval)",
            len(self._config.rss.feeds),
            self._config.rss.poll_interval_hours,
        )

    async def _poll_feeds(self) -> dict[str, Any]:
        items: list[str] = []
        for feed_url in self._config.rss.feeds:
            try:
                feed_items = await self._fetch_feed(feed_url)
                items.extend(feed_items[: self._config.rss.max_items_per_feed])
            except Exception as exc:
                logger.warning("Failed to fetch feed %s: %s", feed_url, exc)

        if not items:
            return {"triggered": False, "digest": "No feed items retrieved."}

        combined = "\n\n".join(items)
        prompt = _SUMMARIZE_PROMPT.format(items=combined)
        try:
            digest = await self._model.generate(prompt, temperature=0.3)
        except Exception as exc:
            logger.warning("RSS summarization failed: %s", exc)
            digest = combined[:4000]

        key = f"rss_digest_{date.today().isoformat()}"
        await self._memory.set_preference(key, digest)
        logger.info("RSS digest stored under key '%s'", key)
        return {"triggered": True, "message": "New RSS digest ready", "digest": digest}

    async def _on_new_digest(self, result: dict[str, Any]) -> None:
        logger.info("RSS digest broadcast: %s", result.get("message", ""))

    # ------------------------------------------------------------------
    # Feed fetching and parsing
    # ------------------------------------------------------------------

    async def _fetch_feed(self, url: str) -> list[str]:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "HeliOS-RSSAgent/1.0"})
            resp.raise_for_status()
        return self._parse_feed(resp.text, url)

    @staticmethod
    def _parse_feed(xml_text: str, source_url: str) -> list[str]:
        """Parse RSS 2.0 or Atom 1.0 XML and return a list of formatted item strings."""
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as exc:
            logger.warning("Failed to parse feed XML from %s: %s", source_url, exc)
            return []

        items: list[str] = []

        # RSS 2.0
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            text = f"[{title}] {desc}"
            if link:
                text += f" ({link})"
            if text.strip():
                items.append(text)

        if items:
            return items

        # Atom 1.0
        ns = _ATOM_NS
        for entry in root.findall(f"{{{ns}}}entry"):
            title_el = entry.find(f"{{{ns}}}title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary_el = entry.find(f"{{{ns}}}summary")
            if summary_el is None:
                summary_el = entry.find(f"{{{ns}}}content")
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            link_el = entry.find(f"{{{ns}}}link")
            link = (link_el.get("href") or "").strip() if link_el is not None else ""
            text = f"[{title}] {summary}"
            if link:
                text += f" ({link})"
            if text.strip():
                items.append(text)

        return items
