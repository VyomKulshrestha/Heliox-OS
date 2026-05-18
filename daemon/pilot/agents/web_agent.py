"""Web Agent — handles browsing, scraping, and information retrieval.

Specializes in all web-related interactions: browser automation,
web scraping, API calls, downloads, and online search.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.web_agent")

WEB_ACTION_TYPES: set[ActionType] = {
    # Browser automation
    ActionType.BROWSER_NAVIGATE,
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_HOVER,
    ActionType.BROWSER_SCROLL,
    ActionType.BROWSER_EXTRACT,
    ActionType.BROWSER_EXTRACT_TABLE,
    ActionType.BROWSER_EXTRACT_LINKS,
    ActionType.BROWSER_EXECUTE_JS,
    ActionType.BROWSER_SCREENSHOT,
    ActionType.BROWSER_FILL_FORM,
    ActionType.BROWSER_NEW_TAB,
    ActionType.BROWSER_CLOSE_TAB,
    ActionType.BROWSER_LIST_TABS,
    ActionType.BROWSER_SWITCH_TAB,
    ActionType.BROWSER_BACK,
    ActionType.BROWSER_FORWARD,
    ActionType.BROWSER_REFRESH,
    ActionType.BROWSER_WAIT,
    ActionType.BROWSER_CLOSE,
    ActionType.BROWSER_PAGE_INFO,
    # URL / downloads
    ActionType.OPEN_URL,
    ActionType.DOWNLOAD_FILE,
    # API
    ActionType.API_REQUEST,
    ActionType.API_GITHUB,
    ActionType.API_SCRAPE,
}


class WebAgent(BaseAgent):
    """Specialist agent for all web-related operations."""

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.WEB, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.BROWSER_NAVIGATE,
                description="Navigate to a URL in an automated browser",
            ),
            AgentCapability(
                action_type=ActionType.BROWSER_EXTRACT,
                description="Extract text/data from a web page",
            ),
            AgentCapability(
                action_type=ActionType.API_REQUEST,
                description="Make HTTP API requests (GET, POST, etc.)",
            ),
            AgentCapability(
                action_type=ActionType.API_SCRAPE,
                description="Scrape structured data from web pages",
            ),
            AgentCapability(
                action_type=ActionType.DOWNLOAD_FILE,
                description="Download files from the internet",
            ),
            AgentCapability(
                action_type=ActionType.API_GITHUB,
                description="Interact with the GitHub API",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the WEB AGENT for Heliox OS. "
            "You handle all web-related tasks: browser automation, web scraping, "
            "HTTP API calls, file downloads, and online information retrieval. "
            "For complex web pages, use browser automation tools. "
            "For simple data fetching, use direct HTTP requests. "
            "Always respect robots.txt and rate limits. "
            "When extracting data, return it in structured format (JSON/tables)."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in WEB_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute web-related actions."""
        import time

        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        sub_plan = ActionPlan(
            actions=my_actions,
            explanation=f"Web Agent executing {len(my_actions)} action(s)",
            raw_input=user_input,
        )

        results = await self._executor.execute(sub_plan)
        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results
