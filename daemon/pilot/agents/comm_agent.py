"""Communication Agent — handles integrations like Discord, Slack, email, webhooks.

Specializes in all outbound communication: sending messages through
various channels, managing webhooks, and handling notifications.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride

logger = logging.getLogger("pilot.agents.comm_agent")

COMM_ACTION_TYPES: set[ActionType] = {
    ActionType.API_SEND_EMAIL,
    ActionType.API_WEBHOOK,
    ActionType.API_SLACK,
    ActionType.API_DISCORD,
    ActionType.NOTIFY,
}


class CommunicationAgent(BaseAgent):
    """Specialist agent for external communications and notifications."""

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.COMMUNICATION, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.API_SEND_EMAIL,
                description="Send emails via SMTP or API",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.API_SLACK,
                description="Send messages to Slack channels",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.API_DISCORD,
                description="Send messages to Discord channels",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.API_WEBHOOK,
                description="Trigger webhooks with custom payloads",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.NOTIFY,
                description="Send local desktop notifications",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the COMMUNICATION AGENT for Heliox OS. "
            "You handle all outbound communications: sending emails, "
            "posting to Slack/Discord, triggering webhooks, and desktop "
            "notifications. You ALWAYS confirm sensitive communications "
            "with the user before sending. You format messages appropriately "
            "for each platform (Markdown for Discord, blocks for Slack, etc)."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in COMM_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        """Execute communication-related actions."""
        import time

        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        sub_plan = ActionPlan(
            actions=my_actions,
            explanation=f"Communication Agent executing {len(my_actions)} action(s)",
            raw_input=user_input,
        )

        results = await self._executor.execute(sub_plan, scope_override=scope_override)
        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results
