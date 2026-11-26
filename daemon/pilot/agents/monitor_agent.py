"""Monitor Agent — runs background monitoring tasks (CPU, disk, network).

Wraps the existing BackgroundTaskManager and extends it with
agent-protocol compliance: structured messaging, orchestrator
integration, and dynamic spawning of monitoring loops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import (
    AgentCapability,
    AgentMessage,
    AgentRole,
    AgentStatus,
    BaseAgent,
)

if TYPE_CHECKING:
    from pilot.agents.background import BackgroundTaskManager
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride

logger = logging.getLogger("pilot.agents.monitor_agent")

# The monitor agent starts, stops and lists background monitors through its messages. It runs no
# plan step: the one-off readings (cpu_usage, memory_usage, disk_usage, network_info, battery_info)
# are the executor's, reached through the system and network agents. Declaring them here once sent
# "how much battery" to the monitor list.
MONITOR_ACTION_TYPES: frozenset[ActionType] = frozenset()


class MonitorAgent(BaseAgent):
    """Specialist agent for continuous system monitoring and alerting."""

    def __init__(
        self,
        model_router: ModelRouter,
        background_manager: BackgroundTaskManager,
    ) -> None:
        super().__init__(role=AgentRole.MONITOR, model_router=model_router)
        self._bg = background_manager
        # Register custom message handlers
        self._message_handlers["start_monitor"] = self._handle_start_monitor
        self._message_handlers["stop_monitor"] = self._handle_stop_monitor
        self._message_handlers["list_monitors"] = self._handle_list_monitors

    def get_capabilities(self) -> list[AgentCapability]:
        return []

    def get_system_prompt(self) -> str:
        return (
            "You are the MONITOR AGENT for Heliox OS. "
            "You manage persistent background monitoring tasks for system health: "
            "CPU, memory, disk, network, and battery. You set up threshold-based "
            "alerts and can run continuous watch loops. When a metric exceeds a "
            "threshold, you notify the user and can trigger automated responses. "
            "You can also relay monitoring data to other agents for analysis."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in MONITOR_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        """Refuse every plan step: this agent declares none (see MONITOR_ACTION_TYPES).

        Nothing routes a step here; a caller that does gets a failed result for each step, never
        the monitor list in place of the reading that was asked for.
        """
        return [
            ActionResult(
                action=action,
                success=False,
                error=f"The monitor agent does not run {action.action_type.value}.",
            )
            for action in plan.actions
        ]

    # ── Message handlers ──

    async def _handle_start_monitor(self, msg: AgentMessage) -> AgentMessage:
        task_id = msg.payload.get("task_id", "cpu_monitor")
        ok = self._bg.start(task_id)
        return msg.reply({"started": ok, "task_id": task_id})

    async def _handle_stop_monitor(self, msg: AgentMessage) -> AgentMessage:
        task_id = msg.payload.get("task_id", "")
        ok = self._bg.stop(task_id)
        return msg.reply({"stopped": ok, "task_id": task_id})

    async def _handle_list_monitors(self, msg: AgentMessage) -> AgentMessage:
        tasks = self._bg.list_tasks()
        return msg.reply({"tasks": tasks})
