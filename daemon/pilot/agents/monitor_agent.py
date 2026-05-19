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

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.agents.monitor_agent")

MONITOR_ACTION_TYPES: set[ActionType] = {
    ActionType.SYSTEM_INFO,
    ActionType.CPU_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.DISK_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
}


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
        return [
            AgentCapability(
                action_type=ActionType.CPU_USAGE,
                description="Monitor CPU usage with threshold alerting",
            ),
            AgentCapability(
                action_type=ActionType.MEMORY_USAGE,
                description="Monitor RAM usage with threshold alerting",
            ),
            AgentCapability(
                action_type=ActionType.DISK_USAGE,
                description="Monitor disk space with threshold alerting",
            ),
            AgentCapability(
                action_type=ActionType.NETWORK_INFO,
                description="Monitor network activity and connectivity",
            ),
            AgentCapability(
                action_type=ActionType.BATTERY_INFO,
                description="Monitor battery level and charging status",
            ),
        ]

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
    ) -> list[ActionResult]:
        """Handle monitoring-related tasks by delegating to BackgroundTaskManager."""
        import time

        start = time.time()
        self.status = AgentStatus.BUSY
        results: list[ActionResult] = []

        input_lower = user_input.lower()

        # Detect intent: start, stop, or status
        if any(kw in input_lower for kw in ["start", "begin", "enable", "turn on"]):
            task_id = self._detect_monitor_type(input_lower)
            ok = self._bg.start(task_id)
            results.append(
                ActionResult(
                    action=plan.actions[0] if plan.actions else _dummy_action(),
                    success=ok,
                    output=f"Monitor '{task_id}' started" if ok else f"Failed to start '{task_id}'",
                )
            )
        elif any(kw in input_lower for kw in ["stop", "disable", "turn off"]):
            task_id = self._detect_monitor_type(input_lower)
            ok = self._bg.stop(task_id)
            results.append(
                ActionResult(
                    action=plan.actions[0] if plan.actions else _dummy_action(),
                    success=ok,
                    output=f"Monitor '{task_id}' stopped" if ok else f"Failed to stop '{task_id}'",
                )
            )
        else:
            # Default: list all monitors
            tasks = self._bg.list_tasks()
            results.append(
                ActionResult(
                    action=plan.actions[0] if plan.actions else _dummy_action(),
                    success=True,
                    output=f"Active monitors: {tasks}",
                )
            )

        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results

    def _detect_monitor_type(self, text: str) -> str:
        """Heuristically detect which monitor the user is referring to."""
        if "cpu" in text:
            return "cpu_monitor"
        if "memory" in text or "ram" in text:
            return "memory_monitor"
        if "disk" in text or "storage" in text:
            return "disk_monitor"
        if "network" in text or "net" in text:
            return "network_monitor"
        return "cpu_monitor"  # default

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


def _dummy_action():
    """Create a minimal Action for results when no plan actions exist."""
    from pilot.actions import Action, EmptyParams

    return Action(
        action_type=ActionType.SYSTEM_INFO,
        target="monitor",
        parameters=EmptyParams(),
    )
