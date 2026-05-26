"""Forensics Agent — parses logs, inspects event sequences, and detects anomalies."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent
from pilot.agents.registry import auto_register

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.forensics_agent")

FORENSICS_ACTION_TYPES: set[ActionType] = {
    ActionType.LOG_ANALYZE,
}

@auto_register
class ForensicsAgent(BaseAgent):
    """Specialist agent for log forensics, incident timeline mapping, and anomaly detection."""

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.FORENSICS, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.LOG_ANALYZE,
                description="Parse system and service logs for anomalous activity, failed logins, and resource spikes.",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the FORENSICS AGENT for Heliox OS. "
            "You specialize in operational visibility, log forensics, and automated incident investigation. "
            "Your job is to read and parse logs (syslog, auth, nginx, apache, Windows Event exports, etc.), "
            "chronologically correlate session events, detect anomalies (like brute force login attempts, service restart loops, "
            "permission failures, or resource spikes), and summarize findings into structured incident reports. "
            "Always follow best security investigation guidelines: present timelines, pinpoint root causes, and propose recommended action pathways."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in FORENSICS_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute forensics-related log analysis tasks."""
        start = time.time()
        self.status = AgentStatus.BUSY

        # Filter to only actions this agent owns
        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        # Build a sub-plan with only our actions
        sub_plan = ActionPlan(
            actions=my_actions,
            explanation=f"Forensics Agent executing {len(my_actions)} log analysis action(s)",
            raw_input=user_input,
        )

        results = await self._executor.execute(sub_plan)
        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results
