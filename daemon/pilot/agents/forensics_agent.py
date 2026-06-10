"""Forensics Agent — parses logs, inspects event sequences, and detects anomalies.

When a completed analysis returns a JSON report with ``"severity": "CRITICAL"``
the attached :class:`~pilot.agents.threat_containment.ThreatContainmentBridge`
autonomously extracts the ``proposed_resolution``, translates it into a
``PROCESS_KILL`` (or ``SHELL_COMMAND``) action, and routes it through the
Orchestrator.  The existing Tier 3/4 Security Gate forces user confirmation
before any destructive command is executed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent
from pilot.agents.registry import auto_register

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.agents.threat_containment import ThreatContainmentBridge
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
        # Injected at server startup via set_threat_bridge(); None = bridge disabled.
        self._threat_bridge: ThreatContainmentBridge | None = None

    def set_threat_bridge(self, bridge: ThreatContainmentBridge) -> None:
        """Attach the ThreatContainmentBridge so CRITICAL reports trigger auto-containment.

        Called once from :meth:`AgentOrchestrator.set_threat_bridge` after the
        bridge is constructed in ``server.py``.

        Args:
            bridge: The initialized :class:`ThreatContainmentBridge` instance.
        """
        self._threat_bridge = bridge
        logger.info("ThreatContainmentBridge attached to ForensicsAgent")

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
            "Always follow best security investigation guidelines: present timelines, pinpoint root causes, and propose recommended action pathways. "
            "When a CRITICAL threat is found, you MUST output findings as a structured JSON object enclosed in a ```json code block. "
            "The JSON object must contain the following fields:\n"
            '  - "severity": "CRITICAL"\n'
            '  - "incident_type": string (e.g., "brute_force", "malware_process", "privilege_escalation")\n'
            '  - "proposed_resolution": string (a clear action statement, e.g., "Kill process 1042")\n'
            '  - "affected_pids": list of integers containing the relevant process IDs\n'
            '  - "summary": string summarizing the incident'
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

        # ── Threat Containment Bridge (Issue #365) ──────────────────────────
        # Scan results for CRITICAL severity JSON reports and, if found,
        # autonomously route a mitigation action through the Security Gate.
        # The interception is scheduled as a background task so it never
        # delays the return of forensics results to the caller.
        if self._threat_bridge is not None:
            if not hasattr(self, "_bg_tasks"):
                self._bg_tasks = set()
            task = asyncio.create_task(
                self._intercept_critical_threats(results),
                name=f"threat_containment_{str(id(results))[:8]}",
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

        return results

    async def _intercept_critical_threats(self, results: list[ActionResult]) -> None:
        """Background task: hand CRITICAL ForensicsAgent results to the bridge.

        Any exception is caught and logged so it never propagates to the daemon.

        Args:
            results: The ActionResult list returned by the executor.
        """
        if self._threat_bridge is None:
            return
        try:
            containment_records = await self._threat_bridge.intercept(results)
            if containment_records:
                logger.info(
                    "[ForensicsAgent] %d CRITICAL threat(s) processed by ThreatContainmentBridge.",
                    len(containment_records),
                )
        except Exception:
            logger.exception("[ForensicsAgent] Unexpected error in ThreatContainmentBridge.intercept()")
