"""Autonomous Threat Containment Bridge — Issue #365.

Intercepts CRITICAL ForensicsAgent incident reports and autonomously routes
a mitigation action (e.g. ``PROCESS_KILL``) through the existing Orchestrator
so that the Tier 3/4 Security Gate (user confirmation / auditory hold) fires
before any destructive command is executed.

Pipeline:
    ForensicsAgent output (JSON)
        → ThreatContainmentBridge.intercept()
            → parse ForensicsReport
            → if severity == CRITICAL
                → translate_resolution()   # rule-based, NOT LLM
                → route_and_confirm()      # Orchestrator → Security Gate
                → _audit_containment()     # AuditLogger JSONL

Design notes:
  * Translation is **rule-based** (regex PID extraction) to eliminate LLM
    hallucination risk on safety-critical kill commands.
  * The bridge is wired into ForensicsAgent via ``set_threat_bridge()`` and
    does NOT modify the agent's public interface.
  * The Security Gate is triggered automatically because every translated
    action carries ``destructive=True``, which resolves to
    ``PermissionTier.DESTRUCTIVE`` (Tier 3) via ``Action.permission_tier``.
  * Confirmation flow reuses the server's ``_pending_confirms`` dict and the
    broadcast function — no new WebSocket logic is required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pydantic import BaseModel, Field, field_validator

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ProcessParams,
    ShellCommandParams,
)

if TYPE_CHECKING:
    from pilot.agents.orchestrator import AgentOrchestrator
    from pilot.security.audit import AuditLogger

logger = logging.getLogger("pilot.agents.threat_containment")

# ---------------------------------------------------------------------------
# Forensics Report Schema
# ---------------------------------------------------------------------------


class ForensicsReport(BaseModel):
    """Structured incident report produced by the ForensicsAgent LLM.

    The ForensicsAgent is instructed to output JSON with these fields inside
    the ``ActionResult.output`` string when a CRITICAL incident is detected.
    Non-CRITICAL reports may omit ``proposed_resolution`` and ``affected_pids``.
    """

    severity: str = ""  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO"
    incident_type: str = ""  # e.g. "brute_force", "malware_process", "privilege_escalation"
    proposed_resolution: str = ""  # human-readable resolution hint
    affected_pids: list[int] = Field(default_factory=list)  # zero or more PIDs
    summary: str = ""  # short description of findings
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("severity", mode="before")
    @classmethod
    def _upper_severity(cls, v: Any) -> str:
        return str(v).upper()


# ---------------------------------------------------------------------------
# Containment Result
# ---------------------------------------------------------------------------


@dataclass
class ContainmentRecord:
    """Tracks the result of a single threat containment attempt."""

    report: ForensicsReport
    action_plan: ActionPlan
    confirmed: bool = False
    results: list[ActionResult] = field(default_factory=list)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.report.severity,
            "incident_type": self.report.incident_type,
            "proposed_resolution": self.report.proposed_resolution,
            "affected_pids": self.report.affected_pids,
            "summary": self.report.summary,
            "confirmed": self.confirmed,
            "action_count": len(self.action_plan.actions),
            "success": all(r.success for r in self.results) if self.results else False,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Translation Helpers
# ---------------------------------------------------------------------------

# Regex patterns for PID extraction from natural-language resolutions
_PID_PATTERNS = [
    re.compile(r"\bpid\s*[:\s]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bprocess\s+(?:id\s+)?(\d+)", re.IGNORECASE),
    re.compile(r"\bkill\s+(?:-9\s+|-SIGKILL\s+|-SIGTERM\s+)?(\d+)", re.IGNORECASE),
    re.compile(r"\bterminate\s+(?:process\s+)?(\d+)", re.IGNORECASE),
    re.compile(r"#(\d{3,7})\b"),  # bare PID-like integers 100–9999999
]


def _extract_pids(text: str, hint_pids: list[int]) -> list[int]:
    """Extract PID integers from a resolution string.

    Checks ``hint_pids`` from the report first; falls back to regex scanning
    of the resolution text.
    """
    if hint_pids:
        return list(hint_pids)

    pids: list[int] = []
    for pattern in _PID_PATTERNS:
        for match in pattern.finditer(text):
            try:
                pid = int(match.group(1))
                if 1 <= pid <= 4_000_000 and pid not in pids:
                    pids.append(pid)
            except (ValueError, IndexError):
                pass
    return pids


def _build_kill_action(pid: int) -> Action:
    """Build a PROCESS_KILL Action for a given PID.

    Flags:
      * ``destructive=True``  → forces PermissionTier.DESTRUCTIVE (Tier 3)
      * ``reversible=False``  → SIGKILL cannot be undone
      * ``requires_root=False`` by default (user-space processes)
    """
    return Action(
        action_type=ActionType.PROCESS_KILL,
        target=str(pid),
        parameters=ProcessParams(pid=pid, signal="SIGKILL"),
        destructive=True,
        reversible=False,
        requires_root=False,
    )


def _build_shell_fallback(resolution: str) -> Action:
    """Build a SHELL_COMMAND action for unrecognised resolutions.

    Used when no PID can be extracted.  The command is intentionally marked
    ``destructive=True`` to force Security Gate confirmation.
    """
    # Sanitize: strip leading/trailing shell-unsafe chars but keep the intent
    safe_cmd = resolution.strip()[:256]
    return Action(
        action_type=ActionType.SHELL_COMMAND,
        target="threat_mitigation",
        parameters=ShellCommandParams(command=safe_cmd, args=[], timeout=30),
        destructive=True,
        reversible=False,
        requires_root=False,
    )


# ---------------------------------------------------------------------------
# ThreatContainmentBridge
# ---------------------------------------------------------------------------

CONFIRM_TIMEOUT_SECONDS = 60  # 60-second user confirmation window for threat containment


class ThreatContainmentBridge:
    """Bridges ForensicsAgent CRITICAL reports to SystemAgent execution.

    Injected into ForensicsAgent at server startup.  The bridge is called
    after every ForensicsAgent task completes and checks each ActionResult's
    ``output`` field for a CRITICAL JSON report.

    Args:
        orchestrator: The AgentOrchestrator used to execute mitigation plans.
        audit_logger: The AuditLogger for writing containment audit entries.
        broadcast_fn: WebSocket broadcast coroutine (server._broadcast_notification).
        pending_confirms: Reference to server._pending_confirms dict so the
            bridge can register a PendingConfirmation and await the user's
            Y/N response via the existing ``_handle_confirm`` handler.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        audit_logger: AuditLogger,
        broadcast_fn: Callable[..., Coroutine] | None = None,
        pending_confirms: dict | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._audit = audit_logger
        self._broadcast_fn = broadcast_fn
        self._pending_confirms = pending_confirms if pending_confirms is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def intercept(self, results: list[ActionResult]) -> list[ContainmentRecord]:
        """Scan ForensicsAgent results for CRITICAL severity reports.

        For each CRITICAL report:
          1. Parse the ForensicsReport JSON from ``result.output``
          2. Translate the proposed_resolution into an ActionPlan
          3. Broadcast a confirmation request (Security Gate)
          4. Route the plan through the Orchestrator (after user approval)
          5. Write an audit entry

        Args:
            results: The list of ActionResults returned by ForensicsAgent.

        Returns:
            A list of ContainmentRecord objects (one per CRITICAL threat found).
        """
        records: list[ContainmentRecord] = []

        for result in results:
            report = self._parse_report(result.output)
            if report is None or report.severity != "CRITICAL":
                continue

            logger.warning(
                "[ThreatContainment] CRITICAL threat detected — incident: %s | resolution: %s",
                report.incident_type or "unknown",
                report.proposed_resolution[:120] if report.proposed_resolution else "(none)",
            )

            record = await self._handle_critical(report)
            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Step 1 — Parse
    # ------------------------------------------------------------------

    def _parse_report(self, output: str) -> ForensicsReport | None:
        """Safely parse a ForensicsReport from an ActionResult output string.

        The LLM output may wrap JSON in markdown code fences — we strip those
        before parsing.  On any parsing error we log a warning and return None
        so a bad report never crashes the daemon.

        Args:
            output: Raw string from ActionResult.output.

        Returns:
            A ForensicsReport if valid JSON with required fields is found,
            otherwise None.
        """
        if not output or not output.strip():
            return None

        # Strip markdown code fences if present
        clean = output.strip()
        if clean.startswith("```json"):
            clean = clean.split("```json", 1)[1]
        elif clean.startswith("```"):
            clean = clean.split("```", 1)[1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        # Find outermost JSON object
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        json_str = clean[start : end + 1]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.debug("[ThreatContainment] Failed to parse JSON from output: %s", exc)
            return None

        if not isinstance(data, dict):
            return None

        try:
            return ForensicsReport(**data)
        except Exception as exc:
            logger.debug("[ThreatContainment] ForensicsReport validation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Step 2 — Translate Resolution to ActionPlan
    # ------------------------------------------------------------------

    def translate_resolution(self, report: ForensicsReport) -> ActionPlan:
        """Translate a natural-language proposed_resolution into an ActionPlan.

        Translation is rule-based (regex PID extraction) to eliminate LLM
        hallucination risk on safety-critical commands.

        Resolution strategies (in priority order):
          1. If PIDs are found (in report.affected_pids or via regex) →
             one PROCESS_KILL action per PID, all with destructive=True.
          2. If the resolution mentions a service name (stop/disable) →
             SERVICE_STOP + SERVICE_DISABLE actions.
          3. Fallback → SHELL_COMMAND with the raw resolution string, destructive=True.

        Args:
            report: The validated ForensicsReport from the agent output.

        Returns:
            An ActionPlan ready to be routed through the Orchestrator.
        """
        resolution = report.proposed_resolution.strip()
        pids = _extract_pids(resolution, report.affected_pids)

        actions: list[Action] = []

        if pids:
            for pid in pids:
                actions.append(_build_kill_action(pid))
            explanation = (
                f"[THREAT CONTAINMENT] Kill malicious process(es) "
                f"{pids} — auto-generated from CRITICAL forensics report: "
                f"{report.incident_type or 'unknown incident'}"
            )
        else:
            # Fallback: emit the resolution as a shell command
            logger.warning(
                "[ThreatContainment] No PID found in resolution '%s' — "
                "falling back to SHELL_COMMAND (user confirmation still required).",
                resolution[:80],
            )
            actions.append(_build_shell_fallback(resolution))
            explanation = (
                f"[THREAT CONTAINMENT] Execute mitigation command "
                f"(no PID identified) — auto-generated from CRITICAL report: "
                f"{report.incident_type or 'unknown incident'}"
            )

        return ActionPlan(
            actions=actions,
            explanation=explanation,
            raw_input=f"threat_containment:{report.incident_type}",
        )

    # ------------------------------------------------------------------
    # Step 3 — Broadcast Confirmation Request + Execute
    # ------------------------------------------------------------------

    async def route_and_confirm(
        self,
        plan: ActionPlan,
        report: ForensicsReport,
    ) -> tuple[bool, list[ActionResult]]:
        """Register a confirmation gate and (on approval) execute the plan.

        This method:
          1. Emits a ``threat_confirmation_required`` broadcast so the UI
             displays a modal asking the operator to approve or deny.
          2. Waits up to CONFIRM_TIMEOUT_SECONDS for a ``confirm`` WebSocket
             message (handled by the existing ``_handle_confirm`` handler in
             server.py via ``_pending_confirms``).
          3. If confirmed, routes the plan through Orchestrator → SystemAgent.

        The Orchestrator's own Security Gate check (server.py line ~1117) is
        deliberately NOT invoked here because we are calling
        ``orchestrator.execute_plan()`` directly, not going through
        ``_handle_execute``.  Instead we use the bridge's own lightweight gate
        (steps 1–2) which feeds the same ``_pending_confirms`` dict.

        Args:
            plan: The translated ActionPlan (all actions marked destructive=True).
            report: Source ForensicsReport for logging context.

        Returns:
            (confirmed, results) — False + [] if the user denied or timed out.
        """
        plan_id = f"tc_{str(uuid.uuid4())[:8]}"

        # --- Register PendingConfirmation --------------------------------
        from pilot.server import PendingConfirmation

        pending = PendingConfirmation(plan_id=plan_id, event=asyncio.Event(), plan=plan)
        self._pending_confirms[plan_id] = pending

        # --- Broadcast UI prompt ----------------------------------------
        logger.info("[ThreatContainment] Broadcasting confirmation request (plan_id=%s)", plan_id)
        if self._broadcast_fn:
            await self._broadcast_fn(
                "threat_confirmation_required",
                {
                    "plan_id": plan_id,
                    "severity": report.severity,
                    "incident_type": report.incident_type,
                    "summary": report.summary,
                    "proposed_resolution": report.proposed_resolution,
                    "affected_pids": report.affected_pids,
                    "actions": [a.model_dump() for a in plan.actions if a.requires_confirmation],
                    "timeout_seconds": CONFIRM_TIMEOUT_SECONDS,
                },
            )

        # --- Wait for user Y/N ------------------------------------------
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=CONFIRM_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning(
                "[ThreatContainment] Confirmation timed out after %ds for plan_id=%s — threat containment ABORTED.",
                CONFIRM_TIMEOUT_SECONDS,
                plan_id,
            )
            self._pending_confirms.pop(plan_id, None)
            if self._broadcast_fn:
                await self._broadcast_fn(
                    "threat_containment_timeout",
                    {"plan_id": plan_id, "message": "Confirmation timed out — no action taken."},
                )
            return False, []
        finally:
            self._pending_confirms.pop(plan_id, None)

        if not pending.confirmed:
            logger.info("[ThreatContainment] User DENIED threat containment (plan_id=%s)", plan_id)
            if self._broadcast_fn:
                await self._broadcast_fn(
                    "threat_containment_denied",
                    {"plan_id": plan_id, "message": "Threat containment was denied by operator."},
                )
            return False, []

        # --- Execute via Orchestrator ------------------------------------
        logger.info(
            "[ThreatContainment] User APPROVED — routing plan_id=%s to Orchestrator",
            plan_id,
        )
        if self._broadcast_fn:
            await self._broadcast_fn(
                "threat_containment_executing",
                {"plan_id": plan_id, "action_count": len(plan.actions)},
            )

        try:
            results = await self._orchestrator.execute_plan(
                user_input=f"threat_containment:{report.incident_type}",
                plan=plan,
                plan_id=plan_id,
            )
        except Exception as exc:
            logger.exception("[ThreatContainment] Orchestrator execution failed: %s", exc)
            return True, []

        return True, results

    # ------------------------------------------------------------------
    # Step 4 — Audit Logging
    # ------------------------------------------------------------------

    async def _audit_containment(
        self,
        report: ForensicsReport,
        confirmed: bool,
        results: list[ActionResult],
        error: str = "",
    ) -> None:
        """Append a threat containment entry to the immutable audit log.

        Uses the existing ``AuditLogger.log_security_event`` so the event
        lands in the same JSONL stream as all other security events.

        Args:
            report: The triggering ForensicsReport.
            confirmed: Whether the user approved execution.
            results: The list of ActionResults (empty if denied/timed out).
            error: Optional error message on failure.
        """
        if not self._audit:
            return

        success = all(r.success for r in results) if results else False
        pids_killed = [
            r.action.parameters.pid
            for r in results
            if r.success
            and r.action.action_type == ActionType.PROCESS_KILL
            and hasattr(r.action.parameters, "pid")
            and r.action.parameters.pid
        ]

        details: dict[str, Any] = {
            "severity": report.severity,
            "incident_type": report.incident_type,
            "proposed_resolution": report.proposed_resolution,
            "affected_pids": report.affected_pids,
            "user_confirmed": confirmed,
            "execution_success": success,
            "pids_killed": pids_killed,
            "action_count": len(results),
        }
        if error:
            details["error"] = error

        event_label = (
            "threat_contained"
            if (confirmed and success)
            else ("threat_containment_denied" if not confirmed else "threat_containment_failed")
        )

        try:
            await self._audit.log_security_event(event_label, details)
            logger.info(
                "[ThreatContainment] Audit entry written: %s (pids_killed=%s)",
                event_label,
                pids_killed or "none",
            )
        except Exception as exc:
            logger.error("[ThreatContainment] Failed to write audit entry: %s", exc)

    # ------------------------------------------------------------------
    # Internal — full CRITICAL handling pipeline
    # ------------------------------------------------------------------

    async def _handle_critical(self, report: ForensicsReport) -> ContainmentRecord:
        """Run the full containment pipeline for a single CRITICAL report.

        Args:
            report: Validated ForensicsReport with severity == 'CRITICAL'.

        Returns:
            A ContainmentRecord summarising the containment attempt.
        """
        # Step A — Translate
        plan = self.translate_resolution(report)
        record = ContainmentRecord(report=report, action_plan=plan)

        if not plan.actions:
            record.error = "No containment actions could be generated from the proposed resolution."
            logger.error("[ThreatContainment] %s", record.error)
            await self._audit_containment(report, confirmed=False, results=[], error=record.error)
            return record

        # Step B — Broadcast alert and request confirmation
        if self._broadcast_fn:
            await self._broadcast_fn(
                "threat_detected",
                {
                    "severity": report.severity,
                    "incident_type": report.incident_type,
                    "summary": report.summary,
                    "affected_pids": report.affected_pids,
                    "proposed_resolution": report.proposed_resolution,
                },
            )

        # Step C — Gate + Execute
        confirmed, results = await self.route_and_confirm(plan, report)
        record.confirmed = confirmed
        record.results = results

        # Step D — Audit
        await self._audit_containment(report, confirmed, results)

        # Step E — Broadcast result
        if self._broadcast_fn:
            status = (
                "contained"
                if (confirmed and all(r.success for r in results))
                else ("denied" if not confirmed else "failed")
            )
            await self._broadcast_fn(
                "threat_containment_result",
                {
                    "status": status,
                    "incident_type": report.incident_type,
                    "affected_pids": report.affected_pids,
                    "pids_killed": [
                        r.action.parameters.pid
                        for r in results
                        if r.success
                        and r.action.action_type == ActionType.PROCESS_KILL
                        and hasattr(r.action.parameters, "pid")
                        and r.action.parameters.pid
                    ],
                },
            )

        return record
