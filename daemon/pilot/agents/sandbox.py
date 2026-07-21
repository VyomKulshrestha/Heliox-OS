"""Simulation Sandbox — dry-run dangerous actions before execution.

Before executing destructive or high-risk commands, the sandbox
estimates the impact and asks for user confirmation with full
transparency of what will happen.

Impact analysis includes:
  - Files/directories affected
  - Estimated number of changes
  - Reversibility assessment
  - Risk score (low / medium / high / critical)

Architecture:
  Plan → Sandbox.simulate(plan) → ImpactReport → User Confirm → Execute
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("pilot.agents.sandbox")


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ImpactItem:
    """A single estimated impact of an action."""

    action_type: str = ""
    target: str = ""
    description: str = ""
    risk: str = RiskLevel.LOW
    reversible: bool = True
    estimated_scope: str = ""  # e.g., "154 files", "1 service"
    cognitive_cost: float = 0.0
    # Pre-execution target assessment (browser click/type/select/fill_form
    # only, and only when a browser session is already open) — set from
    # pilot.system.dom_diff.assess_target() against the CURRENT live DOM,
    # before the action would run. Empty string means "no prediction
    # available" (no active session, or the selector was too complex to
    # statically resolve), never "target confirmed fine".
    predicted_issue: str = ""
    predicted_issue_is_problem: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "description": self.description,
            "risk": self.risk,
            "reversible": self.reversible,
            "estimated_scope": self.estimated_scope,
            "cognitive_cost": self.cognitive_cost,
            "predicted_issue": self.predicted_issue,
        }


@dataclass
class SimulationReport:
    """Full impact report from a sandbox simulation."""

    plan_id: str = ""
    is_safe: bool = True
    overall_risk: str = RiskLevel.LOW
    total_cognitive_cost: float = 0.0
    impacts: list[ImpactItem] = field(default_factory=list)
    total_files_affected: int = 0
    requires_root: bool = False
    has_destructive: bool = False
    has_network: bool = False
    recommendation: str = "safe to execute"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "is_safe": self.is_safe,
            "overall_risk": self.overall_risk,
            "total_cognitive_cost": self.total_cognitive_cost,
            "impacts": [i.to_dict() for i in self.impacts],
            "total_files_affected": self.total_files_affected,
            "requires_root": self.requires_root,
            "has_destructive": self.has_destructive,
            "has_network": self.has_network,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
            "impact_count": len(self.impacts),
        }


# ── Risk classification rules ──
#
# These are plain-string-keyed sets local to the sandbox's own risk model —
# distinct from actions.py's enum-keyed DESTRUCTIVE_ACTIONS/etc. (which drive
# PermissionChecker's tier system). Prefixed SANDBOX_ to avoid the name
# collision/import-confusion risk between the two.
#
# Values below are checked against `action.action_type.value` (see
# simulate()) and must match real ActionType values exactly — several
# entries here (power_action, disk_manage, package_uninstall, registry_delete,
# wifi_control) previously didn't correspond to any real ActionType and were
# silently unreachable dead code; fixed while adding browser/system-control
# coverage below so the new coverage isn't built on the same broken pattern.

SANDBOX_DESTRUCTIVE_ACTIONS = {
    "file_delete",
    "shell_command",
    "shell_script",
    "process_kill",
    "service_stop",
    "service_restart",
    "power_shutdown",
    "power_restart",
    "registry_write",
    "disk_unmount",
    "package_remove",
    # Arbitrary script execution in page context — same bar as a shell command.
    "browser_execute_js",
}

SANDBOX_HIGH_RISK_ACTIONS = {
    "code_execute",
    "shell_command",
    "shell_script",
    "registry_write",
    "power_shutdown",
    "power_restart",
    "disk_mount",
    "disk_unmount",
    "browser_execute_js",
    # Mouse/keyboard control — blind UI interaction with no visual
    # confirmation available in a dry-run.
    "mouse_click",
    "mouse_double_click",
    "mouse_right_click",
    "mouse_drag",
    "keyboard_type",
    "keyboard_hotkey",
}

# State-changing browser actions that interact with visible page elements —
# a step below HIGH_RISK (no arbitrary code execution), but no longer LOW
# either now that they're not blanket-safe (see actions.py's retiering).
SANDBOX_MEDIUM_RISK_ACTIONS = {
    "browser_navigate",
    "browser_click",
    "browser_click_text",
    "browser_type",
    "browser_select",
    "browser_fill_form",
}

NETWORK_ACTIONS = {
    "api_request",
    "download_file",
    "browser_navigate",
    "wifi_connect",
    "wifi_disconnect",
}

ROOT_ACTIONS = {
    "service_start",
    "service_stop",
    "service_restart",
    "registry_write",
    "disk_mount",
    "disk_unmount",
    "package_install",
    "package_remove",
    "power_shutdown",
    "power_restart",
}

# Patterns in shell commands that indicate high risk
DANGEROUS_SHELL_PATTERNS = [
    "rm -rf",
    "rmdir /s",
    "del /f",
    "format",
    "mkfs",
    "dd if=",
    "chmod 777",
    "> /dev/",
    "shutdown",
    "reboot",
    "kill -9",
    "taskkill /f",
    "net stop",
    "reg delete",
    "diskpart",
]


class SimulationSandbox:
    """Pre-execution impact analysis and risk assessment."""

    def __init__(self, allowed_commands: list[str] | None = None):
        self.allowed_commands = allowed_commands or [
            "echo",
            "ls",
            "dir",
            "cat",
            "type",
            "ping",
            "whoami",
            "pwd",
            "grep",
            "find",
        ]

    async def simulate(self, plan: Any) -> SimulationReport:
        """Analyze a plan and produce an impact report without executing anything."""
        report = SimulationReport(plan_id=getattr(plan, "plan_id", "unknown"))
        max_risk = RiskLevel.LOW

        for action in plan.actions:
            action_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
            target = getattr(action, "target", "") or ""

            impact = ImpactItem(
                action_type=action_type,
                target=target,
            )

            # ── Feature 6: ReAct Pipeline Neural Cost Estimator ──
            # Skip TRIBE loading in dry-run mode (synchronous, blocking)
            # Cognitive cost is estimated mathematically instead
            stimulus = f"Execute action {action_type} on {target}"
            impact.cognitive_cost = min(
                1.0, len(stimulus) / 80.0 + (0.4 if action_type in SANDBOX_HIGH_RISK_ACTIONS else 0.1)
            )
            report.total_cognitive_cost += impact.cognitive_cost

            # Classify risk
            if action_type in SANDBOX_DESTRUCTIVE_ACTIONS:
                report.has_destructive = True
                impact.reversible = False

            if action_type in NETWORK_ACTIONS:
                report.has_network = True

            if action_type in ROOT_ACTIONS:
                report.requires_root = True

            # Determine risk level
            risk = self._assess_action_risk(action_type, target, action)
            impact.risk = risk
            impact.description = self._describe_impact(action_type, target)
            impact.estimated_scope = self._estimate_scope(action_type, target)

            # Pre-execution target assessment — checks the CURRENT live DOM
            # (if a browser session is already open) for whether this
            # click/type/select/fill_form target actually resolves, before
            # the action would run. No-op (predicted_issue stays "") when
            # there's no active session or the selector can't be statically
            # resolved — see dom_diff.assess_target().
            from pilot.system.dom_diff import assess_browser_action_target

            assessment = await assess_browser_action_target(action_type, action)
            if assessment is not None and assessment.matchable:
                impact.predicted_issue = assessment.reason
                if not assessment.found or not assessment.visible or assessment.ambiguous:
                    impact.predicted_issue_is_problem = True
                    impact.description += f" — {assessment.reason}"
                    risk_order_local = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
                    if risk_order_local.index(RiskLevel(impact.risk)) < risk_order_local.index(RiskLevel.HIGH):
                        impact.risk = RiskLevel.HIGH
                        risk = RiskLevel.HIGH

            # Track max risk
            risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            if risk_order.index(RiskLevel(risk)) > risk_order.index(RiskLevel(max_risk)):
                max_risk = RiskLevel(risk)

            report.impacts.append(impact)

        report.overall_risk = max_risk

        # Generate warnings
        report.warnings = self._generate_warnings(report)

        # Determine safety
        report.is_safe = max_risk in (RiskLevel.LOW, RiskLevel.MEDIUM) and not report.has_destructive

        # Generate recommendation
        report.recommendation = self._generate_recommendation(report)

        return report

    def _assess_action_risk(self, action_type: str, target: str, action: Any) -> str:
        """Assess the risk level of a single action."""
        if action_type in SANDBOX_HIGH_RISK_ACTIONS:
            # Check for especially dangerous shell patterns
            if action_type in ("shell_command", "shell_script"):
                params = getattr(action, "parameters", getattr(action, "params", None))
                command = ""
                if params:
                    command = getattr(params, "command", "") or getattr(params, "script", "") or ""

                # Verify against the sandbox allowlist
                base_cmd = command.strip().split()[0].lower() if command.strip() else ""
                if base_cmd and base_cmd not in self.allowed_commands:
                    return RiskLevel.CRITICAL

                if any(pattern in command.lower() for pattern in DANGEROUS_SHELL_PATTERNS):
                    return RiskLevel.CRITICAL
            return RiskLevel.HIGH

        if action_type in SANDBOX_DESTRUCTIVE_ACTIONS or action_type in SANDBOX_MEDIUM_RISK_ACTIONS:
            return RiskLevel.MEDIUM

        if action_type.startswith("file_") and "delete" in action_type:
            return RiskLevel.HIGH

        return RiskLevel.LOW

    def _describe_impact(self, action_type: str, target: str) -> str:
        """Human-readable impact description."""
        descriptions = {
            "file_delete": f"Delete file or directory: {target}",
            "file_write": f"Write/modify file: {target}",
            "file_create": f"Create new file: {target}",
            "file_move": f"Move/rename: {target}",
            "shell_command": "Execute shell command on system",
            "shell_script": "Run multi-line script",
            "code_execute": "Execute code in sandbox",
            "process_kill": f"Terminate process: {target}",
            "service_stop": f"Stop system service: {target}",
            "service_restart": f"Restart system service: {target}",
            "package_install": f"Install package: {target}",
            "package_remove": f"Remove package: {target}",
            "power_shutdown": "Shut down the system",
            "power_restart": "Restart the system",
            "registry_write": f"Modify Windows registry: {target}",
            "disk_mount": f"Mount disk/volume: {target}",
            "disk_unmount": f"Unmount disk/volume: {target}",
            "api_request": f"Make HTTP request to: {target}",
            "download_file": f"Download file from: {target}",
            "browser_navigate": f"Navigate the browser to: {target}",
            "browser_execute_js": f"Execute arbitrary JavaScript in the page: {target[:80]}",
            "browser_click": f"Click a page element: {target}",
            "browser_click_text": f"Click page text matching: {target}",
            "browser_type": f"Type into a page field: {target}",
            "browser_select": f"Select a dropdown option: {target}",
            "browser_fill_form": f"Fill and submit a form: {target}",
            "mouse_click": f"Click at screen position: {target}",
            "mouse_double_click": f"Double-click at screen position: {target}",
            "mouse_right_click": f"Right-click at screen position: {target}",
            "mouse_drag": f"Drag the mouse: {target}",
            "keyboard_type": "Type text via keyboard",
            "keyboard_hotkey": f"Send a keyboard hotkey: {target}",
        }
        return descriptions.get(action_type, f"Execute {action_type}: {target}")

    def _estimate_scope(self, action_type: str, target: str) -> str:
        """Estimate the scope of impact."""
        if action_type in ("file_delete", "file_write"):
            if "*" in target or "**" in target:
                return "multiple files (wildcard)"
            return "1 file"
        if action_type in ("shell_command", "shell_script"):
            return "system-wide"
        if action_type in ("service_stop", "service_restart"):
            return "1 service + dependents"
        if action_type in ("power_shutdown", "power_restart"):
            return "entire system"
        if action_type in ("package_install", "package_remove"):
            return "1 package + dependencies"
        if action_type in ("disk_mount", "disk_unmount"):
            return "1 volume"
        if action_type == "browser_execute_js":
            return "current page (script-defined, unbounded)"
        if action_type in SANDBOX_MEDIUM_RISK_ACTIONS:
            return "current page"
        if action_type in (
            "mouse_click",
            "mouse_double_click",
            "mouse_right_click",
            "mouse_drag",
            "keyboard_type",
            "keyboard_hotkey",
        ):
            return "current UI focus (blind interaction)"
        return "targeted"

    def _generate_warnings(self, report: SimulationReport) -> list[str]:
        """Generate human-readable warnings."""
        warnings = []
        if report.has_destructive:
            warnings.append("⚠️ Plan contains destructive actions that cannot be easily undone")
        if report.requires_root:
            warnings.append("🔐 Plan requires elevated privileges (root/admin)")
        if report.has_network:
            warnings.append("🌐 Plan makes external network requests")

        critical_count = sum(1 for i in report.impacts if i.risk == RiskLevel.CRITICAL)
        if critical_count > 0:
            warnings.append(f"🚨 {critical_count} action(s) rated CRITICAL risk")

        irreversible = [i for i in report.impacts if not i.reversible]
        if irreversible:
            warnings.append(f"♻️ {len(irreversible)} action(s) are NOT reversible")

        predicted_problems = [i for i in report.impacts if i.predicted_issue_is_problem]
        if predicted_problems:
            warnings.append(
                f"🎯 {len(predicted_problems)} browser action(s) predicted to fail or misfire "
                "against the current page — see each action's description"
            )

        if report.total_cognitive_cost > 2.0:
            warnings.append(
                "🧠 High cognitive load task sequence detected. Slowing down TTS & requesting verbal verification."
            )

        return warnings

    def _generate_recommendation(self, report: SimulationReport) -> str:
        """Generate a recommendation based on the report."""
        if report.overall_risk == RiskLevel.CRITICAL:
            return "❌ CRITICAL — Review each action carefully before proceeding"
        if report.overall_risk == RiskLevel.HIGH:
            return "⚠️ HIGH RISK — Confirm you understand the impact"
        if report.overall_risk == RiskLevel.MEDIUM:
            return "⚡ MODERATE — Proceed with awareness"
        return "✅ SAFE — Low risk, safe to execute"
