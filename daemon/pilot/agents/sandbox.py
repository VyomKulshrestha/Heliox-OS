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

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "description": self.description,
            "risk": self.risk,
            "reversible": self.reversible,
            "estimated_scope": self.estimated_scope,
            "cognitive_cost": self.cognitive_cost,
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

DESTRUCTIVE_ACTIONS = {
    "file_delete",
    "shell_command",
    "shell_script",
    "process_kill",
    "service_stop",
    "service_restart",
    "power_action",
    "registry_write",
    "registry_delete",
    "disk_manage",
    "package_uninstall",
}

HIGH_RISK_ACTIONS = {
    "code_execute",
    "shell_command",
    "shell_script",
    "registry_write",
    "registry_delete",
    "power_action",
    "disk_manage",
}

NETWORK_ACTIONS = {
    "api_request",
    "download_file",
    "browser_navigate",
    "wifi_control",
}

ROOT_ACTIONS = {
    "service_start",
    "service_stop",
    "service_restart",
    "registry_write",
    "registry_delete",
    "disk_manage",
    "package_install",
    "package_uninstall",
    "power_action",
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
            "echo", "ls", "dir", "cat", "type", "ping", "whoami", "pwd", "grep", "find"
        ]

    def simulate(self, plan: Any) -> SimulationReport:
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
            impact.cognitive_cost = min(1.0, len(stimulus) / 80.0 + (0.4 if action_type in HIGH_RISK_ACTIONS else 0.1))
            report.total_cognitive_cost += impact.cognitive_cost

            # Classify risk
            if action_type in DESTRUCTIVE_ACTIONS:
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
        if action_type in HIGH_RISK_ACTIONS:
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

        if action_type in DESTRUCTIVE_ACTIONS:
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
            "shell_command": f"Execute shell command on system",
            "shell_script": f"Run multi-line script",
            "code_execute": f"Execute code in sandbox",
            "process_kill": f"Terminate process: {target}",
            "service_stop": f"Stop system service: {target}",
            "service_restart": f"Restart system service: {target}",
            "package_install": f"Install package: {target}",
            "package_uninstall": f"Remove package: {target}",
            "power_action": f"System power action: {target}",
            "registry_write": f"Modify Windows registry: {target}",
            "registry_delete": f"Delete registry key: {target}",
            "disk_manage": f"Disk management operation: {target}",
            "api_request": f"Make HTTP request to: {target}",
            "download_file": f"Download file from: {target}",
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
        if action_type in ("power_action",):
            return "entire system"
        if action_type in ("package_install", "package_uninstall"):
            return "1 package + dependencies"
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
