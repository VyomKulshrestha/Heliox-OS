"""Multi-agent router — spawns specialized sub-agents for complex tasks.

Analyzes the user's request and routes it to the most appropriate
specialized agent (or a combination of agents) for execution.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.multi_agent")


class AgentRole(Enum):
    """Specialized agent roles."""

    FILE_AGENT = "file_agent"
    CODE_AGENT = "code_agent"
    WEB_AGENT = "web_agent"
    MONITOR_AGENT = "monitor_agent"
    FORENSICS_AGENT = "forensics_agent"
    COMMUNICATION_AGENT = "comm_agent"
    SYSTEM_AGENT = "system_agent"
    GENERAL = "general"


# Keywords that hint at which specialist to invoke
ROLE_KEYWORDS: dict[AgentRole, list[str]] = {
    AgentRole.FILE_AGENT: [
        "file",
        "folder",
        "directory",
        "rename",
        "move",
        "copy",
        "delete",
        "create file",
        "read file",
        "write file",
        "list files",
        "search files",
        "permissions",
        "zip",
        "unzip",
        "archive",
    ],
    AgentRole.CODE_AGENT: [
        "code",
        "script",
        "python",
        "javascript",
        "rust",
        "compile",
        "run",
        "execute",
        "debug",
        "test",
        "function",
        "class",
        "import",
        "pip",
        "npm",
        "cargo",
        "git",
    ],
    AgentRole.WEB_AGENT: [
        "browse",
        "website",
        "url",
        "http",
        "scrape",
        "download",
        "api",
        "fetch",
        "wikipedia",
        "google",
        "search online",
        "web page",
    ],
    AgentRole.MONITOR_AGENT: [
        "monitor",
        "watch",
        "alert",
        "cpu",
        "memory",
        "ram",
        "disk",
        "network",
        "process",
        "background",
        "continuously",
        "keep checking",
    ],
    AgentRole.FORENSICS_AGENT: [
        "forensic",
        "forensics",
        "log",
        "logs",
        "anomaly",
        "anomalies",
        "suspicious",
        "failed login",
        "failed attempts",
        "auth logs",
        "nginx logs",
        "restart loop",
    ],
    AgentRole.COMMUNICATION_AGENT: [
        "email",
        "send",
        "message",
        "slack",
        "discord",
        "whatsapp",
        "notification",
        "notify",
        "webhook",
    ],
    AgentRole.SYSTEM_AGENT: [
        "install",
        "uninstall",
        "package",
        "service",
        "registry",
        "volume",
        "brightness",
        "wifi",
        "bluetooth",
        "shutdown",
        "restart",
        "sleep",
        "lock",
        "settings",
        "environment",
    ],
}

# Specialized system prompts for each agent role
ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.FILE_AGENT: (
        "You are the File Operations Specialist for Heliox OS. "
        "You excel at filesystem tasks: creating, reading, writing, moving, "
        "copying, deleting files and directories. You understand permissions, "
        "file formats, and directory structures deeply. Always verify paths exist "
        "before operations and suggest safe approaches."
    ),
    AgentRole.CODE_AGENT: (
        "You are the Code Execution Specialist for Heliox OS. "
        "You are an expert programmer who writes clean, functional code. "
        "You can generate, execute, debug, and test code in Python, JavaScript, "
        "Bash, and PowerShell. Always include error handling and logging. "
        "If code fails, analyze the error and fix it automatically."
    ),
    AgentRole.WEB_AGENT: (
        "You are the Web Operations Specialist for Heliox OS. "
        "You handle all web-related tasks: browsing, scraping, API calls, "
        "downloading files, and extracting information from web pages. "
        "Use browser automation for complex pages and direct HTTP for simple fetches."
    ),
    AgentRole.MONITOR_AGENT: (
        "You are the System Monitor Specialist for Heliox OS. "
        "You set up and manage background monitoring tasks for CPU, memory, "
        "disk, network, and custom metrics. You create threshold-based alerts "
        "and can run persistent watch loops."
    ),
    AgentRole.COMMUNICATION_AGENT: (
        "You are the Communication Specialist for Heliox OS. "
        "You handle sending emails, messages via Slack/Discord/WhatsApp, "
        "webhook notifications, and any inter-system communication. "
        "Always confirm sensitive communications before sending."
    ),
    AgentRole.SYSTEM_AGENT: (
        "You are the System Administration Specialist for Heliox OS. "
        "You handle package management, service control, registry editing, "
        "volume/brightness control, network settings, and power management. "
        "Always check current state before making changes."
    ),
    AgentRole.FORENSICS_AGENT: (
        "You are the Forensics and Log Analysis Specialist for Heliox OS. "
        "You excel at system/service log inspection, parsing standard log formats "
        "(syslog, auth, nginx, service logs), analyzing event timelines, and "
        "identifying suspicious patterns or operational anomalies (failed logins, "
        "crashes, restart loops) to generate structural reports."
    ),
}


class MultiAgentRouter:
    """Routes tasks to specialized sub-agents for optimal execution."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router

    def classify(self, user_input: str) -> list[AgentRole]:
        """Determine which specialist agent(s) should handle this request."""
        input_lower = user_input.lower()
        scores: dict[AgentRole, int] = {}

        for role, keywords in ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in input_lower)
            if score > 0:
                scores[role] = score

        if not scores:
            return [AgentRole.GENERAL]

        # Sort by score descending, return top matches
        sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # If top score is significantly higher, use only that specialist
        if len(sorted_roles) >= 2 and sorted_roles[0][1] >= 2 * sorted_roles[1][1]:
            return [sorted_roles[0][0]]

        # Otherwise return top 2 specialists
        return [role for role, _ in sorted_roles[:2]]

    def get_enhanced_prompt(self, roles: list[AgentRole]) -> str:
        """Get a combined system prompt for the assigned specialist roles."""
        if AgentRole.GENERAL in roles:
            return ""

        prompts = [ROLE_PROMPTS[role] for role in roles if role in ROLE_PROMPTS]
        if not prompts:
            return ""

        return "\n\n".join([f"=== SPECIALIST MODE: {', '.join(r.value for r in roles)} ==="] + prompts)

    def get_routing_summary(self, user_input: str) -> dict[str, Any]:
        """Get a full routing analysis for a user input."""
        roles = self.classify(user_input)
        return {
            "input": user_input,
            "assigned_agents": [r.value for r in roles],
            "enhanced_prompt": self.get_enhanced_prompt(roles),
            "is_multi_agent": len(roles) > 1,
        }
