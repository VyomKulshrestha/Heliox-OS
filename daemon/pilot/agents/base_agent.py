"""Base agent protocol — shared interface for all specialist agents.

Every specialist agent inherits from BaseAgent and exposes:
  - A role identifier
  - A list of tools/actions it can perform
  - A system prompt fragment specific to its domain
  - An execute method that runs actions within its specialty
  - An inter-agent messaging interface

This is the backbone of the modular multi-agent architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.actions import ActionPlan, ActionResult, ActionType

if TYPE_CHECKING:
    from pilot.agents.orchestrator import AgentOrchestrator
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.base")


# ── Agent lifecycle states ──


class AgentStatus(StrEnum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"  # waiting for another agent
    ERROR = "error"
    STOPPED = "stopped"


class AgentRole(StrEnum):
    """Canonical agent roles in the system."""

    SYSTEM = "system_agent"
    SSH = "ssh_agent"
    CODE = "code_agent"
    WEB = "web_agent"
    MONITOR = "monitor_agent"
    COMMUNICATION = "comm_agent"
    RSS = "rss_agent"
    CALENDAR = "calendar_agent"
    FORENSICS = "forensics_agent"
    ORCHESTRATOR = "orchestrator"
    SEMANTIC_SEARCH = "semantic_search_agent"
    GENERAL = "general"


# ── Inter-agent messaging ──


class MessagePriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentMessage:
    """Structured message for inter-agent communication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    recipient: str = ""  # agent role or "*" for broadcast
    msg_type: str = "request"  # request, response, event, handoff
    action: str = ""  # what to do
    payload: dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: str | None = None  # links response to original request
    timestamp: float = field(default_factory=time.time)

    def reply(self, payload: dict[str, Any], msg_type: str = "response") -> AgentMessage:
        """Create a reply message linked to this message."""
        return AgentMessage(
            sender=self.recipient,
            recipient=self.sender,
            msg_type=msg_type,
            action=self.action,
            payload=payload,
            correlation_id=self.id,
        )


@dataclass
class AgentCapability:
    """Describes one tool/action an agent can perform."""

    action_type: ActionType
    description: str
    requires_confirmation: bool = False
    estimated_duration_ms: int = 1000


# ── Base Agent ──


class BaseAgent(ABC):
    """Abstract base class for all specialist agents.

    Each agent:
      1. Declares its role and capabilities
      2. Can receive messages from the orchestrator or other agents
      3. Can request help from other agents via the orchestrator
      4. Executes actions within its domain
    """

    def __init__(
        self,
        role: AgentRole,
        model_router: ModelRouter | None = None,
    ) -> None:
        self.role = role
        self.agent_id = f"{role.value}_{str(uuid.uuid4())[:6]}"
        self.status = AgentStatus.IDLE
        self._model = model_router
        self._orchestrator: AgentOrchestrator | None = None
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._message_handlers: dict[str, Callable] = {}
        self._running = False
        self._task_count = 0
        self._error_count = 0
        self._total_duration_ms = 0

    # ── Abstract interface ──

    @abstractmethod
    def get_capabilities(self) -> list[AgentCapability]:
        """Return a list of actions this agent can handle."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialist system prompt for this agent."""
        ...

    @abstractmethod
    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute actions that fall within this agent's domain."""
        ...

    @abstractmethod
    def can_handle(self, action_type: ActionType) -> bool:
        """Return True if this agent handles the given action type."""
        ...

    def get_permission_tier(self) -> int:
        """Return the minimum permission tier required (default: 1)."""
        return 1

    def get_resource_needs(self) -> set[str]:
        """Return set of resource requirements (e.g., 'browser', 'screen', 'audio')."""
        return set()

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the agent's message processing loop."""
        self._running = True
        self.status = AgentStatus.IDLE
        logger.info("Agent %s started", self.agent_id)

    async def stop(self) -> None:
        """Gracefully shut down the agent."""
        self._running = False
        self.status = AgentStatus.STOPPED
        logger.info("Agent %s stopped", self.agent_id)

    def attach_orchestrator(self, orchestrator: AgentOrchestrator) -> None:
        """Register this agent with the central orchestrator."""
        self._orchestrator = orchestrator

    # ── Messaging ──

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent via the orchestrator."""
        if self._orchestrator is None:
            logger.warning("Agent %s has no orchestrator — message dropped", self.agent_id)
            return
        await self._orchestrator.route_message(message)

    async def receive_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message. Override for custom handling."""
        handler = self._message_handlers.get(message.msg_type)
        if handler:
            return await handler(message)

        if message.msg_type == "request":
            # Default: try to handle as a task
            return message.reply({"status": "received", "agent": self.agent_id})
        return None

    async def request_help(self, target_role: str, action: str, payload: dict) -> AgentMessage | None:
        """Ask another agent for help and wait for a response."""
        msg = AgentMessage(
            sender=self.role.value,
            recipient=target_role,
            msg_type="request",
            action=action,
            payload=payload,
            priority=MessagePriority.NORMAL,
        )
        if self._orchestrator:
            return await self._orchestrator.route_message(msg)
        return None

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        """Return agent performance statistics."""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.status.value,
            "task_count": self._task_count,
            "error_count": self._error_count,
            "total_duration_ms": self._total_duration_ms,
            "capabilities": len(self.get_capabilities()),
        }

    def _record_task(self, duration_ms: int, success: bool) -> None:
        self._task_count += 1
        self._total_duration_ms += duration_ms
        if not success:
            self._error_count += 1
