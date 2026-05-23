"""
base_agent.py
-------------
Abstract base class for all Heliox agents.
Every agent must implement `run()` and may optionally override `setup()`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standardised envelope returned by every Heliox agent."""

    agent_name: str
    status: str                         # "success" | "partial" | "error"
    payload: dict[str, Any]            # agent-specific structured output
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    error: str | None = None

    def finish(self, *, error: str | None = None) -> "AgentResult":
        self.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        if error:
            self.status = "error"
            self.error = error
        return self


class BaseAgent(ABC):
    """
    Abstract base for all Heliox agents.

    Subclasses must implement:
        run(**kwargs) -> AgentResult
    """

    #: Override in subclasses to give the agent a meaningful name.
    NAME: str = "BaseAgent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self.setup()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Optional initialisation hook called once during __init__."""

    @abstractmethod
    def run(self, **kwargs: Any) -> AgentResult:
        """
        Execute the agent's primary task.

        Returns
        -------
        AgentResult
            Structured result envelope.
        """