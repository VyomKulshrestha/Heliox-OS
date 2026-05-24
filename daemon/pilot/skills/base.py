"""Abstract base for third-party agent skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillContext:
    """Runtime context passed to every skill invocation.

    Fields may grow over time; skills should accept **kwargs tolerance by
    only using documented attributes.
    """

    pilot_config: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Third-party capability: implement ``skill_id``, ``name``, ``description``, and ``run``.

    Drop a ``.py`` file under a configured skills directory; the loader imports
    the module and registers every concrete ``Skill`` subclass.

    ``skill_id`` must be unique, stable, and safe for JSON (e.g. ``vendor.feature``).
    """

    @property
    @abstractmethod
    def skill_id(self) -> str:
        """Unique id used in ``skill_run`` plans and APIs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label for UIs and logs."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Planner-facing summary of behavior and expected parameters."""

    @abstractmethod
    async def run(self, params: dict[str, Any], ctx: SkillContext) -> str:
        """Execute the skill; return a string result (or error message)."""
