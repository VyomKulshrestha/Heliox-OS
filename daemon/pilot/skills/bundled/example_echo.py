"""Example skill — uppercase a string parameter.

Copy this file to the configured skills directory and modify to experiment.
"""

from __future__ import annotations

from typing import Any

from pilot.skills.base import Skill, SkillContext


class EchoUpperSkill(Skill):
    """Reference implementation for the skill loader."""

    @property
    def skill_id(self) -> str:
        return "heliox.example.echo_upper"

    @property
    def name(self) -> str:
        return "Echo (uppercase)"

    @property
    def description(self) -> str:
        return "Returns the uppercase of parameters['text'] (string)."

    async def run(self, params: dict[str, Any], ctx: SkillContext) -> str:
        _ = ctx
        text = params.get("text", "")
        return str(text).upper()
