"""Code Agent — handles code generation, execution, and debugging.

Specializes in understanding code-related tasks: writing scripts,
running code snippets, debugging errors, testing, and managing
development tools (pip, npm, cargo, git).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.code_agent")

CODE_ACTION_TYPES: set[ActionType] = {
    ActionType.CODE_EXECUTE,
    ActionType.CODE_GENERATE_AND_RUN,
    ActionType.SKILL_RUN,
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.PTY_EXEC,
}


class CodeAgent(BaseAgent):
    """Specialist agent for code generation, execution, and debugging."""

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.CODE, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.CODE_EXECUTE,
                description="Execute code in Python, PowerShell, Bash, or JavaScript",
            ),
            AgentCapability(
                action_type=ActionType.CODE_GENERATE_AND_RUN,
                description="Generate code from a task description and execute it",
            ),
            AgentCapability(
                action_type=ActionType.SKILL_RUN,
                description="Run a Python skill from the config skills directory (skill_id + arguments)",
            ),
            AgentCapability(
                action_type=ActionType.SHELL_COMMAND,
                description="Run shell commands for dev tooling (git, pip, npm, cargo)",
            ),
            AgentCapability(
                action_type=ActionType.SHELL_SCRIPT,
                description="Run multi-line dev scripts",
            ),
            AgentCapability(
                action_type=ActionType.PTY_EXEC,
                description="Run a command in a persistent PTY shell session, preserving env and cwd",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the CODE AGENT for Heliox OS. "
            "You are an expert software engineer who can write, execute, debug, "
            "and test code in Python, JavaScript, Bash, PowerShell, and Rust. "
            "You manage development tools (pip, npm, cargo, git). "
            "When generating code, always include proper error handling, "
            "logging, and type hints. If code fails, analyze the traceback "
            "and produce a corrected version automatically. "
            "For complex tasks, break them into smaller executable units."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in CODE_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute code-related actions."""
        import time

        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        sub_plan = ActionPlan(
            actions=my_actions,
            explanation=f"Code Agent executing {len(my_actions)} action(s)",
            raw_input=user_input,
        )

        results = await self._executor.execute(sub_plan)

        # Auto-debug: if any code execution failed, try to fix and re-run
        failed = [r for r in results if not r.success and r.action.action_type == ActionType.CODE_EXECUTE]
        if failed and self._model:
            logger.info("Code Agent attempting auto-debug for %d failed action(s)", len(failed))
            # Pass the error back for reflection (orchestrator handles retry)

        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results
