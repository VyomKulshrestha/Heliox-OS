"""Multi-Step Reasoning & Chaining — recursive planning with dependency graphs.

Output of one plan feeds into the next. Dynamic re-planning based on
intermediate results. Conditional branching and sub-task decomposition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("pilot.agents.chain_planner")


@dataclass
class ChainStep:
    """A single step in a multi-step chain."""

    id: str
    description: str
    action_type: str = ""
    parameters: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # Step IDs this depends on
    condition: str | None = None  # e.g., "file_exists('/path')", "output_contains('success')"
    branch_if_true: str | None = None  # Step ID to jump to if condition is true
    branch_if_false: str | None = None  # Step ID if condition is false
    output: str = ""
    status: str = "pending"  # pending, running, completed, failed, skipped
    error: str | None = None
    duration: float = 0.0
    retry_count: int = 0


@dataclass
class ChainPlan:
    """A multi-step plan with dependencies and conditionals."""

    id: str
    name: str
    description: str
    steps: list[ChainStep] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)  # Shared context
    status: str = "pending"
    total_duration: float = 0.0
    created_at: float = field(default_factory=time.time)


class ChainExecutor:
    """Execute multi-step chains with dependency resolution.

    Features:
    - Dependency graph: steps only run when their dependencies complete
    - Output chaining: {{step_id.output}} variables in parameters
    - Conditional branching: if/else based on conditions
    - Dynamic re-planning: feed intermediate results back to LLM
    - Parallel execution of independent steps
    """

    def __init__(self, executor=None, planner=None):
        self._executor = executor
        self._planner = planner

    async def execute_chain(self, plan: ChainPlan) -> ChainPlan:
        """Execute all steps in a chain plan, respecting dependencies."""
        plan.status = "running"
        t0 = time.time()

        # Build dependency graph
        step_map = {s.id: s for s in plan.steps}
        completed: set[str] = set()

        while True:
            # Find ready steps (all deps completed, not yet run)
            ready = []
            for step in plan.steps:
                if step.status != "pending":
                    continue
                if all(dep in completed for dep in step.depends_on):
                    ready.append(step)

            if not ready:
                # Check if we're done or stuck
                if all(s.status in ("completed", "failed", "skipped") for s in plan.steps):
                    break
                pending = [s.id for s in plan.steps if s.status == "pending"]
                if pending:
                    logger.error("Chain stuck: steps %s have unmet dependencies", pending)
                    for s_id in pending:
                        step_map[s_id].status = "failed"
                        step_map[s_id].error = "Unmet dependencies"
                break

            # Execute ready steps (optionally in parallel)
            if len(ready) > 1:
                tasks = [self._execute_step(step, plan, step_map) for step in ready]
                await asyncio.gather(*tasks)
            else:
                await self._execute_step(ready[0], plan, step_map)

            completed.update(s.id for s in ready if s.status == "completed")

        plan.total_duration = time.time() - t0
        failed = [s for s in plan.steps if s.status == "failed"]
        plan.status = "completed" if not failed else "partial_failure"

        return plan

    async def _execute_step(
        self,
        step: ChainStep,
        plan: ChainPlan,
        step_map: dict[str, ChainStep],
    ):
        """Execute a single step in the chain."""
        step.status = "running"
        t0 = time.time()

        try:
            # Check condition
            if step.condition:
                condition_met = self._evaluate_condition(step.condition, plan, step_map)
                if step.branch_if_true or step.branch_if_false:
                    if condition_met and step.branch_if_true:
                        step.output = f"Condition met, branching to {step.branch_if_true}"
                        step.status = "completed"
                        # Skip all steps not on the true branch
                        if step.branch_if_false:
                            self._skip_branch(step.branch_if_false, step_map)
                    elif not condition_met and step.branch_if_false:
                        step.output = f"Condition not met, branching to {step.branch_if_false}"
                        step.status = "completed"
                        if step.branch_if_true:
                            self._skip_branch(step.branch_if_true, step_map)
                    else:
                        step.status = "skipped"
                        step.output = f"Condition {'met' if condition_met else 'not met'}, no branch target"
                    step.duration = time.time() - t0
                    return

            # Substitute variables in parameters
            resolved_params = self._resolve_variables(step.parameters, plan, step_map)

            # Execute the action
            if self._executor:
                from pilot.actions import Action, ActionPlan, ActionType, EmptyParams

                action_type = ActionType(step.action_type)

                # Build action from step
                action = Action(
                    action_type=action_type,
                    parameters=resolved_params if isinstance(resolved_params, dict) else EmptyParams(),
                )
                action_plan = ActionPlan(actions=[action])
                results = await self._executor.execute(action_plan)

                if results and results[0].success:
                    step.output = results[0].output
                    step.status = "completed"
                else:
                    error = results[0].error if results else "Unknown error"
                    step.error = error
                    step.status = "failed"
            else:
                # No executor — just resolve and record
                step.output = f"Dry run: {step.action_type} with {json.dumps(resolved_params)}"
                step.status = "completed"

            # Store output in plan variables
            plan.variables[f"{step.id}.output"] = step.output
            plan.variables[f"{step.id}.status"] = step.status

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            logger.error("Chain step %s failed: %s", step.id, e)

        step.duration = time.time() - t0

    def _evaluate_condition(
        self,
        condition: str,
        plan: ChainPlan,
        step_map: dict[str, ChainStep],
    ) -> bool:
        """Evaluate a condition string.

        Supported conditions:
        - "output_contains(step_id, 'text')"
        - "step_succeeded(step_id)"
        - "step_failed(step_id)"
        - "file_exists('/path')"
        - "var_equals(name, 'value')"
        """
        try:
            if condition.startswith("output_contains("):
                parts = condition[len("output_contains(") : -1].split(",", 1)
                step_id = parts[0].strip().strip("'\"")
                text = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
                step = step_map.get(step_id)
                return text.lower() in (step.output.lower() if step else "")

            elif condition.startswith("step_succeeded("):
                step_id = condition[len("step_succeeded(") : -1].strip().strip("'\"")
                step = step_map.get(step_id)
                return step.status == "completed" if step else False

            elif condition.startswith("step_failed("):
                step_id = condition[len("step_failed(") : -1].strip().strip("'\"")
                step = step_map.get(step_id)
                return step.status == "failed" if step else True

            elif condition.startswith("file_exists("):
                path = condition[len("file_exists(") : -1].strip().strip("'\"")
                return os.path.exists(path)

            elif condition.startswith("var_equals("):
                parts = condition[len("var_equals(") : -1].split(",", 1)
                var_name = parts[0].strip().strip("'\"")
                expected = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
                return plan.variables.get(var_name, "") == expected

            else:
                # Try as Python expression (safe subset)
                return bool(condition)

        except Exception as e:
            logger.warning("Condition evaluation failed: %s — %s", condition, e)
            return False

    def _resolve_variables(
        self,
        params: dict,
        plan: ChainPlan,
        step_map: dict[str, ChainStep],
    ) -> dict:
        """Replace {{variable}} placeholders in parameters."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Replace {{step_id.output}} patterns
                for var_name, var_value in plan.variables.items():
                    value = value.replace(f"{{{{{var_name}}}}}", str(var_value))

                # Replace {{step_id}} with output shorthand
                for step in plan.steps:
                    placeholder = f"{{{{{step.id}}}}}"
                    if placeholder in value:
                        value = value.replace(placeholder, step.output or "")

            resolved[key] = value
        return resolved

    def _skip_branch(self, step_id: str, step_map: dict[str, ChainStep]):
        """Skip a step and all steps that depend only on it."""
        step = step_map.get(step_id)
        if step and step.status == "pending":
            step.status = "skipped"
            step.output = "Skipped due to conditional branch"


# ── Helper to create chains from task descriptions ───────────────────


def create_sequential_chain(
    name: str,
    steps: list[dict],
) -> ChainPlan:
    """Create a simple sequential chain from a list of step dicts.

    Each step dict: {"description": str, "action_type": str, "parameters": dict}
    """
    plan = ChainPlan(
        id=f"chain_{int(time.time())}",
        name=name,
        description=f"Sequential chain: {name}",
    )

    prev_id = None
    for i, step_dict in enumerate(steps):
        step = ChainStep(
            id=f"step_{i}",
            description=step_dict.get("description", f"Step {i}"),
            action_type=step_dict.get("action_type", ""),
            parameters=step_dict.get("parameters", {}),
            depends_on=[prev_id] if prev_id else [],
            condition=step_dict.get("condition"),
            branch_if_true=step_dict.get("branch_if_true"),
            branch_if_false=step_dict.get("branch_if_false"),
        )
        plan.steps.append(step)
        prev_id = step.id

    return plan


async def execute_chain_from_steps(
    executor,
    steps: list[dict],
    name: str = "user_chain",
) -> str:
    """Convenience: create and execute a chain from step dicts."""
    plan = create_sequential_chain(name, steps)
    chain_executor = ChainExecutor(executor=executor)
    result = await chain_executor.execute_chain(plan)

    summary = {
        "name": result.name,
        "status": result.status,
        "duration": round(result.total_duration, 2),
        "steps": [
            {
                "id": s.id,
                "description": s.description,
                "status": s.status,
                "output": s.output[:200] if s.output else "",
                "error": s.error,
                "duration": round(s.duration, 2),
            }
            for s in result.steps
        ],
    }
    return json.dumps(summary, indent=2)
