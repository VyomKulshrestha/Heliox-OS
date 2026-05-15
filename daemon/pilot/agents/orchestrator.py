"""Agent Orchestrator — central coordinator for the multi-agent system.

The Orchestrator:
  1. Maintains a registry of all specialist agents
  2. Routes user tasks to the correct agent(s) based on action types
  3. Handles inter-agent messaging
  4. Supports dynamic agent spawning
  5. Integrates with the existing ReAct loop (Planner → Orchestrator → Verifier)

This replaces the simple MultiAgentRouter with a full agent coordination system.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    BaseAgent,
)

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.orchestrator")


class AgentOrchestrator:
    """Central coordinator that manages specialist agents and routes tasks.

    Architecture:
      User Input → Planner → Orchestrator → [Agent₁, Agent₂, ...] → Verifier

    The Orchestrator analyzes the plan's action types and dispatches each
    action to the specialist agent that owns that action type. For multi-agent
    tasks, it coordinates parallel or sequential execution and merges results.
    """

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router
        self._agents: dict[AgentRole, BaseAgent] = {}
        self._action_registry: dict[ActionType, AgentRole] = {}
        self._message_log: list[AgentMessage] = []
        self._broadcast_fn: Callable[..., Coroutine] | None = None

    # ── Agent Registration ──

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent and index its capabilities."""
        agent.attach_orchestrator(self)
        self._agents[agent.role] = agent

        # Index action types → agent role
        for cap in agent.get_capabilities():
            self._action_registry[cap.action_type] = agent.role

        logger.info(
            "Registered agent %s with %d capabilities",
            agent.role.value,
            len(agent.get_capabilities()),
        )

    def unregister_agent(self, role: AgentRole) -> None:
        """Remove an agent from the registry."""
        agent = self._agents.pop(role, None)
        if agent:
            # Clean up action registry
            self._action_registry = {at: r for at, r in self._action_registry.items() if r != role}
            logger.info("Unregistered agent %s", role.value)

    def get_agent(self, role: AgentRole) -> BaseAgent | None:
        """Get a registered agent by role."""
        return self._agents.get(role)

    def set_broadcast(self, fn: Callable[..., Coroutine]) -> None:
        """Set the WebSocket broadcast function for UI notifications."""
        self._broadcast_fn = fn

    # ── Task Routing ──

    def analyze_plan(self, plan: ActionPlan) -> dict[AgentRole, list[int]]:
        """Analyze a plan and determine which agents handle which actions.

        Returns a mapping of AgentRole → list of action indices.
        """
        routing: dict[AgentRole, list[int]] = {}
        for i, action in enumerate(plan.actions):
            role = self._action_registry.get(action.action_type, AgentRole.SYSTEM)
            routing.setdefault(role, []).append(i)
        return routing

    def get_routing_summary(self, plan: ActionPlan) -> dict[str, Any]:
        """Get a human-readable routing summary for the UI."""
        routing = self.analyze_plan(plan)
        return {
            "assigned_agents": [
                {
                    "role": role.value,
                    "action_count": len(indices),
                    "action_types": [plan.actions[i].action_type.value for i in indices],
                    "status": self._agents[role].status.value if role in self._agents else "unregistered",
                }
                for role, indices in routing.items()
            ],
            "is_multi_agent": len(routing) > 1,
            "total_agents": len(routing),
        }

    async def execute_plan(
    self,
    user_input: str,
    plan: ActionPlan,
    on_action_start: Callable | None = None,
    on_action_complete: Callable | None = None,
    cancel_event: asyncio.Event | None = None,   # ← add this
    ) -> list[ActionResult]:
        """Execute a plan by routing actions to specialist agents.

        For single-agent tasks, delegates directly.
        For multi-agent tasks, runs agents sequentially (preserving action order)
        while allowing each agent to process its batch in parallel internally.
        """
        routing = self.analyze_plan(plan)
        all_results: list[ActionResult | None] = [None] * len(plan.actions)

        # Broadcast routing info to UI
        if self._broadcast_fn:
            await self._broadcast_fn(
                "agent_routing",
                {
                    "assigned_agents": [r.value for r in routing],
                    "is_multi_agent": len(routing) > 1,
                },
            )
        # Process actions in order, grouping consecutive same-agent actions
        action_order = self._build_execution_order(plan, routing)

        for batch in action_order:
             # ── Cancellation check ──
            if cancel_event and cancel_event.is_set():
               logger.info("Orchestrator: cancel_event set — halting plan execution")
               break
            
            role, indices = batch
            agent = self._agents.get(role)
            if agent is None:
                # Fallback to system agent
                agent = self._agents.get(AgentRole.SYSTEM)
                if agent is None:
                    logger.error("No agent available for role %s", role.value)
                    continue

            # Build sub-plan for this agent's batch
            batch_actions = [plan.actions[i] for i in indices]
            sub_plan = ActionPlan(
                actions=batch_actions,
                explanation=f"{role.value} handling {len(batch_actions)} action(s)",
                raw_input=user_input,
            )

            # Notify action starts
            if on_action_start:
                for action in batch_actions:
                    await on_action_start(action)

            # Execute via the specialist agent
            results = await agent.handle_task(user_input, sub_plan)

            # Map results back to original indices
            for idx, result in zip(indices, results):
                all_results[idx] = result
                if on_action_complete:
                    await on_action_complete(result)

        # Fill any gaps with error results
        final: list[ActionResult] = []
        for i, r in enumerate(all_results):
            if r is None:
                final.append(
                    ActionResult(
                        action=plan.actions[i],
                        success=False,
                        error="No agent could handle this action",
                    )
                )
            else:
                final.append(r)

        return final

    def _build_execution_order(
        self,
        plan: ActionPlan,
        routing: dict[AgentRole, list[int]],
    ) -> list[tuple[AgentRole, list[int]]]:
        """Build execution batches preserving action order.

        Groups consecutive actions for the same agent, but maintains
        the overall sequential order of the plan.
        """
        if not plan.actions:
            return []

        batches: list[tuple[AgentRole, list[int]]] = []
        current_role: AgentRole | None = None
        current_indices: list[int] = []

        for i in range(len(plan.actions)):
            role = self._action_registry.get(plan.actions[i].action_type, AgentRole.SYSTEM)
            if role != current_role:
                if current_indices:
                    batches.append((current_role, current_indices))  # type: ignore
                current_role = role
                current_indices = [i]
            else:
                current_indices.append(i)

        if current_indices and current_role is not None:
            batches.append((current_role, current_indices))

        return batches

    # ── Inter-Agent Messaging ──

    async def route_message(self, message: AgentMessage) -> AgentMessage | None:
        """Route a message between agents."""
        self._message_log.append(message)

        if message.recipient == "*":
            # Broadcast to all agents
            for agent in self._agents.values():
                await agent.receive_message(message)
            return None

        # Find target agent by role
        target_role = None
        for role in AgentRole:
            if role.value == message.recipient:
                target_role = role
                break

        if target_role and target_role in self._agents:
            return await self._agents[target_role].receive_message(message)

        logger.warning("Message to unknown agent: %s", message.recipient)
        return None

    # ── Dynamic Agent Spawning ──

    async def spawn_agent(self, role: AgentRole, **kwargs: Any) -> BaseAgent | None:
        """Dynamically spawn a new specialist agent at runtime.

        This allows the system to create agents on-demand for tasks
        that need temporary specialized handling.
        """
        if role in self._agents:
            logger.info("Agent %s already exists, returning existing", role.value)
            return self._agents[role]

        # Import and instantiate the agent
        agent: BaseAgent | None = None
        try:
            if role == AgentRole.SYSTEM:
                from pilot.agents.system_agent import SystemAgent

                agent = SystemAgent(self._model, kwargs.get("executor"))
            elif role == AgentRole.CODE:
                from pilot.agents.code_agent import CodeAgent

                agent = CodeAgent(self._model, kwargs.get("executor"))
            elif role == AgentRole.WEB:
                from pilot.agents.web_agent import WebAgent

                agent = WebAgent(self._model, kwargs.get("executor"))
            elif role == AgentRole.MONITOR:
                from pilot.agents.monitor_agent import MonitorAgent

                agent = MonitorAgent(self._model, kwargs.get("background_manager"))
            elif role == AgentRole.COMMUNICATION:
                from pilot.agents.comm_agent import CommunicationAgent

                agent = CommunicationAgent(self._model, kwargs.get("executor"))

            if agent:
                self.register_agent(agent)
                await agent.start()
                logger.info("Dynamically spawned agent: %s", role.value)

        except Exception as e:
            logger.error("Failed to spawn agent %s: %s", role.value, e)

        return agent

    # ── Lifecycle ──

    async def start_all(self) -> None:
        """Start all registered agents."""
        for agent in self._agents.values():
            await agent.start()

    async def stop_all(self) -> None:
        """Stop all registered agents."""
        for agent in self._agents.values():
            await agent.stop()

    # ── Stats & Diagnostics ──

    def get_all_stats(self) -> dict[str, Any]:
        """Return statistics for all registered agents."""
        return {
            "agents": {role.value: agent.get_stats() for role, agent in self._agents.items()},
            "total_agents": len(self._agents),
            "registered_actions": len(self._action_registry),
            "message_count": len(self._message_log),
        }

    def get_all_capabilities(self) -> dict[str, list[str]]:
        """Return all capabilities grouped by agent."""
        return {
            role.value: [c.action_type.value for c in agent.get_capabilities()] for role, agent in self._agents.items()
        }

    def get_input_routing_summary(self, user_input: str) -> dict[str, Any]:
        """Legacy compatibility — route by user input keywords (like old MultiAgentRouter)."""
        input_lower = user_input.lower()
        scores: dict[AgentRole, int] = {}

        keywords: dict[AgentRole, list[str]] = {
            AgentRole.SYSTEM: [
                "file",
                "folder",
                "install",
                "service",
                "process",
                "shutdown",
                "restart",
                "volume",
                "brightness",
                "wifi",
                "screenshot",
                "registry",
            ],
            AgentRole.CODE: [
                "code",
                "script",
                "python",
                "debug",
                "test",
                "compile",
                "run",
                "git",
                "pip",
                "npm",
            ],
            AgentRole.WEB: [
                "browse",
                "website",
                "url",
                "scrape",
                "download",
                "api",
                "http",
                "google",
            ],
            AgentRole.MONITOR: [
                "monitor",
                "watch",
                "alert",
                "cpu",
                "memory",
                "disk",
                "background",
            ],
            AgentRole.COMMUNICATION: [
                "email",
                "slack",
                "discord",
                "message",
                "notify",
                "webhook",
            ],
        }

        for role, kws in keywords.items():
            score = sum(1 for kw in kws if kw in input_lower)
            if score > 0:
                scores[role] = score

        if not scores:
            assigned = [AgentRole.GENERAL.value]
        else:
            sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            assigned = [r.value for r, _ in sorted_roles[:2]]

        return {
            "input": user_input,
            "assigned_agents": assigned,
            "is_multi_agent": len(assigned) > 1,
        }

    def is_complex_prompt(self, user_input: str) -> bool:
        summary = self.get_input_routing_summary(user_input)
        assigned = summary.get("assigned_agents", [])
        return len(assigned) > 1

    async def delegate_to_subagents(self, user_input: str, **kwargs: Any) -> dict[str, Any]:
        """Spawn Researcher and Coder sub-agents in parallel for complex prompts."""
        logger.info("[Orchestrator] Complex prompt detected — spawning sub-agents")

        # Spawn both sub-agents in parallel
        researcher, coder = await asyncio.gather(
            self.spawn_agent(AgentRole.WEB, **kwargs),
            self.spawn_agent(AgentRole.CODE, **kwargs),
        )

        return {
            "researcher": researcher.role.value if researcher else None,
            "coder": coder.role.value if coder else None,
            "status": "delegated",
            "input": user_input,
        }
