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
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    BaseAgent,
)
from pilot.agents.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
)
from pilot.models.budget_tracker import (
    ActionBudgetExceededError,
    BudgetExceededError,
    TaskBudgetExceededError,
    current_task_id,
)


class TaskPriority(IntEnum):
    USER_REALTIME = 0  # Voice commands, UI clicks (Immediate)
    SYSTEM_CRITICAL = 1  # OS-level alerts
    BACKGROUND_BATCH = 2  # Local file indexing, scraping


@dataclass(order=True)
class PrioritizedTask:
    priority: TaskPriority
    task_id: str = field(compare=False)
    coro: Any = field(compare=False)


if TYPE_CHECKING:
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride

logger = logging.getLogger("pilot.agents.orchestrator")


class AgentOrchestrator:
    """Central coordinator that manages specialist agents and routes tasks.

    Architecture:
      User Input → Planner → Orchestrator → [Agent₁, Agent₂, ...] → Verifier

    The Orchestrator analyzes the plan's action types and dispatches each
    action to the specialist agent that owns that action type. For multi-agent
    tasks, it coordinates parallel or sequential execution and merges results.
    """

    def __init__(self, model_router: ModelRouter):
        self._model = model_router
        self._agents: dict[AgentRole, BaseAgent] = {}
        self._action_registry: dict[ActionType, AgentRole] = {}
        self._message_log: list[AgentMessage] = []
        self._broadcast_fn: Callable[..., Coroutine] | None = None
        self._budget_tracker: BudgetTracker | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        # ThreatContainmentBridge — injected via set_threat_bridge() in server.py
        self._threat_bridge: Any = None

        self.task_queue = asyncio.PriorityQueue()

        # This event acts as our "Freeze/Resume" switch for background tasks
        self.background_allowed = asyncio.Event()
        self.background_allowed.set()  # Start unpaused

        # Start the continuous scheduler loop in the background
        self._scheduler_task = asyncio.create_task(self.scheduler_loop())

    async def stop(self) -> None:
        """Cancel the background scheduler task cleanly."""
        if hasattr(self, "_scheduler_task") and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    def set_budget_tracker(self, tracker: BudgetTracker) -> None:
        """Inject the budget tracker. Called by server.py during startup."""
        self._budget_tracker = tracker

    def set_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """Inject the circuit breaker. Called by server.py during startup."""
        self._circuit_breaker = breaker

    def set_threat_bridge(self, bridge: Any) -> None:
        """Inject the ThreatContainmentBridge and wire it to the ForensicsAgent.

        Looks up the ForensicsAgent in the agent registry and calls
        ``set_threat_bridge()`` on it.  Safe to call before or after agent
        registration — if the ForensicsAgent is not yet registered the bridge
        is stored and applied when the agent is registered later.

        Args:
            bridge: An initialized :class:`~pilot.agents.threat_containment.ThreatContainmentBridge`.
        """
        self._threat_bridge = bridge

        # Wire to the already-registered ForensicsAgent (if present)
        forensics_agent = self._agents.get(AgentRole.FORENSICS)
        if forensics_agent is not None and hasattr(forensics_agent, "set_threat_bridge"):
            forensics_agent.set_threat_bridge(bridge)
            logger.info("ThreatContainmentBridge wired to ForensicsAgent via Orchestrator.")
        else:
            logger.info("ThreatContainmentBridge stored; will wire to ForensicsAgent on registration.")

    async def scheduler_loop(self):
        """Continuously pulls tasks from the priority queue and handles context switching."""
        while True:
            # Pull the highest priority task (lowest integer value)
            p_task = await self.task_queue.get()

            if p_task.priority == TaskPriority.USER_REALTIME:
                # INTERRUPT: Freeze all background tasks
                logger.info(f"High-priority interrupt received: {p_task.task_id}. Suspending background tasks.")
                self.background_allowed.clear()

                # Execute the real-time task immediately
                await p_task.coro

                # RESUME: Unfreeze background tasks
                logger.info(f"Real-time task {p_task.task_id} complete. Resuming background tasks.")
                self.background_allowed.set()

            else:
                # Wait until we are allowed to run background tasks
                await self.background_allowed.wait()
                # Fire and forget the background task so the loop can keep listening
                asyncio.create_task(p_task.coro)

            self.task_queue.task_done()

    # ── Agent Registration ──

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent and index its capabilities."""
        agent.attach_orchestrator(self)
        self._agents[agent.role] = agent

        # Index action types → agent role
        for cap in agent.get_capabilities():
            self._action_registry[cap.action_type] = agent.role

        # Auto-wire ThreatContainmentBridge to the ForensicsAgent
        if agent.role == AgentRole.FORENSICS and self._threat_bridge is not None:
            if hasattr(agent, "set_threat_bridge"):
                agent.set_threat_bridge(self._threat_bridge)
                logger.info("ThreatContainmentBridge auto-wired to ForensicsAgent on registration.")

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

    def auto_register_all_agents(
        self,
        executor: Any = None,
        background_manager: Any = None,
        model_router: Any = None,
        config: Any = None,
        vault: Any = None,
        memory: Any = None,
    ) -> int:
        """Auto-register all discovered agents from the registry."""
        import inspect

        from pilot.agents.registry import AgentRegistry

        count = 0
        for name, agent_class in AgentRegistry.get_all_agents().items():
            try:
                kwargs = {}
                if executor:
                    kwargs["executor"] = executor
                if background_manager:
                    kwargs["background_manager"] = background_manager
                if model_router:
                    kwargs["model_router"] = model_router
                if config:
                    kwargs["config"] = config
                if vault:
                    kwargs["vault"] = vault
                if memory:
                    kwargs["memory"] = memory

                # Only pass supported kwargs to avoid breaking agents with narrower constructors.
                sig = inspect.signature(agent_class.__init__)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if accepts_var_kw:
                    filtered = kwargs
                else:
                    accepted = {k for k in sig.parameters if k != "self"}
                    filtered = {k: v for k, v in kwargs.items() if k in accepted}

                agent = agent_class(**filtered)
                self.register_agent(agent)
                count += 1
            except Exception as e:
                logger.warning("Failed to auto-register agent %s: %s", name, e)

        return count

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
        cancel_event: asyncio.Event | None = None,
        plan_id: str | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        """Execute a plan by routing actions to specialist agents.

        For single-agent tasks, delegates directly.
        For multi-agent tasks, runs agents sequentially (preserving action order)
        while allowing each agent to process its batch in parallel internally.

        Establishes a per-task budget context: generates a task_id (or uses
        plan_id if provided), starts a TaskBudget in the tracker, and threads
        the id through via the current_task_id ContextVar so downstream model
        calls and record_usage attribute their tokens to this task. The task
        is cleaned up in a finally block regardless of how execution ends.

        scope_override: an optional gateway narrowing constraint (see
        pilot.security.gateway) applied uniformly across every specialist
        this plan gets routed to — e.g. a voice-originated plan should never
        exceed voice's ceiling, regardless of which agent (or that agent's
        own hardcoded invocation_source) ends up handling a given action.
        None (the default) preserves today's behavior for existing callers.
        """
        task_id = plan_id or str(uuid.uuid4())
        ctx_token = current_task_id.set(task_id)
        if self._budget_tracker:
            self._budget_tracker.start_task(task_id)

        try:
            return await self._execute_plan_inner(
                user_input=user_input,
                plan=plan,
                task_id=task_id,
                on_action_start=on_action_start,
                on_action_complete=on_action_complete,
                cancel_event=cancel_event,
                scope_override=scope_override,
            )
        finally:
            if self._budget_tracker:
                self._budget_tracker.end_task(task_id)
            if self._circuit_breaker:
                self._circuit_breaker.reset(task_id)
            current_task_id.reset(ctx_token)

    async def _execute_plan_inner(
        self,
        user_input: str,
        plan: ActionPlan,
        task_id: str,
        on_action_start: Callable | None,
        on_action_complete: Callable | None,
        cancel_event: asyncio.Event | None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        """Inner execution loop — extracted so the task lifecycle wrapper
        in execute_plan() stays small and the try/finally is obvious."""
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
            # â”€â”€ Cancellation check â”€â”€
            if cancel_event and cancel_event.is_set():
                logger.info("Orchestrator: cancel_event set â€” halting plan execution")
                break

            role, indices = batch

            # Circuit breaker pre-check — bail if too many consecutive failures
            if self._circuit_breaker:
                try:
                    self._circuit_breaker.check()
                except CircuitBreakerOpenError as exc:
                    logger.warning(
                        "Circuit breaker tripped mid-plan (task_id=%s): %s",
                        task_id,
                        exc,
                    )
                    if self._broadcast_fn:
                        await self._broadcast_fn(
                            "circuit_breaker_tripped",
                            {
                                "task_id": task_id,
                                "error": str(exc),
                                "failure_count": self._circuit_breaker.get_failure_count(),
                            },
                        )
                    for idx in indices:
                        if all_results[idx] is None:
                            all_results[idx] = ActionResult(
                                action=plan.actions[idx],
                                success=False,
                                error=f"Circuit breaker tripped: {exc}",
                            )
                    if cancel_event:
                        cancel_event.set()
                    break

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

            # Execute via the specialist agent. Wrap with budget exception
            # handling: if a per-action / per-task / monthly cap fires inside
            # the agent's LLM calls, we halt the plan cleanly instead of
            # letting the exception escape and leave the task in limbo.
            try:
                results = await agent.handle_task(user_input, sub_plan, scope_override=scope_override)
            except (
                ActionBudgetExceededError,
                TaskBudgetExceededError,
                BudgetExceededError,
            ) as exc:
                logger.warning(
                    "Budget exhausted mid-plan (task_id=%s): %s",
                    task_id,
                    exc,
                )
                if self._broadcast_fn:
                    await self._broadcast_fn(
                        "budget_exceeded",
                        {
                            "task_id": task_id,
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                        },
                    )
                # Mark every still-unresolved action in this batch as failed
                # with a budget-specific error so downstream consumers can
                # distinguish budget halts from action failures.
                for idx in indices:
                    if all_results[idx] is None:
                        all_results[idx] = ActionResult(
                            action=plan.actions[idx],
                            success=False,
                            error=f"Budget exceeded: {exc}",
                        )
                # Halt subsequent batches.
                if cancel_event:
                    cancel_event.set()
                break

            # Map results back to original indices
            # Map results back to original indices and update breaker state
            for idx, result in zip(indices, results):
                all_results[idx] = result
                if self._circuit_breaker:
                    if result.success:
                        self._circuit_breaker.record_success()
                    else:
                        self._circuit_breaker.record_failure()
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
            elif role == AgentRole.SSH:
                from pilot.agents.ssh_agent import SshAgent

                agent = SshAgent(self._model)
            elif role == AgentRole.CODE:
                from pilot.agents.code_agent import CodeAgent

                agent = CodeAgent(self._model, kwargs.get("executor"))
            elif role == AgentRole.WEB:
                from pilot.agents.web_agent import WebAgent

                agent = WebAgent(self._model, kwargs.get("executor"))
            elif role == AgentRole.MONITOR:
                from pilot.agents.monitor_agent import MonitorAgent

                agent = MonitorAgent(self._model, kwargs.get("background_manager"))
            elif role == AgentRole.FORENSICS:
                from pilot.agents.forensics_agent import ForensicsAgent

                agent = ForensicsAgent(self._model, kwargs.get("executor"))
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
            AgentRole.SSH: [
                "ssh",
                "remote",
                "server",
                "hostname",
                "host",
                "bastion",
                "jump host",
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
            AgentRole.FORENSICS: [
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
