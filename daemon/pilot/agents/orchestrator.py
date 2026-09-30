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
import inspect
import logging
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.agent_mesh import AgentBudgetExceeded, AssignmentLease
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
from pilot.intelligence.experience import experience_scope
from pilot.models.budget_tracker import (
    ActionBudgetExceededError,
    BudgetExceededError,
    TaskBudgetExceededError,
    current_agent_id,
    current_task_id,
)
from pilot.security.gateway import mark_critic_already_reviewed


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
    from pilot.agents.agent_mesh import AgentMesh
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import AgentGateway, TaskScopeOverride

logger = logging.getLogger("pilot.agents.orchestrator")


class AgentOrchestrator:
    """Central coordinator that manages specialist agents and routes tasks.

    Architecture:
      User Input → Planner → Orchestrator → [Agent₁, Agent₂, ...] → Verifier

    The Orchestrator analyzes the plan's action types and dispatches each
    action to the specialist agent that owns that action type. For multi-agent
    tasks, it coordinates parallel or sequential execution and merges results.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        *,
        agent_mesh: AgentMesh | None = None,
    ):
        from pilot.agents.agent_mesh import AgentMesh

        self._model = model_router
        self._mesh = agent_mesh or AgentMesh()
        self._agents: dict[AgentRole, BaseAgent] = {}
        self._action_registry: dict[ActionType, AgentRole] = {}
        self._message_log: list[AgentMessage] = []
        self._broadcast_fn: Callable[..., Coroutine] | None = None
        self._budget_tracker: BudgetTracker | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._agent_gateway: AgentGateway | None = None
        # ThreatContainmentBridge — injected via set_threat_bridge() in server.py
        self._threat_bridge: Any = None

        self.task_queue = asyncio.PriorityQueue()

        # This event acts as our "Freeze/Resume" switch for background tasks
        self.background_allowed = asyncio.Event()
        self.background_allowed.set()  # Start unpaused

        # Start the continuous scheduler loop in the background
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._scheduler_task = asyncio.create_task(self.scheduler_loop())

    async def stop(self) -> None:
        """Cancel the scheduler and every task it owns."""
        if hasattr(self, "_scheduler_task") and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)

        while not self.task_queue.empty():
            queued = self.task_queue.get_nowait()
            self._dispose_awaitable(queued.coro)
            self.task_queue.task_done()

        background_tasks = tuple(self._background_tasks)
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self.background_allowed.set()

    @staticmethod
    def _dispose_awaitable(awaitable: Any) -> None:
        """Cancel or close work that will never be scheduled."""
        if isinstance(awaitable, asyncio.Future):
            awaitable.cancel()
        elif inspect.iscoroutine(awaitable):
            awaitable.close()

    def _background_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Scheduled background task failed")

    def set_budget_tracker(self, tracker: BudgetTracker) -> None:
        """Inject the budget tracker. Called by server.py during startup."""
        self._budget_tracker = tracker

    def set_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """Inject the circuit breaker. Called by server.py during startup."""
        self._circuit_breaker = breaker

    def set_agent_gateway(self, gateway: AgentGateway) -> None:
        """Guard specialists that perform work without the shared Executor."""
        self._agent_gateway = gateway

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
            work_started = False
            try:
                if p_task.priority == TaskPriority.USER_REALTIME:
                    # INTERRUPT: Freeze all background tasks
                    logger.info("High-priority interrupt received: %s. Suspending background tasks.", p_task.task_id)
                    self.background_allowed.clear()
                    work_started = True
                    try:
                        await p_task.coro
                    except Exception:
                        logger.exception("Real-time scheduled task %s failed", p_task.task_id)
                    finally:
                        # A failed interactive task must never suppress every
                        # lower-priority feature indefinitely.
                        self.background_allowed.set()
                        logger.info("Real-time task %s complete. Resuming background tasks.", p_task.task_id)
                else:
                    # Wait until we are allowed to run background tasks.
                    await self.background_allowed.wait()
                    task = asyncio.create_task(p_task.coro, name=f"scheduled_{p_task.task_id}")
                    work_started = True
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_done)
            except asyncio.CancelledError:
                if not work_started:
                    self._dispose_awaitable(p_task.coro)
                raise
            finally:
                self.task_queue.task_done()

    # ── Agent Registration ──

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent and index its capabilities."""
        agent.attach_orchestrator(self)
        # Multiple providers may share one gateway role. Keep the first as
        # the compatibility primary while retaining every provider in the
        # capability mesh.
        self._agents.setdefault(agent.role, agent)
        self._mesh.register_agent(agent)

        # Index action types → agent role
        for cap in agent.get_capabilities():
            self._action_registry.setdefault(cap.action_type, agent.role)

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
            for specialist in list(self._mesh.status()["specialists"]):
                if specialist["role"] == role.value and specialist["source"] == "builtin":
                    self._mesh.unregister(specialist["agent_key"])
            # Clean up action registry
            self._action_registry = {at: r for at, r in self._action_registry.items() if r != role}
            logger.info("Unregistered agent %s", role.value)

    def get_agent(self, role: AgentRole) -> BaseAgent | None:
        """Get a registered agent by role."""
        return self._agents.get(role)

    def get_agents(self, role: AgentRole) -> tuple[BaseAgent, ...]:
        """Return every specialist sharing a role, including mesh providers."""
        return tuple(agent for agent in self._mesh.executable_agents() if agent.role == role)

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
        critic_already_reviewed: bool = False,
        user_confirmed: bool = False,
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
            review_scope = mark_critic_already_reviewed() if critic_already_reviewed else nullcontext()
            with review_scope:
                return await self._execute_plan_inner(
                    user_input=user_input,
                    plan=plan,
                    task_id=task_id,
                    on_action_start=on_action_start,
                    on_action_complete=on_action_complete,
                    cancel_event=cancel_event,
                    scope_override=scope_override,
                    user_confirmed=user_confirmed,
                )
        finally:
            if self._budget_tracker:
                self._budget_tracker.end_task(task_id)
            if self._circuit_breaker:
                self._circuit_breaker.reset(task_id)
            self._mesh.release_task(task_id)
            current_task_id.reset(ctx_token)

    async def execute_independent_plans(
        self,
        requests: list[tuple[str, ActionPlan]],
        *,
        independence_attested: bool,
        cancel_event: asyncio.Event | None = None,
    ) -> list[dict[str, Any]]:
        """Run only explicitly independent plans with bounded fan-out.

        The ordinary action path remains sequential. Callers must attest that
        these plans have no output/data dependency, and all branches share one
        cancellation event so a safety or budget stop propagates.
        """
        from pilot.agents.agent_mesh import MAX_HANDOFF_FANOUT

        if not independence_attested:
            raise ValueError("parallel execution requires an explicit independence attestation")
        if not requests:
            return []
        if len(requests) > MAX_HANDOFF_FANOUT:
            raise ValueError(f"parallel execution fan-out exceeds {MAX_HANDOFF_FANOUT}")
        shared_cancel = cancel_event or asyncio.Event()

        async def run_branch(index: int, user_input: str, plan: ActionPlan) -> dict[str, Any]:
            if shared_cancel.is_set():
                return {"index": index, "status": "cancelled", "results": []}
            try:
                results = await self.execute_plan(
                    user_input,
                    plan,
                    cancel_event=shared_cancel,
                    plan_id=f"parallel-{uuid.uuid4()}",
                )
                return {
                    "index": index,
                    "status": "completed",
                    "results": results,
                }
            except Exception as exc:
                shared_cancel.set()
                return {
                    "index": index,
                    "status": "failed",
                    "error": str(exc),
                    "results": [],
                }

        branches = [run_branch(index, user_input, plan) for index, (user_input, plan) in enumerate(requests)]
        return list(await asyncio.gather(*branches))

    async def _execute_plan_inner(
        self,
        user_input: str,
        plan: ActionPlan,
        task_id: str,
        on_action_start: Callable | None,
        on_action_complete: Callable | None,
        cancel_event: asyncio.Event | None,
        scope_override: TaskScopeOverride | None = None,
        user_confirmed: bool = False,
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
        action_order = self._build_execution_order(plan, routing, task_id=task_id)
        prior_last_output = ""
        prior_largest_output = ""

        for batch in action_order:
            # â”€â”€ Cancellation check â”€â”€
            if cancel_event and cancel_event.is_set():
                logger.info("Orchestrator: cancel_event set â€” halting plan execution")
                break

            provider, indices = batch
            if isinstance(provider, AgentRole):
                role = provider
                agent = self._agents.get(role)
                agent_key = self._mesh.key_for_agent(agent) if agent is not None else None
            else:
                agent_key = provider
                agent = self._mesh.agent_for(agent_key)
                role = agent.role if agent is not None else AgentRole.SYSTEM

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

            if agent is None:
                # Fallback to system agent
                agent = self._agents.get(AgentRole.SYSTEM)
                agent_key = self._mesh.key_for_agent(agent) if agent is not None else None
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

            # Some specialists integrate directly with a protocol/library
            # instead of using Executor. They must still be constrained by
            # the Agent Gateway and the caller's narrowed scope. Executor-
            # backed agents perform this check inside Executor already.
            if self._agent_gateway is not None and not hasattr(agent, "_executor"):
                gateway_decision = await self._agent_gateway.authorize(
                    sub_plan,
                    agent.get_invocation_source(),
                    scope_override=scope_override,
                    plan_id=task_id,
                )
                if not gateway_decision.allowed:
                    denial = "; ".join(gateway_decision.reasons) or "Denied by Agent Gateway"
                    for idx in indices:
                        all_results[idx] = ActionResult(
                            action=plan.actions[idx],
                            success=False,
                            error=denial,
                        )
                    if cancel_event:
                        cancel_event.set()
                    break

            # Notify action starts
            if on_action_start:
                for action in batch_actions:
                    await on_action_start(action)

            lease: AssignmentLease | None = None
            if agent_key is not None:
                try:
                    lease = self._mesh.begin_assignment(
                        task_id=task_id,
                        agent_key=agent_key,
                        action_type=",".join(action.action_type.value for action in batch_actions),
                        action_count=len(batch_actions),
                    )
                except AgentBudgetExceeded as exc:
                    logger.warning("Specialist budget rejected assignment: %s", exc)
                    for idx in indices:
                        all_results[idx] = ActionResult(
                            action=plan.actions[idx],
                            success=False,
                            error=f"Specialist budget exceeded: {exc}",
                        )
                    if cancel_event:
                        cancel_event.set()
                    break

            # Execute via the specialist agent. Wrap with budget exception
            # handling: if a per-action / per-task / monthly cap fires inside
            # the agent's LLM calls, we halt the plan cleanly instead of
            # letting the exception escape and leave the task in limbo.
            try:
                agent_context_token = current_agent_id.set(agent_key)
                if agent_key is not None and self._budget_tracker is not None:
                    contract = self._mesh.contract_for(agent_key)
                    if contract is not None:
                        self._budget_tracker.start_agent_budget(
                            task_id,
                            agent_key,
                            contract.budget.max_tokens_per_task,
                        )
                try:
                    with experience_scope(
                        task_id=task_id,
                        plan_id=task_id,
                        action_index_offset=indices[0],
                    ):
                        results = await agent.handle_task(
                            user_input,
                            sub_plan,
                            context={
                                "initial_last_output": prior_last_output,
                                "initial_largest_output": prior_largest_output,
                                "user_confirmed": user_confirmed,
                            },
                            scope_override=scope_override,
                        )
                finally:
                    current_agent_id.reset(agent_context_token)
            except (
                ActionBudgetExceededError,
                TaskBudgetExceededError,
                BudgetExceededError,
            ) as exc:
                if lease is not None:
                    await self._mesh.complete_assignment(
                        lease,
                        success=False,
                        error=str(exc),
                    )
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
            except Exception as exc:
                if lease is not None:
                    await self._mesh.complete_assignment(
                        lease,
                        success=False,
                        error=str(exc),
                    )
                raise
            else:
                if lease is not None:
                    await self._mesh.complete_assignment(
                        lease,
                        success=bool(results) and all(result.success for result in results),
                        error="; ".join(result.error or "" for result in results if not result.success),
                    )

            # Map results back to original indices
            # Map results back to original indices and update breaker state
            for idx, result in zip(indices, results):
                all_results[idx] = result
                if result.success:
                    prior_last_output = result.output or ""
                    if len(prior_last_output) > len(prior_largest_output):
                        prior_largest_output = prior_last_output
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
        *,
        task_id: str = "",
    ) -> list[tuple[AgentRole | str, list[int]]]:
        """Build execution batches preserving action order.

        Groups consecutive actions for the same agent, but maintains
        the overall sequential order of the plan.
        """
        if not plan.actions:
            return []

        batches: list[tuple[AgentRole | str, list[int]]] = []
        current_provider: AgentRole | str | None = None
        current_indices: list[int] = []

        for i in range(len(plan.actions)):
            action_type = plan.actions[i].action_type
            selected = self._mesh.select_provider(
                action_type,
                task_id=task_id or "routing-preview",
            )
            if selected is None:
                # A registered specialist may be temporarily saturated or at
                # its per-task limit. Preserve its identity so
                # begin_assignment() reports the bounded-capacity failure;
                # falling back by gateway role can select an unrelated agent
                # that happens to share that role (for example notifications
                # instead of email).
                selected = self._mesh.registered_provider(action_type)
            provider: AgentRole | str
            if selected is not None:
                provider = selected[0]
            else:
                provider = self._action_registry.get(action_type, AgentRole.SYSTEM)
            if provider != current_provider:
                if current_indices:
                    assert current_provider is not None
                    batches.append((current_provider, current_indices))
                current_provider = provider
                current_indices = [i]
            else:
                current_indices.append(i)

        if current_indices and current_provider is not None:
            batches.append((current_provider, current_indices))

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

        exact_target = self._mesh.agent_for(message.recipient)
        if exact_target is not None:
            if message.msg_type == "handoff" or message.delegation_depth:
                await self._mesh.authorize_handoff(
                    root_id=message.root_id or message.correlation_id or message.id,
                    sender=message.sender,
                    recipient=message.recipient,
                    depth=message.delegation_depth,
                    lineage=message.lineage,
                )
            return await exact_target.receive_message(message)

        # Find target agent by role
        target_role = None
        for role in AgentRole:
            if role.value == message.recipient:
                target_role = role
                break

        if target_role and target_role in self._agents:
            target = self._agents[target_role]
            if message.msg_type == "handoff" or message.delegation_depth:
                target_key = self._mesh.key_for_agent(target)
                if target_key is not None:
                    await self._mesh.authorize_handoff(
                        root_id=message.root_id or message.correlation_id or message.id,
                        sender=message.sender,
                        recipient=target_key,
                        depth=message.delegation_depth,
                        lineage=message.lineage,
                    )
            return await target.receive_message(message)

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

    async def spawn_registered_agent(
        self,
        agent_name: str,
        **kwargs: Any,
    ) -> BaseAgent | None:
        """Spawn any discovered specialist class without a role switch."""
        import inspect

        from pilot.agents.registry import AgentRegistry

        agent_class = AgentRegistry.get_agent_class(agent_name)
        if agent_class is None:
            return None
        stable_key = f"builtin:{agent_class.__module__}.{agent_class.__name__}"
        existing = self._mesh.agent_for(stable_key)
        if existing is not None:
            return existing
        signature = inspect.signature(agent_class.__init__)
        accepted = {name for name in signature.parameters if name != "self"}
        accepts_var_kw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        filtered = kwargs if accepts_var_kw else {key: value for key, value in kwargs.items() if key in accepted}
        try:
            agent = agent_class(**filtered)
            self.register_agent(agent)
            await agent.start()
            return agent
        except Exception:
            logger.exception("Failed to spawn discovered specialist %s", agent_name)
            return None

    # ── Lifecycle ──

    async def start_all(self) -> None:
        """Start all registered agents."""
        for agent in self._mesh.executable_agents():
            await agent.start()

    async def stop_all(self) -> None:
        """Stop all registered agents."""
        for agent in self._mesh.executable_agents():
            await agent.stop()

    # ── Stats & Diagnostics ──

    def get_all_stats(self) -> dict[str, Any]:
        """Return statistics for all registered agents."""
        mesh_status = self._mesh.status()
        return {
            "agents": {
                specialist["agent_key"]: {
                    **specialist["performance"],
                    "role": specialist["role"],
                    "display_name": specialist["display_name"],
                    "source": specialist["source"],
                }
                for specialist in mesh_status["specialists"]
            },
            "total_agents": mesh_status["total_specialists"],
            "executable_agents": mesh_status["executable_specialists"],
            "registered_actions": len(self._action_registry),
            "message_count": len(self._message_log),
            "mesh": mesh_status,
        }

    def get_all_capabilities(self) -> dict[str, list[str]]:
        """Return all capabilities grouped by agent."""
        return {
            specialist["agent_key"]: list(specialist["capabilities"])
            for specialist in self._mesh.status()["specialists"]
        }

    def get_input_routing_summary(self, user_input: str) -> dict[str, Any]:
        """Legacy compatibility — route by user input keywords (like old MultiAgentRouter)."""
        mesh_matches = self._mesh.route_text(user_input)
        if mesh_matches:
            assigned = [match["role"] for match in mesh_matches]
            return {
                "input": user_input,
                "assigned_agents": assigned,
                "assigned_specialists": mesh_matches,
                "is_multi_agent": len(assigned) > 1,
                "routing_basis": "capability_contract",
            }
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
