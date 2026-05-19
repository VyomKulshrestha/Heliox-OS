"""Swarm Router Agent - dispatches tasks to appropriate daemon nodes based on hardware requirements."""

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent
from pilot.swarm.swarm_manager import HardwareCapability, SwarmManager, TaskRequirements

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.swarm.swarm_router_agent")


class SwarmRouterAgent(BaseAgent):
    """Specialist agent for distributed task routing in swarm mode."""

    def __init__(
        self,
        model_router: "ModelRouter",
        swarm_manager: SwarmManager,
        executor: Any = None,
    ) -> None:
        super().__init__(role=AgentRole.SYSTEM, model_router=model_router)
        self._swarm = swarm_manager
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        """Return supported capabilities."""
        return [
            AgentCapability(
                action_type=ActionType.SYSTEM_INFO,
                description="System information aggregation across swarm",
            ),
            AgentCapability(
                action_type=ActionType.CODE_EXECUTE,
                description="Distributed code execution with hardware-aware routing",
            ),
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt for swarm router."""
        return (
            "You are the SWARM ROUTER AGENT for Heliox OS. "
            "You handle distributed task routing across multiple daemon nodes. "
            "When tasks are received, route them to the most appropriate node based on "
            "hardware requirements (CPU, GPU VRAM, etc.). "
            "For memory-intensive tasks, prefer nodes with higher VRAM. "
            "For simple tasks, use the local node or any available node. "
            "Monitor swarm status and avoid routing to unhealthy nodes."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        """Check if this agent can handle the given action type."""
        capabilities = [ActionType.SYSTEM_INFO, ActionType.CODE_EXECUTE]
        return action_type in capabilities

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute actions by routing to appropriate swarm nodes."""
        self.status = AgentStatus.BUSY

        if not plan.actions:
            self.status = AgentStatus.IDLE
            return []

        requirements = self._analyze_task_requirements(plan)

        try:
            node = await self._swarm.route_task(requirements)
            logger.info("Routing task to node %s", node.node_id)
        except RuntimeError as e:
            logger.warning("No remote node available: %s. Using local execution.", e)
            node = self._swarm._local_node

        results = await self._execute_on_node(node, plan)

        self.status = AgentStatus.IDLE
        return results

    def _analyze_task_requirements(self, plan: ActionPlan) -> TaskRequirements:
        """Analyze a plan to determine hardware requirements."""
        requirements = TaskRequirements()

        for action in plan.actions:
            action_type = action.action_type

            if action_type == ActionType.CODE_EXECUTE:
                requirements.requires_gpu = True
                requirements.minimal_vram_gb = 8
                requirements.estimated_tokens = 1000

            if action_type == ActionType.SYSTEM_INFO:
                requirements.estimated_tokens = 2000

        return requirements

    async def _execute_on_node(self, node: Any, plan: ActionPlan) -> list[ActionResult]:
        """Execute a plan on a specific node (local or remote)."""
        local_node_id = self._swarm._local_node.node_id if self._swarm._local_node else None
        is_remote = node and hasattr(node, "node_id") and local_node_id is not None and node.node_id != local_node_id

        if is_remote:
            logger.info("Executing on remote node %s", node.node_id)
            try:
                plan_dict = {"input": plan.input, "dry_run": plan.dry_run}
                result = await self._swarm.execute_remote(node, plan_dict)
                remote_results = result.get("results", [])
                if remote_results:
                    return [
                        ActionResult(
                            action=action,
                            success=r.get("success", True),
                            output=r.get("output"),
                            error=r.get("error"),
                        )
                        for action, r in zip(plan.actions, remote_results)
                    ]
                return [
                    ActionResult(
                        action=plan.actions[0] if plan.actions else None,
                        success=result.get("status") != "error",
                        output=result,
                    )
                ]
            except Exception as e:
                logger.error("Remote execution failed: %s. Falling back to local.", e)
                is_remote = False

        if not is_remote:
            if not self._executor:
                logger.warning("Executor not available")
                return [
                    ActionResult(
                        action=plan.actions[0] if plan.actions else None,
                        success=False,
                        error="Swarm executor not initialized - no local fallback available",
                    )
                ]

            logger.info("Executing locally on node %s", local_node_id or "unknown")
            results = await self._executor.execute(plan)

            if node and hasattr(node, "node_id"):
                try:
                    node.tasks_completed += 1
                except Exception as e:
                    logger.warning("Failed to update node stats: %s", e)

            return results

        return []

    async def get_swarm_status(self) -> dict[str, Any]:
        """Get current swarm status."""
        return {
            "total_nodes": len(self._swarm._nodes),
            "healthy_nodes": sum(1 for n in self._swarm._nodes if n.is_healthy),
            "local_node": {
                "node_id": self._swarm._local_node.node_id if self._swarm._local_node else None,
                "addr": self._swarm._local_node.addr if self._swarm._local_node else None,
                "hardware": self._swarm._local_node.hardware if self._swarm._local_node else None,
            },
        }
