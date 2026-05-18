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

    def __init__(self, model_router: ModelRouter, swarm_manager: SwarmManager) -> None:
        super().__init__(role=AgentRole.SYSTEM, model_router=model_router)
        self._swarm = swarm_manager
        self._executor = None

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

        # Analyze task requirements
        requirements = self._analyze_task_requirements(plan)

        # Get the best node for this task
        try:
            node = await self._swarm.route_task(requirements)
            logger.info("Routing task to node %s", node.node_id)
        except RuntimeError as e:
            # Fall back to local execution
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

            # Code execution typically requires GPU for LLM tasks
            if action_type == ActionType.CODE_EXECUTE:
                requirements.requires_gpu = True
                requirements.minimal_vram_gb = 8  # Default to 8GB VRAM
                requirements.estimated_tokens = 1000

            # System-wide operations might need more resources
            if action_type == ActionType.SYSTEM_INFO:
                requirements.estimated_tokens = 2000

        return requirements

    async def _execute_on_node(
        self, node: Any, plan: ActionPlan
    ) -> list[ActionResult]:
        """Execute a plan on a specific node."""
        # For now, execute locally if no executor is set
        if not self._executor:
            self._executor = await self._get_executor()

        # Execute the plan
        results = await self._executor.execute(plan)

        # Update remote node stats if applicable
        if node and node.node_id != self._swarm._local_node.node_id:
            try:
                await self._swarm.execute_remote(
                    node, "task_complete", {"tasks_completed": len(results)}
                )
            except Exception as e:
                logger.warning("Failed to update node stats: %s", e)

        return results

    async def _get_executor(self) -> Any:
        """Get the executor from the orchestrator or create one."""
        if self._orchestrator:
            # Try to get executor from orchestrator
            if hasattr(self._orchestrator, "_executor"):
                return self._orchestrator._executor

        # Fallback: create a local executor
        from pilot.agents.executor import Executor
        from pilot.models.router import ModelRouter

        # This would need proper initialization - simplified for now
        return None

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