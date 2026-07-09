## 🤖 Agent Development Guide — Dynamic Registry

Welcome to the **Heliox OS** Agent Development Guide! This tutorial walks you through how to build and plug in a custom agent using the new dynamic registry system introduced in PR #160.

---

## 📖 Overview

Heliox OS uses a **dynamic agent registry** that automatically discovers and registers agents at startup. Instead of manually wiring up agents, you simply:

1. Subclass `BaseAgent`
2. Decorate your class with `@auto_register`
3. Place it in the `daemon/pilot/agents/` package

The registry handles the rest.

---

## 🧩 Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `AgentRegistry` | `daemon/pilot/agents/registry.py` | Singleton registry that stores all agents |
| `BaseAgent` | `daemon/pilot/agents/base_agent.py` | Abstract base class all agents must extend |
| `auto_register` | `daemon/pilot/agents/registry.py` | Decorator that registers an agent on import |
| `discover_agents()` | `daemon/pilot/agents/registry.py` | Scans the package and imports all agent modules |

---

## 🚀 Step 1 — Understand the Registry

The `AgentRegistry` is a **singleton** — there's only one instance across the entire application.

```python
from pilot.agents.registry import AgentRegistry

# Get the single registry instance
registry = AgentRegistry.get_instance()

# See all registered agents
all_agents = registry.get_all_agents()  # returns dict[str, type[BaseAgent]]
print(all_agents)
```

At startup, `discover_agents()` is called to scan the `pilot.agents` package and import every module, which triggers the `@auto_register` decorator on each agent class.

```python
from pilot.agents.registry import discover_agents

discover_agents(package_name="pilot.agents")
```

---

## 🔧 Step 2 — Create Your Custom Agent

Here's a complete example of a custom agent. Create a new file in `daemon/pilot/agents/`, for example `my_agent.py`:

```python
import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent
from pilot.agents.registry import auto_register

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.my_agent")

# Define the action types this agent handles
MY_ACTION_TYPES: set[ActionType] = {
    ActionType.FILE_READ,  # replace with the relevant ActionType(s)
}

@auto_register
class MyAgent(BaseAgent):
    """Specialist agent for handling my specific domain."""

    def __init__(self, model_router: "ModelRouter") -> None:
        super().__init__(role=AgentRole.GENERAL, model_router=model_router)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.FILE_READ,
                description="Handles file reading tasks",
                requires_sandbox=False,
            )
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are a specialist agent. Your job is to handle "
            "file reading and related tasks efficiently."
        )

    def get_resource_needs(self) -> set[str]:
        return {"filesystem"}

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in MY_ACTION_TYPES

    async def handle_task(self, action_plan: ActionPlan, **kwargs: Any) -> ActionResult:
        self.status = AgentStatus.BUSY
        logger.info("MyAgent handling task: %s", action_plan.action_type)

        try:
            # Your task logic goes here
            result_data = {"message": "Task completed successfully"}

            self.status = AgentStatus.IDLE
            return ActionResult(success=True, data=result_data)

        except Exception as exc:
            logger.exception("MyAgent failed: %s", exc)
            self.status = AgentStatus.IDLE
            return ActionResult(success=False, error=str(exc))
```

---

## ✅ Step 3 — Verify Registration

After creating your agent, you can verify it was registered correctly:

```python
from pilot.agents.registry import AgentRegistry, discover_agents

# Trigger discovery (done automatically at startup)
discover_agents(package_name="pilot.agents")

registry = AgentRegistry.get_instance()
all_agents = registry.get_all_agents()

print("MyAgent" in all_agents)  # True
```

For testing, you can reset the registry:

```python
AgentRegistry.clear()  # Clears all registered agents
```

---

## 🛡️ Step 4 — Permission Tiers

Heliox OS enforces a **five-tier permission system**. Make sure your agent requests only the permissions it actually needs:

| Tier | Description | Example Use Case |
|------|-------------|-----------------|
| 0 | Read-only, sandboxed | Summarisation, analysis |
| 1 | Read + limited write | File creation in temp dirs |
| 2 | Full filesystem access | Refactoring, build tools |
| 3 | Network + system calls | API integrations, scrapers |
| 4 | Unrestricted | Core system agents only |

Set your agent's permission tier by overriding `get_permission_tier()` in `BaseAgent` if needed.

---

## 📂 Related Files

- `daemon/pilot/agents/registry.py` — Registry and `auto_register` decorator
- `daemon/pilot/agents/base_agent.py` — `BaseAgent` abstract class
- `daemon/pilot/agents/code_agent.py` — Real-world example agent
- `daemon/pilot/agents/orchestrator.py` — Agent orchestration logic

---

## 💡 Tips for Contributors

- **One agent per file** — keep each agent in its own module inside `daemon/pilot/agents/`
- **Use `@auto_register`** — don't manually add agents to the registry
- **Override `can_handle()`** — use a set of `ActionType` values for efficient lookup
- **Log with the module logger** — use `logging.getLogger("pilot.agents.your_agent")`
- **Handle exceptions in `handle_task()`** — always return an `ActionResult`, never raise
- **Request minimal permissions** — only declare the resources your agent truly needs

---

*This guide covers the dynamic registry introduced in PR #160. For questions, open an issue or check the existing agents in `daemon/pilot/agents/` for reference.*
