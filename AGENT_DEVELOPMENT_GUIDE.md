# 🤖 Agent Development Guide — Dynamic Registry

This guide walks you through how to build and plug in a custom agent into Helix OS using the dynamic registry system introduced in PR #160.

---

## 📖 Overview

Previously, agents were hardcoded into the orchestrator. With the new dynamic registry, agents **self-register** by declaring their capabilities at startup. The orchestrator dynamically routes actions based on those declared capabilities — no manual wiring needed!

**Key components:**

| Component | File | Role |
|-----------|------|------|
| `AgentRegistry` | `daemon/pilot/agents/registry.py` | Singleton registry that discovers and stores all agents |
| `discover_agents()` | `daemon/pilot/agents/registry.py` | Scans the `pilot.agents` package for all `BaseAgent` subclasses |
| `auto_register` | `daemon/pilot/agents/registry.py` | Decorator to explicitly register an agent class |
| `BaseAgent` | `daemon/pilot/agents/base_agent.py` | Abstract base class all agents must inherit from |
| `AgentRole` | `daemon/pilot/agents/base_agent.py` | Enum defining the agent's canonical role |
| `AgentCapability` | `daemon/pilot/agents/base_agent.py` | Describes one action/tool an agent can perform |

---

## 🏗️ Step 1 — Understand BaseAgent

Every custom agent must inherit from `BaseAgent` and implement **three abstract methods**. Here's the real import pattern used in the codebase:

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.my_agent")
```

The three abstract methods you **must** implement:

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_capabilities()` | `list[AgentCapability]` | Declares what actions this agent can handle |
| `get_system_prompt()` | `str` | The LLM system prompt for this agent |
| `handle_task()` | `list[ActionResult]` | Executes the actual task logic |

---

## 🔧 Step 2 — Create Your Custom Agent

Create a new file in `daemon/pilot/agents/`. For example: `my_agent.py`

```python
from __future__ import annotations

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
    ActionType.FILE,  # replace with the relevant ActionType(s)
}

@auto_register
class MyAgent(BaseAgent):
    """Specialist agent for handling my specific domain."""

    def __init__(self, model_router: ModelRouter) -> None:
        super().__init__(role=AgentRole.GENERAL, model_router=model_router)

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.FILE,
                description="Reads and processes files in my domain",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are a specialist agent for handling my specific domain. "
            "Execute tasks carefully and return structured results."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in MY_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute actions that fall within this agent's domain."""
        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]

        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        self.status = AgentStatus.BUSY
        results = []

        for action in my_actions:
            # Your action logic here
            result = ActionResult(
                action=action,
                success=True,
                output=f"Handled: {action}",
            )
            results.append(result)

        self.status = AgentStatus.IDLE
        return results

    def get_permission_tier(self) -> int:
        return 1  # Safe, read-only

    def get_resource_needs(self) -> set[str]:
        return {"file_system"}
```

> **Note:** The `@auto_register` decorator registers your agent with `AgentRegistry` automatically when the module is imported — no manual wiring needed!

---

## 🔍 Step 3 — How the Registry Works

`AgentRegistry` is a **singleton** — one shared instance across the entire system.

```python
from pilot.agents.registry import AgentRegistry

# Get the singleton instance
registry = AgentRegistry.get_instance()

# Auto-discover all BaseAgent subclasses in the pilot.agents package
AgentRegistry.discover_agents()

# Get all registered agents (returns dict[str, type[BaseAgent]])
all_agents = AgentRegistry.get_all_agents()
for name, agent_class in all_agents.items():
    print(f"Agent: {name} → {agent_class}")

# Get a specific agent class by name
agent_class = AgentRegistry.get_agent_class("my_agent")

# Create an agent instance
agent = AgentRegistry.create_agent("my_agent", model_router=model_router)
```

---

## ✅ Step 4 — Verify Your Agent is Registered

```python
from pilot.agents.registry import AgentRegistry

AgentRegistry.discover_agents()
agents = AgentRegistry.get_all_agents()

print(f"Total agents found: {len(agents)}")
for name in agents:
    print(f"  ✅ {name}")
```

---

## 🛡️ Permission Tiers Reference

| Tier | Level | Examples |
|------|-------|---------|
| **Tier 1** | Safe / Read-only | Reading files, fetching web content |
| **Tier 2** | Requires confirmation | Writing files, sending messages |
| **Tier 3** | Destructive | Deleting files, system changes |
| **Tier 4** | Critical | Recursive deletes, system-level ops |
| **Tier 5** | Root-level | Requires explicit root permission |

> Always use the **lowest tier** that your agent actually needs.

---

## 🎭 AgentRole Reference

| Role | Value |
|------|-------|
| `AgentRole.SYSTEM` | `"system_agent"` |
| `AgentRole.WEB` | `"web_agent"` |
| `AgentRole.MONITOR` | `"monitor_agent"` |
| `AgentRole.COMMUNICATION` | `"comm_agent"` |
| `AgentRole.ORCHESTRATOR` | `"orchestrator"` |
| `AgentRole.GENERAL` | `"general"` |

---

## 💡 Tips for Contributors

- **Implement all 3 abstract methods** — `get_capabilities()`, `get_system_prompt()`, and `handle_task()` are required
- **Override `can_handle()`** — use a set of `ActionType` values for clean routing
- **Use `@auto_register`** — cleanest way to register your agent
- **Pass the right `AgentRole`** to `super().__init__()` — use `GENERAL` if none fit
- **Test before submitting** — run `python -m py_compile daemon/pilot/agents/my_agent.py`
- **Follow naming conventions** — look at `code_agent.py`, `comm_agent.py` for reference
- **To reset the registry in tests** — use `AgentRegistry.clear()`

---

## 📚 Related Files

- `daemon/pilot/agents/base_agent.py` — Base class, enums, and message types
- `daemon/pilot/agents/registry.py` — Dynamic registry implementation
- `daemon/pilot/agents/orchestrator.py` — Routes actions to agents
- `daemon/pilot/agents/code_agent.py` — Real example of an agent implementation

---

*This guide was created as part of GSSoC 2026 contributions to Helix OS.*
