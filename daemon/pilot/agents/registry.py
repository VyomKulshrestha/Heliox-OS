"""Agent Registry - auto-discovery and dynamic registration of agents."""

import importlib
import logging
import pkgutil
from typing import Any

from pilot.agents.base_agent import BaseAgent

logger = logging.getLogger("pilot.agents.registry")


class AgentRegistry:
    """Central registry for agent auto-discovery and dynamic registration."""

    _instance = None
    _agents: dict[str, type[BaseAgent]] = {}
    _initialized = False

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls, agent_class: type[BaseAgent], name: str | None = None) -> None:
        """Register an agent class with the registry."""
        if name is None:
            name = agent_class.__name__
        cls._agents[name] = agent_class
        logger.info("Registered agent: %s", name)

    @classmethod
    def discover_agents(cls, package_name: str = "pilot.agents") -> None:
        """Auto-discover agent classes in the given package."""
        if cls._initialized:
            return

        try:
            package = importlib.import_module(package_name)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseAgent)
                            and attr is not BaseAgent
                        ):
                            cls.register(attr)
                except Exception as e:
                    logger.debug("Could not import %s: %s", module_name, e)
        except Exception as e:
            logger.warning("Agent discovery failed: %s", e)

        cls._initialized = True
        logger.info("Agent discovery complete: %d agents found", len(cls._agents))

    @classmethod
    def get_agent_class(cls, name: str) -> type[BaseAgent] | None:
        """Get an agent class by name."""
        return cls._agents.get(name)

    @classmethod
    def get_all_agents(cls) -> dict[str, type[BaseAgent]]:
        """Get all registered agent classes."""
        return cls._agents.copy()

    @classmethod
    def create_agent(cls, name: str, **kwargs: Any) -> BaseAgent | None:
        """Create an agent instance by name."""
        agent_class = cls.get_agent_class(name)
        if agent_class:
            return agent_class(**kwargs)
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (for testing)."""
        cls._agents.clear()
        cls._initialized = False


def auto_register(agent_class: type[BaseAgent]) -> type[BaseAgent]:
    """Decorator to auto-register an agent class."""
    AgentRegistry.register(agent_class)
    return agent_class
