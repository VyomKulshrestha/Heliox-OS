"""Plugin System — extend Pilot with custom action modules.

Simple Python functions become action types. Auto-discovery of plugins
from a directory, with hot-reload and metadata support.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.plugins")


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""

    name: str
    description: str
    version: str
    author: str
    actions: list[str]  # List of action function names
    file_path: str
    loaded: bool = True
    error: str | None = None


@dataclass
class PluginAction:
    """A single action provided by a plugin."""

    name: str
    plugin_name: str
    description: str
    function: Callable[..., Coroutine | str]
    parameters: dict[str, str]  # param_name -> type hint


class PluginManager:
    """Manages plugin discovery, loading, and execution."""

    DEFAULT_PLUGIN_DIR = os.path.expanduser("~/.config/heliox-os/plugins")

    def __init__(self, plugin_dirs: list[str] | None = None):
        self._plugin_dirs = plugin_dirs or [self.DEFAULT_PLUGIN_DIR]
        self._plugins: dict[str, PluginInfo] = {}
        self._actions: dict[str, PluginAction] = {}

    def discover_plugins(self) -> list[str]:
        """Scan plugin directories for .py plugin files."""
        found = []
        for plugin_dir in self._plugin_dirs:
            p = Path(plugin_dir)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                # Create example plugin
                self._create_example_plugin(p)
                continue

            for py_file in p.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                found.append(str(py_file))

        return found

    def load_plugin(self, file_path: str) -> PluginInfo:
        """Load a single plugin from a Python file.

        A plugin file should have:
        - PLUGIN_NAME (str): Name of the plugin
        - PLUGIN_DESCRIPTION (str): What it does
        - PLUGIN_VERSION (str): Version number
        - PLUGIN_AUTHOR (str): Who made it
        - Functions decorated with @pilot_action or prefixed with 'action_'
        """
        file_path = str(Path(file_path).resolve())
        module_name = f"pilot_plugin_{Path(file_path).stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return PluginInfo(
                    name=Path(file_path).stem,
                    description="",
                    version="0.0.0",
                    author="",
                    actions=[],
                    file_path=file_path,
                    loaded=False,
                    error="Invalid Python file",
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Extract metadata
            name = getattr(module, "PLUGIN_NAME", Path(file_path).stem)
            desc = getattr(module, "PLUGIN_DESCRIPTION", "")
            version = getattr(module, "PLUGIN_VERSION", "1.0.0")
            author = getattr(module, "PLUGIN_AUTHOR", "unknown")

            # Find action functions
            action_names = []
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if not callable(obj):
                    continue

                # Check for @pilot_action decorator or action_ prefix
                is_action = getattr(obj, "_is_pilot_action", False) or attr_name.startswith("action_")

                if is_action:
                    action_id = f"plugin_{name}_{attr_name}"
                    # Get parameter info from signature
                    sig = inspect.signature(obj)
                    params = {}
                    for param_name, param in sig.parameters.items():
                        if param_name == "self":
                            continue
                        type_hint = param.annotation
                        type_str = type_hint.__name__ if hasattr(type_hint, "__name__") else str(type_hint)
                        params[param_name] = type_str

                    self._actions[action_id] = PluginAction(
                        name=action_id,
                        plugin_name=name,
                        description=obj.__doc__ or "",
                        function=obj,
                        parameters=params,
                    )
                    action_names.append(attr_name)

            info = PluginInfo(
                name=name,
                description=desc,
                version=version,
                author=author,
                actions=action_names,
                file_path=file_path,
            )
            self._plugins[name] = info
            logger.info("Loaded plugin: %s (%d actions)", name, len(action_names))
            return info

        except Exception as e:
            info = PluginInfo(
                name=Path(file_path).stem,
                description="",
                version="0.0.0",
                author="",
                actions=[],
                file_path=file_path,
                loaded=False,
                error=str(e),
            )
            self._plugins[info.name] = info
            logger.error("Failed to load plugin %s: %s", file_path, e)
            return info

    def load_all(self) -> list[PluginInfo]:
        """Discover and load all plugins."""
        files = self.discover_plugins()
        results = []
        for f in files:
            info = self.load_plugin(f)
            results.append(info)
        return results

    def reload_plugin(self, name: str) -> PluginInfo | None:
        """Reload a specific plugin (hot-reload)."""
        info = self._plugins.get(name)
        if not info:
            return None

        # Remove old actions
        for action_name in list(self._actions.keys()):
            if self._actions[action_name].plugin_name == name:
                del self._actions[action_name]

        # Remove from sys.modules
        module_name = f"pilot_plugin_{Path(info.file_path).stem}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        return self.load_plugin(info.file_path)

    async def execute_plugin_action(
        self,
        action_id: str,
        params: dict[str, Any],
    ) -> str:
        """Execute a plugin action by ID."""
        action = self._actions.get(action_id)
        if not action:
            available = list(self._actions.keys())
            return f"Plugin action '{action_id}' not found. Available: {available}"

        try:
            result = action.function(**params)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"Plugin action '{action_id}' failed: {e}"

    def list_plugins(self) -> list[dict]:
        """List all loaded plugins."""
        return [asdict(p) for p in self._plugins.values()]

    def list_actions(self) -> list[dict]:
        """List all available plugin actions."""
        return [
            {
                "name": a.name,
                "plugin": a.plugin_name,
                "description": a.description[:200],
                "parameters": a.parameters,
            }
            for a in self._actions.values()
        ]

    def _create_example_plugin(self, plugin_dir: Path):
        """Create an example plugin for reference."""
        example = plugin_dir / "example_hello.py"
        example.write_text('''"""Example Pilot Plugin — Hello World.

This shows how to create a custom plugin for Pilot.
Place .py files in ~/.pilot/plugins/ and they'll be auto-discovered.

Functions prefixed with 'action_' automatically become available actions.
"""

PLUGIN_NAME = "hello_world"
PLUGIN_DESCRIPTION = "A simple example plugin"
PLUGIN_VERSION = "1.0.0"
PLUGIN_AUTHOR = "Pilot User"


def action_greet(name: str = "World") -> str:
    """Greet someone by name."""
    return f"Hello, {name}! This is a Pilot plugin action."


def action_add(a: int = 0, b: int = 0) -> str:
    """Add two numbers together."""
    return f"{a} + {b} = {a + b}"


async def action_count_files(directory: str = ".") -> str:
    """Count files in a directory."""
    import os
    count = sum(1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
    return f"Found {count} files in {directory}"
''')
        logger.info("Created example plugin at %s", example)


# ── Decorator for plugin actions ─────────────────────────────────────


def pilot_action(func):
    """Decorator to mark a function as a Pilot plugin action."""
    func._is_pilot_action = True
    return func


# ── Global plugin manager ────────────────────────────────────────────

_manager = PluginManager()


async def plugin_list() -> str:
    """List all loaded plugins."""
    plugins = _manager.list_plugins()
    if not plugins:
        # Try to load
        _manager.load_all()
        plugins = _manager.list_plugins()
    if not plugins:
        return "No plugins found. Place .py files in ~/.pilot/plugins/"
    return json.dumps(plugins, indent=2)


async def plugin_list_actions() -> str:
    """List all available plugin actions."""
    actions = _manager.list_actions()
    if not actions:
        _manager.load_all()
        actions = _manager.list_actions()
    if not actions:
        return "No plugin actions available"
    return json.dumps(actions, indent=2)


async def plugin_execute(action_id: str, params: dict | None = None) -> str:
    """Execute a plugin action."""
    return await _manager.execute_plugin_action(action_id, params or {})


async def plugin_reload(name: str) -> str:
    """Reload a plugin."""
    info = _manager.reload_plugin(name)
    if info:
        return json.dumps(asdict(info), indent=2)
    return f"Plugin '{name}' not found"


async def plugin_install(source: str) -> str:
    """Install a plugin from a URL or local path."""
    import shutil

    src = Path(source)
    if src.exists():
        dest = Path(_manager.DEFAULT_PLUGIN_DIR) / src.name
        shutil.copy2(src, dest)
        info = _manager.load_plugin(str(dest))
        return f"Installed plugin from {source}: {json.dumps(asdict(info), indent=2)}"
    return f"Plugin source not found: {source}"


def get_manager() -> PluginManager:
    """Get the global plugin manager."""
    return _manager
