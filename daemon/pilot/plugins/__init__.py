"""Plugin Ecosystem — dynamic tool/agent extension system.

Allows developers to add new capabilities to Heliox OS by dropping
plugin manifests into a plugins directory.

Plugin Manifest (JSON):
{
  "name": "docker-agent",
  "version": "1.0.0",
  "description": "Docker container management",
  "author": "community",
  "tools": [
    {
      "name": "docker_build",
      "description": "Build a Docker image",
      "inputs": ["dockerfile_path", "tag"],
      "outputs": ["image_id"],
      "permission_tier": 2,
      "action_type": "shell_command"
    }
  ],
  "agent_type": "system",
  "entry_point": "docker_plugin.py",
  "dependencies": ["docker"]
}

Plugin directory structure:
  ~/.heliox/plugins/
    docker-agent/
      manifest.json
      docker_plugin.py
    spotify-agent/
      manifest.json
      spotify_plugin.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.plugins")


@dataclass
class PluginTool:
    """A single tool exposed by a plugin."""

    name: str = ""
    description: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    permission_tier: int = 1
    action_type: str = "shell_command"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "permission_tier": self.permission_tier,
            "action_type": self.action_type,
        }


@dataclass
class PluginManifest:
    """Parsed plugin manifest describing capabilities."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tools: list[PluginTool] = field(default_factory=list)
    agent_type: str = "system"
    entry_point: str = ""
    dependencies: list[str] = field(default_factory=list)
    enabled: bool = True
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tools": [t.to_dict() for t in self.tools],
            "agent_type": self.agent_type,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "tool_count": len(self.tools),
        }


class PluginRegistry:
    """Discovers, loads, and manages Heliox OS plugins.

    Plugins are discovered from:
      1. ~/.heliox/plugins/     (user plugins)
      2. <data_dir>/plugins/    (system plugins)
    """

    def __init__(self, plugin_dirs: list[Path] | None = None) -> None:
        self._plugin_dirs: list[Path] = plugin_dirs or []
        self._plugins: dict[str, PluginManifest] = {}
        self._tool_index: dict[str, PluginManifest] = {}

        # Add default plugin directories
        home_plugins = Path.home() / ".heliox" / "plugins"
        if home_plugins not in self._plugin_dirs:
            self._plugin_dirs.insert(0, home_plugins)

    def discover(self) -> int:
        """Scan plugin directories and load manifests. Returns count of loaded plugins."""
        loaded: int = 0

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue

            for child in plugin_dir.iterdir():
                if child.is_dir():
                    manifest_path = child / "manifest.json"
                    if manifest_path.exists():
                        manifest = self._load_manifest(manifest_path)
                        if manifest:
                            self._plugins[manifest.name] = manifest
                            # Index tools
                            for tool in manifest.tools:
                                self._tool_index[tool.name] = manifest
                            loaded += 1
                            logger.info(
                                "Loaded plugin: %s v%s (%d tools)",
                                manifest.name,
                                manifest.version,
                                len(manifest.tools),
                            )

        logger.info("Plugin discovery complete: %d plugins loaded", loaded)
        return loaded

    def _load_manifest(self, path: Path) -> PluginManifest | None:
        """Parse a plugin manifest.json file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            tools = []
            for tool_data in data.get("tools", []):
                tools.append(
                    PluginTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        inputs=tool_data.get("inputs", []),
                        outputs=tool_data.get("outputs", []),
                        permission_tier=tool_data.get("permission_tier", 1),
                        action_type=tool_data.get("action_type", "shell_command"),
                    )
                )

            return PluginManifest(
                name=data.get("name", path.parent.name),
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                author=data.get("author", ""),
                tools=tools,
                agent_type=data.get("agent_type", "system"),
                entry_point=data.get("entry_point", ""),
                dependencies=data.get("dependencies", []),
                enabled=data.get("enabled", True),
                path=str(path.parent),
            )
        except Exception:
            logger.error("Failed to load plugin manifest: %s", path, exc_info=True)
            return None

    # ── Query APIs ──

    def get_plugin(self, name: str) -> PluginManifest | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_all_plugins(self) -> list[PluginManifest]:
        """Return all loaded plugins."""
        return list(self._plugins.values())

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all tools from all loaded plugins."""
        tools = []
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            for tool in plugin.tools:
                tool_info = tool.to_dict()
                tool_info["plugin"] = plugin.name
                tools.append(tool_info)
        return tools

    def find_tool(self, tool_name: str) -> tuple[PluginManifest, PluginTool] | None:
        """Find a tool by name across all plugins."""
        plugin = self._tool_index.get(tool_name)
        if plugin and plugin.enabled:
            for tool in plugin.tools:
                if tool.name == tool_name:
                    return (plugin, tool)
        return None

    def get_tools_for_planner(self) -> str:
        """Generate a tool listing that can be injected into planner prompts."""
        tools = self.get_all_tools()
        if not tools:
            return ""

        lines = ["Available plugin tools:"]
        for tool in tools:
            inputs_str = ", ".join(tool["inputs"]) if tool["inputs"] else "none"
            lines.append(f"  - {tool['name']}: {tool['description']} (inputs: {inputs_str}, plugin: {tool['plugin']})")

        return "\n".join(lines)

    # ── Management ──

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin by name."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = True
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin by name."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = False
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Return plugin ecosystem statistics."""
        enabled = sum(1 for p in self._plugins.values() if p.enabled)
        total_tools = sum(len(p.tools) for p in self._plugins.values() if p.enabled)

        return {
            "total_plugins": len(self._plugins),
            "enabled_plugins": enabled,
            "total_tools": total_tools,
            "plugin_dirs": [str(d) for d in self._plugin_dirs],
            "plugins": [p.to_dict() for p in self._plugins.values()],
        }
