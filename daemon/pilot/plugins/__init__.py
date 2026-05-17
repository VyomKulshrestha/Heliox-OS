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
      plugin.ed25519.pub
      plugin.ed25519.sig
      docker_plugin.py
    spotify-agent/
      manifest.json
      plugin.ed25519.pub
      plugin.ed25519.sig
      spotify_plugin.py

Plugin signatures:
  Each plugin directory must include an Ed25519 signature file. Production
  registries should provide bundled trusted public keys to PluginRegistry; local
  development may also include plugin.ed25519.pub beside the signature. The
  signature is verified before manifest parsing and covers a deterministic
  digest of every plugin file except signature metadata, so changes to either
  manifest.json or plugin code are rejected before loading.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger("pilot.plugins")

PLUGIN_PUBLIC_KEY_FILE = "plugin.ed25519.pub"
PLUGIN_SIGNATURE_FILE = "plugin.ed25519.sig"
PLUGIN_SIGNATURE_EXCLUDES = frozenset(
    {
        PLUGIN_PUBLIC_KEY_FILE,
        PLUGIN_SIGNATURE_FILE,
    }
)


class PluginSignatureError(ValueError):
    """Raised when a plugin signature is missing, malformed, or invalid."""


def _load_public_key(raw: bytes, *, source: str) -> Ed25519PublicKey:
    """Load an Ed25519 public key from PEM or base64-encoded raw bytes."""

    stripped = raw.strip()
    if stripped.startswith(b"-----BEGIN"):
        key = serialization.load_pem_public_key(stripped)
        if not isinstance(key, Ed25519PublicKey):
            raise PluginSignatureError(f"Public key is not Ed25519: {source}")
        return key

    try:
        key_bytes = base64.b64decode(stripped, validate=True)
    except ValueError as exc:
        raise PluginSignatureError(f"Public key is not valid base64: {source}") from exc

    try:
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    except ValueError as exc:
        raise PluginSignatureError(f"Public key is not valid Ed25519: {source}") from exc


def _read_public_key(path: Path) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a file."""

    return _load_public_key(path.read_bytes(), source=str(path))


def _coerce_public_key(source: Ed25519PublicKey | bytes | str | Path) -> Ed25519PublicKey:
    """Normalize supported trusted-key inputs into Ed25519 public keys."""

    if isinstance(source, Ed25519PublicKey):
        return source
    if isinstance(source, bytes):
        return _load_public_key(source, source="trusted public key bytes")
    return _read_public_key(Path(source))


def _read_signature(path: Path) -> bytes:
    """Load a base64-encoded Ed25519 signature."""

    try:
        return base64.b64decode(path.read_bytes().strip(), validate=True)
    except ValueError as exc:
        raise PluginSignatureError(f"Signature is not valid base64: {path}") from exc


def _plugin_digest(plugin_dir: Path) -> bytes:
    """Build a deterministic digest over plugin package contents."""

    digest = hashlib.sha256()
    for file_path in sorted(path for path in plugin_dir.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(plugin_dir).as_posix()
        if relative_path in PLUGIN_SIGNATURE_EXCLUDES:
            continue
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.digest()


def verify_plugin_signature(plugin_dir: Path, trusted_public_keys: tuple[Ed25519PublicKey, ...] = ()) -> None:
    """Verify a plugin directory's Ed25519 signature before loading code."""

    public_key_path = plugin_dir / PLUGIN_PUBLIC_KEY_FILE
    signature_path = plugin_dir / PLUGIN_SIGNATURE_FILE
    if not signature_path.exists():
        raise PluginSignatureError(f"Plugin is missing Ed25519 signature: {signature_path}")

    if trusted_public_keys:
        public_keys = trusted_public_keys
    elif public_key_path.exists():
        public_keys = (_read_public_key(public_key_path),)
    else:
        raise PluginSignatureError(
            f"Plugin is missing Ed25519 public key and no trusted key was configured: {public_key_path}"
        )

    signature = _read_signature(signature_path)
    payload = _plugin_digest(plugin_dir)
    for public_key in public_keys:
        try:
            public_key.verify(signature, payload)
            return
        except InvalidSignature:
            continue
    raise PluginSignatureError(f"Plugin signature verification failed: {plugin_dir}")


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

    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        *,
        require_signatures: bool = True,
        trusted_public_keys: list[Ed25519PublicKey | bytes | str | Path] | None = None,
    ) -> None:
        self._plugin_dirs: list[Path] = plugin_dirs or []
        self._plugins: dict[str, PluginManifest] = {}
        self._tool_index: dict[str, PluginManifest] = {}
        self._require_signatures = require_signatures
        self._trusted_public_keys = tuple(_coerce_public_key(key) for key in (trusted_public_keys or []))

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
                        if self._require_signatures and not self._verify_plugin(child):
                            continue
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

    def _verify_plugin(self, plugin_dir: Path) -> bool:
        """Return True when a plugin directory passes signature verification."""

        try:
            verify_plugin_signature(plugin_dir, self._trusted_public_keys)
            return True
        except PluginSignatureError as exc:
            logger.error("Rejected unsigned or tampered plugin %s: %s", plugin_dir, exc)
            return False

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
