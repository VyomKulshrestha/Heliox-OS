"""Plugin Ecosystem — dynamic tool/agent extension system.

Allows developers to add new capabilities to Heliox OS by dropping
plugin manifests into a plugins directory.

Python plugin manifest (JSON):
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

WASM plugin manifest (JSON):
{
  "name": "image-resizer",
  "version": "1.0.0",
  "description": "Resize images inside a WASM sandbox",
  "author": "community",
  "runtime_type": "wasm",
  "wasm_module": "plugin.wasm",
  "tools": [
    {
      "name": "resize_image",
      "description": "Resize an image to given dimensions",
      "inputs": ["path", "width", "height"],
      "outputs": ["out_path"],
      "permission_tier": 1,
      "action_type": "wasm_call"
    }
  ]
}

Plugin directory structure:
  ~/.config/heliox-os/plugins/
    docker-agent/
      manifest.json
      plugin.ed25519.pub
      plugin.ed25519.sig
      docker_plugin.py
    image-resizer/            # WASM plugin
      manifest.json
      plugin.ed25519.pub
      plugin.ed25519.sig
      plugin.wasm

Plugin signatures:
  Each plugin directory must include an Ed25519 signature file. Production
  registries should provide bundled trusted public keys to PluginRegistry; local
  development may also include plugin.ed25519.pub beside the signature. The
  signature is verified before manifest parsing and covers a deterministic
  digest of every plugin file except signature metadata, so changes to either
  manifest.json or plugin code are rejected before loading.
  For WASM plugins the .wasm binary is automatically included in the digest.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pilot.config import PLUGINS_DIR
from pilot.plugins.capabilities import (
    PluginCapabilities,
    PluginCapabilityError,
    parse_plugin_capabilities,
    validate_credential_urls,
)

logger = logging.getLogger("pilot.plugins")
_PLUGIN_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

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


def _safe_plugin_file(raw_path: str, *, field_name: str) -> str:
    path = Path(raw_path)
    if not raw_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a plugin-relative file")
    return str(path)


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


def verify_plugin_signature(plugin_dir: Path, trusted_public_keys: tuple[Ed25519PublicKey, ...] = ()) -> bool:
    """Verify a plugin directory's Ed25519 signature before loading code.

    True when a trusted key vouched for it; False when the signature is good but made with the key beside the plugin
    (the plugin vouches for itself: plugin_create signs every plugin it writes with a new key of its own, so such a
    signature says the files are unchanged since signing, not that anyone checked their code). Raises when missing,
    malformed or wrong."""

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
            return bool(trusted_public_keys)
        except InvalidSignature:
            continue
    raise PluginSignatureError(f"Plugin signature verification failed: {plugin_dir}")


def sign_plugin_directory(plugin_dir: Path) -> None:
    """Generate Ed25519 keypair, sign all non-metadata files in plugin_dir, and save key & signature."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (plugin_dir / PLUGIN_PUBLIC_KEY_FILE).write_bytes(public_pem)

    payload = _plugin_digest(plugin_dir)
    sig = private_key.sign(payload)
    (plugin_dir / PLUGIN_SIGNATURE_FILE).write_bytes(base64.b64encode(sig) + b"\n")


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
    runtime_type: str = "python"  # "python" | "wasm"
    wasm_module: str = ""  # relative path to .wasm file, e.g. "plugin.wasm"
    capabilities: PluginCapabilities = field(default_factory=PluginCapabilities)
    # A trusted key vouched for its code (False: signed only by its own key, see verify_plugin_signature).
    verified: bool = False

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
            "verified": self.verified,
            "enabled": self.enabled,
            "runtime_type": self.runtime_type,
            "wasm_module": self.wasm_module,
            "capabilities": self.capabilities.to_dict(),
            "capability_risks": self.capabilities.risk_labels,
            "tool_count": len(self.tools),
        }


class PluginRegistry:
    """Discovers, loads, and manages Heliox OS plugins.

    Plugins are discovered from:
      1. <config_dir>/plugins/     (user plugins, from PLUGINS_DIR)
      2. <data_dir>/plugins/       (system plugins)

    Both Python and WASM plugins are supported. WASM plugins require the
    optional `wasmtime` package (pip install pilot-daemon[wasm]).
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
        self._wasm_plugins: dict[str, Any] = {}  # name -> WasmPlugin

        # Add default plugin directories
        if PLUGINS_DIR not in self._plugin_dirs:
            self._plugin_dirs.insert(0, PLUGINS_DIR)

    def discover(self) -> int:
        """Scan plugin directories and load manifests. Returns count of loaded plugins."""
        for plugin in self._wasm_plugins.values():
            with contextlib.suppress(Exception):
                plugin.close()
        self._plugins.clear()
        self._tool_index.clear()
        self._wasm_plugins.clear()

        loaded: int = 0

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue

            for child in plugin_dir.iterdir():
                if child.is_dir():
                    manifest_path = child / "manifest.json"
                    if manifest_path.exists():
                        verified = self._verify_plugin(child) if self._require_signatures else None
                        if verified is None and self._require_signatures:
                            continue
                        manifest = self._load_manifest(manifest_path)
                        if manifest:
                            manifest.verified = bool(verified)
                            collisions = sorted(tool.name for tool in manifest.tools if tool.name in self._tool_index)
                            if collisions:
                                logger.error(
                                    "Rejected plugin %s because tool names already exist: %s",
                                    manifest.name,
                                    ", ".join(collisions),
                                )
                                continue
                            self._plugins[manifest.name] = manifest
                            # Index tools
                            for tool in manifest.tools:
                                self._tool_index[tool.name] = manifest
                            if manifest.runtime_type == "wasm":
                                self._load_wasm_plugin(manifest)
                            loaded += 1
                            logger.info(
                                "Loaded plugin: %s v%s (%d tools, runtime=%s)",
                                manifest.name,
                                manifest.version,
                                len(manifest.tools),
                                manifest.runtime_type,
                            )

        logger.info("Plugin discovery complete: %d plugins loaded", loaded)
        return loaded

    def find_tool_plugin(self, tool_name: str) -> PluginManifest | None:
        """The loaded plugin that provides ``tool_name``, or None."""
        return self._tool_index.get(tool_name)

    def _verify_plugin(self, plugin_dir: Path) -> bool | None:
        """Whether a trusted key vouched for the plugin (True), only its own key (False), or the signature is missing
        or wrong (None: not loaded)."""

        try:
            return verify_plugin_signature(plugin_dir, self._trusted_public_keys)
        except PluginSignatureError as exc:
            logger.error("Rejected unsigned or tampered plugin %s: %s", plugin_dir, exc)
            return None

    def _load_manifest(self, path: Path) -> PluginManifest | None:
        """Parse a plugin manifest.json file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name", path.parent.name)
            if not isinstance(name, str) or not _PLUGIN_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Unsafe plugin name: {name!r}")
            runtime_type = data.get("runtime_type", "python")
            if runtime_type not in {"python", "wasm"}:
                raise ValueError(f"Unsupported plugin runtime: {runtime_type!r}")
            entry_point = _safe_plugin_file(
                data.get("entry_point", "plugin.py"),
                field_name="entry_point",
            )
            wasm_module = data.get("wasm_module", "")
            if wasm_module:
                wasm_module = _safe_plugin_file(
                    wasm_module,
                    field_name="wasm_module",
                )

            tools = []
            for tool_data in data.get("tools", []):
                tool_name = tool_data.get("name", "")
                if not isinstance(tool_name, str) or not _TOOL_NAME_PATTERN.fullmatch(tool_name):
                    raise ValueError(f"Unsafe plugin tool name: {tool_name!r}")
                tools.append(
                    PluginTool(
                        name=tool_name,
                        description=tool_data.get("description", ""),
                        inputs=tool_data.get("inputs", []),
                        outputs=tool_data.get("outputs", []),
                        permission_tier=tool_data.get("permission_tier", 1),
                        action_type=tool_data.get("action_type", "shell_command"),
                    )
                )

            return PluginManifest(
                name=name,
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                author=data.get("author", ""),
                tools=tools,
                agent_type=data.get("agent_type", "system"),
                entry_point=entry_point,
                dependencies=data.get("dependencies", []),
                enabled=data.get("enabled", True),
                path=str(path.parent),
                runtime_type=runtime_type,
                wasm_module=wasm_module,
                capabilities=parse_plugin_capabilities(data.get("capabilities")),
            )
        except (OSError, json.JSONDecodeError, PluginCapabilityError, TypeError, ValueError):
            logger.error("Failed to load plugin manifest: %s", path, exc_info=True)
            return None

    def _load_wasm_plugin(self, manifest: PluginManifest) -> None:
        """Instantiate and cache a WasmPlugin for the given manifest."""
        from pilot.plugins.wasm_runtime import WasmConfig, WasmPlugin, WasmRuntimeError

        if not manifest.wasm_module:
            logger.error("WASM plugin %s has no wasm_module specified in manifest", manifest.name)
            return

        wasm_path = Path(manifest.path) / manifest.wasm_module
        try:
            if manifest.capabilities.network_domains:
                raise WasmRuntimeError(
                    "WASM network access is not brokered; remove network_domains or use a reviewed native plugin"
                )
            plugin = WasmPlugin(
                wasm_path,
                WasmConfig(
                    read_paths=manifest.capabilities.filesystem.read,
                    write_paths=manifest.capabilities.filesystem.write,
                    env_vars={
                        name: os.environ[name] for name in manifest.capabilities.credentials if name in os.environ
                    },
                ),
            )
            self._wasm_plugins[manifest.name] = plugin
            logger.info("Loaded WASM plugin: %s (%s)", manifest.name, wasm_path.name)
        except (WasmRuntimeError, ImportError) as exc:
            logger.error("Failed to load WASM plugin %s: %s", manifest.name, exc)

    # ── Query APIs ──

    def call_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        approved: bool = False,
    ) -> dict[str, Any]:
        """Call a tool provided by any loaded plugin (Python or WASM) and return its JSON result."""
        result = self.find_tool(tool_name)
        if result is None:
            return {"error": f"Tool not found: {tool_name}"}
        manifest, tool = result
        if not manifest.enabled:
            return {"error": f"Plugin {manifest.name!r} is disabled"}
        if manifest.capabilities.destructive_actions and not approved:
            return {
                "error": (
                    f"Plugin {manifest.name!r} declares destructive actions and must run "
                    "through the guarded planner approval flow"
                )
            }

        if manifest.runtime_type == "wasm":
            return self.call_wasm_tool(tool_name, params)

        # Native Python executes once in a constrained child process. Plugin
        # code is never imported into the long-lived daemon.
        entry = manifest.entry_point or "plugin.py"
        script_path = Path(manifest.path) / entry
        if not script_path.exists():
            return {"error": f"Plugin script not found: {script_path}"}

        try:
            inherited_environment = os.environ
            validate_credential_urls(manifest.capabilities, inherited_environment)
            child_environment = {
                name: inherited_environment[name]
                for name in (
                    "COMSPEC",
                    "PATH",
                    "SYSTEMROOT",
                    "TEMP",
                    "TMP",
                    "WINDIR",
                )
                if name in inherited_environment
            }
            child_environment["PYTHONIOENCODING"] = "utf-8"
            child_environment.update(
                {
                    name: inherited_environment[name]
                    for name in manifest.capabilities.credentials
                    if name in inherited_environment
                }
            )
            runner = Path(__file__).with_name("native_runner.py")
            request = {
                "plugin_dir": manifest.path,
                "script_path": str(script_path),
                "tool_name": tool_name,
                "params": params,
                "capabilities": manifest.capabilities.to_dict(),
            }
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            completed = subprocess.run(
                [sys.executable, "-I", str(runner)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=manifest.path,
                env=child_environment,
                creationflags=creation_flags,
                check=False,
            )
            stdout = completed.stdout.strip()
            if not stdout:
                return {"error": (f"Plugin broker exited with code {completed.returncode} without a JSON response")}
            try:
                response = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                return {"error": "Plugin broker returned invalid JSON"}
            if not isinstance(response, dict):
                return {"result": response}
            return response
        except subprocess.TimeoutExpired:
            return {"error": "Plugin execution exceeded the 30-second broker timeout"}
        except (OSError, PluginCapabilityError, TypeError, ValueError) as exc:
            logger.error("Error executing Python plugin tool %s: %s", tool_name, exc, exc_info=True)
            return {"error": f"Plugin execution exception: {exc}"}

    def call_wasm_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a tool provided by a WASM plugin and return its JSON result.

        Returns ``{"error": "..."}`` when the tool is not found or the plugin
        is not a WASM plugin — never raises on lookup failures.
        """
        result = self.find_tool(tool_name)
        if result is None:
            return {"error": f"Tool not found: {tool_name}"}
        manifest, _ = result
        if manifest.runtime_type != "wasm":
            return {"error": f"Tool {tool_name!r} belongs to a Python plugin, not a WASM plugin"}
        wasm_plugin = self._wasm_plugins.get(manifest.name)
        if wasm_plugin is None:
            return {"error": f"WASM plugin {manifest.name!r} is not loaded (missing wasmtime?)"}
        return wasm_plugin.call_tool(tool_name, params)

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

        lines = [
            "=== INSTALLED PLUGIN TOOLS ===",
            'Use action_type plugin_call with parameters {"tool": "<tool name>", "args": {<tool arguments>}}.',
            "Plugin actions require user confirmation before execution.",
        ]
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

    def remove_plugin(self, name: str) -> bool:
        """Remove all in-memory state for a plugin after uninstall."""
        plugin = self._plugins.pop(name, None)
        if plugin is None:
            return False
        for tool_name, owner in list(self._tool_index.items()):
            if owner.name == name:
                del self._tool_index[tool_name]
        wasm_plugin = self._wasm_plugins.pop(name, None)
        if wasm_plugin is not None:
            with contextlib.suppress(Exception):
                wasm_plugin.close()
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return plugin ecosystem statistics."""
        enabled = sum(1 for p in self._plugins.values() if p.enabled)
        total_tools = sum(len(p.tools) for p in self._plugins.values() if p.enabled)

        return {
            "total_plugins": len(self._plugins),
            "enabled_plugins": enabled,
            "total_tools": total_tools,
            "wasm_plugins": len(self._wasm_plugins),
            "plugin_dirs": [str(d) for d in self._plugin_dirs],
            "plugins": [p.to_dict() for p in self._plugins.values()],
        }
