"""Tests for the WASM plugin runtime and PluginRegistry WASM integration."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pilot.plugins import PluginManifest, PluginRegistry
from pilot.plugins.wasm_runtime import WasmConfig, WasmPlugin, WasmRuntimeError

# ---------------------------------------------------------------------------
# WasmConfig tests
# ---------------------------------------------------------------------------


def test_wasm_config_defaults():
    cfg = WasmConfig()
    assert cfg.memory_mb == 128
    assert cfg.timeout_secs == 30
    assert cfg.allow_network is False
    assert cfg.allow_filesystem is False
    assert cfg.env_vars == {}


def test_wasm_config_custom():
    cfg = WasmConfig(memory_mb=64, timeout_secs=10, allow_network=True)
    assert cfg.memory_mb == 64
    assert cfg.timeout_secs == 10
    assert cfg.allow_network is True


# ---------------------------------------------------------------------------
# WasmPlugin load-time error paths
# ---------------------------------------------------------------------------


def test_wasm_plugin_load_missing_wasmtime():
    """WasmPlugin raises ImportError when wasmtime is not installed."""
    with patch("pilot.plugins.wasm_runtime._WASMTIME_AVAILABLE", False), pytest.raises(ImportError, match="wasmtime"):
        WasmPlugin(Path("nonexistent.wasm"))


def test_wasm_plugin_load_invalid_path():
    """WasmPlugin raises WasmRuntimeError for a non-existent .wasm file."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    with pytest.raises(WasmRuntimeError, match="not found"):
        WasmPlugin(Path("/tmp/does_not_exist_xyz.wasm"))


def test_wasm_plugin_load_invalid_bytes():
    """WasmPlugin raises WasmRuntimeError when file contains invalid WASM bytes."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as f:
        f.write(b"this is not valid wasm bytecode at all")
        bad_path = Path(f.name)

    try:
        with pytest.raises(WasmRuntimeError, match="compile"):
            WasmPlugin(bad_path)
    finally:
        bad_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Minimal WAT module for functional call_tool tests
# ---------------------------------------------------------------------------

# This WAT module implements the required ABI:
#   alloc(size) -> ptr    (bump allocator from a static offset)
#   dealloc(ptr, size)    (no-op; we rely on GC)
#   call_tool(ptr, len) -> result_ptr
#       Always returns {"result": "ok"} regardless of input.
#
# Memory layout used by alloc: keeps a running offset at address 0 (4 bytes).
_MINIMAL_WAT = r"""
(module
  (memory (export "memory") 2)

  ;; bump allocator — next free byte offset stored at address 0
  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (i32.load (i32.const 0)))
    ;; if first call, start after the header word
    (if (i32.eqz (local.get $ptr))
      (then (local.set $ptr (i32.const 16)))
    )
    (i32.store (i32.const 0) (i32.add (local.get $ptr) (local.get $size)))
    (local.get $ptr)
  )

  ;; dealloc is a no-op for this minimal module
  (func (export "dealloc") (param $ptr i32) (param $size i32))

  ;; call_tool — writes length-prefixed {"result":"ok"} to a static location
  ;; and returns a pointer to it.
  (func (export "call_tool") (param $ptr i32) (param $len i32) (result i32)
    ;; Response bytes: 4-byte LE length + JSON
    ;; {"result":"ok"} = 15 bytes  =>  length prefix = 15 (0x0F 0x00 0x00 0x00)
    (local $out i32)
    (local.set $out (i32.const 4096))

    ;; write length prefix: 15 as little-endian i32
    (i32.store8 (local.get $out)                        (i32.const 15))
    (i32.store8 (i32.add (local.get $out) (i32.const 1)) (i32.const 0))
    (i32.store8 (i32.add (local.get $out) (i32.const 2)) (i32.const 0))
    (i32.store8 (i32.add (local.get $out) (i32.const 3)) (i32.const 0))

    ;; write {"result":"ok"} starting at offset 4
    ;; { = 123
    (i32.store8 (i32.add (local.get $out) (i32.const 4))  (i32.const 123))
    ;; " = 34
    (i32.store8 (i32.add (local.get $out) (i32.const 5))  (i32.const 34))
    ;; r = 114
    (i32.store8 (i32.add (local.get $out) (i32.const 6))  (i32.const 114))
    ;; e = 101
    (i32.store8 (i32.add (local.get $out) (i32.const 7))  (i32.const 101))
    ;; s = 115
    (i32.store8 (i32.add (local.get $out) (i32.const 8))  (i32.const 115))
    ;; u = 117
    (i32.store8 (i32.add (local.get $out) (i32.const 9))  (i32.const 117))
    ;; l = 108
    (i32.store8 (i32.add (local.get $out) (i32.const 10)) (i32.const 108))
    ;; t = 116
    (i32.store8 (i32.add (local.get $out) (i32.const 11)) (i32.const 116))
    ;; " = 34
    (i32.store8 (i32.add (local.get $out) (i32.const 12)) (i32.const 34))
    ;; : = 58
    (i32.store8 (i32.add (local.get $out) (i32.const 13)) (i32.const 58))
    ;; " = 34
    (i32.store8 (i32.add (local.get $out) (i32.const 14)) (i32.const 34))
    ;; o = 111
    (i32.store8 (i32.add (local.get $out) (i32.const 15)) (i32.const 111))
    ;; k = 107
    (i32.store8 (i32.add (local.get $out) (i32.const 16)) (i32.const 107))
    ;; " = 34
    (i32.store8 (i32.add (local.get $out) (i32.const 17)) (i32.const 34))
    ;; } = 125
    (i32.store8 (i32.add (local.get $out) (i32.const 18)) (i32.const 125))

    (local.get $out)
  )
)
"""


@pytest.fixture()
def minimal_wasm_file(tmp_path: Path) -> Path:
    """Write a minimal WAT module to a temp file and return its path.

    wasmtime.Module() accepts WAT text format directly alongside binary WASM,
    so we skip a separate compile step and write the WAT bytes straight to
    the .wasm file.
    """
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    wasm_path = tmp_path / "plugin.wasm"
    wasm_path.write_bytes(_MINIMAL_WAT.strip().encode())
    return wasm_path


# ---------------------------------------------------------------------------
# WasmPlugin functional tests
# ---------------------------------------------------------------------------


def test_wasm_plugin_call_tool(minimal_wasm_file: Path):
    """A compiled WASM module can be loaded and its call_tool export invoked."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    with WasmPlugin(minimal_wasm_file) as plugin:
        result = plugin.call_tool("any_tool", {"key": "value"})

    assert isinstance(result, dict)
    assert result.get("result") == "ok"


def test_wasm_plugin_context_manager(minimal_wasm_file: Path):
    """WasmPlugin works as a context manager without raising."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    with WasmPlugin(minimal_wasm_file) as plugin:
        assert plugin is not None


# ---------------------------------------------------------------------------
# PluginManifest WASM field tests
# ---------------------------------------------------------------------------


def test_manifest_runtime_type_defaults_to_python():
    manifest = PluginManifest()
    assert manifest.runtime_type == "python"
    assert manifest.wasm_module == ""


def test_manifest_to_dict_includes_wasm_fields():
    manifest = PluginManifest(runtime_type="wasm", wasm_module="plugin.wasm")
    d = manifest.to_dict()
    assert d["runtime_type"] == "wasm"
    assert d["wasm_module"] == "plugin.wasm"


def test_load_manifest_wasm_fields(tmp_path: Path):
    """PluginRegistry._load_manifest parses runtime_type and wasm_module."""
    manifest_data = {
        "name": "test-wasm-plugin",
        "version": "1.0.0",
        "description": "A test WASM plugin",
        "author": "test",
        "runtime_type": "wasm",
        "wasm_module": "plugin.wasm",
        "tools": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data))

    registry = PluginRegistry(plugin_dirs=[], require_signatures=False)
    manifest = registry._load_manifest(manifest_path)

    assert manifest is not None
    assert manifest.runtime_type == "wasm"
    assert manifest.wasm_module == "plugin.wasm"


def test_load_manifest_defaults_runtime_type(tmp_path: Path):
    """Manifests without runtime_type default to 'python'."""
    manifest_data = {"name": "py-plugin", "tools": []}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data))

    registry = PluginRegistry(plugin_dirs=[], require_signatures=False)
    manifest = registry._load_manifest(manifest_path)

    assert manifest is not None
    assert manifest.runtime_type == "python"


# ---------------------------------------------------------------------------
# PluginRegistry WASM integration tests
# ---------------------------------------------------------------------------


def test_registry_discover_wasm_plugin(tmp_path: Path, minimal_wasm_file: Path):
    """PluginRegistry.discover() loads a WASM plugin and stores it."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    plugin_dir = tmp_path / "my-wasm-plugin"
    plugin_dir.mkdir()

    wasm_dest = plugin_dir / "plugin.wasm"
    wasm_dest.write_bytes(minimal_wasm_file.read_bytes())

    manifest_data = {
        "name": "my-wasm-plugin",
        "version": "1.0.0",
        "description": "WASM test plugin",
        "author": "test",
        "runtime_type": "wasm",
        "wasm_module": "plugin.wasm",
        "tools": [
            {
                "name": "wasm_greet",
                "description": "Greet via WASM",
                "inputs": [],
                "outputs": ["result"],
                "permission_tier": 1,
                "action_type": "wasm_call",
            }
        ],
    }
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest_data))

    registry = PluginRegistry(plugin_dirs=[tmp_path], require_signatures=False)
    count = registry.discover()

    assert count == 1
    assert "my-wasm-plugin" in registry._wasm_plugins


def test_registry_discover_wasm_plugin_mocked(tmp_path: Path):
    """PluginRegistry.discover() calls _load_wasm_plugin for wasm manifests."""
    plugin_dir = tmp_path / "mock-wasm-plugin"
    plugin_dir.mkdir()

    manifest_data = {
        "name": "mock-wasm-plugin",
        "runtime_type": "wasm",
        "wasm_module": "plugin.wasm",
        "tools": [],
    }
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest_data))

    registry = PluginRegistry(plugin_dirs=[tmp_path], require_signatures=False)
    with patch.object(registry, "_load_wasm_plugin") as mock_load:
        registry.discover()
        mock_load.assert_called_once()
        called_manifest = mock_load.call_args[0][0]
        assert called_manifest.runtime_type == "wasm"


def test_call_wasm_tool_not_found():
    """call_wasm_tool returns an error dict for unknown tools."""
    registry = PluginRegistry(plugin_dirs=[], require_signatures=False)
    result = registry.call_wasm_tool("nonexistent_tool", {})
    assert "error" in result
    assert "nonexistent_tool" in result["error"]


def test_call_wasm_tool_python_plugin(tmp_path: Path):
    """call_wasm_tool returns an error dict when the tool belongs to a Python plugin."""
    plugin_dir = tmp_path / "py-plugin"
    plugin_dir.mkdir()

    manifest_data = {
        "name": "py-plugin",
        "runtime_type": "python",
        "tools": [{"name": "py_tool", "description": "", "inputs": [], "outputs": []}],
    }
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest_data))

    registry = PluginRegistry(plugin_dirs=[tmp_path], require_signatures=False)
    registry.discover()

    result = registry.call_wasm_tool("py_tool", {})
    assert "error" in result
    assert "Python plugin" in result["error"]


def test_get_stats_includes_wasm_count():
    """get_stats() includes a wasm_plugins key."""
    registry = PluginRegistry(plugin_dirs=[], require_signatures=False)
    stats = registry.get_stats()
    assert "wasm_plugins" in stats
    assert stats["wasm_plugins"] == 0


def test_wasm_plugin_timeout(tmp_path: Path):
    """Verify that a loop-heavy WAT module times out and raises WasmRuntimeError."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        pytest.skip("wasmtime not installed")

    looping_wat = """
    (module
      (memory (export "memory") 2)
      (func (export "alloc") (param $size i32) (result i32)
        (i32.const 16)
      )
      (func (export "dealloc") (param $ptr i32) (param $size i32))
      (func (export "call_tool") (param $ptr i32) (param $len i32) (result i32)
        (loop $infinite
          (br $infinite)
        )
        (i32.const 0)
      )
    )
    """
    wasm_path = tmp_path / "timeout_plugin.wasm"
    wasm_path.write_bytes(looping_wat.strip().encode())

    # Configure a short timeout (e.g. 0.1 seconds)
    config = WasmConfig(timeout_secs=0.1)
    with WasmPlugin(wasm_path, config) as plugin, pytest.raises(WasmRuntimeError, match="trapped or failed"):
        plugin.call_tool("any_tool", {})


async def test_wasm_executor_integration(default_config, tmp_path: Path):
    """Verify that ActionType.WASM_CALL is correctly routed by Executor to PluginRegistry."""
    from pilot.actions import Action, ActionPlan, ActionType, WasmCallParams
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    # Set up executor
    validator = ActionValidator(default_config)
    permissions = PermissionChecker(default_config)
    audit = AuditLogger(audit_file=tmp_path / "audit.log")
    executor = Executor(default_config, validator, permissions, audit)

    # Mock plugin registry
    mock_registry = MagicMock()
    mock_registry.call_tool.return_value = {"status": "success", "result": 123}
    executor.set_plugin_registry(mock_registry)

    # 1. Success case
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.WASM_CALL,
                target="my_wasm_tool",
                parameters=WasmCallParams(tool="my_wasm_tool", args={"x": 42}),
            )
        ]
    )

    results = await executor.execute(plan)
    assert len(results) == 1
    assert results[0].success is True
    assert json.loads(results[0].output) == {"status": "success", "result": 123}
    mock_registry.call_tool.assert_called_once_with("my_wasm_tool", {"x": 42})

    # 2. Failure case (registry returns error)
    mock_registry.reset_mock()
    mock_registry.call_tool.return_value = {"error": "Some WASM trap/panic"}

    results = await executor.execute(plan)
    assert len(results) == 1
    assert results[0].success is False
    assert "Plugin tool execution failed: Some WASM trap/panic" in results[0].error
    mock_registry.call_tool.assert_called_once_with("my_wasm_tool", {"x": 42})


async def test_wasm_executor_integration_uninitialized(default_config, tmp_path: Path):
    """Verify that Executor raises RuntimeError if PluginRegistry is not set."""
    from pilot.actions import Action, ActionPlan, ActionType, WasmCallParams
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    validator = ActionValidator(default_config)
    permissions = PermissionChecker(default_config)
    audit = AuditLogger(audit_file=tmp_path / "audit.log")
    executor = Executor(default_config, validator, permissions, audit)
    # do NOT set plugin registry

    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.WASM_CALL,
                target="my_wasm_tool",
                parameters=WasmCallParams(tool="my_wasm_tool", args={"x": 42}),
            )
        ]
    )

    results = await executor.execute(plan)
    assert len(results) == 1
    assert results[0].success is False
    assert "Plugin registry not initialized in Executor" in results[0].error


def test_wasm_validator_checks(default_config):
    """Verify that ActionValidator rejects WASM call actions without a tool name."""
    from pilot.actions import Action, ActionPlan, ActionType, WasmCallParams
    from pilot.security.validator import ActionValidator

    validator = ActionValidator(default_config)

    # 1. Invalid: both empty
    plan_invalid = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.WASM_CALL,
                target="",
                parameters=WasmCallParams(tool="", args={}),
            )
        ]
    )
    errors = validator.validate_plan(plan_invalid)
    assert any("WASM call requires a tool name" in e for e in errors)

    # 2. Valid: target provided
    plan_target = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.WASM_CALL,
                target="my_tool",
                parameters=WasmCallParams(tool="", args={}),
            )
        ]
    )
    assert len(validator.validate_plan(plan_target)) == 0

    # 3. Valid: parameters.tool provided
    plan_param = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.WASM_CALL,
                target="",
                parameters=WasmCallParams(tool="my_tool", args={}),
            )
        ]
    )
    assert len(validator.validate_plan(plan_param)) == 0
