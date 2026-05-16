"""Tests for Ed25519 plugin package verification."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pilot.plugins import (
    PLUGIN_PUBLIC_KEY_FILE,
    PLUGIN_SIGNATURE_FILE,
    PluginRegistry,
    _plugin_digest,
)


def _write_plugin(plugin_dir: Path, *, name: str = "signed-plugin") -> None:
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0.0",
                "description": "Signed test plugin",
                "entry_point": "plugin.py",
                "tools": [
                    {
                        "name": "signed_tool",
                        "description": "A signed plugin tool",
                        "inputs": ["value"],
                        "outputs": ["result"],
                        "permission_tier": 2,
                        "action_type": "shell_command",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        "def run(value: str) -> str:\n    return value\n",
        encoding="utf-8",
    )


def _sign_plugin(plugin_dir: Path, private_key: Ed25519PrivateKey | None = None) -> Ed25519PrivateKey:
    private_key = private_key or Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signature = private_key.sign(_plugin_digest(plugin_dir))
    (plugin_dir / PLUGIN_PUBLIC_KEY_FILE).write_text(
        base64.b64encode(public_key).decode("ascii"),
        encoding="utf-8",
    )
    (plugin_dir / PLUGIN_SIGNATURE_FILE).write_text(
        base64.b64encode(signature).decode("ascii"),
        encoding="utf-8",
    )
    return private_key


def test_discover_loads_valid_signed_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "signed-plugin"
    _write_plugin(plugin_dir)
    _sign_plugin(plugin_dir)

    registry = PluginRegistry(plugin_dirs=[plugin_root])

    assert registry.discover() == 1
    assert registry.get_plugin("signed-plugin") is not None
    assert registry.find_tool("signed_tool") is not None


def test_discover_accepts_plugin_signed_by_trusted_bundled_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "trusted-plugin"
    private_key = Ed25519PrivateKey.generate()
    _write_plugin(plugin_dir, name="trusted-plugin")
    _sign_plugin(plugin_dir, private_key)
    trusted_key = private_key.public_key()

    registry = PluginRegistry(plugin_dirs=[plugin_root], trusted_public_keys=[trusted_key])

    assert registry.discover() == 1
    assert registry.get_plugin("trusted-plugin") is not None


def test_discover_rejects_plugin_signed_by_untrusted_replacement_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "forged-plugin"
    trusted_private_key = Ed25519PrivateKey.generate()
    attacker_private_key = Ed25519PrivateKey.generate()
    _write_plugin(plugin_dir, name="forged-plugin")
    _sign_plugin(plugin_dir, attacker_private_key)

    registry = PluginRegistry(
        plugin_dirs=[plugin_root],
        trusted_public_keys=[trusted_private_key.public_key()],
    )

    assert registry.discover() == 0
    assert registry.get_plugin("forged-plugin") is None


def test_discover_rejects_missing_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    _write_plugin(plugin_root / "unsigned-plugin", name="unsigned-plugin")

    registry = PluginRegistry(plugin_dirs=[plugin_root])

    assert registry.discover() == 0
    assert registry.get_plugin("unsigned-plugin") is None


def test_discover_rejects_tampered_plugin_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "tampered-plugin"
    _write_plugin(plugin_dir, name="tampered-plugin")
    _sign_plugin(plugin_dir)
    (plugin_dir / "plugin.py").write_text(
        "def run(value: str) -> str:\n    return 'tampered'\n",
        encoding="utf-8",
    )

    registry = PluginRegistry(plugin_dirs=[plugin_root])

    assert registry.discover() == 0
    assert registry.get_plugin("tampered-plugin") is None


def test_discover_rejects_invalid_public_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "bad-key-plugin"
    _write_plugin(plugin_dir, name="bad-key-plugin")
    _sign_plugin(plugin_dir)
    (plugin_dir / PLUGIN_PUBLIC_KEY_FILE).write_text(
        base64.b64encode(b"not-an-ed25519-key").decode("ascii"),
        encoding="utf-8",
    )

    registry = PluginRegistry(plugin_dirs=[plugin_root])

    assert registry.discover() == 0
    assert registry.get_plugin("bad-key-plugin") is None


def test_discover_can_disable_signature_requirement_for_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plugin_root = tmp_path / "plugins"
    _write_plugin(plugin_root / "local-plugin", name="local-plugin")

    registry = PluginRegistry(plugin_dirs=[plugin_root], require_signatures=False)

    assert registry.discover() == 1
    assert registry.get_plugin("local-plugin") is not None
