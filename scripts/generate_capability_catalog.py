"""Generate the public Heliox capability catalog from runtime authorities.

The catalog intentionally distinguishes declared platform targets and explicit
post-condition verification from runtime availability and executor success.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
DAEMON_ROOT = REPO_ROOT / "daemon"
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from pilot.actions import Action, ActionType, PermissionTier  # noqa: E402
from pilot.agents.agent_mesh import AgentMesh  # noqa: E402
from pilot.agents.orchestrator import AgentOrchestrator  # noqa: E402
from pilot.agents.registry import AgentRegistry  # noqa: E402
from pilot.agents.verifier import POSTCONDITION_VERIFIERS  # noqa: E402

ALL_PLATFORMS = ("windows", "macos", "linux")
WINDOWS_ONLY = {
    ActionType.REGISTRY_READ,
    ActionType.REGISTRY_WRITE,
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
}
LINUX_ONLY = {
    ActionType.GNOME_SETTING_READ,
    ActionType.GNOME_SETTING_WRITE,
    ActionType.DBUS_CALL,
}


def _sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".json", ".py"}:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _action_family(action_type: ActionType) -> str:
    value = action_type.value
    prefixes = (
        "browser",
        "calendar",
        "clipboard",
        "disk",
        "email",
        "file",
        "git",
        "keyboard",
        "mouse",
        "package",
        "power",
        "process",
        "registry",
        "schedule",
        "screen",
        "service",
        "ssh",
        "trigger",
        "user",
        "volume",
        "wifi",
        "window",
        "workspace",
    )
    for prefix in prefixes:
        if value.startswith(f"{prefix}_"):
            return prefix
    if value.startswith("api_"):
        return "integration"
    if value.startswith("gnome_") or value == "dbus_call":
        return "desktop"
    if value.startswith("code_") or value in {
        "shell_command",
        "shell_script",
        "pty_exec",
    }:
        return "code"
    if value in {"wasm_call", "plugin_call", "skill_run"}:
        return "extension"
    return "system"


def _platforms(action_type: ActionType) -> tuple[list[str], str]:
    if action_type in WINDOWS_ONLY:
        return ["windows"], "explicit_runtime_guard"
    if action_type in LINUX_ONLY:
        return ["linux"], "explicit_platform_adapter"
    return list(ALL_PLATFORMS), "declared_product_target"


async def _provider_map() -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    AgentRegistry.clear()
    AgentRegistry.discover_agents()
    mesh = AgentMesh()
    model_router = MagicMock()
    orchestrator = AgentOrchestrator(model_router, agent_mesh=mesh)
    try:
        orchestrator.auto_register_all_agents(
            executor=MagicMock(),
            background_manager=MagicMock(),
            model_router=model_router,
            config=MagicMock(),
            vault=MagicMock(),
            memory=MagicMock(),
        )
        status = mesh.status()
        providers: dict[str, list[dict[str, str]]] = defaultdict(list)
        for specialist in status["specialists"]:
            provider = {
                "id": specialist["agent_key"],
                "name": specialist["display_name"],
                "role": specialist["role"],
                "source": specialist["source"],
            }
            for action_name in specialist["capabilities"]:
                if action_name in ActionType._value2member_map_:
                    providers[action_name].append(provider)
        for entries in providers.values():
            entries.sort(key=lambda item: (item["source"], item["name"], item["id"]))
        mesh_summary = {
            "specialists": status["executable_specialists"],
            "registered_action_types": status["registered_action_types"],
            "available_action_types": status["available_action_types"],
            "coverage_complete": status["coverage_complete"],
            "uncovered_action_types": status["uncovered_action_types"],
        }
        return dict(providers), mesh_summary
    finally:
        await orchestrator.stop()


def _plugin_catalog() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    plugins: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    root = DAEMON_ROOT / "pilot" / "plugins"
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(REPO_ROOT).as_posix()
        sources.append({"path": relative, "sha256": _sha256(path)})
        plugins.append(
            {
                "name": manifest.get("name", path.parent.name),
                "version": manifest.get("version", ""),
                "description": manifest.get("description", ""),
                "author": manifest.get("author", ""),
                "catalog": "marketplace" if "marketplace_catalog" in path.parts else "builtin",
                "runtime_type": manifest.get("runtime_type", ""),
                "source_manifest": relative,
                "tools": [
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "action_type": tool.get("action_type", ""),
                        "permission_tier": tool.get("permission_tier"),
                    }
                    for tool in manifest.get("tools", [])
                ],
                "capabilities": manifest.get("capabilities", {}),
            }
        )
    return plugins, sources


async def build_catalog() -> dict[str, Any]:
    providers, mesh_summary = await _provider_map()
    plugins, plugin_sources = _plugin_catalog()
    version_config = json.loads(
        (REPO_ROOT / "tauri-app" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )

    actions: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    independently_verified = 0
    for action_type in ActionType:
        action = Action(action_type=action_type, parameters={})
        platforms, platform_basis = _platforms(action_type)
        verifier = POSTCONDITION_VERIFIERS.get(action_type)
        independent = verifier is not None
        independently_verified += int(independent)
        tier_counts[action.permission_tier.name.lower()] += 1
        family = _action_family(action_type)
        family_counts[family] += 1
        actions.append(
            {
                "id": action_type.value,
                "family": family,
                "permission": {
                    "tier": int(action.permission_tier),
                    "name": action.permission_tier.name.lower(),
                    "approval_required": action.requires_confirmation,
                    "snapshot_required": action.requires_snapshot,
                    "irreversible": action.is_irreversible,
                },
                "platform_support": {
                    "platforms": platforms,
                    "basis": platform_basis,
                    "note": (
                        "Availability can still depend on host tools, permissions, hardware, and "
                        "configured integrations."
                    ),
                },
                "providers": providers.get(action_type.value, []),
                "verification": {
                    "method": verifier or "executor_result_only",
                    "independent_postcondition": independent,
                    "note": (
                        "Verifier checks this post-condition only when the live state is unambiguous; "
                        "other outcomes remain executor-reported."
                        if verifier and verifier.startswith("conditional_")
                        else "Verifier checks an observed post-condition."
                        if independent
                        else "No independent post-condition check is currently registered."
                    ),
                },
            }
        )

    source_path = DAEMON_ROOT / "pilot" / "actions.py"
    verifier_path = DAEMON_ROOT / "pilot" / "agents" / "verifier.py"
    return {
        "schema_version": "1.0.0",
        "product": {
            "name": "Heliox OS",
            "version": version_config["version"],
            "kind": "desktop AI system-control agent",
            "license": "MIT",
            "declared_platforms": list(ALL_PLATFORMS),
        },
        "interpretation": {
            "platforms": (
                "Declared platform targets are not a guarantee that every action is available on every host."
            ),
            "verification": (
                "Executor success and independent post-condition verification are reported separately."
            ),
            "neural": (
                "Synthetic or recorded-EEG validation does not establish live brain control or clinical use."
            ),
        },
        "sources": {
            "action_registry": {
                "path": source_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(source_path),
            },
            "postcondition_verifier": {
                "path": verifier_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(verifier_path),
            },
            "plugin_manifests": plugin_sources,
        },
        "summary": {
            "action_types": len(actions),
            "families": dict(sorted(family_counts.items())),
            "permission_tiers": dict(sorted(tier_counts.items())),
            "independent_postcondition_verifiers": independently_verified,
            "plugins": len(plugins),
            "mesh": mesh_summary,
        },
        "permission_tiers": [
            {"value": int(tier), "name": tier.name.lower()} for tier in PermissionTier
        ],
        "actions": actions,
        "plugins": plugins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "capabilities.json",
        help="Destination for the deterministic JSON catalog.",
    )
    args = parser.parse_args()
    catalog = asyncio.run(build_catalog())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {len(catalog['actions'])} actions and {len(catalog['plugins'])} plugins "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
