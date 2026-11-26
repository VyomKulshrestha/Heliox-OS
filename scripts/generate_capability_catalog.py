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
from pilot.agents.postconditions._read_only import READ_ONLY_WITHOUT_POSTCONDITION  # noqa: E402
from pilot.agents.postconditions._unobservable import UNOBSERVABLE_POSTCONDITION  # noqa: E402
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


DECLARED_PREFIX = "declared_"


def _postcondition_state(action_type: ActionType, verifier: str | None) -> str:
    """independent: always checked; declared: checked only when a plan declares a read-only probe;
    read_only: changes nothing, needs no check; none: the executor's word only."""
    if verifier is not None:
        return "declared" if verifier.startswith(DECLARED_PREFIX) else "independent"
    return "read_only" if action_type in READ_ONLY_WITHOUT_POSTCONDITION else "none"


def _verification(action_type: ActionType, verifier: str | None) -> dict[str, Any]:
    state = _postcondition_state(action_type, verifier)
    if state == "declared":
        return {
            "method": verifier,
            "postcondition": state,
            "independent_postcondition": False,
            "note": (
                "Checked only when the plan declares a read-only probe of the effect; without one the outcome "
                "is the executor's word."
            ),
        }
    if state == "read_only":
        return {
            "method": "read_only",
            "postcondition": state,
            "independent_postcondition": False,
            "reason": READ_ONLY_WITHOUT_POSTCONDITION[action_type],
            "note": (
                "Read-only: it changes nothing, so there is no post-condition to observe; a read that fails "
                "fails its action instead of returning an error message as data."
            ),
        }
    if verifier is None and action_type in UNOBSERVABLE_POSTCONDITION:
        return {
            "method": "unobservable_at_run_time",
            "postcondition": state,
            "independent_postcondition": False,
            "reason": UNOBSERVABLE_POSTCONDITION[action_type],
            "note": "No check is possible when it would run; the result says the change was requested, not done.",
        }
    return {
        "method": verifier or "executor_result_only",
        "postcondition": state,
        "independent_postcondition": verifier is not None,
        "note": (
            "Verifier checks this post-condition only when the live state is unambiguous; "
            "other outcomes remain executor-reported."
            if verifier and verifier.startswith("conditional_")
            else "Verifier checks an observed post-condition."
            if verifier
            else "No independent post-condition check is currently registered."
        ),
    }


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
    declared_verified = 0
    both = sorted(
        action.value
        for action in set(READ_ONLY_WITHOUT_POSTCONDITION) & set(POSTCONDITION_VERIFIERS)
    )
    if both:
        raise ValueError(
            f"Listed as read-only without a check but also verified: {', '.join(both)}"
        )
    for action_type in ActionType:
        action = Action(action_type=action_type, parameters={})
        platforms, platform_basis = _platforms(action_type)
        verifier = POSTCONDITION_VERIFIERS.get(action_type)
        state = _postcondition_state(action_type, verifier)
        independently_verified += int(state == "independent")
        declared_verified += int(state == "declared")
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
                "verification": _verification(action_type, verifier),
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
            "postcondition_area_modules": [
                {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(path)}
                for path in sorted((verifier_path.parent / "postconditions").glob("*.py"))
            ],
            "plugin_manifests": plugin_sources,
        },
        "summary": {
            "action_types": len(actions),
            "families": dict(sorted(family_counts.items())),
            "permission_tiers": dict(sorted(tier_counts.items())),
            "independent_postcondition_verifiers": independently_verified,
            "declared_postcondition_verifiers": declared_verified,
            "read_only_actions": len(READ_ONLY_WITHOUT_POSTCONDITION),
            "effect_actions": len(actions) - len(READ_ONLY_WITHOUT_POSTCONDITION),
            "plugins": len(plugins),
            "mesh": mesh_summary,
        },
        "permission_tiers": [
            {"value": int(tier), "name": tier.name.lower()} for tier in PermissionTier
        ],
        "actions": actions,
        "plugins": plugins,
    }


README_VERIFICATION_ROW = "| Outcome verification |"


def readme_with_verifier_counts(
    text: str, verified: int, effects: int, read_only: int, declared: int = 0
) -> str:
    """Rewrite the README's outcome-verification row from the catalog, keeping each cell's width where it fits."""

    lines = text.split("\n")
    rows = [index for index, line in enumerate(lines) if line.startswith(README_VERIFICATION_ROW)]
    if len(rows) != 1:
        raise ValueError(
            f"README must have exactly one row starting with {README_VERIFICATION_ROW!r}"
        )
    cells = lines[rows[0]].split("|")
    if len(cells) != 5:
        raise ValueError("README outcome-verification row must have three cells")
    contents = {
        2: (
            f"{verified} of {effects} actions that change something have an independent post-condition check"
            + (f", plus {declared} checked when a plan declares a check" if declared else "")
            + f"; {read_only} read-only actions need none"
        ),
        3: (
            f"The other {effects - verified - declared} rely on the executor result; a read that fails now fails "
            "its action instead of returning an error message as data"
        ),
    }
    for index, content in contents.items():
        width = len(cells[index]) - 2
        cells[index] = f" {content.ljust(width)} "
    lines[rows[0]] = "|".join(cells)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "capabilities.json",
        help="Destination for the deterministic JSON catalog.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="README whose verifier counts to rewrite (default: the repository's, only when --output is the default).",
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
    readme = args.readme
    if readme is None and args.output.resolve() == (REPO_ROOT / "capabilities.json").resolve():
        readme = REPO_ROOT / "README.md"
    if readme is not None:
        original = readme.read_bytes().decode("utf-8")
        summary = catalog["summary"]
        updated = readme_with_verifier_counts(
            original,
            summary["independent_postcondition_verifiers"],
            summary["effect_actions"],
            summary["read_only_actions"],
            summary.get("declared_postcondition_verifiers", 0),
        )
        if updated != original:
            readme.write_bytes(updated.encode("utf-8"))
            print(f"Updated the verifier counts in {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
