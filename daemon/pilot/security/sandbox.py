"""Sandboxed Preview — simulate destructive commands before execution.

Dry-run mode, undo history, and state snapshots. Lets users see
exactly what a command will do before dropping the `force` flag.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pilot.security.sandbox")


@dataclass
class SandboxPreview:
    """Result of a simulated action."""

    action_type: str
    target: str
    will_delete: list[str]
    will_modify: list[str]
    will_create: list[str]
    estimated_duration: float
    is_reversible: bool
    risk_level: str  # low, medium, high, critical
    warnings: list[str]


class SandboxEnvironment:
    """Manages dry-run simulations and temporary workspaces."""

    def __init__(self, workspace_dir: str | None = None):
        self._workspace = workspace_dir or os.path.join(tempfile.gettempdir(), "pilot_sandbox")
        os.makedirs(self._workspace, exist_ok=True)

    async def preview_file_delete(self, target_path: str, recursive: bool = False) -> SandboxPreview:
        """Simulate deleting a file or directory."""
        path = Path(target_path).resolve()
        preview = SandboxPreview(
            action_type="file_delete",
            target=str(path),
            will_delete=[],
            will_modify=[],
            will_create=[],
            estimated_duration=0.1,
            is_reversible=False,  # Deletions bypass recycle bin in scripts usually
            risk_level="high" if path.is_dir() and recursive else "medium",
            warnings=[],
        )

        if not path.exists():
            preview.warnings.append(f"Target does not exist: {path}")
            preview.risk_level = "low"
            return preview

        if path.is_dir():
            if not recursive:
                preview.warnings.append(f"Directory {path} cannot be deleted without recursive=True")
                return preview

            # Count files
            file_count = sum(1 for _ in path.rglob("*"))
            if file_count > 1000:
                preview.warnings.append(f"Mass deletion warning: ~{file_count} files will be deleted")
                preview.risk_level = "critical"
            preview.will_delete.append(f"Directory and {file_count} contents at {path}")
        else:
            preview.will_delete.append(str(path))
            # Check if system file
            if "Windows" in str(path) or "System32" in str(path):
                preview.warnings.append("CRITICAL: Attempting to delete a core system file!")
                preview.risk_level = "critical"

        return preview

    async def preview_shell_command(self, command: str) -> SandboxPreview:
        """Heuristically preview a shell command's impact."""
        command_lower = command.lower()
        preview = SandboxPreview(
            action_type="shell_command",
            target=command[:50],
            will_delete=[],
            will_modify=["System state (unknown degree)"],
            will_create=[],
            estimated_duration=1.0,
            is_reversible=False,
            risk_level="medium",
            warnings=[],
        )

        # Basic dumb heuristics
        if any(w in command_lower for w in ["rm ", "del ", "format ", "diskpart"]):
            preview.risk_level = "critical"
            preview.warnings.append("Command contains destructive keywords (rm, del, format)")
            preview.will_delete.append("Unknown files/directories based on command")

        if any(w in command_lower for w in ["apt", "dpkg", "yum", "dnf", "pacman", "winget", "choco"]):
            preview.risk_level = "high"
            preview.warnings.append("System package modification detected")

        if any(w in command_lower for w in ["systemctl", "sc ", "net stop", "net start"]):
            preview.warnings.append("Service state modification detected")

        if "sudo " in command_lower or "runas " in command_lower:
            preview.risk_level = "critical"
            preview.warnings.append("Command requests elevated privileges")

        return preview

    async def create_snapshot(self, path: str) -> str | None:
        """Create a backup snapshot of a file before modifying it."""
        src = Path(path).resolve()
        if not src.exists() or not src.is_file():
            return None

        # Don't snapshot giant files (>10MB)
        if src.stat().st_size > 10 * 1024 * 1024:
            logger.warning("File too large to snapshot: %s", src)
            return None

        snapshot_dir = Path(self._workspace) / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = snapshot_dir / f"{src.name}_{timestamp}.bak"

        shutil.copy2(src, dest)
        return str(dest)

    async def restore_snapshot(self, snapshot_path: str, target_path: str) -> bool:
        """Restore a file from a snapshot."""
        src = Path(snapshot_path)
        dest = Path(target_path)

        if not src.exists():
            return False

        try:
            shutil.copy2(src, dest)
            return True
        except Exception as e:
            logger.error("Failed to restore snapshot: %s", e)
            return False

    def clear(self):
        """Clean up the sandbox environment."""
        with contextlib.suppress(Exception):
            shutil.rmtree(self._workspace)


_sandbox = SandboxEnvironment()


async def preview_action(action_type: str, parameters: dict) -> str:
    """Generate a human-readable preview of what an action will do."""
    if action_type == "file_delete":
        preview = await _sandbox.preview_file_delete(parameters.get("path", ""), parameters.get("recursive", False))
    elif action_type in ("shell_command", "shell_script"):
        preview = await _sandbox.preview_shell_command(parameters.get("command", "") or parameters.get("script", ""))
    else:
        # Generic fallback
        preview = SandboxPreview(
            action_type=action_type,
            target=str(parameters.get("path") or parameters.get("name") or "system"),
            will_delete=[],
            will_modify=["System state"],
            will_create=[],
            estimated_duration=0.5,
            is_reversible=False,
            risk_level="medium",
            warnings=["Generic preview: effect depends entirely on parameters"],
        )

    return json.dumps(
        {
            "risk_level": preview.risk_level.upper(),
            "warnings": preview.warnings,
            "impact": {
                "delete": preview.will_delete,
                "modify": preview.will_modify,
                "create": preview.will_create,
            },
            "reversible": preview.is_reversible,
        },
        indent=2,
    )


async def safe_execute_with_snapshot(file_path: str, edit_func) -> bool:
    """Execute a function that modifies a file, with automatic snapshot rollback on failure."""
    snapshot = await _sandbox.create_snapshot(file_path)

    try:
        await edit_func()
        return True
    except Exception as e:
        logger.error("Edit failed, rolling back: %s", e)
        if snapshot:
            await _sandbox.restore_snapshot(snapshot, file_path)
        raise
