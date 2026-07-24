"""Snapshot and rollback integration — Btrfs and Timeshift.

Automatically detects the filesystem type and uses the appropriate
snapshot mechanism. Falls back gracefully if neither is available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.system.snapshots")


class SnapshotBackend(StrEnum):
    BTRFS = "btrfs"
    TIMESHIFT = "timeshift"
    WINDOWS_RESTORE_POINT = "windows_restore_point"
    NONE = "none"


async def _run(args: list[str], *, root: bool = False) -> tuple[int, str, str]:
    cmd = ["pkexec"] + args if root and sys.platform != "win32" else args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        return 127, "", str(error)
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


class SnapshotManager:
    """Manages system snapshots for rollback capability."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._backend: SnapshotBackend | None = None

    async def detect_backend(self) -> SnapshotBackend:
        """Auto-detect the best available snapshot backend."""
        if self._backend is not None:
            return self._backend

        configured = self._config.security.snapshot_backend
        if configured != "auto":
            self._backend = SnapshotBackend(configured)
            return self._backend

        if sys.platform == "win32" and await self._is_windows_restore_available():
            self._backend = SnapshotBackend.WINDOWS_RESTORE_POINT
        elif await self._is_btrfs_root():
            self._backend = SnapshotBackend.BTRFS
        elif await self._is_timeshift_available():
            self._backend = SnapshotBackend.TIMESHIFT
        else:
            self._backend = SnapshotBackend.NONE

        logger.info("Snapshot backend: %s", self._backend.value)
        return self._backend

    async def create_snapshot(self, action_id: str, description: str = "") -> str | None:
        """Create a pre-action snapshot. Returns snapshot ID or None."""
        backend = await self.detect_backend()
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        tag = f"pilot-{action_id}-{timestamp}"

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_snapshot(tag, description)
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_snapshot(tag, description)
        elif backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            return await self._windows_restore_snapshot(tag, description)
        else:
            logger.warning("No snapshot backend available")
            return None

    async def rollback(self, snapshot_id: str) -> str:
        """Rollback to a previous snapshot."""
        backend = await self.detect_backend()

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_rollback(snapshot_id)
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_rollback(snapshot_id)
        elif backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            return await self._windows_restore_rollback(snapshot_id)
        else:
            raise RuntimeError("No snapshot backend available for rollback")

    async def list_snapshots(self) -> list[dict[str, str]]:
        """List available Pilot snapshots."""
        backend = await self.detect_backend()

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_list()
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_list()
        elif backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            return await self._windows_restore_list()
        return []

    async def status(self) -> dict[str, str | bool]:
        """Return whether configured snapshot protection can run right now."""
        backend = await self.detect_backend()
        enabled = self._config.security.snapshot_on_destructive
        available = backend != SnapshotBackend.NONE
        ready = available

        if backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            from pilot.security.privileges import has_elevated_privileges

            ready = has_elevated_privileges()
            detail = (
                "Windows Restore Point is ready for destructive actions."
                if ready
                else (
                    "Windows Restore Point is installed, but the daemon is not Administrator. "
                    "Required snapshots will fail closed and destructive actions will not run."
                )
            )
        elif available:
            detail = f"{backend.value} is available for pre-action snapshots."
        else:
            detail = (
                "No supported snapshot backend is available. When Auto-Snapshot is enabled, "
                "destructive actions will fail closed instead of running without rollback protection."
            )

        return {
            "enabled": enabled,
            "backend": backend.value,
            "available": available,
            "ready": ready,
            "detail": detail,
        }

    async def cleanup(self) -> int:
        """Remove old snapshots per retention policy. Returns count removed."""
        retention = self._config.security.snapshot_retention_count
        snapshots = await self.list_snapshots()
        pilot_snapshots = [s for s in snapshots if s.get("tag", "").startswith("pilot-")]

        if len(pilot_snapshots) <= retention:
            return 0

        to_remove = pilot_snapshots[retention:]
        removed = 0
        for snap in to_remove:
            try:
                sid = snap.get("id", "")
                if sid:
                    backend = await self.detect_backend()
                    if backend == SnapshotBackend.BTRFS:
                        await _run(["btrfs", "subvolume", "delete", sid], root=True)
                    elif backend == SnapshotBackend.TIMESHIFT:
                        await _run(["timeshift", "--delete", "--snapshot", sid], root=True)
                    removed += 1
            except Exception:
                logger.warning("Failed to remove snapshot: %s", snap)

        return removed

    # -- Btrfs --

    async def _is_btrfs_root(self) -> bool:
        code, out, _ = await _run(["stat", "-f", "--format=%T", "/"])
        return "btrfs" in out.lower()

    async def _btrfs_snapshot(self, tag: str, description: str) -> str:
        snapshot_path = f"/.snapshots/{tag}"
        code, out, err = await _run(["btrfs", "subvolume", "snapshot", "/", snapshot_path], root=True)
        if code != 0:
            raise RuntimeError(f"Btrfs snapshot failed: {err.strip()}")
        logger.info("Created Btrfs snapshot: %s", snapshot_path)
        return snapshot_path

    async def _btrfs_rollback(self, snapshot_id: str) -> str:
        code, _, err = await _run(
            ["btrfs", "subvolume", "snapshot", snapshot_id, "/rollback-target"],
            root=True,
        )
        if code != 0:
            raise RuntimeError(f"Btrfs rollback failed: {err.strip()}")
        return f"Rollback snapshot created from {snapshot_id}. Reboot to apply."

    async def _btrfs_list(self) -> list[dict[str, str]]:
        code, out, _ = await _run(["btrfs", "subvolume", "list", "/.snapshots"], root=True)
        if code != 0:
            return []
        snapshots = []
        for line in out.strip().split("\n"):
            if "pilot-" in line:
                parts = line.split()
                if len(parts) >= 9:
                    snapshots.append({"id": parts[-1], "tag": parts[-1].split("/")[-1]})
        return snapshots

    # -- Timeshift --

    async def _is_timeshift_available(self) -> bool:
        code, _, _ = await _run(["which", "timeshift"])
        return code == 0

    # -- Windows System Restore --

    async def _is_windows_restore_available(self) -> bool:
        code, out, _ = await _run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-Command Checkpoint-Computer -ErrorAction SilentlyContinue).Name",
            ]
        )
        return code == 0 and "Checkpoint-Computer" in out

    async def _windows_restore_snapshot(self, tag: str, description: str) -> str:
        label = (description or tag).replace("'", "''")[:200]
        script = (
            "$ErrorActionPreference='Stop'; "
            f"Checkpoint-Computer -Description '{label}' -RestorePointType MODIFY_SETTINGS; "
            "(Get-ComputerRestorePoint | Sort-Object SequenceNumber | "
            "Select-Object -Last 1 -ExpandProperty SequenceNumber)"
        )
        code, out, err = await _run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            root=True,
        )
        sequence = out.strip().splitlines()[-1] if out.strip() else ""
        if code != 0 or not sequence.isdigit():
            detail = err.strip() or out.strip() or "no restore point ID returned"
            raise RuntimeError(f"Windows Restore Point failed: {detail}")
        snapshot_id = f"windows-restore:{sequence}"
        logger.info("Created Windows Restore Point: %s", snapshot_id)
        return snapshot_id

    async def _windows_restore_rollback(self, snapshot_id: str) -> str:
        sequence = snapshot_id.removeprefix("windows-restore:")
        if not sequence.isdigit():
            raise ValueError("Invalid Windows restore point ID")
        script = (
            "$ErrorActionPreference='Stop'; "
            f"Restore-Computer -RestorePoint {sequence} -Confirm:$false"
        )
        code, out, err = await _run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            root=True,
        )
        if code != 0:
            raise RuntimeError(f"Windows restore failed: {(err or out).strip()}")
        return f"Windows restore point {sequence} selected. Restart Windows to apply it."

    async def _windows_restore_list(self) -> list[dict[str, str]]:
        code, out, _ = await _run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-ComputerRestorePoint | Select-Object SequenceNumber,Description | ConvertTo-Json -Compress",
            ]
        )
        if code != 0 or not out.strip():
            return []
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return []
        rows = parsed if isinstance(parsed, list) else [parsed]
        return [
            {
                "id": f"windows-restore:{row['SequenceNumber']}",
                "tag": str(row.get("Description", "")),
            }
            for row in rows
            if isinstance(row, dict) and "SequenceNumber" in row
        ]

    async def _timeshift_snapshot(self, tag: str, description: str) -> str:
        comment = description or f"Pilot pre-action snapshot: {tag}"
        code, out, err = await _run(
            ["timeshift", "--create", f"--comments={comment}", "--tags=D"],
            root=True,
        )
        if code != 0:
            raise RuntimeError(f"Timeshift snapshot failed: {err.strip()}")
        logger.info("Created Timeshift snapshot: %s", tag)
        return tag

    async def _timeshift_rollback(self, snapshot_id: str) -> str:
        code, _, err = await _run(
            ["timeshift", "--restore", "--snapshot", snapshot_id, "--yes"],
            root=True,
        )
        if code != 0:
            raise RuntimeError(f"Timeshift rollback failed: {err.strip()}")
        return f"Timeshift rollback to {snapshot_id} complete. Reboot recommended."

    async def _timeshift_list(self) -> list[dict[str, str]]:
        code, out, _ = await _run(["timeshift", "--list"], root=True)
        if code != 0:
            return []
        snapshots = []
        for line in out.strip().split("\n"):
            if "pilot-" in line.lower():
                parts = line.split()
                if parts:
                    snapshots.append({"id": parts[0], "tag": line.strip()})
        return snapshots
