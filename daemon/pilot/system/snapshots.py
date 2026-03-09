"""Snapshot and rollback integration — Btrfs and Timeshift.

Automatically detects the filesystem type and uses the appropriate
snapshot mechanism. Falls back gracefully if neither is available.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.system.snapshots")


class SnapshotBackend(StrEnum):
    BTRFS = "btrfs"
    TIMESHIFT = "timeshift"
    NONE = "none"


async def _run(args: list[str], *, root: bool = False) -> tuple[int, str, str]:
    cmd = ["pkexec"] + args if root else args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
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

        if await self._is_btrfs_root():
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
        else:
            logger.warning("No snapshot backend available — proceeding without snapshot")
            return None

    async def rollback(self, snapshot_id: str) -> str:
        """Rollback to a previous snapshot."""
        backend = await self.detect_backend()

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_rollback(snapshot_id)
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_rollback(snapshot_id)
        else:
            raise RuntimeError("No snapshot backend available for rollback")

    async def list_snapshots(self) -> list[dict[str, str]]:
        """List available Pilot snapshots."""
        backend = await self.detect_backend()

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_list()
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_list()
        return []

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
