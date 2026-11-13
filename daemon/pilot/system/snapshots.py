"""Snapshot and rollback integration — Btrfs and Timeshift.

Automatically detects the filesystem type and uses the appropriate
snapshot mechanism. Falls back gracefully if neither is available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import sys
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from pilot.actions import Action
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.system.snapshots")


class SnapshotBackend(StrEnum):
    BTRFS = "btrfs"
    TIMESHIFT = "timeshift"
    WINDOWS_RESTORE_POINT = "windows_restore_point"
    NONE = "none"


_PILOT_TIMESTAMP = re.compile(r"(\d{8}-\d{6})")
_FILE_SNAPSHOT_ID = re.compile(r"^[a-zA-Z0-9._-]+$")
_MAX_FILE_SNAPSHOT_BYTES = 100 * 1024 * 1024
_MAX_FILE_SNAPSHOT_ENTRIES = 10_000


def _snapshot_sort_key(snapshot: dict[str, str]) -> tuple[str, int]:
    """Return a newest-first-compatible key for Pilot-created snapshots."""
    tag = snapshot.get("tag", "")
    timestamp_match = _PILOT_TIMESTAMP.search(tag)
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    snapshot_id = snapshot.get("id", "")
    numeric_id = snapshot_id.removeprefix("windows-restore:")
    return timestamp, int(numeric_id) if numeric_id.isdigit() else 0


def _snapshot_created_at(snapshot: dict[str, str]) -> datetime | None:
    timestamp_match = _PILOT_TIMESTAMP.search(snapshot.get("tag", ""))
    if timestamp_match is None:
        return None
    try:
        return datetime.strptime(timestamp_match.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _as_root(args: list[str], *, root: bool) -> list[str]:
    """The command as it is run: through pkexec on macOS and Linux when it needs root (the system then asks for an
    administrator's password, unless a polkit rule allows it), as it is on Windows (where it runs with the daemon's
    own rights: nothing elevates it)."""
    return ["pkexec", *args] if root and sys.platform != "win32" else list(args)


def windows_restore_command(snapshot_id: str) -> list[str]:
    sequence = snapshot_id.removeprefix("windows-restore:")
    if not sequence.isdigit():
        raise ValueError("Invalid Windows restore point ID")
    script = f"$ErrorActionPreference='Stop'; Restore-Computer -RestorePoint {sequence} -Confirm:$false"
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def timeshift_restore_command(snapshot_id: str) -> list[str]:
    return ["timeshift", "--restore", "--snapshot", snapshot_id, "--yes"]


def btrfs_restore_command(snapshot_id: str) -> list[str]:
    return ["btrfs", "subvolume", "snapshot", snapshot_id, "/rollback-target"]


def rollback_request(snapshot_id: str, backend: str, taken_at: datetime | None) -> tuple[Action, str]:
    """What the approval dialog shows before a rollback: the one thing that will run (the exact command, or the
    files put back) and, in plain words, what is lost and since when. Pure."""
    from pilot.actions import Action, ActionType, FileParams, ShellCommandParams

    when = taken_at.astimezone().strftime("%d %B %Y at %H:%M") if taken_at else "just before that task ran"
    if snapshot_id.startswith("file-snapshot:"):
        action = Action(
            action_type=ActionType.FILE_WRITE,
            target=snapshot_id,
            parameters=FileParams(path=snapshot_id),
            destructive=True,
        )
        reason = (
            f"Put back the files that task changed, as they were on {when}. Any change made to those files since "
            "then is lost."
        )
        return action, reason
    if backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
        command = windows_restore_command(snapshot_id)
        reason = (
            f"Return Windows to the restore point made on {when}. Programs, drivers, updates and system settings "
            "changed since then are undone; Windows restarts to do it. Your documents are not changed by a restore "
            "point."
        )
    elif backend == SnapshotBackend.TIMESHIFT:
        command = _as_root(timeshift_restore_command(snapshot_id), root=True)
        reason = (
            f"Restore the system with Timeshift to the snapshot made on {when}. Every change to the system's files "
            "since then is lost (with Timeshift's usual settings your home folder is not part of it). The system "
            "asks for an administrator's password first."
        )
    elif backend == SnapshotBackend.BTRFS:
        command = _as_root(btrfs_restore_command(snapshot_id), root=True)
        reason = (
            f"Make a copy of the system snapshot taken on {when} at /rollback-target. The system asks for an "
            "administrator's password first."
        )
    else:
        raise RuntimeError("No snapshot backend available for rollback")
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target=" ".join(command),
        parameters=ShellCommandParams(command=command[0], args=command[1:]),
        destructive=True,
    )
    return action, reason


async def _run(args: list[str], *, root: bool = False) -> tuple[int, str, str]:
    cmd = _as_root(args, root=root)
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

    def __init__(self, config: PilotConfig, *, file_snapshot_dir: Path | None = None) -> None:
        from pilot.config import DATA_DIR

        self._config = config
        self._backend: SnapshotBackend | None = None
        self._file_snapshot_dir = file_snapshot_dir or (DATA_DIR / "file_snapshots")

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

        # Status checks construct short-lived managers, so INFO here floods the
        # daemon log while Settings waits for an unavailable backend to recover.
        logger.debug("Snapshot backend: %s", self._backend.value)
        return self._backend

    async def create_snapshot(self, action_id: str, description: str = "") -> str | None:
        """Create a pre-action snapshot. Returns snapshot ID or None."""
        backend = await self.detect_backend()
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        tag = f"pilot-{action_id}-{timestamp}"

        if backend == SnapshotBackend.BTRFS:
            snapshot_id = await self._btrfs_snapshot(tag, description)
        elif backend == SnapshotBackend.TIMESHIFT:
            snapshot_id = await self._timeshift_snapshot(tag, description)
        elif backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            snapshot_id = await self._windows_restore_snapshot(tag, description)
        else:
            logger.warning("No snapshot backend available")
            return None

        if backend in {SnapshotBackend.BTRFS, SnapshotBackend.TIMESHIFT}:
            try:
                removed = await self.cleanup()
                if removed:
                    logger.info("Snapshot retention removed %d old Pilot snapshot(s)", removed)
            except Exception:
                logger.warning("Snapshot retention cleanup failed", exc_info=True)

        return snapshot_id

    async def rollback(self, snapshot_id: str) -> str:
        """Rollback to a previous snapshot."""
        if snapshot_id.startswith("file-snapshot:"):
            return await self._rollback_file_snapshot(snapshot_id)

        backend = await self.detect_backend()

        if backend == SnapshotBackend.BTRFS:
            return await self._btrfs_rollback(snapshot_id)
        elif backend == SnapshotBackend.TIMESHIFT:
            return await self._timeshift_rollback(snapshot_id)
        elif backend == SnapshotBackend.WINDOWS_RESTORE_POINT:
            return await self._windows_restore_rollback(snapshot_id)
        else:
            raise RuntimeError("No snapshot backend available for rollback")

    async def create_file_snapshot(self, action_id: str, paths: list[str]) -> str | None:
        """Snapshot exact file targets without requiring a system restore point."""
        unique_paths = list(dict.fromkeys(str(Path(path).resolve()) for path in paths if path))
        if not unique_paths:
            return None
        return await asyncio.to_thread(self._create_file_snapshot_sync, action_id, unique_paths)

    def _create_file_snapshot_sync(self, action_id: str, paths: list[str]) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        safe_action_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", action_id).strip("-") or "plan"
        directory_name = f"{safe_action_id}-{timestamp}-{uuid4().hex[:8]}"
        self._file_snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = self._file_snapshot_dir / directory_name
        entries_dir = snapshot_dir / "entries"
        entries_dir.mkdir(parents=True)

        manifest_entries: list[dict[str, str | bool]] = []
        total_bytes = 0
        total_entries = 0
        try:
            for index, raw_path in enumerate(paths):
                original = Path(raw_path).resolve()
                if original == Path(original.anchor):
                    raise ValueError(f"Refusing to snapshot filesystem root: {original}")

                entry: dict[str, str | bool] = {
                    "path": str(original),
                    "existed": original.exists(),
                    "kind": "absent",
                    "backup": "",
                }
                if not original.exists():
                    manifest_entries.append(entry)
                    continue
                if original.is_symlink():
                    raise ValueError(f"Symbolic links are not supported by file snapshots: {original}")

                backup_name = f"{index:06d}"
                backup_path = entries_dir / backup_name
                entry["backup"] = f"entries/{backup_name}"
                if original.is_file():
                    size = original.stat().st_size
                    total_bytes += size
                    total_entries += 1
                    entry["kind"] = "file"
                    self._check_file_snapshot_limits(total_bytes, total_entries)
                    shutil.copy2(original, backup_path)
                elif original.is_dir():
                    entry["kind"] = "directory"
                    for child in original.rglob("*"):
                        if child.is_symlink():
                            raise ValueError(f"Symbolic links are not supported by file snapshots: {child}")
                        total_entries += 1
                        if child.is_file():
                            total_bytes += child.stat().st_size
                        self._check_file_snapshot_limits(total_bytes, total_entries)
                    shutil.copytree(original, backup_path)
                else:
                    raise ValueError(f"Unsupported file target: {original}")
                manifest_entries.append(entry)

            manifest = {
                "version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "action_id": action_id,
                "entries": manifest_entries,
            }
            (snapshot_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise

        self._cleanup_file_snapshots()
        snapshot_id = f"file-snapshot:{directory_name}"
        logger.info("Created local file snapshot %s for %d target(s)", snapshot_id, len(paths))
        return snapshot_id

    @staticmethod
    def _check_file_snapshot_limits(total_bytes: int, total_entries: int) -> None:
        if total_bytes > _MAX_FILE_SNAPSHOT_BYTES:
            raise ValueError("File snapshot exceeds the 100 MiB safety limit")
        if total_entries > _MAX_FILE_SNAPSHOT_ENTRIES:
            raise ValueError("File snapshot exceeds the 10,000-entry safety limit")

    async def _rollback_file_snapshot(self, snapshot_id: str) -> str:
        return await asyncio.to_thread(self._rollback_file_snapshot_sync, snapshot_id)

    def _rollback_file_snapshot_sync(self, snapshot_id: str) -> str:
        directory_name = snapshot_id.removeprefix("file-snapshot:")
        if not _FILE_SNAPSHOT_ID.fullmatch(directory_name):
            raise ValueError("Invalid file snapshot ID")
        snapshot_dir = (self._file_snapshot_dir / directory_name).resolve()
        if snapshot_dir.parent != self._file_snapshot_dir.resolve():
            raise ValueError("Invalid file snapshot path")

        manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Invalid file snapshot manifest")

        restored = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid file snapshot entry")
            target = Path(str(entry.get("path", ""))).resolve()
            if not str(target) or target == Path(target.anchor):
                raise ValueError("Invalid file snapshot target")
            self._remove_exact_path(target)
            if not entry.get("existed"):
                restored += 1
                continue

            backup = (snapshot_dir / str(entry.get("backup", ""))).resolve()
            if snapshot_dir not in backup.parents:
                raise ValueError("Invalid file snapshot backup path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.get("kind") == "file":
                shutil.copy2(backup, target)
            elif entry.get("kind") == "directory":
                shutil.copytree(backup, target)
            else:
                raise ValueError("Invalid file snapshot entry kind")
            restored += 1
        return f"Restored {restored} file target(s) from {snapshot_id}."

    @staticmethod
    def _remove_exact_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    def _cleanup_file_snapshots(self) -> None:
        retention_count = max(1, self._config.security.snapshot_retention_count)
        retention_days = max(1, self._config.security.snapshot_retention_days)
        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=retention_days)).timestamp()
        snapshots = sorted(
            (path for path in self._file_snapshot_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for index, snapshot in enumerate(snapshots):
            if index >= retention_count or snapshot.stat().st_mtime < cutoff_timestamp:
                shutil.rmtree(snapshot, ignore_errors=True)

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

    async def status(self) -> dict[str, str | bool | int]:
        """Return whether configured snapshot protection can run right now."""
        backend = await self.detect_backend()
        enabled = self._config.security.snapshot_on_destructive
        available = backend != SnapshotBackend.NONE
        ready = available
        system_retention_supported = backend in {SnapshotBackend.BTRFS, SnapshotBackend.TIMESHIFT}

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
            "status": "ok",
            "enabled": enabled,
            "backend": backend.value,
            "available": available,
            "ready": ready,
            "detail": detail,
            "retention_supported": True,
            "system_retention_supported": system_retention_supported,
            "retention_count": self._config.security.snapshot_retention_count,
            "retention_days": self._config.security.snapshot_retention_days,
            "file_snapshot_available": True,
            "file_snapshot_detail": (
                "File-only destructive workflows use local content snapshots and do not require Administrator access."
            ),
            "retention_detail": (
                "Heliox enforces count and age limits for both system and local file snapshots."
                if system_retention_supported
                else (
                    "Heliox enforces count and age limits for local file snapshots. Windows manages "
                    "Restore Point retention because its supported APIs do not expose individual deletion."
                    if backend == SnapshotBackend.WINDOWS_RESTORE_POINT
                    else "Heliox enforces count and age limits for local file snapshots."
                )
            ),
        }

    async def cleanup(self) -> int:
        """Remove old snapshots per retention policy. Returns count removed."""
        backend = await self.detect_backend()
        if backend not in {SnapshotBackend.BTRFS, SnapshotBackend.TIMESHIFT}:
            return 0

        retention_count = max(1, self._config.security.snapshot_retention_count)
        retention_days = max(1, self._config.security.snapshot_retention_days)
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        snapshots = await self.list_snapshots()
        pilot_snapshots = [s for s in snapshots if s.get("tag", "").startswith("pilot-")]
        pilot_snapshots.sort(key=_snapshot_sort_key, reverse=True)

        to_remove = [
            snapshot
            for index, snapshot in enumerate(pilot_snapshots)
            if index >= retention_count
            or ((created_at := _snapshot_created_at(snapshot)) is not None and created_at < cutoff)
        ]
        removed = 0
        for snap in to_remove:
            try:
                sid = snap.get("id", "")
                if sid:
                    if backend == SnapshotBackend.BTRFS:
                        code, _, error = await _run(
                            ["btrfs", "subvolume", "delete", sid],
                            root=True,
                        )
                    elif backend == SnapshotBackend.TIMESHIFT:
                        code, _, error = await _run(
                            ["timeshift", "--delete", "--snapshot", sid],
                            root=True,
                        )
                    if code == 0:
                        removed += 1
                    else:
                        logger.warning(
                            "Failed to remove snapshot %s: %s",
                            sid,
                            error.strip(),
                        )
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
        code, _, err = await _run(btrfs_restore_command(snapshot_id), root=True)
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
        label = f"{tag}: {description}".rstrip(": ").replace("'", "''")[:200]
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
        code, out, err = await _run(windows_restore_command(snapshot_id), root=True)
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
                (
                    "Get-ComputerRestorePoint | Sort-Object SequenceNumber -Descending | "
                    "Select-Object SequenceNumber,Description | ConvertTo-Json -Compress"
                ),
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
        comment = f"{tag}: {description}".rstrip(": ")
        code, out, err = await _run(
            ["timeshift", "--create", f"--comments={comment}", "--tags=D"],
            root=True,
        )
        if code != 0:
            raise RuntimeError(f"Timeshift snapshot failed: {err.strip()}")
        logger.info("Created Timeshift snapshot: %s", tag)
        return tag

    async def _timeshift_rollback(self, snapshot_id: str) -> str:
        code, _, err = await _run(timeshift_restore_command(snapshot_id), root=True)
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
