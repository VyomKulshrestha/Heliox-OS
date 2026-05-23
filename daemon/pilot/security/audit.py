"""Append-only audit logger.

Every executed action is recorded to an immutable JSONL log file.
This log is tamper-evident and can be shipped to external SIEM systems.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from pilot.actions import Action, ActionResult
from pilot.config import AUDIT_FILE, DATA_DIR

logger = logging.getLogger("pilot.security.audit")


class AuditEntry:
    __slots__ = ("timestamp", "event_type", "action_type", "target", "success", "details", "user")

    def __init__(
        self,
        event_type: str,
        action_type: str = "",
        target: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.timestamp = datetime.now(UTC).isoformat()
        self.event_type = event_type
        self.action_type = action_type
        self.target = target
        self.success = success
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "action_type": self.action_type,
            "target": self.target,
            "success": self.success,
            "details": self.details,
        }


class AuditLogger:
    """Writes audit entries to an append-only JSONL file."""

    def __init__(self, audit_file: Path | None = None) -> None:
        self._file = audit_file or AUDIT_FILE
        self._lock = asyncio.Lock()

    async def log_action_start(self, action: Action, plan_id: str, dry_run: bool = False) -> None:
        action_type = action.action_type.value
        if dry_run:
            action_type = f"(dry run) {action_type}"
        entry = AuditEntry(
            event_type="action_start",
            action_type=action_type,
            target=action.target,
            details={
                "plan_id": plan_id,
                "requires_root": action.requires_root,
                "destructive": action.destructive,
                "permission_tier": action.permission_tier.value,
                "dry_run": dry_run,
            },
        )
        await self._write(entry)

    async def log_action_result(self, result: ActionResult, plan_id: str, dry_run: bool = False) -> None:
        action_type = result.action.action_type.value
        if dry_run:
            action_type = f"(dry run) {action_type}"
        entry = AuditEntry(
            event_type="action_complete",
            action_type=action_type,
            target=result.action.target,
            success=result.success,
            details={
                "plan_id": plan_id,
                "output_preview": result.output[:200] if result.output else "",
                "error": result.error,
                "snapshot_id": result.snapshot_id,
                "dry_run": dry_run,
            },
        )
        await self._write(entry)

    async def log_rollback(self, snapshot_id: str, plan_id: str, reason: str) -> None:
        entry = AuditEntry(
            event_type="rollback",
            details={
                "plan_id": plan_id,
                "snapshot_id": snapshot_id,
                "reason": reason,
            },
        )
        await self._write(entry)

    async def log_config_change(self, section: str, key: str, old_value: Any, new_value: Any) -> None:
        entry = AuditEntry(
            event_type="config_change",
            details={
                "section": section,
                "key": key,
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )
        await self._write(entry)

    async def log_security_event(self, event: str, details: dict[str, Any] | None = None) -> None:
        entry = AuditEntry(
            event_type="security",
            details={"event": event, **(details or {})},
        )
        await self._write(entry)

    async def _write(self, entry: AuditEntry) -> None:
        line = json.dumps(entry.to_dict(), separators=(",", ":")) + "\n"
        try:
            await aiofiles.os.makedirs(self._file.parent, exist_ok=True)
            async with self._lock, aiofiles.open(self._file, "a", encoding="utf-8") as f:
                await f.write(line)
        except OSError:
            logger.exception("Failed to write audit log")
