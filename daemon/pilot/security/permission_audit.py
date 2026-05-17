"""Tamper-evident audit store for permission escalation events."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from pilot.config import PERMISSION_AUDIT_DB_FILE, PERMISSION_AUDIT_KEY_FILE

logger = logging.getLogger("pilot.security.permission_audit")


@dataclass(frozen=True)
class ChainVerificationResult:
    valid: bool
    checked_entries: int
    error: str = ""


class PermissionEscalationAuditStore:
    """Append-only SQLite audit log with an HMAC chain.

    The database stores high-risk permission events separately from the general
    JSONL audit log. Every row commits to the previous row's HMAC, so deleting,
    reordering, or modifying any prior record is detectable by ``verify_chain``.
    """

    def __init__(
        self,
        db_file: Path | None = None,
        key_file: Path | None = None,
        key: bytes | None = None,
    ) -> None:
        self._db_file = db_file or PERMISSION_AUDIT_DB_FILE
        self._key_file = key_file or PERMISSION_AUDIT_KEY_FILE
        self._key = key

    async def initialize(self) -> None:
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        self._key = self._key or self._load_or_create_key()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_escalation_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    permission_tier TEXT NOT NULL,
                    requires_root INTEGER NOT NULL,
                    destructive INTEGER NOT NULL,
                    confirmation_decision TEXT NOT NULL,
                    critic_verdict TEXT NOT NULL,
                    execution_success INTEGER,
                    execution_error TEXT NOT NULL,
                    previous_hmac TEXT NOT NULL,
                    entry_hmac TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_permission_audit_plan_id
                ON permission_escalation_audit(plan_id)
                """
            )
            await db.commit()

    async def record_event(
        self,
        *,
        plan_id: str,
        action_index: int,
        action_type: str,
        target: str,
        permission_tier: str,
        requires_root: bool,
        destructive: bool,
        confirmation_decision: str,
        critic_verdict: dict[str, Any] | None = None,
        execution_success: bool | None = None,
        execution_error: str = "",
    ) -> str:
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            previous_hmac = await self._last_hmac(db)
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "plan_id": plan_id,
                "action_index": action_index,
                "action_type": action_type,
                "target": target,
                "permission_tier": permission_tier,
                "requires_root": bool(requires_root),
                "destructive": bool(destructive),
                "confirmation_decision": confirmation_decision,
                "critic_verdict": critic_verdict or {},
                "execution_success": execution_success,
                "execution_error": execution_error,
                "previous_hmac": previous_hmac,
            }
            entry_hmac = self._sign_payload(payload)
            await db.execute(
                """
                INSERT INTO permission_escalation_audit (
                    timestamp,
                    plan_id,
                    action_index,
                    action_type,
                    target,
                    permission_tier,
                    requires_root,
                    destructive,
                    confirmation_decision,
                    critic_verdict,
                    execution_success,
                    execution_error,
                    previous_hmac,
                    entry_hmac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["timestamp"],
                    plan_id,
                    action_index,
                    action_type,
                    target,
                    permission_tier,
                    int(requires_root),
                    int(destructive),
                    confirmation_decision,
                    self._json_dumps(critic_verdict or {}),
                    None if execution_success is None else int(execution_success),
                    execution_error,
                    previous_hmac,
                    entry_hmac,
                ),
            )
            await db.commit()
            return entry_hmac

    async def verify_chain(self) -> ChainVerificationResult:
        await self.initialize()
        expected_previous = ""
        checked = 0
        async with (
            aiosqlite.connect(self._db_file) as db,
            db.execute(
                """
                SELECT
                    id,
                    timestamp,
                    plan_id,
                    action_index,
                    action_type,
                    target,
                    permission_tier,
                    requires_root,
                    destructive,
                    confirmation_decision,
                    critic_verdict,
                    execution_success,
                    execution_error,
                    previous_hmac,
                    entry_hmac
                FROM permission_escalation_audit
                ORDER BY id ASC
                """
            ) as cursor,
        ):
            async for row in cursor:
                checked += 1
                (
                    row_id,
                    timestamp,
                    plan_id,
                    action_index,
                    action_type,
                    target,
                    permission_tier,
                    requires_root,
                    destructive,
                    confirmation_decision,
                    critic_verdict,
                    execution_success,
                    execution_error,
                    previous_hmac,
                    entry_hmac,
                ) = row
                if previous_hmac != expected_previous:
                    return ChainVerificationResult(
                        valid=False,
                        checked_entries=checked,
                        error=f"Row {row_id} previous_hmac mismatch",
                    )

                payload = {
                    "timestamp": timestamp,
                    "plan_id": plan_id,
                    "action_index": action_index,
                    "action_type": action_type,
                    "target": target,
                    "permission_tier": permission_tier,
                    "requires_root": bool(requires_root),
                    "destructive": bool(destructive),
                    "confirmation_decision": confirmation_decision,
                    "critic_verdict": json.loads(critic_verdict),
                    "execution_success": None if execution_success is None else bool(execution_success),
                    "execution_error": execution_error,
                    "previous_hmac": previous_hmac,
                }
                expected_hmac = self._sign_payload(payload)
                if not hmac.compare_digest(entry_hmac, expected_hmac):
                    return ChainVerificationResult(
                        valid=False,
                        checked_entries=checked,
                        error=f"Row {row_id} entry_hmac mismatch",
                    )
                expected_previous = entry_hmac

        return ChainVerificationResult(valid=True, checked_entries=checked)

    async def _last_hmac(self, db: aiosqlite.Connection) -> str:
        async with db.execute(
            """
            SELECT entry_hmac FROM permission_escalation_audit
            ORDER BY id DESC
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row else ""

    def _load_or_create_key(self) -> bytes:
        self._key_file.parent.mkdir(parents=True, exist_ok=True)
        if self._key_file.exists():
            return base64.b64decode(self._key_file.read_text(encoding="utf-8"))

        key = secrets.token_bytes(32)
        self._key_file.write_text(base64.b64encode(key).decode("ascii"), encoding="utf-8")
        try:
            os.chmod(self._key_file, 0o600)
        except OSError:
            logger.warning("Unable to restrict permission audit key file permissions", exc_info=True)
        return key

    def _sign_payload(self, payload: dict[str, Any]) -> str:
        if self._key is None:
            self._key = self._load_or_create_key()
        return hmac.new(self._key, self._json_dumps(payload).encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _json_dumps(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
