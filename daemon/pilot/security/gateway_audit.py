"""Tamper-evident audit store for Agent Gateway decisions.

A separate HMAC-chained store from PermissionEscalationAuditStore
(permission_audit.py), reusing the same chain-of-custody pattern (see
ChainVerificationResult, imported rather than redefined) but with its own
database/key file and a schema tailored to gateway decisions — source
profile, action family, override narrowing, allow/deny/confirm outcome.

Kept separate rather than folded into the existing escalation-audit table
because that table's semantics (escalation decisions at confirmation/
execution time — a small, security-critical volume the existing
verify_chain() and Settings UI are already built around) shouldn't be
diluted by every gateway-covered action regardless of tier, and there is no
schema-migration path on that existing table today. Two independent HMAC
chains also mean a compromise of one key doesn't help forge the other.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from pilot.config import AGENT_GATEWAY_AUDIT_DB_FILE, AGENT_GATEWAY_AUDIT_KEY_FILE
from pilot.security.permission_audit import ChainVerificationResult
from pilot.security.secret_urls import redact_text

logger = logging.getLogger("pilot.security.gateway_audit")


class AgentGatewayAuditStore:
    """Append-only SQLite audit log with an HMAC chain for AgentGateway
    decisions — every row commits to the previous row's HMAC, so deleting,
    reordering, or modifying any prior record is detectable via
    verify_chain(), mirroring PermissionEscalationAuditStore exactly."""

    def __init__(
        self,
        db_file: Path | None = None,
        key_file: Path | None = None,
        key: bytes | None = None,
    ) -> None:
        self._db_file = db_file or AGENT_GATEWAY_AUDIT_DB_FILE
        self._key_file = key_file or AGENT_GATEWAY_AUDIT_KEY_FILE
        self._key = key

    async def initialize(self) -> None:
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        self._key = self._key or self._load_or_create_key()
        async with aiosqlite.connect(self._db_file) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_gateway_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    action_family TEXT NOT NULL,
                    target TEXT NOT NULL,
                    source_profile TEXT NOT NULL,
                    permission_tier TEXT NOT NULL,
                    override_applied INTEGER NOT NULL,
                    override_restricted INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    denial_reason TEXT NOT NULL,
                    dry_run INTEGER NOT NULL,
                    execution_success INTEGER,
                    execution_error TEXT NOT NULL,
                    policy_snapshot TEXT NOT NULL,
                    previous_hmac TEXT NOT NULL,
                    entry_hmac TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gateway_audit_plan_id
                ON agent_gateway_audit(plan_id)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gateway_audit_source
                ON agent_gateway_audit(source_profile)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gateway_audit_family
                ON agent_gateway_audit(action_family)
                """
            )
            await db.commit()

    async def record_event(
        self,
        *,
        plan_id: str,
        action_index: int,
        action_type: str,
        action_family: str,
        target: str,
        source_profile: str,
        permission_tier: str,
        override_applied: bool,
        override_restricted: bool,
        decision: str,
        denial_reason: str = "",
        dry_run: bool = False,
        execution_success: bool | None = None,
        execution_error: str = "",
        policy_snapshot: dict[str, Any] | None = None,
    ) -> str:
        target = redact_text(target) or ""
        execution_error = redact_text(execution_error) or ""
        await self.initialize()
        async with aiosqlite.connect(self._db_file) as db:
            previous_hmac = await self._last_hmac(db)
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "plan_id": plan_id,
                "action_index": action_index,
                "action_type": action_type,
                "action_family": action_family,
                "target": target,
                "source_profile": source_profile,
                "permission_tier": permission_tier,
                "override_applied": bool(override_applied),
                "override_restricted": bool(override_restricted),
                "decision": decision,
                "denial_reason": denial_reason,
                "dry_run": bool(dry_run),
                "execution_success": execution_success,
                "execution_error": execution_error,
                "policy_snapshot": policy_snapshot or {},
                "previous_hmac": previous_hmac,
            }
            entry_hmac = self._sign_payload(payload)
            await db.execute(
                """
                INSERT INTO agent_gateway_audit (
                    timestamp,
                    plan_id,
                    action_index,
                    action_type,
                    action_family,
                    target,
                    source_profile,
                    permission_tier,
                    override_applied,
                    override_restricted,
                    decision,
                    denial_reason,
                    dry_run,
                    execution_success,
                    execution_error,
                    policy_snapshot,
                    previous_hmac,
                    entry_hmac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["timestamp"],
                    plan_id,
                    action_index,
                    action_type,
                    action_family,
                    target,
                    source_profile,
                    permission_tier,
                    int(override_applied),
                    int(override_restricted),
                    decision,
                    denial_reason,
                    int(dry_run),
                    None if execution_success is None else int(execution_success),
                    execution_error,
                    self._json_dumps(policy_snapshot or {}),
                    previous_hmac,
                    entry_hmac,
                ),
            )
            await db.commit()
            return entry_hmac

    async def list_events(
        self,
        limit: int = 50,
        plan_id: str | None = None,
        source_profile: str | None = None,
        action_family: str | None = None,
        decision: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent gateway audit events for display, most
        recent first, optionally filtered by any combination of plan_id,
        source_profile, action_family, and decision."""
        await self.initialize()
        query = """
            SELECT id, timestamp, plan_id, action_index, action_type, action_family,
                   target, source_profile, permission_tier, override_applied,
                   override_restricted, decision, denial_reason, dry_run,
                   execution_success, execution_error, policy_snapshot
            FROM agent_gateway_audit
        """
        clauses: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("plan_id", plan_id),
            ("source_profile", source_profile),
            ("action_family", action_family),
            ("decision", decision),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                args.append(value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        args.append(limit)

        async with aiosqlite.connect(self._db_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, tuple(args)) as cursor:
                rows = await cursor.fetchall()

        events = []
        for row in rows:
            event = dict(row)
            event["override_applied"] = bool(event["override_applied"])
            event["override_restricted"] = bool(event["override_restricted"])
            event["dry_run"] = bool(event["dry_run"])
            if event["execution_success"] is not None:
                event["execution_success"] = bool(event["execution_success"])
            try:
                event["policy_snapshot"] = json.loads(event["policy_snapshot"]) if event["policy_snapshot"] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            events.append(event)
        return events

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
                    action_family,
                    target,
                    source_profile,
                    permission_tier,
                    override_applied,
                    override_restricted,
                    decision,
                    denial_reason,
                    dry_run,
                    execution_success,
                    execution_error,
                    policy_snapshot,
                    previous_hmac,
                    entry_hmac
                FROM agent_gateway_audit
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
                    action_family,
                    target,
                    source_profile,
                    permission_tier,
                    override_applied,
                    override_restricted,
                    decision,
                    denial_reason,
                    dry_run,
                    execution_success,
                    execution_error,
                    policy_snapshot,
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
                    "action_family": action_family,
                    "target": target,
                    "source_profile": source_profile,
                    "permission_tier": permission_tier,
                    "override_applied": bool(override_applied),
                    "override_restricted": bool(override_restricted),
                    "decision": decision,
                    "denial_reason": denial_reason,
                    "dry_run": bool(dry_run),
                    "execution_success": None if execution_success is None else bool(execution_success),
                    "execution_error": execution_error,
                    "policy_snapshot": json.loads(policy_snapshot),
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
            SELECT entry_hmac FROM agent_gateway_audit
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
            logger.warning("Unable to restrict agent gateway audit key file permissions", exc_info=True)
        return key

    def _sign_payload(self, payload: dict[str, Any]) -> str:
        if self._key is None:
            self._key = self._load_or_create_key()
        return hmac.new(self._key, self._json_dumps(payload).encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _json_dumps(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
