from __future__ import annotations

import aiosqlite
import pytest

from pilot.security.permission_audit import PermissionEscalationAuditStore


async def _store(tmp_path):
    store = PermissionEscalationAuditStore(
        db_file=tmp_path / "permission_audit.db",
        key_file=tmp_path / "permission_audit.key",
        key=b"0" * 32,
    )
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_permission_audit_records_and_verifies_hmac_chain(tmp_path):
    store = await _store(tmp_path)

    first_hmac = await store.record_event(
        plan_id="plan-1",
        action_index=0,
        action_type="service_restart",
        target="nginx",
        permission_tier="SYSTEM_MODIFY",
        requires_root=False,
        destructive=False,
        confirmation_decision="approved",
        critic_verdict={"verdict": "APPROVE", "risk_score": 0.2},
        execution_success=True,
    )
    second_hmac = await store.record_event(
        plan_id="plan-1",
        action_index=1,
        action_type="file_delete",
        target="/tmp/demo",
        permission_tier="DESTRUCTIVE",
        requires_root=False,
        destructive=True,
        confirmation_decision="approved",
        critic_verdict={"verdict": "WARN", "risk_score": 0.5},
        execution_success=False,
        execution_error="permission denied",
    )

    assert first_hmac != second_hmac
    verification = await store.verify_chain()
    assert verification.valid is True
    assert verification.checked_entries == 2


@pytest.mark.asyncio
async def test_permission_audit_detects_tampered_rows(tmp_path):
    store = await _store(tmp_path)
    await store.record_event(
        plan_id="plan-2",
        action_index=0,
        action_type="package_remove",
        target="openssl",
        permission_tier="DESTRUCTIVE",
        requires_root=True,
        destructive=True,
        confirmation_decision="denied",
        critic_verdict={"verdict": "BLOCK", "risk_score": 0.9},
        execution_success=None,
        execution_error="Plan was denied by user.",
    )

    async with aiosqlite.connect(tmp_path / "permission_audit.db") as db:
        await db.execute(
            """
            UPDATE permission_escalation_audit
            SET target = ?
            WHERE id = 1
            """,
            ("/etc/passwd",),
        )
        await db.commit()

    verification = await store.verify_chain()
    assert verification.valid is False
    assert verification.checked_entries == 1
    assert "entry_hmac mismatch" in verification.error
