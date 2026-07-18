from __future__ import annotations

import aiosqlite
import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmptyParams
from pilot.config import PilotConfig
from pilot.security.gateway import AgentGateway, InvocationSource
from pilot.security.gateway_audit import AgentGatewayAuditStore
from pilot.security.permissions import PermissionChecker


async def _store(tmp_path):
    store = AgentGatewayAuditStore(
        db_file=tmp_path / "agent_gateway_audit.db",
        key_file=tmp_path / "agent_gateway_audit.key",
        key=b"0" * 32,
    )
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_gateway_audit_records_and_verifies_hmac_chain(tmp_path):
    store = await _store(tmp_path)

    first_hmac = await store.record_event(
        plan_id="plan-1",
        action_index=0,
        action_type="browser_navigate",
        action_family="browsing",
        target="https://example.com",
        source_profile="autonomous",
        permission_tier="SYSTEM_MODIFY",
        override_applied=False,
        override_restricted=False,
        decision="allowed",
        execution_success=True,
    )
    second_hmac = await store.record_event(
        plan_id="plan-1",
        action_index=1,
        action_type="browser_execute_js",
        action_family="browsing",
        target="document.cookie",
        source_profile="autonomous",
        permission_tier="DESTRUCTIVE",
        override_applied=False,
        override_restricted=False,
        decision="denied",
        denial_reason="browser_execute_js is denied for source 'autonomous' by gateway policy.",
        policy_snapshot={"deny_action_types": ["browser_execute_js"]},
    )

    assert first_hmac != second_hmac
    verification = await store.verify_chain()
    assert verification.valid is True
    assert verification.checked_entries == 2


@pytest.mark.asyncio
async def test_gateway_audit_list_events_filters_by_source_and_family(tmp_path):
    store = await _store(tmp_path)
    await store.record_event(
        plan_id="plan-a",
        action_index=0,
        action_type="shell_command",
        action_family="shell",
        target="ls",
        source_profile="autonomous",
        permission_tier="SYSTEM_MODIFY",
        override_applied=False,
        override_restricted=False,
        decision="allowed",
    )
    await store.record_event(
        plan_id="plan-b",
        action_index=0,
        action_type="mouse_click",
        action_family="system_control",
        target="100,200",
        source_profile="gesture",
        permission_tier="USER_WRITE",
        override_applied=True,
        override_restricted=True,
        decision="allowed",
    )

    shell_events = await store.list_events(action_family="shell")
    assert len(shell_events) == 1
    assert shell_events[0]["action_type"] == "shell_command"

    gesture_events = await store.list_events(source_profile="gesture")
    assert len(gesture_events) == 1
    assert gesture_events[0]["override_restricted"] is True


@pytest.mark.asyncio
async def test_gateway_audit_detects_tampered_rows(tmp_path):
    store = await _store(tmp_path)
    await store.record_event(
        plan_id="plan-2",
        action_index=0,
        action_type="registry_write",
        action_family="system_control",
        target="HKCU\\Software\\Foo",
        source_profile="autonomous",
        permission_tier="SYSTEM_MODIFY",
        override_applied=False,
        override_restricted=False,
        decision="denied",
        denial_reason="registry_write is denied for source 'autonomous' by gateway policy.",
    )

    async with aiosqlite.connect(tmp_path / "agent_gateway_audit.db") as db:
        await db.execute(
            """
            UPDATE agent_gateway_audit
            SET target = ?
            WHERE id = 1
            """,
            ("HKLM\\SYSTEM",),
        )
        await db.commit()

    verification = await store.verify_chain()
    assert verification.valid is False
    assert verification.checked_entries == 1
    assert "entry_hmac mismatch" in verification.error


@pytest.mark.asyncio
async def test_gateway_audit_key_persists_across_instances(tmp_path):
    db_file = tmp_path / "agent_gateway_audit.db"
    key_file = tmp_path / "agent_gateway_audit.key"

    store1 = AgentGatewayAuditStore(db_file=db_file, key_file=key_file)
    await store1.initialize()
    await store1.record_event(
        plan_id="plan-3",
        action_index=0,
        action_type="shell_command",
        action_family="shell",
        target="ls",
        source_profile="interactive",
        permission_tier="SYSTEM_MODIFY",
        override_applied=False,
        override_restricted=False,
        decision="allowed",
    )

    # A fresh instance loading the same key file must verify the same chain.
    store2 = AgentGatewayAuditStore(db_file=db_file, key_file=key_file)
    verification = await store2.verify_chain()
    assert verification.valid is True
    assert verification.checked_entries == 1


@pytest.mark.asyncio
async def test_agent_gateway_authorize_writes_audit_events(tmp_path):
    """AgentGateway.authorize() itself must actually populate the audit
    store when one is attached — not just be a store that nothing writes to."""
    config = PilotConfig()
    permissions = PermissionChecker(config)
    audit_store = await _store(tmp_path)
    gateway = AgentGateway(config, permissions, audit_store=audit_store)

    plan = ActionPlan(
        actions=[Action(action_type=ActionType.BROWSER_EXECUTE_JS, target="page", parameters=EmptyParams())],
        raw_input="test",
    )

    decision = await gateway.authorize(plan, InvocationSource.AUTONOMOUS, plan_id="plan-xyz")
    assert decision.allowed is False

    events = await audit_store.list_events(plan_id="plan-xyz")
    assert len(events) == 1
    assert events[0]["decision"] == "denied"
    assert events[0]["action_type"] == "browser_execute_js"
    assert events[0]["source_profile"] == "autonomous"

    verification = await audit_store.verify_chain()
    assert verification.valid is True
