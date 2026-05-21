"""Tests for non-blocking audit log writes."""

from __future__ import annotations

import asyncio
import json

import pytest

from pilot.actions import Action, ActionResult, ActionType, EmptyParams
from pilot.security.audit import AuditLogger


def _action() -> Action:
    return Action(
        action_type=ActionType.SYSTEM_INFO,
        target="system",
        parameters=EmptyParams(),
    )


@pytest.mark.asyncio
async def test_audit_logger_writes_action_events_as_jsonl(tmp_path):
    audit_file = tmp_path / "nested" / "audit.jsonl"
    logger = AuditLogger(audit_file)
    action = _action()

    await logger.log_action_start(action, "plan-1")
    await logger.log_action_result(ActionResult(action=action, success=True, output="ok"), "plan-1")

    lines = audit_file.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    assert [entry["event_type"] for entry in entries] == ["action_start", "action_complete"]
    assert entries[0]["details"]["plan_id"] == "plan-1"
    assert entries[1]["success"] is True


@pytest.mark.asyncio
async def test_audit_logger_serializes_concurrent_writes(tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(audit_file)

    await asyncio.gather(
        logger.log_security_event("one"),
        logger.log_security_event("two"),
        logger.log_security_event("three"),
    )

    events = [json.loads(line)["details"]["event"] for line in audit_file.read_text(encoding="utf-8").splitlines()]

    assert sorted(events) == ["one", "three", "two"]
