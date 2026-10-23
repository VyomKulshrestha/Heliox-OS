"""Signed task receipts: hashed details, Ed25519 signatures, a hash chain, and honest verification."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, EmptyParams, FileParams
from pilot.config import PilotConfig, ReceiptsConfig
from pilot.workflows.task_receipts import TaskReceipts, _main, verify_envelope


class FakeVault:
    def __init__(self) -> None:
        self.available = True
        self.keys: dict[str, str] = {}

    async def get_key(self, name: str) -> str | None:
        return self.keys.get(name)

    async def store_key(self, name: str, value: str) -> None:
        self.keys[name] = value


def _write_action(path: str = "notes.txt", content: str = "secret plan") -> Action:
    return Action(action_type=ActionType.FILE_WRITE, target=path, parameters=FileParams(path=path, content=content))


def _record(receipts: TaskReceipts, task: str = "task-1", *, output: str = "Written 11 bytes") -> None:
    action = _write_action()
    receipts.record_action(
        task,
        "plan-1",
        action,
        ActionResult(action=action, success=True, output=output),
        user_confirmed=True,
        undo_kind="undoable",
    )


async def _finish(receipts: TaskReceipts, task: str = "task-1", status: str = "success") -> dict | None:
    return await receipts.finalize(task, request="write my notes", status=status, summary="Done.", source="voice")


@pytest.fixture
def receipts(tmp_path) -> TaskReceipts:
    return TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=FakeVault())


@pytest.mark.asyncio
async def test_receipt_is_signed_and_verifies(receipts):
    _record(receipts)
    short = await _finish(receipts)

    assert short["signed"] is True and short["action_count"] == 1
    envelope = await receipts.get(short["receipt_id"])
    result = await receipts.verify(envelope)
    assert result["valid"] is True
    assert result["this_computer"] is True
    action = envelope["receipt"]["actions"][0]
    assert action["user_confirmed"] is True and action["undo"] == "undoable"
    assert envelope["receipt"]["request"] == "write my notes"


@pytest.mark.asyncio
async def test_contents_and_outputs_are_only_hashed(receipts, tmp_path):
    _record(receipts, output="the private output text")
    await _finish(receipts)

    stored = next((tmp_path / "receipts").glob("*.json")).read_text(encoding="utf-8")
    assert "secret plan" not in stored
    assert "the private output text" not in stored
    assert "parameters_sha256" in stored and "output_sha256" in stored


@pytest.mark.asyncio
async def test_any_edit_breaks_the_signature(receipts):
    _record(receipts)
    short = await _finish(receipts)
    envelope = await receipts.get(short["receipt_id"])

    envelope["receipt"]["status"] = "error"
    result = verify_envelope(envelope)

    assert result["valid"] is False
    assert "changed after it was signed" in result["reasons"][0]


@pytest.mark.asyncio
async def test_removing_a_receipt_breaks_the_chain(receipts, tmp_path):
    for task in ("a", "b", "c"):
        _record(receipts, task)
        await _finish(receipts, task)
        time.sleep(0.002)
    assert (await receipts.verify_chain())["valid"] is True

    files = sorted((tmp_path / "receipts").glob("*.json"))
    files[1].unlink()
    chain = await receipts.verify_chain()

    assert chain["valid"] is False
    assert "changed or removed" in chain["problems"][0]["reason"]


@pytest.mark.asyncio
async def test_tasks_without_actions_get_no_receipt(receipts):
    assert await _finish(receipts, "nothing-ran") is None


@pytest.mark.asyncio
async def test_key_file_is_reused_when_no_keyring(tmp_path):
    first = TaskReceipts(tmp_path / "r", key_dir=tmp_path / "config")
    second = TaskReceipts(tmp_path / "r", key_dir=tmp_path / "config")

    one, two = await first.key_info(), await second.key_info()

    assert one["available"] and one["fingerprint"] == two["fingerprint"]
    assert (tmp_path / "config" / "task_receipt_signing_key").is_file()


@pytest.mark.asyncio
async def test_without_any_key_store_the_receipt_says_it_is_unsigned(tmp_path):
    receipts = TaskReceipts(tmp_path / "r")
    _record(receipts)
    short = await _finish(receipts)

    envelope = await receipts.get(short["receipt_id"])
    assert short["signed"] is False
    assert verify_envelope(envelope)["valid"] is False
    assert envelope["unsigned_reason"]


@pytest.mark.asyncio
async def test_a_receipt_from_another_key_is_not_this_computers(receipts, tmp_path):
    other = TaskReceipts(tmp_path / "other", vault=FakeVault())
    _record(other)
    short = await _finish(other)
    envelope = await other.get(short["receipt_id"])

    result = await receipts.verify(envelope)

    assert result["valid"] is True
    assert result["this_computer"] is False


@pytest.mark.asyncio
async def test_disabled_receipts_record_nothing(tmp_path):
    receipts = TaskReceipts(tmp_path / "r", config=ReceiptsConfig(enabled=False), vault=FakeVault())
    _record(receipts)
    assert await _finish(receipts) is None


@pytest.mark.asyncio
async def test_prune_keeps_the_newest_receipt(receipts, tmp_path):
    for task in ("a", "b"):
        _record(receipts, task)
        await _finish(receipts, task)
        time.sleep(0.002)
    old = time.time() - 400 * 86400
    for path in (tmp_path / "receipts").glob("*.json"):
        os.utime(path, (old, old))

    assert await receipts.prune() == 1
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_command_line_verifier(receipts, tmp_path, capsys):
    _record(receipts)
    short = await _finish(receipts)
    path = tmp_path / "exported.json"
    path.write_text(json.dumps(await receipts.get(short["receipt_id"])), encoding="utf-8")

    assert _main(["verify", str(path)]) == 0
    assert "Valid receipt" in capsys.readouterr().out

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["receipt"]["actions"] = []
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert _main(["verify", str(path)]) == 1


def test_receipt_store_is_off_limits_to_actions(monkeypatch, tmp_path):
    import pilot.config
    from pilot.security.validator import inside_heliox_private_dirs

    monkeypatch.setattr(pilot.config, "DATA_DIR", tmp_path / "data")

    assert inside_heliox_private_dirs(str(tmp_path / "data" / "receipts" / "x.json"))


@pytest.mark.asyncio
async def test_executor_adds_executed_actions_to_the_receipt(tmp_path):
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.gateway import InvocationSource
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    receipts = TaskReceipts(tmp_path / "receipts", vault=FakeVault())
    executor.set_task_receipts(receipts)
    plan = ActionPlan(
        actions=[Action(action_type=ActionType.CPU_USAGE, target="", parameters=EmptyParams())], raw_input="cpu"
    )

    await executor.execute(plan, invocation_source=InvocationSource.INTERACTIVE, plan_id="plan-r")
    short = await receipts.finalize("plan-r", request="cpu?", status="success", summary="ok", source="text")

    assert short is not None and short["action_count"] == 1
    envelope = await receipts.get(short["receipt_id"])
    assert envelope["receipt"]["actions"][0]["action_type"] == "cpu_usage"
    assert envelope["receipt"]["plan_ids"] == ["plan-r"]


@pytest.mark.asyncio
async def test_background_job_end_adds_a_receipt(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._task_receipts = TaskReceipts(tmp_path / "receipts", vault=FakeVault())
    server._broadcast_notification = AsyncMock()
    _record(server._task_receipts, "job-1")

    await server._autonomous_broadcast(
        "autonomous_complete", {"job_id": "job-1", "goal": "tidy notes", "status": "completed", "result_summary": "ok"}
    )

    method, params = server._broadcast_notification.await_args.args
    assert method == "autonomous_complete"
    assert params["receipt_id"]
    verified = await server._handle_receipt_verify({"receipt_id": params["receipt_id"]}, MagicMock())
    assert verified["result"]["valid"] is True


def test_receipts_settings_are_validated():
    import asyncio

    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    bad = asyncio.run(server._handle_update_config({"section": "receipts", "values": {"enabled": "yes"}}, MagicMock()))
    assert bad["status"] == "error"
    good = asyncio.run(
        server._handle_update_config({"section": "receipts", "values": {"retention_days": 30}}, MagicMock())
    )
    assert good["status"] == "ok" and config.receipts.retention_days == 30


@pytest.mark.asyncio
async def test_what_did_you_do_today_is_answered_from_receipts(receipts):
    _record(receipts, "a")
    await receipts.finalize("a", request="tidy my downloads", status="success", summary="ok", source="voice")
    _record(receipts, "b")
    await receipts.finalize("b", request="email the report", status="error", summary="smtp down", source="text")

    answer = receipts.answer("What did you do today?")

    assert answer.startswith("Today I ran 2 tasks.")
    assert '"tidy my downloads" (1 action)' in answer and '"email the report" (1 action)' in answer
    assert "1 did not finish cleanly" in answer
    assert receipts.answer("what is the weather") is None


def test_nothing_ran_is_said_plainly(tmp_path):
    from datetime import datetime

    receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())

    assert receipts.answer("what did you do yesterday", now=datetime(2026, 10, 23, 12)) == (
        "I didn't run any actions yesterday."
    )


@pytest.mark.asyncio
async def test_local_answers_try_the_timeline_then_receipts(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._task_receipts = TaskReceipts(tmp_path / "r", vault=FakeVault())
    _record(server._task_receipts, "a")
    await server._task_receipts.finalize("a", request="tidy", status="success", summary="ok", source="voice")

    assert (await server._local_answer("what did you do today")).startswith("Today I ran 1 task.")
    assert await server._local_answer("open the report") is None
