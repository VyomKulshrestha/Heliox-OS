"""A target that is not there is said as found ("There was no file at ..."), never as done, from the executor's
result through the narration, the reply and the task receipt. Fakes only: nothing is deleted, stopped or closed."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    DiskManageParams,
    FileParams,
    PackageParams,
    ProcessParams,
    WindowParams,
)
from pilot.agents.executor import Executor
from pilot.agents.narrator import ExecutionNarrator
from pilot.agents.verifier import Verifier
from pilot.config import PilotConfig, ReceiptsConfig
from pilot.response_contract import success_message
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system.platform_detect import Platform
from pilot.workflows.task_receipts import TaskReceipts

SUCCESS_VERBS = ("Done", "Already satisfied", "successfully", "Killed", "Closed window", "Unmounted ")


class _Vault:
    available = True

    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    async def get_key(self, name: str) -> str | None:
        return self.keys.get(name)

    async def store_key(self, name: str, value: str) -> None:
        self.keys[name] = value


def _absent(monkeypatch: pytest.MonkeyPatch, module: str, observer: str) -> None:
    imported = __import__(module, fromlist=[observer])
    monkeypatch.setattr(imported, observer, AsyncMock(return_value=False))


CASES = [
    (
        ActionType.FILE_DELETE,
        lambda tmp: FileParams(path=str(tmp / "old-report.txt")),
        None,
        lambda tmp: f"There was no file at {tmp / 'old-report.txt'}, so nothing was deleted",
    ),
    (
        ActionType.PROCESS_KILL,
        lambda tmp: ProcessParams(name="backup-agent"),
        ("pilot.system.processes", "process_exists"),
        lambda tmp: "backup-agent was not running, so nothing was stopped",
    ),
    (
        ActionType.PACKAGE_REMOVE,
        lambda tmp: PackageParams(name="cowsay"),
        ("pilot.system.package_mgr", "is_installed"),
        lambda tmp: "cowsay is not installed, so nothing was removed",
    ),
    (
        ActionType.WINDOW_CLOSE,
        lambda tmp: WindowParams(title="Quarterly plan"),
        ("pilot.system.window_mgr", "window_exists"),
        lambda tmp: "Quarterly plan was not open, so nothing was closed",
    ),
    (
        ActionType.DISK_UNMOUNT,
        lambda tmp: DiskManageParams(mount_point="/mnt/backup"),
        ("pilot.system.disks", "mount_exists"),
        lambda tmp: "/mnt/backup was not mounted, so nothing was unmounted",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("action_type", "parameters", "observer", "expected"), CASES)
async def test_an_absent_target_is_said_as_found_everywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path, action_type, parameters, observer, expected
) -> None:
    if observer is not None:
        _absent(monkeypatch, *observer)
    sentence = expected(tmp_path)
    built = parameters(tmp_path)
    action = Action(action_type=action_type, target=getattr(built, "path", None) or "the target", parameters=built)
    plan = ActionPlan(actions=[action], explanation="tidy up")
    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    executor.set_step_checker(Verifier(MagicMock()))
    # Never the real snapshot backend (on Windows it would ask for a System Restore point).
    executor._snapshot_mgr = SimpleNamespace(
        create_file_snapshot=AsyncMock(return_value="snap-1"), create_snapshot=AsyncMock(return_value="snap-1")
    )
    executor._execute_single = AsyncMock(side_effect=AssertionError("an absent target must not be acted on"))

    results = await executor.execute(plan, user_confirmed=True)
    [result] = results

    assert result.success is True and result.executed is False, result
    assert result.skip_reason == "already_absent"
    assert result.output == sentence
    assert result.notice == sentence

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)
    assert sentence in reply
    assert reply.count(sentence) == 1
    assert not any(verb in reply for verb in SUCCESS_VERBS)

    config.narration.enabled = True
    narrated: list[dict] = []

    async def broadcast(_event: str, payload: dict) -> None:
        narrated.append(payload)

    await ExecutionNarrator(config, {}, broadcast).on_action_complete(result)
    assert narrated[-1]["text"] == sentence

    receipts = TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=_Vault())
    receipts.record_action("task-1", "plan-1", action, result, user_confirmed=True)
    short = await receipts.finalize("task-1", request="tidy up", status="success", summary=reply, source="voice")
    recorded = (await receipts.get(short["receipt_id"]))["receipt"]["actions"][0]
    assert recorded["notice"] == sentence
    assert recorded["skip_reason"] == "already_absent"
    assert recorded["executed"] is False


async def _run(action_type: ActionType, handler: str, parameters: object) -> ActionResult:
    executor = object.__new__(Executor)
    executor._stress_gate = None
    executor._dispatch_table = {action_type: getattr(executor, handler)}
    action = Action(action_type=action_type, target="test", parameters=parameters)
    return await executor._execute_single(action, snapshot_id=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "handler", "verb"),
    [
        (ActionType.WINDOW_CLOSE, "_exec_window_close", "closed"),
        (ActionType.WINDOW_MINIMIZE, "_exec_window_minimize", "minimized"),
        (ActionType.WINDOW_MAXIMIZE, "_exec_window_maximize", "maximized"),
    ],
)
async def test_windows_window_action_with_no_matching_title_says_nothing_happened(
    monkeypatch: pytest.MonkeyPatch, action_type: ActionType, handler: str, verb: str
) -> None:
    from pilot.system import window_mgr

    async def no_match(*_args: object, **_kwargs: object) -> tuple[int, str, str]:
        return 0, f"{window_mgr.NOTHING_MATCHED}\n", ""

    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", no_match)

    result = await _run(action_type, handler, WindowParams(title="Quarterly plan"))

    sentence = f"Quarterly plan was not open, so nothing was {verb}"
    assert result.success is True
    assert result.output == sentence and result.notice == sentence
    assert result.skip_reason == "already_absent"


@pytest.mark.asyncio
async def test_windows_window_close_that_matched_still_says_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import window_mgr

    async def closed(*_args: object, **_kwargs: object) -> tuple[int, str, str]:
        return 0, "", ""

    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(window_mgr, "run_powershell", closed)

    result = await _run(ActionType.WINDOW_CLOSE, "_exec_window_close", WindowParams(title="Quarterly plan"))

    assert result.output == "Closed window: Quarterly plan"
    assert result.skip_reason is None and result.notice is None


@pytest.mark.asyncio
async def test_closing_a_tab_when_none_is_open_says_nothing_happened(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import browser

    async def no_tabs() -> object:
        return SimpleNamespace(pages=[])

    monkeypatch.setattr(browser, "_ensure_browser", no_tabs)

    output = await browser.browser_close_tab()

    assert str(output) == "No browser tab was open, so nothing was closed"
    assert output.nothing_done is True


def test_the_wording_rule_does_not_touch_trigger_or_schedule_deletes() -> None:
    from pilot.agents.verifier import ALREADY_ABSENT_WORDING

    assert ActionType.TRIGGER_DELETE not in ALREADY_ABSENT_WORDING
    assert ActionType.SCHEDULE_DELETE not in ALREADY_ABSENT_WORDING
