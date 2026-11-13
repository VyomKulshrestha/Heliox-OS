"""A snapshot rollback runs only after the person approves it in a dialog the daemon asks for itself.

Until 13 November 2026 one rollback_plan call restored the snapshot, system-wide, with no approval asked by the
daemon: the window asked first in a dialog of its own, and any other holder of the window's token did not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionType
from pilot.config import PilotConfig
from pilot.server import PilotServer
from pilot.system import snapshots
from pilot.system.snapshots import SnapshotBackend, rollback_request

TAKEN = datetime(2026, 11, 12, 21, 30, tzinfo=UTC)


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id: str, plan: Any, ws: Any, **kwargs: Any):
        self.asked.append({"plan_id": plan_id, "actions": plan.actions, **kwargs})
        return self.confirmed, ({0} if self.confirmed else set()), {0}


@pytest.fixture
def restored(monkeypatch) -> list[str]:
    done: list[str] = []

    async def rollback(self, snapshot_id: str) -> str:
        done.append(snapshot_id)
        return "restored"

    async def backend(self):
        return SnapshotBackend.WINDOWS_RESTORE_POINT

    monkeypatch.setattr(snapshots.SnapshotManager, "rollback", rollback)
    monkeypatch.setattr(snapshots.SnapshotManager, "detect_backend", backend)
    return done


def _server(person: Person) -> PilotServer:
    server = PilotServer(PilotConfig())
    server._plan_snapshots["plan-1"] = "windows-restore:12"
    server._plan_snapshot_times["plan-1"] = TAKEN
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    return server


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_without_the_persons_approval_nothing_is_restored(restored):
    person = Person(confirmed=False)  # denied, or the approval expired unanswered
    server = _server(person)

    result = await server._handle_rollback_plan({"plan_id": "plan-1"}, _ws())

    assert result["status"] == "cancelled" and restored == []
    assert server._plan_snapshots["plan-1"] == "windows-restore:12"  # still there to ask about again


@pytest.mark.asyncio
async def test_the_daemon_asks_with_the_command_and_what_is_lost_since_when(restored):
    person = Person(confirmed=True)
    server = _server(person)

    result = await server._handle_rollback_plan({"plan_id": "plan-1"}, _ws())

    assert result["status"] == "ok" and restored == ["windows-restore:12"]
    asked = person.asked[0]
    assert asked["force_all_actions"] is True and asked["plan_id"] == "rollback-plan-1"
    assert "Restore-Computer -RestorePoint 12" in asked["actions"][0].target
    when = TAKEN.astimezone().strftime("%d %B %Y at %H:%M")
    assert when in asked["reason"] and "restarts" in asked["reason"]


@pytest.mark.asyncio
async def test_a_second_request_while_asking_is_refused(restored):
    server = _server(Person(confirmed=True))
    server._rollbacks_asking.add("plan-1")

    result = await server._handle_rollback_plan({"plan_id": "plan-1"}, _ws())

    assert result["status"] == "error" and restored == []


def test_each_system_shows_the_command_that_really_runs(monkeypatch):
    monkeypatch.setattr(snapshots.sys, "platform", "linux")
    timeshift, reason = rollback_request("2026-11-12_21-30-00", SnapshotBackend.TIMESHIFT, TAKEN)
    assert timeshift.target.startswith("pkexec timeshift --restore --snapshot 2026-11-12_21-30-00 --yes")
    assert "administrator's password" in reason
    btrfs, _ = rollback_request("/.snapshots/pilot-x", SnapshotBackend.BTRFS, TAKEN)
    assert btrfs.target == "pkexec btrfs subvolume snapshot /.snapshots/pilot-x /rollback-target"

    monkeypatch.setattr(snapshots.sys, "platform", "win32")
    windows, _ = rollback_request("windows-restore:7", SnapshotBackend.WINDOWS_RESTORE_POINT, TAKEN)
    assert windows.action_type == ActionType.SHELL_COMMAND and windows.target.startswith("powershell ")
    assert "pkexec" not in windows.target  # nothing elevates it on Windows

    files, reason = rollback_request("file-snapshot:abc", "", None)
    assert files.action_type == ActionType.FILE_WRITE and "just before that task ran" in reason

    with pytest.raises(RuntimeError):
        rollback_request("x", SnapshotBackend.NONE, TAKEN)  # macOS: no system snapshot to restore
