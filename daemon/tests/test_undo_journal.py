"""Per-task undo: capture before each local change, restore newest first, never overwrite later edits."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ClipboardParams,
    EmptyParams,
    FileParams,
)
from pilot.config import PilotConfig, UndoConfig, _config_to_dict, _merge_config
from pilot.workflows.undo_journal import UndoJournal, undo_support


def _file_action(action_type: ActionType, path, destination=None, content=None, recursive=False) -> Action:
    return Action(
        action_type=action_type,
        target=str(path),
        parameters=FileParams(
            path=str(path),
            destination=str(destination) if destination is not None else None,
            content=content,
            recursive=recursive,
        ),
    )


def _ok(action: Action) -> ActionResult:
    return ActionResult(action=action, success=True, output="done")


async def _run(journal: UndoJournal, action: Action, effect, *, task="task-1", success=True) -> None:
    token = await journal.before(action, task_key=task)
    if success:
        effect()
        await journal.after(token, _ok(action))
    else:
        await journal.after(token, ActionResult(action=action, success=False, error="boom"))


async def _undo(journal: UndoJournal, task="task-1") -> dict:
    preview = await journal.preview(task)
    assert preview is not None
    return await journal.undo(task, preview["preview_token"])


@pytest.fixture
def journal(tmp_path) -> UndoJournal:
    return UndoJournal(tmp_path / "undo-store", config=UndoConfig())


def test_support_is_decided_by_action_type():
    write = _file_action(ActionType.FILE_WRITE, "a.txt", content="x")
    email = Action(action_type=ActionType.API_SEND_EMAIL, target="x", parameters=EmptyParams())
    status = Action(action_type=ActionType.CPU_USAGE, target="", parameters=EmptyParams())
    shell = Action(action_type=ActionType.SHELL_COMMAND, target="ls", parameters=EmptyParams())

    assert undo_support(write).kind == "undoable"
    assert undo_support(email).kind == "not_undoable"
    assert "outside your computer" in undo_support(email).reason
    assert undo_support(status).kind == "no_change"
    assert undo_support(shell).kind == "not_undoable"


@pytest.mark.asyncio
async def test_overwritten_file_comes_back_and_created_file_goes(journal, tmp_path):
    existing = tmp_path / "notes.txt"
    existing.write_text("original", encoding="utf-8")
    created = tmp_path / "new" / "draft.txt"

    await _run(
        journal,
        _file_action(ActionType.FILE_WRITE, existing, content="changed"),
        lambda: existing.write_text("changed"),
    )
    await _run(
        journal,
        _file_action(ActionType.FILE_WRITE, created, content="hello"),
        lambda: (created.parent.mkdir(parents=True), created.write_text("hello")),
    )

    preview = await journal.preview("task-1")
    assert [item["state"] for item in preview["items"]] == ["ready", "ready"]
    result = await journal.undo("task-1", preview["preview_token"])

    assert result["status"] == "ok" and result["undone"] == 2
    assert existing.read_text(encoding="utf-8") == "original"
    assert not created.exists()
    summary = await journal.summary("task-1")
    assert summary["available"] is False


@pytest.mark.asyncio
async def test_deleted_file_and_folder_are_restored(journal, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("keep me", encoding="utf-8")
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.jpg").write_bytes(b"\x01\x02")

    await _run(journal, _file_action(ActionType.FILE_DELETE, doc), doc.unlink)
    await _run(
        journal,
        _file_action(ActionType.FILE_DELETE, folder, recursive=True),
        lambda: __import__("shutil").rmtree(folder),
    )

    result = await _undo(journal)

    assert result["undone"] == 2
    assert doc.read_text(encoding="utf-8") == "keep me"
    assert (folder / "sub" / "a.jpg").read_bytes() == b"\x01\x02"


@pytest.mark.asyncio
async def test_move_goes_back_and_replaced_file_returns(journal, tmp_path):
    import shutil

    source = tmp_path / "report.txt"
    source.write_text("report", encoding="utf-8")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    replaced_source = tmp_path / "new.txt"
    replaced_source.write_text("new", encoding="utf-8")
    replaced = tmp_path / "old.txt"
    replaced.write_text("old", encoding="utf-8")

    await _run(
        journal,
        _file_action(ActionType.FILE_MOVE, source, destination=inbox),
        lambda: shutil.move(str(source), str(inbox)),
    )
    await _run(
        journal,
        _file_action(ActionType.FILE_MOVE, replaced_source, destination=replaced),
        lambda: shutil.move(str(replaced_source), str(replaced)),
    )

    result = await _undo(journal)

    assert result["undone"] == 2
    assert source.read_text(encoding="utf-8") == "report"
    assert not (inbox / "report.txt").exists()
    assert replaced_source.read_text(encoding="utf-8") == "new"
    assert replaced.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_copy_is_removed(journal, tmp_path):
    import shutil

    source = tmp_path / "a.txt"
    source.write_text("a", encoding="utf-8")
    target_dir = tmp_path / "backup"
    target_dir.mkdir()

    await _run(
        journal,
        _file_action(ActionType.FILE_COPY, source, destination=target_dir),
        lambda: shutil.copy2(source, target_dir),
    )

    result = await _undo(journal)

    assert result["undone"] == 1
    assert not (target_dir / "a.txt").exists()
    assert source.read_text(encoding="utf-8") == "a"


@pytest.mark.asyncio
async def test_later_edits_are_never_overwritten(journal, tmp_path):
    path = tmp_path / "plan.md"
    path.write_text("v1", encoding="utf-8")

    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="v2"), lambda: path.write_text("v2"))
    path.write_text("my own edit", encoding="utf-8")

    preview = await journal.preview("task-1")
    assert preview["items"][0]["state"] == "conflict"
    assert preview["ready"] == 0
    result = await journal.undo("task-1", preview["preview_token"])

    assert result["undone"] == 0 and result["skipped"] == 1
    assert path.read_text(encoding="utf-8") == "my own edit"


@pytest.mark.asyncio
async def test_two_writes_to_one_file_unwind_to_the_original(journal, tmp_path):
    path = tmp_path / "todo.txt"
    path.write_text("v1", encoding="utf-8")

    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="v2"), lambda: path.write_text("v2"))
    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="v3"), lambda: path.write_text("v3"))

    preview = await journal.preview("task-1")
    assert [item["state"] for item in preview["items"]] == ["ready", "ready"]
    await journal.undo("task-1", preview["preview_token"])

    assert path.read_text(encoding="utf-8") == "v1"


@pytest.mark.asyncio
async def test_stale_preview_is_refused(journal, tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("one", encoding="utf-8")
    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="two"), lambda: path.write_text("two"))
    preview = await journal.preview("task-1")

    path.write_text("three", encoding="utf-8")
    result = await journal.undo("task-1", preview["preview_token"])

    assert result["status"] == "changed"
    assert path.read_text(encoding="utf-8") == "three"


@pytest.mark.asyncio
async def test_failed_action_leaves_no_entry_and_no_copy(journal, tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("one", encoding="utf-8")

    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="two"), lambda: None, success=False)

    preview = await journal.preview("task-1")
    assert preview["items"] == []
    assert not list((tmp_path / "undo-store").rglob("blobs/*"))


@pytest.mark.asyncio
async def test_unsupported_and_external_actions_are_listed(journal):
    shell = Action(action_type=ActionType.SHELL_COMMAND, target="rm -rf build", parameters=EmptyParams())
    email = Action(action_type=ActionType.API_SEND_EMAIL, target="boss@example.com", parameters=EmptyParams())
    status = Action(action_type=ActionType.CPU_USAGE, target="", parameters=EmptyParams())

    for action in (shell, email, status):
        await _run(journal, action, lambda: None)

    preview = await journal.preview("task-1")
    assert [row["target"] for row in preview["not_undoable"]] == ["rm -rf build", "boss@example.com"]
    assert preview["ready"] == 0


@pytest.mark.asyncio
async def test_copy_too_large_for_the_task_limit_is_not_undoable(tmp_path):
    journal = UndoJournal(tmp_path / "undo-store", config=UndoConfig(max_task_mb=1))
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (2 * 1024 * 1024))

    await _run(journal, _file_action(ActionType.FILE_DELETE, big), big.unlink)

    preview = await journal.preview("task-1")
    assert preview["ready"] == 0
    assert "larger than the undo limit" in preview["not_undoable"][0]["reason"]


@pytest.mark.asyncio
async def test_paths_inside_heliox_folders_are_refused(journal, tmp_path, monkeypatch):
    import pilot.config

    private = tmp_path / "heliox-config"
    private.mkdir()
    secret = private / "config.toml"
    secret.write_text("token", encoding="utf-8")
    monkeypatch.setattr(pilot.config, "CONFIG_DIR", private)

    await _run(journal, _file_action(ActionType.FILE_DELETE, secret), secret.unlink)

    preview = await journal.preview("task-1")
    assert preview["ready"] == 0
    assert not secret.exists()


def test_undo_store_is_off_limits_to_actions(monkeypatch, tmp_path):
    import pilot.config
    from pilot.security.validator import inside_heliox_private_dirs

    monkeypatch.setattr(pilot.config, "DATA_DIR", tmp_path / "data")

    assert inside_heliox_private_dirs(str(tmp_path / "data" / "undo" / "t" / "journal.json"))
    assert not inside_heliox_private_dirs(str(tmp_path / "data" / "notes.txt"))


@pytest.mark.asyncio
async def test_clipboard_text_is_put_back_only_if_unchanged(journal, monkeypatch):
    board = {"text": "before"}

    async def read():
        return board["text"]

    async def write(content):
        board["text"] = content
        return "ok"

    monkeypatch.setattr("pilot.system.clipboard.clipboard_read", read)
    monkeypatch.setattr("pilot.system.clipboard.clipboard_write", write)
    action = Action(action_type=ActionType.CLIPBOARD_WRITE, target="", parameters=ClipboardParams(content="secret"))

    await _run(journal, action, lambda: board.update(text="secret"))
    result = await _undo(journal)

    assert result["undone"] == 1
    assert board["text"] == "before"

    await _run(journal, action, lambda: board.update(text="secret"), task="task-2")
    board["text"] = "copied by me"
    preview = await journal.preview("task-2")
    result = await journal.undo("task-2", preview["preview_token"])
    assert result["failed"] == 1
    assert board["text"] == "copied by me"


@pytest.mark.asyncio
async def test_disabled_journal_records_nothing(tmp_path):
    journal = UndoJournal(tmp_path / "undo-store", config=UndoConfig(enabled=False))
    path = tmp_path / "x.txt"

    assert await journal.before(_file_action(ActionType.FILE_WRITE, path, content="x"), task_key="t") is None
    assert await journal.preview("t") is None


@pytest.mark.asyncio
async def test_prune_drops_expired_and_oldest_history(tmp_path):
    journal = UndoJournal(tmp_path / "undo-store", config=UndoConfig(retention_days=1))
    path = tmp_path / "x.txt"
    path.write_text("a", encoding="utf-8")
    await _run(journal, _file_action(ActionType.FILE_WRITE, path, content="b"), lambda: path.write_text("b"))

    assert await journal.prune(now=time.time()) == 0
    assert await journal.prune(now=time.time() + 2 * 86400) == 1
    assert await journal.preview("task-1") is None


def test_undo_config_merges_and_clamps():
    config = PilotConfig()
    _merge_config(config, {"undo": {"enabled": False, "retention_days": 999, "max_task_mb": 0, "max_total_mb": 64}})

    assert config.undo.enabled is False
    assert config.undo.retention_days == 90
    assert config.undo.max_task_mb == 1
    assert config.undo.max_total_mb == 64
    assert _config_to_dict(PilotConfig())["undo"] == {
        "enabled": True,
        "retention_days": 7,
        "max_task_mb": 512,
        "max_total_mb": 2048,
    }


@pytest.mark.asyncio
async def test_executor_journals_real_file_writes(tmp_path):
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.gateway import InvocationSource
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    journal = UndoJournal(tmp_path / "undo-store", config=config.undo)
    executor.set_undo_journal(journal)
    path = tmp_path / "hello.txt"
    plan = ActionPlan(actions=[_file_action(ActionType.FILE_WRITE, path, content="hi")], raw_input="write hello")

    results = await executor.execute(
        plan, invocation_source=InvocationSource.INTERACTIVE, user_confirmed=True, plan_id="plan-a"
    )

    assert results[0].success is True
    preview = await journal.preview("plan-a")
    assert preview is not None and preview["ready"] == 1
    assert preview["plan_ids"] == ["plan-a"]
    await journal.undo("plan-a", preview["preview_token"])
    assert not path.exists()


@pytest.mark.asyncio
async def test_dry_run_journals_nothing(tmp_path):
    from pilot.agents.executor import Executor
    from pilot.security.audit import AuditLogger
    from pilot.security.gateway import InvocationSource
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    journal = UndoJournal(tmp_path / "undo-store", config=config.undo)
    executor.set_undo_journal(journal)
    plan = ActionPlan(
        actions=[_file_action(ActionType.FILE_WRITE, tmp_path / "x.txt", content="hi")], raw_input="write"
    )

    await executor.execute(
        plan, invocation_source=InvocationSource.INTERACTIVE, user_confirmed=True, plan_id="plan-d", dry_run=True
    )

    assert await journal.preview("plan-d") is None


@pytest.mark.asyncio
async def test_undo_rpc_needs_confirmation_and_an_idle_daemon(tmp_path):
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._undo_journal = UndoJournal(tmp_path / "undo-store", config=server.config.undo)
    path = tmp_path / "x.txt"
    path.write_text("a", encoding="utf-8")
    await _run(
        server._undo_journal, _file_action(ActionType.FILE_WRITE, path, content="b"), lambda: path.write_text("b")
    )
    preview = (await server._handle_undo_preview({"task_key": "task-1"}, MagicMock()))["preview"]
    server._broadcast_notification = MagicMock(side_effect=lambda *a, **k: _noop())

    for confirmed in (None, "true", 1):
        refused = await server._handle_undo_task(
            {"task_key": "task-1", "preview_token": preview["preview_token"], "confirmed": confirmed}, MagicMock()
        )
        assert refused["status"] == "error"

    server._interactive_request_active = True
    busy = await server._handle_undo_task(
        {"task_key": "task-1", "preview_token": preview["preview_token"], "confirmed": True}, MagicMock()
    )
    assert "running tasks" in busy["message"]
    assert path.read_text(encoding="utf-8") == "b"

    server._interactive_request_active = False
    done = await server._handle_undo_task(
        {"task_key": "task-1", "preview_token": preview["preview_token"], "confirmed": True}, MagicMock()
    )
    assert done["status"] == "ok"
    assert path.read_text(encoding="utf-8") == "a"
    assert server._broadcast_notification.call_args.args[0] == "undo_complete"


def test_approval_badge_reflects_the_setting():
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    write = _file_action(ActionType.FILE_WRITE, "a.txt", content="x")

    assert server._undo_badge(write)["kind"] == "undoable"
    server.config.undo.enabled = False
    assert server._undo_badge(write) == {"kind": "not_undoable", "reason": "Undo is turned off in Settings."}


async def _noop() -> None:
    return None


def test_update_config_validates_undo_values():
    import asyncio

    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)

    bad = asyncio.run(server._handle_update_config({"section": "undo", "values": {"retention_days": 0}}, MagicMock()))
    assert bad["status"] == "error"
    good = asyncio.run(server._handle_update_config({"section": "undo", "values": {"retention_days": 14}}, MagicMock()))
    assert good["status"] == "ok"
    assert config.undo.retention_days == 14


def _level_action(action_type: ActionType, level: int) -> Action:
    from pilot.actions import BrightnessParams, VolumeParams

    params = VolumeParams(level=level) if action_type == ActionType.VOLUME_SET else BrightnessParams(level=level)
    return Action(action_type=action_type, target=f"{level}%", parameters=params)


@pytest.fixture
def levels(monkeypatch):
    state = {"volume": 30, "brightness": 70}

    async def read(kind):
        return state[kind]

    async def write(kind, level):
        state[kind] = level

    monkeypatch.setattr("pilot.workflows.undo_journal._read_level", read)
    monkeypatch.setattr("pilot.workflows.undo_journal._write_level", write)
    return state


@pytest.mark.asyncio
async def test_volume_and_brightness_go_back_to_their_earlier_levels(journal, levels):
    await _run(journal, _level_action(ActionType.VOLUME_SET, 80), lambda: levels.update(volume=80))
    await _run(journal, _level_action(ActionType.VOLUME_SET, 50), lambda: levels.update(volume=50))
    await _run(journal, _level_action(ActionType.BRIGHTNESS_SET, 20), lambda: levels.update(brightness=20))

    preview = await journal.preview("task-1")
    assert [item["state"] for item in preview["items"]] == ["ready", "ready", "ready"]
    assert preview["items"][0]["description"] == "Set the brightness back to 70%."
    result = await journal.undo("task-1", preview["preview_token"])

    assert result["undone"] == 3
    assert levels == {"volume": 30, "brightness": 70}


@pytest.mark.asyncio
async def test_a_level_changed_since_is_left_alone(journal, levels):
    await _run(journal, _level_action(ActionType.VOLUME_SET, 80), lambda: levels.update(volume=80))
    levels["volume"] = 20

    preview = await journal.preview("task-1")

    assert preview["items"][0]["state"] == "conflict"
    await journal.undo("task-1", preview["preview_token"])
    assert levels["volume"] == 20


@pytest.mark.asyncio
async def test_an_unreadable_level_is_reported_not_guessed(journal, monkeypatch):
    async def unreadable(kind):
        return None

    monkeypatch.setattr("pilot.workflows.undo_journal._read_level", unreadable)
    await _run(journal, _level_action(ActionType.VOLUME_SET, 80), lambda: None)

    preview = await journal.preview("task-1")

    assert preview["items"][0]["state"] == "unavailable"
    assert "could not read the volume" in preview["items"][0]["description"]


def _env_action(name: str, value: str, persistent: bool = False) -> Action:
    from pilot.actions import EnvParams

    return Action(
        action_type=ActionType.ENV_SET, target=name, parameters=EnvParams(name=name, value=value, persistent=persistent)
    )


@pytest.mark.asyncio
async def test_session_environment_variable_is_put_back_or_removed(journal, monkeypatch):
    monkeypatch.setenv("HELIOX_UNDO_A", "before")
    monkeypatch.delenv("HELIOX_UNDO_B", raising=False)

    await _run(journal, _env_action("HELIOX_UNDO_A", "after"), lambda: os.environ.__setitem__("HELIOX_UNDO_A", "after"))
    await _run(journal, _env_action("HELIOX_UNDO_B", "new"), lambda: os.environ.__setitem__("HELIOX_UNDO_B", "new"))

    result = await _undo(journal)

    assert result["undone"] == 2
    assert os.environ["HELIOX_UNDO_A"] == "before"
    assert "HELIOX_UNDO_B" not in os.environ
    stored = next((journal._root).rglob("journal.json")).read_text(encoding="utf-8")
    assert "before" not in stored  # earlier values stay in memory only


@pytest.mark.asyncio
async def test_persistent_variable_restores_the_registry_value_and_its_type(journal, monkeypatch):
    registry = {"HELIOX_UNDO_R": ("C:/old", 2)}
    monkeypatch.setattr("pilot.workflows.undo_journal._uses_registry", lambda: True)
    monkeypatch.setattr("pilot.workflows.undo_journal._registry_env_get", lambda name: registry.get(name))
    monkeypatch.setattr(
        "pilot.workflows.undo_journal._registry_env_set",
        lambda name, value: registry.pop(name, None) if value is None else registry.__setitem__(name, value),
    )
    monkeypatch.delenv("HELIOX_UNDO_R", raising=False)

    def set_both():
        os.environ["HELIOX_UNDO_R"] = "C:\new"
        registry["HELIOX_UNDO_R"] = ("C:\new", 1)

    await _run(journal, _env_action("HELIOX_UNDO_R", "C:\new", persistent=True), set_both)
    result = await _undo(journal)

    assert result["undone"] == 1
    assert registry["HELIOX_UNDO_R"] == ("C:/old", 2)
    assert "HELIOX_UNDO_R" not in os.environ


@pytest.mark.asyncio
async def test_persistent_variable_restores_the_shell_profile(journal, monkeypatch, tmp_path):
    profile = tmp_path / ".bashrc"
    profile.write_text("alias ll='ls -l'\n", encoding="utf-8")
    monkeypatch.setattr("pilot.workflows.undo_journal._uses_registry", lambda: False)
    monkeypatch.setattr("pilot.workflows.undo_journal._profile_path", lambda: profile)
    monkeypatch.delenv("HELIOX_UNDO_P", raising=False)

    def set_both():
        os.environ["HELIOX_UNDO_P"] = "1"
        profile.write_text(profile.read_text(encoding="utf-8") + "export HELIOX_UNDO_P=1\n", encoding="utf-8")

    await _run(journal, _env_action("HELIOX_UNDO_P", "1", persistent=True), set_both)
    await _undo(journal)

    assert profile.read_text(encoding="utf-8") == "alias ll='ls -l'\n"
    assert "HELIOX_UNDO_P" not in os.environ


@pytest.mark.asyncio
async def test_environment_undo_is_unavailable_after_a_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("HELIOX_UNDO_C", raising=False)
    first = UndoJournal(tmp_path / "undo-store", config=UndoConfig())
    await _run(first, _env_action("HELIOX_UNDO_C", "x"), lambda: os.environ.__setitem__("HELIOX_UNDO_C", "x"))
    monkeypatch.delenv("HELIOX_UNDO_C", raising=False)

    restarted = UndoJournal(tmp_path / "undo-store", config=UndoConfig())
    preview = await restarted.preview("task-1")

    assert preview["items"][0]["state"] == "unavailable"
    assert "until Heliox restarts" in preview["items"][0]["description"]
