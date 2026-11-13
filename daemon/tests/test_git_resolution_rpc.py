"""apply_git_resolution writes only the person's choice for one real git conflict, from the window only.

Until 13 November 2026 any client holding the window's token could replace any text found in any file the validator
allowed with any text: ``full_block`` only had to occur somewhere in the file and ``resolved_code`` could be
anything. Each refusal below is one of the gaps closed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.actions import ActionResult, ActionType
from pilot.config import PilotConfig
from pilot.security.rpc_identity import RpcClientRole, rpc_method_allowed
from pilot.security.validator import ActionValidator
from pilot.server import PilotServer
from pilot.system.git_conflict import git_resolution_action, resolution_refusal
from pilot.workflows.undo_journal import undo_support

BLOCK = "<<<<<<< HEAD\nmine\n=======\ntheirs\n>>>>>>> branch\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    return tmp_path / "repo"


def conflicted(repo: Path, text: str = f"before\n{BLOCK}after\n") -> Path:
    path = repo / "app.py"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("side", ["mine\n", "theirs\n"])
def test_either_side_of_a_conflict_in_a_repository_may_be_written(repo, side):
    assert resolution_refusal(str(conflicted(repo)), BLOCK, side) is None


def test_the_resolution_heliox_proposed_may_be_written_and_no_other_text(repo):
    path = str(conflicted(repo))
    assert resolution_refusal(path, BLOCK, "merged\n", ["merged\n"]) is None
    assert "Only one side" in resolution_refusal(path, BLOCK, "import os; os.system('x')\n", ["merged\n"])


def test_any_text_that_is_not_a_whole_conflict_block_is_refused(repo):
    path = str(conflicted(repo))
    # Before: "before\n" was found in the file, so it could be replaced with anything.
    assert "not in the file" in resolution_refusal(path, "before\n", "mine\n")
    assert "not in the file" in resolution_refusal(path, BLOCK.replace("mine", "other"), "other\n")


def test_a_conflict_that_appears_twice_is_refused(repo):
    path = str(conflicted(repo, f"{BLOCK}middle\n{BLOCK}"))
    assert "more than once" in resolution_refusal(path, BLOCK, "mine\n")


def test_a_file_outside_a_repository_is_refused(tmp_path):
    path = tmp_path / "loose.txt"
    path.write_text(f"x\n{BLOCK}", encoding="utf-8")
    assert "not inside a git repository" in resolution_refusal(str(path), BLOCK, "mine\n")


def test_a_missing_file_a_folder_and_a_relative_path_are_refused(repo):
    assert "not a file that exists" in resolution_refusal(str(repo / "gone.py"), BLOCK, "mine\n")
    assert "not a file that exists" in resolution_refusal(str(repo), BLOCK, "mine\n")
    assert "full path" in resolution_refusal("app.py", BLOCK, "mine\n")
    assert "must be text" in resolution_refusal(str(conflicted(repo)), BLOCK, 5)


def test_a_protected_folder_is_refused_by_the_validator_as_for_any_file_write(repo):
    config = PilotConfig()
    config.restrictions.protected_folders = [str(repo)]
    action = git_resolution_action(str(conflicted(repo)), BLOCK, "mine\n")
    with pytest.raises(Exception, match="protected folder"):
        ActionValidator(config).validate_action(action, 0)


def test_the_write_is_undoable_as_a_file_write():
    action = git_resolution_action("C:/repo/app.py", BLOCK, "mine\n")
    assert action.action_type == ActionType.GIT_RESOLVE
    assert undo_support(action).kind == "undoable"


class RecordingExecutor:
    def __init__(self) -> None:
        self.plans: list = []

    async def execute(self, plan, **kwargs):
        self.plans.append((plan, kwargs))
        return [ActionResult(action=plan.actions[0], success=True, output="done")]


def server_for(role: RpcClientRole | None) -> tuple[PilotServer, object, RecordingExecutor]:
    server = PilotServer.__new__(PilotServer)
    ws = object()
    server._client_roles = {ws: role} if role is not None else {}
    server._executor = RecordingExecutor()
    return server, ws, server._executor


@pytest.mark.asyncio
async def test_the_window_writes_its_choice_through_the_executor(repo):
    server, ws, executor = server_for(RpcClientRole.UI)
    path = conflicted(repo)

    result = await server._handle_apply_git_resolution(
        {"path": str(path), "full_block": BLOCK, "resolved_code": "theirs\n"}, ws
    )

    assert result["status"] == "success"
    plan, kwargs = executor.plans[0]
    action = plan.actions[0]
    assert action.action_type == ActionType.GIT_RESOLVE and action.target == str(path.resolve())
    assert kwargs["plan_id"].startswith("git-resolution-")


@pytest.mark.asyncio
async def test_a_refused_resolution_never_reaches_the_executor(repo):
    server, ws, executor = server_for(RpcClientRole.UI)
    result = await server._handle_apply_git_resolution(
        {"path": str(conflicted(repo)), "full_block": "before\n", "resolved_code": "anything"}, ws
    )
    assert result["status"] == "error" and executor.plans == []


@pytest.mark.asyncio
async def test_a_proposal_from_resolve_git_conflict_is_accepted_for_its_own_block_only(repo):
    server, ws, executor = server_for(RpcClientRole.UI)
    path = conflicted(repo)
    server._git_proposals()[(str(path.resolve()), BLOCK)] = "merged\n"

    accepted = await server._handle_apply_git_resolution(
        {"path": str(path), "full_block": BLOCK, "resolved_code": "merged\n"}, ws
    )
    refused = await server._handle_apply_git_resolution(
        {"path": str(path), "full_block": BLOCK, "resolved_code": "something else\n"}, ws
    )

    assert accepted["status"] == "success" and refused["status"] == "error"
    assert len(executor.plans) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [RpcClientRole.NEURAL_SIDECAR, RpcClientRole.MCP_LOCAL, None])
async def test_no_other_client_role_may_write(repo, role):
    if role is not None:
        assert not rpc_method_allowed(role, "apply_git_resolution")  # refused at dispatch
    server, ws, executor = server_for(role)
    with pytest.raises(PermissionError):  # and by the handler itself, should an allow-list ever grow
        await server._handle_apply_git_resolution(
            {"path": str(conflicted(repo)), "full_block": BLOCK, "resolved_code": "mine\n"}, ws
        )
    assert executor.plans == []


def test_the_sides_offered_are_the_files_own_not_the_models(repo, monkeypatch):
    """resolve_git_conflict used to pass on a model's "original_hunk"; "keep mine" then wrote the model's text."""
    import asyncio

    from pilot.system import git_conflict

    path = conflicted(repo)
    router = SimpleNamespace(
        generate=lambda *a, **k: asyncio.sleep(
            0, result='{"original_hunk": "rm -rf\\n", "conflict_hunk": "x\\n", "proposed_resolution_code": "m\\n"}'
        )
    )
    blocks = asyncio.run(git_conflict.resolve_conflicts_in_file(str(path), router))
    assert blocks[0]["original_hunk"] == "mine\n" and blocks[0]["conflict_hunk"] == "theirs\n"
    assert blocks[0]["proposed_resolution_code"] == "m\n"


@pytest.mark.asyncio
async def test_undo_puts_the_conflicted_file_back(repo, tmp_path):
    from pilot.config import UndoConfig
    from pilot.workflows.undo_journal import UndoJournal

    journal = UndoJournal(tmp_path / "undo-store", config=UndoConfig())
    path = conflicted(repo)
    before = path.read_text(encoding="utf-8")
    action = git_resolution_action(str(path), BLOCK, "mine\n")

    token = await journal.before(action, task_key="git-resolution-1")
    path.write_text(before.replace(BLOCK, "mine\n"), encoding="utf-8")
    await journal.after(token, ActionResult(action=action, success=True, output="done"))
    preview = await journal.preview("git-resolution-1")
    result = await journal.undo("git-resolution-1", preview["preview_token"])

    assert result["status"] == "ok" and result["undone"] == 1
    assert path.read_text(encoding="utf-8") == before
