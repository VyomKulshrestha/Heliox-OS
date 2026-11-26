"""git_commit and git_push post-conditions, against real temporary repositories.

Every repository here is made under pytest's tmp_path with a local bare repository
as its remote. The developer's global and system git configuration is hidden from
git for these tests (GIT_CONFIG_GLOBAL points at an empty file, GIT_CONFIG_NOSYSTEM
is set), user.name and user.email are set inside each temporary repository only,
and nothing is pushed anywhere but the temporary bare repositories.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, GitParams
from pilot.agents.executor import Executor
from pilot.agents.postconditions import git_repo
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _repo(root: Path, name: str = "work") -> Path:
    repo = root / name
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Heliox Test")
    _git(repo, "config", "user.email", "heliox-test@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    _git(repo, "config", "core.hooksPath", str(repo / ".git" / "hooks"))
    return repo


def _commit_file(repo: Path, name: str, text: str, message: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _bare(root: Path, name: str = "remote.git") -> Path:
    bare = root / name
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    return bare


def _hook(repo: Path, name: str, body: str) -> None:
    hook = repo / ".git" / "hooks" / name
    hook.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8", newline="\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


async def _run(action_type: ActionType, params: GitParams) -> ActionResult:
    """Run the action through the executor's own git handler and wrap it as the executor does."""
    action = Action(action_type=action_type, target=params.repo_path, parameters=params)
    try:
        output = await Executor._exec_git_action(MagicMock(), action)
    except Exception as error:  # noqa: BLE001 - mirrors Executor._execute_single
        return ActionResult(action=action, success=False, error=str(error))
    return ActionResult(action=action, success=True, output=output)


async def _verify(result: ActionResult):
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[result.action]), [result])


def _assert_verified(verification) -> None:
    assert verification.passed is True, verification.details
    assert verification.independently_observed_actions == [0]
    assert "VERIFIED" in verification.details[0]


def _assert_mismatch(verification, *words: str) -> None:
    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "MISMATCH" in verification.details[0]
    for word in words:
        assert word in verification.details[0]


def test_git_verifiers_are_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.GIT_COMMIT] == "commit_content_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.GIT_PUSH] == "remote_ref_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.GIT_STAGE] == "index_matches_worktree_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.GIT_BRANCH] == "current_branch_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.GIT_RESOLVE] == "conflict_block_replaced_postcondition"


# ── git_commit ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_verifies(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    before = _commit_file(repo, "a.txt", "one", "start")
    (repo / "a.txt").write_text("two", encoding="utf-8")

    result = await _run(
        ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="fix: change a\n\nwhy  ")
    )

    assert result.success is True
    assert json.loads(result.output)["head_before"] == before
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_first_commit_of_an_empty_repository_verifies(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("one", encoding="utf-8")

    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="first"))

    assert json.loads(result.output)["head_before"] == ""
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_commit_rejected_by_pre_commit_hook_is_a_failed_action(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    before = _commit_file(repo, "a.txt", "one", "start")
    _hook(repo, "pre-commit", "echo rejected by policy >&2\nexit 1")
    (repo / "a.txt").write_text("two", encoding="utf-8")

    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="blocked"))
    verification = await _verify(result)

    assert result.success is False
    assert verification.passed is False
    assert "FAILED" in verification.details[0]
    assert _git(repo, "rev-parse", "HEAD") == before


@pytest.mark.asyncio
async def test_commit_message_rewritten_by_hook_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    _hook(repo, "commit-msg", 'echo "something else" > "$1"')
    (repo / "a.txt").write_text("two", encoding="utf-8")

    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="fix: a"))

    assert result.success is True
    _assert_mismatch(await _verify(result), "different message")


@pytest.mark.asyncio
async def test_commit_undone_by_post_commit_hook_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    _hook(repo, "post-commit", "git reset --soft HEAD~1")
    (repo / "a.txt").write_text("two", encoding="utf-8")

    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="fix: a"))

    assert result.success is True
    _assert_mismatch(await _verify(result), "did not move")


@pytest.mark.asyncio
async def test_commit_reset_away_after_the_action_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    before = _commit_file(repo, "a.txt", "one", "start")
    (repo / "a.txt").write_text("two", encoding="utf-8")
    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="fix: a"))

    _git(repo, "reset", "--hard", before)

    _assert_mismatch(await _verify(result), "no longer on main")


@pytest.mark.asyncio
async def test_commit_followed_by_another_commit_still_verifies(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    (repo / "a.txt").write_text("two", encoding="utf-8")
    result = await _run(ActionType.GIT_COMMIT, GitParams(repo_path=str(repo), files=["a.txt"], message="fix: a"))

    _commit_file(repo, "b.txt", "three", "later")

    verification = await _verify(result)
    _assert_verified(verification)


@pytest.mark.asyncio
async def test_commit_without_a_record_is_reported_not_observed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    action = Action(action_type=ActionType.GIT_COMMIT, parameters=GitParams(repo_path=str(repo), message="x"))

    verification = await _verify(ActionResult(action=action, success=True, output="Committed"))

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


def test_commit_message_is_compared_as_git_stores_it() -> None:
    assert git_repo.normalized_message("\n\nsubject  \n\n\n\nbody\t\n\n") == "subject\n\nbody"
    assert git_repo.normalized_message("a\r\nb\n") == "a\nb"


# ── git_push ─────────────────────────────────────────────────────────


def _pushed_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = _repo(tmp_path)
    bare = _bare(tmp_path)
    _git(repo, "remote", "add", "origin", str(bare))
    _commit_file(repo, "a.txt", "one", "start")
    return repo, bare


@pytest.mark.asyncio
async def test_push_verifies_against_the_remote(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))

    assert result.success is True
    record = json.loads(result.output)
    assert (record["pushed_sha"], record["remote_ref"]) == (head, "refs/heads/main")
    assert _git(bare, "rev-parse", "refs/heads/main") == head
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_push_of_the_current_branch_without_a_name(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    _git(repo, "switch", "-c", "feature")
    head = _commit_file(repo, "b.txt", "two", "feature work")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin"))

    assert json.loads(result.output)["remote_ref"] == "refs/heads/feature"
    assert _git(bare, "rev-parse", "refs/heads/feature") == head
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_push_from_detached_head_without_a_branch_fails_and_pushes_nothing(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    _git(repo, "push", "origin", "main")
    remote_before = _git(bare, "rev-parse", "refs/heads/main")
    _git(repo, "switch", "--detach")
    _commit_file(repo, "b.txt", "two", "detached work")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin"))

    assert result.success is False
    assert "detached" in (result.error or "")
    assert _git(bare, "rev-parse", "refs/heads/main") == remote_before


@pytest.mark.asyncio
async def test_push_outside_a_repository_fails_instead_of_pushing_main(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(not_a_repo), remote="origin"))

    assert result.success is False
    assert "name the branch" in (result.error or "")


@pytest.mark.asyncio
async def test_push_refspec_that_would_delete_is_refused(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    _git(repo, "push", "origin", "main")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch=":main"))

    assert result.success is False
    assert _git(bare, "rev-parse", "refs/heads/main")


@pytest.mark.asyncio
async def test_push_then_remote_reset_is_a_mismatch(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "origin", "main")
    _commit_file(repo, "b.txt", "two", "second")
    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))
    _assert_verified(await _verify(result))

    _git(bare, "update-ref", "refs/heads/main", first)

    _assert_mismatch(await _verify(result), "not at the pushed commit")


@pytest.mark.asyncio
async def test_push_whose_remote_branch_is_gone_is_a_mismatch(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))

    _git(bare, "update-ref", "-d", "refs/heads/main")

    _assert_mismatch(await _verify(result), "has no refs/heads/main")


@pytest.mark.asyncio
async def test_push_recorded_for_one_branch_but_landing_on_another_is_a_mismatch(tmp_path: Path) -> None:
    repo, _bare_repo = _pushed_repo(tmp_path)
    _git(repo, "push", "origin", "main")
    _git(repo, "switch", "-c", "feature")
    _commit_file(repo, "b.txt", "two", "feature work")
    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="feature"))
    record = json.loads(result.output)
    record["remote_ref"] = "refs/heads/main"
    wrong = result.model_copy(update={"output": json.dumps(record)})

    _assert_mismatch(await _verify(wrong), "refs/heads/main")


@pytest.mark.asyncio
async def test_push_overtaken_by_a_later_push_that_contains_it_verifies(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))
    _commit_file(repo, "b.txt", "two", "next")
    _git(repo, "push", "origin", "main")

    _assert_verified(await _verify(result))
    assert "contains the pushed commit" in (await git_repo.push_landed(result)).detail


@pytest.mark.asyncio
async def test_push_to_an_unreachable_remote_is_reported_not_observed(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))

    bare.rename(tmp_path / "moved-away.git")

    verification = await _verify(result)
    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_push_reads_the_push_url_of_a_fork(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    upstream = _bare(tmp_path, "upstream.git")
    fork = _bare(tmp_path, "fork.git")
    _git(repo, "remote", "add", "origin", str(upstream))
    _git(repo, "remote", "set-url", "--push", "origin", str(fork))
    head = _commit_file(repo, "a.txt", "one", "start")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main"))

    assert _git(fork, "rev-parse", "refs/heads/main") == head
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_push_of_an_annotated_tag_verifies(tmp_path: Path) -> None:
    repo, _bare_repo = _pushed_repo(tmp_path)
    _git(repo, "tag", "-a", "v1.0", "-m", "release")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="v1.0"))

    assert json.loads(result.output)["remote_ref"] == "refs/tags/v1.0"
    _assert_verified(await _verify(result))


@pytest.mark.asyncio
async def test_force_push_of_a_rewritten_branch_verifies(tmp_path: Path) -> None:
    repo, bare = _pushed_repo(tmp_path)
    _git(repo, "push", "origin", "main")
    _git(repo, "fetch", "origin")
    _git(repo, "commit", "--amend", "-m", "start, reworded")
    rewritten = _git(repo, "rev-parse", "HEAD")

    result = await _run(ActionType.GIT_PUSH, GitParams(repo_path=str(repo), remote="origin", branch="main", force=True))

    assert result.success is True, result.error
    assert _git(bare, "rev-parse", "refs/heads/main") == rewritten
    _assert_verified(await _verify(result))


def test_remote_in_a_message_never_carries_a_credential() -> None:
    assert git_repo.plain_remote("https://user:token@example.invalid/r.git") == "https://example.invalid/r.git"
    assert git_repo.plain_remote("origin") == "origin"


def test_remote_reads_never_wait_for_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)
    env = git_repo.non_interactive_git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]

    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i custom-key")
    assert git_repo.non_interactive_git_env()["GIT_SSH_COMMAND"] == "ssh -i custom-key"
    assert os.environ["GIT_SSH_COMMAND"] == "ssh -i custom-key"


# ── git_stage, git_branch, git_resolve ───────────────────────────────


@pytest.mark.asyncio
async def test_stage_verifies_and_a_later_edit_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    (repo / "a.txt").write_text("two", encoding="utf-8")
    (repo / "new.txt").write_text("new", encoding="utf-8")

    result = await _run(ActionType.GIT_STAGE, GitParams(repo_path=str(repo), files=["a.txt", "new.txt"]))

    assert result.success is True
    _assert_verified(await _verify(result))
    (repo / "a.txt").write_text("three", encoding="utf-8")
    _assert_mismatch(await _verify(result), "not staged")


@pytest.mark.asyncio
async def test_stage_leaving_an_untracked_file_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    (repo / "new.txt").write_text("new", encoding="utf-8")
    action = Action(action_type=ActionType.GIT_STAGE, parameters=GitParams(repo_path=str(repo), files=["new.txt"]))

    verification = await _verify(ActionResult(action=action, success=True, output="{}"))

    _assert_mismatch(verification, "untracked")


@pytest.mark.asyncio
async def test_stage_of_a_missing_file_is_a_failed_action(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    result = await _run(ActionType.GIT_STAGE, GitParams(repo_path=str(repo), files=["missing.txt"]))

    assert result.success is False


@pytest.mark.asyncio
async def test_branch_create_verifies_and_switching_away_is_a_mismatch(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")

    result = await _run(ActionType.GIT_BRANCH, GitParams(repo_path=str(repo), name="feature/login", create=True))

    assert json.loads(result.output)["branch"] == "feature-login"
    _assert_verified(await _verify(result))
    _git(repo, "switch", "main")
    _assert_mismatch(await _verify(result), "not on branch feature-login")


@pytest.mark.asyncio
async def test_branch_that_does_not_exist_is_a_mismatch_and_a_failed_switch_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit_file(repo, "a.txt", "one", "start")
    action = Action(
        action_type=ActionType.GIT_BRANCH, parameters=GitParams(repo_path=str(repo), name="nope", create=False)
    )

    _assert_mismatch(await _verify(ActionResult(action=action, success=True, output="{}")), "does not exist")
    assert (await _run(ActionType.GIT_BRANCH, action.parameters)).success is False


_CONFLICT = "<<<<<<< HEAD\nx = 1\n=======\nx = 2\n>>>>>>> other\n"


@pytest.mark.asyncio
async def test_resolve_verifies_and_a_restored_block_is_a_mismatch(tmp_path: Path) -> None:
    from pilot.actions import GitResolveParams

    path = tmp_path / "code.py"
    path.write_text("a = 0\n" + _CONFLICT + "b = 3\n", encoding="utf-8")
    params = GitResolveParams(path=str(path), full_block=_CONFLICT, resolved_code="x = 2\n")
    action = Action(action_type=ActionType.GIT_RESOLVE, parameters=params)
    output = await Executor._exec_git_resolve(MagicMock(), action)
    result = ActionResult(action=action, success=True, output=output)

    _assert_verified(await _verify(result))
    path.write_text("a = 0\n" + _CONFLICT + "b = 3\n", encoding="utf-8")
    _assert_mismatch(await _verify(result), "still holds the conflict block")
    path.write_text("a = 0\nb = 3\n", encoding="utf-8")
    _assert_mismatch(await _verify(result), "does not hold the resolved code")
