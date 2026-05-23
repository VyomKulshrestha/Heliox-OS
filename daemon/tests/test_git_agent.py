"""
Unit tests for GitAgent.

Tests use a real temporary git repository — no mocking of git itself,
so these tests require git to be installed. LLM client is mocked.
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.agents.git_agent import GitAgent, _parse_status, _run_git


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a real temporary git repository for testing."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    return str(tmp_path)


@pytest.fixture
def agent(tmp_repo):
    """GitAgent pointed at the temp repo, with a mock LLM client."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="feat(test): add initial test files")
    return GitAgent(repo_path=tmp_repo, llm_client=mock_llm)


# Helper tests


def test_parse_status_empty():
    result = _parse_status("")
    assert result == {"staged": [], "unstaged": [], "untracked": []}


def test_parse_status_untracked():
    result = _parse_status("?? new_file.py\n")
    assert "new_file.py" in result["untracked"]


def test_parse_status_staged():
    result = _parse_status("A  staged_file.py\n")
    assert "staged_file.py" in result["staged"]


def test_parse_status_modified():
    result = _parse_status(" M modified_file.py\n")
    assert "modified_file.py" in result["unstaged"]


# git_status


@pytest.mark.asyncio
async def test_status_clean_repo(agent, tmp_repo):
    """A fresh repo with no files should report clean working tree."""
    # Make an initial commit so HEAD exists
    readme = Path(tmp_repo) / "README.md"
    readme.write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=tmp_repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_repo,
        capture_output=True,
    )

    result = await agent.execute("git_status", {})
    assert result["success"] is True
    assert "clean" in result["output"].lower() or result["staged"] == []


@pytest.mark.asyncio
async def test_status_shows_untracked(agent, tmp_repo):
    """New files should appear in untracked list."""
    new_file = Path(tmp_repo) / "hello.py"
    new_file.write_text("print('hello')")

    result = await agent.execute("git_status", {})
    assert result["success"] is True
    assert "hello.py" in result["untracked"]


# git_branch


@pytest.mark.asyncio
async def test_branch_missing_name(agent):
    """Should fail gracefully when no branch name given."""
    result = await agent.execute("git_branch", {})
    assert result["success"] is False
    assert "name" in result["output"].lower()


@pytest.mark.asyncio
async def test_branch_create(agent, tmp_repo):
    """Should create a new branch successfully."""
    # Need at least one commit first
    f = Path(tmp_repo) / "f.txt"
    f.write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_repo, capture_output=True)

    result = await agent.execute("git_branch", {"name": "feature/test", "create": True})
    assert result["success"] is True
    assert result["branch"] == "feature-test"  # slashes replaced with dashes


# git_stage


@pytest.mark.asyncio
async def test_stage_specific_file(agent, tmp_repo):
    """Should stage a specific file successfully."""
    f = Path(tmp_repo) / "main.py"
    f.write_text("x = 1")

    result = await agent.execute("git_stage", {"files": ["main.py"]})
    assert result["success"] is True
    assert "main.py" in result["staged_files"]


@pytest.mark.asyncio
async def test_stage_no_files(agent):
    """Should fail when files list is empty."""
    result = await agent.execute("git_stage", {"files": []})
    assert result["success"] is False


# git_commit (with LLM mock)


@pytest.mark.asyncio
async def test_commit_generates_llm_message(agent, tmp_repo):
    """Should use LLM to generate commit message and succeed."""
    f = Path(tmp_repo) / "app.py"
    f.write_text("def hello(): return 'world'")

    result = await agent.execute("git_commit", {"files": ["app.py"]})
    assert result["success"] is True
    assert result["commit_message"] == "feat(test): add initial test files"
    assert result["commit_hash"] != ""


@pytest.mark.asyncio
async def test_commit_uses_manual_message(agent, tmp_repo):
    """Should use manual message when provided, skipping LLM."""
    f = Path(tmp_repo) / "manual.py"
    f.write_text("x = 42")

    result = await agent.execute(
        "git_commit",
        {"files": ["manual.py"], "message": "fix: manual message test"},
    )
    assert result["success"] is True
    assert result["commit_message"] == "fix: manual message test"


@pytest.mark.asyncio
async def test_commit_nothing_staged(agent, tmp_repo):
    """Should fail gracefully when nothing to commit."""
    # Fresh repo, no files
    result = await agent.execute("git_commit", {"files": ["nonexistent.py"]})
    assert result["success"] is False


# git_log


@pytest.mark.asyncio
async def test_log_returns_commits(agent, tmp_repo):
    """Should return commit history after commits are made."""
    f = Path(tmp_repo) / "log_test.py"
    f.write_text("x = 1")
    subprocess.run(["git", "add", "."], cwd=tmp_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "test commit"], cwd=tmp_repo, capture_output=True)

    result = await agent.execute("git_log", {"limit": 5})
    assert result["success"] is True
    assert len(result["commits"]) >= 1
    assert result["commits"][0]["message"] == "test commit"


# Unknown action


@pytest.mark.asyncio
async def test_unknown_action(agent):
    """Unknown action_type should return failure, not raise exception."""
    result = await agent.execute("git_teleport", {})
    assert result["success"] is False
    assert "unknown" in result["output"].lower()
