"""Git automation agent for Heliox-OS."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Helpers


async def _run_git(args, cwd=None):
    """Run a git command asynchronously and return structured output."""
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": stderr.decode("utf-8", errors="replace").strip(),
            "returncode": proc.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "git is not installed or not in PATH",
            "returncode": -1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
        }


def _parse_status(porcelain: str) -> dict[str, list[str]]:
    """
    Parse `git status --porcelain` output into categorised file lists.

    Returns dict with keys: staged, unstaged, untracked
    """
    staged, unstaged, untracked = [], [], []
    for line in porcelain.splitlines():
        if len(line) < 2:
            continue
        index_status = line[0]  # staged area
        work_status = line[1]  # working tree
        filepath = line[2:].strip().strip('"')

        if index_status != " " and index_status != "?":
            staged.append(filepath)
        if work_status not in (" ", "?"):
            unstaged.append(filepath)
        if index_status == "?" and work_status == "?":
            untracked.append(filepath)

    return {"staged": staged, "unstaged": unstaged, "untracked": untracked}


# Conventional commit generation

COMMIT_SYSTEM_PROMPT = """\
You are an expert software engineer writing Git commit messages.
Given a git diff, produce a single conventional commit message.

Rules:
1. Format: <type>(<scope>): <short description>
   - type: feat | fix | docs | refactor | test | chore | perf | ci
   - scope: optional, the module/file affected (e.g. agents, auth, ui)
   - description: imperative mood, max 72 chars, no period at end
2. If the change is large, add a blank line then a short body (max 3 lines).
3. Output ONLY the commit message. No explanations, no markdown, no quotes.

Examples of good commit messages:
  feat(agents): add GitAgent for automated repo management
  fix(auth): handle expired JWT tokens gracefully
  docs(readme): add troubleshooting FAQ section
  refactor(sandbox): extract file operations into helper module
"""


async def _generate_commit_message(diff: str, llm_client: Any) -> str:
    """
    Use the LLM planner to generate a conventional commit message from a diff.

    Args:
        diff:        output of `git diff --staged`
        llm_client:  the project's existing LLM client instance

    Returns:
        A conventional commit message string, or a safe fallback.
    """
    if not diff.strip():
        return "chore: update files"

    # Truncate very large diffs to stay within context limits
    max_diff_chars = 6000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n... [diff truncated for context]"

    prompt = f"Generate a conventional commit message for this diff:\n\n{diff}"

    try:
        # Use the existing LLM client pattern from the project
        response = await llm_client.complete(
            system=COMMIT_SYSTEM_PROMPT,
            user=prompt,
        )
        # Clean up the response — remove any accidental quotes or newlines
        message = response.strip().strip('"').strip("'")
        # Enforce max length on subject line
        lines = message.splitlines()
        if lines and len(lines[0]) > 72:
            lines[0] = lines[0][:72]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM commit generation failed: %s — using fallback", exc)
        return "chore: update repository files"


# GitAgent


class GitAgent:
    # Action types this agent handles — used by the Orchestrator router
    SUPPORTED_ACTIONS = {
        "git_status",
        "git_branch",
        "git_stage",
        "git_commit",
        "git_push",
        "git_diff",
        "git_log",
    }

    def __init__(self, repo_path: str | None = None, llm_client: Any = None) -> None:
        """
        Initialise the GitAgent.

        Args:
            repo_path:  Path to the git repo. Defaults to current directory.
            llm_client: LLM client for commit message generation.
                        If None, falls back to a safe static message.
        """
        self.repo_path = str(Path(repo_path).resolve()) if repo_path else None
        self.llm_client = llm_client
        logger.info("GitAgent initialised — repo_path=%s", self.repo_path)

    # Public interface — called by the Orchestrator

    async def execute(self, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Route an action to the correct tool method.

        Args:
            action_type: One of SUPPORTED_ACTIONS.
            params:      Action-specific parameters dict.

        Returns:
            Result dict always containing at least {"success": bool, "output": str}.
        """
        handlers = {
            "git_status": self._handle_status,
            "git_branch": self._handle_branch,
            "git_stage": self._handle_stage,
            "git_commit": self._handle_commit,
            "git_push": self._handle_push,
            "git_diff": self._handle_diff,
            "git_log": self._handle_log,
        }
        # TODO: add support for git stash operations
        handler = handlers.get(action_type)
        if handler is None:
            return {
                "success": False,
                "output": f"GitAgent: unknown action '{action_type}'",
            }

        try:
            return await handler(params)
        except Exception as exc:  # noqa: BLE001
            logger.exception("GitAgent.execute failed for action=%s", action_type)
            return {"success": False, "output": f"GitAgent error: {exc}"}

    # Tool: git_status

    async def _handle_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get working tree status and branch info."""
        # Get current branch
        branch_result = await _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
        )
        branch = branch_result["stdout"] if branch_result["success"] else "unknown"

        # Get porcelain status
        status_result = await _run_git(
            ["status", "--porcelain"],
            cwd=self.repo_path,
        )

        if not status_result["success"]:
            return {
                "success": False,
                "output": f"git status failed: {status_result['stderr']}",
            }

        parsed = _parse_status(status_result["stdout"])

        # Build human-readable summary
        lines = [f"On branch: {branch}"]
        if parsed["staged"]:
            lines.append(f"Staged ({len(parsed['staged'])}): {', '.join(parsed['staged'])}")
        if parsed["unstaged"]:
            lines.append(f"Unstaged ({len(parsed['unstaged'])}): {', '.join(parsed['unstaged'])}")
        if parsed["untracked"]:
            lines.append(f"Untracked ({len(parsed['untracked'])}): {', '.join(parsed['untracked'])}")
        if not any(parsed.values()):
            lines.append("Working tree clean — nothing to commit.")

        return {
            "success": True,
            "output": "\n".join(lines),
            "staged": parsed["staged"],
            "unstaged": parsed["unstaged"],
            "untracked": parsed["untracked"],
            "branch": branch,
        }

    # Tool: git_branch

    async def _handle_branch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new branch or switch to an existing one."""
        name = params.get("name", "").strip()
        if not name:
            return {"success": False, "output": "git_branch: 'name' parameter is required"}

        # Sanitise branch name — no spaces, no special chars
        safe_name = name.replace(" ", "-").replace("/", "-")

        create = params.get("create", True)

        if create:
            result = await _run_git(
                ["checkout", "-b", safe_name],
                cwd=self.repo_path,
            )
        else:
            result = await _run_git(
                ["checkout", safe_name],
                cwd=self.repo_path,
            )

        return {
            "success": result["success"],
            "output": result["stdout"] or result["stderr"],
            "branch": safe_name if result["success"] else "unchanged",
        }

    # Tool: git_stage

    async def _handle_stage(self, params: dict[str, Any]) -> dict[str, Any]:
        """Stage files for the next commit."""
        files = params.get("files", ["."])
        if isinstance(files, str):
            files = [files]

        if not files:
            return {"success": False, "output": "git_stage: no files specified"}

        result = await _run_git(
            ["add", "--"] + files,
            cwd=self.repo_path,
        )

        return {
            "success": result["success"],
            "output": result["stdout"] or result["stderr"] or f"Staged: {files}",
            "staged_files": files if result["success"] else [],
        }

    # Tool: git_diff

    async def _handle_diff(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get the diff of staged or unstaged changes."""
        staged = params.get("staged", True)
        args = ["diff"]
        if staged:
            args.append("--staged")

        result = await _run_git(args, cwd=self.repo_path)

        return {
            "success": result["success"],
            "output": result["stdout"] or "(no changes)",
            "diff": result["stdout"],
        }

    # Tool: git_commit
    async def _handle_commit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Stage files (optional), generate an LLM commit message, and commit."""
        files = params.get("files", ["."])
        if isinstance(files, str):
            files = [files]

        manual_message = params.get("message", "").strip()

        # Stage files first
        stage_result = await self._handle_stage({"files": files})
        if not stage_result["success"]:
            return {
                "success": False,
                "output": f"Staging failed: {stage_result['output']}",
            }

        # Get staged diff for LLM
        diff_result = await self._handle_diff({"staged": True})
        diff = diff_result.get("diff", "")

        if not diff.strip():
            return {
                "success": False,
                "output": "Nothing staged to commit — working tree may be clean.",
                "commit_message": "",
                "commit_hash": "",
            }

        # Generate commit message
        if manual_message:
            commit_message = manual_message
            logger.info("GitAgent: using manual commit message")
        elif self.llm_client:
            commit_message = await _generate_commit_message(diff, self.llm_client)
            logger.info("GitAgent: LLM generated commit message: %s", commit_message)
        else:
            commit_message = "chore: update files"
            logger.warning("GitAgent: no LLM client — using fallback commit message")

        # Execute the commit
        commit_result = await _run_git(
            ["commit", "-m", commit_message],
            cwd=self.repo_path,
        )

        if not commit_result["success"]:
            return {
                "success": False,
                "output": f"Commit failed: {commit_result['stderr']}",
                "commit_message": commit_message,
                "commit_hash": "",
            }

        # Get the commit hash for reference
        hash_result = await _run_git(
            ["rev-parse", "--short", "HEAD"],
            cwd=self.repo_path,
        )
        commit_hash = hash_result["stdout"] if hash_result["success"] else "unknown"

        return {
            "success": True,
            "output": f"Committed as {commit_hash}: {commit_message}",
            "commit_message": commit_message,
            "commit_hash": commit_hash,
        }

    # Tool: git_push

    async def _handle_push(self, params: dict[str, Any]) -> dict[str, Any]:
        """Push the current branch to a remote."""
        remote = params.get("remote", "origin")
        force = params.get("force", False)

        # Get current branch if not specified
        branch = params.get("branch", "")
        if not branch:
            branch_result = await _run_git(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
            )
            branch = branch_result["stdout"] if branch_result["success"] else "main"

        args = ["push", remote, branch]
        if force:
            # Safety: log a warning when force push is used
            logger.warning("GitAgent: force push requested to %s/%s", remote, branch)
            args.append("--force-with-lease")  # safer than --force

        result = await _run_git(args, cwd=self.repo_path)

        return {
            "success": result["success"],
            "output": result["stdout"] or result["stderr"],
        }

    # Tool: git_log

    async def _handle_log(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get recent commit history."""
        limit = int(params.get("limit", 10))

        result = await _run_git(
            [
                "log",
                f"-{limit}",
                "--pretty=format:%H|%h|%an|%ar|%s",
            ],
            cwd=self.repo_path,
        )

        if not result["success"]:
            return {"success": False, "output": result["stderr"], "commits": []}

        commits = []
        for line in result["stdout"].splitlines():
            parts = line.split("|", 4)
            if len(parts) == 5:
                commits.append(
                    {
                        "hash": parts[0],
                        "short": parts[1],
                        "author": parts[2],
                        "when": parts[3],
                        "message": parts[4],
                    }
                )

        readable = "\n".join(f"{c['short']} ({c['when']}) — {c['message']}" for c in commits)

        return {
            "success": True,
            "output": readable or "No commits found.",
            "commits": commits,
        }
