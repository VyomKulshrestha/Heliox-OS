"""Git automation agent for Heliox-OS."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.agent_mesh import AgentBudgetPolicy
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride

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


class GitAgent(BaseAgent):
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
    ACTION_TYPES = frozenset(
        {
            ActionType.GIT_STATUS,
            ActionType.GIT_DIFF,
            ActionType.GIT_LOG,
            ActionType.GIT_BRANCH,
            ActionType.GIT_STAGE,
            ActionType.GIT_COMMIT,
            ActionType.GIT_PUSH,
            ActionType.GIT_RESOLVE,
        }
    )
    mesh_keywords = ("git", "repository", "branch", "commit", "diff", "status", "push", "conflict")
    mesh_budget = AgentBudgetPolicy(
        max_tokens_per_task=6_000,
        max_actions_per_task=12,
        max_latency_ms_per_action=90_000,
        max_concurrency=1,
    )
    mesh_filesystem_read = ("selected_repository",)
    mesh_filesystem_write = ("selected_repository",)
    mesh_network_domains = ("configured_git_remotes",)
    mesh_credential_names = ("git_credential_helper",)

    def __init__(
        self,
        repo_path: str | None = None,
        llm_client: Any = None,
        model_router: ModelRouter | None = None,
        executor: Executor | None = None,
    ) -> None:
        """
        Initialise the GitAgent.

        Args:
            repo_path:  Path to the git repo. Defaults to current directory.
            llm_client: LLM client for commit message generation.
                        If None, falls back to a safe static message.
        """
        super().__init__(role=AgentRole.GIT, model_router=model_router)
        self.repo_path = str(Path(repo_path).resolve()) if repo_path else None
        self.llm_client = llm_client
        self._executor = executor
        logger.info("GitAgent initialised — repo_path=%s", self.repo_path)

    # Public interface — called by the Orchestrator

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=action_type,
                description=f"Git repository operation: {action_type.value}",
                requires_confirmation=action_type in {ActionType.GIT_COMMIT, ActionType.GIT_PUSH},
            )
            for action_type in sorted(self.ACTION_TYPES, key=lambda item: item.value)
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the GIT AGENT. Inspect repository state before changing it, keep every "
            "operation scoped to the selected repository, require an explicit commit message, "
            "and never push without confirmation."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in self.ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        if self._executor is None:
            raise RuntimeError("GitAgent requires the shared Executor for orchestrated tasks")
        self.status = AgentStatus.BUSY
        actions = [action for action in plan.actions if self.can_handle(action.action_type)]
        if not actions:
            self.status = AgentStatus.IDLE
            return []
        try:
            return await self._executor.execute(
                ActionPlan(
                    actions=actions,
                    explanation=f"GitAgent executing {len(actions)} action(s)",
                    raw_input=user_input,
                ),
                initial_last_output=str((context or {}).get("initial_last_output", "")),
                initial_largest_output=str((context or {}).get("initial_largest_output", "")),
                invocation_source=self.get_invocation_source(),
                scope_override=scope_override,
                user_confirmed=bool((context or {}).get("user_confirmed", False)),
            )
        finally:
            self.status = AgentStatus.IDLE

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
        """The diff of staged or unstaged changes, saying what is on the other side too.

        A diff of one side alone answered "(no changes)" when every change was on the other side (a commit message
        or a review then saw nothing). The output now names both sides, and when the side asked for is empty it shows
        the other one. ``diff`` stays the raw diff of the side asked for, which git_commit reads.
        """
        staged = params.get("staged", True)
        asked, other = (["diff", "--staged"], ["diff"]) if staged else (["diff"], ["diff", "--staged"])
        asked_name, other_name = ("staged", "unstaged") if staged else ("unstaged", "staged")

        result = await _run_git(asked, cwd=self.repo_path)
        if not result["success"]:
            return {"success": False, "output": result["stderr"] or "git diff failed", "diff": ""}
        other_files = await _run_git([*other, "--name-only"], cwd=self.repo_path)
        other_count = len(other_files["stdout"].splitlines()) if other_files["success"] else 0
        untracked = await _run_git(["ls-files", "--others", "--exclude-standard"], cwd=self.repo_path)
        untracked_count = len(untracked["stdout"].splitlines()) if untracked["success"] else 0
        tail = f"\n{untracked_count} untracked file(s) are in no diff." if untracked_count else ""

        if result["stdout"]:
            also = (
                f"\nAlso {other_count} file(s) with {other_name} changes, not shown (set staged to "
                f"{str(not staged).lower()} to see them)."
                if other_count
                else ""
            )
            output = result["stdout"] + also + tail
        elif other_count:
            shown = await _run_git(other, cwd=self.repo_path)
            output = f"Nothing {asked_name}; {other_count} file(s) {other_name}:\n{shown['stdout']}" + tail
        else:
            output = "(no changes: nothing staged or unstaged)" + tail
        return {"success": True, "output": output, "diff": result["stdout"]}

    # Tool: git_commit
    async def _handle_commit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Stage files (optional), generate an LLM commit message, and commit."""
        files = params.get("files", ["."])
        if isinstance(files, str):
            files = [files]

        manual_message = params.get("message", "").strip()
        files = [name for name in files if str(name).strip()]

        if files:
            # Stage files first
            stage_result = await self._handle_stage({"files": files})
            if not stage_result["success"]:
                return {
                    "success": False,
                    "output": f"Staging failed: {stage_result['output']}",
                }
        else:
            # No files named: commit what is already staged, or say what to do.
            staged_result = await _run_git(["diff", "--cached", "--quiet"], cwd=self.repo_path)
            if staged_result["returncode"] != 1:
                return {
                    "success": False,
                    "output": (
                        "git_commit: no files were named and nothing is staged. Name the files to commit "
                        "(or '.' for every change), or stage them first with git_stage."
                    ),
                    "commit_message": "",
                    "commit_hash": "",
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

        # What HEAD and the branch were before, so the verifier can tell a new commit from none.
        before_result = await _run_git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=self.repo_path)
        head_before = before_result["stdout"] if before_result["success"] else ""
        branch_result = await _run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=self.repo_path)
        branch = branch_result["stdout"] if branch_result["success"] else ""

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
            ["rev-parse", "--verify", "HEAD"],
            cwd=self.repo_path,
        )
        commit_sha = hash_result["stdout"] if hash_result["success"] else ""
        commit_hash = commit_sha[:7] if commit_sha else "unknown"

        return {
            "success": True,
            "output": f"Committed as {commit_hash}: {commit_message}",
            "commit_message": commit_message,
            # How many files the commit holds, read from the staged diff it was made from (the reply says it).
            "files_committed": len(re.findall(r"^diff --git ", diff, re.MULTILINE)),
            "commit_hash": commit_hash,
            "commit_sha": commit_sha,
            "head_before": head_before,
            "branch": branch,
        }

    # Tool: git_push

    async def _handle_push(self, params: dict[str, Any]) -> dict[str, Any]:
        """Push the current branch to a remote."""
        remote = (params.get("remote") or "origin").strip()
        force = params.get("force", False)

        # Get current branch if not specified. Never guess one: pushing a branch
        # nobody named (for example "main" when HEAD is detached) sends the wrong work.
        branch = (params.get("branch") or "").strip()
        if not branch:
            branch_result = await _run_git(
                ["symbolic-ref", "--quiet", "--short", "HEAD"],
                cwd=self.repo_path,
            )
            if not branch_result["success"] or not branch_result["stdout"]:
                return {
                    "success": False,
                    "output": (
                        "git_push: no branch was named and HEAD is not on a branch (detached HEAD, "
                        "or the repository could not be read); name the branch to push"
                    ),
                }
            branch = branch_result["stdout"]

        source, _, destination = branch.removeprefix("+").partition(":")
        if not source:
            return {
                "success": False,
                "output": (
                    f"git_push: {branch!r} has no source, so it would delete the remote branch, and git_push does "
                    "not delete. To push, name the local branch (for example 'main' or 'main:main'); to delete a "
                    "remote branch, do it yourself in git or on the hosting site"
                ),
            }

        # Record what is being pushed, so the verifier can ask the remote whether it landed.
        pushed_sha, remote_ref = await self._push_target(source, destination)

        args = ["push", remote, branch]
        if force:
            # Safety: log a warning when force push is used
            logger.warning("GitAgent: force push requested to %s/%s", remote, branch)
            args.append("--force-with-lease")  # safer than --force

        result = await _run_git(args, cwd=self.repo_path)

        return {
            "success": result["success"],
            "output": result["stdout"] or result["stderr"],
            "remote": remote,
            "branch": branch,
            "pushed_sha": pushed_sha,
            "remote_ref": remote_ref,
        }

    async def _push_target(self, source: str, destination: str) -> tuple[str, str]:
        """The object a push of ``source[:destination]`` sends and the remote ref it updates ("" when unknown)."""
        full_result = await _run_git(["rev-parse", "--symbolic-full-name", source], cwd=self.repo_path)
        source_ref = full_result["stdout"] if full_result["success"] else ""
        if not source_ref.startswith(("refs/heads/", "refs/tags/")):
            source_ref = ""
        sha_result = await _run_git(
            ["rev-parse", "--verify", "--quiet", source_ref or source],
            cwd=self.repo_path,
        )
        pushed_sha = sha_result["stdout"] if sha_result["success"] else ""
        if not destination:
            return pushed_sha, source_ref
        if destination.startswith("refs/"):
            return pushed_sha, destination
        if source_ref:
            prefix = "refs/tags/" if source_ref.startswith("refs/tags/") else "refs/heads/"
            return pushed_sha, prefix + destination
        return pushed_sha, ""

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
