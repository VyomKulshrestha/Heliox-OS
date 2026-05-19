"""Developer Tools Plugin — Jira, Git, code generation, and PR automation.

Provides tools for a complete developer workflow:
  1. Read Jira tickets (via REST API)
  2. Clone repositories
  3. Create feature branches
  4. Generate code from specifications
  5. Commit, push, and open pull requests
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.plugins.developer")


async def jira_read_ticket(
    ticket_id: str,
    jira_base_url: str,
    api_token: str = "",
    email: str = "",
) -> dict[str, Any]:
    """Fetch a Jira ticket's details via the REST API."""
    url = f"{jira_base_url.rstrip('/')}/rest/api/3/issue/{ticket_id}"
    headers: dict[str, str] = {"Accept": "application/json"}

    if api_token and email:
        import base64

        creds = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        fields = data.get("fields", {})
        return {
            "ticket_id": ticket_id,
            "summary": fields.get("summary", ""),
            "description": _extract_text(fields.get("description")),
            "status": fields.get("status", {}).get("name", "Unknown"),
            "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
            "priority": (fields.get("priority") or {}).get("name", "None"),
        }
    except URLError as e:
        logger.error("Jira API error: %s", e)
        return {"error": str(e), "ticket_id": ticket_id}


def _extract_text(description: Any) -> str:
    """Extract plain text from Atlassian Document Format or string."""
    if isinstance(description, str):
        return description
    if isinstance(description, dict):
        parts: list[str] = []
        for content in description.get("content", []):
            for item in content.get("content", []):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
        return " ".join(parts)
    return ""


async def git_clone_repo(repo_url: str, target_dir: str) -> dict[str, Any]:
    """Clone a Git repository."""
    target = Path(target_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", repo_url, str(target)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "clone_path": ""}
    return {"clone_path": str(target)}


async def git_create_branch(repo_path: str, branch_name: str) -> dict[str, Any]:
    """Create and checkout a new branch."""
    result = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "branch_created": False}
    return {"branch_created": True, "branch_name": branch_name}


async def git_commit_push(
    repo_path: str,
    commit_message: str,
    remote: str = "origin",
) -> dict[str, Any]:
    """Stage, commit, and push changes."""
    cmds = [
        ["git", "add", "-A"],
        ["git", "commit", "-m", commit_message],
        ["git", "push", remote, "HEAD"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": f"'{' '.join(cmd)}' failed: {result.stderr.strip()}"}

    # Get commit hash
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {"commit_hash": hash_result.stdout.strip()}


async def github_open_pr(
    repo_path: str,
    title: str,
    body: str = "",
    base_branch: str = "main",
) -> dict[str, Any]:
    """Open a GitHub pull request using the gh CLI."""
    cmd = ["gh", "pr", "create", "--title", title, "--base", base_branch]
    if body:
        cmd.extend(["--body", body])

    result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "pr_url": ""}
    return {"pr_url": result.stdout.strip()}


# Tool dispatch map — the plugin loader uses this to route tool calls.
TOOL_HANDLERS = {
    "jira_read_ticket": jira_read_ticket,
    "git_clone_repo": git_clone_repo,
    "git_create_branch": git_create_branch,
    "git_commit_push": git_commit_push,
    "github_open_pr": github_open_pr,
}
