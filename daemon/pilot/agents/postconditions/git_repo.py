"""Post-conditions for Git actions, read back from the repository and from the remote.

Every check runs read-only git commands (``rev-parse``, ``log``, ``diff --quiet``,
``ls-remote``, ``merge-base --is-ancestor``) in a fresh process. None of them runs a
hook: hooks belong to commit, push, merge and checkout, never to these reads, and
nothing here commits, fetches, pushes or moves a ref.

The handler records what it set out to do in its JSON result (the commit it was on
before, the commit it made, the commit and remote ref it pushed). That record is the
claim; the check asks the repository, or for a push the remote itself, whether the
claim holds. The output git printed while committing or pushing is never read.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pilot.actions import ActionResult, ActionType, GitParams
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

LOCAL_GIT_TIMEOUT_SECONDS = 10
REMOTE_GIT_TIMEOUT_SECONDS = 25
_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def non_interactive_git_env() -> dict[str, str]:
    """The environment for a read-only git process that must never wait for a person.

    A terminal password prompt, a credential manager window and an SSH password,
    passphrase or host-key question are all turned off, so a remote that wants a
    credential nobody stored fails at once instead of hanging. Credentials a helper
    already holds, and an SSH command the user set in the environment, still apply.
    """

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["GIT_ASKPASS"] = ""
    env["SSH_ASKPASS"] = ""
    env["SSH_ASKPASS_REQUIRE"] = "never"
    if "GIT_SSH_COMMAND" not in env and "GIT_SSH" not in env:
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=10"
    return env


async def _git(repo: str, *args: str, timeout: int = LOCAL_GIT_TIMEOUT_SECONDS) -> tuple[int, str]:
    """Run one read-only git command: (exit code, stdout). -1 when git is missing or ran out of time."""
    from pilot.system.platform_detect import run_command

    # A file-system monitor the repository configures is a program git would start; the reads need none.
    code, out, _err = await run_command(
        ["git", "-c", "core.fsmonitor=false", *args], cwd=repo, timeout=timeout, env=non_interactive_git_env()
    )
    return code, out


def _repo(params: GitParams) -> str:
    return str(Path(params.repo_path or ".").resolve())


def _record(result: ActionResult) -> dict[str, Any]:
    try:
        record = json.loads(result.output or "")
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    return value.strip() if isinstance(value, str) else ""


def _short(sha: str) -> str:
    return sha[:12]


def plain_remote(remote: str) -> str:
    """A remote name or URL fit for a message: a user name or token inside a URL is dropped."""
    return _USERINFO.sub("", remote)


def normalized_message(message: str) -> str:
    """A commit message as ``git commit -m`` stores it by default (cleanup mode ``whitespace``)."""
    lines = [line.rstrip() for line in message.replace("\r\n", "\n").split("\n")]
    collapsed: list[str] = []
    for line in lines:
        if not line and (not collapsed or not collapsed[-1]):
            continue
        collapsed.append(line)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed)


# ── git_commit ───────────────────────────────────────────────────────


@register(ActionType.GIT_COMMIT, method="commit_content_postcondition")
async def commit_landed(result: ActionResult) -> Outcome:
    """The new commit exists, carries the requested message, sits on the commit HEAD was on
    before, is still on the branch it was made on, and nothing of the requested files is left staged."""

    params: GitParams = result.action.parameters  # type: ignore[assignment]
    repo = _repo(params)
    record = _record(result)
    commit = _text(record, "commit_sha")
    before = _text(record, "head_before")
    branch = _text(record, "branch")
    if not commit:
        return unobserved("The commit handler did not record the commit it made; nothing to read back")
    if before and commit == before:
        return mismatch(f"HEAD did not move: it is still at {_short(before)} after the commit")

    code, out = await _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if code == -1:
        return unobserved("git could not be run to read the commit back")
    if code != 0:
        return mismatch(f"Commit {_short(commit)} does not exist in {repo}")

    code, out = await _git(repo, "log", "-1", "--format=%B", commit)
    if code != 0:
        return unobserved(f"git could not read the message of {_short(commit)}")
    expected = normalized_message(params.message)
    actual = normalized_message(out)
    if actual != expected:
        return mismatch(
            f"Commit {_short(commit)} has a different message from the one requested "
            f"(first line {actual.splitlines()[0] if actual else '(empty)'!r}, "
            f"expected {expected.splitlines()[0] if expected else '(empty)'!r}); a commit-msg hook may have changed it"
        )

    code, parent = await _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^")
    parent = parent.strip() if code == 0 else ""
    if parent != before:
        return mismatch(
            f"Commit {_short(commit)} sits on {_short(parent) or 'no parent'}, "
            f"not on {_short(before) or 'an empty branch'} where HEAD was before"
        )

    tip_ref = f"refs/heads/{branch}" if branch else "HEAD"
    code, tip = await _git(repo, "rev-parse", "--verify", "--quiet", tip_ref)
    tip = tip.strip() if code == 0 else ""
    if not tip:
        return mismatch(f"{branch or 'HEAD'} no longer exists after the commit")
    if tip != commit:
        code, _ = await _git(repo, "merge-base", "--is-ancestor", commit, tip)
        if code == 1:
            return mismatch(
                f"Commit {_short(commit)} is no longer on {branch or 'HEAD'} (now at {_short(tip)}); "
                "something reset or rewrote it after the commit"
            )
        if code != 0:
            return unobserved(f"git could not tell whether {branch or 'HEAD'} still holds {_short(commit)}")

    code, head = await _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    if code == 0 and head.strip() == commit:
        files = [name for name in params.files if name] or ["."]
        code, _ = await _git(repo, "diff", "--cached", "--quiet", "--", *files)
        if code == 1:
            return mismatch(f"Commit {_short(commit)} was made but changes to the requested files are still staged")
        if code != 0:
            return unobserved("git could not compare the index with the new commit")
        return verified(
            f"Commit {_short(commit)} is on {branch or 'HEAD'} with the requested message; nothing left staged"
        )
    return verified(
        f"Commit {_short(commit)} is on {branch or 'HEAD'} with the requested message "
        "(later commits followed it, so the index was not compared)"
    )


# ── git_push ─────────────────────────────────────────────────────────


@register(ActionType.GIT_PUSH, method="remote_ref_postcondition")
async def push_landed(result: ActionResult) -> Outcome:
    """The remote ref points at the commit that was pushed, asked of the remote with ``git ls-remote``."""

    params: GitParams = result.action.parameters  # type: ignore[assignment]
    repo = _repo(params)
    record = _record(result)
    pushed = _text(record, "pushed_sha")
    remote_ref = _text(record, "remote_ref")
    remote = _text(record, "remote") or params.remote or "origin"
    shown = plain_remote(remote)
    if not pushed or not remote_ref:
        return unobserved("The push handler did not record which commit and remote ref it pushed")
    if not (remote_ref.startswith("refs/heads/") or remote_ref.startswith("refs/tags/")):
        return unobserved(f"{remote_ref} is not a branch or a tag the remote lists; the push cannot be read back")

    # Ask the repository pushed to: a remote with a separate push URL (a fork) is read at that URL.
    code, url = await _git(repo, "remote", "get-url", "--push", remote)
    target = url.strip() if code == 0 and url.strip() else remote

    code, listing = await _git(repo, "ls-remote", target, remote_ref, timeout=REMOTE_GIT_TIMEOUT_SECONDS)
    if code != 0:
        return unobserved(
            f"Remote {shown} could not be asked for {remote_ref} (no network, a credential it would have to "
            "ask for, or no answer in time); the push is unverified"
        )
    found = ""
    for line in listing.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.strip() == remote_ref:
            found = sha.strip()
            break
    if not found:
        return mismatch(f"Remote {shown} has no {remote_ref} after the push; expected it at {_short(pushed)}")
    if found == pushed:
        return verified(f"Remote {shown} {remote_ref} points at the pushed commit {_short(pushed)}")

    code, _ = await _git(repo, "merge-base", "--is-ancestor", pushed, found)
    if code == 0:
        return verified(
            f"Remote {shown} {remote_ref} is at {_short(found)}, which already contains the pushed commit {_short(pushed)}"
        )
    if code == 1:
        return mismatch(
            f"Remote {shown} {remote_ref} points at {_short(found)}, not at the pushed commit {_short(pushed)} "
            "and not at a commit that contains it"
        )
    return mismatch(
        f"Remote {shown} {remote_ref} points at {_short(found)}, a commit this repository does not have, "
        f"instead of the pushed commit {_short(pushed)}"
    )
