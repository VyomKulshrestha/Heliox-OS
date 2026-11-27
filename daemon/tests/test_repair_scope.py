"""A value the request gives stays in its step, and a repair keeps to the failed step's own action.

Second live run: "Scrape https://example.com and extract the text of the elements matching '.price-table td.amount'"
was repaired by scraping "body" and the whole page came back as the answer; a push the remote rejected was repaired
with "git pull" as a shell command (two approval cards, 26.5 s, seven model calls). Fakes only: plans are built here
and a scripted model stands in for the planner's."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    ApiRequestParams,
    FileParams,
    GitParams,
    ShellCommandParams,
)
from pilot.agents import request_match
from pilot.agents.planner import Planner

W = "C:\\w"
SCRAPE = "Scrape https://example.com and extract the text of the elements matching '.price-table td.amount'"
PUSH = f"Push the main branch of the git repository at {W}\\repo to origin"


def scrape(selector: str) -> Action:
    return Action(
        action_type=ActionType.API_SCRAPE,
        target="https://example.com",
        parameters=ApiRequestParams(url="https://example.com", selector=selector, extract="text"),
    )


def push(force: bool = False) -> Action:
    return Action(
        action_type=ActionType.GIT_PUSH,
        target=f"{W}\\repo",
        parameters=GitParams(repo_path=f"{W}\\repo", remote="origin", branch="main", force=force),
    )


def shell(*args: str) -> Action:
    return Action(
        action_type=ActionType.SHELL_COMMAND,
        target="git",
        parameters=ShellCommandParams(command="git", args=list(args)),
    )


def status() -> Action:
    return Action(action_type=ActionType.GIT_STATUS, target=f"{W}\\repo", parameters=GitParams(repo_path=f"{W}\\repo"))


# ── values the request gives ─────────────────────────────────────────


def test_the_values_a_request_gives_are_code_like_quotes_and_line_counts() -> None:
    assert request_match.given_values(SCRAPE) == (".price-table td.amount",)
    assert request_match.given_values("List the files matching `*.log` in C:\\w") == ("*.log",)
    assert request_match.given_values("Show me the last 5 lines of C:\\w\\app.log") == ("5",)
    assert request_match.given_values("Search C:\\w for files that mention 'quarterly budget'") == ()
    # A sentence's full stop is not code: the line to add is text, held by the write rules, not here.
    assert request_match.given_values("Add the line 'Also renew the parking permit.' to C:\\w\\note.txt") == ()


def test_a_read_that_drops_the_selector_is_refused_and_one_that_keeps_it_passes() -> None:
    problem = request_match.check_plan(SCRAPE, [scrape("body")])
    assert problem is not None and problem.rule == "value_dropped"
    assert request_match.check_plan(SCRAPE, [scrape(".price-table td.amount")]) is None


def test_a_line_count_must_reach_the_read() -> None:
    log = f"{W}\\app.log"
    request_text = f"Show me the last 5 lines of {log}"
    whole = Action(action_type=ActionType.FILE_READ, target=log, parameters=FileParams(path=log))
    tail = Action(action_type=ActionType.FILE_READ, target=log, parameters=FileParams(path=log, tail_lines=5))
    assert request_match.check_plan(request_text, [whole]) is not None
    assert request_match.check_plan(request_text, [tail]) is None


def test_a_repair_that_drops_the_selector_is_not_run() -> None:
    problem = request_match.check_repair(SCRAPE, [scrape("body")], failed=[scrape(".price-table td.amount")])
    assert problem is not None
    assert request_match.repair_refusal(problem) == (
        "I did not try again: the retry would leave out '.price-table td.amount', which you asked for."
    )


# ── a repair keeps to the failed step's action ─────────────────────────


def test_a_repair_may_not_get_around_a_rejected_push_with_a_shell_command() -> None:
    repair = [shell("-C", f"{W}\\repo", "pull", "--no-rebase", "origin", "main"), push()]
    problem = request_match.check_repair(PUSH, repair, failed=[push()])
    assert problem is not None and problem.rule == "repair_escalation"
    assert request_match.repair_refusal(problem) == (
        "I did not try again: the retry would run a shell command instead of the git push that failed."
    )


def test_a_repair_may_not_force_a_push_the_failed_step_did_not_force() -> None:
    problem = request_match.check_repair(PUSH, [push(force=True)], failed=[push()])
    assert problem is not None and problem.rule == "repair_escalation" and "'force'" in problem.step


def test_a_repair_may_not_start_another_kind_of_change() -> None:
    commit = Action(
        action_type=ActionType.GIT_COMMIT,
        target=f"{W}\\repo",
        parameters=GitParams(repo_path=f"{W}\\repo", message="merge"),
    )
    problem = request_match.check_repair(PUSH, [commit, push()], failed=[push()])
    assert problem is not None and problem.rule == "repair_new_kind"
    assert request_match.repair_refusal(problem).endswith("start a git commit, which you did not ask for.")


def test_a_repair_may_retry_the_step_or_add_a_read() -> None:
    assert request_match.check_repair(PUSH, [push()], failed=[push()]) is None
    assert request_match.check_repair(PUSH, [status(), push()], failed=[push()]) is None


def test_a_plan_that_had_a_shell_command_may_retry_it() -> None:
    run = shell("--version")
    assert request_match.repair_escalation([run], [run]) is None


# ── the planner holds a repair round to this ─────────────────────────


class ScriptedModel:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.sent: list[list[dict[str, Any]]] = []

    async def generate(self, messages, **_kwargs) -> str:
        self.sent.append([dict(m) for m in messages])
        return self.answers.pop(0)


class EmptyMemory:
    async def get_context(self, *_a, **_k) -> str:
        return ""

    async def get_history(self, *_a, **_k) -> list[dict[str, Any]]:
        return []


@pytest.mark.asyncio
async def test_the_repair_round_after_a_rejected_push_is_refused_with_one_model_call() -> None:
    answer = json.dumps(
        {
            "explanation": "Pull, then push again.",
            "actions": [
                {
                    "action_type": "shell_command",
                    "target": "git",
                    "parameters": {"command": "git", "args": ["-C", f"{W}\\repo", "pull", "origin", "main"]},
                },
                {
                    "action_type": "git_push",
                    "target": f"{W}\\repo",
                    "parameters": {"repo_path": f"{W}\\repo", "remote": "origin", "branch": "main"},
                },
            ],
        }
    )
    model = ScriptedModel(answer)
    plan = await Planner(model, EmptyMemory()).plan(  # type: ignore[arg-type]
        PUSH,
        error_context="Action 0 (git_push): FAILED \u2014 ! [rejected] main -> main (fetch first)",
        repair_of=ActionPlan(actions=[push()]),
    )

    assert plan.actions == [] and request_match.is_refusal(plan.error)
    assert "shell command" in (plan.error or "")
    assert len(model.sent) == 1
    sent = model.sent[0][-1]["content"]
    assert "Never switch to a shell command, a script or code" in sent
