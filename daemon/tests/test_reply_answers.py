"""Round twelve: what the second live run found in the replies.

- An HTTP answer is said by what a person can do about it, by host only, and the error page never reaches the reply.
- Git's common endings are final failures with their own sentences; the reply leads with the failure of the plan the
  person asked for, never a repair step's.
- A read answers the question asked: a tail of a log is "the last 5 lines", one scraped title is "the title of",
  a process list is how many and the biggest by name, system information is the facts the question was about.
"""

from __future__ import annotations

import json

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    FileIntelParams,
    FileParams,
    GitParams,
    ProcessParams,
    ServiceParams,
    SystemInfoParams,
    TriggerParams,
    VerificationResult,
)
from pilot.failure_sentences import FailureKind, classify, failure_sentence, is_final
from pilot.reply_rules import ear_faults, screen_faults
from pilot.response_contract import failure_reply, success_reply
from pilot.result_sentences import step_said
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute

W = r"C:\Users\me\work"
REPO = W + r"\repo"
PAGE = "<!doctype html><html><head><title>404</title></head><body>Not here</body></html>"


def _get(url: str = "https://example.com/missing") -> Action:
    return Action(action_type=ActionType.API_REQUEST, target=url, parameters=ApiRequestParams(url=url))


def _push() -> Action:
    return Action(
        action_type=ActionType.GIT_PUSH,
        target=REPO,
        parameters=GitParams(repo_path=REPO, remote="origin", branch="main"),
    )


def _result(kind: ActionType, params: object, output: str, target: str = "") -> ActionResult:
    return ActionResult(action=Action(action_type=kind, target=target, parameters=params), success=True, output=output)


# ── the web ───────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "kind", "spoken", "final"),
    [
        (f"GET to https://example.com failed with HTTP 404 Not Found (the server has nothing at this address): {PAGE}",
         FailureKind.PAGE_NOT_THERE, "Example.com has no page at that address. Check the address and ask me again.", True),
        ("GET to https://example.com failed with HTTP 410 Gone", FailureKind.PAGE_NOT_THERE,
         "Example.com has no page at that address. Check the address and ask me again.", True),
        ("GET to https://example.com failed with HTTP 401 Unauthorized", FailureKind.SITE_REFUSED,
         "Example.com refused the request: it needs a sign-in or a permission I don't have.", True),
        (f"GET to https://example.com failed with HTTP 403 Forbidden: {PAGE}", FailureKind.SITE_REFUSED,
         "Example.com refused the request: it needs a sign-in or a permission I don't have.", True),
        ("GET to https://example.com failed with HTTP 429 Too Many Requests", FailureKind.SITE_TROUBLE,
         "Example.com is overloaded or having trouble right now. Try again in a few minutes.", False),
        (f"GET to https://example.com failed with HTTP 503 Service Unavailable: {PAGE}", FailureKind.SITE_TROUBLE,
         "Example.com is overloaded or having trouble right now. Try again in a few minutes.", False),
        ("GET to https://example.com failed with HTTP 400 Bad Request", FailureKind.SITE_DECLINED,
         "Example.com turned the request down. The details are on screen.", False),
        ("Outbound request exceeded 5 redirects", FailureKind.REDIRECT_LOOP,
         "Example.com kept sending me to another address without reaching a page, so I stopped.", True),
        ("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate",
         FailureKind.CERTIFICATE, "I didn't open example.com: its security certificate could not be checked, so the "
         "connection could not be trusted.", True),
        ("[Errno 11001] getaddrinfo failed", FailureKind.NAME_NOT_RESOLVED,
         "I couldn't reach example.com: that address could not be found. Check its spelling and your connection.", False),
        ("ConnectTimeout: timed out", FailureKind.TIMED_OUT,
         "I couldn't reach example.com: it took too long to answer. You can ask me again in a moment.", False),
    ],
)  # fmt: skip
def test_an_http_outcome_is_said_by_what_a_person_can_do(error, kind, spoken, final) -> None:
    assert classify(error, platform="win32") is kind
    shown, said = failure_sentence(_get(), error, platform="win32")
    assert said == spoken
    assert is_final(_get(), error, platform="win32") is final
    assert "<" not in shown and "doctype" not in shown.casefold() and "Not here" not in shown
    assert "/missing" not in shown and "/missing" not in said
    assert ear_faults(said) == [] and screen_faults(shown) == []


def test_the_run_404_shows_the_status_and_never_the_page() -> None:
    error = (
        "GET to https://example.com failed with HTTP 404 Not Found (the server has nothing at this address; the same "
        f"request will get the same answer): {PAGE}"
    )
    said = failure_reply([ActionResult(action=_get(), success=False, error=error)], None)
    assert said.shown == (
        "example.com has no page at that address. Check the address and ask me again.\n"
        "The server answered: HTTP 404 Not Found"
    )


# ── git ───────────────────────────────────────────────────────────────────────────────────────────────────────────

RUN_REJECTED = (
    f"To {W}\\remote.git\n ! [rejected]        main -> main (fetch first)\nerror: failed to push some refs to "
    f"'{W}\\remote.git'\nhint: Updates were rejected because the remote contains work that you do not\nhint: have "
    "locally."
)


@pytest.mark.parametrize(
    ("error", "kind", "spoken"),
    [
        (RUN_REJECTED, FailureKind.PUSH_BEHIND,
         "I couldn't push main: origin has commits you don't have yet. Pull first, then push again."),
        (" ! [rejected] main -> main (non-fast-forward)", FailureKind.PUSH_BEHIND,
         "I couldn't push main: origin has commits you don't have yet. Pull first, then push again."),
        ("remote: error: GH006: Protected branch update failed for refs/heads/main.", FailureKind.PUSH_PROTECTED,
         "I couldn't push main: origin protects that branch, so it takes changes only through a pull request."),
        ("Nothing staged to commit — working tree may be clean.", FailureKind.NOTHING_TO_COMMIT,
         "There was nothing to commit in repo: no changes are staged."),
        ("fatal: not a git repository (or any of the parent directories): .git", FailureKind.NOT_A_REPOSITORY,
         "Repo is not a git repository."),
        ("fatal: 'origin' does not appear to be a git repository", FailureKind.NO_REMOTE,
         "I couldn't push main: repo has no remote called origin."),
        ("CONFLICT (content): Merge conflict in README.md", FailureKind.MERGE_CONFLICT,
         "I couldn't finish that in repo: there is a merge conflict to resolve first."),
        ("fatal: Authentication failed for 'https://github.com/me/repo.git/'", FailureKind.GIT_SIGN_IN,
         "I couldn't reach origin: it refused the sign-in. Check your Git credentials."),
    ],
)  # fmt: skip
def test_gits_common_endings_are_final_and_said_plainly(error, kind, spoken) -> None:
    assert classify(error) is kind
    assert is_final(_push(), error)
    shown, said = failure_sentence(_push(), error)
    assert said == spoken and shown == spoken


def test_the_reply_leads_with_the_asked_plans_failure_not_a_repair_step() -> None:
    push = ActionResult(action=_push(), success=False, error=RUN_REJECTED)
    shell = ActionResult(
        action=Action(action_type=ActionType.SHELL_COMMAND, target="git", parameters={"command": "git"}),
        success=False,
        error="Process exited with code 1",
    )
    said = failure_reply([shell, push], None, asked=([push], None))
    # The asked plan's failure leads; the repair step that ran and failed is one clause after it.
    assert said.spoken == (
        "I couldn't push main: origin has commits you don't have yet. Pull first, then push again. I tried again, "
        "and that failed too: the shell command step failed. The details are on screen."
    )
    assert said.shown.splitlines()[0] == (
        "I couldn't push main: origin has commits you don't have yet. Pull first, then push again."
    )


@pytest.mark.asyncio
async def test_a_rejected_push_is_not_repaired_and_asks_once() -> None:
    class _Planner:
        calls = 0

        async def plan(self, *args, **kwargs):
            _Planner.calls += 1
            return ActionPlan(actions=[_push()], explanation="Push main.")

    class _Executor:
        runs = 0

        async def execute(self, plan, **kwargs):
            _Executor.runs += 1
            return [ActionResult(action=a, success=False, error=RUN_REJECTED) for a in plan.actions]

    class _Verifier:
        async def verify(self, plan, results):
            return VerificationResult(passed=False, failed_actions=[0], details=["Action 0 (git_push): FAILED"])

    server = _server_ready_for_handle_execute(_Executor())
    server._planner, server._verifier = _Planner(), _Verifier()
    server._permission_checker = _FakePermissionChecker(requires_confirmation=True)
    ws = _FakeWs(confirmation=True)
    ws.server = server

    response = await server._handle_execute({"input": "push the repo to origin"}, ws)

    approvals = sum(1 for sent in ws.sent if json.loads(sent).get("method") == "confirm_required")
    assert (response["status"], _Planner.calls, _Executor.runs, approvals) == ("partial_failure", 1, 1, 1)
    assert (
        response["spoken"]
        == "I couldn't push main: origin has commits you don't have yet. Pull first, then push again."
    )


# ── a read answers the question asked ─────────────────────────────────────────────────────────────────────────────

LOG = W + r"\logs\app.log"
TAIL = "\n".join(f"2026-11-26 23:03:1{i} INFO job step 19{i}" for i in range(5, 10))
PROCESSES = "\n".join(
    [
        "PID=  4321  CPU= 12.5%  MEM=  9.1%  STATUS=running       chrome.exe",
        "PID=  4322  CPU=  2.5%  MEM=  6.0%  STATUS=running       chrome.exe",
        "PID=   812  CPU=  0.4%  MEM=  0.9%  STATUS=running       explorer.exe",
        "PID=  9001  CPU=  1.1%  MEM=  8.2%  STATUS=sleeping      Code.exe",
        "PID=  7002  CPU=  0.1%  MEM=  4.0%  STATUS=sleeping      Teams.exe",
    ]
)
SYSTEM = (
    "=== Operating System ===\n  system: Windows\n  release: 11\n=== CPU ===\n  Logical cores: 8\n  Average usage: 12.0%\n"
    "=== Memory ===\n  Total: 32.0 GB\n  Used:  12.0 GB (37.5%)\n  Free:  20.0 GB\n=== Disk ===\n"
    "  C:\\ (C:\\) - 1024.0 GB total, 896.0 GB used (87.5%), fstype=NTFS\n=== Network ===\n  Wi-Fi (UP):\n"
    "    IPv4: 203.0.113.7\n  Loopback Pseudo-Interface 1 (UP):\n    IPv4: 127.0.0.1"
)


@pytest.mark.parametrize(
    ("result", "request_words", "spoken"),
    [
        (_result(ActionType.FILE_READ, FileParams(path=LOG, tail_lines=5), TAIL, LOG), "show the last 5 lines of app.log",
         "The last 5 lines of app.log; the newest is: the twenty-sixth of November 23:03:19 INFO job step 199."),
        (_result(ActionType.FILE_READ, FileParams(path=W + r"\notes.txt", head_lines=2), "first\nsecond", W),
         "first two lines", "The first 2 lines of notes.txt; the first is: first."),
        (_result(ActionType.API_SCRAPE, ApiRequestParams(url="https://example.com", selector="title"), "Example Domain"),
         "what is the title of example.com", "The title of example.com is Example Domain."),
        (_result(ActionType.PROCESS_LIST, ProcessParams(), PROCESSES), "what programs are running",
         "4 programs are running; the biggest are Chrome, Code and Teams."),
        (_result(ActionType.PROCESS_LIST, ProcessParams(name="chrome"), PROCESSES), "is chrome running",
         "Chrome is running, as 5 processes, using fifteen percent of memory."),
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(), SYSTEM), "how much memory is in use",
         "12.0 of 32.0 gigabytes of memory are in use."),
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(), SYSTEM), "what programs are running",
         "The system information does not include which programs are running. This computer runs Windows 11, the "
         "processor is at 12 percent, and 12.0 of 32.0 gigabytes of memory are in use."),
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(), SYSTEM), "what is the CPU temperature",
         "The system information does not include the temperature. The processor is at 12 percent."),
        (_result(ActionType.FILE_LIST, FileParams(path=W),
                 "[f] a.txt  (1 bytes)\n[f] b.txt  (2 bytes)\n[d] old  (0 bytes)\nNewest: b.txt, a.txt"),
         "what is in work", "Work has 2 files and 1 folder; the newest are b.txt and a.txt."),
        (_result(ActionType.SERVICE_STATUS, ServiceParams(name="nginx"), "Active: active (running) since Mon"),
         "is nginx running", "The nginx service is running."),
        (_result(ActionType.GIT_STATUS, GitParams(repo_path=REPO), json.dumps(
            {"success": True, "output": "On branch: main", "staged": ["a"], "unstaged": ["b", "c"], "untracked": [],
             "branch": "main"})), "git status", "Repo is on main; 1 file is staged and 2 files have changes not staged."),
        (_result(ActionType.FILE_DELETE, FileParams(path=W + r"\archive\gone.txt"),
                 f"There was no file at {W}\\archive\\gone.txt, so nothing was deleted"), "delete gone.txt",
         "There was no gone.txt in archive, so nothing was deleted."),
    ],
)  # fmt: skip
def test_a_read_answers_the_question_asked(result: ActionResult, request_words: str, spoken: str) -> None:
    if result.action.action_type is ActionType.FILE_DELETE:
        result = result.model_copy(update={"skip_reason": "already_absent", "executed": False})
    said = success_reply(ActionPlan(actions=[result.action], raw_input=request_words), [result],
                         VerificationResult(passed=True), dry_run=False)  # fmt: skip
    assert said.spoken == spoken
    assert ear_faults(said.spoken) == [] and screen_faults(said.shown) == []
    assert "..." not in said.shown and "===" not in said.shown


def test_system_information_is_shown_as_a_summary_not_a_cut_dump() -> None:
    said = step_said(_result(ActionType.SYSTEM_INFO, SystemInfoParams(), SYSTEM))
    assert said.shown.splitlines() == [
        "This computer runs Windows 11, the processor is at 12 percent, and 12.0 of 32.0 gigabytes of memory are in use.",
        "System: Windows 11, 8 cores",
        "Processor: 12.0% in use",
        "Memory: 12.0 of 32.0 GB in use (37.5%)",
        "Drive C: 128 GB free of 1024.0 GB",
        "Network: Wi-Fi connected",
    ]


def test_search_lines_are_shown_relative_to_the_folder_searched() -> None:
    found = json.dumps(
        {
            "matches": [
                {"file": W + r"\workspace\plan.md", "line": 2, "content": "The annual report review"},
                {"file": W + r"\workspace\sub\notes.txt", "line": 1, "content": "Draft the annual report"},
            ],
            "count": 2,
        }
    )
    params = FileIntelParams(directory=W + r"\workspace", search_text="annual report")
    said = step_said(_result(ActionType.FILE_SEARCH_CONTENT, params, found))
    assert said.shown.splitlines()[1:] == [
        "- plan.md:2 The annual report review",
        "- sub\\notes.txt:1 Draft the annual report",
    ]


def test_two_triggers_on_one_file_are_told_apart_and_removed_together() -> None:
    def entry(name: str, job: str) -> dict:
        return {"id": name[:8], "name": name, "trigger_type": "file_modified",
                "condition": {"path": W + r"\trigger\watched.txt"}, "action_command": job}  # fmt: skip

    both = json.dumps([entry("notes-log", "echo file changed >> fired.log"), entry("watched-txt-tell", "")])
    listed = step_said(_result(ActionType.TRIGGER_LIST, TriggerParams(), both))
    assert listed.spoken == (
        'You have 2 triggers: one will add "file changed" to fired.log when watched.txt changes; one will tell '
        "you when watched.txt changes."
    )
    removed = [
        _result(ActionType.TRIGGER_DELETE, TriggerParams(trigger_id=e["id"]), json.dumps({**e, "deleted": True}))
        for e in json.loads(both)
    ]
    said = success_reply(ActionPlan(actions=[r.action for r in removed], raw_input="delete both"), removed,
                         VerificationResult(passed=True), dry_run=False)  # fmt: skip
    assert said.spoken == "I've removed 2 triggers; both watched watched.txt."


def test_nothing_matching_a_selector_and_a_service_not_connected() -> None:
    scrape = Action(
        action_type=ActionType.API_SCRAPE,
        target="https://example.com",
        parameters=ApiRequestParams(url="https://example.com", selector=".price-table td.amount"),
    )
    error = (
        "No element matches the selector '.price-table td.amount' on example.com. The page loaded (577 characters, "
        "title 'Example Domain')."
    )
    shown, said = failure_sentence(scrape, error)
    assert (
        said
        == "Nothing on example.com matches .price-table td.amount. The page loaded; what it does hold is on screen."
    )
    assert shown.splitlines()[1] == "The page loaded (577 characters, title 'Example Domain')."
    slack = Action(action_type=ActionType.API_SLACK, target="slack", parameters=ApiRequestParams(message="hi"))
    error = "The Slack message was not sent: Slack is not connected. Connect it in Settings > Connections > Slack."
    assert failure_sentence(slack, error)[1] == (
        "Slack isn't connected yet. Connect it in Settings, Connections, Slack, then ask me again."
    )
    assert is_final(slack, error)
