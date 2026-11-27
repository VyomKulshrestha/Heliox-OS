"""Round thirteen: what the third live run found in the replies.

- Every read's shown reply starts with the answer sentence that is said; a reader's own printout (section headers,
  whoami's table, process rows, Windows line ends) is never shown as the reply.
- Uptime, the time and the computer's name are answered from system information; the total memory or disk size
  when the question asks how much there is.
- A step skipped because its outcome already held never leads a plan's reply.
- A trigger removed while another still watches the same file says so.
- A repair that was refused or failed is said in one clause after the failure.
- A step Heliox's own rules refused is its own failure, "gateway_denied", not "could not be reached".
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
    EmptyParams,
    FileParams,
    ProcessParams,
    SystemInfoParams,
    TriggerParams,
    VerificationResult,
)
from pilot.failure_sentences import FailureKind, classify, failure_sentence, is_final
from pilot.reply_rules import ear_faults, for_ear, screen_faults
from pilot.response_contract import failure_reply, success_reply
from pilot.result_sentences import machine_output, plan_said, step_said
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute
from tests.test_reply_contract import CASES

W = r"C:\Users\me\work"
UPTIME = (
    "=== System Time ===\n  Current Local Time: Tuesday, March 3, 2026 4:05:09 PM\n  Timezone: Newfoundland Standard Time\n"
    "  Up for: 2 day(s), 3 hour(s), 0 minute(s)"
)
UPTIME_WEST = (
    "=== System Time ===\n  Current Local Time: Sunday, July 19, 2026 7:30:00 AM\n  Timezone: Pacific Daylight Time\n"
    "  Up for: 0 day(s), 0 hour(s), 45 minute(s)"
)
NETWORK = (
    "=== Network ===\n  Ethernet (DOWN):\n    IPv4: 169.254.77.8\n  Wi-Fi (UP):\n    IPv4: 192.0.2.15\n"
    "    IPv6: 2001:db8::15\n  vEthernet (WSL (Hyper-V firewall)) (UP):\n    IPv4: 198.51.100.1\n"
    "  Loopback Pseudo-Interface 1 (UP):\n    IPv4: 127.0.0.1"
)
WHOAMI = (
    "USER INFORMATION\r\n----------------\r\n\r\nUser Name        SID                                           \r\n"
    "================ ==============================================\r\nstudio-pc\\sam   S-1-5-21-1000-2000-3000-1001\r\n"
)
MEMORY = "=== Memory ===\n  Total: 16.0 GB\n  Used:  12.0 GB (75.0%)\n  Free:  4.0 GB"
DISK = "=== Disk ===\n  C:\\ (C:\\) - 512.0 GB total, 400.0 GB used (78.1%), fstype=NTFS"
PROCESSES = "\n".join(
    [
        "PID=     0  CPU=1895.2%  MEM=  0.0%  STATUS=running       System Idle Process",
        "PID=   336  CPU=  0.0%  MEM=  0.4%  STATUS=stopped       ",
        "PID=  4321  CPU= 12.5%  MEM=  5.5%  STATUS=running       pixelforge.exe",
        "PID=  4322  CPU= 30.0%  MEM=  5.2%  STATUS=running       notenest.exe",
        "PID=   812  CPU=  0.4%  MEM=  1.7%  STATUS=running       SyncDaemon",
    ]
)


def _result(kind: ActionType, params: object, output: str, target: str = "", **extra) -> ActionResult:
    action = Action(action_type=kind, target=target, parameters=params)
    return ActionResult(action=action, success=True, output=output, **extra)


def _said(result: ActionResult, request: str):
    plan = ActionPlan(actions=[result.action], raw_input=request)
    return success_reply(plan, [result], VerificationResult(passed=True), dry_run=False)


# ── the first line of every read's shown reply is its spoken answer ──────────────────────────────────────────────

READS = [kind for kind in CASES if kind in __import__("pilot.actions").actions.READ_ONLY_ACTIONS]


@pytest.mark.parametrize("kind", READS, ids=[kind.value for kind in READS])
def test_every_reads_shown_reply_starts_with_its_spoken_answer(kind: ActionType) -> None:
    params, target, output = CASES[kind]
    said = _said(_result(kind, params, output, target), "tell me")
    first = said.shown.splitlines()[0]
    assert for_ear(first) == said.spoken, (kind, said)
    assert not machine_output(said.shown), (kind, said.shown)
    assert "characters long, starting" not in said.spoken or kind in {
        ActionType.CLIPBOARD_READ,
        ActionType.FILE_READ,
        ActionType.SCREEN_OCR,
        ActionType.BROWSER_EXTRACT,
    }


@pytest.mark.parametrize(
    ("result", "request_words", "shown", "spoken"),
    [
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(categories=["time"]), UPTIME),
         "How long has this computer been running?",
         "This computer has been up for 2 days and 3 hours.",
         "This computer has been up for 2 days and 3 hours."),
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(categories=["time"]), UPTIME), "What time is it?",
         "It's 4:05 PM on Tuesday.", "It's four oh five p m on Tuesday."),
        # A zone behind UTC, and an uptime under an hour.
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(categories=["time"]), UPTIME_WEST), "What time is it?",
         "It's 7:30 AM on Sunday.", "It's seven thirty a m on Sunday."),
        (_result(ActionType.SYSTEM_INFO, SystemInfoParams(categories=["time"]), UPTIME_WEST),
         "How long has this computer been up?", "This computer has been up for 45 minutes.",
         "This computer has been up for 45 minutes."),
        (_result(ActionType.NETWORK_INFO, SystemInfoParams(), NETWORK), "What is my IP address?",
         "Your IP address is 192.0.2.15, on Wi-Fi.", "Your IP address is 192.0.2.15, on Wi-Fi."),
        (_result(ActionType.USER_INFO, EmptyParams(), WHOAMI), "Which user am I logged in as?",
         "You're logged in as sam.", "You're logged in as sam."),
        (_result(ActionType.MEMORY_USAGE, SystemInfoParams(), MEMORY), "How much RAM does this computer have?",
         "This computer has 16.0 gigabytes of memory; 12.0 are in use.",
         "This computer has 16.0 gigabytes of memory; 12.0 are in use."),
        (_result(ActionType.DISK_USAGE, SystemInfoParams(), DISK), "How big is my disk?",
         "Drive C holds 512.0 gigabytes; 112 are free.", "Drive C holds 512.0 gigabytes; 112 are free."),
        (_result(ActionType.DISK_USAGE, SystemInfoParams(), DISK), "How much space is left?",
         "You have 112 gigabytes free on drive C.", "You have 112 gigabytes free on drive C."),
        (_result(ActionType.PROCESS_LIST, ProcessParams(), PROCESSES), "What is using the most memory right now?",
         "Pixelforge uses the most memory, 5.5%; then Notenest and SyncDaemon.",
         "Pixelforge uses the most memory, five point five percent; then Notenest and SyncDaemon."),
        (_result(ActionType.PROCESS_LIST, ProcessParams(), PROCESSES), "What is using the CPU?",
         "Notenest is using the most processor time, 30.0%; then Pixelforge and SyncDaemon.",
         "Notenest is using the most processor time, thirty point zero percent; then Pixelforge and SyncDaemon."),
        (_result(ActionType.PROCESS_LIST, ProcessParams(), PROCESSES), "What programs are running?",
         "4 programs are running; the biggest are Pixelforge, Notenest and SyncDaemon.",
         "4 programs are running; the biggest are Pixelforge, Notenest and SyncDaemon."),
    ],
)  # fmt: skip
def test_the_third_runs_reads(result: ActionResult, request_words: str, shown: str, spoken: str) -> None:
    said = _said(result, request_words)
    assert said.shown.splitlines()[0] == shown
    assert said.spoken == spoken
    assert not machine_output(said.shown) and "STATUS=" not in said.shown and "\r" not in said.shown
    assert ear_faults(said.spoken) == [] and screen_faults(said.shown) == []


# ── a skipped step never leads ───────────────────────────────────────────────────────────────────────────────────

TRIGGER = json.dumps(
    {"id": "t1f2a3b4", "name": "notes-log", "trigger_type": "file_modified",
     "condition": {"path": W + r"\trigger\watched.txt"}, "action_command": "echo file changed >> fired.log"}
)  # fmt: skip
ALREADY = (
    "Already satisfied: The trigger engine's evaluation loop is running, and the store keeps it running after a restart"
)


def test_a_step_skipped_because_it_already_held_never_leads_and_is_not_counted() -> None:
    created = _result(ActionType.TRIGGER_CREATE, TriggerParams(trigger_type="file_modified"), TRIGGER)
    started = _result(
        ActionType.TRIGGER_START, TriggerParams(), ALREADY, skip_reason="already_satisfied", executed=False
    )
    said = plan_said([created, started], "watch watched.txt and log changes to fired.log")
    assert said.spoken == 'I\'ll watch watched.txt and add "file changed" to fired.log when it changes.'
    assert "Already satisfied" not in said.shown and "steps" not in said.spoken


def test_already_satisfied_and_absent_notices_read_as_sentences() -> None:
    started = _result(
        ActionType.TRIGGER_START, TriggerParams(), ALREADY, skip_reason="already_satisfied", executed=False
    )
    assert step_said(started).spoken == "Your triggers were already on."
    other = _result(
        ActionType.SERVICE_START,
        __import__("pilot.actions").actions.ServiceParams(name="nginx"),
        "Already satisfied: The nginx service is already running",
        skip_reason="already_satisfied",
        executed=False,
    )
    assert step_said(other).spoken == "Nothing needed doing: the nginx service is already running."


# ── a trigger removed while another still watches the same file ──────────────────────────────────────────────────


def test_removing_one_of_two_triggers_on_a_file_says_what_remains() -> None:
    record = {**json.loads(TRIGGER), "watched": "watched.txt", "deleted": True, "others_watching": 1}
    said = step_said(_result(ActionType.TRIGGER_DELETE, TriggerParams(trigger_id="t1f2a3b4"), json.dumps(record)))
    assert said.spoken == (
        'I\'ve removed the trigger that adds "file changed" to fired.log. The other one still watches watched.txt.'
    )
    alone = {**record, "others_watching": 0}
    said = step_said(_result(ActionType.TRIGGER_DELETE, TriggerParams(trigger_id="t1f2a3b4"), json.dumps(alone)))
    assert said.spoken == "I've stopped watching watched.txt."


# ── what happened to the retry ───────────────────────────────────────────────────────────────────────────────────

REFUSED = "I could not keep a copy of the earlier file, so I did not replace it."


def _copy() -> Action:
    return Action(
        action_type=ActionType.FILE_COPY,
        target=W + r"\big\source.bin",
        parameters=FileParams(path=W + r"\big\source.bin", destination=W + r"\big\copy.bin"),
    )


def test_a_refused_retry_and_a_failed_retry_are_said_after_the_failure() -> None:
    first = ActionResult(action=_copy(), success=True, output="Copied it")
    mismatch = VerificationResult(
        passed=False, failed_actions=[0], details=["Action 0 (file_copy): MISMATCH — Copy destination content mismatch"]
    )
    refused = ActionResult(action=_copy(), success=False, executed=False, skip_reason="no_undo_copy", error=REFUSED)
    said = failure_reply([refused], None, asked=([first], mismatch))
    assert said.spoken == (
        "That didn't work for copy.bin: the check afterwards found something else. I did not try again: I could "
        "not keep a copy of the earlier file, so I did not replace it."
    )
    tried = ActionResult(action=_copy(), success=False, error="[Errno 28] No space left on device")
    said = failure_reply([tried], None, asked=([first], mismatch))
    assert said.spoken.endswith(
        "I tried again, and that failed too: I couldn't copy source.bin in big: the disk is full. Free some space "
        "and ask me again."
    )


@pytest.mark.asyncio
async def test_the_run_copy_mismatch_with_a_refused_retry_says_both() -> None:
    class _Planner:
        async def plan(self, *args, **kwargs):
            return ActionPlan(actions=[_copy()], explanation="Copy it.")

    class _Executor:
        runs = 0

        async def execute(self, plan, **kwargs):
            _Executor.runs += 1
            if _Executor.runs == 1:
                return [ActionResult(action=a, success=True, output="Copied it") for a in plan.actions]
            return [
                ActionResult(action=a, success=False, executed=False, skip_reason="no_undo_copy", error=REFUSED)
                for a in plan.actions
            ]

    class _Verifier:
        async def verify(self, plan, results):
            if all(result.success for result in results):
                return VerificationResult(
                    passed=False,
                    failed_actions=[0],
                    details=["Action 0 (file_copy): MISMATCH — Copy destination content mismatch"],
                )
            return VerificationResult(passed=False, failed_actions=[0], details=["Action 0 (file_copy): FAILED"])

    server = _server_ready_for_handle_execute(_Executor())
    server._planner, server._verifier = _Planner(), _Verifier()
    server._permission_checker = _FakePermissionChecker()
    response = await server._handle_execute({"input": "copy source.bin to copy.bin"}, _FakeWs())

    assert response["status"] == "partial_failure"
    assert response["spoken"].startswith("That didn't work for copy.bin: the check afterwards found something else.")
    assert "I did not try again: I could not keep a copy of the earlier file" in response["spoken"]


# ── refused by Heliox's own rules ────────────────────────────────────────────────────────────────────────────────


def test_a_step_heliox_refused_is_its_own_final_failure() -> None:
    get = Action(
        action_type=ActionType.API_REQUEST,
        target="http://127.0.0.1:8797/busy",
        parameters=ApiRequestParams(url="http://127.0.0.1:8797/busy", allow_private_network=True),
    )
    error = (
        "Gateway denied: api_request (tier=SYSTEM_MODIFY, family=browsing) exceeds the 'integration_agent' source's "
        "allowed ceiling (READ_ONLY)."
    )
    assert classify(error) is FailureKind.GATEWAY_DENIED
    assert FailureKind.GATEWAY_DENIED.value == "gateway_denied"
    assert is_final(get, error)
    shown, said = failure_sentence(get, error)
    assert (
        said
        == shown
        == ("I didn't call 127.0.0.1: Heliox's own rules don't allow that request from here, so nothing was sent.")
    )
    assert "reach" not in said and "integration_agent" not in shown
