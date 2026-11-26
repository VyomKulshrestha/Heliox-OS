"""Declared post-conditions: the ``expect`` probe of the arbitrary-effect actions (postconditions/declared.py).

Fakes only: the ssh probe, the local command runner, HTTP, MCP and the browser are replaced on their modules. No test
opens a connection, runs a command or reads outside its temp folder.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    BrowserParams,
    CodeExecParams,
    DBusParams,
    Expectation,
    FileParams,
    McpToolParams,
    PermissionTier,
    ShellCommandParams,
    ShellScriptParams,
    SkillRunParams,
    SshCommandParams,
    SshScriptParams,
)
from pilot.agents.postconditions import declared
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier

# ---------------------------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------------------------


def ssh(command: str, host: str = "prod-1", expect: Expectation | None = None) -> Action:
    return Action(
        action_type=ActionType.SSH_COMMAND,
        target=host,
        parameters=SshCommandParams(host=host, command=command),
        expect=expect,
    )


def shell(command: str, args: list[str] | None = None, **flags: Any) -> Action:
    return Action(
        action_type=ActionType.SHELL_COMMAND,
        parameters=ShellCommandParams(command=command, args=args or [], **flags),
    )


def file_probe(kind: ActionType, path: str) -> Action:
    return Action(action_type=kind, target=path, parameters=FileParams(path=path))


def checked(action: Action, probe: Action, kind: str = "exit_code", value: Any = 0, field: str = "") -> Action:
    return action.model_copy(update={"expect": Expectation(probe=probe, kind=kind, value=value, field=field)})


def verify(action: Action, result: ActionResult | None = None):
    result = result or ActionResult(action=action, success=True, output="done")
    return asyncio.run(Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result]))


@pytest.fixture(autouse=True)
def _router_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Verifier binds its router into the module; put the original back after each test."""
    monkeypatch.setattr(declared, "_router", None)


@pytest.fixture
def ssh_probe(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    from pilot.agents import ssh_agent

    fake = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(ssh_agent.SshAgent, "run_probe", fake)
    return fake


@pytest.fixture
def local_runner(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    from pilot.system import platform_detect

    fake = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(platform_detect, "run_command", fake)
    return fake


TOUCH = ssh("touch /tmp/x", expect=Expectation(probe=ssh("test -f /tmp/x"), kind="exit_code", value=0))


# ---------------------------------------------------------------------------------------------------------------
# The analysis's acceptance: ssh_command with expect = ssh_command test -f /tmp/x
# ---------------------------------------------------------------------------------------------------------------


def test_ssh_probe_verifies_when_the_file_exists(ssh_probe: AsyncMock) -> None:
    ssh_probe.return_value = (0, "", "")

    verification = verify(TOUCH)

    assert verification.passed
    assert verification.independently_observed_actions == [0]
    assert verification.details[0] == (
        "Action 0 (ssh_command): VERIFIED (runs `test -f /tmp/x` on prod-1; it succeeded, as declared)"
    )
    ssh_probe.assert_awaited_once()
    assert ssh_probe.await_args.args == ("prod-1", "test -f /tmp/x")


def test_ssh_probe_gives_mismatch_when_the_file_is_missing(ssh_probe: AsyncMock) -> None:
    ssh_probe.return_value = (1, "", "")

    verification = verify(TOUCH)

    assert not verification.passed
    assert verification.failed_actions == [0]
    assert verification.details[0] == (
        "Action 0 (ssh_command): MISMATCH — runs `test -f /tmp/x` on prod-1: expected exit code 0, found exit code 1"
    )


def test_ssh_probe_exists_kind_reads_the_exit_code_as_presence(ssh_probe: AsyncMock) -> None:
    action = checked(TOUCH, ssh("test -f /tmp/x"), kind="exists", value=None)
    ssh_probe.return_value = (1, "", "")
    assert "expected it to be there, found it is not there" in verify(action).details[0]
    ssh_probe.return_value = (127, "", "test: not found")
    reported = verify(action)
    assert reported.passed and reported.independently_observed_actions == []
    assert "could not tell whether it is there (exit code 127)" in reported.details[0]


def test_an_unreachable_host_is_not_verified_and_does_not_fail_the_step(ssh_probe: AsyncMock) -> None:
    ssh_probe.side_effect = RuntimeError("Unknown SSH host alias 'prod-1'")

    verification = verify(TOUCH)

    assert verification.passed
    assert verification.independently_observed_actions == []
    assert verification.details[0].startswith("Action 0 (ssh_command): REPORTED — no independent postcondition (")
    assert "could not reach prod-1" in verification.details[0]


def test_a_probe_that_is_not_read_only_is_refused_and_never_runs(ssh_probe: AsyncMock) -> None:
    action = checked(TOUCH, ssh("rm -f /tmp/x"))

    verification = verify(action)

    assert verification.passed
    assert verification.independently_observed_actions == []
    assert (
        "not verified: the check was refused because 'rm' is not one of the read-only commands"
        in (verification.details[0])
    )
    ssh_probe.assert_not_awaited()


def test_without_expect_the_step_stays_reported(ssh_probe: AsyncMock) -> None:
    verification = verify(ssh("touch /tmp/x"))

    assert verification.passed
    assert verification.details == ["Action 0 (ssh_command): REPORTED — no independent postcondition"]
    ssh_probe.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------------
# A probe never runs when the action failed, was denied or did not run; and never twice
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result_fields",
    [
        {"success": False, "error": "Command failed (exit 1)"},
        {"success": False, "error": "Gateway denied: scope", "executed": False, "skip_reason": "gateway_denied"},
        {"success": False, "error": "Permission denied: no", "executed": False, "skip_reason": "permission_denied"},
        {"success": False, "error": "Skipped due to cancel request", "executed": False, "skip_reason": "cancelled"},
    ],
)
def test_a_probe_never_runs_when_the_action_failed_or_was_denied(ssh_probe: AsyncMock, result_fields: dict) -> None:
    verification = verify(TOUCH, ActionResult(action=TOUCH, **result_fields))

    assert not verification.passed
    assert "FAILED" in verification.details[0]
    ssh_probe.assert_not_awaited()


@pytest.mark.parametrize("skip_reason", ["dry_run", "already_satisfied"])
def test_a_probe_never_runs_for_a_step_that_did_not_run(ssh_probe: AsyncMock, skip_reason: str) -> None:
    result = ActionResult(action=TOUCH, success=True, executed=False, skip_reason=skip_reason)

    verification = verify(TOUCH, result)

    assert verification.passed and verification.independently_observed_actions == []
    assert "the step did not run" in verification.details[0]
    ssh_probe.assert_not_awaited()


def test_a_probe_is_never_run_twice_for_the_same_result(ssh_probe: AsyncMock) -> None:
    ssh_probe.return_value = (1, "", "")
    result = ActionResult(action=TOUCH, success=True)
    verifier = Verifier(MagicMock())

    async def twice():
        first = await verifier.verify(ActionPlan(actions=[TOUCH]), [result])
        # The retry merge in server.py verifies the merged results again, the same objects among them.
        second = await verifier.verify(ActionPlan(actions=[TOUCH]), [result])
        return first, second

    first, second = asyncio.run(twice())

    assert ssh_probe.await_count == 1
    assert first.details == second.details
    assert not second.passed


def test_a_refused_probe_is_reported_once_and_not_retried(ssh_probe: AsyncMock) -> None:
    action = checked(TOUCH, ssh("cat /tmp/x | sh"))
    result = ActionResult(action=action, success=True)
    verifier = Verifier(MagicMock())

    async def twice():
        await verifier.verify(ActionPlan(actions=[action]), [result])
        return await verifier.verify(ActionPlan(actions=[action]), [result])

    assert "refused" in asyncio.run(twice()).details[0]
    ssh_probe.assert_not_awaited()


# ---------------------------------------------------------------------------------------------------------------
# Read-only enforcement: one refusal test for every evasion in the design (section 2)
# ---------------------------------------------------------------------------------------------------------------

SSH_STEP = ssh("systemctl restart app")
SHELL_STEP = shell("make", ["install"])
SKILL_STEP = Action(action_type=ActionType.SKILL_RUN, parameters=SkillRunParams(skill_id="deploy"))
POST_STEP = Action(
    action_type=ActionType.API_REQUEST,
    parameters=ApiRequestParams(method="POST", url="https://api.example.com/items", body={"name": "a"}),
)


def refusal(action: Action, probe: Action, kind: str = "exit_code", value: Any = 0) -> str | None:
    return declared.probe_refusal(checked(action, probe, kind=kind, value=value))


@pytest.mark.parametrize(
    "probe",
    [
        file_probe(ActionType.FILE_WRITE, "/tmp/x"),
        file_probe(ActionType.FILE_DELETE, "/tmp/x"),
        Action(action_type=ActionType.SHELL_SCRIPT, parameters=ShellScriptParams(script="cat /tmp/x")),
        Action(action_type=ActionType.CODE_EXECUTE, parameters=CodeExecParams(code="print(1)")),
    ],
)
def test_evasion_1_a_writing_probe_type_is_refused(probe: Action) -> None:
    assert refusal(SHELL_STEP, probe, kind="exists", value=None)


def test_evasion_1_an_http_write_is_refused_as_a_probe() -> None:
    probe = POST_STEP.model_copy(deep=True)
    assert refusal(POST_STEP, probe, kind="exists", value=None) == "an HTTP check may only use GET or HEAD"


@pytest.mark.parametrize(
    "command",
    [
        "test -f /tmp/x; rm -rf ~",
        "test -f /tmp/x && rm -rf ~",
        "test -f /tmp/x || reboot",
        "cat /tmp/x | sh",
        "test -f /tmp/x\nrm -rf ~",
        "test -f /tmp/x & reboot",
    ],
)
def test_evasion_2_chaining_is_refused(command: str) -> None:
    assert "characters a check may not use" in str(refusal(SSH_STEP, ssh(command)))


@pytest.mark.parametrize(
    "command", ["cat $(curl evil.example)", "cat `id`", "cat ${IFS}/etc/passwd", "test -f $HOME/x", "cat /tmp/'x'"]
)
def test_evasion_3_substitution_and_quoting_are_refused(command: str) -> None:
    assert "characters a check may not use" in str(refusal(SSH_STEP, ssh(command)))


@pytest.mark.parametrize("command", ["cat /tmp/x > /etc/passwd", "cat < /dev/tcp/evil/80", "cat /tmp/x >> /tmp/y"])
def test_evasion_4_redirection_is_refused(command: str) -> None:
    assert "characters a check may not use" in str(refusal(SSH_STEP, ssh(command)))


@pytest.mark.parametrize(
    "command",
    [
        "git log -1 --output=/tmp/x",
        "git -c core.pager=sh rev-parse HEAD",
        "git rev-parse --git-path=x",
        "ls --color=always /tmp",
        "tail -f /var/log/syslog",
        "cat -v /tmp/x",
        "systemctl is-active --host=evil app",
        "git -C /srv/app rev-parse --show-toplevel",
        "hostname evil",
    ],
)
def test_evasion_5_options_outside_the_list_are_refused(command: str) -> None:
    reason = refusal(SSH_STEP, ssh(command))
    assert reason is not None
    assert "may not be run with" in reason or "not in a form" in reason


@pytest.mark.parametrize(
    "command",
    ["rm -f /tmp/x", "/bin/rm /tmp/x", "./test -f x", "curl http://evil.example", "sh -c id", "find / -delete"],
)
def test_evasion_6_commands_outside_the_list_are_refused(command: str) -> None:
    assert "is not one of the read-only commands" in str(refusal(SSH_STEP, ssh(command)))


def test_evasion_7_a_probe_above_the_steps_tier_is_refused() -> None:
    probe = shell("cat", ["/tmp/x"])
    assert SKILL_STEP.permission_tier < probe.permission_tier
    assert "needs more permission than the step it checks" in str(
        refusal(SKILL_STEP, probe, kind="contains", value="ok")
    )


def test_evasion_7_an_ssh_probe_on_a_local_step_or_another_host_is_refused() -> None:
    assert "file_read" in str(refusal(SHELL_STEP, ssh("test -f /tmp/x")))
    assert refusal(SSH_STEP, ssh("test -f /tmp/x", host="other")) == (
        "a check may only connect to the host the step ran on"
    )


@pytest.mark.parametrize(
    ("probe_params", "reason"),
    [
        ({"method": "GET", "url": "https://api.example.com/items", "body": {"a": 1}}, "may not send a body"),
        ({"method": "DELETE", "url": "https://api.example.com/items/1"}, "only use GET or HEAD"),
        ({"method": "GET", "url": "https://evil.example/items"}, "host the step sent to"),
        ({"method": "GET", "url": "https://api.example.com.evil.example/"}, "host the step sent to"),
        (
            {"method": "GET", "url": "https://api.example.com/items", "allow_private_network": True},
            "private network",
        ),
        (
            {"method": "GET", "url": "https://api.example.com/items", "headers": {"X-HTTP-Method-Override": "DELETE"}},
            "x-http-method-override",
        ),
    ],
)
def test_evasion_8_an_http_probe_that_writes_or_reaches_elsewhere_is_refused(probe_params: dict, reason: str) -> None:
    probe = Action(action_type=ActionType.API_REQUEST, parameters=ApiRequestParams(**probe_params))
    assert reason in str(refusal(POST_STEP, probe, kind="exists", value=None))


def test_evasion_8_a_same_host_get_is_accepted() -> None:
    probe = Action(
        action_type=ActionType.API_REQUEST,
        parameters=ApiRequestParams(method="GET", url="https://API.example.com/items/1", headers={"Accept": "x"}),
    )
    assert (
        declared.probe_refusal(
            POST_STEP.model_copy(
                update={"expect": Expectation(probe=probe, kind="json_equals", field="name", value="a")}
            )
        )
        is None
    )


def test_evasion_9_a_local_probe_into_helioxs_private_folder_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.config import PilotConfig
    from pilot.security.validator import heliox_private_dirs

    router = MagicMock()
    router.get_config.return_value = PilotConfig()
    monkeypatch.setattr(declared, "_router", router)
    secret = str(heliox_private_dirs()[0] / "config.toml")
    action = checked(SHELL_STEP, file_probe(ActionType.FILE_READ, secret), kind="contains", value="x")

    assert declared.probe_refusal(action) is None  # allowed by type ...
    assert "same rules as any action refuse it" in str(declared.validator_refusal(action))  # ... refused by rules
    assert "refused" in str(declared.describe_check(action))


def test_evasion_9_found_text_is_short_and_masked() -> None:
    key = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ\n-----END OPENSSH PRIVATE KEY-----"
    expect = Expectation(probe=ssh("cat /home/a/.ssh/id_ed25519"), kind="equals", value="x")

    outcome = declared.compare(expect, declared.Reading(output=key, exit_code=0, present=True), "reads")

    assert not outcome.passed
    assert "PRIVATE KEY" not in outcome.detail.replace("[REDACTED]", "")
    assert "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ" not in outcome.detail
    long = declared.compare(expect, declared.Reading(output="y" * 5000, exit_code=0, present=True), "reads")
    assert len(long.detail) < 260
    token = declared.compare(expect, declared.Reading(output="token ghp_abcdefghijklmnop1234", present=True), "r")
    assert "ghp_abcdefghijklmnop1234" not in token.detail


def test_evasion_10_an_mcp_probe_must_be_a_read_only_tool_of_the_same_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot import mcp_client

    tiers = {("github", "get_issue"): PermissionTier.READ_ONLY}
    monkeypatch.setattr(
        mcp_client,
        "tool_permission_tier",
        lambda server, tool: tiers.get((server, tool), PermissionTier.SYSTEM_MODIFY),
    )

    def mcp(server: str, tool: str) -> Action:
        return Action(
            action_type=ActionType.MCP_CALL_TOOL,
            parameters=McpToolParams(mcp_server=server, mcp_tool=tool, mcp_arguments={"number": 1}),
        )

    step = mcp("github", "create_issue")
    assert refusal(step, mcp("github", "get_issue"), kind="contains", value="open") is None
    assert "does not declare itself read-only" in str(refusal(step, mcp("github", "close_issue"), "contains", "x"))
    assert "server the step used" in str(refusal(step, mcp("other", "get_issue"), "contains", "x"))


def test_evasion_11_a_probe_cannot_nest_another_action() -> None:
    inner = Expectation(probe=ssh("test -f /tmp/y"), kind="exit_code", value=0)
    with pytest.raises(ValidationError):
        Expectation(probe=ssh("test -f /tmp/x", expect=inner), kind="exit_code", value=0)
    with pytest.raises(ValidationError):
        Expectation(probe=ssh("test -f /tmp/x").model_copy(update={"rollback_action": ssh("reboot")}), kind="exists")


@pytest.mark.parametrize(
    "probe",
    [
        ssh("test -f /tmp/x").model_copy(update={"requires_root": True}),
        ssh("test -f /tmp/x").model_copy(update={"destructive": True}),
        shell("cat", ["/tmp/x"], elevated=True),
    ],
)
def test_evasion_12_an_elevated_probe_is_refused(probe: Action) -> None:
    step = SSH_STEP if probe.action_type == ActionType.SSH_COMMAND else SHELL_STEP
    assert refusal(step, probe) == "a check may not run elevated or be marked destructive"


def test_evasion_13_tilde_and_percent_are_plain_words() -> None:
    assert declared.command_refusal(["cat", "~/app/ready"]) is None
    assert declared.command_refusal(["git", "-C", "~/repo", "log", "-1", "--format=%H"]) is None


def test_evasion_14_a_probe_that_would_hang_is_refused_or_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "may not be run with -f" in str(declared.command_refusal(["tail", "-f", "/var/log/x"]))

    async def slow(_probe: Action) -> declared.Reading:
        raise TimeoutError

    monkeypatch.setattr(declared, "run_probe", slow)
    verification = verify(TOUCH)
    assert verification.passed and verification.independently_observed_actions == []
    assert "did not finish in time" in verification.details[0]


def test_a_page_script_is_checked_only_by_reading_the_page() -> None:
    step = Action(action_type=ActionType.BROWSER_EXECUTE_JS, parameters=BrowserParams(script="submit()"))
    more_script = Action(action_type=ActionType.BROWSER_EXECUTE_JS, parameters=BrowserParams(script="return 1"))
    many = Action(action_type=ActionType.BROWSER_EXTRACT, parameters=BrowserParams(selector="p", multiple=True))
    html = Action(action_type=ActionType.BROWSER_EXTRACT, parameters=BrowserParams(selector="p", attribute="innerHTML"))
    page = Action(action_type=ActionType.BROWSER_PAGE_INFO, parameters=BrowserParams())

    assert "only be checked by reading the page" in str(refusal(step, more_script, "contains", "1"))
    assert "reads one element" in str(refusal(step, many, "contains", "1"))
    assert "innerText, value" in str(refusal(step, html, "contains", "1"))
    assert (
        declared.probe_refusal(
            step.model_copy(
                update={"expect": Expectation(probe=page, kind="json_equals", field="url", value="https://a/done")}
            )
        )
        is None
    )


def test_a_dbus_probe_only_reads_properties_of_the_same_service() -> None:
    def dbus(service: str, interface: str, method: str) -> Action:
        return Action(
            action_type=ActionType.DBUS_CALL,
            parameters=DBusParams(service=service, object_path="/a", interface=interface, method=method),
        )

    step = dbus("org.example.App", "org.example.App", "Enable")
    assert refusal(step, dbus("org.example.App", "org.freedesktop.DBus.Properties", "Get"), "contains", "x") is None
    assert "Properties Get or GetAll" in str(
        refusal(step, dbus("org.example.App", "org.freedesktop.DBus.Properties", "Set"), "contains", "x")
    )
    assert "bus and service" in str(
        refusal(step, dbus("org.other", "org.freedesktop.DBus.Properties", "Get"), "contains", "x")
    )


def test_a_webhook_has_no_read_back_and_is_not_registered() -> None:
    hook = Action(
        action_type=ActionType.API_WEBHOOK,
        parameters=ApiRequestParams(method="POST", url="https://hooks.example/x"),
    )
    probe = Action(action_type=ActionType.API_REQUEST, parameters=ApiRequestParams(url="https://hooks.example/x"))
    assert "receipt" in str(refusal(hook, probe, "exists", None))
    assert ActionType.API_WEBHOOK not in POSTCONDITION_VERIFIERS
    assert "receipt" in str(declared.describe_check(hook))


def test_the_fifteen_are_registered_as_declared() -> None:
    for action_type in declared.DECLARED_ACTIONS:
        assert POSTCONDITION_VERIFIERS[action_type] == "declared_expect_postcondition"
    assert len(declared.DECLARED_ACTIONS) == 15


def test_a_read_api_call_takes_no_check() -> None:
    get = Action(action_type=ActionType.API_REQUEST, parameters=ApiRequestParams(method="GET", url="https://a.example"))
    assert "a read changes nothing" in str(refusal(get, get.model_copy(deep=True), "exists", None))


# ---------------------------------------------------------------------------------------------------------------
# The model rules
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "value", "field"),
    [
        ("exit_code", "0", ""),
        ("exit_code", 300, ""),
        ("exit_code", True, ""),
        ("equals", "", ""),
        ("contains", "x" * 501, ""),
        ("pattern", "*" * 9, ""),
        ("json_equals", "a", ""),
        ("json_equals", "a", "a..b"),
        ("exists", "x", ""),
        ("contains", "x", "a"),
        ("regex", "x", ""),
    ],
)
def test_malformed_expectations_are_refused_by_the_model(kind: str, value: Any, field: str) -> None:
    with pytest.raises(ValidationError):
        Expectation(probe=ssh("test -f /tmp/x"), kind=kind, value=value, field=field)


# ---------------------------------------------------------------------------------------------------------------
# Comparison and local probes
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "value", "field", "output", "passed"),
    [
        ("equals", "active", "", "active\n", True),
        ("equals", "active", "", "inactive\n", False),
        ("contains", "ready", "", "service is ready now", True),
        ("pattern", "v1.*", "", "v1.4.2\n", True),
        ("pattern", "v2.*", "", "v1.4.2\n", False),
        ("json_equals", "merged", "pull.state", '{"pull": {"state": "merged"}}', True),
        ("json_equals", 3, "items.1.id", '{"items": [{"id": 1}, {"id": 3}]}', True),
        ("json_equals", "true", "ok", '{"ok": true}', True),
        ("json_equals", "merged", "pull.state", '{"pull": {"state": "open"}}', False),
        ("json_equals", "merged", "pull.state", "not json", False),
    ],
)
def test_compare(kind: str, value: Any, field: str, output: str, passed: bool) -> None:
    expect = Expectation(probe=ssh("cat /tmp/x"), kind=kind, value=value, field=field)
    outcome = declared.compare(expect, declared.Reading(output=output, exit_code=0, present=True), "reads x")
    assert outcome.passed is passed
    assert outcome.observed


def test_mismatch_says_what_was_expected_and_found() -> None:
    expect = Expectation(probe=ssh("systemctl is-active app"), kind="equals", value="active")
    outcome = declared.compare(expect, declared.Reading(output="inactive\n", exit_code=0, present=True), "runs it")
    assert outcome.detail == "runs it: expected the output `active`, found `inactive`"


def test_local_file_probe_reads_the_file_system(tmp_path) -> None:
    target = tmp_path / "out.txt"
    step = Action(action_type=ActionType.CODE_EXECUTE, parameters=CodeExecParams(code="open('out.txt','w')"))
    action = checked(step, file_probe(ActionType.FILE_READ, str(target)), kind="contains", value="done")

    assert "expected output containing `done`, found nothing there" in verify(action).details[0]
    target.write_text("all done", encoding="utf-8")
    fresh = verify(action)
    assert fresh.passed and fresh.independently_observed_actions == [0]
    absent = checked(step, file_probe(ActionType.FILE_READ, str(target)), kind="absent", value=None)
    assert "expected it to be gone, found it is still there" in verify(absent).details[0]


def test_local_shell_probe_runs_without_a_shell(local_runner: AsyncMock) -> None:
    step = shell("make", ["install"])
    action = checked(step, shell("git -C /srv/app rev-parse HEAD"), kind="equals", value="abc123")
    local_runner.return_value = (0, "abc123\n", "")

    verification = verify(action)

    assert verification.passed and verification.independently_observed_actions == [0]
    assert local_runner.await_args.args[0] == ["git", "-C", "/srv/app", "rev-parse", "HEAD"]


def test_a_missing_local_tool_is_not_verified(local_runner: AsyncMock) -> None:
    local_runner.return_value = (-1, "", "Command not found: sha256sum")
    action = checked(shell("make", ["dist"]), shell("sha256sum /srv/dist.tar"), kind="contains", value="ab")

    verification = verify(action)

    assert verification.passed and verification.independently_observed_actions == []
    assert "'sha256sum' is not available on this computer" in verification.details[0]


def test_http_probe_reads_status_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.config import PilotConfig
    from pilot.system import api_client, http_client

    class _Response:
        def __init__(self, status: int, text: str) -> None:
            self.status_code, self.text = status, text

    calls: list[tuple[str, str]] = []
    answers = iter([_Response(200, '{"name": "a"}'), _Response(404, "")])

    async def fake_request(_client, method, url, **_kwargs):
        calls.append((method, url))
        return next(answers)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(api_client, "_request_with_validated_redirects", fake_request)
    monkeypatch.setattr(http_client, "create_httpx_client", lambda *a, **k: _Client())
    monkeypatch.setattr(declared, "_router", None)  # restored after the test; Verifier(router) binds its own
    router = MagicMock()
    router.get_config.return_value = PilotConfig()
    probe = Action(
        action_type=ActionType.API_REQUEST,
        parameters=ApiRequestParams(method="GET", url="https://api.example.com/items/1"),
    )
    action = POST_STEP.model_copy(
        update={"expect": Expectation(probe=probe, kind="json_equals", field="name", value="a")}
    )

    verifier = Verifier(router)
    first = asyncio.run(verifier.verify(ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]))
    second = asyncio.run(verifier.verify(ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]))

    assert first.passed and first.independently_observed_actions == [0]
    assert not second.passed and "found nothing there" in second.details[0]
    assert calls == [("GET", "https://api.example.com/items/1")] * 2


# ---------------------------------------------------------------------------------------------------------------
# The plan card's line
# ---------------------------------------------------------------------------------------------------------------


def test_plan_card_line_says_the_check_or_why_there_is_none() -> None:
    assert declared.describe_check(TOUCH) == "Runs `test -f /tmp/x` on prod-1 and expects it to succeed."
    assert declared.describe_check(ssh("touch /tmp/x")) == (
        "None declared: Heliox will report what the step said, without checking it."
    )
    refused = declared.describe_check(checked(TOUCH, ssh("rm /tmp/x")))
    assert refused is not None and refused.startswith("None: the check was refused because 'rm'")
    assert declared.describe_check(file_probe(ActionType.FILE_WRITE, "/tmp/x")) is None
    # A reversible local step without a check says nothing new: it is reported, as today.
    assert declared.describe_check(shell("make", ["install"])) is None


def test_ssh_script_steps_are_checked_over_ssh_too(ssh_probe: AsyncMock) -> None:
    step = Action(
        action_type=ActionType.SSH_SCRIPT,
        parameters=SshScriptParams(host="prod-1", script="systemctl restart app\n"),
    )
    action = checked(step, ssh("systemctl is-active app"), kind="equals", value="active")
    ssh_probe.return_value = (0, "active\n", "")
    verification = verify(action)
    assert verification.passed and verification.independently_observed_actions == [0]
