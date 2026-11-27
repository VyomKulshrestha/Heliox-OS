"""A handler that did not do what was asked must fail the action, never return a string the executor reads as success.

The executor marks an action successful whenever its handler returns (executor.py, ``_execute_single``); these tests
drive that path with fakes for SMTP, HTTP, the registry, the scheduler of triggers and the platform, so nothing is sent,
posted or changed.
"""

from __future__ import annotations

import json
import os
import smtplib
import stat

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    FileParams,
    HumanHelpParams,
    RegistryParams,
    TriggerParams,
)
from pilot.agents.executor import Executor
from pilot.agents.verifier import Verifier
from pilot.system import api_client, platform_detect
from pilot.system.platform_detect import Platform

SECRET_PATH = "/services/T000/B000/secret-token-in-path"


def _executor(action_type: ActionType, handler: str, **attributes: object) -> Executor:
    executor = object.__new__(Executor)
    executor._stress_gate = None
    for name, value in attributes.items():
        setattr(executor, name, value)
    executor._dispatch_table = {action_type: getattr(executor, handler)}
    return executor


async def _run(action_type: ActionType, handler: str, parameters: object, **attributes: object) -> ActionResult:
    action = Action(action_type=action_type, target="test", parameters=parameters)
    return await _executor(action_type, handler, **attributes)._execute_single(action, snapshot_id=None)


# ── HTTP fakes (no network) ──────────────────────────────────────────


class _Response:
    def __init__(self, url: str, status_code: int, text: str) -> None:
        self.url = url
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = text
        self.reason_phrase = {404: "Not Found", 500: "Internal Server Error", 400: "Bad Request"}.get(status_code, "")

    def json(self) -> object:
        raise ValueError("not json")


class _Client:
    def __init__(self, status_code: int, text: str, requests: list) -> None:
        self._status_code = status_code
        self._text = text
        self._requests = requests

    async def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self._requests.append((method, url, kwargs))
        return _Response(url, self._status_code, self._text)


class _ClientContext:
    def __init__(self, client: _Client) -> None:
        self._client = client

    async def __aenter__(self) -> _Client:
        return self._client

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch):
    """Answer every outbound request with the status the test sets; record what was asked."""

    state = {"status": 200, "text": "ok", "requests": []}

    def factory(*_args: object, **_kwargs: object) -> _ClientContext:
        return _ClientContext(_Client(state["status"], state["text"], state["requests"]))

    async def no_dns(_url: str, *, allow_private_network: bool) -> None:
        return None

    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", factory)
    monkeypatch.setattr(api_client, "_validate_outbound_url", no_dns)
    return state


# ── Row 1 and 2: api_send_email ─────────────────────────────────────


class _SMTP:
    """Fake SMTP server; ``fail`` names the step that raises."""

    sent: list[tuple[str, str]] = []

    def __init__(self, fail: str | None) -> None:
        self._fail = fail

    def __call__(self, host: str, port: int) -> _SMTP:
        return self

    def __enter__(self) -> _SMTP:
        return self

    def __exit__(self, *_args: object) -> None:
        if self._fail == "quit":
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    def starttls(self, context: object) -> None:
        if self._fail == "starttls":
            raise smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server.")

    def login(self, user: str, password: str) -> None:
        if self._fail == "login":
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

    def sendmail(self, sender: str, to: str, message: str) -> dict:
        if self._fail == "sendmail":
            raise smtplib.SMTPRecipientsRefused({to: (550, b"No such user")})
        self.sent.append((sender, to))
        return {}


def _email() -> ApiRequestParams:
    return ApiRequestParams(to="friend@example.com", subject="Hello", message="Body text")


@pytest.mark.asyncio
async def test_send_email_without_smtp_credentials_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASS", raising=False)

    result = await _run(ActionType.API_SEND_EMAIL, "_exec_api_send_email", _email())

    assert result.success is False
    assert "no mail account is connected" in (result.error or "")
    assert "Settings > Email integration" in (result.error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["starttls", "login", "sendmail"])
async def test_send_email_fails_when_smtp_throws_before_the_server_accepts(
    monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    monkeypatch.setattr(api_client.smtplib, "SMTP", _SMTP(step))

    result = await _run(ActionType.API_SEND_EMAIL, "_exec_api_send_email", _email())

    assert result.success is False
    assert "was not sent" in (result.error or "")
    assert "app-password" not in (result.error or "")


@pytest.mark.asyncio
async def test_send_email_accepted_then_disconnect_stays_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after the server took the message must not invite a second send."""
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    _SMTP.sent = []
    monkeypatch.setattr(api_client.smtplib, "SMTP", _SMTP("quit"))

    result = await _run(ActionType.API_SEND_EMAIL, "_exec_api_send_email", _email())

    assert result.success is True
    assert _SMTP.sent == [("sender@example.com", "friend@example.com")]


# ── Rows 3 and 4: api_slack, api_discord without a webhook URL ──────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "handler", "variable", "service"),
    [
        (ActionType.API_SLACK, "_exec_api_slack", "SLACK_WEBHOOK_URL", "Slack"),
        (ActionType.API_DISCORD, "_exec_api_discord", "DISCORD_WEBHOOK_URL", "Discord"),
    ],
)
async def test_chat_message_without_webhook_url_fails(
    monkeypatch: pytest.MonkeyPatch, http: dict, action_type: ActionType, handler: str, variable: str, service: str
) -> None:
    monkeypatch.delenv(variable, raising=False)

    result = await _run(action_type, handler, ApiRequestParams(message="Deploy finished"))

    assert result.success is False
    assert f"The {service} message was not sent: {service} is not connected" in (result.error or "")
    assert f"Settings > Connections > {service}" in (result.error or "")
    assert http["requests"] == []


# ── Row 5: HTTP 4xx and 5xx through api_request ─────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 500])
@pytest.mark.parametrize(
    ("action_type", "handler", "parameters"),
    [
        (ActionType.API_REQUEST, "_exec_api_request", ApiRequestParams(method="POST", url="https://8.8.8.8/items")),
        (
            ActionType.API_SLACK,
            "_exec_api_slack",
            ApiRequestParams(message="hi", webhook_url=f"https://hooks.slack.example{SECRET_PATH}"),
        ),
        (
            ActionType.API_DISCORD,
            "_exec_api_discord",
            ApiRequestParams(message="hi", webhook_url=f"https://discord.example/api/webhooks{SECRET_PATH}"),
        ),
        (
            ActionType.API_WEBHOOK,
            "_exec_api_webhook",
            ApiRequestParams(method="POST", url=f"https://hooks.example{SECRET_PATH}", body={"a": 1}),
        ),
        (
            ActionType.API_GITHUB,
            "_exec_api_github",
            ApiRequestParams(
                method="POST", endpoint="/repos/o/r/issues", body={"title": "x"}, token="ghp_fake-token-in-plan"
            ),
        ),
    ],
)
async def test_http_error_status_fails_the_action(
    http: dict, status: int, action_type: ActionType, handler: str, parameters: ApiRequestParams
) -> None:
    http["status"] = status
    http["text"] = "invalid_payload"

    result = await _run(action_type, handler, parameters)

    assert result.success is False
    assert f"HTTP {status}" in (result.error or "")
    assert "invalid_payload" in (result.error or "")
    assert "secret-token-in-path" not in (result.error or "")
    assert len(http["requests"]) == 1


@pytest.mark.asyncio
async def test_http_success_status_still_succeeds(http: dict) -> None:
    http["status"] = 201
    http["text"] = '{"id": 7}'

    result = await _run(ActionType.API_REQUEST, "_exec_api_request", ApiRequestParams(url="https://8.8.8.8/items"))

    assert result.success is True
    assert '"status": 201' in (result.output or "")


@pytest.mark.asyncio
async def test_scrape_of_an_error_page_fails(http: dict) -> None:
    http["status"] = 404
    http["text"] = "<html>Not here</html>"

    result = await _run(ActionType.API_SCRAPE, "_exec_api_scrape", ApiRequestParams(url="https://8.8.8.8/page"))

    assert result.success is False
    assert "HTTP 404" in (result.error or "")


# ── Row 6: trigger_delete when nothing matches ──────────────────────


@pytest.mark.asyncio
async def test_trigger_delete_of_an_unknown_trigger_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import triggers

    engine = triggers.TriggerEngine()
    engine.create_trigger("disk-alert", "disk_threshold", {"threshold": 95}, "tell me")
    monkeypatch.setattr(triggers, "_engine", engine)

    result = await _run(ActionType.TRIGGER_DELETE, "_exec_trigger_delete", TriggerParams(name="cpu-alert"))

    assert result.success is False
    assert 'No trigger matches "cpu-alert", so nothing was deleted.' in (result.error or "")
    assert [t["name"] for t in engine.list_triggers()] == ["disk-alert"]


@pytest.mark.asyncio
async def test_trigger_delete_of_a_known_trigger_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import triggers

    engine = triggers.TriggerEngine()
    made = engine.create_trigger("cpu-alert", "cpu_threshold", {"threshold": 90}, "tell me")
    monkeypatch.setattr(triggers, "_engine", engine)

    result = await _run(ActionType.TRIGGER_DELETE, "_exec_trigger_delete", TriggerParams(name="cpu-alert"))

    assert result.success is True
    record = json.loads(result.output)
    assert record["id"] == made.id and record["name"] == "cpu-alert" and record["trigger_type"] == "cpu_threshold"
    assert record["text"] == f'Deleted the trigger "cpu-alert" (cpu threshold) (id {made.id}).'
    assert engine.list_triggers() == []


# ── Row 7: registry_write (and registry_read) off Windows ───────────


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.LINUX, Platform.MACOS])
@pytest.mark.parametrize(
    ("action_type", "handler"),
    [(ActionType.REGISTRY_WRITE, "_exec_registry_write"), (ActionType.REGISTRY_READ, "_exec_registry_read")],
)
async def test_registry_action_off_windows_fails(
    monkeypatch: pytest.MonkeyPatch, platform: Platform, action_type: ActionType, handler: str
) -> None:
    async def no_powershell(*_args: object, **_kwargs: object) -> tuple[int, str, str]:
        raise AssertionError("PowerShell must not run off Windows")

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(platform_detect, "run_powershell", no_powershell)
    parameters = RegistryParams(key_path=r"HKCU\Software\Heliox", value_name="x", value_data="1")

    result = await _run(action_type, handler, parameters)

    assert result.success is False
    assert "The registry exists only on Windows" in (result.error or "")


# ── Row 8: file_permissions with a mode on Windows ──────────────────


@pytest.mark.asyncio
async def test_file_permissions_set_on_windows_fails_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pilot.system import filesystem

    target = tmp_path / "notes.txt"
    target.write_text("x", encoding="utf-8")
    before = os.stat(target).st_mode
    monkeypatch.setattr(filesystem, "CURRENT_PLATFORM", Platform.WINDOWS)

    result = await _run(
        ActionType.FILE_PERMISSIONS, "_exec_file_permissions", FileParams(path=str(target), permissions="600")
    )

    assert result.success is False
    assert "Windows has no Unix permission modes" in (result.error or "")
    assert os.stat(target).st_mode == before


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="chmod sets Unix modes only off Windows")
async def test_file_permissions_set_off_windows_still_succeeds(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("x", encoding="utf-8")

    result = await _run(
        ActionType.FILE_PERMISSIONS, "_exec_file_permissions", FileParams(path=str(target), permissions="600")
    )

    assert result.success is True
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


# ── Row 9: request_human_help without a takeover gate ───────────────


@pytest.mark.asyncio
async def test_request_human_help_without_takeover_fails() -> None:
    result = await _run(
        ActionType.REQUEST_HUMAN_HELP,
        "_exec_request_human_help",
        HumanHelpParams(help_needed="Enter the 2FA code"),
    )

    assert result.success is False
    assert "the person was not asked: Enter the 2FA code" in (result.error or "")


# ── The failure reaches verification and the repair decision ───────


@pytest.mark.asyncio
async def test_a_failed_send_is_reported_failed_and_not_retried_without_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pilot.server import _is_terminal_execution_failure

    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASS", raising=False)
    action = Action(action_type=ActionType.API_SEND_EMAIL, target="test", parameters=_email())
    result = await _executor(ActionType.API_SEND_EMAIL, "_exec_api_send_email")._execute_single(
        action, snapshot_id=None
    )

    verification = await Verifier(object()).verify(ActionPlan(actions=[action]), [result])

    assert verification.passed is False
    assert "FAILED" in verification.details[0]
    assert _is_terminal_execution_failure([result]) is True


@pytest.mark.asyncio
async def test_an_overloaded_site_is_left_to_the_normal_repair_path_and_a_missing_page_is_final(http: dict) -> None:
    from pilot.server import _is_terminal_execution_failure

    # 503: the site is overloaded or broken now and may answer later (pilot.failure_sentences.SITE_TROUBLE).
    http["status"] = 503
    result = await _run(ActionType.API_REQUEST, "_exec_api_request", ApiRequestParams(url="https://8.8.8.8/x"))
    assert _is_terminal_execution_failure([result]) is False

    # 404: the site answered that the page is not there; the same request gets the same answer (PAGE_NOT_THERE).
    http["status"] = 404
    result = await _run(ActionType.API_REQUEST, "_exec_api_request", ApiRequestParams(url="https://8.8.8.8/x"))
    assert _is_terminal_execution_failure([result]) is True


# ── Row 10: notify when the desktop notification could not be shown ──


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.MACOS, Platform.LINUX])
async def test_notification_not_shown_fails_and_the_reply_quotes_its_text(
    monkeypatch: pytest.MonkeyPatch, platform: Platform
) -> None:
    from pilot.actions import NotifyParams
    from pilot.response_contract import partial_failure_message
    from pilot.server import _is_terminal_execution_failure

    async def refused(*_args: object, **_kwargs: object) -> tuple[int, str, str]:
        return 1, "", "no notification service"

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(platform_detect, "run_command", refused)
    monkeypatch.setattr(platform_detect, "run_powershell", refused)
    action = Action(
        action_type=ActionType.NOTIFY,
        target="all",
        parameters=NotifyParams(summary="Backup", body="finished at 14:00"),
    )
    result = await _executor(ActionType.NOTIFY, "_exec_notify")._execute_single(action, snapshot_id=None)
    verification = await Verifier(object()).verify(ActionPlan(actions=[action]), [result])

    assert result.success is False
    assert "Could not show a desktop notification (no notification service)" in (result.error or "")
    reply = partial_failure_message([result], verification)
    assert "Backup: finished at 14:00" in reply
    assert _is_terminal_execution_failure([result]) is True


@pytest.mark.asyncio
async def test_notification_shown_still_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.actions import NotifyParams

    async def shown(*_args: object, **_kwargs: object) -> tuple[int, str, str]:
        return 0, "", ""

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "run_command", shown)

    result = await _run(ActionType.NOTIFY, "_exec_notify", NotifyParams(summary="Backup", body="done"))

    assert result.success is True
    assert result.output == "Notification sent: Backup"


# ── A partly refused email: a success whose refused recipients reach the reply and the receipt ──


class _PartlyRefusingSMTP(_SMTP):
    def sendmail(self, sender: str, to: str, message: str) -> dict:
        return {"gone@example.com": (550, b"No such user")}


@pytest.mark.asyncio
async def test_partly_refused_email_names_the_refused_recipients_to_the_person(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pilot.config import ReceiptsConfig
    from pilot.response_contract import success_message
    from pilot.workflows.task_receipts import TaskReceipts

    class _Vault:
        available = True

        def __init__(self) -> None:
            self.keys: dict[str, str] = {}

        async def get_key(self, name: str) -> str | None:
            return self.keys.get(name)

        async def store_key(self, name: str, value: str) -> None:
            self.keys[name] = value

    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    monkeypatch.setattr(api_client.smtplib, "SMTP", _PartlyRefusingSMTP(None))
    parameters = ApiRequestParams(to="friend@example.com, gone@example.com", subject="Hello", message="Body text")
    action = Action(action_type=ActionType.API_SEND_EMAIL, target="friend@example.com", parameters=parameters)

    result = await _executor(ActionType.API_SEND_EMAIL, "_exec_api_send_email")._execute_single(
        action, snapshot_id=None
    )
    verification = await Verifier(object()).verify(ActionPlan(actions=[action]), [result])
    reply = success_message(ActionPlan(actions=[action]), [result], verification, dry_run=False)
    receipts = TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=_Vault())
    receipts.record_action("task-1", "plan-1", action, result, user_confirmed=True)
    short = await receipts.finalize("task-1", request="send it", status="success", summary="Sent.", source="voice")
    envelope = await receipts.get(short["receipt_id"])

    assert result.success is True
    assert result.notice == "The mail server refused the email for: gone@example.com"
    assert type(result.output) is str
    assert "The mail server refused the email for: gone@example.com" in reply
    assert envelope["receipt"]["actions"][0]["notice"] == result.notice
    assert "Body text" not in json.dumps(envelope)
