"""Mail, Slack, Discord and GitHub connections: Settings first, environment second, then a sentence naming Settings.

Fakes only: the credential store is a dict, HTTP answers come from a recorder, SMTP is a fake class. Nothing is
sent, posted or read from GitHub, and no real secret is read.
"""

from __future__ import annotations

import json
import smtplib
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.security.vault import VaultUnavailableError
from pilot.server import RESERVED_VAULT_NAMES, PilotServer
from pilot.system import api_client, connections

SLACK = "https://hooks.slack.com/services/T0FAKE/B0FAKE/fakeFakeFake1234"
SLACK_ENV = "https://hooks.slack.com/services/T0ENV/B0ENV/environmentFake99"
DISCORD = "https://discord.com/api/webhooks/123456789/fakeDiscordToken9876"
TOKEN = "ghp_fakefakefakefakefake0000"
BOT = "xoxb-fake-fake-fake"
PASSWORD = "fake-app-password-1234"
SECRETS = (SLACK, SLACK_ENV, DISCORD, TOKEN, BOT, PASSWORD, "fakeFakeFake1234")


class FakeVault:
    backend_name = "FakeKeyring"

    def __init__(self, available: bool = True, **entries: str) -> None:
        self.available = available
        self.entries: dict[str, str] = dict(entries)

    async def get_key(self, name: str) -> str | None:
        return self.entries.get(name) if self.available else None

    async def store_key(self, name: str, value: str) -> None:
        if not self.available:
            raise VaultUnavailableError("no credential store")
        self.entries[name] = value

    async def delete_key(self, name: str) -> None:
        if not self.available:
            raise VaultUnavailableError("no credential store")
        self.entries.pop(name, None)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    for variable in (*connections.ENVIRONMENT.values(), "SMTP_USER", "SMTP_PASS", "SMTP_SERVER", "SMTP_PORT"):
        monkeypatch.delenv(variable, raising=False)
    connections.configure(None, None)
    yield
    connections.configure(None, None)


class Calls(list):
    """The requests made, and the answers to give (an exception is raised) before the default 200."""

    def __init__(self) -> None:
        super().__init__()
        self.answers: list[Any] = []


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """api_request replaced: records each call and answers 200 unless told otherwise."""
    calls = Calls()
    answers = calls.answers

    async def _request(method: str, url: str, headers=None, body=None, *_a: Any, **_k: Any) -> str:
        calls.append({"method": method, "url": url, "headers": dict(headers or {}), "body": body})
        if answers:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return json.dumps({"status": 200, "body": {"login": "octo-fake", "id": "1"}})

    monkeypatch.setattr(api_client, "api_request", _request)
    return calls


def _config(enabled: bool = True) -> PilotConfig:
    config = PilotConfig()
    config.email.enabled = enabled
    config.email.smtp_host = "smtp.settings.example"
    config.email.smtp_port = 2525
    config.email.username = "me@settings.example"
    config.email.password_provider = "email"
    return config


def _no_secret(value: Any) -> None:
    text = json.dumps(value, default=str)
    for secret in SECRETS:
        assert secret not in text


# ── lookup order ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_settings_win_over_the_environment_and_the_status_says_both_are_set(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_ENV)
    connections.configure(_config(), FakeVault(slack_webhook=SLACK))

    found = await connections.service("slack")
    status = await connections.status()

    assert found.source == "settings" and found.get(connections.SLACK_WEBHOOK) == SLACK
    assert status["connections"]["slack"]["source"] == "settings"
    assert status["connections"]["slack"]["environment_also_set"] is True
    assert status["connections"]["slack"]["webhook"] == {"host": "hooks.slack.com", "last4": "1234"}
    _no_secret(status)


@pytest.mark.asyncio
async def test_someone_with_only_environment_variables_keeps_working_unchanged(monkeypatch, http) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_ENV)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    connections.configure(_config(enabled=False), FakeVault())

    await api_client.send_slack_message("hi")
    await api_client.github_api("/user")

    assert http[0]["url"] == SLACK_ENV
    assert http[1]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    status = await connections.status()
    assert status["connections"]["slack"]["source"] == "environment"
    assert status["connections"]["github"]["source"] == "environment"
    assert status["connections"]["github"]["environment_also_set"] is False


@pytest.mark.asyncio
async def test_on_linux_without_a_secret_service_the_environment_still_works_and_saving_says_why_not(
    monkeypatch, http
) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", DISCORD)
    vault = FakeVault(available=False)
    connections.configure(_config(), vault)

    await api_client.send_discord_message("hi")
    status = await connections.status()

    assert http[0]["url"].startswith(DISCORD)
    assert status["available"] is False and "Environment variables still work" in status["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send", "where"),
    [
        (lambda: api_client.send_slack_message("hi"), "Settings > Connections > Slack"),
        (lambda: api_client.send_discord_message("hi"), "Settings > Connections > Discord"),
        (lambda: api_client.github_api("/repos/o/r/issues", "POST", {"title": "x"}), "Settings > Connections > GitHub"),
        (lambda: api_client.send_email("you@example.com", "Hi", "Body"), "Settings > Email integration"),
    ],
)
async def test_with_nothing_connected_every_action_fails_and_names_where_to_connect(http, send, where) -> None:
    connections.configure(_config(enabled=False), FakeVault())

    with pytest.raises(ValueError) as failure:
        await send()

    assert where in str(failure.value)
    assert http == []


# ── no secret in a plan ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_github_uses_the_connected_token_and_a_plan_token_never_shows(http) -> None:
    from pilot.actions import ApiRequestParams

    connections.configure(_config(), FakeVault(github_token=TOKEN))
    await api_client.github_api("/user")
    assert http[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"

    params = ApiRequestParams(endpoint="/user", token="ghp_literalFakeInPlanWXYZ")
    for shown in (params.model_dump_json(), repr(params)):
        assert "ghp_literalFakeInPlanWXYZ" not in shown
    kept = ApiRequestParams.model_validate_json(params.model_dump_json())
    with pytest.raises(ValueError, match="hidden when the plan was kept"):
        await api_client.github_api("/user", token=kept.token)

    same = ApiRequestParams.model_validate_json(ApiRequestParams(endpoint="/user", token=TOKEN).model_dump_json())
    await api_client.github_api("/user", token=same.token)
    assert http[-1]["headers"]["Authorization"] == f"Bearer {TOKEN}"


@pytest.mark.asyncio
async def test_the_connected_github_token_never_goes_to_another_host(http) -> None:
    connections.configure(_config(), FakeVault(github_token=TOKEN))

    with pytest.raises(ValueError, match="api.github.com only"):
        await api_client.github_api("https://collector.example/steal")
    assert http == []


def test_the_mail_password_is_masked_in_every_dump() -> None:
    from pilot.actions import EmailParams

    params = EmailParams(username="me@example.com", app_password=PASSWORD)
    for shown in (params.model_dump_json(), repr(params), str(params)):
        assert PASSWORD not in shown
    assert params.app_password == PASSWORD


# ── rejected secrets ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "send", "status"),
    [
        ("Slack", lambda: api_client.send_slack_message("hi"), 404),
        ("Discord", lambda: api_client.send_discord_message("hi"), 401),
        ("GitHub", lambda: api_client.github_api("/user"), 401),
    ],
)
async def test_a_secret_the_service_rejects_fails_with_a_plain_reason(http, service, send, status) -> None:
    connections.configure(_config(), FakeVault(slack_webhook=SLACK, discord_webhook=DISCORD, github_token=TOKEN))
    http.answers.append(api_client.HttpStatusError("POST", "https://service.example/x", status, "", "invalid_token"))

    with pytest.raises(api_client.HttpStatusError) as failure:
        await send()

    text = str(failure.value)
    assert f"{service} refused the" in text and f"Settings > Connections > {service}" in text
    assert failure.value.status == status
    for secret in SECRETS:
        assert secret not in text


# ── mail: one account for both paths ─────────────────────────────────


class FakeSMTP:
    sent: list[tuple[str, int, str, str]] = []
    refuse_login = False

    def __init__(self, host: str, port: int, *_a: Any, **_k: Any) -> None:
        self.host, self.port = host, port

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def starttls(self, context: Any) -> None:
        return None

    def login(self, user: str, password: str) -> None:
        if FakeSMTP.refuse_login:
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 not accepted")
        self.user = user

    def sendmail(self, sender: str, to: str, message: str) -> dict:
        FakeSMTP.sent.append((self.host, self.port, self.user, sender))
        return {}


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.sent = []
    FakeSMTP.refuse_login = False
    monkeypatch.setattr(api_client.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


@pytest.mark.asyncio
async def test_api_send_email_sends_from_the_email_settings_account(monkeypatch, smtp) -> None:
    monkeypatch.setenv("SMTP_USER", "old@environment.example")
    monkeypatch.setenv("SMTP_PASS", "fake-environment-password")
    connections.configure(_config(), FakeVault(email=PASSWORD))

    output = await api_client.send_email("you@example.com", "Hi", "Body")

    assert smtp.sent == [("smtp.settings.example", 2525, "me@settings.example", "me@settings.example")]
    assert "Message-ID <" in str(output)
    assert await connections.mail_sender() == "me@settings.example"
    status = await connections.status()
    assert status["connections"]["mail"]["source"] == "settings"
    assert status["connections"]["mail"]["environment_also_set"] is True
    _no_secret(status)


@pytest.mark.asyncio
async def test_mail_falls_back_to_smtp_variables_when_settings_hold_no_password(monkeypatch, smtp) -> None:
    monkeypatch.setenv("SMTP_USER", "old@environment.example")
    monkeypatch.setenv("SMTP_PASS", "fake-environment-password")
    monkeypatch.setenv("SMTP_SERVER", "smtp.environment.example")
    connections.configure(_config(), FakeVault())

    await api_client.send_email("you@example.com", "Hi", "Body")

    assert smtp.sent[0][:3] == ("smtp.environment.example", 587, "old@environment.example")
    assert await connections.mail_sender() == "old@environment.example"


@pytest.mark.asyncio
async def test_a_refused_mail_sign_in_names_the_account_and_where_its_password_is(smtp) -> None:
    smtp.refuse_login = True
    connections.configure(_config(), FakeVault(email=PASSWORD))

    with pytest.raises(RuntimeError) as failure:
        await api_client.send_email("you@example.com", "Hi", "Body")

    assert "refused the sign-in for me@settings.example" in str(failure.value)
    assert "Settings > Email integration" in str(failure.value) and PASSWORD not in str(failure.value)


# ── the Settings RPCs ────────────────────────────────────────────────


def _server(vault: FakeVault) -> PilotServer:
    server = PilotServer(_config())
    server._vault = vault
    connections.configure(server.config, vault)
    return server


@pytest.mark.asyncio
async def test_saving_checks_the_value_and_answers_with_status_only() -> None:
    vault = FakeVault()
    server = _server(vault)

    bad = await server._handle_connection_save(
        {"service": "slack", "values": {"webhook_url": "https://evil.example/x"}}, MagicMock()
    )
    alone = await server._handle_connection_save({"service": "slack", "values": {"bot_token": BOT}}, MagicMock())
    saved = await server._handle_connection_save(
        {"service": "slack", "values": {"webhook_url": SLACK, "bot_token": BOT, "channel_id": "C0FAKE123"}}, MagicMock()
    )

    assert bad["status"] == "error" and bad["field"] == "webhook_url"
    assert alone["status"] == "error" and "webhook address first" in alone["message"]
    assert saved["status"] == "ok" and saved["connections"]["slack"]["read_back"] is True
    assert vault.entries == {"slack_webhook": SLACK, "slack_bot_token": BOT, "slack_channel_id": "C0FAKE123"}
    _no_secret(saved)
    for response in (bad, alone):
        _no_secret(response)


@pytest.mark.asyncio
async def test_removing_leaves_the_environment_alone_and_the_generic_key_rpcs_cannot_touch_connections(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_environmentFakeToken00")
    vault = FakeVault(github_token=TOKEN)
    server = _server(vault)

    removed = await server._handle_connection_remove({"service": "github"}, MagicMock())
    generic = await server._handle_store_api_key({"provider": "github_token", "api_key": "x"}, MagicMock())

    assert vault.entries == {}
    assert removed["connections"]["github"]["source"] == "environment"
    assert generic["status"] == "error"
    assert set(connections.STORE_NAMES) <= RESERVED_VAULT_NAMES


@pytest.mark.asyncio
async def test_without_a_credential_store_saving_is_refused_with_the_reason() -> None:
    server = _server(FakeVault(available=False))

    answer = await server._handle_connection_save({"service": "github", "values": {"token": TOKEN}}, MagicMock())

    assert answer["status"] == "error" and answer["available"] is False
    assert "no usable credential store" in answer["message"]


@pytest.mark.asyncio
async def test_a_chat_test_posts_only_after_the_person_confirms_and_github_reads_its_own_user(http) -> None:
    server = _server(FakeVault(slack_webhook=SLACK, github_token=TOKEN))

    unasked = await server._handle_connection_test({"service": "slack"}, MagicMock())
    assert unasked["status"] == "error" and http == []

    posted = await server._handle_connection_test({"service": "slack", "confirm_post": True}, MagicMock())
    github = await server._handle_connection_test({"service": "github"}, MagicMock())
    mail = await server._handle_connection_test({"service": "mail"}, MagicMock())

    assert posted["status"] == "ok" and http[0]["url"] == SLACK and http[0]["method"] == "POST"
    assert github == {"status": "ok", "message": "GitHub accepted the token for octo-fake."}
    assert http[1]["method"] == "GET" and http[1]["url"] == "https://api.github.com/user"
    assert mail["status"] == "error" and len(http) == 2


@pytest.mark.asyncio
async def test_opening_settings_sends_nothing_anywhere(http) -> None:
    server = _server(FakeVault(slack_webhook=SLACK, discord_webhook=DISCORD, github_token=TOKEN, email=PASSWORD))

    status = await server._handle_connections_status({}, MagicMock())

    assert http == []
    assert all(status["connections"][name]["connected"] for name in ("mail", "slack", "discord", "github"))
    _no_secret(status)


def test_every_connection_name_is_cleaned_up_on_uninstall() -> None:
    from pilot.security.vault import KNOWN_PROVIDERS

    assert set(connections.STORE_NAMES) <= set(KNOWN_PROVIDERS)


@pytest.mark.asyncio
async def test_the_slack_read_back_reason_names_settings(monkeypatch) -> None:
    from pilot.actions import Action, ActionResult, ActionType, ApiRequestParams
    from pilot.agents.postconditions import messaging

    connections.configure(_config(), FakeVault(slack_webhook=SLACK))
    action = Action(action_type=ActionType.API_SLACK, target="post", parameters=ApiRequestParams(message="hi"))
    outcome = await messaging.slack_message_in_channel(
        ActionResult(action=action, success=True, output=json.dumps({"status": 200, "body": "ok", "sent_at": 1.0}))
    )

    assert "Settings > Connections > Slack" in outcome.detail
