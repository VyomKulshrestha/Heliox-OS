"""Slack and Discord sends are read back, or reported as a receipt, never trusted as written.

Fakes only: no request leaves the test. The webhook token below must never appear in a detail, a log line or
the handler's output.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams
from pilot.agents.postconditions import messaging
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import api_client

SECRET = "s3cr3t-webhook-token-value"
SLACK_HOOK = f"https://hooks.slack.com/services/T000/B000/{SECRET}"
DISCORD_HOOK = f"https://discord.com/api/webhooks/123456/{SECRET}"
SENT_AT = 1_790_000_000.0


class _ClientContext:
    def __init__(self, client: object) -> None:
        self._client = client

    async def __aenter__(self) -> object:
        return self._client

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Response:
    def __init__(self, url: str, status_code: int, body: object) -> None:
        self.url = url
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> object:
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class _Network:
    """Records every request; answers from a list in order."""

    def __init__(self, *answers: tuple[int, object]) -> None:
        self.answers = list(answers)
        self.requests: list[tuple[str, str, dict]] = []

    async def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.requests.append((method, url, kwargs))
        status, body = self.answers.pop(0)
        return _Response(url, status, body)


def _install_network(monkeypatch: pytest.MonkeyPatch, network: _Network) -> None:
    async def _public(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", lambda *_a, **_k: _ClientContext(network))
    monkeypatch.setattr(api_client, "_validate_outbound_url", _public)


@pytest.fixture(autouse=True)
def _no_settings(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    for name in ("SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID", "DISCORD_WEBHOOK_URL"):
        monkeypatch.delenv(name, raising=False)
    pauses: list[float] = []

    async def _pause(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(messaging, "_pause", _pause)
    return pauses


def _slack_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_HOOK)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C0123")


def _action(action_type: ActionType, message: str = "Deploy finished", **params: object) -> Action:
    return Action(action_type=action_type, target="chat", parameters=ApiRequestParams(message=message, **params))


def _receipt(**fields: object) -> str:
    return json.dumps({"status": 200, "host": "example", "sent_at": SENT_AT, **fields})


async def _verify(action: Action, output: str):
    return await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output=output)]
    )


def test_both_senders_are_registered_from_the_messaging_module() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.API_SLACK] == "conditional_slack_read_back_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.API_DISCORD] == "discord_message_read_back_postcondition"
    # api_webhook has no read-back; it belongs to the declared post-condition contract.
    assert ActionType.API_WEBHOOK not in POSTCONDITION_VERIFIERS


# ── handlers ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_handler_returns_a_receipt_without_the_webhook_url(monkeypatch, caplog) -> None:
    network = _Network((200, "ok"))
    _install_network(monkeypatch, network)
    caplog.set_level(logging.DEBUG)

    output = await api_client.send_slack_message("Deploy finished", SLACK_HOOK)

    receipt = json.loads(output)
    assert receipt["status"] == 200 and receipt["body"] == "ok" and receipt["host"] == "hooks.slack.com"
    assert isinstance(receipt["sent_at"], float)
    assert SECRET not in output and SECRET not in caplog.text
    assert network.requests[0][1] == SLACK_HOOK


@pytest.mark.asyncio
async def test_discord_handler_waits_for_the_message_and_keeps_its_id(monkeypatch, caplog) -> None:
    network = _Network((200, {"id": "987654321", "channel_id": "555", "content": "Deploy finished"}))
    _install_network(monkeypatch, network)
    caplog.set_level(logging.DEBUG)

    output = await api_client.send_discord_message("Deploy finished", DISCORD_HOOK + "?thread_id=42")

    receipt = json.loads(output)
    assert receipt["message_id"] == "987654321" and receipt["channel_id"] == "555"
    assert "content" not in receipt and "Deploy finished" not in output
    assert SECRET not in output and SECRET not in caplog.text
    sent_url = network.requests[0][1]
    assert "wait=true" in sent_url and "thread_id=42" in sent_url


@pytest.mark.asyncio
async def test_a_handler_failure_never_reaches_the_verifier(monkeypatch) -> None:
    async def _never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("no read-back for a failed send")

    monkeypatch.setattr(messaging, "_http_get", _never)
    action = _action(ActionType.API_DISCORD)
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=False, error="connection refused")]
    )

    assert verification.passed is False
    assert "FAILED" in verification.details[0]


# ── Slack ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_without_a_token_is_an_honest_receipt(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_HOOK)

    verification = await _verify(_action(ActionType.API_SLACK), _receipt(body="ok"))

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]
    detail = await messaging.slack_message_in_channel(
        ActionResult(action=_action(ActionType.API_SLACK), success=True, output=_receipt(body="ok"))
    )
    assert "receipt, no read-back" in detail.detail and "SLACK_BOT_TOKEN" in detail.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [{"status": 404, "body": "no_service"}, {"status": 200, "body": "invalid_payload"}])
async def test_slack_refusal_in_the_receipt_is_a_mismatch(answer) -> None:
    verification = await _verify(_action(ActionType.API_SLACK), _receipt(**answer))

    assert verification.passed is False
    assert "MISMATCH" in verification.details[0]


@pytest.mark.asyncio
async def test_slack_message_found_in_history_is_verified(monkeypatch, _no_settings) -> None:
    _slack_settings(monkeypatch)
    calls: list[dict] = []

    async def _get(url: str, *, headers=None, params=None):
        calls.append({"url": url, "headers": headers, "params": params})
        return 200, {"ok": True, "messages": [{"text": "other", "ts": "1"}, {"text": "Deploy  finished", "ts": "2"}]}

    monkeypatch.setattr(messaging, "_http_get", _get)

    verification = await _verify(_action(ActionType.API_SLACK), _receipt(body="ok"))

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert verification.details[0].endswith("VERIFIED")
    assert calls[0]["url"] == messaging.SLACK_HISTORY_URL
    assert calls[0]["params"]["channel"] == "C0123"
    assert float(calls[0]["params"]["oldest"]) == SENT_AT - messaging.SLACK_CLOCK_MARGIN_SECONDS
    assert _no_settings == []


@pytest.mark.asyncio
async def test_slack_entities_and_markup_still_match(monkeypatch) -> None:
    _slack_settings(monkeypatch)

    async def _get(*_args, **_kwargs):
        return 200, {"ok": True, "messages": [{"text": "Q&amp;A at 5 &lt;3", "ts": "3"}]}

    monkeypatch.setattr(messaging, "_http_get", _get)

    verification = await _verify(_action(ActionType.API_SLACK, "Q&A at 5 <3"), _receipt(body="ok"))

    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_slack_message_absent_after_the_retries_is_a_mismatch_and_says_so(monkeypatch, _no_settings) -> None:
    _slack_settings(monkeypatch)
    calls: list[int] = []

    async def _get(*_args, **_kwargs):
        calls.append(1)
        return 200, {"ok": True, "messages": [{"text": "something else", "ts": "4"}]}

    monkeypatch.setattr(messaging, "_http_get", _get)

    verification = await _verify(_action(ActionType.API_SLACK), _receipt(body="ok"))

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "shows no message with its text" in verification.details[0]
    assert "Deploy finished" not in verification.details[0]
    assert len(calls) == messaging.SLACK_READ_ATTEMPTS
    assert _no_settings == [messaging.SLACK_READ_PAUSE_SECONDS] * (messaging.SLACK_READ_ATTEMPTS - 1)


@pytest.mark.asyncio
async def test_slack_message_that_appears_on_the_second_read_is_verified(monkeypatch) -> None:
    _slack_settings(monkeypatch)
    answers = [
        (200, {"ok": True, "messages": []}),
        (200, {"ok": True, "messages": [{"text": "Deploy finished", "ts": "5"}]}),
    ]

    async def _get(*_args, **_kwargs):
        return answers.pop(0)

    monkeypatch.setattr(messaging, "_http_get", _get)

    verification = await _verify(_action(ActionType.API_SLACK), _receipt(body="ok"))

    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        (200, {"ok": False, "error": "not_in_channel"}),
        (429, None),
        (503, None),
        (401, None),
        (200, None),  # an answer that is not JSON
        (200, {"ok": True}),  # no message list
    ],
)
async def test_slack_read_path_that_cannot_look_is_reported_never_failed(monkeypatch, answer) -> None:
    _slack_settings(monkeypatch)

    async def _get(*_args, **_kwargs):
        return answer

    monkeypatch.setattr(messaging, "_http_get", _get)

    verification = await _verify(_action(ActionType.API_SLACK), _receipt(body="ok"))

    assert verification.passed is True
    assert verification.independently_observed_actions == []


@pytest.mark.asyncio
async def test_slack_with_a_plan_webhook_or_a_workflow_hook_is_not_read_back(monkeypatch) -> None:
    _slack_settings(monkeypatch)

    async def _never(*_args, **_kwargs):
        raise AssertionError("the configured channel is not this webhook's channel")

    monkeypatch.setattr(messaging, "_http_get", _never)
    other = _action(ActionType.API_SLACK, webhook_url="https://hooks.slack.com/services/T1/B1/other")
    workflow = _action(ActionType.API_SLACK, webhook_url="https://hooks.slack.com/triggers/T1/1/abc")

    for action in (other, workflow):
        verification = await _verify(action, _receipt(body="ok"))
        assert verification.passed is True
        assert verification.independently_observed_actions == []


# ── Discord ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discord_message_read_back_with_matching_content_is_verified(monkeypatch, caplog) -> None:
    network = _Network((200, {"id": "987654321", "content": "Deploy finished"}))
    _install_network(monkeypatch, network)
    caplog.set_level(logging.DEBUG)
    action = _action(ActionType.API_DISCORD, webhook_url=DISCORD_HOOK + "?thread_id=42")

    verification = await _verify(action, _receipt(message_id="987654321"))

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    method, url, kwargs = network.requests[0]
    assert method == "GET"
    assert url == f"https://discord.com/api/webhooks/123456/{SECRET}/messages/987654321"
    assert kwargs["params"] == {"thread_id": "42"}
    assert SECRET not in str(verification) and SECRET not in caplog.text


@pytest.mark.asyncio
async def test_discord_404_is_a_mismatch_without_the_token(monkeypatch, caplog) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", DISCORD_HOOK)
    _install_network(monkeypatch, _Network((404, {"message": "Unknown Message", "code": 10008})))
    caplog.set_level(logging.DEBUG)

    verification = await _verify(_action(ActionType.API_DISCORD), _receipt(message_id="987654321"))

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "HTTP 404" in verification.details[0] and "discord.com" in verification.details[0]
    assert SECRET not in str(verification) and SECRET not in caplog.text


@pytest.mark.asyncio
async def test_discord_message_with_other_content_is_a_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", DISCORD_HOOK)
    _install_network(monkeypatch, _Network((200, {"id": "987654321", "content": "Deploy fin"})))

    verification = await _verify(_action(ActionType.API_DISCORD), _receipt(message_id="987654321"))

    assert verification.passed is False
    assert "text differs" in verification.details[0]
    assert "Deploy" not in verification.details[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt, answer",
    [
        (_receipt(), None),  # no message id (an older handler, or a webhook proxy)
        (_receipt(message_id="987654321"), (404, {"message": "Unknown Webhook", "code": 10015})),
        (_receipt(message_id="987654321"), (429, {"retry_after": 1})),
        (_receipt(message_id="987654321"), (200, {"id": "1", "content": "Deploy finished"})),  # not that message
        (_receipt(message_id="987654321"), (200, "<html>maintenance</html>")),
        ("Sent", None),  # not a receipt at all
    ],
)
async def test_discord_cases_that_cannot_be_read_are_reported(monkeypatch, receipt, answer) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", DISCORD_HOOK)
    _install_network(monkeypatch, _Network(*([answer] if answer else [])))

    verification = await _verify(_action(ActionType.API_DISCORD), receipt)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_discord_offline_is_reported_and_the_error_never_quotes_the_url(monkeypatch, caplog) -> None:
    import httpx

    class _Offline:
        async def request(self, _method: str, url: str, **_kwargs: object) -> object:
            raise httpx.ConnectError(f"cannot connect to {url}")

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", DISCORD_HOOK)
    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", lambda *_a, **_k: _ClientContext(_Offline()))

    async def _public(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(api_client, "_validate_outbound_url", _public)
    caplog.set_level(logging.DEBUG)

    outcome = await messaging.discord_message_in_channel(
        ActionResult(action=_action(ActionType.API_DISCORD), success=True, output=_receipt(message_id="987654321"))
    )

    assert outcome.passed is True and outcome.observed is False
    assert "ConnectError" in outcome.detail
    assert SECRET not in outcome.detail and SECRET not in caplog.text


@pytest.mark.asyncio
async def test_discord_refusal_in_the_receipt_is_a_mismatch() -> None:
    verification = await _verify(_action(ActionType.API_DISCORD), _receipt(status=400, error="bad"))

    assert verification.passed is False
    assert "Discord refused the message" in verification.details[0]


def test_a_failed_read_back_of_a_send_stops_for_the_person_instead_of_resending() -> None:
    from pilot.server import _postcondition_failure_requires_reconciliation

    for action_type in (ActionType.API_SLACK, ActionType.API_DISCORD):
        action = _action(action_type)
        result = ActionResult(action=action, success=True, output=_receipt(message_id="1"))
        verification = MagicMock(failed_actions=[0])
        assert action.is_irreversible is True
        assert _postcondition_failure_requires_reconciliation([result], verification) is True
