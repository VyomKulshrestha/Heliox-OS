"""A credential in an api_request header is masked wherever the plan is shown or kept; only the HTTP call sees it.

Fakes only: no network, no keyring, temporary stores.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
from typing import Any

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams
from pilot.network.collab_executor import is_delegable_action
from pilot.security.secret_urls import (
    SecretUrlFilter,
    is_secret_header,
    mask_header_value,
    redact,
    redact_headers,
)

BEARER = "Bearer fakeHeaderSecret0000WXYZ"
API_KEY = "fakeApiKeyValue12345678"
COOKIE = "session=fakeCookieValue999"
SECRETS = ("fakeHeaderSecret0000WXYZ", API_KEY, "fakeCookieValue999")
ORDINARY = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Heliox-test/1.0"}


def _headers() -> dict[str, str]:
    return {"Authorization": BEARER, "X-Api-Key": API_KEY, "Cookie": COOKIE, **ORDINARY}


def _action() -> Action:
    return Action(
        action_type=ActionType.API_REQUEST,
        target="read my account",
        parameters=ApiRequestParams(method="GET", url="https://api.example.com/v1/me", headers=_headers()),
    )


def _no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


# ── which headers, and how ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("Authorization", True),
        ("proxy-authorization", True),
        ("Cookie", True),
        ("Set-Cookie", True),
        ("X-Api-Key", True),
        ("X-Auth-Token", True),
        ("X-Client-Secret", True),
        ("apikey", True),
        ("X-Password", True),
        ("Accept", False),
        ("Content-Type", False),
        ("User-Agent", False),
    ],
)
def test_which_headers_carry_a_credential(name: str, secret: bool) -> None:
    assert is_secret_header(name) is secret


def test_the_scheme_word_and_last_four_characters_stay_and_a_short_secret_is_masked_whole() -> None:
    assert mask_header_value(BEARER) == "Bearer ***WXYZ"
    assert mask_header_value(API_KEY) == "***5678"
    assert mask_header_value("Bearer ab1") == "Bearer ***"
    assert mask_header_value("k3y4") == "***"
    assert redact_headers(_headers()) | ORDINARY == redact_headers(_headers())
    assert {name: redact_headers(_headers())[name] for name in ORDINARY} == ORDINARY


# ── every place the parameters travel ────────────────────────────────


def test_dumps_reprs_and_the_value_the_step_uses() -> None:
    params = _action().parameters
    assert params.headers["Authorization"] == BEARER  # the HTTP call itself gets the real value
    for shown in (params.model_dump_json(), json.dumps(params.model_dump()), repr(params), str(_action())):
        _no_secret(shown)
    assert params.model_dump()["headers"]["Authorization"] == "Bearer ***WXYZ"
    assert redact({"headers": _headers()})["headers"]["X-Api-Key"] == "***5678"


def test_the_approval_card_says_which_host_receives_a_credential() -> None:
    from pilot.server import _approval_effect

    action = _action()
    card = {**action.model_dump(), "effect": _approval_effect(action)}

    _no_secret(json.dumps(card))
    assert card["effect"].startswith("Sends a credential to api.example.com")
    assert "Authorization" in card["effect"]
    plain = Action(
        action_type=ActionType.API_REQUEST,
        target="read",
        parameters=ApiRequestParams(url="https://api.example.com/x", headers=dict(ORDINARY)),
    )
    assert _approval_effect(plain) == ""


def test_log_lines_carry_the_masked_parameters() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretUrlFilter())
    logger = logging.getLogger("tests.secret_headers")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("running %s with %r", _action(), _action().parameters)
    finally:
        logger.removeHandler(handler)
    assert "Bearer ***WXYZ" in stream.getvalue()
    _no_secret(stream.getvalue())


@pytest.mark.asyncio
async def test_the_audit_log_receipts_saved_plan_task_history_and_repair_plan_are_masked(tmp_path) -> None:
    from pilot.config import ReceiptsConfig
    from pilot.security.audit import AuditLogger
    from pilot.server import PlanHistoryStore
    from pilot.workflows.checkpoints import WorkflowCheckpointStore
    from pilot.workflows.task_receipts import TaskReceipts

    action = _action()
    result = ActionResult(action=action, success=True, output="{}")
    plan = ActionPlan(actions=[action], raw_input="read my account")

    audit = AuditLogger(audit_file=tmp_path / "audit.jsonl")
    await audit.log_action_start(action, "plan-1")
    await audit.log_action_result(result, "plan-1")
    _no_secret((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))

    class _Vault:
        available = True

        def __init__(self) -> None:
            self.keys: dict[str, str] = {}

        async def get_key(self, name: str) -> str | None:
            return self.keys.get(name)

        async def store_key(self, name: str, value: str) -> None:
            self.keys[name] = value

    receipts = TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=_Vault())
    receipts.record_action("task-1", "plan-1", action, result, user_confirmed=True, undo_kind="not_undoable")
    await receipts.finalize("task-1", request="read my account", status="succeeded", summary="read", source="voice")
    written = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "receipts").rglob("*.json"))
    assert written
    _no_secret(written)

    store = WorkflowCheckpointStore(tmp_path / "workflows.db")  # no keeper: the row alone is checked here
    await store.start_plan("plan-1", "read my account", plan)
    with sqlite3.connect(tmp_path / "workflows.db") as db:
        _no_secret(json.dumps(db.execute("SELECT * FROM workflow_checkpoints").fetchall(), default=str))

    history = PlanHistoryStore(tmp_path / "history.db")
    await history.initialize()
    await history.record(
        plan_id="p1",
        raw_input="read my account",
        plan=plan,
        critic_verdict=None,
        confirmation_decision="approved",
        execution_status="ok",
        results=[result],
        verification=None,
        dry_run=False,
        duration_ms=5,
    )
    await history.close()
    with sqlite3.connect(tmp_path / "history.db") as db:
        _no_secret(json.dumps(db.execute("SELECT * FROM plan_history").fetchall(), default=str))

    # The planner's repair round and the experience trace read the plan through its dump.
    _no_secret(plan.model_dump_json())
    _no_secret(json.dumps(result.model_dump(mode="json")))


def test_a_mesh_peer_is_never_sent_a_request_with_a_credential() -> None:
    action = _action()
    assert is_delegable_action(action) is False
    _no_secret(json.dumps(action.model_dump(mode="json")))


@pytest.mark.asyncio
async def test_the_http_call_gets_the_real_header_and_a_masked_one_is_never_sent(monkeypatch) -> None:
    from pilot.system import api_client

    sent: list[dict[str, Any]] = []

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}
        text = "{}"
        url = "https://api.example.com/v1/me"

        def json(self) -> object:
            return {}

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_a: object) -> None:
            return None

    async def _send(client: Any, method: str, url: str, *, allow_private_network: bool, request_kwargs: Any):
        sent.append(dict(request_kwargs.get("headers") or {}))
        return _Response()

    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", lambda *_a, **_k: _Client())
    monkeypatch.setattr(api_client, "_request_with_validated_redirects", _send)

    params = _action().parameters
    await api_client.api_request("GET", params.url, headers=params.headers)
    kept = ApiRequestParams.model_validate_json(params.model_dump_json())
    with pytest.raises(ValueError, match="Authorization header"):
        await api_client.api_request("GET", kept.url, headers=kept.headers)

    assert sent == [_headers()]
