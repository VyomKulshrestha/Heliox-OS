"""A webhook or token URL a plan carries is shown and kept masked everywhere; only the running step has it whole.

Fakes only: no request leaves the test, every store writes under tmp_path.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams
from pilot.security import secret_urls
from pilot.security.secret_urls import SecretUrlFilter, carries_secret, mask_url, redact, redact_text

SLACK = "https://hooks.slack.com/services/T0001/B0001/s3cr3tSlackTokenXyZ9"
DISCORD = "https://discord.com/api/webhooks/123456789/s3cr3tDiscordTokenAbC7"
WITH_USER = "https://ada:hunter2pass@api.example.com/v1/items"
WITH_TOKEN = "https://api.example.com/v1/items?limit=5&access_token=s3cr3tQueryToken"
SECRETS = ("s3cr3tSlackToken", "s3cr3tDiscordToken", "hunter2pass", "s3cr3tQueryToken")


def _no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, secret


def _slack_action(webhook: str = SLACK, target: str = "") -> Action:
    return Action(
        action_type=ActionType.API_SLACK,
        target=target or f"post to {webhook}",
        parameters=ApiRequestParams(message="Deploy finished", webhook_url=webhook),
    )


def _request_action(url: str) -> Action:
    return Action(action_type=ActionType.API_REQUEST, target=url, parameters=ApiRequestParams(method="GET", url=url))


# ── the masking function ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        (SLACK, True),
        (DISCORD, True),
        (WITH_USER, True),
        (WITH_TOKEN, True),
        ("https://hooks.zapier.com/hooks/catch/123/abcdefghijklmnopq", True),
        ("https://storage.example.com/f.zip?X-Amz-Signature=abc", True),
        ("https://example.com/docs?page=2", False),
        ("https://api.github.com/repos/o/r/issues", False),
        ("", False),
    ],
)
def test_which_urls_carry_a_secret(url: str, secret: bool) -> None:
    assert carries_secret(url) is secret


def test_a_masked_url_keeps_the_host_and_the_last_four_characters() -> None:
    assert mask_url(SLACK) == "https://hooks.slack.com/***XyZ9"
    assert mask_url(WITH_USER) == "https://api.example.com/***tems"
    assert mask_url(mask_url(SLACK)) == mask_url(SLACK)


def test_text_and_structures_are_masked_and_nothing_else_changes() -> None:
    text = f"Posting to {SLACK}. See https://example.com/docs for help."
    assert redact_text(text) == "Posting to https://hooks.slack.com/***XyZ9. See https://example.com/docs for help."
    assert redact({"webhook_url": "https://my.own/hook", "nested": [WITH_TOKEN]}) == {
        "webhook_url": "https://my.own/***hook",
        "nested": [mask_url(WITH_TOKEN)],
    }


# ── the models: plan card, approval, saved plan and repair prompt all read their dumps ──


def test_parameters_keep_the_whole_address_in_memory_and_mask_it_in_every_dump_and_repr() -> None:
    params = ApiRequestParams(message="hi", webhook_url=SLACK, url=WITH_TOKEN)

    assert params.webhook_url == SLACK and params.url == WITH_TOKEN  # what the running step uses
    for shown in (json.dumps(params.model_dump()), params.model_dump_json(), repr(params), str(params)):
        _no_secret(shown)
    assert params.model_dump()["webhook_url"] == "https://hooks.slack.com/***XyZ9"


def test_target_explanation_request_output_and_error_are_masked() -> None:
    action = _slack_action()
    plan = ActionPlan(actions=[action], explanation=f"I will post to {DISCORD}", raw_input=f"post to {SLACK}")
    result = ActionResult(action=action, success=False, output=f"from {WITH_USER}", error=f"failed at {WITH_TOKEN}")

    assert "***" in action.target  # masked as soon as the plan exists: it is only ever shown
    assert plan.raw_input.endswith("XyZ9") and "s3cr3t" in plan.raw_input  # the request itself stays in memory
    _no_secret(plan.model_dump_json())
    _no_secret(result.model_dump_json())
    _no_secret(json.dumps(result.model_dump()))


def test_a_url_without_a_secret_is_shown_whole() -> None:
    action = _request_action("https://api.github.com/repos/o/r/issues")

    assert action.model_dump()["parameters"]["url"] == "https://api.github.com/repos/o/r/issues"


# ── the running step, and a restored plan ────────────────────────────


class _Response:
    def __init__(self, url: str) -> None:
        self.url = url
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = "ok"

    def json(self) -> object:
        raise ValueError("not json")


def _network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from tests import fake_outbound

    sent: list[str] = []

    class _Client:
        async def request(self, _method: str, url: str, **_kwargs: object) -> _Response:
            sent.append(url)
            return _Response(url)

    # The masked check runs in pilot.system.outbound; the name lookup goes to a fake resolver.
    fake_outbound.install(monkeypatch, _Client())
    return sent


@pytest.mark.asyncio
async def test_the_step_sends_to_the_whole_address(monkeypatch) -> None:
    from pilot.agents.executor import Executor

    sent = _network(monkeypatch)
    # The handler alone, on a stand-in with no system manager: it reaches only api_client and the fake network.
    output = await Executor._exec_api_slack(SimpleNamespace(), _slack_action())  # type: ignore[arg-type]

    assert sent == [SLACK]
    _no_secret(output)


@pytest.mark.asyncio
async def test_a_restored_plan_never_sends_to_a_masked_address(monkeypatch) -> None:
    from pilot.system import api_client

    sent = _network(monkeypatch)
    restored = ActionPlan.model_validate_json(ActionPlan(actions=[_slack_action()]).model_dump_json())
    masked = restored.actions[0].parameters.webhook_url  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="hidden when the plan was kept"):
        await api_client.send_slack_message("Deploy finished", masked)
    with pytest.raises(ValueError, match="hidden when the plan was kept"):
        await api_client.api_request("GET", mask_url(WITH_TOKEN) or "")
    assert sent == []


@pytest.mark.asyncio
async def test_an_api_answer_names_its_address_masked(monkeypatch) -> None:
    from pilot.system import api_client

    _network(monkeypatch)
    answer = await api_client.api_request("GET", WITH_TOKEN)

    _no_secret(answer)


# ── stores: audit log, signed audits, receipts, saved plan, task history ──


@pytest.mark.asyncio
async def test_the_audit_log_is_written_masked(tmp_path) -> None:
    from pilot.security.audit import AuditLogger

    audit = AuditLogger(audit_file=tmp_path / "audit.jsonl")
    action = _slack_action()
    await audit.log_action_start(action, "plan-1")
    await audit.log_action_result(
        ActionResult(action=action, success=False, output=f"sent via {SLACK}", error=f"refused by {DISCORD}"), "plan-1"
    )
    await audit.log_security_event("note", {"url": WITH_USER})

    _no_secret((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_the_signed_permission_and_gateway_audits_are_written_masked(tmp_path) -> None:
    from pilot.security.gateway_audit import AgentGatewayAuditStore
    from pilot.security.permission_audit import PermissionEscalationAuditStore

    permission = PermissionEscalationAuditStore(db_file=tmp_path / "permission.db", key=b"k" * 32)
    await permission.record_event(
        plan_id="p",
        action_index=0,
        action_type="api_slack",
        target=f"post to {SLACK}",
        permission_tier="SYSTEM_MODIFY",
        requires_root=False,
        destructive=False,
        confirmation_decision="approved",
        execution_error=f"failed at {WITH_TOKEN}",
    )
    gateway = AgentGatewayAuditStore(db_file=tmp_path / "gateway.db", key=b"k" * 32)
    await gateway.record_event(
        plan_id="p",
        action_index=0,
        action_type="api_discord",
        action_family="network",
        target=f"post to {DISCORD}",
        source_profile="interactive",
        permission_tier="SYSTEM_MODIFY",
        override_applied=False,
        override_restricted=False,
        decision="allowed",
        execution_error=f"failed at {WITH_USER}",
    )

    for name in ("permission.db", "gateway.db"):
        _no_secret((tmp_path / name).read_bytes().decode("utf-8", "replace"))


@pytest.mark.asyncio
async def test_receipts_are_written_masked(tmp_path) -> None:
    from pilot.config import ReceiptsConfig
    from pilot.workflows.task_receipts import TaskReceipts

    class _Vault:
        available = True

        def __init__(self) -> None:
            self.keys: dict[str, str] = {}

        async def get_key(self, name: str) -> str | None:
            return self.keys.get(name)

        async def store_key(self, name: str, value: str) -> None:
            self.keys[name] = value

    receipts = TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=_Vault())
    action = _slack_action()
    receipts.record_action(
        "task-1",
        "plan-1",
        action,
        ActionResult(action=action, success=False, error=f"refused by {SLACK}"),
        user_confirmed=True,
        undo_kind="not_undoable",
    )
    await receipts.finalize(
        "task-1", request=f"post to {SLACK}", status="failed", summary=f"tried {DISCORD}", source="voice"
    )

    written = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "receipts").rglob("*.json"))
    assert written
    _no_secret(written)


@pytest.mark.asyncio
async def test_the_saved_plan_is_written_masked(tmp_path) -> None:
    from pilot.workflows.checkpoints import WorkflowCheckpointStore

    store = WorkflowCheckpointStore(tmp_path / "workflows.db")
    action = _slack_action()
    await store.start_plan("plan-1", f"post to {SLACK}", ActionPlan(actions=[action]))
    await store.record_result("plan-1", ActionResult(action=action, success=True, output=f"via {WITH_TOKEN}"))

    with sqlite3.connect(tmp_path / "workflows.db") as db:
        rows = db.execute("SELECT * FROM workflow_checkpoints").fetchall()
    _no_secret(json.dumps(rows, default=str))
    restored = await store.get("plan-1")
    assert restored is not None and restored.plan.actions[0].parameters.webhook_url == mask_url(SLACK)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_task_history_and_the_repair_prompt_are_masked(tmp_path) -> None:
    from pilot.server import PlanHistoryStore, _repair_error_context

    history = PlanHistoryStore(tmp_path / "history.db")
    await history.initialize()
    action = _slack_action()
    await history.record(
        plan_id="p1",
        raw_input=f"post to {SLACK}",
        plan=ActionPlan(actions=[action], raw_input=f"post to {SLACK}"),
        critic_verdict=None,
        confirmation_decision="approved",
        execution_status="error",
        results=[ActionResult(action=action, success=False, error=f"refused by {DISCORD}")],
        verification=None,
        dry_run=False,
        duration_ms=5,
    )
    await history.close()

    with sqlite3.connect(tmp_path / "history.db") as db:
        _no_secret(json.dumps(db.execute("SELECT * FROM plan_history").fetchall(), default=str))
    _no_secret(_repair_error_context([f"Action 0 (api_request): FAILED — {WITH_TOKEN}"], [f"POST {WITH_USER}"]))


def test_log_lines_and_tracebacks_are_masked() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretUrlFilter())
    logger = logging.getLogger("tests.secret_urls")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("posting to %s", SLACK)
        logger.info("params %s", ApiRequestParams(webhook_url=DISCORD))
        try:
            raise RuntimeError(f"cannot connect to {WITH_USER}")
        except RuntimeError:
            logger.exception("request failed")
    finally:
        logger.removeHandler(handler)

    written = stream.getvalue()
    assert "hooks.slack.com/***" in written and "cannot connect to" in written
    _no_secret(written)
