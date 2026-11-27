"""A kept plan stays masked on disk and still resumes: its secrets wait in the credential store (fake here).

The three ways a plan is rebuilt from storage: the window's resume of an interrupted plan (resume_plan), a task
approved later from a phone or through MCP and continued with its token (resume_task), and a plan sent to a mesh
peer (which never carries a secret). Nothing here reaches the system keyring, a network or a mail server.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    NotifyParams,
    VerificationResult,
)
from pilot.config import PilotConfig
from pilot.network.collab_executor import is_delegable_action
from pilot.security import plan_secrets
from pilot.security.plan_secrets import KEEP_SECONDS, PlanSecretKeeper, restore, secrets_of
from pilot.security.secret_urls import carries_secret, mask_url
from pilot.security.vault import VaultUnavailableError
from pilot.server import PilotServer
from pilot.system import connections
from pilot.workflows.checkpoints import WorkflowCheckpointStore
from pilot.workflows.durable_tasks import DurableTaskStore, TaskStatus

SLACK = "https://hooks.slack.com/services/T0FAKE/B0FAKE/fakeFakeFake1234"
DISCORD = "https://discord.com/api/webhooks/123456789/fakeDiscordToken9876"
TOKEN = "ghp_fakefakefakefakefake0000"
KEYED = "https://api.weather.example/v1/current.json?key=fakekey5678&q=Paris"
BEARER = "Bearer fakeHeaderToken0000WXYZ"
COOKIE = "sid=fakeCookieValue7777"
SECRETS = (SLACK, DISCORD, TOKEN, "fakekey5678", "fakeHeaderToken0000WXYZ", "fakeCookieValue7777")


class FakeVault:
    """The credential store: a dict. ``available=False`` is a computer with no usable store."""

    backend_name = "FakeKeyring"

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.entries: dict[str, str] = {}

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


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _no_connections(monkeypatch: pytest.MonkeyPatch):
    for variable in connections.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    connections.configure(None, None)
    yield
    connections.configure(None, None)


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.API_SLACK,
                target="post",
                parameters=ApiRequestParams(message="hi", webhook_url=SLACK),
            ),
            Action(
                action_type=ActionType.API_DISCORD,
                target="post",
                parameters=ApiRequestParams(message="hi", webhook_url=DISCORD),
            ),
            Action(
                action_type=ActionType.API_GITHUB,
                target="read",
                parameters=ApiRequestParams(endpoint="/user", token=TOKEN),
            ),
            Action(action_type=ActionType.API_REQUEST, target="weather", parameters=ApiRequestParams(url=KEYED)),
            Action(
                action_type=ActionType.API_REQUEST,
                target="private api",
                parameters=ApiRequestParams(
                    url="https://api.example.com/v1/me",
                    headers={"Authorization": BEARER, "Cookie": COOKIE, "Accept": "application/json"},
                ),
            ),
        ],
        explanation="post and read",
        raw_input="post it",
    )


def _no_secret(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


def _values(plan: ActionPlan) -> list[Any]:
    p = [action.parameters for action in plan.actions]
    return [
        p[0].webhook_url,
        p[1].webhook_url,
        p[2].token,
        p[3].url,
        p[4].headers.get("Authorization"),
        p[4].headers.get("Cookie"),
    ]


# ── the pure parts ───────────────────────────────────────────────────


def test_a_masked_plan_gets_its_values_back_matched_by_their_masked_form_whatever_the_order() -> None:
    plan = _plan()
    held = secrets_of(plan)
    stored = ActionPlan.model_validate_json(plan.model_dump_json())
    _no_secret(plan.model_dump_json())

    assert _values(restore(stored, held)) == [SLACK, DISCORD, TOKEN, KEYED, BEARER, COOKIE]
    reordered = stored.model_copy(update={"actions": list(reversed(stored.actions))})
    assert [a.parameters for a in restore(reordered, held).actions][1].url == KEYED
    # A masked value nobody kept stays masked, and the HTTP layer refuses it.
    assert _values(restore(stored, {}))[0] == mask_url(SLACK)


def test_ordinary_query_words_are_no_longer_taken_for_secrets() -> None:
    assert not carries_secret("https://shop.example/offer?code=SPRING24&session=2")
    assert carries_secret(KEYED)


# ── the checkpoint store ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_stored_row_holds_no_secret_and_a_restart_reads_the_plan_back_whole(tmp_path) -> None:
    vault = FakeVault()
    db = tmp_path / "checkpoints.db"
    await WorkflowCheckpointStore(db, secrets=PlanSecretKeeper(vault)).start_plan("plan-1", "post it", _plan())

    _no_secret(db.read_bytes().decode("utf-8", "replace"))
    for wal in tmp_path.glob("checkpoints.db-*"):
        _no_secret(wal.read_bytes().decode("utf-8", "replace"))
    assert set(vault.entries) == {"plan_secret:plan-1", "plan_secret:index"}

    after_restart = WorkflowCheckpointStore(db, secrets=PlanSecretKeeper(vault))
    checkpoint = await after_restart.get("plan-1")
    assert _values(checkpoint.plan) == [SLACK, DISCORD, TOKEN, KEYED, BEARER, COOKIE]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["complete", "cancelled", "superseded", "denied"])
async def test_a_finished_or_refused_plan_takes_its_secrets_with_it(tmp_path, ending: str) -> None:
    vault = FakeVault()
    store = WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault))
    await store.start_plan("plan-1", "post it", _plan())
    if ending == "denied":
        await store.mark_approval("plan-1", "denied")
    else:
        await store.mark_status("plan-1", ending)

    assert "plan_secret:plan-1" not in vault.entries
    assert json.loads(vault.entries["plan_secret:index"]) == {}


@pytest.mark.asyncio
async def test_failed_and_interrupted_plans_keep_their_secrets_for_a_resume(tmp_path) -> None:
    vault = FakeVault()
    store = WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault))
    await store.start_plan("plan-1", "post it", _plan())
    await store.mark_status("plan-1", "interrupted")
    await store.mark_status("plan-1", "failed")

    assert "plan_secret:plan-1" in vault.entries


@pytest.mark.asyncio
async def test_secrets_older_than_seven_days_are_not_restored_and_the_start_up_sweep_removes_them(tmp_path) -> None:
    vault, clock = FakeVault(), Clock()
    keeper = PlanSecretKeeper(vault, clock=clock)
    store = WorkflowCheckpointStore(tmp_path / "c.db", secrets=keeper)
    await store.start_plan("old", "post it", _plan())
    clock.now += KEEP_SECONDS - 60
    await store.start_plan("fresh", "post it", _plan())
    clock.now += 120  # "old" is now past seven days, "fresh" is two minutes old

    restarted = PlanSecretKeeper(vault, clock=clock)
    assert await restarted.sweep() == 1
    assert "plan_secret:old" not in vault.entries and "plan_secret:fresh" in vault.entries
    expired = await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault, clock=clock)).get("old")
    assert _values(expired.plan)[0] == mask_url(SLACK)


@pytest.mark.asyncio
async def test_without_a_credential_store_the_secrets_live_in_memory_for_this_run_only(tmp_path) -> None:
    vault = FakeVault(available=False)
    keeper = PlanSecretKeeper(vault)
    store = WorkflowCheckpointStore(tmp_path / "c.db", secrets=keeper)
    await store.start_plan("plan-1", "post it", _plan())

    assert _values((await store.get("plan-1")).plan) == [SLACK, DISCORD, TOKEN, KEYED, BEARER, COOKIE]
    restarted = WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault))
    assert _values((await restarted.get("plan-1")).plan)[0] == mask_url(SLACK)


# ── the three resume paths ───────────────────────────────────────────


class Recorder:
    """The executor: the parameters each step was given."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    async def execute(self, plan: ActionPlan, **kwargs: Any) -> list[ActionResult]:
        results = []
        for action in plan.actions:
            await kwargs["on_action_start"](action)
            self.params.append(action.parameters)
            result = ActionResult(action=action, success=True, output="sent")
            await kwargs["on_action_complete"](result)
            results.append(result)
        return results


def _server(checkpoints: WorkflowCheckpointStore, durable: Any = None) -> tuple[PilotServer, Recorder]:
    config = PilotConfig()
    config.security.dry_run = False
    server = PilotServer(config)
    server._checkpoint_store = checkpoints
    server._durable_tasks = durable
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    recorder = Recorder()
    server._executor = recorder
    return server, recorder


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_the_window_resumes_an_interrupted_plan_with_its_webhook_and_token_after_a_restart(tmp_path) -> None:
    vault = FakeVault()
    before = WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault))
    await before.start_plan("plan-1", "post it", _plan())
    await before.mark_approval("plan-1", "approved")
    await before.mark_status("plan-1", "interrupted")

    server, recorder = _server(WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)))
    resumed = await server._handle_resume_plan({"plan_id": "plan-1"}, _ws())

    assert resumed["status"] == "success"
    assert [recorder.params[0].webhook_url, recorder.params[1].webhook_url, recorder.params[2].token] == [
        SLACK,
        DISCORD,
        TOKEN,
    ]
    assert recorder.params[3].url == KEYED
    assert recorder.params[4].headers == {"Authorization": BEARER, "Cookie": COOKIE, "Accept": "application/json"}
    _no_secret(json.dumps(resumed, default=str))
    assert "plan_secret:plan-1" not in vault.entries  # the plan completed


@pytest.mark.asyncio
async def test_a_task_approved_later_from_a_phone_or_mcp_continues_with_its_webhook(tmp_path) -> None:
    vault = FakeVault()
    durable = DurableTaskStore(tmp_path / "durable.db")
    await durable.initialize()
    try:
        created = await durable.create_task(task_id="task-1", session_id="s", user_input="post it")
        await durable.transition("task-1", TaskStatus.PLANNING)
        await durable.transition("task-1", TaskStatus.AWAITING_APPROVAL, plan_id="plan-1")
        plan = ActionPlan(
            actions=[
                Action(
                    action_type=ActionType.API_SLACK,
                    target="post",
                    parameters=ApiRequestParams(message="hi", webhook_url=SLACK),
                ),
                Action(
                    action_type=ActionType.API_REQUEST,
                    target="private api",
                    parameters=ApiRequestParams(
                        method="POST", url="https://api.example.com/v1/items", headers={"Authorization": BEARER}
                    ),
                ),
            ],
            explanation="post",
            raw_input="post it",
        )
        await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)).start_plan(
            "plan-1", "post it", plan
        )
        await durable.create_approval(
            task_id="task-1",
            plan_id="plan-1",
            request={
                "action_indices": [0, 1],
                "actions": [{"index": 0, "action_type": "api_slack"}, {"index": 1, "action_type": "api_request"}],
            },
            timeout_seconds=300,
        )

        # The daemon restarted before the person approved on the phone.
        server, recorder = _server(WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)), durable)
        ws = _ws()
        await server._handle_confirm({"plan_id": "plan-1", "confirmed": True}, ws)
        resumed = await server._handle_resume_task({"task_id": "task-1", "resume_token": created.resume_token}, ws)

        assert resumed["status"] == "success"
        assert recorder.params[0].webhook_url == SLACK
        assert recorder.params[1].headers == {"Authorization": BEARER}
    finally:
        await durable.close()


def test_a_mesh_peer_is_never_sent_a_step_that_needs_a_secret() -> None:
    for action in _plan().actions:
        assert is_delegable_action(action) is False
        _no_secret(json.dumps(action.model_dump(mode="json")))


# ── after a restart with no credential store: the stored connection still runs the step ──


@pytest.mark.asyncio
async def test_a_masked_webhook_runs_through_the_matching_connection_and_any_other_is_refused(monkeypatch) -> None:
    from pilot.system import api_client

    sent: list[str] = []

    async def _request(method: str, url: str, **_kwargs: Any) -> str:
        sent.append(url)
        return json.dumps({"status": 200, "body": "ok"})

    monkeypatch.setattr(api_client, "api_request", _request)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK)

    await api_client.send_slack_message("hi", mask_url(SLACK))
    assert sent == [SLACK]

    other = "https://hooks.slack.com/services/T0FAKE/B0FAKE/anotherFakeAAAA"
    with pytest.raises(ValueError, match="hidden when the plan was kept"):
        await api_client.send_slack_message("hi", mask_url(other))
    assert sent == [SLACK]


def test_the_module_text_names_the_lifetime_it_enforces() -> None:
    assert plan_secrets.KEEP_DAYS == 7 and "KEEP_DAYS" in (plan_secrets.__doc__ or "")


# ── why a secret could not be put back, said in the step's own failure ──


class SmallVault(FakeVault):
    """A store that refuses an entry longer than Windows Credential Manager holds."""

    async def store_key(self, name: str, value: str) -> None:
        if len(value) > plan_secrets.WINDOWS_ENTRY_CHARS:
            raise VaultUnavailableError("could not save")
        await super().store_key(name, value)


def _big_plan() -> ActionPlan:
    long_token = "Bearer " + "fakeLongToken" * 120
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.API_REQUEST,
                target="private api",
                parameters=ApiRequestParams(url="https://api.example.com/v1/me", headers={"Authorization": long_token}),
            ),
            Action(
                action_type=ActionType.API_SLACK,
                target="post",
                parameters=ApiRequestParams(message="hi", webhook_url=SLACK),
            ),
        ],
        explanation="read and post",
        raw_input="do it",
    )


@pytest.mark.asyncio
async def test_a_secret_too_large_to_keep_is_named_in_the_failure_after_a_restart(tmp_path, monkeypatch) -> None:
    from pilot.system import api_client

    vault = SmallVault()
    await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)).start_plan(
        "big", "do it", _big_plan()
    )
    assert "plan_secret:big" not in vault.entries
    assert json.loads(vault.entries["plan_secret:index"])["big"]["not_kept"] == "too_large"

    restarted = await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)).get("big")
    request, slack = (action.parameters for action in restarted.plan.actions)
    sent: list[str] = []

    async def _no_send(*args: Any, **kwargs: Any) -> str:
        sent.append("sent")
        return "{}"

    from pilot.system import outbound

    monkeypatch.setattr(outbound, "create_httpx_client", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(ValueError) as header_failure:
        await api_client.api_request("GET", request.url, headers=request.headers)
    monkeypatch.setattr(api_client, "api_request", _no_send)
    with pytest.raises(ValueError) as webhook_failure:
        await api_client.send_slack_message("hi", slack.webhook_url)

    for failure in (header_failure, webhook_failure):
        text = str(failure.value)
        assert "too large to keep in the credential store" in text and "Give the request again" in text
        _no_secret(text)
    assert sent == []


@pytest.mark.asyncio
async def test_with_no_credential_store_the_failure_after_a_restart_says_so(tmp_path) -> None:
    from pilot.system import api_client

    vault = FakeVault(available=False)
    await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)).start_plan("p", "do it", _plan())

    restarted = await WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(vault)).get("p")
    request = restarted.plan.actions[4].parameters
    with pytest.raises(ValueError, match="no credential store"):
        await api_client.api_request("GET", request.url, headers=request.headers)


def test_two_different_secrets_that_mask_alike_are_never_swapped() -> None:
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.API_REQUEST,
                target=str(index),
                parameters=ApiRequestParams(
                    url="https://api.example.com/x", headers={"Authorization": f"Bearer ab{index}"}
                ),
            )
            for index in range(2)
        ]
    )
    held = secrets_of(plan)
    kept = ActionPlan.model_validate_json(plan.model_dump_json())

    assert held == {}
    assert [a.parameters.headers["Authorization"] for a in restore(kept, held).actions] == ["Bearer ***", "Bearer ***"]
