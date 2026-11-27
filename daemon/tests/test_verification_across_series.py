"""Rules that hold across the verifier series once they are stacked together.

Each verifier series tested its own module. These tests check the places where they meet: the registry and the
executor's failure rule (groundwork), the declared checks (expect), the reconcile set (mail and calendar, desktop
input), the irreversible sends (messaging, packages), secret URL masking, the pre-check rule and the catalog's
categories. Fakes only: no handler runs, no request leaves the test, every store writes under tmp_path.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    IRREVERSIBLE_ACTIONS,
    RECONCILE_NOT_RETRY_ACTIONS,
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    CalendarParams,
    Expectation,
    FileParams,
    KeyboardParams,
    ScheduleParams,
    SshCommandParams,
    TriggerParams,
    VerificationResult,
)
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.autonomous import AutonomousExecutor, AutonomousJob, JobStep, settle_instead_of_retrying
from pilot.agents.postconditions import Outcome, declared, messaging
from pilot.agents.postconditions._read_only import READ_ONLY_WITHOUT_POSTCONDITION
from pilot.agents.postconditions._unobservable import UNOBSERVABLE_POSTCONDITION
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.response_contract import success_message
from pilot.server import _postcondition_failure_requires_reconciliation
from tests import fake_outbound

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET = "s3cr3tDiscordTokenAbC7"
DISCORD_HOOK = f"https://discord.com/api/webhooks/123456789/{SECRET}"


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Register into a copy, so nothing a test puts there reaches later tests."""
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


@pytest.fixture(autouse=True)
def _declared_router_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(declared, "_router", None)


# ── 1. A failed action is a failure, and its verifier never runs ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", sorted(POSTCONDITION_VERIFIERS, key=lambda a: a.value))
async def test_a_failed_action_is_a_failure_and_no_verifier_runs(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict, action_type: ActionType
) -> None:
    calls: list[ActionType] = []
    for registered, registration in list(scratch_registry.items()):

        async def spy(result: ActionResult, _real=registration.check) -> Outcome:
            calls.append(result.action.action_type)
            return await _real(result)

        scratch_registry[registered] = dataclasses.replace(registration, check=spy)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action = Action(action_type=action_type, parameters={})

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=False, error="the handler raised")]
    )

    assert verification.passed is False
    assert verification.details[0] == f"Action 0 ({action_type.value}): FAILED — the handler raised"
    assert verification.independently_observed_actions == []
    assert calls == []


# ── 2. A declared check that is refused leaves the step completed as reported ────────────────


@pytest.mark.asyncio
async def test_a_refused_declared_check_leaves_the_step_reported_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never(_probe: Action) -> object:
        raise AssertionError("a refused probe must never run")

    monkeypatch.setattr(declared, "run_probe", never)
    probe = Action(
        action_type=ActionType.SSH_COMMAND,
        target="prod-1",
        parameters=SshCommandParams(host="prod-1", command="rm -f /tmp/marker"),
    )
    action = Action(
        action_type=ActionType.SSH_COMMAND,
        target="prod-1",
        parameters=SshCommandParams(host="prod-1", command="touch /tmp/marker"),
        expect=Expectation(probe=probe, kind="exit_code", value=0),
    )
    plan = ActionPlan(actions=[action])
    results = [ActionResult(action=action, success=True, output="")]

    verification = await Verifier(MagicMock()).verify(plan, results)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert verification.failed_actions == []
    assert "refused" in verification.reasons[0]
    assert "refused" in verification.details[0]
    # The run is not stopped for reconciliation and the step is not repaired: it completed, as reported.
    assert _postcondition_failure_requires_reconciliation(results, verification) is False
    # An ssh command cannot be undone: the window's one line leads with the phrase said aloud, never the reason.
    reply = success_message(plan, results, verification, dry_run=False)
    assert reply.endswith("Note: I couldn't check that it went through; details in the receipt.")


# ── 3. A mismatch on an irreversible send stops; on a retryable action it retries ─────────────


def _mismatch(action: Action) -> tuple[list[ActionResult], VerificationResult]:
    results = [ActionResult(action=action, success=True, output="done")]
    return results, VerificationResult(passed=False, details=["MISMATCH"], failed_actions=[0])


@pytest.mark.parametrize(
    "action_type",
    [
        ActionType.API_SEND_EMAIL,
        ActionType.EMAIL_REPLY,
        ActionType.API_SLACK,
        ActionType.API_DISCORD,
        ActionType.PACKAGE_UPDATE,
        ActionType.GIT_PUSH,
    ],
)
def test_a_mismatch_on_an_irreversible_action_stops_the_run(action_type: ActionType) -> None:
    assert action_type in IRREVERSIBLE_ACTIONS
    results, verification = _mismatch(Action(action_type=action_type, parameters={}))

    assert _postcondition_failure_requires_reconciliation(results, verification) is True


@pytest.mark.parametrize("action_type", sorted(RECONCILE_NOT_RETRY_ACTIONS, key=lambda a: a.value))
def test_a_mismatch_on_a_reconcile_action_stops_the_run_and_the_autonomous_rounds(action_type: ActionType) -> None:
    results, verification = _mismatch(Action(action_type=action_type, parameters={}))

    assert _postcondition_failure_requires_reconciliation(results, verification) is True
    assert settle_instead_of_retrying(results, verification.failed_actions)


@pytest.mark.parametrize("action_type", [ActionType.FILE_WRITE, ActionType.SERVICE_ENABLE, ActionType.TRIGGER_CREATE])
def test_a_mismatch_on_a_retryable_action_goes_to_repair(action_type: ActionType) -> None:
    action = Action(action_type=action_type, parameters={})
    assert not action.is_irreversible and not action.requires_snapshot
    results, verification = _mismatch(action)

    assert _postcondition_failure_requires_reconciliation(results, verification) is False
    assert settle_instead_of_retrying(results, verification.failed_actions) is None


# ── 4. A secret URL is masked in the receipt of a verified send ──────────────────────────────


class _Response:
    def __init__(self, url: str, status_code: int, body: object) -> None:
        self.url = url
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> object:
        return self._body


class _Network:
    def __init__(self, *answers: tuple[int, object]) -> None:
        self.answers = list(answers)
        self.requests: list[tuple[str, str]] = []

    async def request(self, method: str, url: str, **_kwargs: object) -> _Response:
        self.requests.append((method, url))
        status, body = self.answers.pop(0)
        return _Response(url, status, body)


class _Vault:
    available = True

    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    async def get_key(self, name: str) -> str | None:
        return self.keys.get(name)

    async def store_key(self, name: str, value: str) -> None:
        self.keys[name] = value


@pytest.mark.asyncio
async def test_a_verified_send_keeps_its_webhook_secret_out_of_the_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pilot.config import ReceiptsConfig
    from pilot.system import api_client
    from pilot.workflows.task_receipts import TaskReceipts

    async def _no_pause(_seconds: float) -> None:
        return None

    network = _Network(
        (200, {"id": "987654321", "channel_id": "555", "content": "Deploy finished"}),
        (200, {"id": "987654321", "content": "Deploy finished"}),
    )
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    fake_outbound.install(monkeypatch, network)
    monkeypatch.setattr(messaging, "_pause", _no_pause)

    action = Action(
        action_type=ActionType.API_DISCORD,
        target=f"post to {DISCORD_HOOK}",
        parameters=ApiRequestParams(message="Deploy finished", webhook_url=DISCORD_HOOK),
    )
    output = await api_client.send_discord_message("Deploy finished", DISCORD_HOOK)
    plan = ActionPlan(actions=[action], raw_input=f"post Deploy finished to {DISCORD_HOOK}")
    results = [ActionResult(action=action, success=True, output=output)]

    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)
    receipts = TaskReceipts(tmp_path / "receipts", config=ReceiptsConfig(), vault=_Vault())
    receipts.record_action("task-1", "plan-1", action, results[0], user_confirmed=True, undo_kind="not_undoable")
    await receipts.finalize("task-1", request=plan.raw_input, status="success", summary=reply, source="voice")

    assert verification.passed is True and verification.independently_observed_actions == [0]
    assert [method for method, _url in network.requests] == ["POST", "GET"]
    written = "".join(path.read_text(encoding="utf-8") for path in (tmp_path / "receipts").rglob("*.json"))
    assert written
    for text in (written, reply, str(verification), output, results[0].model_dump_json(), action.target):
        assert SECRET not in text


# ── 5. Every registered pre-check skips only on an observed "already holds" ──────────────────

PRECHECKED_AREA_ACTIONS = {
    ActionType.FILE_WRITE,
    ActionType.SCHEDULE_DELETE,
    ActionType.TRIGGER_DELETE,
    ActionType.TRIGGER_START,
    ActionType.TRIGGER_STOP,
}


def test_the_area_prechecks_are_the_ones_covered_here() -> None:
    """A new pre-check must get a case below before it is trusted to skip an action."""
    prechecked = {action for action, registration in postconditions.registrations().items() if registration.precheck}
    assert prechecked == PRECHECKED_AREA_ACTIONS


def _engine_without_store() -> SimpleNamespace:
    return SimpleNamespace(store_path=None, list_triggers=lambda: [], is_running=lambda: True)


def _unobservable_case(action_type: ActionType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Action:
    from pilot.system import scheduler, triggers

    if action_type == ActionType.FILE_WRITE:
        path = tmp_path / "exists.txt"
        path.write_text("anything", encoding="utf-8")
        # The file is there but the plan gives no content to compare it with: not observed.
        return Action(action_type=action_type, target=str(path), parameters=FileParams(path=str(path)))
    if action_type == ActionType.SCHEDULE_DELETE:
        monkeypatch.setattr(scheduler, "schedule_exists", AsyncMock(side_effect=RuntimeError("scheduler unreadable")))
        return Action(action_type=action_type, target="backup", parameters=ScheduleParams(name="backup"))
    monkeypatch.setattr(triggers, "get_engine", _engine_without_store)
    return Action(action_type=action_type, target="nightly", parameters=TriggerParams(name="nightly"))


def _absent_case(action_type: ActionType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Action:
    """The target is not there and this run did not delete it: a delete must still run (and fail in its handler)."""
    from pilot.system import scheduler, triggers

    if action_type == ActionType.FILE_WRITE:
        path = tmp_path / "missing.txt"
        return Action(action_type=action_type, target=str(path), parameters=FileParams(path=str(path), content="x"))
    if action_type == ActionType.SCHEDULE_DELETE:
        monkeypatch.setattr(scheduler, "schedule_exists", AsyncMock(return_value=False))
        return Action(action_type=action_type, target="no-such", parameters=ScheduleParams(name="no-such"))
    store = tmp_path / "triggers.json"
    store.write_text(json.dumps({"version": 1, "engine_running": False, "triggers": []}), encoding="utf-8")
    engine = SimpleNamespace(store_path=store, list_triggers=lambda: [], is_running=lambda: False)
    monkeypatch.setattr(triggers, "get_engine", lambda: engine)
    if action_type == ActionType.TRIGGER_STOP:
        # The store says stopped but the loop still runs: not "already holds".
        engine.is_running = lambda: True
    return Action(action_type=action_type, target="no-such", parameters=TriggerParams(name="no-such"))


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", sorted(PRECHECKED_AREA_ACTIONS, key=lambda a: a.value))
@pytest.mark.parametrize("case", ["unobservable", "absent"])
async def test_an_area_precheck_that_did_not_observe_the_state_never_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action_type: ActionType, case: str
) -> None:
    builder = _unobservable_case if case == "unobservable" else _absent_case
    action = builder(action_type, monkeypatch, tmp_path)

    state = await Verifier(MagicMock()).check_already_satisfied(action)

    assert state is None or state[0] is False


@pytest.mark.asyncio
async def test_an_observed_precheck_still_skips(tmp_path: Path) -> None:
    path = tmp_path / "same.txt"
    path.write_text("hello", encoding="utf-8")
    action = Action(
        action_type=ActionType.FILE_WRITE, target=str(path), parameters=FileParams(path=str(path), content="hello")
    )

    state = await Verifier(MagicMock()).check_already_satisfied(action)

    assert state is not None and state[0] is True


# ── 6. Every action type is in exactly one catalog category ──────────────────────────────────


def _category(action_type: ActionType) -> str:
    method = POSTCONDITION_VERIFIERS.get(action_type)
    if method is not None:
        return "declared" if method.startswith("declared_") else "independent"
    if action_type in READ_ONLY_WITHOUT_POSTCONDITION:
        return "read_only"
    if action_type in UNOBSERVABLE_POSTCONDITION:
        return "unobservable"
    return "none"


def test_the_category_lists_do_not_overlap() -> None:
    verified = set(POSTCONDITION_VERIFIERS)
    read_only = set(READ_ONLY_WITHOUT_POSTCONDITION)
    unobservable = set(UNOBSERVABLE_POSTCONDITION)

    assert not verified & read_only
    assert not verified & unobservable
    assert not read_only & unobservable


def test_every_action_type_has_exactly_one_category_in_the_committed_catalog() -> None:
    catalog = json.loads((REPO_ROOT / "capabilities.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry["verification"] for entry in catalog["actions"]}

    assert set(entries) == {action_type.value for action_type in ActionType}
    counts: dict[str, int] = {}
    for action_type in ActionType:
        category = _category(action_type)
        counts[category] = counts.get(category, 0) + 1
        verification = entries[action_type.value]
        if category == "unobservable":
            assert verification["postcondition"] == "none"
            assert verification["method"] == "unobservable_at_run_time"
        else:
            assert verification["postcondition"] == category
            assert verification["method"] != "unobservable_at_run_time"
    assert sum(counts.values()) == len(ActionType)
    summary = catalog["summary"]
    assert summary["independent_postcondition_verifiers"] == counts.get("independent", 0)
    assert summary["declared_postcondition_verifiers"] == counts.get("declared", 0)
    assert summary["read_only_actions"] == counts.get("read_only", 0)


# ── 7. Where the fix series meet: the desktop and browser readings, kept plans, Settings connections, late checks ──


def _executor(tmp_path: Path):
    from pilot.agents.executor import Executor
    from pilot.config import PilotConfig
    from pilot.security.audit import AuditLogger
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator

    config = PilotConfig()
    return Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )


@pytest.mark.asyncio
async def test_the_desktop_and_browser_readings_each_read_only_their_own_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Open an app, type, press Enter, then click on a page: both observers are wired in _execute_single, the
    desktop one around every step, the browser one after browser steps only, and neither runs inside the other."""
    from pilot.actions import BrowserParams, OpenApplicationParams
    from pilot.agents.postconditions import _in_step
    from pilot.system import desktop_observe

    executor = _executor(tmp_path)
    inside: list[tuple[str, bool, bool]] = []

    async def handler(action: Action) -> str:
        inside.append((action.action_type.value, desktop_observe.current_step() is action, _in_step.right_after_step()))
        return f"done: {action.action_type.value}"

    readings: list[tuple[str, object]] = []

    async def observe(result: ActionResult) -> str | None:
        readings.append((result.action.action_type.value, desktop_observe.current_step()))
        return _in_step.STEP_CHECK_MARKER + ' {"passed":true,"detail":"read","observed":true}'

    steps = [
        Action(action_type=ActionType.OPEN_APPLICATION, target="Notes", parameters=OpenApplicationParams(name="Notes")),
        Action(action_type=ActionType.KEYBOARD_TYPE, target="Notes", parameters=KeyboardParams(text="hello")),
        Action(action_type=ActionType.KEYBOARD_PRESS, target="Notes", parameters=KeyboardParams(key="enter")),
        Action(action_type=ActionType.BROWSER_CLICK, target="Play", parameters=BrowserParams(selector="#play")),
    ]
    for step in steps:
        monkeypatch.setitem(executor._dispatch_table, step.action_type, handler)
    monkeypatch.setattr(_in_step, "observe", observe)

    results = [await executor._execute_single(step, None) for step in steps]

    assert all(result.success for result in results)
    # Every handler ran inside its own desktop step, never while a browser reading was being taken.
    assert inside == [(step.action_type.value, True, False) for step in steps]
    # One browser reading, for the browser step only, taken after its desktop step had closed.
    assert readings == [("browser_click", None)]
    assert [bool(result.observation) for result in results] == [False, False, False, True]
    assert all(desktop_observe.recalled(step) is None for step in steps)  # fake handlers read nothing


@pytest.mark.asyncio
async def test_a_scroll_read_in_its_step_stays_verified_after_a_click_changes_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scratch_registry: dict
) -> None:
    from pilot.actions import BrowserParams
    from pilot.agents.postconditions import mismatch, verified

    page = {"address": "news", "y": 0}

    async def scroll(_action: Action) -> str:
        page["y"] = 600
        return "Scrolled down by 500px"

    async def click(_action: Action) -> str:
        page.update(address="article", y=0)
        return "Clicked: a.article"

    async def scrolled(_result: ActionResult) -> Outcome:
        return verified("The page moved down") if page["y"] > 0 else mismatch("The page did not move")

    async def opened(_result: ActionResult) -> Outcome:
        return verified("The article opened") if page["address"] == "article" else mismatch("No article")

    for action_type, check in ((ActionType.BROWSER_SCROLL, scrolled), (ActionType.BROWSER_CLICK, opened)):
        scratch_registry[action_type] = dataclasses.replace(scratch_registry[action_type], check=check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    executor = _executor(tmp_path)
    monkeypatch.setitem(executor._dispatch_table, ActionType.BROWSER_SCROLL, scroll)
    monkeypatch.setitem(executor._dispatch_table, ActionType.BROWSER_CLICK, click)
    steps = [
        Action(action_type=ActionType.BROWSER_SCROLL, target="page", parameters=BrowserParams(direction="down")),
        Action(action_type=ActionType.BROWSER_CLICK, target="article", parameters=BrowserParams(selector="a.article")),
    ]

    results = [await executor._execute_single(step, None) for step in steps]
    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=steps), results)

    # Read now, the scroll would look undone (the click opened a page at the top); it was read in its step.
    assert (await scrolled(results[0])).passed is False
    assert verification.passed is True
    assert verification.independently_observed_actions == [0, 1]


@pytest.fixture
def no_connections(monkeypatch: pytest.MonkeyPatch):
    from pilot.system import connections

    for variable in connections.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    connections.configure(None, None)
    yield connections
    connections.configure(None, None)


class _Store:
    """The credential store: a dict."""

    backend_name = "FakeKeyring"
    available = True

    def __init__(self, **entries: str) -> None:
        self.entries: dict[str, str] = dict(entries)

    async def get_key(self, name: str) -> str | None:
        return self.entries.get(name)

    async def store_key(self, name: str, value: str) -> None:
        self.entries[name] = value

    async def delete_key(self, name: str) -> None:
        self.entries.pop(name, None)


HEADER_SECRET = "fakeHeaderToken0000WXYZ"
BEARER = f"Bearer {HEADER_SECRET}"


def _kept_plan() -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.API_DISCORD,
                target=f"post to {DISCORD_HOOK}",
                parameters=ApiRequestParams(message="Deploy finished", webhook_url=DISCORD_HOOK),
            ),
            Action(
                action_type=ActionType.API_REQUEST,
                target="status",
                parameters=ApiRequestParams(
                    method="GET", url="https://api.example.com/v1/status", headers={"Authorization": BEARER}
                ),
            ),
        ],
        explanation="post and read",
        raw_input="post it",
    )


def _network(monkeypatch: pytest.MonkeyPatch, *answers: tuple[int, object]) -> _Network:
    from pilot.system import api_client

    async def _no_pause(_seconds: float) -> None:
        return None

    network = _Network(*answers)
    fake_outbound.install(monkeypatch, network)
    monkeypatch.setattr(messaging, "_pause", _no_pause)
    return network


@pytest.mark.asyncio
async def test_a_kept_plan_with_a_masked_webhook_and_header_resumes_after_a_restart_and_is_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_connections, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from pilot.security.plan_secrets import PlanSecretKeeper
    from pilot.system import api_client
    from pilot.workflows.checkpoints import WorkflowCheckpointStore

    caplog.set_level(logging.DEBUG)
    store = _Store()
    db = tmp_path / "checkpoints.db"
    await WorkflowCheckpointStore(db, secrets=PlanSecretKeeper(store)).start_plan("plan-1", "post it", _kept_plan())
    assert SECRET not in db.read_bytes().decode("utf-8", "replace")
    assert HEADER_SECRET not in db.read_bytes().decode("utf-8", "replace")

    # The daemon restarts: a new store object reads the plan back, its secrets restored from the credential store.
    resumed = (await WorkflowCheckpointStore(db, secrets=PlanSecretKeeper(store)).get("plan-1")).plan
    discord, status = resumed.actions
    assert discord.parameters.webhook_url == DISCORD_HOOK
    assert status.parameters.headers["Authorization"] == BEARER

    network = _network(
        monkeypatch,
        (200, {"id": "987654321", "channel_id": "555", "content": "Deploy finished"}),
        (200, {"ok": True}),
        (200, {"id": "987654321", "content": "Deploy finished"}),
    )
    sent = await api_client.send_discord_message("Deploy finished", discord.parameters.webhook_url)
    answer = await api_client.api_request("GET", status.parameters.url, headers=status.parameters.headers)
    results = [
        ActionResult(action=discord, success=True, output=sent),
        ActionResult(action=status, success=True, output=str(answer)),
    ]
    verification = await Verifier(MagicMock()).verify(resumed, results)

    assert verification.passed is True and 0 in verification.independently_observed_actions
    assert [method for method, _url in network.requests] == ["POST", "GET", "GET"]
    assert network.requests[0][1].startswith(DISCORD_HOOK)
    for text in (str(verification), resumed.model_dump_json(), caplog.text, sent):
        assert SECRET not in text and HEADER_SECRET not in text


@pytest.mark.asyncio
async def test_a_slack_post_from_a_settings_connection_says_where_to_set_up_its_read_back(
    monkeypatch: pytest.MonkeyPatch, no_connections
) -> None:
    from pilot.system import api_client

    connections = no_connections
    webhook = "https://hooks.slack.com/services/T0FAKE/B0FAKE/fakeSettingsHook1234"
    connections.configure(MagicMock(), _Store(**{connections.SLACK_WEBHOOK: webhook}))
    network = _network(monkeypatch, (200, "ok"))
    action = Action(action_type=ActionType.API_SLACK, target="team", parameters=ApiRequestParams(message="Deploy"))

    output = await api_client.send_slack_message("Deploy")
    plan = ActionPlan(actions=[action])
    results = [ActionResult(action=action, success=True, output=output)]
    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)

    assert network.requests[0][1] == webhook
    assert verification.passed is True and verification.independently_observed_actions == []
    assert connections.WHERE["slack"] in verification.reasons[0]
    assert verification.reason_kinds[0] == "receipt_only"
    assert "fakeSettingsHook1234" not in reply + str(verification)


@pytest.mark.asyncio
async def test_a_late_check_after_its_plan_finished_needs_no_kept_secret_and_writes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_connections, caplog: pytest.LogCaptureFixture
) -> None:
    """The reply went out before the Discord read-back finished; the plan then completed, which removes its kept
    secrets. The late check still reads the message back (the result holds its webhook in memory) and nothing it
    records or logs holds the webhook or the header."""
    import asyncio
    import logging

    from pilot.agents.late_checks import FollowUp, LateVerifications
    from pilot.security.plan_secrets import PlanSecretKeeper
    from pilot.system import api_client
    from pilot.workflows.checkpoints import WorkflowCheckpointStore

    caplog.set_level(logging.DEBUG)
    store = _Store()
    checkpoints = WorkflowCheckpointStore(tmp_path / "c.db", secrets=PlanSecretKeeper(store))
    plan = _kept_plan()
    await checkpoints.start_plan("plan-1", "post it", plan)
    network = _network(
        monkeypatch,
        (200, {"id": "987654321", "channel_id": "555", "content": "Deploy finished"}),
        (200, {"id": "987654321", "content": "Deploy finished"}),
    )
    sent = await api_client.send_discord_message("Deploy finished", DISCORD_HOOK)
    results = [
        ActionResult(action=plan.actions[0], success=True, output=sent),
        ActionResult(action=plan.actions[1], success=True, output='{"status": 200}'),
    ]
    budget_spent = asyncio.Event()

    async def reply_goes_out_first(task, _budget: float) -> None:
        budget_spent.set()  # the reply does not wait for any check

    monkeypatch.setattr(Verifier, "_wait_budget", staticmethod(reply_goes_out_first))
    reports: list = []
    told: list[str] = []

    async def record(report) -> None:
        reports.append(report)

    async def tell(sentence: str, _report, _spoken: bool) -> None:
        told.append(sentence)

    verification, late = await Verifier(MagicMock()).verify_within(plan, results, budget=3.0)
    await checkpoints.mark_status("plan-1", "complete")
    assert "plan_secret:plan-1" not in store.entries
    assert late is not None and 0 in late.pending

    follower = LateVerifications(record=record, tell=tell)
    follower.follow(late, FollowUp(task_key="task-1", plan_id="plan-1", request="post it"))
    await late.task
    for _ in range(50):
        if reports:
            break
        await asyncio.sleep(0)

    assert reports and verification.passed is True
    assert any(check["step"] == 1 and check["kind"] == "verified" for check in reports[0].checks)
    assert told == []
    assert [method for method, _url in network.requests] == ["POST", "GET"]
    written = repr(reports) + caplog.text
    assert SECRET not in written and HEADER_SECRET not in written
