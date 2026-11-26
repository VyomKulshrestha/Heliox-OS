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


class _ClientContext:
    def __init__(self, client: object) -> None:
        self._client = client

    async def __aenter__(self) -> object:
        return self._client

    async def __aexit__(self, *_args: object) -> None:
        return None


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

    async def _public(*_args: object, **_kwargs: object) -> None:
        return None

    async def _no_pause(_seconds: float) -> None:
        return None

    network = _Network(
        (200, {"id": "987654321", "channel_id": "555", "content": "Deploy finished"}),
        (200, {"id": "987654321", "content": "Deploy finished"}),
    )
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", lambda *_a, **_k: _ClientContext(network))
    monkeypatch.setattr(api_client, "_validate_outbound_url", _public)
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
