"""The area registry of post-condition verifiers: loading, registration rules and dispatch."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents import postconditions
from pilot.agents import verifier as verifier_module
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier

# The 22 verifiers that existed before the area registry; new ones only add to this set.
ORIGINAL_22 = {
    ActionType.BROWSER_NAVIGATE: "conditional_exact_url_postcondition",
    ActionType.BROWSER_TYPE: "conditional_field_value_postcondition",
    ActionType.BROWSER_SELECT: "conditional_field_value_postcondition",
    ActionType.BROWSER_FILL_FORM: "conditional_field_value_postcondition",
    ActionType.FILE_WRITE: "file_content_postcondition",
    ActionType.FILE_DELETE: "file_absence_postcondition",
    ActionType.FILE_COPY: "copy_destination_postcondition",
    ActionType.FILE_MOVE: "move_source_and_destination_postcondition",
    ActionType.PACKAGE_INSTALL: "package_installed_postcondition",
    ActionType.PACKAGE_REMOVE: "package_removed_postcondition",
    ActionType.SERVICE_START: "service_active_postcondition",
    ActionType.SERVICE_RESTART: "service_active_postcondition",
    ActionType.SERVICE_STOP: "service_inactive_postcondition",
    ActionType.GNOME_SETTING_WRITE: "setting_value_postcondition",
    ActionType.DOWNLOAD_FILE: "download_file_exists_postcondition",
    ActionType.PROCESS_KILL: "process_absence_postcondition",
    ActionType.POWER_SHUTDOWN: "shutdown_transition_postcondition",
    ActionType.POWER_RESTART: "restart_transition_postcondition",
    ActionType.POWER_LOGOUT: "logout_transition_postcondition",
    ActionType.WINDOW_CLOSE: "window_absence_postcondition",
    ActionType.DISK_UNMOUNT: "mount_absence_postcondition",
    ActionType.SCHEDULE_DELETE: "schedule_absence_postcondition",
}

# The pre-checks of the original 22, less schedule_delete: a name that matches nothing must run and fail.
ORIGINAL_PRECHECKABLE = {
    ActionType.FILE_WRITE,
    ActionType.FILE_DELETE,
    ActionType.PACKAGE_INSTALL,
    ActionType.PACKAGE_REMOVE,
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.GNOME_SETTING_WRITE,
    ActionType.PROCESS_KILL,
    ActionType.WINDOW_CLOSE,
    ActionType.DISK_UNMOUNT,
}


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Register into a copy, so a test's fake verifier never reaches the real registry."""
    registry = dict(postconditions._REGISTRY)
    monkeypatch.setattr(postconditions, "_REGISTRY", registry)
    return registry


def test_the_original_22_keep_their_catalog_names() -> None:
    assert {action: POSTCONDITION_VERIFIERS[action] for action in ORIGINAL_22} == ORIGINAL_22


def test_the_original_prechecks_are_kept() -> None:
    assert ORIGINAL_PRECHECKABLE <= verifier_module._PRECHECKABLE_POSTCONDITIONS


def test_catalog_names_follow_the_registry_and_its_rules() -> None:
    area = postconditions.registrations()
    for action, method in POSTCONDITION_VERIFIERS.items():
        assert method.endswith("_postcondition"), action
        if action in area:
            assert area[action].method == method
            assert area[action].module.startswith("pilot.agents.postconditions.")
    assert not set(area) & set(verifier_module._BUILTIN_POSTCONDITION_VERIFIERS)


def test_schedule_delete_lives_in_its_area_module() -> None:
    registration = postconditions.registrations()[ActionType.SCHEDULE_DELETE]
    assert registration.module == "pilot.agents.postconditions.triggers_schedules"


def test_a_second_verifier_for_one_action_is_refused(scratch_registry: dict) -> None:
    async def other(_result: ActionResult) -> Outcome:
        return verified("never")

    with pytest.raises(ValueError, match="already has a verifier"):
        register(ActionType.SCHEDULE_DELETE, method="other_postcondition")(other)


@pytest.mark.parametrize("method", ["", "schedule_absence", "check"])
def test_a_method_name_must_name_a_postcondition(method: str) -> None:
    with pytest.raises(ValueError, match="_postcondition"):
        register(ActionType.NOTIFY, method=method)


def _verify(action_type: ActionType) -> tuple[Action, ActionResult]:
    action = Action(action_type=action_type, target="x", parameters=FileParams(path="x"))
    return action, ActionResult(action=action, success=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "passed", "observed", "label"),
    [
        (verified("seen"), True, [0], "VERIFIED"),
        (mismatch("expected a, found b"), False, [0], "MISMATCH — expected a, found b"),
        (unobserved("no credentials"), True, [], "REPORTED — no independent postcondition"),
    ],
)
async def test_registered_outcomes_reach_the_verification_result(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict, outcome: Outcome, passed: bool, observed, label: str
) -> None:
    async def check(_result: ActionResult) -> Outcome:
        return outcome

    register(ActionType.FILE_HASH, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action, result = _verify(ActionType.FILE_HASH)

    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])

    assert verification.passed is passed
    assert verification.independently_observed_actions == observed
    assert label in verification.details[0]


@pytest.mark.asyncio
async def test_a_raising_verifier_fails_closed(monkeypatch: pytest.MonkeyPatch, scratch_registry: dict) -> None:
    async def broken(_result: ActionResult) -> Outcome:
        raise RuntimeError("observer unavailable")

    register(ActionType.FILE_HASH, method="test_postcondition")(broken)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action, result = _verify(ActionType.FILE_HASH)

    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])

    assert verification.passed is False
    assert verification.independently_observed_actions == []
    assert "Verification error: observer unavailable" in verification.details[0]


@pytest.mark.asyncio
async def test_a_verifier_past_its_time_limit_is_unverified(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    never = asyncio.Event()

    async def hangs(_result: ActionResult) -> Outcome:
        await never.wait()
        return verified("never")

    # A zero limit makes bounded() cancel at once: no real time passes.
    register(ActionType.FILE_HASH, method="test_postcondition", timeout=1)(hangs)
    registration = scratch_registry[ActionType.FILE_HASH]
    monkeypatch.setitem(
        scratch_registry,
        ActionType.FILE_HASH,
        postconditions.Registration(**{**registration.__dict__, "timeout": 0}),
    )
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action, result = _verify(ActionType.FILE_HASH)

    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_a_failed_action_never_reaches_its_verifier(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    calls: list[ActionResult] = []

    async def check(result: ActionResult) -> Outcome:
        calls.append(result)
        return verified("seen")

    register(ActionType.FILE_HASH, method="test_postcondition")(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    action = Action(action_type=ActionType.FILE_HASH, target="x", parameters=FileParams(path="x"))

    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=False, error="handler raised")]
    )

    assert calls == []
    assert verification.passed is False
    assert "FAILED — handler raised" in verification.details[0]


# ── Pre-checks: only an observed "the post-condition already holds" may skip an action ──


def _unobservable(kind: str):
    async def check(_result: ActionResult) -> Outcome:
        if kind == "unobserved":
            return unobserved("no credentials to look with")
        if kind == "raises":
            raise RuntimeError("observer unavailable")
        if kind == "mismatch":
            return mismatch("expected absent, found present")
        await asyncio.Event().wait()  # never finishes; a zero time limit cancels it at once
        return verified("never")

    return check


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unobserved", "raises", "times_out", "mismatch"])
@pytest.mark.parametrize("action_type", sorted(verifier_module._PRECHECKABLE_POSTCONDITIONS, key=lambda a: a.value))
async def test_a_precheck_that_did_not_observe_the_state_never_skips_the_action(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict, action_type: ActionType, kind: str
) -> None:
    """Every action with a pre-check, area-registered or not, put behind a check that cannot confirm the state."""
    scratch_registry[action_type] = postconditions.Registration(
        action_type=action_type,
        method="test_postcondition",
        check=_unobservable(kind),
        precheck=True,
        timeout=0 if kind == "times_out" else 20,
        module="tests",
    )
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    from pilot.actions import PackageParams

    parameters = PackageParams(name="x") if action_type == ActionType.PACKAGE_INSTALL else FileParams(path="x")
    action = Action(action_type=action_type, target="x", parameters=parameters)

    state = await Verifier(MagicMock()).check_already_satisfied(action)

    assert state is None or state[0] is False


@pytest.mark.asyncio
async def test_an_observed_precheck_still_skips(monkeypatch: pytest.MonkeyPatch, scratch_registry: dict) -> None:
    async def present(_result: ActionResult) -> Outcome:
        return verified("already there")

    register(ActionType.FILE_HASH, method="test_postcondition", precheck=True)(present)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    monkeypatch.setattr(
        verifier_module,
        "_PRECHECKABLE_POSTCONDITIONS",
        verifier_module._PRECHECKABLE_POSTCONDITIONS | {ActionType.FILE_HASH},
    )
    action = Action(action_type=ActionType.FILE_HASH, target="x", parameters=FileParams(path="x"))

    assert await Verifier(MagicMock()).check_already_satisfied(action) == (True, "already there")


def test_every_registered_precheck_is_declared() -> None:
    area = postconditions.registrations()
    for action, registration in area.items():
        assert (action in verifier_module._PRECHECKABLE_POSTCONDITIONS) is registration.precheck


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", [ActionType.SCHEDULE_DELETE, ActionType.TRIGGER_DELETE])
async def test_a_delete_of_a_name_that_matches_nothing_is_never_skipped(
    monkeypatch: pytest.MonkeyPatch, action_type: ActionType
) -> None:
    """The rule, whatever pre-check a series gives these deletes: a name that matches nothing reaches the handler
    and fails there. The only skip allowed is a repeat of a delete this same run already made and verified."""
    from unittest.mock import AsyncMock

    from pilot.actions import ScheduleParams, TriggerParams
    from pilot.system import scheduler

    monkeypatch.setattr(scheduler, "schedule_exists", AsyncMock(return_value=False))
    parameters = (
        ScheduleParams(name="no-such-task")
        if action_type == ActionType.SCHEDULE_DELETE
        else TriggerParams(name="no-such-trigger")
    )
    action = Action(action_type=action_type, target="no-such", parameters=parameters)

    state = await Verifier(MagicMock()).check_already_satisfied(action)

    assert state is None or state[0] is False


@pytest.mark.asyncio
async def test_a_verifier_can_tell_a_precheck_from_a_check_after_the_action(
    monkeypatch: pytest.MonkeyPatch, scratch_registry: dict
) -> None:
    seen: list[bool] = []

    async def check(result: ActionResult) -> Outcome:
        seen.append(postconditions.is_precheck(result))
        return mismatch("not there yet")

    register(ActionType.FILE_HASH, method="test_postcondition", precheck=True)(check)
    monkeypatch.setattr(verifier_module, "_AREA_VERIFIERS", postconditions.registrations())
    monkeypatch.setattr(
        verifier_module,
        "_PRECHECKABLE_POSTCONDITIONS",
        verifier_module._PRECHECKABLE_POSTCONDITIONS | {ActionType.FILE_HASH},
    )
    action = Action(action_type=ActionType.FILE_HASH, target="x", parameters=FileParams(path="x"))

    await Verifier(MagicMock()).check_already_satisfied(action)
    await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])

    assert seen == [True, False]
