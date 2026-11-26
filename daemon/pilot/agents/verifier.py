"""Verifier agent — confirms execution outcomes match intended results.

Can trigger rollback if verification detects a mismatch between
expected and actual system state after execution.

Updated for the expanded action set with cross-platform support.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    DiskManageParams,
    FileParams,
    GnomeSettingParams,
    PackageParams,
    ProcessParams,
    ServiceParams,
    VerificationResult,
    WindowParams,
)
from pilot.agents import postconditions
from pilot.agents.postconditions import PRECHECK
from pilot.agents.unconfirmed import VERIFY_GRACE_SECONDS, Kind, classify, no_check_kind
from pilot.timeouts import bounded

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.verifier")


# Verifiers written before the area registry. New verifiers go in a module of
# pilot.agents.postconditions and register themselves there.
_BUILTIN_POSTCONDITION_VERIFIERS: dict[ActionType, str] = {
    ActionType.FILE_DELETE: "file_absence_postcondition",
    ActionType.FILE_MOVE: "move_source_and_destination_postcondition",
    ActionType.PACKAGE_INSTALL: "package_installed_postcondition",
    ActionType.PACKAGE_REMOVE: "package_removed_postcondition",
    ActionType.SERVICE_START: "service_active_postcondition",
    ActionType.SERVICE_RESTART: "service_active_postcondition",
    ActionType.SERVICE_STOP: "service_inactive_postcondition",
    ActionType.GNOME_SETTING_WRITE: "setting_value_postcondition",
    ActionType.PROCESS_KILL: "process_absence_postcondition",
    ActionType.POWER_SHUTDOWN: "shutdown_transition_postcondition",
    ActionType.POWER_RESTART: "restart_transition_postcondition",
    ActionType.POWER_LOGOUT: "logout_transition_postcondition",
    ActionType.WINDOW_CLOSE: "window_absence_postcondition",
    ActionType.DISK_UNMOUNT: "mount_absence_postcondition",
}

_AREA_VERIFIERS = postconditions.load_area_modules()
_DOUBLY_REGISTERED = sorted(action.value for action in set(_AREA_VERIFIERS) & set(_BUILTIN_POSTCONDITION_VERIFIERS))
if _DOUBLY_REGISTERED:
    raise RuntimeError(f"Post-condition verifiers registered twice: {', '.join(_DOUBLY_REGISTERED)}")

# Public evidence generation imports this registry so the catalog cannot claim
# an independent verifier that is absent from the runtime verification path.
POSTCONDITION_VERIFIERS: dict[ActionType, str] = {
    **_BUILTIN_POSTCONDITION_VERIFIERS,
    **{action: registration.method for action, registration in _AREA_VERIFIERS.items()},
}

_PRECHECKABLE_POSTCONDITIONS = frozenset(
    {
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
    | {action for action, registration in _AREA_VERIFIERS.items() if registration.precheck}
)


def _process_label(params: Any) -> str:
    return f"Process {params.pid}" if getattr(params, "pid", None) is not None else str(getattr(params, "name", ""))


# Deleting, stopping, removing, closing or unmounting something observed to be absent skips the action: the goal
# state holds, and plans rely on that (a clean-up step for a file that may not exist, closing a window that is already
# gone). Heliox then says what it found, never a success verb for something that did not happen; the executor marks
# these skips "already_absent". Triggers and scheduled tasks are different on purpose: they have no pre-check and fail
# on a name that matches nothing, because a wrongly named automation keeps running unseen, so the plan must stop and
# the person must be told. An absent file, process, package, window or mount leaves nothing running.
ALREADY_ABSENT_WORDING: dict[ActionType, Callable[[Any], str]] = {
    ActionType.FILE_DELETE: lambda p: f"There was no file at {p.path}, so nothing was deleted",
    ActionType.PROCESS_KILL: lambda p: f"{_process_label(p)} was not running, so nothing was stopped",
    ActionType.PACKAGE_REMOVE: lambda p: f"{p.name} is not installed, so nothing was removed",
    ActionType.WINDOW_CLOSE: (
        lambda p: f"{p.window_id or p.title or p.process_name} was not open, so nothing was closed"
    ),
    ActionType.DISK_UNMOUNT: lambda p: f"{p.mount_point or p.device} was not mounted, so nothing was unmounted",
}


NO_POSTCONDITION = "No independent postcondition for this action type"
STILL_CHECKING = "The check was still running when the reply went out; its outcome goes to the receipt"


@dataclass
class LateOutcome:
    """A check that finished after the reply: its step, its outcome and its kind (None: plainly verified)."""

    index: int
    result: ActionResult
    outcome: postconditions.Outcome
    kind: Kind | None


@dataclass
class LateChecks:
    """Checks still running behind the reply (``Verifier.verify_within``). ``task`` runs them in the plan's order."""

    task: asyncio.Task[Any]
    pending: list[int]
    results: list[ActionResult]

    async def outcomes(self) -> list[LateOutcome]:
        """Wait for the checks to finish; the outcome of each step that was still being checked, in order."""
        found = await asyncio.shield(self.task)
        late = []
        for index in self.pending:
            outcome = found.get(index)
            if outcome is None:
                continue
            result = self.results[index]
            kind = (
                classify(
                    result.action.action_type,
                    outcome.detail,
                    observed=outcome.observed,
                    passed=outcome.passed,
                    hint=outcome.kind,
                )
                if outcome.detail != NO_POSTCONDITION
                else no_check_kind(result.action.action_type)
            )
            late.append(LateOutcome(index, result, outcome, kind))
        return late

    async def cancel(self) -> None:
        """Stop the checks and wait until they have stopped."""
        if not self.task.done():
            self.task.cancel()
        await asyncio.wait({self.task})


class Verifier:
    """Verifies execution results against expected outcomes."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router
        # Declared checks (Action.expect) read the same configuration and vault the actions use.
        from pilot.agents.postconditions.declared import bind_router

        bind_router(model_router)

    async def check_already_satisfied(self, action: Action) -> tuple[bool, str] | None:
        """Prove whether an action's intended state is already present.

        Only deterministic, side-effect-free checks are eligible. ``None``
        means execution must continue; a failed or unavailable check never
        suppresses an action.
        """

        if action.action_type in {
            ActionType.BROWSER_NAVIGATE,
            ActionType.BROWSER_TYPE,
            ActionType.BROWSER_SELECT,
            ActionType.BROWSER_FILL_FORM,
        }:
            if getattr(action.parameters, "tab_index", -1) != -1:
                return None
            from pilot.system.browser import browser_action_already_satisfied

            return await browser_action_already_satisfied(action.action_type.value, action.parameters)

        if action.action_type not in _PRECHECKABLE_POSTCONDITIONS:
            return None
        if action.action_type == ActionType.PACKAGE_INSTALL:
            params: PackageParams = action.parameters  # type: ignore[assignment]
            if params.version or params.repository:
                return None
        probe = ActionResult(action=action, success=True, executed=False, skip_reason=PRECHECK)
        passed, detail, observed = await self._verify_single(probe)
        if not observed:
            # Could not look (no credentials, no probe on this platform, time limit, an error): that is not
            # evidence the state is already there, so the action runs.
            return None
        if passed and action.action_type in ALREADY_ABSENT_WORDING:
            return True, ALREADY_ABSENT_WORDING[action.action_type](action.parameters)
        return passed, detail

    async def verify(self, plan: ActionPlan, results: list[ActionResult]) -> VerificationResult:
        """Verify all action results against the plan, waiting for every check."""
        verification, _late = await self.verify_within(plan, results, budget=None)
        return verification

    async def verify_within(
        self, plan: ActionPlan, results: list[ActionResult], *, budget: float | None
    ) -> tuple[VerificationResult, LateChecks | None]:
        """Verify, waiting at most ``budget`` seconds for the checks together (None: for all of them).

        Checks run one after another, in the plan's order, as before. A check still running when the budget is spent
        goes on behind the reply: its step is reported "still being checked" now (``pending_actions``), and the
        returned ``LateChecks`` gives its outcome when it arrives. A late outcome only informs: whoever holds the
        ``LateChecks`` writes it to the receipt and may tell the person; it never fails this result or starts a
        repair. The caller must cancel and await ``LateChecks.task`` when it stops caring (``LateChecks.cancel``).
        """
        details: dict[int, str] = {}
        head: list[str] = []
        failed_indices: list[int] = []
        observed_indices: list[int] = []
        reasons: dict[int, str] = {}
        kinds: dict[int, str] = {}
        notices: dict[int, str] = {}
        to_check: list[int] = []

        if len(results) != len(plan.actions):
            head.append(
                f"RESULT COUNT MISMATCH — planned {len(plan.actions)} action(s), received {len(results)} result(s)"
            )

        for i, planned_action in enumerate(plan.actions):
            if i >= len(results):
                details[i] = f"Action {i} ({planned_action.action_type.value}): MISSING — executor returned no result"
                failed_indices.append(i)
                continue

            result = results[i]
            if result.action.model_dump(mode="json") != planned_action.model_dump(mode="json"):
                details[i] = (
                    f"Action {i} ({planned_action.action_type.value}): MISMATCH — "
                    f"executor returned a result for {result.action.action_type.value}"
                )
                failed_indices.append(i)
                continue

            if not result.success:
                details[i] = f"Action {i} ({result.action.action_type.value}): FAILED — {result.error}"
                failed_indices.append(i)
                continue
            to_check.append(i)

        outcomes: dict[int, postconditions.Outcome] = {}

        async def run_checks() -> dict[int, postconditions.Outcome]:
            for index in to_check:
                outcomes[index] = await self._verify_one(results[index])
            return outcomes

        late: LateChecks | None = None
        if budget is None or not to_check:
            await run_checks()
        else:
            task = asyncio.create_task(run_checks(), name="verification-checks")
            try:
                # The grace covers the last read of a check whose own wait ends inside the budget.
                await self._wait_budget(task, budget + VERIFY_GRACE_SECONDS)
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.wait({task})
                raise
            if not task.done():
                pending = [index for index in to_check if index not in outcomes]
                late = LateChecks(task=task, pending=pending, results=list(results))
            elif not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]

        for i in to_check:
            result = results[i]
            label = f"Action {i} ({result.action.action_type.value})"
            if i not in outcomes:
                details[i] = f"{label}: CHECKING — the check was still running when the reply went out"
                reasons[i] = STILL_CHECKING
                kinds[i] = Kind.STILL_CHECKING.value
                continue
            outcome = outcomes[i]
            check_passed, check_detail, independently_observed = outcome.passed, outcome.detail, outcome.observed
            if independently_observed:
                observed_indices.append(i)
            if outcome.notice:
                notices[i] = outcome.notice
            kind = (
                classify(
                    result.action.action_type,
                    check_detail,
                    observed=independently_observed,
                    passed=check_passed,
                    hint=outcome.kind,
                )
                if check_detail != NO_POSTCONDITION
                else no_check_kind(result.action.action_type)
            )
            if kind is not None:
                kinds[i] = kind.value
            if check_passed:
                status = "VERIFIED" if independently_observed else "REPORTED — no independent postcondition"
                if not independently_observed and check_detail and check_detail != NO_POSTCONDITION:
                    # Why it was not confirmed (a receipt only, no credentials to look with, a time limit).
                    status += f" ({check_detail})"
                    reasons[i] = check_detail
                elif result.action.expect is not None and check_detail:
                    # A declared check says what it ran and found (postconditions.declared).
                    status = f"{status} ({check_detail})"
                details[i] = f"{label}: {status}"
            else:
                details[i] = f"{label}: MISMATCH — {check_detail}"
                failed_indices.append(i)
                reasons[i] = check_detail

        tail = [
            f"Action {i} ({result.action.action_type.value}): UNEXPECTED — result was not present in the plan"
            for i, result in enumerate(results[len(plan.actions) :], start=len(plan.actions))
        ]
        failed_indices.extend(range(len(plan.actions), len(results)))

        passed = len(failed_indices) == 0
        verification = VerificationResult(
            passed=passed,
            details=[*head, *(details[i] for i in sorted(details)), *tail],
            failed_actions=sorted(failed_indices),
            independently_observed_actions=observed_indices,
            rollback_triggered=False,
            reasons=reasons,
            reason_kinds=kinds,
            notices=notices,
            pending_actions=late.pending if late is not None else [],
        )
        return verification, late

    @staticmethod
    async def _wait_budget(task: asyncio.Task[Any], budget: float) -> None:
        """Wait for ``task`` at most ``budget`` seconds; it goes on running past that (tests replace this wait)."""
        try:
            await bounded(asyncio.shield(task), budget)
        except TimeoutError:
            pass

    async def _verify_single(self, result: ActionResult) -> tuple[bool, str, bool]:
        """Return (passed, detail, independent observation completed)."""
        outcome = await self._verify_one(result)
        return outcome.passed, outcome.detail, outcome.observed

    async def _verify_one(self, result: ActionResult) -> postconditions.Outcome:
        """The check of one step, with the kind of an outcome that is not plainly verified."""
        action = result.action

        try:
            registration = _AREA_VERIFIERS.get(action.action_type)
            if registration is not None:
                from pilot.agents.postconditions._in_step import recalled

                # A browser step is read right after it ran (the executor keeps that reading); a later step may
                # have left its page by now.
                read_then = recalled(result)
                if read_then is not None:
                    return read_then
                try:
                    return await bounded(registration.check(result), registration.timeout)
                except TimeoutError:
                    return postconditions.unobserved(
                        f"Post-condition check did not finish within {registration.timeout:g} s; outcome unverified",
                        kind=Kind.COULD_NOT_LOOK.value,
                    )

            if action.action_type == ActionType.FILE_DELETE:
                check = await self._verify_file_delete(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.PACKAGE_INSTALL:
                check = await self._verify_package_install(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.PACKAGE_REMOVE:
                check = await self._verify_package_remove(action.parameters)  # type: ignore[arg-type]

            elif action.action_type in (ActionType.SERVICE_START, ActionType.SERVICE_RESTART):
                check = await self._verify_service_running(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.SERVICE_STOP:
                check = await self._verify_service_stopped(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.GNOME_SETTING_WRITE:
                check = await self._verify_gnome_setting(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.FILE_MOVE:
                check = await self._verify_file_move(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.PROCESS_KILL:
                check = await self._verify_process_kill(action.parameters)  # type: ignore[arg-type]

            elif action.action_type in {
                ActionType.POWER_SHUTDOWN,
                ActionType.POWER_RESTART,
                ActionType.POWER_LOGOUT,
            }:
                check = await self._verify_power_transition(action.action_type)

            elif action.action_type == ActionType.WINDOW_CLOSE:
                check = await self._verify_window_close(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.DISK_UNMOUNT:
                check = await self._verify_disk_unmount(action.parameters)  # type: ignore[arg-type]

            else:
                # The executor callback alone is not evidence of a state transition.
                return postconditions.Outcome(True, NO_POSTCONDITION, False)

            return postconditions.Outcome(check[0], check[1], True)

        except Exception as e:
            logger.warning("Verification check failed: %s", e)
            return postconditions.Outcome(False, f"Verification error: {e}", False)

    async def _verify_file_delete(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.path:
            raise ValueError("File delete has no path to observe")
        if Path(params.path).exists():
            return False, f"File still exists after delete: {params.path}"
        return True, "File successfully deleted"

    async def _verify_file_move(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.path or not params.destination:
            raise ValueError("File move has no source or destination to observe")
        if not Path(params.destination).exists():
            return False, f"Destination does not exist after move: {params.destination}"
        if Path(params.path).exists():
            return False, f"Source still exists after move: {params.path}"
        return True, "Move verified: source removed, destination exists"

    async def _verify_package_install(self, params: PackageParams) -> tuple[bool, str]:
        from pilot.system.package_mgr import is_installed

        if not params.name:
            raise ValueError("Package install has no name to observe")
        if await is_installed(params.name):
            return True, f"Package {params.name} is installed"
        return False, f"Package {params.name} is not installed after install"

    async def _verify_package_remove(self, params: PackageParams) -> tuple[bool, str]:
        from pilot.system.package_mgr import is_installed

        if not params.name:
            raise ValueError("Package remove has no name to observe")
        if not await is_installed(params.name):
            return True, f"Package {params.name} is removed"
        return False, f"Package {params.name} is still installed after remove"

    async def _verify_service_running(self, params: ServiceParams) -> tuple[bool, str]:
        from pilot.system.systemctl import is_active

        if not params.name:
            raise ValueError("Service action has no name to observe")
        if await is_active(params.name, user_scope=params.user_scope):
            return True, f"Service {params.name} is active"
        return False, f"Service {params.name} is not active"

    async def _verify_service_stopped(self, params: ServiceParams) -> tuple[bool, str]:
        from pilot.system.systemctl import is_active

        if not params.name:
            raise ValueError("Service stop has no name to observe")
        if not await is_active(params.name, user_scope=params.user_scope):
            return True, f"Service {params.name} is stopped"
        return False, f"Service {params.name} is still active after stop"

    async def _verify_gnome_setting(self, params: GnomeSettingParams) -> tuple[bool, str]:
        from pilot.system.gnome import get_setting

        if not params.schema_id or not params.key or params.value is None:
            raise ValueError("Setting write has no schema, key, or value to observe")
        actual = await get_setting(params.schema_id, params.key)
        expected = params.value.strip("'\"")
        actual_clean = actual.strip("'\"")
        if actual_clean == expected:
            return True, f"Setting {params.key} = {actual}"
        return False, f"Setting mismatch: expected {expected}, got {actual_clean}"

    async def _verify_process_kill(self, params: ProcessParams) -> tuple[bool, str]:
        from pilot.system.processes import process_exists

        if params.pid is None and not params.name:
            raise ValueError("Process kill has no PID or process name to observe")
        if await process_exists(pid=params.pid, name=params.name):
            return False, f"Process is still running: {params.pid if params.pid is not None else params.name}"
        return True, f"Process is absent: {params.pid if params.pid is not None else params.name}"

    async def _verify_power_transition(self, action_type: ActionType) -> tuple[bool, str]:
        from pilot.system.power import power_transition_observed

        transition = action_type.value.removeprefix("power_")
        if await power_transition_observed(transition):
            return True, f"Host independently reported the {transition} transition"
        return False, f"Host did not independently report the {transition} transition"

    async def _verify_window_close(self, params: WindowParams) -> tuple[bool, str]:
        from pilot.system.window_mgr import window_exists

        target = params.window_id or params.title or params.process_name
        if not target:
            raise ValueError("Window close has no window selector to observe")
        if await window_exists(
            window_id=params.window_id,
            title=params.title,
            process_name=params.process_name,
        ):
            return False, f"Window is still open: {target}"
        return True, f"Window is absent: {target}"

    async def _verify_disk_unmount(self, params: DiskManageParams) -> tuple[bool, str]:
        from pilot.system.disks import mount_exists

        target = params.mount_point or params.device
        if not target:
            raise ValueError("Disk unmount has no device or mount point to observe")
        if await mount_exists(device=params.device, mount_point=params.mount_point):
            return False, f"Mount is still active: {target}"
        return True, f"Mount is absent: {target}"
