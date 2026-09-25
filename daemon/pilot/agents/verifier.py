"""Verifier agent — confirms execution outcomes match intended results.

Can trigger rollback if verification detects a mismatch between
expected and actual system state after execution.

Updated for the expanded action set with cross-platform support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    ScheduleParams,
    ServiceParams,
    VerificationResult,
    WindowParams,
)

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.verifier")


# Public evidence generation imports this registry so the catalog cannot claim
# an independent verifier that is absent from the runtime verification path.
POSTCONDITION_VERIFIERS: dict[ActionType, str] = {
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

_PRECHECKABLE_POSTCONDITIONS = frozenset(
    {
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
        ActionType.SCHEDULE_DELETE,
    }
)


class Verifier:
    """Verifies execution results against expected outcomes."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router

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
            from pilot.system.browser import browser_action_already_satisfied

            return await browser_action_already_satisfied(action.action_type.value, action.parameters)

        if action.action_type not in _PRECHECKABLE_POSTCONDITIONS:
            return None
        if action.action_type == ActionType.PACKAGE_INSTALL:
            params: PackageParams = action.parameters  # type: ignore[assignment]
            if params.version or params.repository:
                return None
        passed, detail, _ = await self._verify_single(ActionResult(action=action, success=True))
        return passed, detail

    async def verify(self, plan: ActionPlan, results: list[ActionResult]) -> VerificationResult:
        """Verify all action results against the plan."""
        details: list[str] = []
        failed_indices: list[int] = []
        observed_indices: list[int] = []

        if len(results) != len(plan.actions):
            details.append(
                f"RESULT COUNT MISMATCH — planned {len(plan.actions)} action(s), received {len(results)} result(s)"
            )

        for i, planned_action in enumerate(plan.actions):
            if i >= len(results):
                details.append(
                    f"Action {i} ({planned_action.action_type.value}): MISSING — executor returned no result"
                )
                failed_indices.append(i)
                continue

            result = results[i]
            if result.action.model_dump(mode="json") != planned_action.model_dump(mode="json"):
                details.append(
                    f"Action {i} ({planned_action.action_type.value}): MISMATCH — "
                    f"executor returned a result for {result.action.action_type.value}"
                )
                failed_indices.append(i)
                continue

            if not result.success:
                details.append(f"Action {i} ({result.action.action_type.value}): FAILED — {result.error}")
                failed_indices.append(i)
                continue

            check_passed, check_detail, independently_observed = await self._verify_single(result)
            if independently_observed:
                observed_indices.append(i)
            if check_passed:
                details.append(f"Action {i} ({result.action.action_type.value}): VERIFIED")
            else:
                details.append(f"Action {i} ({result.action.action_type.value}): MISMATCH — {check_detail}")
                failed_indices.append(i)

        for i, result in enumerate(results[len(plan.actions) :], start=len(plan.actions)):
            details.append(
                f"Action {i} ({result.action.action_type.value}): UNEXPECTED — result was not present in the plan"
            )
            failed_indices.append(i)

        passed = len(failed_indices) == 0
        return VerificationResult(
            passed=passed,
            details=details,
            failed_actions=failed_indices,
            independently_observed_actions=observed_indices,
            rollback_triggered=False,
        )

    async def _verify_single(self, result: ActionResult) -> tuple[bool, str, bool]:
        """Return (passed, detail, independent observation completed)."""
        action = result.action

        try:
            if action.action_type == ActionType.FILE_WRITE:
                check = await self._verify_file_write(action.parameters)  # type: ignore[arg-type]

            elif action.action_type == ActionType.FILE_DELETE:
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

            elif action.action_type == ActionType.DOWNLOAD_FILE:
                check = await self._verify_download(result)

            elif action.action_type == ActionType.FILE_COPY:
                check = await self._verify_file_copy(action.parameters)  # type: ignore[arg-type]

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

            elif action.action_type == ActionType.SCHEDULE_DELETE:
                check = await self._verify_schedule_delete(action.parameters)  # type: ignore[arg-type]

            else:
                # The executor callback alone is not evidence of a state transition.
                return True, "No independent postcondition for this action type", False

            return check[0], check[1], True

        except Exception as e:
            logger.warning("Verification check failed: %s", e)
            return False, f"Verification error: {e}", False

    async def _verify_file_write(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.path:
            raise ValueError("File write has no path to observe")
        p = Path(params.path)
        if not p.exists():
            return False, f"File does not exist after write: {params.path}"
        if params.content is not None:
            actual = p.read_text("utf-8")
            if actual != params.content:
                return False, f"File content mismatch (expected {len(params.content)} bytes, got {len(actual)})"
        return True, "File exists with expected content"

    async def _verify_file_delete(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.path:
            raise ValueError("File delete has no path to observe")
        if Path(params.path).exists():
            return False, f"File still exists after delete: {params.path}"
        return True, "File successfully deleted"

    async def _verify_file_copy(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.path or not params.destination:
            raise ValueError("File copy has no source or destination to observe")
        if not Path(params.destination).exists():
            return False, f"Destination does not exist after copy: {params.destination}"
        return True, "Copy destination exists"

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

    async def _verify_download(self, result: ActionResult) -> tuple[bool, str]:
        from pathlib import Path

        from pilot.actions import DownloadParams

        params: DownloadParams = result.action.parameters  # type: ignore[assignment]
        if not params.output_path:
            raise ValueError("Download has no output path to observe")
        if Path(params.output_path).exists():
            size = Path(params.output_path).stat().st_size
            return True, f"File downloaded: {params.output_path} ({size:,} bytes)"
        return False, f"Downloaded file not found: {params.output_path}"

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

    async def _verify_schedule_delete(self, params: ScheduleParams) -> tuple[bool, str]:
        from pilot.system.scheduler import schedule_exists

        target = params.task_id or params.name
        if not target:
            raise ValueError("Schedule delete has no task identifier to observe")
        if await schedule_exists(target):
            return False, f"Scheduled task still exists: {target}"
        return True, f"Scheduled task is absent: {target}"
