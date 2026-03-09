"""Verifier agent — confirms execution outcomes match intended results.

Can trigger rollback if verification detects a mismatch between
expected and actual system state after execution.

Updated for the expanded action set with cross-platform support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pilot.actions import (
    ActionPlan,
    ActionResult,
    ActionType,
    FileParams,
    GnomeSettingParams,
    PackageParams,
    ServiceParams,
    VerificationResult,
)

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.verifier")


class Verifier:
    """Verifies execution results against expected outcomes."""

    def __init__(self, model_router: ModelRouter) -> None:
        self._model = model_router

    async def verify(self, plan: ActionPlan, results: list[ActionResult]) -> VerificationResult:
        """Verify all action results against the plan."""
        details: list[str] = []
        failed_indices: list[int] = []

        for i, result in enumerate(results):
            if not result.success:
                details.append(f"Action {i} ({result.action.action_type.value}): FAILED — {result.error}")
                failed_indices.append(i)
                continue

            check_passed, check_detail = await self._verify_single(result)
            if check_passed:
                details.append(f"Action {i} ({result.action.action_type.value}): VERIFIED")
            else:
                details.append(f"Action {i} ({result.action.action_type.value}): MISMATCH — {check_detail}")
                failed_indices.append(i)

        passed = len(failed_indices) == 0
        return VerificationResult(
            passed=passed,
            details=details,
            failed_actions=failed_indices,
            rollback_triggered=False,
        )

    async def _verify_single(self, result: ActionResult) -> tuple[bool, str]:
        """Verify a single action result. Returns (passed, detail)."""
        action = result.action

        try:
            if action.action_type == ActionType.FILE_WRITE:
                return await self._verify_file_write(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.FILE_DELETE:
                return await self._verify_file_delete(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.PACKAGE_INSTALL:
                return await self._verify_package_install(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.PACKAGE_REMOVE:
                return await self._verify_package_remove(action.parameters)  # type: ignore[arg-type]

            if action.action_type in (ActionType.SERVICE_START, ActionType.SERVICE_RESTART):
                return await self._verify_service_running(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.SERVICE_STOP:
                return await self._verify_service_stopped(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.GNOME_SETTING_WRITE:
                return await self._verify_gnome_setting(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.DOWNLOAD_FILE:
                return await self._verify_download(result)

            if action.action_type == ActionType.FILE_COPY:
                return await self._verify_file_copy(action.parameters)  # type: ignore[arg-type]

            if action.action_type == ActionType.FILE_MOVE:
                return await self._verify_file_move(action.parameters)  # type: ignore[arg-type]

            # For most actions, success in execution = verified
            # (process_list, clipboard_write, volume_set, etc. are self-verifying)
            return True, "No additional verification needed for this action type"

        except Exception as e:
            logger.warning("Verification check failed: %s", e)
            return False, f"Verification error: {e}"

    async def _verify_file_write(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

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

        if Path(params.path).exists():
            return False, f"File still exists after delete: {params.path}"
        return True, "File successfully deleted"

    async def _verify_file_copy(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.destination:
            return True, "No destination to verify"
        if not Path(params.destination).exists():
            return False, f"Destination does not exist after copy: {params.destination}"
        return True, "Copy destination exists"

    async def _verify_file_move(self, params: FileParams) -> tuple[bool, str]:
        from pathlib import Path

        if not params.destination:
            return True, "No destination to verify"
        if not Path(params.destination).exists():
            return False, f"Destination does not exist after move: {params.destination}"
        if Path(params.path).exists():
            return False, f"Source still exists after move: {params.path}"
        return True, "Move verified: source removed, destination exists"

    async def _verify_package_install(self, params: PackageParams) -> tuple[bool, str]:
        from pilot.system.package_mgr import is_installed

        if await is_installed(params.name):
            return True, f"Package {params.name} is installed"
        return False, f"Package {params.name} is not installed after install"

    async def _verify_package_remove(self, params: PackageParams) -> tuple[bool, str]:
        from pilot.system.package_mgr import is_installed

        if not await is_installed(params.name):
            return True, f"Package {params.name} is removed"
        return False, f"Package {params.name} is still installed after remove"

    async def _verify_service_running(self, params: ServiceParams) -> tuple[bool, str]:
        from pilot.system.systemctl import is_active

        if await is_active(params.name, user_scope=params.user_scope):
            return True, f"Service {params.name} is active"
        return False, f"Service {params.name} is not active"

    async def _verify_service_stopped(self, params: ServiceParams) -> tuple[bool, str]:
        from pilot.system.systemctl import is_active

        if not await is_active(params.name, user_scope=params.user_scope):
            return True, f"Service {params.name} is stopped"
        return False, f"Service {params.name} is still active after stop"

    async def _verify_gnome_setting(self, params: GnomeSettingParams) -> tuple[bool, str]:
        from pilot.system.gnome import get_setting

        if params.value is None:
            return True, "No value to verify for read operation"
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
        if Path(params.output_path).exists():
            size = Path(params.output_path).stat().st_size
            return True, f"File downloaded: {params.output_path} ({size:,} bytes)"
        return False, f"Downloaded file not found: {params.output_path}"
