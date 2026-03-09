"""Action validator — enforces schema compliance and parameter safety.

Every action from the Planner must pass through this validator before
reaching the Executor. Malformed or dangerous actions are rejected entirely.

Updated for the expanded action set with cross-platform support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    DBusParams,
    DownloadParams,
    FileParams,
    GnomeSettingParams,
    NotifyParams,
    OpenApplicationParams,
    OpenUrlParams,
    PackageParams,
    RegistryParams,
    ServiceParams,
    ShellCommandParams,
    ShellScriptParams,
)
from pilot.security.sanitizer import SanitizationError, Sanitizer

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.security.validator")


class ValidationError(Exception):
    def __init__(self, action_index: int, message: str) -> None:
        self.action_index = action_index
        super().__init__(f"Action [{action_index}]: {message}")


# Action types that don't need a target field
NO_TARGET_REQUIRED = {
    ActionType.OPEN_URL,
    ActionType.OPEN_APPLICATION,
    ActionType.NOTIFY,
    ActionType.PACKAGE_UPDATE,
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_KILL,
    ActionType.PROCESS_INFO,
    ActionType.CLIPBOARD_READ,
    ActionType.CLIPBOARD_WRITE,
    ActionType.SYSTEM_INFO,
    ActionType.DISK_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.CPU_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_SLEEP,
    ActionType.POWER_LOCK,
    ActionType.POWER_LOGOUT,
    ActionType.SCHEDULE_CREATE,
    ActionType.SCHEDULE_LIST,
    ActionType.SCHEDULE_DELETE,
    ActionType.ENV_GET,
    ActionType.ENV_SET,
    ActionType.ENV_LIST,
    ActionType.WINDOW_LIST,
    ActionType.WINDOW_FOCUS,
    ActionType.WINDOW_CLOSE,
    ActionType.WINDOW_MINIMIZE,
    ActionType.WINDOW_MAXIMIZE,
    ActionType.VOLUME_GET,
    ActionType.VOLUME_SET,
    ActionType.VOLUME_MUTE,
    ActionType.BRIGHTNESS_GET,
    ActionType.BRIGHTNESS_SET,
    ActionType.SCREENSHOT,
    ActionType.WIFI_LIST,
    ActionType.WIFI_CONNECT,
    ActionType.WIFI_DISCONNECT,
    ActionType.DISK_LIST,
    ActionType.DISK_MOUNT,
    ActionType.DISK_UNMOUNT,
    ActionType.USER_LIST,
    ActionType.USER_INFO,
    ActionType.DOWNLOAD_FILE,
    ActionType.REGISTRY_READ,
    ActionType.REGISTRY_WRITE,
    # Tier 1: Game Changers
    ActionType.MOUSE_CLICK,
    ActionType.MOUSE_DOUBLE_CLICK,
    ActionType.MOUSE_RIGHT_CLICK,
    ActionType.MOUSE_MOVE,
    ActionType.MOUSE_DRAG,
    ActionType.MOUSE_SCROLL,
    ActionType.MOUSE_POSITION,
    ActionType.KEYBOARD_TYPE,
    ActionType.KEYBOARD_PRESS,
    ActionType.KEYBOARD_HOTKEY,
    ActionType.KEYBOARD_HOLD,
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_ELEMENT_MAP,
    ActionType.BROWSER_NAVIGATE,
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_HOVER,
    ActionType.BROWSER_SCROLL,
    ActionType.BROWSER_EXTRACT,
    ActionType.BROWSER_EXTRACT_TABLE,
    ActionType.BROWSER_EXTRACT_LINKS,
    ActionType.BROWSER_EXECUTE_JS,
    ActionType.BROWSER_SCREENSHOT,
    ActionType.BROWSER_FILL_FORM,
    ActionType.BROWSER_NEW_TAB,
    ActionType.BROWSER_CLOSE_TAB,
    ActionType.BROWSER_LIST_TABS,
    ActionType.BROWSER_SWITCH_TAB,
    ActionType.BROWSER_BACK,
    ActionType.BROWSER_FORWARD,
    ActionType.BROWSER_REFRESH,
    ActionType.BROWSER_WAIT,
    ActionType.BROWSER_CLOSE,
    ActionType.BROWSER_PAGE_INFO,
    ActionType.TRIGGER_CREATE,
    ActionType.TRIGGER_LIST,
    ActionType.TRIGGER_DELETE,
    ActionType.TRIGGER_START,
    ActionType.TRIGGER_STOP,
    # Tier 2: Massive Multipliers
    ActionType.CODE_EXECUTE,
    ActionType.CODE_GENERATE_AND_RUN,
    ActionType.FILE_PARSE,
    ActionType.FILE_SEARCH_CONTENT,
    ActionType.API_REQUEST,
    ActionType.API_GITHUB,
    ActionType.API_SEND_EMAIL,
    ActionType.API_WEBHOOK,
    ActionType.API_SLACK,
    ActionType.API_DISCORD,
    ActionType.API_SCRAPE,
}

# File action types
FILE_ACTION_TYPES = {
    ActionType.FILE_READ,
    ActionType.FILE_WRITE,
    ActionType.FILE_DELETE,
    ActionType.FILE_MOVE,
    ActionType.FILE_COPY,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.FILE_PERMISSIONS,
}


class ActionValidator:
    """Validates action plans against security constraints."""

    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._sanitizer = Sanitizer(config)

    def validate_plan(self, plan: ActionPlan) -> list[str]:
        """Validate all actions in a plan. Returns list of error messages (empty if valid)."""
        errors: list[str] = []
        for i, action in enumerate(plan.actions):
            try:
                self.validate_action(action, i)
            except (ValidationError, SanitizationError) as e:
                errors.append(str(e))
        return errors

    def validate_action(self, action: Action, index: int = 0) -> None:
        """Validate a single action. Raises ValidationError on failure."""
        self._validate_target(action, index)
        self._validate_parameters(action, index)
        self._validate_root_requirement(action, index)
        self._validate_restrictions(action, index)

    def _validate_target(self, action: Action, idx: int) -> None:
        if action.action_type not in NO_TARGET_REQUIRED and (not action.target or not action.target.strip()):
            raise ValidationError(idx, "Empty target")

        if action.action_type in FILE_ACTION_TYPES:
            self._sanitizer.validate_path(action.target, idx)

    def _validate_parameters(self, action: Action, idx: int) -> None:
        params = action.parameters

        if isinstance(params, FileParams):
            self._sanitizer.validate_path(params.path, idx)
            if params.destination:
                self._sanitizer.validate_path(params.destination, idx)

        elif isinstance(params, PackageParams):
            self._sanitizer.validate_package_name(params.name, idx)

        elif isinstance(params, ServiceParams):
            self._sanitizer.validate_service_name(params.name, idx)

        elif isinstance(params, GnomeSettingParams):
            self._sanitizer.validate_gsettings_schema(params.schema_id, idx)
            self._sanitizer.validate_gsettings_key(params.key, idx)

        elif isinstance(params, ShellCommandParams):
            self._sanitizer.validate_shell_command(params.command, params.args, idx)

        elif isinstance(params, ShellScriptParams):
            # Scripts get validated more loosely — they're inherently powerful
            if not params.script or not params.script.strip():
                raise ValidationError(idx, "Empty script")

        elif isinstance(params, DBusParams):
            self._sanitizer.validate_dbus_params(params, idx)

        elif isinstance(params, OpenUrlParams):
            self._sanitizer.validate_url(params.url, idx)

        elif isinstance(params, DownloadParams):
            self._sanitizer.validate_url(params.url, idx)
            self._sanitizer.validate_path(params.output_path, idx)

        elif isinstance(params, RegistryParams):
            if not params.key_path:
                raise ValidationError(idx, "Empty registry key_path")

        elif isinstance(params, (OpenApplicationParams, NotifyParams)):
            pass

    def _validate_root_requirement(self, action: Action, idx: int) -> None:
        if action.requires_root and not self._config.security.root_enabled:
            raise ValidationError(
                idx,
                "Action requires root but root access is disabled. Enable it in settings first.",
            )

    def _validate_restrictions(self, action: Action, idx: int) -> None:
        restrictions = self._config.restrictions

        if isinstance(action.parameters, FileParams):
            path_str = action.parameters.path
            for protected in restrictions.protected_folders:
                if path_str.startswith(protected):
                    raise ValidationError(idx, f"Path {path_str} is in protected folder {protected}")

        if isinstance(action.parameters, PackageParams) and action.action_type == ActionType.PACKAGE_REMOVE:
            if action.parameters.name in restrictions.protected_packages:
                raise ValidationError(
                    idx,
                    f"Package '{action.parameters.name}' is protected and cannot be removed",
                )

        if isinstance(action.parameters, ShellCommandParams):
            if action.parameters.command in restrictions.blocked_commands:
                raise ValidationError(idx, f"Command '{action.parameters.command}' is blocked by policy")
