"""Executor agent — executes validated action plans via system interfaces.

The Executor ONLY accepts validated Action objects. It dispatches each action
to the appropriate system interface module based on action_type.

Now cross-platform with 50+ action types covering full system control.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    BrightnessParams,
    BrowserParams,
    ClipboardParams,
    CodeExecParams,
    DBusParams,
    DiskManageParams,
    DownloadParams,
    EnvParams,
    FileIntelParams,
    FileParams,
    GnomeSettingParams,
    KeyboardParams,
    MouseParams,
    NotifyParams,
    OpenApplicationParams,
    OpenUrlParams,
    PackageParams,
    PowerParams,
    ProcessParams,
    RegistryParams,
    ScheduleParams,
    ScreenshotParams,
    ScreenVisionParams,
    ServiceParams,
    ShellCommandParams,
    ShellScriptParams,
    SystemInfoParams,
    TriggerParams,
    VolumeParams,
    WifiParams,
    WindowParams,
)
from pilot.security.audit import AuditLogger
from pilot.security.permissions import PermissionChecker
from pilot.security.validator import ActionValidator
from pilot.system.snapshots import SnapshotManager

if TYPE_CHECKING:
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.agents.executor")


class Executor:
    """Executes validated action plans against the system."""

    def __init__(
        self,
        config: PilotConfig,
        validator: ActionValidator,
        permissions: PermissionChecker,
        audit: AuditLogger,
    ) -> None:
        self._config = config
        self._validator = validator
        self._permissions = permissions
        self._audit = audit
        self._snapshot_mgr = SnapshotManager(config)
        self._last_output: str = ""  # For output chaining between steps
        self._largest_output: str = ""  # Largest output from any step in the pipeline

        self._dispatch_table: dict[ActionType, callable] = {
            # -- File operations --
            ActionType.FILE_READ: self._exec_file_read,
            ActionType.FILE_WRITE: self._exec_file_write,
            ActionType.FILE_DELETE: self._exec_file_delete,
            ActionType.FILE_MOVE: self._exec_file_move,
            ActionType.FILE_COPY: self._exec_file_copy,
            ActionType.FILE_LIST: self._exec_file_list,
            ActionType.FILE_SEARCH: self._exec_file_search,
            ActionType.FILE_PERMISSIONS: self._exec_file_permissions,
            # -- Package operations --
            ActionType.PACKAGE_INSTALL: self._exec_package_install,
            ActionType.PACKAGE_REMOVE: self._exec_package_remove,
            ActionType.PACKAGE_UPDATE: self._exec_package_update,
            ActionType.PACKAGE_SEARCH: self._exec_package_search,
            # -- Service operations --
            ActionType.SERVICE_START: self._exec_service_start,
            ActionType.SERVICE_STOP: self._exec_service_stop,
            ActionType.SERVICE_RESTART: self._exec_service_restart,
            ActionType.SERVICE_ENABLE: self._exec_service_enable,
            ActionType.SERVICE_DISABLE: self._exec_service_disable,
            ActionType.SERVICE_STATUS: self._exec_service_status,
            # -- GNOME settings --
            ActionType.GNOME_SETTING_READ: self._exec_gnome_read,
            ActionType.GNOME_SETTING_WRITE: self._exec_gnome_write,
            # -- DBus --
            ActionType.DBUS_CALL: self._exec_dbus_call,
            # -- Shell commands --
            ActionType.SHELL_COMMAND: self._exec_shell_command,
            ActionType.SHELL_SCRIPT: self._exec_shell_script,
            # -- Open URL / App / Notify --
            ActionType.OPEN_URL: self._exec_open_url,
            ActionType.OPEN_APPLICATION: self._exec_open_application,
            ActionType.NOTIFY: self._exec_notify,
            # -- Process management --
            ActionType.PROCESS_LIST: self._exec_process_list,
            ActionType.PROCESS_KILL: self._exec_process_kill,
            ActionType.PROCESS_INFO: self._exec_process_info,
            # -- Clipboard --
            ActionType.CLIPBOARD_READ: self._exec_clipboard_read,
            ActionType.CLIPBOARD_WRITE: self._exec_clipboard_write,
            # -- System info --
            ActionType.SYSTEM_INFO: self._exec_system_info,
            ActionType.DISK_USAGE: self._exec_disk_usage,
            ActionType.MEMORY_USAGE: self._exec_memory_usage,
            ActionType.CPU_USAGE: self._exec_cpu_usage,
            ActionType.NETWORK_INFO: self._exec_network_info,
            ActionType.BATTERY_INFO: self._exec_battery_info,
            # -- Power management --
            ActionType.POWER_SHUTDOWN: self._exec_power_shutdown,
            ActionType.POWER_RESTART: self._exec_power_restart,
            ActionType.POWER_SLEEP: self._exec_power_sleep,
            ActionType.POWER_LOCK: self._exec_power_lock,
            ActionType.POWER_LOGOUT: self._exec_power_logout,
            # -- Scheduled tasks --
            ActionType.SCHEDULE_CREATE: self._exec_schedule_create,
            ActionType.SCHEDULE_LIST: self._exec_schedule_list,
            ActionType.SCHEDULE_DELETE: self._exec_schedule_delete,
            # -- Environment variables --
            ActionType.ENV_GET: self._exec_env_get,
            ActionType.ENV_SET: self._exec_env_set,
            ActionType.ENV_LIST: self._exec_env_list,
            # -- Window management --
            ActionType.WINDOW_LIST: self._exec_window_list,
            ActionType.WINDOW_FOCUS: self._exec_window_focus,
            ActionType.WINDOW_CLOSE: self._exec_window_close,
            ActionType.WINDOW_MINIMIZE: self._exec_window_minimize,
            ActionType.WINDOW_MAXIMIZE: self._exec_window_maximize,
            # -- Volume / audio --
            ActionType.VOLUME_GET: self._exec_volume_get,
            ActionType.VOLUME_SET: self._exec_volume_set,
            ActionType.VOLUME_MUTE: self._exec_volume_mute,
            # -- Brightness / screen --
            ActionType.BRIGHTNESS_GET: self._exec_brightness_get,
            ActionType.BRIGHTNESS_SET: self._exec_brightness_set,
            ActionType.SCREENSHOT: self._exec_screenshot,
            # -- Network --
            ActionType.WIFI_LIST: self._exec_wifi_list,
            ActionType.WIFI_CONNECT: self._exec_wifi_connect,
            ActionType.WIFI_DISCONNECT: self._exec_wifi_disconnect,
            # -- Disk management --
            ActionType.DISK_LIST: self._exec_disk_list,
            ActionType.DISK_MOUNT: self._exec_disk_mount,
            ActionType.DISK_UNMOUNT: self._exec_disk_unmount,
            # -- User info --
            ActionType.USER_LIST: self._exec_user_list,
            ActionType.USER_INFO: self._exec_user_info,
            # -- Download --
            ActionType.DOWNLOAD_FILE: self._exec_download_file,
            # -- Registry (Windows) --
            ActionType.REGISTRY_READ: self._exec_registry_read,
            ActionType.REGISTRY_WRITE: self._exec_registry_write,
            # ============================================================
            # TIER 1: GAME CHANGERS
            # ============================================================
            # -- Mouse control --
            ActionType.MOUSE_CLICK: self._exec_mouse_click,
            ActionType.MOUSE_DOUBLE_CLICK: self._exec_mouse_double_click,
            ActionType.MOUSE_RIGHT_CLICK: self._exec_mouse_right_click,
            ActionType.MOUSE_MOVE: self._exec_mouse_move,
            ActionType.MOUSE_DRAG: self._exec_mouse_drag,
            ActionType.MOUSE_SCROLL: self._exec_mouse_scroll,
            ActionType.MOUSE_POSITION: self._exec_mouse_position,
            # -- Keyboard control --
            ActionType.KEYBOARD_TYPE: self._exec_keyboard_type,
            ActionType.KEYBOARD_PRESS: self._exec_keyboard_press,
            ActionType.KEYBOARD_HOTKEY: self._exec_keyboard_hotkey,
            ActionType.KEYBOARD_HOLD: self._exec_keyboard_hold,
            # -- Screen understanding / Vision --
            ActionType.SCREEN_OCR: self._exec_screen_ocr,
            ActionType.SCREEN_FIND_TEXT: self._exec_screen_find_text,
            ActionType.SCREEN_ANALYZE: self._exec_screen_analyze,
            ActionType.SCREEN_ELEMENT_MAP: self._exec_screen_element_map,
            # -- Browser automation --
            ActionType.BROWSER_NAVIGATE: self._exec_browser_navigate,
            ActionType.BROWSER_CLICK: self._exec_browser_click,
            ActionType.BROWSER_CLICK_TEXT: self._exec_browser_click_text,
            ActionType.BROWSER_TYPE: self._exec_browser_type,
            ActionType.BROWSER_SELECT: self._exec_browser_select,
            ActionType.BROWSER_HOVER: self._exec_browser_hover,
            ActionType.BROWSER_SCROLL: self._exec_browser_scroll,
            ActionType.BROWSER_EXTRACT: self._exec_browser_extract,
            ActionType.BROWSER_EXTRACT_TABLE: self._exec_browser_extract_table,
            ActionType.BROWSER_EXTRACT_LINKS: self._exec_browser_extract_links,
            ActionType.BROWSER_EXECUTE_JS: self._exec_browser_execute_js,
            ActionType.BROWSER_SCREENSHOT: self._exec_browser_screenshot,
            ActionType.BROWSER_FILL_FORM: self._exec_browser_fill_form,
            ActionType.BROWSER_NEW_TAB: self._exec_browser_new_tab,
            ActionType.BROWSER_CLOSE_TAB: self._exec_browser_close_tab,
            ActionType.BROWSER_LIST_TABS: self._exec_browser_list_tabs,
            ActionType.BROWSER_SWITCH_TAB: self._exec_browser_switch_tab,
            ActionType.BROWSER_BACK: self._exec_browser_back,
            ActionType.BROWSER_FORWARD: self._exec_browser_forward,
            ActionType.BROWSER_REFRESH: self._exec_browser_refresh,
            ActionType.BROWSER_WAIT: self._exec_browser_wait,
            ActionType.BROWSER_CLOSE: self._exec_browser_close,
            ActionType.BROWSER_PAGE_INFO: self._exec_browser_page_info,
            # -- Reactive triggers --
            ActionType.TRIGGER_CREATE: self._exec_trigger_create,
            ActionType.TRIGGER_LIST: self._exec_trigger_list,
            ActionType.TRIGGER_DELETE: self._exec_trigger_delete,
            ActionType.TRIGGER_START: self._exec_trigger_start,
            ActionType.TRIGGER_STOP: self._exec_trigger_stop,
            # ============================================================
            # TIER 2: MASSIVE MULTIPLIERS
            # ============================================================
            # -- Code execution --
            ActionType.CODE_EXECUTE: self._exec_code_execute,
            ActionType.CODE_GENERATE_AND_RUN: self._exec_code_generate,
            # -- File intelligence --
            ActionType.FILE_PARSE: self._exec_file_parse,
            ActionType.FILE_SEARCH_CONTENT: self._exec_file_search_content,
            # -- API integration --
            ActionType.API_REQUEST: self._exec_api_request,
            ActionType.API_GITHUB: self._exec_api_github,
            ActionType.API_SEND_EMAIL: self._exec_api_send_email,
            ActionType.API_WEBHOOK: self._exec_api_webhook,
            ActionType.API_SLACK: self._exec_api_slack,
            ActionType.API_DISCORD: self._exec_api_discord,
            ActionType.API_SCRAPE: self._exec_api_scrape,
        }

    async def execute(self, plan: ActionPlan) -> list[ActionResult]:
        """Execute all actions in a plan sequentially, with output chaining."""
        plan_id = str(uuid.uuid4())[:8]
        results: list[ActionResult] = []
        self._last_output = ""
        self._largest_output = ""

        allowed, reasons = self._permissions.plan_allowed(plan)
        if not allowed:
            return [
                ActionResult(
                    action=plan.actions[0]
                    if plan.actions
                    else Action(
                        action_type=ActionType.FILE_READ,
                        target="",
                        parameters=FileParams(path="/"),
                    ),
                    success=False,
                    error=f"Permission denied: {'; '.join(reasons)}",
                )
            ]

        errors = self._validator.validate_plan(plan)
        if errors:
            # Check if ALL actions are bad — if so, reject
            # Otherwise, filter out the bad actions and continue with the good ones
            bad_indices = set()
            for err in errors:
                # Extract action index from error like "Action [6]: ..."
                import re as _re

                m = _re.search(r"Action \[(\d+)\]", err)
                if m:
                    bad_indices.add(int(m.group(1)))

            if bad_indices and len(bad_indices) < len(plan.actions):
                # Some actions are good — remove bad ones and continue
                good_actions = [a for i, a in enumerate(plan.actions) if i not in bad_indices]
                if good_actions:
                    logger.warning(
                        "Validation: removing %d bad actions (indices %s), keeping %d good ones",
                        len(bad_indices),
                        bad_indices,
                        len(good_actions),
                    )
                    plan.actions = good_actions
                else:
                    return [
                        ActionResult(
                            action=plan.actions[0],
                            success=False,
                            error=f"Validation failed: {'; '.join(errors)}",
                        )
                    ]
            else:
                return [
                    ActionResult(
                        action=plan.actions[0],
                        success=False,
                        error=f"Validation failed: {'; '.join(errors)}",
                    )
                ]

        snapshot_id: str | None = None
        if plan.needs_snapshot and self._config.security.snapshot_on_destructive:
            try:
                snapshot_id = await self._snapshot_mgr.create_snapshot(
                    plan_id, f"Pre-action snapshot for plan {plan_id}"
                )
            except Exception as e:
                logger.warning("Snapshot creation failed: %s", e)

        for i, action in enumerate(plan.actions):
            self._audit.log_action_start(action, plan_id)

            # Auto-inject previous output into action content/code
            action = self._inject_previous_output(action)

            result = await self._execute_single(action, snapshot_id)
            self._audit.log_action_result(result, plan_id)
            results.append(result)

            # Store output for chaining
            if result.success:
                self._last_output = result.output
                # Track the largest output from any step (for data-heavy actions like browser_extract)
                if not hasattr(self, "_largest_output") or len(result.output or "") > len(self._largest_output or ""):
                    self._largest_output = result.output

            if not result.success:
                logger.error(
                    "Action %d failed: %s — stopping plan execution",
                    i,
                    result.error,
                )
                break

        return results

    # Placeholder patterns the LLM might use
    _OUTPUT_PLACEHOLDERS = [
        "{PREV_OUTPUT}",
        "{{PREV_OUTPUT}}",
        "{PREVIOUS_OUTPUT}",
        "{{PREVIOUS_OUTPUT}}",
        "{LAST_OUTPUT}",
        "{{LAST_OUTPUT}}",
        "{OUTPUT}",
        "{{OUTPUT}}",
        "{EXTRACTED_TEXT}",
        "{{EXTRACTED_TEXT}}",
        "{FIRST_3_TEXT}",
        "{{FIRST_3_TEXT}}",
        "{FIRST_3_PARAGRAPHS}",
        "{{FIRST_3_PARAGRAPHS}}",
        "{PARAGRAPHS}",
        "{{PARAGRAPHS}}",
        "{TEXT}",
        "{{TEXT}}",
        "{CONTENT}",
        "{{CONTENT}}",
        "{OCR_TEXT}",
        "{{OCR_TEXT}}",
        "{SCREEN_TEXT}",
        "{{SCREEN_TEXT}}",
    ]

    # Regex to catch ANY {UPPERCASE_PLACEHOLDER} or {{UPPERCASE_PLACEHOLDER}}
    _PLACEHOLDER_REGEX = __import__("re").compile(r"^\{?\{[A-Z][A-Z0-9_]*\}\}?$")

    def _inject_previous_output(self, action: Action) -> Action:
        """Auto-inject previous step output into the action's content/code fields."""
        if not self._last_output:
            return action

        params = action.parameters

        # For file_write: inject into content
        if hasattr(params, "content") and params.content is not None:
            content = params.content
            # Check if content IS a known placeholder
            if content.strip() in self._OUTPUT_PLACEHOLDERS:
                params.content = self._last_output
                action.use_previous_output = True
            # Check if content matches any {UPPERCASE_VAR} pattern
            elif self._PLACEHOLDER_REGEX.match(content.strip()):
                logger.info("Replacing unknown placeholder %r with previous output", content.strip())
                params.content = self._last_output
                action.use_previous_output = True
            else:
                # Replace any placeholder within the content
                for ph in self._OUTPUT_PLACEHOLDERS:
                    if ph in content:
                        params.content = content.replace(ph, self._last_output)
                        action.use_previous_output = True
                        break

        # NOTE: Do NOT replace placeholders in code strings!
        # Code gets PREV_OUTPUT injected as a proper Python variable via _exec_code_execute.
        # String-replacing raw text into code creates syntax errors.

        # If use_previous_output is set but no placeholder was found,
        # inject into content if it's empty
        if action.use_previous_output and hasattr(params, "content") and not params.content:
            params.content = self._last_output

        return action

    async def _execute_single(self, action: Action, snapshot_id: str | None) -> ActionResult:
        """Execute a single validated action."""
        handler = self._dispatch_table.get(action.action_type)
        if handler is None:
            return ActionResult(
                action=action,
                success=False,
                error=f"No handler for action type: {action.action_type}",
            )

        try:
            output = await handler(action)
            return ActionResult(action=action, success=True, output=output, snapshot_id=snapshot_id)
        except Exception as e:
            logger.exception("Action execution failed: %s", action.action_type)
            return ActionResult(action=action, success=False, error=str(e), snapshot_id=snapshot_id)

    # ======================================================================
    # FILE OPERATIONS
    # ======================================================================

    async def _exec_file_read(self, action: Action) -> str:
        from pilot.system.filesystem import file_read

        params: FileParams = action.parameters  # type: ignore[assignment]
        return await file_read(params.path)

    async def _exec_file_write(self, action: Action) -> str:
        from pilot.system.filesystem import file_write

        params: FileParams = action.parameters  # type: ignore[assignment]
        content = params.content or ""

        # Prefer the largest output (e.g., browser_extract data)
        best_output = self._largest_output or self._last_output or ""

        # Replace any placeholder patterns in content
        for ph in self._OUTPUT_PLACEHOLDERS:
            if ph in content:
                content = content.replace(ph, best_output)
                break

        # Catch any remaining {UPPERCASE_VAR} placeholders via regex
        if self._PLACEHOLDER_REGEX.match(content.strip()):
            logger.info("file_write: replacing unknown placeholder %r with best output", content.strip())
            content = best_output

        # If use_previous_output is set, inject the best available output
        if action.use_previous_output and not content:
            content = best_output

        return await file_write(params.path, content)

    async def _exec_file_delete(self, action: Action) -> str:
        from pilot.system.filesystem import file_delete

        params: FileParams = action.parameters  # type: ignore[assignment]
        return await file_delete(params.path, recursive=params.recursive)

    async def _exec_file_move(self, action: Action) -> str:
        from pilot.system.filesystem import file_move

        params: FileParams = action.parameters  # type: ignore[assignment]
        if not params.destination:
            raise ValueError("file_move requires a destination")
        return await file_move(params.path, params.destination)

    async def _exec_file_copy(self, action: Action) -> str:
        from pilot.system.filesystem import file_copy

        params: FileParams = action.parameters  # type: ignore[assignment]
        if not params.destination:
            raise ValueError("file_copy requires a destination")
        return await file_copy(params.path, params.destination, recursive=params.recursive)

    async def _exec_file_list(self, action: Action) -> str:
        from pilot.system.filesystem import file_list

        params: FileParams = action.parameters  # type: ignore[assignment]
        return await file_list(params.path, recursive=params.recursive)

    async def _exec_file_search(self, action: Action) -> str:
        from pilot.system.filesystem import file_search

        params: FileParams = action.parameters  # type: ignore[assignment]
        return await file_search(params.path, params.pattern or "*")

    async def _exec_file_permissions(self, action: Action) -> str:
        from pilot.system.filesystem import file_permissions

        params: FileParams = action.parameters  # type: ignore[assignment]
        return await file_permissions(params.path, params.permissions)

    # ======================================================================
    # PACKAGE OPERATIONS
    # ======================================================================

    async def _exec_package_install(self, action: Action) -> str:
        from pilot.system.package_mgr import package_install

        params: PackageParams = action.parameters  # type: ignore[assignment]
        return await package_install(params.name, params.version)

    async def _exec_package_remove(self, action: Action) -> str:
        from pilot.system.package_mgr import package_remove

        params: PackageParams = action.parameters  # type: ignore[assignment]
        return await package_remove(params.name)

    async def _exec_package_update(self, action: Action) -> str:
        from pilot.system.package_mgr import package_update

        return await package_update()

    async def _exec_package_search(self, action: Action) -> str:
        from pilot.system.package_mgr import package_search

        params: PackageParams = action.parameters  # type: ignore[assignment]
        return await package_search(params.name)

    # ======================================================================
    # SERVICE OPERATIONS
    # ======================================================================

    async def _exec_service_start(self, action: Action) -> str:
        from pilot.system.systemctl import service_start

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_start(params.name, user_scope=params.user_scope)

    async def _exec_service_stop(self, action: Action) -> str:
        from pilot.system.systemctl import service_stop

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_stop(params.name, user_scope=params.user_scope)

    async def _exec_service_restart(self, action: Action) -> str:
        from pilot.system.systemctl import service_restart

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_restart(params.name, user_scope=params.user_scope)

    async def _exec_service_enable(self, action: Action) -> str:
        from pilot.system.systemctl import service_enable

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_enable(params.name, user_scope=params.user_scope)

    async def _exec_service_disable(self, action: Action) -> str:
        from pilot.system.systemctl import service_disable

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_disable(params.name, user_scope=params.user_scope)

    async def _exec_service_status(self, action: Action) -> str:
        from pilot.system.systemctl import service_status

        params: ServiceParams = action.parameters  # type: ignore[assignment]
        return await service_status(params.name, user_scope=params.user_scope)

    # ======================================================================
    # GNOME SETTINGS
    # ======================================================================

    async def _exec_gnome_read(self, action: Action) -> str:
        from pilot.system.gnome import get_setting

        params: GnomeSettingParams = action.parameters  # type: ignore[assignment]
        return await get_setting(params.schema_id, params.key)

    async def _exec_gnome_write(self, action: Action) -> str:
        from pilot.system.gnome import set_setting

        params: GnomeSettingParams = action.parameters  # type: ignore[assignment]
        if not params.value:
            raise ValueError("gnome_setting_write requires a value")
        return await set_setting(params.schema_id, params.key, params.value)

    # ======================================================================
    # DBUS
    # ======================================================================

    async def _exec_dbus_call(self, action: Action) -> str:
        from pilot.system.dbus_client import call_dbus_method

        params: DBusParams = action.parameters  # type: ignore[assignment]
        return await call_dbus_method(
            params.bus,
            params.service,
            params.object_path,
            params.interface,
            params.method,
            params.args or None,
        )

    # ======================================================================
    # SHELL COMMANDS
    # ======================================================================

    async def _exec_shell_command(self, action: Action) -> str:
        params: ShellCommandParams = action.parameters  # type: ignore[assignment]
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        cmd = [params.command] + (params.args or [])
        # On Windows, built-in commands (tree, dir, type) and .com/.bat
        # files need cmd /c to execute properly
        if CURRENT_PLATFORM == Platform.WINDOWS:
            cmd = ["cmd", "/c"] + cmd
        code, out, err = await run_command(
            cmd,
            timeout=params.timeout,
            cwd=params.working_directory,
            root=params.elevated,
        )
        if code != 0:
            raise RuntimeError(f"Command failed (exit {code}): {err.strip()}")
        return out

    async def _exec_shell_script(self, action: Action) -> str:
        params: ShellScriptParams = action.parameters  # type: ignore[assignment]
        from pilot.system.platform_detect import run_shell_script

        code, out, err = await run_shell_script(
            params.script,
            interpreter=params.interpreter,
            timeout=params.timeout,
            cwd=params.working_directory,
            elevated=params.elevated,
        )
        if code != 0:
            raise RuntimeError(f"Script failed (exit {code}): {err.strip()}")
        return out

    # ======================================================================
    # OPEN URL / APPLICATION / NOTIFY
    # ======================================================================

    async def _exec_open_url(self, action: Action) -> str:
        params: OpenUrlParams = action.parameters  # type: ignore[assignment]
        url = params.url
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, _, err = await run_command(["cmd", "/c", "start", "", url])
        elif CURRENT_PLATFORM == Platform.MACOS:
            code, _, err = await run_command(["open", url])
        else:
            code, _, err = await run_command(["xdg-open", url])

        return f"Opened {url} in default browser"

    async def _exec_open_application(self, action: Action) -> str:
        import shutil

        params: OpenApplicationParams = action.parameters  # type: ignore[assignment]
        app_name = params.name
        args = params.args or []

        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, _, err = await run_command(["cmd", "/c", "start", "", app_name, *args])
        else:
            binary = shutil.which(app_name)
            if not binary:
                code, _, err = await run_command(["gtk-launch", app_name])
            else:
                code, _, err = await run_command([binary, *args])

        return f"Launched {app_name}"

    async def _exec_notify(self, action: Action) -> str:
        params: NotifyParams = action.parameters  # type: ignore[assignment]
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, _, err = await run_powershell(
                f"[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
                f"$notify = New-Object System.Windows.Forms.NotifyIcon; "
                f"$notify.Icon = [System.Drawing.SystemIcons]::Information; "
                f"$notify.Visible = $true; "
                f"$notify.ShowBalloonTip(5000, '{params.summary}', '{params.body}', "
                f"[System.Windows.Forms.ToolTipIcon]::Info); "
                f"Start-Sleep -Seconds 3; $notify.Dispose()"
            )
            if code != 0:
                return f"Notification: {params.summary} — {params.body}"
            return f"Notification sent: {params.summary}"
        elif CURRENT_PLATFORM == Platform.MACOS:
            code, _, err = await run_command(
                ["osascript", "-e", f'display notification "{params.body}" with title "{params.summary}"']
            )
        else:
            code, _, err = await run_command(["notify-send", params.summary, params.body or ""])

        return f"Notification sent: {params.summary}"

    # ======================================================================
    # PROCESS MANAGEMENT
    # ======================================================================

    async def _exec_process_list(self, action: Action) -> str:
        from pilot.system.processes import process_list

        params: ProcessParams = action.parameters  # type: ignore[assignment]
        return await process_list(filter_name=params.name)

    async def _exec_process_kill(self, action: Action) -> str:
        from pilot.system.processes import process_kill

        params: ProcessParams = action.parameters  # type: ignore[assignment]
        return await process_kill(pid=params.pid, name=params.name, sig=params.signal)

    async def _exec_process_info(self, action: Action) -> str:
        from pilot.system.processes import process_info

        params: ProcessParams = action.parameters  # type: ignore[assignment]
        if params.pid is None:
            raise ValueError("process_info requires a PID")
        return await process_info(params.pid)

    # ======================================================================
    # CLIPBOARD
    # ======================================================================

    async def _exec_clipboard_read(self, action: Action) -> str:
        from pilot.system.clipboard import clipboard_read

        return await clipboard_read()

    async def _exec_clipboard_write(self, action: Action) -> str:
        from pilot.system.clipboard import clipboard_write

        params: ClipboardParams = action.parameters  # type: ignore[assignment]
        content = params.content or ""
        if action.use_previous_output and self._last_output:
            content = self._last_output
        return await clipboard_write(content)

    # ======================================================================
    # SYSTEM INFO
    # ======================================================================

    async def _exec_system_info(self, action: Action) -> str:
        from pilot.system.sysinfo import system_info

        params: SystemInfoParams = action.parameters  # type: ignore[assignment]
        return await system_info(params.categories)

    async def _exec_disk_usage(self, action: Action) -> str:
        from pilot.system.sysinfo import disk_usage

        return await disk_usage()

    async def _exec_memory_usage(self, action: Action) -> str:
        from pilot.system.sysinfo import memory_usage

        return await memory_usage()

    async def _exec_cpu_usage(self, action: Action) -> str:
        from pilot.system.sysinfo import cpu_usage

        return await cpu_usage()

    async def _exec_network_info(self, action: Action) -> str:
        from pilot.system.sysinfo import network_info

        return await network_info()

    async def _exec_battery_info(self, action: Action) -> str:
        from pilot.system.sysinfo import battery_info

        return await battery_info()

    # ======================================================================
    # POWER MANAGEMENT
    # ======================================================================

    async def _exec_power_shutdown(self, action: Action) -> str:
        from pilot.system.power import shutdown

        params: PowerParams = action.parameters  # type: ignore[assignment]
        return await shutdown(params.delay_seconds, params.force)

    async def _exec_power_restart(self, action: Action) -> str:
        from pilot.system.power import restart

        params: PowerParams = action.parameters  # type: ignore[assignment]
        return await restart(params.delay_seconds, params.force)

    async def _exec_power_sleep(self, action: Action) -> str:
        from pilot.system.power import sleep

        return await sleep()

    async def _exec_power_lock(self, action: Action) -> str:
        from pilot.system.power import lock_screen

        return await lock_screen()

    async def _exec_power_logout(self, action: Action) -> str:
        from pilot.system.power import logout

        return await logout()

    # ======================================================================
    # SCHEDULED TASKS
    # ======================================================================

    async def _exec_schedule_create(self, action: Action) -> str:
        from pilot.system.scheduler import schedule_create

        params: ScheduleParams = action.parameters  # type: ignore[assignment]
        return await schedule_create(params.name, params.command, params.schedule)

    async def _exec_schedule_list(self, action: Action) -> str:
        from pilot.system.scheduler import schedule_list

        return await schedule_list()

    async def _exec_schedule_delete(self, action: Action) -> str:
        from pilot.system.scheduler import schedule_delete

        params: ScheduleParams = action.parameters  # type: ignore[assignment]
        return await schedule_delete(params.name, params.task_id)

    # ======================================================================
    # ENVIRONMENT VARIABLES
    # ======================================================================

    async def _exec_env_get(self, action: Action) -> str:
        from pilot.system.environment import env_get

        params: EnvParams = action.parameters  # type: ignore[assignment]
        return await env_get(params.name)

    async def _exec_env_set(self, action: Action) -> str:
        from pilot.system.environment import env_set

        params: EnvParams = action.parameters  # type: ignore[assignment]
        return await env_set(params.name, params.value or "", params.persistent)

    async def _exec_env_list(self, action: Action) -> str:
        from pilot.system.environment import env_list

        params: EnvParams = action.parameters  # type: ignore[assignment]
        return await env_list(params.name if params.name else None)

    # ======================================================================
    # WINDOW MANAGEMENT
    # ======================================================================

    async def _exec_window_list(self, action: Action) -> str:
        from pilot.system.window_mgr import window_list

        return await window_list()

    async def _exec_window_focus(self, action: Action) -> str:
        from pilot.system.window_mgr import window_focus

        params: WindowParams = action.parameters  # type: ignore[assignment]
        return await window_focus(params.window_id, params.title, params.process_name)

    async def _exec_window_close(self, action: Action) -> str:
        from pilot.system.window_mgr import window_close

        params: WindowParams = action.parameters  # type: ignore[assignment]
        return await window_close(params.window_id, params.title, params.process_name)

    async def _exec_window_minimize(self, action: Action) -> str:
        from pilot.system.window_mgr import window_minimize

        params: WindowParams = action.parameters  # type: ignore[assignment]
        return await window_minimize(params.title, params.process_name)

    async def _exec_window_maximize(self, action: Action) -> str:
        from pilot.system.window_mgr import window_maximize

        params: WindowParams = action.parameters  # type: ignore[assignment]
        return await window_maximize(params.title, params.process_name)

    # ======================================================================
    # VOLUME / AUDIO
    # ======================================================================

    async def _exec_volume_get(self, action: Action) -> str:
        from pilot.system.volume import volume_get

        return await volume_get()

    async def _exec_volume_set(self, action: Action) -> str:
        from pilot.system.volume import volume_set

        params: VolumeParams = action.parameters  # type: ignore[assignment]
        if params.level is None:
            raise ValueError("volume_set requires a level")
        return await volume_set(params.level)

    async def _exec_volume_mute(self, action: Action) -> str:
        from pilot.system.volume import volume_mute

        params: VolumeParams = action.parameters  # type: ignore[assignment]
        return await volume_mute(params.mute if params.mute is not None else True)

    # ======================================================================
    # BRIGHTNESS / SCREEN
    # ======================================================================

    async def _exec_brightness_get(self, action: Action) -> str:
        from pilot.system.screen import brightness_get

        return await brightness_get()

    async def _exec_brightness_set(self, action: Action) -> str:
        from pilot.system.screen import brightness_set

        params: BrightnessParams = action.parameters  # type: ignore[assignment]
        if params.level is None:
            raise ValueError("brightness_set requires a level")
        return await brightness_set(params.level)

    async def _exec_screenshot(self, action: Action) -> str:
        from pilot.system.screen import screenshot

        params: ScreenshotParams = action.parameters  # type: ignore[assignment]
        return await screenshot(params.output_path, params.region)

    # ======================================================================
    # WIFI / NETWORK
    # ======================================================================

    async def _exec_wifi_list(self, action: Action) -> str:
        from pilot.system.network import wifi_list

        return await wifi_list()

    async def _exec_wifi_connect(self, action: Action) -> str:
        from pilot.system.network import wifi_connect

        params: WifiParams = action.parameters  # type: ignore[assignment]
        return await wifi_connect(params.ssid, params.password, params.interface)

    async def _exec_wifi_disconnect(self, action: Action) -> str:
        from pilot.system.network import wifi_disconnect

        params: WifiParams = action.parameters  # type: ignore[assignment]
        return await wifi_disconnect(params.interface)

    # ======================================================================
    # DISK MANAGEMENT
    # ======================================================================

    async def _exec_disk_list(self, action: Action) -> str:
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, out, err = await run_powershell("Get-Disk | Format-Table -AutoSize | Out-String -Width 200")
        else:
            code, out, err = await run_command(["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE"])
        if code != 0:
            raise RuntimeError(f"Disk list failed: {err.strip()}")
        return out.strip()

    async def _exec_disk_mount(self, action: Action) -> str:
        params: DiskManageParams = action.parameters  # type: ignore[assignment]
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        if not params.device or not params.mount_point:
            raise ValueError("disk_mount requires device and mount_point")
        if CURRENT_PLATFORM == Platform.WINDOWS:
            raise RuntimeError("Manual mount not supported on Windows via this interface")
        code, out, err = await run_command(["mount", params.device, params.mount_point], root=True)
        if code != 0:
            raise RuntimeError(f"Mount failed: {err.strip()}")
        return f"Mounted {params.device} at {params.mount_point}"

    async def _exec_disk_unmount(self, action: Action) -> str:
        params: DiskManageParams = action.parameters  # type: ignore[assignment]
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        target = params.mount_point or params.device
        if not target:
            raise ValueError("disk_unmount requires mount_point or device")
        if CURRENT_PLATFORM == Platform.WINDOWS:
            raise RuntimeError("Manual unmount not supported on Windows via this interface")
        code, out, err = await run_command(["umount", target], root=True)
        if code != 0:
            raise RuntimeError(f"Unmount failed: {err.strip()}")
        return f"Unmounted {target}"

    # ======================================================================
    # USER INFO
    # ======================================================================

    async def _exec_user_list(self, action: Action) -> str:
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, out, err = await run_command(["net", "user"])
        else:
            code, out, err = await run_command(["cut", "-d:", "-f1", "/etc/passwd"])
        if code != 0:
            raise RuntimeError(f"User list failed: {err.strip()}")
        return out.strip()

    async def _exec_user_info(self, action: Action) -> str:
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

        if CURRENT_PLATFORM == Platform.WINDOWS:
            code, out, err = await run_command(["whoami", "/all"])
        else:
            code, out, err = await run_command(["id"])
        if code != 0:
            raise RuntimeError(f"User info failed: {err.strip()}")
        return out.strip()

    # ======================================================================
    # DOWNLOAD
    # ======================================================================

    async def _exec_download_file(self, action: Action) -> str:
        from pilot.system.download import download_file

        params: DownloadParams = action.parameters  # type: ignore[assignment]
        return await download_file(params.url, params.output_path, params.overwrite)

    # ======================================================================
    # REGISTRY (WINDOWS)
    # ======================================================================

    async def _exec_registry_read(self, action: Action) -> str:
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

        if CURRENT_PLATFORM != Platform.WINDOWS:
            return "Registry operations are only available on Windows"
        params: RegistryParams = action.parameters  # type: ignore[assignment]
        code, out, err = await run_powershell(
            f"Get-ItemProperty -Path 'Registry::{params.key_path}' "
            + (f"-Name '{params.value_name}'" if params.value_name else "")
            + " | Format-List"
        )
        if code != 0:
            raise RuntimeError(f"Registry read failed: {err.strip()}")
        return out.strip()

    async def _exec_registry_write(self, action: Action) -> str:
        from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

        if CURRENT_PLATFORM != Platform.WINDOWS:
            return "Registry operations are only available on Windows"
        params: RegistryParams = action.parameters  # type: ignore[assignment]
        if not params.value_name or params.value_data is None:
            raise ValueError("registry_write requires value_name and value_data")
        code, out, err = await run_powershell(
            f"Set-ItemProperty -Path 'Registry::{params.key_path}' "
            f"-Name '{params.value_name}' -Value '{params.value_data}' "
            f"-Type {params.value_type}"
        )
        if code != 0:
            raise RuntimeError(f"Registry write failed: {err.strip()}")
        return f"Registry: set {params.key_path}\\{params.value_name} = {params.value_data}"

    # ======================================================================
    # TIER 1: MOUSE CONTROL
    # ======================================================================

    async def _exec_mouse_click(self, action: Action) -> str:
        from pilot.system.input_control import mouse_click

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_click(p.x, p.y, p.button, p.clicks)

    async def _exec_mouse_double_click(self, action: Action) -> str:
        from pilot.system.input_control import mouse_double_click

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_double_click(p.x, p.y)

    async def _exec_mouse_right_click(self, action: Action) -> str:
        from pilot.system.input_control import mouse_right_click

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_right_click(p.x, p.y)

    async def _exec_mouse_move(self, action: Action) -> str:
        from pilot.system.input_control import mouse_move

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_move(p.x, p.y, p.duration, p.relative)

    async def _exec_mouse_drag(self, action: Action) -> str:
        from pilot.system.input_control import mouse_drag

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_drag(p.x, p.y, p.end_x or 0, p.end_y or 0, p.duration, p.button)

    async def _exec_mouse_scroll(self, action: Action) -> str:
        from pilot.system.input_control import mouse_scroll

        p: MouseParams = action.parameters  # type: ignore[assignment]
        return await mouse_scroll(p.amount, p.x if p.x else None, p.y if p.y else None, p.horizontal)

    async def _exec_mouse_position(self, action: Action) -> str:
        from pilot.system.input_control import mouse_position

        return await mouse_position()

    # ======================================================================
    # TIER 1: KEYBOARD CONTROL
    # ======================================================================

    async def _exec_keyboard_type(self, action: Action) -> str:
        from pilot.system.input_control import keyboard_type

        p: KeyboardParams = action.parameters  # type: ignore[assignment]
        return await keyboard_type(p.text, p.interval)

    async def _exec_keyboard_press(self, action: Action) -> str:
        from pilot.system.input_control import keyboard_press

        p: KeyboardParams = action.parameters  # type: ignore[assignment]
        return await keyboard_press(p.key, p.presses)

    async def _exec_keyboard_hotkey(self, action: Action) -> str:
        from pilot.system.input_control import keyboard_hotkey

        p: KeyboardParams = action.parameters  # type: ignore[assignment]
        return await keyboard_hotkey(*p.keys)

    async def _exec_keyboard_hold(self, action: Action) -> str:
        from pilot.system.input_control import keyboard_hold

        p: KeyboardParams = action.parameters  # type: ignore[assignment]
        return await keyboard_hold(p.key, p.duration)

    # ======================================================================
    # TIER 1: SCREEN UNDERSTANDING / VISION
    # ======================================================================

    async def _exec_screen_ocr(self, action: Action) -> str:
        from pilot.system.vision import screen_ocr

        region = None
        language = "eng"
        try:
            p: ScreenVisionParams = action.parameters  # type: ignore[assignment]
            if p.region:
                parts = [int(x) for x in str(p.region).split(",")]
                region = tuple(parts[:4]) if len(parts) >= 4 else None
            language = p.language or "eng"
        except Exception as e:
            logger.warning("screen_ocr: failed to parse params (%s), using defaults", e)
        return await screen_ocr(region, language)

    async def _exec_screen_find_text(self, action: Action) -> str:
        from pilot.system.vision import screen_find_text

        p: ScreenVisionParams = action.parameters  # type: ignore[assignment]
        region = None
        if p.region:
            parts = [int(x) for x in p.region.split(",")]
            region = tuple(parts[:4]) if len(parts) >= 4 else None
        return await screen_find_text(p.target_text, region)

    async def _exec_screen_analyze(self, action: Action) -> str:
        from pilot.system.vision import screen_analyze

        p: ScreenVisionParams = action.parameters  # type: ignore[assignment]
        return await screen_analyze(p.prompt)

    async def _exec_screen_element_map(self, action: Action) -> str:
        from pilot.system.vision import screen_element_map

        return await screen_element_map()

    # ======================================================================
    # TIER 1: BROWSER AUTOMATION
    # ======================================================================

    async def _exec_browser_navigate(self, action: Action) -> str:
        from pilot.system.browser import browser_navigate

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_navigate(p.url, p.wait_until)

    async def _exec_browser_click(self, action: Action) -> str:
        from pilot.system.browser import browser_click

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_click(p.selector, p.button, timeout=p.timeout)

    async def _exec_browser_click_text(self, action: Action) -> str:
        from pilot.system.browser import browser_click_text

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_click_text(p.text, p.exact)

    async def _exec_browser_type(self, action: Action) -> str:
        from pilot.system.browser import browser_type

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_type(p.selector, p.text, p.clear_first, p.press_enter)

    async def _exec_browser_select(self, action: Action) -> str:
        from pilot.system.browser import browser_select

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_select(p.selector, p.value)

    async def _exec_browser_hover(self, action: Action) -> str:
        from pilot.system.browser import browser_hover

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_hover(p.selector)

    async def _exec_browser_scroll(self, action: Action) -> str:
        from pilot.system.browser import browser_scroll

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_scroll(p.direction, p.amount)

    async def _exec_browser_extract(self, action: Action) -> str:
        from pilot.system.browser import browser_extract

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_extract(p.selector or "body", p.attribute, p.multiple)

    async def _exec_browser_extract_table(self, action: Action) -> str:
        from pilot.system.browser import browser_extract_table

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_extract_table(p.selector or "table")

    async def _exec_browser_extract_links(self, action: Action) -> str:
        from pilot.system.browser import browser_extract_links

        return await browser_extract_links()

    async def _exec_browser_execute_js(self, action: Action) -> str:
        from pilot.system.browser import browser_execute_js

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_execute_js(p.script)

    async def _exec_browser_screenshot(self, action: Action) -> str:
        from pilot.system.browser import browser_screenshot

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_screenshot(p.output_path, p.full_page, p.selector or None)

    async def _exec_browser_fill_form(self, action: Action) -> str:
        from pilot.system.browser import browser_fill_form

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_fill_form(p.fields, p.submit_selector)

    async def _exec_browser_new_tab(self, action: Action) -> str:
        from pilot.system.browser import browser_new_tab

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_new_tab(p.url or None)

    async def _exec_browser_close_tab(self, action: Action) -> str:
        from pilot.system.browser import browser_close_tab

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_close_tab(p.tab_index)

    async def _exec_browser_list_tabs(self, action: Action) -> str:
        from pilot.system.browser import browser_list_tabs

        return await browser_list_tabs()

    async def _exec_browser_switch_tab(self, action: Action) -> str:
        from pilot.system.browser import browser_switch_tab

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_switch_tab(p.tab_index)

    async def _exec_browser_back(self, action: Action) -> str:
        from pilot.system.browser import browser_back

        return await browser_back()

    async def _exec_browser_forward(self, action: Action) -> str:
        from pilot.system.browser import browser_forward

        return await browser_forward()

    async def _exec_browser_refresh(self, action: Action) -> str:
        from pilot.system.browser import browser_refresh

        return await browser_refresh()

    async def _exec_browser_wait(self, action: Action) -> str:
        from pilot.system.browser import browser_wait

        p: BrowserParams = action.parameters  # type: ignore[assignment]
        return await browser_wait(p.selector or None, p.timeout, p.state)

    async def _exec_browser_close(self, action: Action) -> str:
        from pilot.system.browser import browser_close

        return await browser_close()

    async def _exec_browser_page_info(self, action: Action) -> str:
        from pilot.system.browser import browser_get_page_info

        return await browser_get_page_info()

    # ======================================================================
    # TIER 1: REACTIVE TRIGGERS
    # ======================================================================

    async def _exec_trigger_create(self, action: Action) -> str:
        from pilot.system.triggers import trigger_create

        p: TriggerParams = action.parameters  # type: ignore[assignment]
        return await trigger_create(
            p.name,
            p.trigger_type,
            p.condition,
            p.action_command,
            p.max_fires,
            p.cooldown_seconds,
        )

    async def _exec_trigger_list(self, action: Action) -> str:
        from pilot.system.triggers import trigger_list

        return await trigger_list()

    async def _exec_trigger_delete(self, action: Action) -> str:
        from pilot.system.triggers import trigger_delete

        p: TriggerParams = action.parameters  # type: ignore[assignment]
        return await trigger_delete(p.trigger_id or p.name)

    async def _exec_trigger_start(self, action: Action) -> str:
        from pilot.system.triggers import trigger_start_engine

        return await trigger_start_engine()

    async def _exec_trigger_stop(self, action: Action) -> str:
        from pilot.system.triggers import trigger_stop_engine

        return await trigger_stop_engine()

    # ======================================================================
    # TIER 2: CODE EXECUTION
    # ======================================================================

    async def _exec_code_execute(self, action: Action) -> str:
        import tempfile

        from pilot.system.code_exec import execute_code

        p: CodeExecParams = action.parameters  # type: ignore[assignment]
        code = p.code

        # If there's previous output available, inject it as Python variables
        if p.language.lower().strip() in ("python", "py", "python3"):
            best_text = self._largest_output or self._last_output or ""

            if best_text:
                # Normalize line endings
                normalized = best_text.replace("\r\n", "\n").replace("\r", "\n")

                # Write in binary mode to avoid Windows \r issues
                data_file = tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False, prefix="pilot_data_")
                data_file.write(normalized.encode("utf-8"))
                data_file.close()

                # Build preamble using helper module
                from pilot.agents._code_preamble import build_preamble

                preamble = build_preamble(data_file.name)
                code = preamble + code

                para_count = len([p for p in normalized.split("\n\n") if p.strip() and len(p.strip()) > 30])
                logger.info(
                    "Code execute: injected PREV_OUTPUT (%d chars), ~%d paragraphs", len(normalized), para_count
                )
                logger.debug("LLM generated code:\n%s", p.code[:500] if p.code else "(empty)")

            # --- Sanitize code before execution ---
            from pilot.agents.code_sanitizer import sanitize_python_code

            code = sanitize_python_code(code)

        logger.info("Code execute: running %d chars of code", len(code))
        result = await execute_code(code, p.language, p.timeout)
        logger.info("Code execute result (%d chars): %s", len(result), result[:200] if result else "(empty)")

        # --- Auto-retry on failure: ask LLM to fix the code ---
        if result and ("[STDERR]" in result or "[EXIT CODE:" in result):
            # Code failed — try to fix it with the LLM
            logger.warning("Code execution failed, attempting auto-fix via LLM")
            try:
                fix_prompt = (
                    "The following Python code crashed with this error:\n\n"
                    "```\n" + result[:500] + "\n```\n\n"
                    "Here is the code:\n\n"
                    "```python\n" + code[-2000:] + "\n```\n\n"
                    "Fix the code. Return ONLY the complete fixed Python code, no explanation. "
                    "Do NOT use self.method() calls. This is a standalone script. "
                    "Make sure all variables are defined. Use simple, safe code."
                )
                fixed_code = await self._model.generate(
                    fix_prompt,
                    system="You are a Python code fixer. Return ONLY valid Python code. No markdown, no explanation.",
                    temperature=0.0,
                )
                # Strip markdown code fences if present
                fixed_code = fixed_code.strip()
                if fixed_code.startswith("```"):
                    lines = fixed_code.split("\n")
                    # Remove first and last lines (```python and ```)
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    fixed_code = "\n".join(lines)

                # Re-sanitize the fixed code
                if p.language.lower().strip() in ("python", "py", "python3"):
                    from pilot.agents.code_sanitizer import sanitize_python_code

                    fixed_code = sanitize_python_code(fixed_code)

                logger.info("Auto-fix: retrying with LLM-fixed code (%d chars)", len(fixed_code))
                retry_result = await execute_code(fixed_code, p.language, p.timeout)

                # Only use the retry if it's better (no error)
                if "[STDERR]" not in retry_result and "[EXIT CODE:" not in retry_result:
                    logger.info("Auto-fix succeeded!")
                    return retry_result
                elif len(retry_result) > len(result):
                    # Partial improvement — use it if it produced more output
                    logger.info("Auto-fix partially improved output")
                    return retry_result
                else:
                    logger.warning("Auto-fix didn't improve, keeping original result")
            except Exception as e:
                logger.warning("Auto-fix LLM call failed: %s", e)

        return result

    async def _exec_code_generate(self, action: Action) -> str:
        from pilot.system.code_exec import generate_and_execute

        p: CodeExecParams = action.parameters  # type: ignore[assignment]
        return await generate_and_execute(p.task_description, p.language, p.timeout)

    # ======================================================================
    # TIER 2: FILE CONTENT INTELLIGENCE
    # ======================================================================

    async def _exec_file_parse(self, action: Action) -> str:
        from pilot.system.file_intel import parse_file

        p: FileIntelParams = action.parameters  # type: ignore[assignment]
        return await parse_file(p.path)

    async def _exec_file_search_content(self, action: Action) -> str:
        from pilot.system.file_intel import search_file_contents

        p: FileIntelParams = action.parameters  # type: ignore[assignment]
        return await search_file_contents(p.directory, p.search_text, p.pattern, p.max_results)

    # ======================================================================
    # TIER 2: API INTEGRATION
    # ======================================================================

    async def _exec_api_request(self, action: Action) -> str:
        from pilot.system.api_client import api_request

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await api_request(p.method, p.url, p.headers or None, p.body, p.params or None, timeout=p.timeout)

    async def _exec_api_github(self, action: Action) -> str:
        from pilot.system.api_client import github_api

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await github_api(p.endpoint, p.method, p.body, p.token)

    async def _exec_api_send_email(self, action: Action) -> str:
        from pilot.system.api_client import send_email

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await send_email(p.to, p.subject, p.message, html=p.html)

    async def _exec_api_webhook(self, action: Action) -> str:
        from pilot.system.api_client import send_webhook

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await send_webhook(p.url, p.body or {}, p.method, p.headers or None)

    async def _exec_api_slack(self, action: Action) -> str:
        from pilot.system.api_client import send_slack_message

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await send_slack_message(p.message, p.webhook_url, p.channel)

    async def _exec_api_discord(self, action: Action) -> str:
        from pilot.system.api_client import send_discord_message

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await send_discord_message(p.message, p.webhook_url)

    async def _exec_api_scrape(self, action: Action) -> str:
        from pilot.system.api_client import scrape_url

        p: ApiRequestParams = action.parameters  # type: ignore[assignment]
        return await scrape_url(p.url, p.selector, p.extract)
