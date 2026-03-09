"""Pydantic models for structured action plans.

Every LLM output is parsed into these models. The Executor only accepts
validated Action objects — there is no path from raw LLM text to system calls.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    # -- File operations --
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_MOVE = "file_move"
    FILE_COPY = "file_copy"
    FILE_LIST = "file_list"
    FILE_SEARCH = "file_search"
    FILE_PERMISSIONS = "file_permissions"

    # -- Package management --
    PACKAGE_INSTALL = "package_install"
    PACKAGE_REMOVE = "package_remove"
    PACKAGE_UPDATE = "package_update"
    PACKAGE_SEARCH = "package_search"

    # -- Service management --
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    SERVICE_RESTART = "service_restart"
    SERVICE_ENABLE = "service_enable"
    SERVICE_DISABLE = "service_disable"
    SERVICE_STATUS = "service_status"

    # -- GNOME/Desktop settings --
    GNOME_SETTING_READ = "gnome_setting_read"
    GNOME_SETTING_WRITE = "gnome_setting_write"

    # -- DBus --
    DBUS_CALL = "dbus_call"

    # -- Shell / command execution --
    SHELL_COMMAND = "shell_command"
    SHELL_SCRIPT = "shell_script"  # Run a multi-line script

    # -- Open URL / Application / Notify (original) --
    OPEN_URL = "open_url"
    OPEN_APPLICATION = "open_application"
    NOTIFY = "notify"

    # -- Process management --
    PROCESS_LIST = "process_list"
    PROCESS_KILL = "process_kill"
    PROCESS_INFO = "process_info"

    # -- Clipboard --
    CLIPBOARD_READ = "clipboard_read"
    CLIPBOARD_WRITE = "clipboard_write"

    # -- System info & monitoring --
    SYSTEM_INFO = "system_info"
    DISK_USAGE = "disk_usage"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_INFO = "network_info"
    BATTERY_INFO = "battery_info"

    # -- Power management --
    POWER_SHUTDOWN = "power_shutdown"
    POWER_RESTART = "power_restart"
    POWER_SLEEP = "power_sleep"
    POWER_LOCK = "power_lock"
    POWER_LOGOUT = "power_logout"

    # -- Scheduled tasks --
    SCHEDULE_CREATE = "schedule_create"
    SCHEDULE_LIST = "schedule_list"
    SCHEDULE_DELETE = "schedule_delete"

    # -- Environment variables --
    ENV_GET = "env_get"
    ENV_SET = "env_set"
    ENV_LIST = "env_list"

    # -- Window management --
    WINDOW_LIST = "window_list"
    WINDOW_FOCUS = "window_focus"
    WINDOW_CLOSE = "window_close"
    WINDOW_MINIMIZE = "window_minimize"
    WINDOW_MAXIMIZE = "window_maximize"

    # -- Volume / audio --
    VOLUME_GET = "volume_get"
    VOLUME_SET = "volume_set"
    VOLUME_MUTE = "volume_mute"

    # -- Display / screen --
    BRIGHTNESS_GET = "brightness_get"
    BRIGHTNESS_SET = "brightness_set"
    SCREENSHOT = "screenshot"

    # -- Network management --
    WIFI_LIST = "wifi_list"
    WIFI_CONNECT = "wifi_connect"
    WIFI_DISCONNECT = "wifi_disconnect"

    # -- Disk management --
    DISK_LIST = "disk_list"
    DISK_MOUNT = "disk_mount"
    DISK_UNMOUNT = "disk_unmount"

    # -- User / group management --
    USER_LIST = "user_list"
    USER_INFO = "user_info"

    # -- Download --
    DOWNLOAD_FILE = "download_file"

    # -- Registry (Windows) --
    REGISTRY_READ = "registry_read"
    REGISTRY_WRITE = "registry_write"

    # ============================================================
    # TIER 1: GAME CHANGERS
    # ============================================================

    # -- Mouse control --
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    MOUSE_POSITION = "mouse_position"

    # -- Keyboard control --
    KEYBOARD_TYPE = "keyboard_type"
    KEYBOARD_PRESS = "keyboard_press"
    KEYBOARD_HOTKEY = "keyboard_hotkey"
    KEYBOARD_HOLD = "keyboard_hold"

    # -- Screen understanding / Vision --
    SCREEN_OCR = "screen_ocr"
    SCREEN_FIND_TEXT = "screen_find_text"
    SCREEN_ANALYZE = "screen_analyze"
    SCREEN_ELEMENT_MAP = "screen_element_map"

    # -- Browser automation --
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_CLICK = "browser_click"
    BROWSER_CLICK_TEXT = "browser_click_text"
    BROWSER_TYPE = "browser_type"
    BROWSER_SELECT = "browser_select"
    BROWSER_HOVER = "browser_hover"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_EXTRACT = "browser_extract"
    BROWSER_EXTRACT_TABLE = "browser_extract_table"
    BROWSER_EXTRACT_LINKS = "browser_extract_links"
    BROWSER_EXECUTE_JS = "browser_execute_js"
    BROWSER_SCREENSHOT = "browser_screenshot"
    BROWSER_FILL_FORM = "browser_fill_form"
    BROWSER_NEW_TAB = "browser_new_tab"
    BROWSER_CLOSE_TAB = "browser_close_tab"
    BROWSER_LIST_TABS = "browser_list_tabs"
    BROWSER_SWITCH_TAB = "browser_switch_tab"
    BROWSER_BACK = "browser_back"
    BROWSER_FORWARD = "browser_forward"
    BROWSER_REFRESH = "browser_refresh"
    BROWSER_WAIT = "browser_wait"
    BROWSER_CLOSE = "browser_close"
    BROWSER_PAGE_INFO = "browser_page_info"

    # -- Reactive triggers --
    TRIGGER_CREATE = "trigger_create"
    TRIGGER_LIST = "trigger_list"
    TRIGGER_DELETE = "trigger_delete"
    TRIGGER_START = "trigger_start"
    TRIGGER_STOP = "trigger_stop"

    # ============================================================
    # TIER 2: MASSIVE MULTIPLIERS
    # ============================================================

    # -- Code generation & execution --
    CODE_EXECUTE = "code_execute"
    CODE_GENERATE_AND_RUN = "code_generate_and_run"

    # -- File content intelligence --
    FILE_PARSE = "file_parse"
    FILE_SEARCH_CONTENT = "file_search_content"

    # -- API integration --
    API_REQUEST = "api_request"
    API_GITHUB = "api_github"
    API_SEND_EMAIL = "api_send_email"
    API_WEBHOOK = "api_webhook"
    API_SLACK = "api_slack"
    API_DISCORD = "api_discord"
    API_SCRAPE = "api_scrape"


class PermissionTier(int, Enum):
    READ_ONLY = 0
    USER_WRITE = 1
    SYSTEM_MODIFY = 2
    DESTRUCTIVE = 3
    ROOT_CRITICAL = 4


READ_ONLY_ACTIONS = {
    ActionType.FILE_READ,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.PACKAGE_SEARCH,
    ActionType.SERVICE_STATUS,
    ActionType.GNOME_SETTING_READ,
    ActionType.OPEN_URL,
    ActionType.OPEN_APPLICATION,
    ActionType.NOTIFY,
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_INFO,
    ActionType.CLIPBOARD_READ,
    ActionType.SYSTEM_INFO,
    ActionType.DISK_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.CPU_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
    ActionType.ENV_GET,
    ActionType.ENV_LIST,
    ActionType.WINDOW_LIST,
    ActionType.VOLUME_GET,
    ActionType.BRIGHTNESS_GET,
    ActionType.SCREENSHOT,
    ActionType.WIFI_LIST,
    ActionType.DISK_LIST,
    ActionType.USER_LIST,
    ActionType.USER_INFO,
    ActionType.SCHEDULE_LIST,
    ActionType.REGISTRY_READ,
    # Tier 1 read-only
    ActionType.MOUSE_POSITION,
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_ELEMENT_MAP,
    ActionType.BROWSER_EXTRACT,
    ActionType.BROWSER_EXTRACT_TABLE,
    ActionType.BROWSER_EXTRACT_LINKS,
    ActionType.BROWSER_SCREENSHOT,
    ActionType.BROWSER_LIST_TABS,
    ActionType.BROWSER_PAGE_INFO,
    ActionType.TRIGGER_LIST,
    # Tier 2 read-only
    ActionType.FILE_PARSE,
    ActionType.FILE_SEARCH_CONTENT,
    ActionType.API_SCRAPE,
}

DESTRUCTIVE_ACTIONS = {
    ActionType.FILE_DELETE,
    ActionType.PACKAGE_REMOVE,
    ActionType.PROCESS_KILL,
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_LOGOUT,
    ActionType.SCHEDULE_DELETE,
    ActionType.DISK_UNMOUNT,
    ActionType.WINDOW_CLOSE,
}

SYSTEM_MODIFY_ACTIONS = {
    ActionType.PACKAGE_INSTALL,
    ActionType.PACKAGE_UPDATE,
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.SERVICE_RESTART,
    ActionType.SERVICE_ENABLE,
    ActionType.SERVICE_DISABLE,
    ActionType.GNOME_SETTING_WRITE,
    ActionType.SHELL_SCRIPT,
    ActionType.SCHEDULE_CREATE,
    ActionType.FILE_PERMISSIONS,
    ActionType.WIFI_CONNECT,
    ActionType.WIFI_DISCONNECT,
    ActionType.DISK_MOUNT,
    ActionType.REGISTRY_WRITE,
    # API actions that send data externally
    ActionType.API_SEND_EMAIL,
    ActionType.API_WEBHOOK,
    ActionType.API_SLACK,
    ActionType.API_DISCORD,
}


# -- Parameter models --


class FileParams(BaseModel):
    path: str = ""
    content: str | None = None
    destination: str | None = None
    recursive: bool = False
    pattern: str | None = None  # For file_search
    permissions: str | None = None  # e.g. "755" for file_permissions


class PackageParams(BaseModel):
    name: str = ""
    version: str | None = None
    repository: str | None = None


class ServiceParams(BaseModel):
    name: str = ""
    user_scope: bool = False


class GnomeSettingParams(BaseModel):
    schema_id: str = Field(default="", alias="schema")
    key: str = ""
    value: str | None = None


class DBusParams(BaseModel):
    bus: Literal["system", "session"] = "session"
    service: str = ""
    object_path: str = ""
    interface: str = ""
    method: str = ""
    args: list[str] = Field(default_factory=list)


class ShellCommandParams(BaseModel):
    command: str = ""
    args: list[str] = Field(default_factory=list)
    working_directory: str | None = None
    timeout: int = 60  # seconds (Windows commands can be slow)
    elevated: bool = False  # run as admin/root


class ShellScriptParams(BaseModel):
    """Run a multi-line script (bash/powershell)."""

    script: str = ""
    interpreter: str | None = None  # auto-detect: bash (linux), powershell (win)
    working_directory: str | None = None
    timeout: int = 60
    elevated: bool = False


class OpenUrlParams(BaseModel):
    url: str = ""


class OpenApplicationParams(BaseModel):
    name: str = ""
    args: list[str] = Field(default_factory=list)


class NotifyParams(BaseModel):
    summary: str = "Pilot notification"
    body: str = ""


class ProcessParams(BaseModel):
    """For process_list, process_kill, process_info."""

    pid: int | None = None  # for kill/info
    name: str | None = None  # for kill by name
    signal: str = "SIGTERM"  # for kill


class ClipboardParams(BaseModel):
    """For clipboard_read / clipboard_write."""

    content: str | None = None  # for write


class SystemInfoParams(BaseModel):
    """What system info to gather."""

    categories: list[str] = Field(default_factory=lambda: ["os", "cpu", "memory", "disk", "network"])


class PowerParams(BaseModel):
    """For power management actions."""

    delay_seconds: int = 0
    force: bool = False


class ScheduleParams(BaseModel):
    """For scheduled task management."""

    name: str = ""
    command: str = ""
    schedule: str = ""  # cron expression or Windows task schedule
    task_id: str | None = None  # for delete


class EnvParams(BaseModel):
    """For environment variable operations."""

    name: str = ""
    value: str | None = None
    persistent: bool = False  # write to profile


class WindowParams(BaseModel):
    """For window management."""

    window_id: str | None = None
    title: str | None = None  # match by title substring
    process_name: str | None = None


class VolumeParams(BaseModel):
    """For audio volume control."""

    level: int | None = None  # 0-100
    mute: bool | None = None


class BrightnessParams(BaseModel):
    """For display brightness."""

    level: int | None = None  # 0-100


class ScreenshotParams(BaseModel):
    """For taking screenshots."""

    output_path: str | None = None
    region: str | None = None  # "x,y,w,h" or "fullscreen" or "active_window"


class WifiParams(BaseModel):
    """For WiFi management."""

    ssid: str = ""
    password: str | None = None
    interface: str | None = None


class DiskManageParams(BaseModel):
    """For disk mount/unmount/list."""

    device: str | None = None
    mount_point: str | None = None


class DownloadParams(BaseModel):
    """Download a file from a URL."""

    url: str = ""
    output_path: str = ""
    overwrite: bool = False


class RegistryParams(BaseModel):
    """Windows registry operations."""

    key_path: str = ""  # e.g. HKCU\\Software\\...
    value_name: str = ""
    value_data: str | None = None
    value_type: str = "REG_SZ"


# ======= TIER 1: GAME CHANGER PARAMS =======


class MouseParams(BaseModel):
    """For mouse control actions."""

    x: int = 0
    y: int = 0
    button: str = "left"  # left, right, middle
    clicks: int = 1
    end_x: int | None = None  # For drag
    end_y: int | None = None  # For drag
    duration: float = 0.3
    amount: int = 3  # For scroll
    relative: bool = False
    horizontal: bool = False  # For horizontal scroll


class KeyboardParams(BaseModel):
    """For keyboard control actions."""

    text: str = ""  # For type
    key: str = ""  # For press (enter, tab, escape, f1...)
    keys: list[str] = Field(default_factory=list)  # For hotkey (ctrl, c)
    interval: float = 0.03  # Typing speed
    presses: int = 1
    duration: float = 0.5  # For hold


class ScreenVisionParams(BaseModel):
    """For OCR, text finding, screen analysis."""

    target_text: str = ""  # For find_text
    prompt: str = "Describe what you see on the screen"  # For analysis
    region: str | None = None  # "x,y,w,h" or None for fullscreen
    language: str = "eng"


class BrowserParams(BaseModel):
    """For browser automation actions."""

    url: str = ""
    selector: str = ""  # CSS selector
    text: str = ""  # For typing, clicking by text
    value: str = ""  # For select, attribute extraction
    script: str = ""  # For JS execution
    fields: dict[str, str] = Field(default_factory=dict)  # For form fill
    submit_selector: str | None = None  # For form submit
    full_page: bool = False  # For screenshot
    output_path: str | None = None  # For screenshot
    clear_first: bool = True  # For type
    press_enter: bool = False  # For type
    direction: str = "down"  # For scroll
    amount: int = 500  # For scroll px
    tab_index: int = -1  # For tab operations
    multiple: bool = False  # For extract
    attribute: str = "innerText"  # For extract
    timeout: int = 10000
    wait_until: str = "domcontentloaded"
    state: str = "visible"  # For wait
    exact: bool = False  # For click_text
    button: str = "left"  # For click


class TriggerParams(BaseModel):
    """For reactive trigger management."""

    name: str = ""
    trigger_type: str = ""  # cpu_threshold, file_created, etc.
    condition: dict = Field(default_factory=dict)  # Type-specific conditions
    action_command: str = ""  # What to do when triggered
    trigger_id: str = ""  # For delete
    max_fires: int = 0  # 0 = unlimited
    cooldown_seconds: int = 60


# ======= TIER 2: MULTIPLIER PARAMS =======


class CodeExecParams(BaseModel):
    """For code generation and execution."""

    code: str = ""
    language: str = "python"  # python, powershell, bash, cmd, javascript
    timeout: int = 30
    task_description: str = ""  # For generate_and_run (LLM writes the code)


class FileIntelParams(BaseModel):
    """For file content intelligence."""

    path: str = ""
    search_text: str = ""
    directory: str = ""
    pattern: str = "*"  # For search_content glob
    max_results: int = 50


class ApiRequestParams(BaseModel):
    """For generic API and service calls."""

    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict | None = None
    params: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    # GitHub specific
    endpoint: str = ""  # e.g. /user/repos
    token: str | None = None
    # Email specific
    to: str = ""
    subject: str = ""
    message: str = ""
    html: bool = False
    # Slack/Discord
    webhook_url: str | None = None
    channel: str | None = None
    # Scrape
    selector: str | None = None
    extract: str = "text"  # text, html, links, tables
    # GitHub issue
    owner: str = ""
    repo: str = ""
    title: str = ""
    labels: list[str] = Field(default_factory=list)


class EmptyParams(BaseModel):
    """For actions that need no parameters."""

    pass


# Union of all parameter types
ActionParameters = (
    FileParams
    | PackageParams
    | ServiceParams
    | GnomeSettingParams
    | DBusParams
    | ShellCommandParams
    | ShellScriptParams
    | OpenUrlParams
    | OpenApplicationParams
    | NotifyParams
    | ProcessParams
    | ClipboardParams
    | SystemInfoParams
    | PowerParams
    | ScheduleParams
    | EnvParams
    | WindowParams
    | VolumeParams
    | BrightnessParams
    | ScreenshotParams
    | WifiParams
    | DiskManageParams
    | DownloadParams
    | RegistryParams
    | MouseParams
    | KeyboardParams
    | ScreenVisionParams
    | BrowserParams
    | TriggerParams
    | CodeExecParams
    | FileIntelParams
    | ApiRequestParams
    | EmptyParams
)


class Action(BaseModel):
    """A single validated system action."""

    action_type: ActionType
    target: str = ""
    parameters: ActionParameters
    requires_root: bool = False
    destructive: bool = False
    reversible: bool = True
    rollback_action: Action | None = None
    use_previous_output: bool = False  # Inject previous step's output into this action

    @property
    def permission_tier(self) -> PermissionTier:
        # These actions are ALWAYS safe — never require confirmation
        ALWAYS_SAFE = {
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
            ActionType.FILE_LIST,
            ActionType.FILE_SEARCH,
            ActionType.FILE_COPY,
            ActionType.CODE_EXECUTE,
            ActionType.CODE_GENERATE_AND_RUN,
            ActionType.SHELL_COMMAND,
            ActionType.BROWSER_NAVIGATE,
            ActionType.BROWSER_EXTRACT,
            ActionType.BROWSER_EXTRACT_TABLE,
            ActionType.BROWSER_EXTRACT_LINKS,
            ActionType.BROWSER_CLICK,
            ActionType.BROWSER_CLICK_TEXT,
            ActionType.BROWSER_TYPE,
            ActionType.BROWSER_SELECT,
            ActionType.BROWSER_HOVER,
            ActionType.BROWSER_SCROLL,
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
            ActionType.OPEN_URL,
            ActionType.OPEN_APPLICATION,
            ActionType.NOTIFY,
            ActionType.CLIPBOARD_READ,
            ActionType.CLIPBOARD_WRITE,
            ActionType.PROCESS_LIST,
            ActionType.PROCESS_INFO,
            ActionType.SYSTEM_INFO,
            ActionType.DISK_USAGE,
            ActionType.MEMORY_USAGE,
            ActionType.SCREENSHOT,
            ActionType.API_REQUEST,
            ActionType.API_SCRAPE,
            ActionType.DOWNLOAD_FILE,
        }
        if self.action_type in ALWAYS_SAFE:
            return PermissionTier.USER_WRITE
        if self.action_type in READ_ONLY_ACTIONS:
            return PermissionTier.READ_ONLY
        if self.requires_root:
            return PermissionTier.ROOT_CRITICAL
        if self.action_type in DESTRUCTIVE_ACTIONS:
            return PermissionTier.DESTRUCTIVE
        if self.action_type in SYSTEM_MODIFY_ACTIONS:
            return PermissionTier.SYSTEM_MODIFY
        return PermissionTier.USER_WRITE

    @property
    def requires_confirmation(self) -> bool:
        return self.permission_tier >= PermissionTier.SYSTEM_MODIFY

    @property
    def requires_snapshot(self) -> bool:
        return self.permission_tier >= PermissionTier.DESTRUCTIVE


Action.model_rebuild()


class ActionPlan(BaseModel):
    """A complete plan generated by the Planner from user input."""

    actions: list[Action] = Field(default_factory=list)
    explanation: str = ""
    error: str | None = None
    raw_input: str = ""

    @property
    def max_tier(self) -> PermissionTier:
        if not self.actions:
            return PermissionTier.READ_ONLY
        return max(a.permission_tier for a in self.actions)

    @property
    def needs_snapshot(self) -> bool:
        return any(a.requires_snapshot for a in self.actions)


class ActionResult(BaseModel):
    """Result of executing a single action."""

    action: Action
    success: bool
    output: str = ""
    error: str | None = None
    snapshot_id: str | None = None


class VerificationResult(BaseModel):
    """Result of post-execution verification."""

    passed: bool
    details: list[str] = Field(default_factory=list)
    failed_actions: list[int] = Field(default_factory=list)
    rollback_triggered: bool = False
