"""Pydantic models for structured action plans.

Every LLM output is parsed into these models. The Executor only accepts
validated Action objects ΓÇö there is no path from raw LLM text to system calls.
"""

from __future__ import annotations

import re
from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator, model_validator

from pilot.security.secret_urls import mask_if_secret, mask_secret, mask_url, redact_headers, redact_text


class ActionType(StrEnum):
    # -- File operations --
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_MOVE = "file_move"
    FILE_COPY = "file_copy"
    FILE_LIST = "file_list"
    FILE_SEARCH = "file_search"
    DIRECTORY_SUMMARY = "directory_summary"
    DIRECTORY_SIZE = "directory_size"
    FILE_HASH = "file_hash"
    FILE_COMPARE = "file_compare"
    FILE_PERMISSIONS = "file_permissions"
    GIT_RESOLVE = "git_resolve"

    # -- Git repository operations --
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"
    GIT_LOG = "git_log"
    GIT_BRANCH = "git_branch"
    GIT_STAGE = "git_stage"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"

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
    PTY_EXEC = "pty_exec"  # Run command in a persistent PTY shell session

    # -- Open URL / Application / Notify (original) --
    OPEN_URL = "open_url"
    OPEN_APPLICATION = "open_application"
    NOTIFY = "notify"
    REQUEST_HUMAN_HELP = "request_human_help"

    # -- Process management --
    PROCESS_LIST = "process_list"
    PROCESS_KILL = "process_kill"
    PROCESS_INFO = "process_info"

    # -- Clipboard --
    CLIPBOARD_READ = "clipboard_read"
    CLIPBOARD_WRITE = "clipboard_write"

    # -- System info & monitoring --
    SYSTEM_INFO = "system_info"
    SYSTEM_HEALTH_REVIEW = "system_health_review"
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

    # -- Forensics & Log Analysis --
    LOG_ANALYZE = "log_analyze"

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
    SCREEN_ACCESSIBILITY_SNAPSHOT = "screen_accessibility_snapshot"

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
    BROWSER_SNAPSHOT = "browser_snapshot"

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

    # -- Calendar operations --
    CALENDAR_PARSE = "calendar_parse"
    CALENDAR_SYNC = "calendar_sync"
    CALENDAR_CREATE_EVENT = "calendar_create_event"
    CALENDAR_LIST_EVENTS = "calendar_list_events"
    CALENDAR_DELETE_EVENT = "calendar_delete_event"

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
    # -- Workspace semantic search (RAG) --
    WORKSPACE_INDEX = "workspace_index"
    WORKSPACE_SEARCH = "workspace_search"

    # -- Email (IMAP/SMTP) --
    EMAIL_FETCH = "email_fetch"
    EMAIL_SUMMARIZE = "email_summarize"
    EMAIL_REPLY = "email_reply"

    # -- Calendar reconciliation --
    CALENDAR_FETCH = "calendar_fetch"
    CALENDAR_RECONCILE = "calendar_reconcile"

    # -- Remote execution (SSH) --
    SSH_COMMAND = "ssh_command"
    SSH_SCRIPT = "ssh_script"

    # -- VLM zero-shot element detection --
    SCREEN_DETECT_ELEMENTS = "screen_detect_elements"

    # -- WebAssembly & Python Plugin execution --
    WASM_CALL = "wasm_call"
    PLUGIN_CALL = "plugin_call"

    # -- Third-party skills (dynamic ``pilot/skills`` loader) --
    SKILL_RUN = "skill_run"

    # -- Tools on MCP servers the person configured (``pilot.mcp_client``) --
    MCP_LIST_TOOLS = "mcp_list_tools"
    MCP_CALL_TOOL = "mcp_call_tool"


class PermissionTier(int, Enum):
    READ_ONLY = 0
    USER_WRITE = 1
    SYSTEM_MODIFY = 2
    DESTRUCTIVE = 3
    ROOT_CRITICAL = 4


READ_ONLY_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
SUPPORTED_HTTP_METHODS = READ_ONLY_HTTP_METHODS | frozenset({"POST", "PUT", "PATCH", "DELETE"})


READ_ONLY_ACTIONS = {
    ActionType.FILE_READ,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.DIRECTORY_SUMMARY,
    ActionType.DIRECTORY_SIZE,
    ActionType.FILE_HASH,
    ActionType.FILE_COMPARE,
    ActionType.GIT_STATUS,
    ActionType.GIT_DIFF,
    ActionType.GIT_LOG,
    ActionType.PACKAGE_SEARCH,
    ActionType.SERVICE_STATUS,
    ActionType.GNOME_SETTING_READ,
    ActionType.OPEN_URL,
    ActionType.OPEN_APPLICATION,
    ActionType.NOTIFY,
    ActionType.REQUEST_HUMAN_HELP,
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_INFO,
    ActionType.CLIPBOARD_READ,
    ActionType.SYSTEM_INFO,
    ActionType.SYSTEM_HEALTH_REVIEW,
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
    ActionType.LOG_ANALYZE,
    # Tier 1 read-only
    ActionType.MOUSE_POSITION,
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_ELEMENT_MAP,
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
    ActionType.SCREEN_DETECT_ELEMENTS,
    ActionType.BROWSER_EXTRACT,
    ActionType.BROWSER_EXTRACT_TABLE,
    ActionType.BROWSER_EXTRACT_LINKS,
    ActionType.BROWSER_SCREENSHOT,
    ActionType.BROWSER_LIST_TABS,
    ActionType.BROWSER_PAGE_INFO,
    ActionType.BROWSER_SNAPSHOT,
    ActionType.BROWSER_WAIT,
    ActionType.TRIGGER_LIST,
    # Tier 2 read-only
    ActionType.FILE_PARSE,
    ActionType.FILE_SEARCH_CONTENT,
    ActionType.API_SCRAPE,
    ActionType.WORKSPACE_INDEX,
    ActionType.WORKSPACE_SEARCH,
    # Email agent read-only
    ActionType.EMAIL_FETCH,
    ActionType.EMAIL_SUMMARIZE,
    # Calendar reconciliation (read-only — only reads ICS data)
    ActionType.CALENDAR_FETCH,
    ActionType.CALENDAR_RECONCILE,
    ActionType.CALENDAR_PARSE,
    ActionType.CALENDAR_LIST_EVENTS,
    # Lists tools of the person's MCP servers from the local cache
    ActionType.MCP_LIST_TOOLS,
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
    ActionType.CALENDAR_DELETE_EVENT,
    # Arbitrary script execution in page context can exfiltrate cookies/
    # tokens or bypass CSP — a higher bar than the other browser actions
    # below, which only interact with visible page elements.
    ActionType.BROWSER_EXECUTE_JS,
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
    # Email agent actions (IMAP fetch is read-only; reply/send require confirmation)
    ActionType.EMAIL_REPLY,
    # SSH is always a remote system modification surface
    ActionType.SSH_COMMAND,
    ActionType.SSH_SCRIPT,
    ActionType.CODE_EXECUTE,
    ActionType.SHELL_COMMAND,
    ActionType.CALENDAR_SYNC,
    ActionType.CALENDAR_CREATE_EVENT,
    ActionType.GIT_COMMIT,
    ActionType.GIT_PUSH,
    # State-changing browser actions — previously hardcoded to ALWAYS_SAFE
    # regardless of what they actually did (see the ALWAYS_SAFE set below).
    # Navigating, clicking, typing, selecting, and submitting forms can
    # trigger purchases, account changes, or arbitrary site-defined side
    # effects, so they now require confirmation like any other Tier 2 action.
    ActionType.BROWSER_NAVIGATE,
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_FILL_FORM,
    # Marketplace plugins are reviewed before publication, but Python plugin
    # tools can still perform external or local side effects at runtime.
    ActionType.PLUGIN_CALL,
}

# Actions that cannot be undone by the local snapshot/rollback mechanism
# (SnapshotManager only reverts local disk/filesystem state), regardless of
# their permission tier. A Tier 2 email send is not "destructive" to local
# files but is just as unrecoverable as a Tier 3 file delete once it fires.
IRREVERSIBLE_ACTIONS = {
    # External communications — cannot be recalled once sent
    ActionType.API_SEND_EMAIL,
    ActionType.API_WEBHOOK,
    ActionType.API_SLACK,
    ActionType.API_DISCORD,
    ActionType.EMAIL_REPLY,
    # Remote hosts — outside the local snapshot's reach entirely
    ActionType.SSH_COMMAND,
    ActionType.SSH_SCRIPT,
    # Power actions — cannot be "rolled back" once they take effect
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_LOGOUT,
    # Package removal can lose config/data a simple reinstall won't restore
    ActionType.PACKAGE_REMOVE,
    # Upgrading every installed package: nothing Heliox keeps can put the earlier versions back
    ActionType.PACKAGE_UPDATE,
    # Arbitrary script execution can exfiltrate data before a snapshot
    # rollback would ever have a chance to matter.
    ActionType.BROWSER_EXECUTE_JS,
    ActionType.GIT_PUSH,
}

# Actions whose post-condition MISMATCH is settled by the person, never by running the action again: the action
# succeeded and left something others can see (an event on a shared calendar), so a retry would add a second one
# beside the wrong one. Unlike IRREVERSIBLE_ACTIONS this adds no approval prompt; it only stops the retry.
# Desktop input that may have gone to another window than the one meant (keys typed or pressed, a click or a drag)
# is in the same set: repeating it blindly could put the text there again or do the click's harm twice, so the run
# stops and says which window was in front instead (pilot.system.desktop_observe.input_went_elsewhere).
RECONCILE_NOT_RETRY_ACTIONS = frozenset(
    {
        ActionType.CALENDAR_CREATE_EVENT,
        ActionType.KEYBOARD_TYPE,
        ActionType.KEYBOARD_PRESS,
        ActionType.KEYBOARD_HOTKEY,
        ActionType.KEYBOARD_HOLD,
        ActionType.MOUSE_CLICK,
        ActionType.MOUSE_DOUBLE_CLICK,
        ActionType.MOUSE_RIGHT_CLICK,
        ActionType.MOUSE_DRAG,
    }
)


# -- Parameter models --


class FileParams(BaseModel):
    path: str = ""
    content: str | None = None
    destination: str | None = None
    recursive: bool = False
    pattern: str | None = None  # For file_search
    max_depth: int = 3  # For directory_summary
    max_entries: int = 200  # For directory_summary
    ignore_dirs: list[str] = Field(default_factory=lambda: [".git", "node_modules"])  # For directory_summary
    permissions: str | None = None  # e.g. "755" for file_permissions
    hash_algorithm: Literal["sha256", "sha512", "blake2b"] = "sha256"
    # file_read only: the first or the last N lines instead of the whole file ("show the last lines of this log").
    head_lines: int | None = Field(default=None, ge=1, le=100_000)
    tail_lines: int | None = Field(default=None, ge=1, le=100_000)
    # file_write only: add ``content`` as new lines at the end of the file (made when missing) instead of replacing
    # the file. Exactly one line break separates the earlier text from the new; the file's line endings and encoding
    # are kept (pilot.system.filesystem.file_append).
    append: bool = False

    @model_validator(mode="after")
    def _one_line_limit(self) -> FileParams:
        if self.head_lines is not None and self.tail_lines is not None:
            raise ValueError("give head_lines or tail_lines, not both")
        return self


class GitResolveParams(BaseModel):
    path: str = ""
    full_block: str = ""
    resolved_code: str = ""


class GitParams(BaseModel):
    """Validated parameters for repository-scoped Git actions."""

    repo_path: str = "."
    name: str = ""
    create: bool = True
    files: list[str] = Field(default_factory=list)
    message: str = ""
    remote: str = "origin"
    branch: str = ""
    force: bool = False
    limit: int = 20
    staged: bool = False


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


class PtyExecParams(BaseModel):
    """Run a command inside a persistent PTY shell session."""

    session_id: str = "default"
    command: str = ""
    timeout: int = 30


class OpenUrlParams(BaseModel):
    url: str = ""


class OpenApplicationParams(BaseModel):
    name: str = ""
    args: list[str] = Field(default_factory=list)


class NotifyParams(BaseModel):
    summary: str = "Pilot notification"
    body: str = ""


class HumanHelpParams(BaseModel):
    """For REQUEST_HUMAN_HELP: hand the desktop to the person for something Heliox must not do."""

    help_needed: str = Field(min_length=1, max_length=200)  # e.g. "Sign in to Outlook"
    instructions: str = Field(default="", max_length=500)


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
    step: int | None = None  # volume_set: change by this many points (-100 to 100) instead of to a level


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
    # As for api_request: true lets the download reach an address on this computer or the local network, and the step
    # then needs the person's approval (Action.permission_tier, pilot.system.outbound).
    allow_private_network: bool = False
    # Optional: the file's published sha256; the post-condition check compares the downloaded file with it.
    sha256: str | None = None


class RegistryParams(BaseModel):
    """Windows registry operations."""

    key_path: str = ""  # e.g. HKCU\\Software\\...
    value_name: str = ""
    value_data: str | None = None
    value_type: str = "REG_SZ"


class LogAnalyzeParams(BaseModel):
    """Parameters for log analysis operations."""

    log_path: str = ""  # path to log file (or log category e.g., 'auth', 'syslog', 'nginx')
    log_type: str | None = None  # syslog, auth, json, service, nginx, apache, windows
    query: str | None = None  # specific query or filter to apply
    time_window: str | None = None  # e.g., "1h", "24h"
    llm_contextual: bool = False  # whether to run LLM-based contextual analysis


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
    # Accessibility grounding: click an element from screen_accessibility_snapshot
    # instead of coordinates; refused when the snapshot no longer matches the screen.
    element_ref: str | None = None
    snapshot_id: str | None = None


class KeyboardParams(BaseModel):
    """For keyboard control actions."""

    text: str = ""  # For type
    key: str = ""  # For press (enter, tab, escape, f1...)
    keys: list[str] = Field(default_factory=list)  # For hotkey (ctrl, c)
    interval: float = 0.03  # Typing speed
    presses: int = 1
    duration: float = 0.5  # For hold
    window_title: str | None = None  # Bind background-safe text entry to a native app window


class ScreenVisionParams(BaseModel):
    """For OCR, text finding, screen analysis."""

    target_text: str = ""  # For find_text
    prompt: str = "Describe what you see on the screen"  # For analysis
    region: str | None = None  # "x,y,w,h" or None for fullscreen
    language: str = "eng"


class ElementDetectionParams(BaseModel):
    """For VLM zero-shot element detection (SCREEN_DETECT_ELEMENTS).

    The VLM is asked to locate all interactive UI elements in a screenshot
    and return structured bounding-box coordinates so the agent can click
    or type without relying on DOM trees or accessibility APIs.
    """

    description: str = ""  # Optional: natural-language filter, e.g. "login button"
    region: str | None = None  # "x,y,w,h" or None for full screen
    max_elements: int = 20  # cap on returned elements
    action_filter: str = ""  # "click" | "type" | "" (all)


class AccessibilitySnapshotParams(BaseModel):
    """For SCREEN_ACCESSIBILITY_SNAPSHOT: list the foreground window's UI elements with refs."""

    element_limit: int = Field(default=120, ge=1, le=300)


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
    # Act on an element from browser_snapshot instead of a CSS selector; refused if the page changed.
    element_ref: str | None = None
    snapshot_id: str | None = None
    button: str = "left"  # For click


class TriggerParams(BaseModel):
    """For reactive trigger management."""

    name: str = ""
    trigger_type: str = ""  # cpu_threshold, file_created, etc.
    condition: dict = Field(default_factory=dict)  # Type-specific conditions
    action_command: str = ""  # What to do when triggered, in words
    trigger_id: str = ""  # For delete
    max_fires: int = 0  # 0 = unlimited
    cooldown_seconds: int = 60
    # trigger_create: the exact steps a fire runs, fixed now (at most 5; no trigger or schedule steps). Their
    # approval is asked for with this step (Action.permission_tier), and a fire runs these and nothing else.
    steps: list[Action] = Field(default_factory=list, max_length=5)
    # trigger_create: the job is only to tell the person ("tell me when notes.txt changes"). A fire is Heliox's own
    # notice, shown and spoken; nothing is planned or run. No steps and no action_command are needed.
    notify_only: bool = False


# ======= TIER 2: MULTIPLIER PARAMS =======


class SkillRunParams(BaseModel):
    """Invoke a dynamically loaded :class:`pilot.skills.base.Skill`."""

    skill_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "arguments" not in data and "params" in data:
            data = dict(data)
            data["arguments"] = data.pop("params")
        return data


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
    """For generic API and service calls.

    ``webhook_url`` is always a secret (whoever has it can post), and ``url`` is one when it carries a user,
    a password or a token. Both stay whole in memory for the step that sends; every serialization (the plan
    card, the approval, the audit log, receipts, the saved plan, a repair prompt) and the repr in log lines
    shows them masked by ``pilot.security.secret_urls``.
    """

    @field_serializer("webhook_url")
    def _mask_webhook(self, value: str | None) -> str | None:
        return mask_url(value)

    @field_serializer("url")
    def _mask_secret_url(self, value: str) -> str:
        return mask_if_secret(value) or ""

    @field_serializer("token")
    def _mask_token(self, value: str | None) -> str | None:
        return mask_secret(value)

    @field_serializer("headers")
    def _mask_secret_headers(self, value: dict[str, str]) -> dict[str, str]:
        # Authorization, Cookie, X-Api-Key and the like (pilot.security.secret_urls.is_secret_header).
        return redact_headers(value)

    def __repr_args__(self):  # type: ignore[no-untyped-def]
        for name, value in super().__repr_args__():
            if name == "webhook_url":
                yield name, mask_url(value)
            elif name == "url":
                yield name, mask_if_secret(value)
            elif name == "token":
                yield name, mask_secret(value)
            elif name == "headers":
                yield name, redact_headers(value)
            else:
                yield name, value

    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict | None = None
    params: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    allow_private_network: bool = False
    # GitHub specific
    endpoint: str = ""  # e.g. /user/repos
    # Not for new plans: the token comes from the GitHub connection (pilot.system.connections). A plan that still
    # carries one works; every dump shows it masked.
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


class WorkspaceParams(BaseModel):
    """For workspace indexing and semantic search."""

    folder_path: str = ""
    query: str = ""
    n_results: int = 5


class EmailParams(BaseModel):
    """Parameters for IMAP/SMTP email operations.

    The daemon fills the connection fields from the Email settings and the credential store just before a step
    runs; ``app_password`` is masked in every dump and repr, so it never reaches a plan card, log or saved plan.
    """

    @field_serializer("app_password")
    def _mask_app_password(self, value: str) -> str:
        return mask_secret(value) or ""

    def __repr_args__(self):  # type: ignore[no-untyped-def]
        for name, value in super().__repr_args__():
            yield name, (mask_secret(value) if name == "app_password" else value)

    # Connection settings
    imap_host: str = ""  # e.g. imap.gmail.com
    smtp_host: str = ""  # e.g. smtp.gmail.com
    smtp_port: int = 587
    username: str = ""  # full email address
    app_password: str = ""  # App Password (not account password)

    # Fetch options
    mailbox: str = "INBOX"
    max_emails: int = 10  # max unread emails to fetch
    mark_as_read: bool = False  # mark fetched emails as read

    # Reply / send options
    reply_to_uid: str = ""  # UID of the email to reply to
    in_reply_to: str = ""  # Message-ID of the email to reply to (email_fetch reports it as message_id)
    reply_body: str = ""  # pre-written reply body (empty = LLM drafts it)
    subject: str = ""  # subject override for new emails
    to: str = ""  # recipient for new emails

    # Summarise options
    emails_json: str = ""  # JSON-serialised list of fetched emails to summarise


class CalendarParams(BaseModel):
    """Parameters for calendar operations (.ics and CalDAV)."""

    file_path: str | None = None
    summary: str | None = None
    start: str | None = None
    end: str | None = None
    description: str | None = None
    location: str | None = None
    calendar_id: str | None = None
    event_uid: str | None = None
    # Reconciliation fields
    emails_json: str = ""
    lookahead_hours: int = 24
    check_conflicts: bool = True
    check_missing_links: bool = True
    notify: bool = True


class SshCommandParams(BaseModel):
    """Parameters for executing a single command over SSH on a configured host."""

    host: str = ""
    command: str = ""
    timeout_seconds: int = 60


class SshScriptParams(BaseModel):
    """Parameters for executing a multi-line bash script over SSH on a configured host."""

    host: str = ""
    script: str = ""
    timeout_seconds: int = 300


class WasmCallParams(BaseModel):
    """Parameters for wasm_call action."""

    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class McpToolParams(BaseModel):
    """For MCP_LIST_TOOLS / MCP_CALL_TOOL: a tool on an MCP server the person configured."""

    mcp_server: str = Field(default="", max_length=40)
    mcp_tool: str = Field(default="", max_length=80)
    mcp_arguments: dict[str, Any] = Field(default_factory=dict)


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
    | PtyExecParams
    | OpenUrlParams
    | OpenApplicationParams
    | NotifyParams
    | HumanHelpParams
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
    | LogAnalyzeParams
    | MouseParams
    | KeyboardParams
    | ScreenVisionParams
    | BrowserParams
    | TriggerParams
    | CodeExecParams
    | FileIntelParams
    | ApiRequestParams
    | WorkspaceParams
    | EmailParams
    | CalendarParams
    | SshCommandParams
    | SshScriptParams
    | ElementDetectionParams
    | AccessibilitySnapshotParams
    | WasmCallParams
    | SkillRunParams
    | GitParams
    | GitResolveParams
    | McpToolParams
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
    dangerous_flags: list[str] = Field(default_factory=list)  # populated by ActionValidator
    # A declared post-condition: a read-only probe run after this action, and what it must find.
    # Checked and run by pilot.agents.postconditions.declared; absent means the step stays "reported".
    expect: Expectation | None = None

    @field_validator("target")
    @classmethod
    def _target_without_secrets(cls, value: str) -> str:
        # The target is only ever shown (plan card, approval, audit, receipts); a webhook or token URL in it is
        # masked at once. The handlers take their address from the parameters.
        return redact_text(value) or ""

    @property
    def permission_tier(self) -> PermissionTier:
        # These actions are ALWAYS safe ΓÇö never require confirmation
        #
        # NOTE: browser actions used to all be listed here regardless of
        # what they actually did (see the Agent Gateway feature / SECURITY.md
        # for the full writeup) — they're now individually classified via
        # READ_ONLY_ACTIONS (extraction/inspection), left at the default
        # USER_WRITE tier below (passive/local: hover, scroll, tab/nav
        # housekeeping), SYSTEM_MODIFY_ACTIONS (state-changing: navigate,
        # click, type, select, fill_form), or DESTRUCTIVE_ACTIONS
        # (BROWSER_EXECUTE_JS — arbitrary script execution).
        ALWAYS_SAFE = {
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
            ActionType.GIT_RESOLVE,
            ActionType.FILE_LIST,
            ActionType.FILE_SEARCH,
            ActionType.FILE_COPY,
            ActionType.CODE_GENERATE_AND_RUN,
            ActionType.SKILL_RUN,
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
            ActionType.DOWNLOAD_FILE,
            ActionType.WASM_CALL,
        }
        if self.action_type == ActionType.DOWNLOAD_FILE and bool(
            getattr(self.parameters, "allow_private_network", False)
        ):
            return PermissionTier.SYSTEM_MODIFY
        if self.action_type in ALWAYS_SAFE:
            return PermissionTier.USER_WRITE
        if self.action_type == ActionType.TRIGGER_CREATE:
            # Making a trigger approves what it will run unattended: its steps' approval is asked for now. Capped
            # at SYSTEM_MODIFY (an approval, no snapshot: making the trigger changes no file).
            steps = getattr(self.parameters, "steps", None) or []
            tiers = [PermissionTier.USER_WRITE, *(step.permission_tier for step in steps)]
            return min(max(tiers), PermissionTier.SYSTEM_MODIFY)
        if self.action_type in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
            if self.action_type == ActionType.API_REQUEST and bool(
                getattr(self.parameters, "allow_private_network", False)
            ):
                return PermissionTier.SYSTEM_MODIFY
            method = str(getattr(self.parameters, "method", "GET")).upper()
            if method in READ_ONLY_HTTP_METHODS:
                return PermissionTier.READ_ONLY
            return PermissionTier.SYSTEM_MODIFY
        if self.action_type == ActionType.API_SCRAPE and bool(getattr(self.parameters, "allow_private_network", False)):
            return PermissionTier.SYSTEM_MODIFY
        if self.action_type in READ_ONLY_ACTIONS:
            return PermissionTier.READ_ONLY
        if self.requires_root:
            return PermissionTier.ROOT_CRITICAL
        if self.action_type == ActionType.MCP_CALL_TOOL:
            from pilot.mcp_client import tool_permission_tier

            return tool_permission_tier(
                str(getattr(self.parameters, "mcp_server", "")), str(getattr(self.parameters, "mcp_tool", ""))
            )
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

    @property
    def is_irreversible(self) -> bool:
        """True when this action cannot be undone via local snapshot rollback.

        Distinct from ``permission_tier``/``destructive``: a Tier 2 email send
        is irreversible even though it isn't a Tier 3+ "destructive" action,
        while most Tier 3 file operations ARE recoverable via snapshot and so
        are NOT irreversible. Confirmation gating must never be bypassable for
        an irreversible action regardless of global config toggles.
        """
        if self.permission_tier == PermissionTier.ROOT_CRITICAL:
            return True
        if self.action_type == ActionType.TRIGGER_CREATE and any(
            step.is_irreversible or step.permission_tier == PermissionTier.ROOT_CRITICAL
            for step in getattr(self.parameters, "steps", None) or []
        ):
            return True
        if self.dangerous_flags:
            return True
        if self.action_type in {ActionType.API_REQUEST, ActionType.API_SCRAPE, ActionType.DOWNLOAD_FILE} and bool(
            getattr(self.parameters, "allow_private_network", False)
        ):
            return True
        if self.action_type in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
            method = str(getattr(self.parameters, "method", "GET")).upper()
            return method not in READ_ONLY_HTTP_METHODS
        if self.action_type == ActionType.MCP_CALL_TOOL:
            # A tool on another server can send, post or delete; local snapshots cannot undo that.
            return self.permission_tier > PermissionTier.READ_ONLY
        return self.action_type in IRREVERSIBLE_ACTIONS


EXPECT_KINDS = ("exit_code", "equals", "contains", "pattern", "json_equals", "exists", "absent")
MAX_EXPECT_TEXT = 500
MAX_EXPECT_WILDCARDS = 8
MAX_EXPECTS_PER_PLAN = 4
_EXPECT_FIELD = re.compile(r"^[A-Za-z0-9_\-]{1,64}(\.[A-Za-z0-9_\-]{1,64}){0,7}$")


class Expectation(BaseModel):
    """What a read-only probe must find after the action ran.

    ``kind`` says how the probe's reading is compared: its exit code, its whole output (``equals``), part of it
    (``contains``), a glob over it (``pattern``, ``*`` and ``?`` only, never a regular expression), one JSON field
    (``json_equals`` with a dotted ``field``), or whether the thing it reads is there (``exists``, ``absent``).
    Whether the probe is allowed at all is decided by ``pilot.agents.postconditions.declared.probe_refusal``.
    """

    probe: Action
    kind: Literal["exit_code", "equals", "contains", "pattern", "json_equals", "exists", "absent"]
    value: str | int | float | bool | None = None
    field: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def _value_fits_kind(self) -> Expectation:
        if self.probe.expect is not None or self.probe.rollback_action is not None:
            raise ValueError("a probe cannot carry its own expect or rollback_action")
        kind, value = self.kind, self.value
        if kind == "exit_code":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError("exit_code needs an integer from 0 to 255")
        elif kind in {"equals", "contains", "pattern"}:
            if not isinstance(value, str) or not value or len(value) > MAX_EXPECT_TEXT:
                raise ValueError(f"{kind} needs text of 1 to {MAX_EXPECT_TEXT} characters")
            if kind == "pattern" and value.count("*") + value.count("?") > MAX_EXPECT_WILDCARDS:
                raise ValueError(f"a pattern may hold at most {MAX_EXPECT_WILDCARDS} wildcards")
        elif kind == "json_equals":
            if not _EXPECT_FIELD.match(self.field):
                raise ValueError("json_equals needs a dotted field such as state or items.0.id")
            if isinstance(value, str) and len(value) > MAX_EXPECT_TEXT:
                raise ValueError(f"json_equals takes a value of at most {MAX_EXPECT_TEXT} characters")
        elif value is not None:
            raise ValueError(f"{kind} takes no value")
        if kind != "json_equals" and self.field:
            raise ValueError("field is only for json_equals")
        return self


TriggerParams.model_rebuild()
Action.model_rebuild()
Expectation.model_rebuild()


class ActionPlan(BaseModel):
    """A complete plan generated by the Planner from user input."""

    actions: list[Action] = Field(default_factory=list)
    explanation: str = ""
    error: str | None = None
    raw_input: str = ""

    @field_validator("explanation")
    @classmethod
    def _explanation_without_secrets(cls, value: str) -> str:
        return redact_text(value) or ""

    @field_serializer("raw_input")
    def _raw_input_without_secrets(self, value: str) -> str:
        # The request stays whole in memory (the planner needs the address it names); kept or shown, it is masked.
        return redact_text(value) or ""

    @property
    def max_tier(self) -> PermissionTier:
        if not self.actions:
            return PermissionTier.READ_ONLY
        return max(a.permission_tier for a in self.actions)

    @property
    def needs_snapshot(self) -> bool:
        return any(a.requires_snapshot for a in self.actions)

    @property
    def needs_confirmation_unconditional(self) -> bool:
        """True if the plan contains an action that must be confirmed no
        matter what confirmation-bypass config toggles are set."""
        return any(a.is_irreversible for a in self.actions)


class ActionResult(BaseModel):
    """Result of executing a single action."""

    action: Action
    success: bool
    output: str = ""
    error: str | None = None
    snapshot_id: str | None = None
    executed: bool = True
    skip_reason: str | None = None
    # A success with a part the person must see, such as recipients the mail server refused. The reply and the
    # task receipt carry it in full; the output itself is kept in the receipt only as a hash.
    notice: str | None = None
    # Lines a browser handler adds for its own post-condition check ([PAGE_BEFORE], [DOM_DIFF]): hashes and numbers
    # for the verifier. They are taken out of ``output`` (split_machine_lines), so they are never shown, spoken,
    # narrated or sent to a model, and they are left out of every serialised copy of the result.
    observation: str | None = Field(default=None, exclude=True)
    # The outcome of this result's declared ``expect`` probe, kept so a second verification of the same
    # result (the retry merge in server.py verifies merged results again) never runs the probe twice.
    _expect_outcome: Any = PrivateAttr(default=None)

    @field_serializer("output", "error")
    def _without_secret_urls(self, value: str | None) -> str | None:
        return redact_text(value)


# The browser handlers that end their output with machine lines for their check, and the lines' prefixes. Only
# these actions are split, so a file or page that happens to contain such a line is never cut.
MACHINE_LINE_ACTIONS = frozenset(
    {
        ActionType.BROWSER_CLICK,
        ActionType.BROWSER_CLICK_TEXT,
        ActionType.BROWSER_TYPE,
        ActionType.BROWSER_SELECT,
        ActionType.BROWSER_FILL_FORM,
        ActionType.BROWSER_SCROLL,
        ActionType.BROWSER_HOVER,
        ActionType.BROWSER_BACK,
        ActionType.BROWSER_FORWARD,
        ActionType.BROWSER_REFRESH,
        ActionType.BROWSER_NEW_TAB,
        ActionType.BROWSER_CLOSE_TAB,
        ActionType.BROWSER_CLOSE,
    }
)
MACHINE_LINE_PREFIXES = ("[PAGE_BEFORE]", "[DOM_DIFF]")


def split_machine_lines(action_type: ActionType, output: Any) -> tuple[Any, str | None]:
    """Separate a browser handler's words from its machine lines: (words, machine lines or None).

    The single point where handler output becomes the text a person sees or hears, or a model reads. Output with
    no machine lines (and every other action's output) is returned unchanged, notice and all.
    """
    if action_type not in MACHINE_LINE_ACTIONS or not isinstance(output, str):
        return output, None
    lines = output.splitlines()
    machine = [line for line in lines if line.startswith(MACHINE_LINE_PREFIXES)]
    if not machine:
        return output, None
    words = "\n".join(line for line in lines if not line.startswith(MACHINE_LINE_PREFIXES))
    notice = getattr(output, "notice", None)
    if notice is not None:
        words = OutputWithNotice(words, notice, nothing_done=bool(getattr(output, "nothing_done", False)))
    return words, "\n".join(machine)


class OutputWithNotice(str):
    """A handler's output that succeeded with a caveat; the executor moves ``notice`` into ``ActionResult.notice``."""

    notice: str
    nothing_done: bool

    def __new__(cls, text: str, notice: str, *, nothing_done: bool = False) -> OutputWithNotice:
        output = super().__new__(cls, text)
        output.notice = notice
        # True when the target was not there, so nothing happened: the executor marks the result "already_absent".
        output.nothing_done = nothing_done
        return output


class VerificationResult(BaseModel):
    """Result of post-execution verification."""

    passed: bool
    details: list[str] = Field(default_factory=list)
    failed_actions: list[int] = Field(default_factory=list)
    independently_observed_actions: list[int] = Field(default_factory=list)
    rollback_triggered: bool = False
    # Action index to the verifier's one-sentence reason, for every outcome that is not a plain VERIFIED and has a
    # specific reason (a receipt only, a look that was refused or could not be made, a mismatch).
    reasons: dict[int, str] = Field(default_factory=dict)
    # Action index to the kind of a step that was not plainly verified (pilot.agents.unconfirmed.Kind): what decides
    # whether it is said aloud, counted in the window's line, or only kept in the receipt.
    reason_kinds: dict[int, str] = Field(default_factory=dict)
    # Action index to a sentence the person sees beside a result that holds (packages an upgrade left behind).
    notices: dict[int, str] = Field(default_factory=dict)
    # Steps whose check was still running when the reply went out; their outcome reaches the receipt later.
    pending_actions: list[int] = Field(default_factory=list)
