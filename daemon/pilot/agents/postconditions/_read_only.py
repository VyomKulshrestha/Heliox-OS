"""Read-only actions that need no post-condition check, each with the reason.

A read changes nothing, so there is no state to observe afterwards: what it returns is its result. A second reading
through the same source proves nothing, and values such as CPU load or the window list move between two readings.
What protects a read is that a read which fails now fails its action instead of returning an error message as data
(tests/test_failed_actions_more.py). Six reads that can lose data on the way (file_read, file_parse, file_hash,
browser_extract, browser_extract_table, browser_screenshot) get real checks instead and are not listed here.

The catalog generator shows these as ``verification.method = "read_only"`` with the reason, and counts them apart
from the actions that change something. An action listed here must not also register a verifier.
"""

from __future__ import annotations

from pilot.actions import ActionType

_SAME_SOURCE = "a second reading would ask the same source"
_MOVING = "the value moves between two readings"
_CUT_SAID = "a cut is said in the output"

READ_ONLY_WITHOUT_POSTCONDITION: dict[ActionType, str] = {
    ActionType.GIT_STATUS: f"Reads the working tree through git; {_SAME_SOURCE}.",
    ActionType.GIT_DIFF: f"Reads a diff through git; {_SAME_SOURCE}.",
    ActionType.GIT_LOG: f"Reads history through git; {_SAME_SOURCE}.",
    ActionType.FILE_COMPARE: "Compares two files by hash; the comparison is the result.",
    ActionType.SERVICE_STATUS: f"Reads a unit's state from systemd; {_SAME_SOURCE}.",
    ActionType.PROCESS_LIST: f"Lists running processes; {_MOVING}, and {_CUT_SAID}.",
    ActionType.PROCESS_INFO: f"Reads one process's details; {_MOVING}.",
    ActionType.SCHEDULE_LIST: f"Lists scheduled tasks from the system scheduler; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.TRIGGER_LIST: f"Lists triggers from the daemon's trigger engine; {_SAME_SOURCE}.",
    ActionType.DISK_LIST: f"Lists disks from the operating system; {_SAME_SOURCE}.",
    ActionType.REGISTRY_READ: f"Reads a registry value; {_SAME_SOURCE}.",
    ActionType.ENV_GET: f"Reads an environment variable; {_SAME_SOURCE}.",
    ActionType.ENV_LIST: f"Lists environment variables; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.WINDOW_LIST: f"Lists open windows; {_MOVING}.",
    ActionType.FILE_LIST: f"Lists a folder; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.FILE_SEARCH: f"Searches file names in a folder; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.FILE_SEARCH_CONTENT: (
        f"Searches file contents in a folder; {_SAME_SOURCE}, and where it stopped or what it skipped is said."
    ),
    ActionType.EMAIL_FETCH: (
        "Fetches mail from the mail server; a second fetch would ask the same server and mark messages read, "
        "and messages or bodies left out are said."
    ),
    ActionType.CALENDAR_FETCH: f"Fetches invitations from the calendar server; {_SAME_SOURCE}.",
    ActionType.CALENDAR_LIST_EVENTS: (
        f"Lists events from the calendar server; {_SAME_SOURCE}, and calendars or events not read are said."
    ),
    ActionType.CALENDAR_RECONCILE: "Compares local and remote calendars; the comparison is the result.",
    ActionType.CALENDAR_PARSE: "Parses event text; the parsed fields are the result.",
    ActionType.EMAIL_SUMMARIZE: "Summarises fetched mail with a model; the summary is the result.",
    ActionType.BROWSER_EXTRACT_LINKS: f"Reads the links of the open page; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.BROWSER_SNAPSHOT: f"Reads the open page's structure; {_MOVING}.",
    ActionType.BROWSER_PAGE_INFO: f"Reads the open page's address and title; {_SAME_SOURCE}.",
    ActionType.BROWSER_LIST_TABS: f"Lists the browser's tabs; {_SAME_SOURCE}.",
    ActionType.BROWSER_WAIT: "Waits for a page condition; it fails when the condition does not arrive.",
    ActionType.API_SCRAPE: f"Fetches a web page; {_MOVING}, an error status fails the action, and {_CUT_SAID}.",
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT: f"Reads the accessibility tree; {_MOVING}.",
    ActionType.SCREEN_ELEMENT_MAP: f"Maps on-screen elements; {_MOVING}.",
    ActionType.SCREEN_DETECT_ELEMENTS: f"Detects on-screen elements; {_MOVING}.",
    ActionType.SCREEN_FIND_TEXT: f"Finds text on screen; {_MOVING}.",
    ActionType.SCREEN_OCR: f"Reads screen text by OCR; {_MOVING}, and no working OCR engine fails the action.",
    ActionType.SCREEN_ANALYZE: f"Describes the screen with a model; {_MOVING}.",
    ActionType.WORKSPACE_SEARCH: f"Searches the workspace index; {_SAME_SOURCE}.",
    ActionType.DIRECTORY_SUMMARY: f"Summarises a folder; {_SAME_SOURCE}, and what its limits leave out is said.",
    ActionType.DIRECTORY_SIZE: f"Measures a folder; {_SAME_SOURCE}.",
    ActionType.DISK_USAGE: f"Reads free disk space; {_MOVING}.",
    ActionType.LOG_ANALYZE: f"Reads and summarises a log; {_MOVING}.",
    ActionType.SYSTEM_HEALTH_REVIEW: f"Reads system health; {_MOVING}.",
    ActionType.PACKAGE_SEARCH: f"Searches the package index; {_SAME_SOURCE}, and {_CUT_SAID}.",
    ActionType.MCP_LIST_TOOLS: f"Lists the tools of configured MCP servers; {_SAME_SOURCE}.",
    ActionType.WIFI_LIST: f"Lists Wi-Fi networks; {_MOVING}.",
    ActionType.NETWORK_INFO: f"Reads network state; {_MOVING}.",
    ActionType.USER_LIST: f"Lists user accounts; {_SAME_SOURCE}.",
    ActionType.USER_INFO: f"Reads the current user's details; {_SAME_SOURCE}.",
    ActionType.GNOME_SETTING_READ: f"Reads a GNOME setting; {_SAME_SOURCE}.",
    ActionType.CLIPBOARD_READ: "Reads the clipboard; there is no second source, and anyone can copy over it at any moment.",
    ActionType.SYSTEM_INFO: f"Reads system details; {_SAME_SOURCE}.",
    ActionType.MEMORY_USAGE: f"Reads memory use; {_MOVING}.",
    ActionType.CPU_USAGE: f"Reads CPU load; {_MOVING}.",
    ActionType.BATTERY_INFO: f"Reads battery state; {_MOVING}.",
    ActionType.VOLUME_GET: f"Reads the volume; {_SAME_SOURCE}.",
    ActionType.BRIGHTNESS_GET: f"Reads the brightness; {_SAME_SOURCE}.",
    ActionType.MOUSE_POSITION: f"Reads the pointer position; {_MOVING}.",
}
