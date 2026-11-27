"""What people call the things Heliox's actions read or do, keyed by the action catalog (``pilot.actions.ActionType``).

The router's rules (pilot.conversation.router) decide most turns. This table is the net under them. In the live run
of 26 November "what is on my clipboard", "how much free disk space is there", "append the line ... to note.txt",
"post ... to Slack" and "watch the file ..." all ended as talk: no rule knew those words, and the talk model then said
the work was under way when no task existed. With the table, a request that names something a read action answers,
or starts with a verb that maps to an action, becomes a task whatever the model would have made of it.

Every action type is in exactly one of NAMED and UNNAMED (tests/test_conversation_catalog_net.py walks the catalog),
so a new action cannot be added without deciding how a person asks for it. Each named read action carries a
question a person would ask, and the test checks that the router sends that question to the planner.

Words:
- ``subjects``: what the action reads or acts on, named so plainly that a question about it is about this computer
  ("the clipboard", "disk space", "git status", "brightness"), or a service only an action reaches ("Slack", "a
  webhook"). Any action's subject makes an anchored verb (ANCHORED_VERBS) a request.
- ``topics``: words that need a sign of here and now first ("my", "right now", "left", a path) before a question
  about them is about this computer ("memory", "volume", "git", "hash").
- ``verbs``: imperatives that ask for the action ("append", "hash", "scrape").
- ``asked``: for a read action, a question that must reach the planner.

Each entry is a regular expression fragment over the words as ``router.normalize`` leaves them (lower case, no
punctuation, single spaces). All of it is a few compiled patterns: matching takes microseconds, and no model is asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pilot.actions import READ_ONLY_ACTIONS, ActionType

A = ActionType


@dataclass(frozen=True, slots=True)
class Words:
    subjects: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ()
    asked: str = ""


NAMED: dict[ActionType, Words] = {
    # files and folders
    A.FILE_READ: Words(topics=("contents",), verbs=("read",), asked="What are the contents of C:\\work\\note.txt?"),
    A.FILE_WRITE: Words(verbs=("write", "append", "overwrite", "save", "prepend")),
    A.FILE_DELETE: Words(verbs=("delete", "remove", "erase")),
    A.FILE_MOVE: Words(verbs=("move", "rename")),
    A.FILE_COPY: Words(verbs=("copy", "duplicate")),
    A.FILE_LIST: Words(
        subjects=("what(?:'s| s| is) in (?:my|the|this) (?:\\w+ )?folder",),
        topics=("files",),
        verbs=("list",),
        asked="What files are in my downloads folder right now?",
    ),
    A.FILE_SEARCH: Words(verbs=("find", "search", "locate")),
    A.DIRECTORY_SUMMARY: Words(
        topics=("folder summary",), verbs=("summarize", "summarise"), asked="Give me a folder summary of my downloads"
    ),
    A.DIRECTORY_SIZE: Words(
        subjects=(
            "folder size",
            "size of (?:the|my|this) (?:\\w+ )?folder",
            "how big is (?:my|the|this) (?:\\w+ )?folder",
        ),
        topics=("size",),
        asked="What is the folder size of my downloads?",
    ),
    A.FILE_HASH: Words(
        subjects=("sha ?256", "sha ?1", "md5", "checksum"),
        topics=("hash",),
        verbs=("hash",),
        asked="What is the SHA-256 hash of C:\\work\\note.txt?",
    ),
    A.FILE_COMPARE: Words(verbs=("diff",)),
    A.FILE_PERMISSIONS: Words(topics=("file permissions",), verbs=("chmod",)),
    A.FILE_PARSE: Words(verbs=("parse",)),
    A.FILE_SEARCH_CONTENT: Words(verbs=("grep", "search")),
    # git
    A.GIT_STATUS: Words(
        subjects=("git status", "uncommitted (?:changes|files)", "staged (?:changes|files)"),
        topics=("git", "repo", "repository"),
        asked="What's the git status of the repository at C:\\work\\repo?",
    ),
    A.GIT_DIFF: Words(subjects=("git diff",), topics=("diff",), asked="What's the git diff in my repo?"),
    A.GIT_LOG: Words(
        subjects=("git log", "commit history", "(?:last|recent|latest) commits?"),
        topics=("commits",),
        asked="What are the last commits in my repo?",
    ),
    A.GIT_BRANCH: Words(topics=("branch", "branches"), verbs=("branch", "checkout")),
    A.GIT_STAGE: Words(verbs=("stage", "unstage")),
    A.GIT_COMMIT: Words(verbs=("commit",)),
    A.GIT_PUSH: Words(verbs=("push",)),
    A.GIT_RESOLVE: Words(topics=("merge conflicts?",), verbs=("resolve",)),
    # packages and services
    A.PACKAGE_INSTALL: Words(verbs=("install",)),
    A.PACKAGE_REMOVE: Words(verbs=("uninstall",)),
    A.PACKAGE_UPDATE: Words(verbs=("update", "upgrade")),
    A.PACKAGE_SEARCH: Words(
        subjects=("package manager", "packages? (?:called|named)"),
        topics=("packages?",),
        asked="Is there a package for ffmpeg on my computer?",
    ),
    A.SERVICE_START: Words(topics=("service",), verbs=("start",)),
    A.SERVICE_STOP: Words(topics=("service",), verbs=("stop",)),
    A.SERVICE_RESTART: Words(topics=("service",), verbs=("restart",)),
    A.SERVICE_ENABLE: Words(topics=("service",), verbs=("enable",)),
    A.SERVICE_DISABLE: Words(topics=("service",), verbs=("disable",)),
    A.SERVICE_STATUS: Words(
        subjects=(
            "service status",
            "the [\\w ]{1,30} service (?:is )?(?:up|down|running|stopped|started|in|status|state)",
        ),
        topics=("services?",),
        asked="What's the service status of nginx right now?",
    ),
    A.GNOME_SETTING_WRITE: Words(verbs=("set", "turn", "switch")),
    A.SHELL_COMMAND: Words(verbs=("run", "execute")),
    A.SHELL_SCRIPT: Words(verbs=("run", "execute")),
    A.PTY_EXEC: Words(verbs=("run", "execute")),
    A.OPEN_URL: Words(verbs=("open", "visit", "browse")),
    A.OPEN_APPLICATION: Words(verbs=("open", "launch", "start")),
    A.NOTIFY: Words(verbs=("notify",)),
    # processes, clipboard, system
    A.PROCESS_LIST: Words(
        subjects=(
            "(?:programs|processes|apps|applications) (?:are |is )?running",
            "running (?:programs|processes|apps|applications)",
            "what s running",
            "what is running",
            "(?:processes|programs|apps|applications) (?:are )?using",
            "using the most (?:cpu|memory|ram)",
        ),
        topics=("processes", "programs"),
        asked="What programs are running right now?",
    ),
    A.PROCESS_INFO: Words(
        subjects=("process (?:with )?pid \\w+", "pid \\w+", "process \\w+ doing"),
        topics=("process",),
        asked="Is the process chrome using a lot of my memory right now?",
    ),
    A.PROCESS_KILL: Words(verbs=("kill", "terminate")),
    A.CLIPBOARD_READ: Words(
        subjects=("clipboard", "(?:did )?i just cop(?:y|ied)"),
        asked="What is on my clipboard? Just read it, don't change it.",
    ),
    A.CLIPBOARD_WRITE: Words(subjects=("clipboard",), verbs=("copy",)),
    A.SYSTEM_INFO: Words(
        subjects=(
            "system info(?:rmation)?",
            "os version",
            "windows version",
            "macos version",
            "uptime",
            "operating system",
            "(?:computer|pc|laptop|machine|system) been (?:up|on|running)",
            # one fact of the overview: the host name, the system's version, the time zone, the architecture
            "host ?name",
            "(?:computer|pc|laptop|machine)(?:'s| s)? name",
            "(?:this|my) (?:computer|pc|laptop|machine|mac) called",
            "version of (?:windows|macos|mac os|linux|ubuntu)",
            "os is (?:this|my) (?:computer|pc|laptop|machine|mac)",
            "(?:last|latest) (?:restart|reboot|boot)",
            "time ?zone",
            "(?:64|32) ?bit",
            "architecture",
        ),
        topics=("specs",),
        asked="What's my system information?",
    ),
    A.SYSTEM_HEALTH_REVIEW: Words(
        subjects=(
            "system health",
            "health check",
            "(?:computer|pc|laptop|machine|system) (?:is )?healthy",
            "wrong with (?:my|the|this) (?:computer|pc|laptop|machine|system)",
        ),
        asked="How is my system health?",
    ),
    A.DISK_USAGE: Words(
        subjects=(
            "disk space",
            "free space",
            "storage space",
            "drive space",
            "space left",
            "space (?:on|in) (?:my|the|this) (?:disk|drive|computer|laptop|machine)",
        ),
        topics=("disk", "drive", "storage", "space"),
        asked="How much free disk space is there?",
    ),
    A.MEMORY_USAGE: Words(
        subjects=("memory usage", "memory use", "ram usage", "free memory", "free ram"),
        topics=("memory", "ram"),
        asked="How much memory is in use right now?",
    ),
    A.CPU_USAGE: Words(
        subjects=(
            "cpu usage",
            "cpu load",
            "cpu use",
            "processor usage",
            "how busy is (?:the|my) (?:cpu|processor)",
            "(?:cpu|processor) (?:is )?maxed(?: out)?",
        ),
        topics=("cpu", "processor"),
        asked="What's the CPU usage?",
    ),
    A.NETWORK_INFO: Words(
        subjects=(
            "ip address",
            "ipv[46] address",
            "network info(?:rmation)?",
            "internet connection",
            "network am i on",
        ),
        topics=("network", "internet", "ip"),
        asked="What's my IP address?",
    ),
    A.BATTERY_INFO: Words(
        subjects=("battery", "charge level", "charge (?:is )?left", "(?:laptop|computer|pc|machine) (?:is )?charging"),
        asked="How much battery do I have left?",
    ),
    # power and schedules
    A.POWER_SHUTDOWN: Words(verbs=("shut", "shutdown")),
    A.POWER_RESTART: Words(verbs=("restart", "reboot")),
    A.POWER_SLEEP: Words(verbs=("sleep", "suspend", "hibernate")),
    A.POWER_LOCK: Words(verbs=("lock",)),
    A.POWER_LOGOUT: Words(verbs=("logout",)),
    A.SCHEDULE_CREATE: Words(verbs=("schedule",)),
    A.SCHEDULE_LIST: Words(
        subjects=("scheduled tasks", "scheduled jobs", "tasks are scheduled", "scheduled to run"),
        topics=("schedule",),
        asked="What scheduled tasks do I have?",
    ),
    A.SCHEDULE_DELETE: Words(verbs=("delete", "remove", "unschedule")),
    A.ENV_GET: Words(
        subjects=("environment variables?", "path variable", "env var(?:iable)?s?", "my path"),
        asked="What is my PATH variable right now?",
    ),
    A.ENV_SET: Words(verbs=("set",)),
    A.ENV_LIST: Words(
        subjects=("environment variables",), asked="Which environment variables are set on this computer?"
    ),
    # windows, sound, display
    A.WINDOW_LIST: Words(
        subjects=("open windows", "windows (?:are |that are |do i have )?open"), asked="What windows are open?"
    ),
    A.WINDOW_FOCUS: Words(verbs=("focus", "switch")),
    A.WINDOW_CLOSE: Words(verbs=("close",)),
    A.WINDOW_MINIMIZE: Words(verbs=("minimize", "minimise")),
    A.WINDOW_MAXIMIZE: Words(verbs=("maximize", "maximise")),
    A.VOLUME_GET: Words(
        subjects=(
            "volume level",
            "sound level",
            "how loud is (?:my|the) (?:sound|volume|audio|speakers?|computer|laptop|music)",
            "the volume at",
        ),
        topics=("volume",),
        asked="What is the volume level? Just tell me, don't change it.",
    ),
    A.VOLUME_SET: Words(verbs=("set", "turn")),
    A.VOLUME_MUTE: Words(verbs=("mute", "unmute")),
    A.BRIGHTNESS_GET: Words(
        subjects=(
            "brightness",
            "how bright is (?:my|the|this) (?:screen|display|monitor|laptop|computer)",
            "(?:screen|display) (?:is )?dimmed",
        ),
        asked="What is the display brightness? Just tell me, don't change it.",
    ),
    A.BRIGHTNESS_SET: Words(verbs=("dim", "brighten")),
    A.SCREENSHOT: Words(verbs=("screenshot",)),
    # network, disks, users
    A.WIFI_LIST: Words(
        subjects=("wi ?fi networks", "wireless networks"), topics=("wi ?fi",), asked="What wifi networks are there?"
    ),
    A.WIFI_CONNECT: Words(verbs=("connect",)),
    A.WIFI_DISCONNECT: Words(verbs=("disconnect",)),
    A.DISK_LIST: Words(
        subjects=(
            "connected drives",
            "mounted drives",
            "usb drives?",
            "drives (?:are )?(?:connected|mounted|plugged in)",
        ),
        asked="What drives are connected?",
    ),
    A.DISK_MOUNT: Words(verbs=("mount",)),
    A.DISK_UNMOUNT: Words(verbs=("unmount", "eject")),
    A.USER_LIST: Words(
        subjects=(
            "user accounts",
            "users are on (?:this|my) (?:pc|computer|laptop|machine|mac)",
            "accounts? on (?:this|my) (?:pc|computer|laptop|machine|mac)",
        ),
        asked="Which user accounts are on this computer?",
    ),
    A.USER_INFO: Words(
        subjects=(
            "my user account",
            "am i (?:an )?admin(?:istrator)?",
            "account am i using",
            "logged in as",
            "user am i",
            "my user ?name",
            "groups am i in",
            "admin(?:istrator)? rights",
        ),
        asked="What groups is my user account in?",
    ),
    A.DOWNLOAD_FILE: Words(verbs=("download",)),
    A.REGISTRY_READ: Words(
        subjects=("registry (?:key|value)s?",),
        topics=("registry",),
        asked="What is the registry value for my wallpaper?",
    ),
    A.REGISTRY_WRITE: Words(verbs=("set",)),
    A.LOG_ANALYZE: Words(
        subjects=("log file", "error log", "last \\w+ lines"),
        topics=("logs?",),
        asked="What errors are in the log file C:\\work\\app.log?",
    ),
    # mouse, keyboard
    A.MOUSE_CLICK: Words(verbs=("click",)),
    A.MOUSE_DOUBLE_CLICK: Words(verbs=("click",)),
    A.MOUSE_RIGHT_CLICK: Words(verbs=("click",)),
    A.MOUSE_MOVE: Words(verbs=("move",)),
    A.MOUSE_DRAG: Words(verbs=("drag",)),
    A.MOUSE_SCROLL: Words(verbs=("scroll",)),
    A.MOUSE_POSITION: Words(
        subjects=("mouse position", "cursor position", "mouse pointer", "where(?:'s| s| is) (?:my|the) mouse"),
        asked="What's the mouse position?",
    ),
    A.KEYBOARD_TYPE: Words(verbs=("type",)),
    A.KEYBOARD_PRESS: Words(verbs=("press",)),
    A.KEYBOARD_HOTKEY: Words(verbs=("press",)),
    A.KEYBOARD_HOLD: Words(verbs=("hold",)),
    # browser
    A.BROWSER_NAVIGATE: Words(verbs=("go", "navigate")),
    A.BROWSER_CLICK: Words(verbs=("click",)),
    A.BROWSER_CLICK_TEXT: Words(verbs=("click",)),
    A.BROWSER_TYPE: Words(verbs=("type",)),
    A.BROWSER_SELECT: Words(verbs=("select", "choose", "pick")),
    A.BROWSER_HOVER: Words(verbs=("hover",)),
    A.BROWSER_SCROLL: Words(verbs=("scroll",)),
    A.BROWSER_EXTRACT: Words(
        topics=("page text",), verbs=("extract",), asked="What's the page text of https://example.com?"
    ),
    A.BROWSER_EXTRACT_TABLE: Words(
        topics=("table",), verbs=("extract",), asked="What's in the table on https://example.com?"
    ),
    A.BROWSER_EXTRACT_LINKS: Words(subjects=("links on",), asked="What are the links on https://example.com?"),
    A.BROWSER_SCREENSHOT: Words(verbs=("screenshot",)),
    A.BROWSER_FILL_FORM: Words(verbs=("fill",)),
    A.BROWSER_NEW_TAB: Words(topics=("new tab",), verbs=("open",)),
    A.BROWSER_CLOSE_TAB: Words(verbs=("close",)),
    A.BROWSER_LIST_TABS: Words(
        subjects=("open tabs", "browser tabs", "tabs (?:do i have )?open"), asked="Which browser tabs are open?"
    ),
    A.BROWSER_SWITCH_TAB: Words(verbs=("switch",)),
    A.BROWSER_BACK: Words(verbs=("go",)),
    A.BROWSER_FORWARD: Words(verbs=("go",)),
    A.BROWSER_REFRESH: Words(verbs=("refresh", "reload")),
    A.BROWSER_CLOSE: Words(verbs=("close",)),
    A.BROWSER_PAGE_INFO: Words(
        subjects=("page title", "page am i on"), asked="What's the page title of https://example.com?"
    ),
    # triggers
    A.TRIGGER_CREATE: Words(subjects=("trigger", "watcher"), verbs=("watch", "monitor")),
    A.TRIGGER_LIST: Words(
        subjects=("my triggers", "my watchers", "(?:triggers|watchers) (?:are )?(?:active|running|set up)"),
        topics=("triggers",),
        asked="What triggers do I have?",
    ),
    A.TRIGGER_DELETE: Words(verbs=("delete", "remove")),
    A.TRIGGER_START: Words(verbs=("start", "resume")),
    A.TRIGGER_STOP: Words(verbs=("stop", "pause")),
    # code
    A.CODE_EXECUTE: Words(verbs=("run", "execute")),
    A.CODE_GENERATE_AND_RUN: Words(verbs=("compute", "calculate")),
    # calendar
    A.CALENDAR_PARSE: Words(verbs=("parse",)),
    A.CALENDAR_SYNC: Words(verbs=("sync",)),
    A.CALENDAR_CREATE_EVENT: Words(verbs=("schedule", "book")),
    A.CALENDAR_LIST_EVENTS: Words(
        # Not "meeting" alone: "what did my boss say in this morning's meeting?" is about a meeting, not the calendar.
        subjects=("my calendar",),
        topics=("calendar",),
        asked="What meetings are on my calendar tomorrow?",
    ),
    A.CALENDAR_DELETE_EVENT: Words(verbs=("delete", "remove")),
    A.CALENDAR_FETCH: Words(topics=("calendar",), verbs=("sync",), asked="What's on my calendar today?"),
    A.CALENDAR_RECONCILE: Words(verbs=("reconcile",)),
    # web, services, mail
    A.API_REQUEST: Words(
        topics=("http (?:get|request|post)", "api", "status code"),
        verbs=("fetch", "get", "request", "query", "call"),
        asked="Fetch https://example.com with an HTTP GET request and tell me the page title",
    ),
    A.API_GITHUB: Words(subjects=("github", "pull requests?"), verbs=("open", "create")),
    A.API_SEND_EMAIL: Words(topics=("email", "e ?mail"), verbs=("send", "email", "mail")),
    A.API_WEBHOOK: Words(subjects=("webhook",), verbs=("post", "send", "call")),
    A.API_SLACK: Words(subjects=("slack",), verbs=("post", "send", "message")),
    A.API_DISCORD: Words(subjects=("discord",), verbs=("post", "send", "message")),
    A.API_SCRAPE: Words(
        topics=("css selector",),
        verbs=("scrape", "extract"),
        asked="What is the text of the elements matching the CSS selector '.price' on https://example.com?",
    ),
    A.WORKSPACE_INDEX: Words(verbs=("index", "reindex")),
    A.WORKSPACE_SEARCH: Words(verbs=("search",)),
    A.EMAIL_FETCH: Words(
        subjects=("my inbox", "(?:new|unread) (?:emails?|mails?|messages)"),
        topics=("emails?", "inbox"),
        asked="Do I have any unread emails?",
    ),
    A.EMAIL_SUMMARIZE: Words(verbs=("summarize", "summarise")),
    A.EMAIL_REPLY: Words(verbs=("reply",)),
    A.SSH_COMMAND: Words(subjects=("ssh",), verbs=("ssh",)),
    A.SSH_SCRIPT: Words(subjects=("ssh",), verbs=("ssh",)),
    A.MCP_LIST_TOOLS: Words(subjects=("mcp tools", "mcp servers"), asked="What MCP tools do I have?"),
}

# Action types a person does not ask for by name, each with the reason. Moving one to NAMED is a decision, not an
# oversight: the contract test fails when an action type is in neither.
UNNAMED: dict[ActionType, str] = {
    A.GNOME_SETTING_READ: "a desktop setting is named by the setting itself; the planner maps it to a key",
    A.DBUS_CALL: "an internal call the planner chooses; nobody asks for D-Bus",
    A.REQUEST_HUMAN_HELP: "Heliox asks the person; the person never asks for it",
    A.SCREEN_OCR: "the talk model is given the screen itself (pilot.conversation.context.Screen)",
    A.SCREEN_FIND_TEXT: "the talk model is given the screen itself",
    A.SCREEN_ANALYZE: "the talk model is given the screen itself",
    A.SCREEN_ELEMENT_MAP: "a step inside a desktop plan",
    A.SCREEN_ACCESSIBILITY_SNAPSHOT: "a step inside a desktop plan",
    A.SCREEN_DETECT_ELEMENTS: "a step inside a desktop plan",
    A.BROWSER_EXECUTE_JS: "a step inside a browser plan; nobody asks for script by name",
    A.BROWSER_SNAPSHOT: "a step inside a browser plan",
    A.BROWSER_WAIT: "a step inside a browser plan",
    A.WASM_CALL: "a plugin's own words name it; the planner maps them",
    A.PLUGIN_CALL: "a plugin's own words name it; the planner maps them",
    A.SKILL_RUN: "a skill's own words name it; the planner maps them",
    A.MCP_CALL_TOOL: "a tool's own words name it; the planner maps them",
}

# Verbs that ask for an action only when the sentence also names something an action works on (a subject or topic
# of any action, a path, a file name or a web address): "watch the file ...", "post it to Slack", "get the text from
# https://..." are requests; "watch out", "get some rest", "stop worrying" are not.
ANCHORED_VERBS = frozenset(
    {
        "watch",
        "monitor",
        "post",
        "get",
        "fetch",
        "request",
        "query",
        "call",
        "compute",
        "calculate",
        "stop",
        "start",
        "resume",
        "pause",
        "hold",
        "sleep",
        "enable",
        "disable",
        "connect",
        "disconnect",
        "select",
        "choose",
        "pick",
        "focus",
        "message",
        "mail",
        "read",
        "list",
        "branch",
        "diff",
        "duplicate",
        "fill",
        "hover",
        "index",
        "drag",
        "resolve",
        "checkout",
        "sync",
        "reconcile",
        "parse",
        "notify",
        "terminate",
    }
)


def _alternation(fragments: list[str]) -> re.Pattern[str]:
    ordered = sorted(set(fragments), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(ordered) + r")\b")


# A read action's subjects and topics make a question about this computer; any action's subjects anchor a verb.
STRONG = _alternation([s for kind, words in NAMED.items() if kind in READ_ONLY_ACTIONS for s in words.subjects])
WEAK = _alternation([t for kind, words in NAMED.items() if kind in READ_ONLY_ACTIONS for t in words.topics])
ANCHOR = _alternation([s for words in NAMED.values() for s in words.subjects])
VERBS = frozenset(v for words in NAMED.values() for v in words.verbs)


def read_subject(words: str) -> tuple[bool, bool]:
    """(names a read subject plainly, names a read topic) in normalized words."""
    return bool(STRONG.search(words)), bool(WEAK.search(words))


def names_anything(words: str) -> bool:
    """A subject of any action (a read one, or "Slack", "a webhook", "a trigger") in normalized words."""
    return bool(ANCHOR.search(words))


# Words that ask for a reading without changing what is read: the question's frame, here and now, and what a reading
# is called. Anything else left once the action's own subjects and topics are taken out (a temperature, a graphics
# card, a named app or drive, "per core", "yesterday", "average", "than") is a qualifier the read does not cover.
_FRAME = frozenset(
    [
        "what",
        "what's",
        "whats",
        "how",
        "busy",
        "much",
        "many",
        "is",
        "are",
        "was",
        "the",
        "my",
        "me",
        "i",
        "i'm",
        "do",
        "does",
        "have",
        "has",
        "got",
        "left",
        "right",
        "now",
        "currently",
        "current",
        "at",
        "the",
        "moment",
        "this",
        "computer",
        "laptop",
        "machine",
        "pc",
        "mac",
        "there",
        "of",
        "on",
        "in",
        "a",
        "an",
        "it",
        "tell",
        "show",
        "check",
        "please",
        "just",
        "free",
        "used",
        "use",
        "usage",
        "level",
        "remaining",
        "available",
        "space",
        "percent",
        "percentage",
        "status",
        "still",
        "much",
        "amount",
        "total",
        "how's",
        "can",
        "you",
        "could",
        "would",
        "let",
        "know",
        "see",
        "give",
        "s",
        "quick",
        "quickly",
        "whole",
        "full",
        "charge",
        "charged",
        "reading",
    ]
)


def asks_only_for(kind: ActionType, words: str) -> bool:
    """Whether normalized ``words`` ask for exactly what read action ``kind`` returns, with no qualifier it does not
    take (the quick road runs it only then; anything more goes to the planner)."""
    entry = NAMED.get(kind)
    if entry is None:
        return False
    rest = _alternation([*entry.subjects, *entry.topics]).sub(" ", words)
    return all(word in _FRAME for word in rest.split())


def asked_for(kind: ActionType) -> str:
    return NAMED[kind].asked if kind in NAMED else ""


__all__ = [
    "ANCHORED_VERBS",
    "NAMED",
    "UNNAMED",
    "VERBS",
    "Words",
    "asked_for",
    "names_anything",
    "read_subject",
]
