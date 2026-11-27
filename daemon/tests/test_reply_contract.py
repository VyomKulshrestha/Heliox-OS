"""Every action type has a sentence to show and one to say, and raw data is neither.

For each action type a representative handler result (the handlers' own shapes, and the shapes recorded in the live
run of 26 November: git's JSON with a repr's eight backslashes, a file write with its sha256, a trigger's JSON, an HTTP
answer) goes through the reply path. The test fails when the shown text holds raw JSON or doubled backslashes, when
the spoken text holds a hash, a path separator, a brace or bracket, is cut inside a word, or when either is empty. An
action type added without an entry here, or without a sentence in pilot.result_sentences, fails it too.
"""

from __future__ import annotations

import json

import pytest

from pilot.actions import (
    AccessibilitySnapshotParams,
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    BrightnessParams,
    BrowserParams,
    CalendarParams,
    ClipboardParams,
    CodeExecParams,
    DBusParams,
    DiskManageParams,
    DownloadParams,
    ElementDetectionParams,
    EmailParams,
    EmptyParams,
    EnvParams,
    FileIntelParams,
    FileParams,
    GitParams,
    GitResolveParams,
    GnomeSettingParams,
    HumanHelpParams,
    KeyboardParams,
    LogAnalyzeParams,
    McpToolParams,
    MouseParams,
    NotifyParams,
    OpenApplicationParams,
    OpenUrlParams,
    PackageParams,
    PowerParams,
    ProcessParams,
    PtyExecParams,
    RegistryParams,
    ScheduleParams,
    ScreenshotParams,
    ScreenVisionParams,
    ServiceParams,
    ShellCommandParams,
    ShellScriptParams,
    SkillRunParams,
    SshCommandParams,
    SshScriptParams,
    SystemInfoParams,
    TriggerParams,
    VerificationResult,
    VolumeParams,
    WasmCallParams,
    WifiParams,
    WindowParams,
    WorkspaceParams,
)
from pilot.reply_rules import SPOKEN_LIMIT, ear_faults, for_ear, screen_faults
from pilot.response_contract import failure_reply, success_reply
from pilot.result_sentences import SENTENCES, step_said
from pilot.task_reply import with_reply

W = r"C:\Users\me\work"
NOTE = W + r"\notes\note.txt"
REPO = W + r"\repo"
SHA = "9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c9f2c"
LONG_TEXT = (
    "Dear team, the annual report review moved to Thursday afternoon because the finance office needs another day "
    "to reconcile the travel claims. Please bring your updated forecasts, the list of open purchase orders and any "
    "questions about the new approval limits. Thank you all for the quick turnaround last month."
)
EXAMPLE_HTML = (
    "<!doctype html><html><head><title>Example Domain</title><style>body{font:16px/1.6 system-ui}</style></head>"
    "<body><h1>Example Domain</h1><p>This domain is for use in documentation examples without needing permission."
    "</p></body></html>"
)


def _git(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# The run's git_stage output: the handler's repr of a list inside JSON, eight backslashes per separator when shown.
STAGED = REPO + r"\README.md"
RUN_STAGE = _git({"success": True, "output": f"Staged: [{STAGED!r}]", "staged_files": [STAGED]})
RUN_TRIGGER = json.dumps(
    {
        "id": "t0a1b2c3",
        "name": "watch_inbox_file",
        "trigger_type": "file_modified",
        "condition": {"path": W + r"\trigger\watched.txt"},
        "action_command": "echo file touched >> fired.log",
    },
    indent=2,
)
RUN_HTTP = json.dumps(
    {
        "status": 200,
        "url": "https://example.com",
        "headers": {"content-type": "text/html; charset=utf-8", "server": "cloudflare"},
        "body": EXAMPLE_HTML,
    },
    indent=2,
)

# action type: (parameters, target, the handler's output)
CASES: dict[ActionType, tuple[object, str, str]] = {
    ActionType.FILE_READ: (FileParams(path=NOTE), NOTE, "Water the ferns and call Sam on Fridays."),
    ActionType.FILE_WRITE: (
        FileParams(path=NOTE, content="Water the ferns and call Sam on Fridays."),
        NOTE,
        f"Written 40 bytes to {NOTE} (sha256 {SHA}, 40 bytes on disk)",
    ),
    ActionType.FILE_DELETE: (FileParams(path=NOTE), NOTE, f"Deleted file: {NOTE}"),
    ActionType.FILE_MOVE: (
        FileParams(path=NOTE, destination=W + r"\archive\note.txt"),
        NOTE,
        f"Moved {NOTE} -> {W}\\archive\\note.txt",
    ),
    ActionType.FILE_COPY: (
        FileParams(path=NOTE, destination=W + r"\notes\note_copy.txt"),
        NOTE,
        f"Copied {NOTE} -> {W}\\notes\\note_copy.txt",
    ),
    ActionType.FILE_LIST: (
        FileParams(path=W + r"\notes"),
        W + r"\notes",
        "[f] note.txt  (40 bytes)\n[f] todo.md  (120 bytes)\n[d] old  (0 bytes)",
    ),
    ActionType.FILE_SEARCH: (
        FileParams(path=W, pattern="*.txt"),
        W,
        "[f] notes\\note.txt  (40 bytes)\n[f] archive\\old.txt  (12 bytes)",
    ),
    ActionType.DIRECTORY_SUMMARY: (
        FileParams(path=W),
        W,
        "work/\n|-- notes/\n|   `-- note.txt (40 B)\n`-- readme.md (1.2 KB)",
    ),
    ActionType.DIRECTORY_SIZE: (FileParams(path=W), W, f"{W}: 1.2 MB across 34 files"),
    ActionType.FILE_HASH: (FileParams(path=NOTE), NOTE, f"sha256:{SHA}  {NOTE}"),
    ActionType.FILE_COMPARE: (
        FileParams(path=NOTE, destination=W + r"\notes\note_copy.txt"),
        NOTE,
        f"Identical: {NOTE} <> {W}\\notes\\note_copy.txt\nsha256: {SHA} <> {SHA}",
    ),
    ActionType.FILE_PERMISSIONS: (FileParams(path=NOTE, permissions="644"), NOTE, f"Set permissions on {NOTE} to 644"),
    ActionType.FILE_PARSE: (
        FileIntelParams(path=W + r"\budget.pdf"),
        W + r"\budget.pdf",
        "PDF: 3 pages\n\nQuarterly budget summary for the travel team.",
    ),
    ActionType.FILE_SEARCH_CONTENT: (
        FileIntelParams(directory=W + r"\workspace", search_text="annual report"),
        W + r"\workspace",
        json.dumps(
            {
                "matches": [
                    {"file": W + r"\workspace\budget.txt", "line": 3, "text": "the annual report is due"},
                    {"file": W + r"\workspace\plan.md", "line": 9, "text": "see the annual report"},
                ],
                "count": 2,
            },
            indent=2,
        ),
    ),
    ActionType.DOWNLOAD_FILE: (
        DownloadParams(url="https://example.com/files/a.zip?token=x", output_path=W + r"\a.zip"),
        "https://example.com/files/a.zip",
        f"Downloaded https://example.com/files/a.zip -> {W}\\a.zip (1,048,576 bytes)",
    ),
    ActionType.GIT_RESOLVE: (
        GitResolveParams(path=REPO + r"\app.py"),
        REPO + r"\app.py",
        f"Successfully resolved git conflict in {REPO}\\app.py",
    ),
    ActionType.GIT_STATUS: (
        GitParams(repo_path=REPO),
        REPO,
        _git(
            {
                "success": True,
                "output": "On branch: main\nUnstaged (1): README.md\nUntracked (1): notes.txt",
                "staged": [],
                "unstaged": ["README.md"],
                "untracked": ["notes.txt"],
                "branch": "main",
            }
        ),
    ),
    ActionType.GIT_DIFF: (
        GitParams(repo_path=REPO),
        REPO,
        _git(
            {
                "success": True,
                "output": "diff --git a/README.md b/README.md\nindex 1111111..2222222 100644\n--- a/README.md\n"
                "+++ b/README.md\n@@ -1 +1,3 @@\n # demo repo\n+\n+A line added for the diff.",
                "diff": "diff --git a/README.md b/README.md\nindex 1111111..2222222 100644\n--- a/README.md\n"
                "+++ b/README.md\n@@ -1 +1,3 @@\n # demo repo\n+\n+A line added for the diff.",
            }
        ),
    ),
    ActionType.GIT_LOG: (
        GitParams(repo_path=REPO),
        REPO,
        _git(
            {
                "success": True,
                "output": "1a2b3c4 Fix the intro (2 hours ago)\n9e8d7c6 First commit (3 days ago)",
                "commits": [
                    {
                        "hash": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d",
                        "short": "1a2b3c4",
                        "author": "Ann",
                        "when": "2 hours ago",
                        "message": "Fix the intro",
                    },
                    {
                        "hash": "9e8d7c6b5a493827160f1e2d3c4b5a6978695a4b",
                        "short": "9e8d7c6",
                        "author": "Ann",
                        "when": "3 days ago",
                        "message": "First commit",
                    },
                ],
            }
        ),
    ),  # fmt: skip
    ActionType.GIT_BRANCH: (
        GitParams(repo_path=REPO, name="topic-branch", create=True),
        REPO,
        _git({"success": True, "output": "Switched to a new branch 'topic-branch'", "branch": "topic-branch"}),
    ),
    ActionType.GIT_STAGE: (GitParams(repo_path=REPO, files=[STAGED]), STAGED, RUN_STAGE),
    ActionType.GIT_COMMIT: (
        GitParams(repo_path=REPO, message="Fix the intro"),
        REPO,
        _git(
            {
                "success": True,
                "output": "Committed as 1a2b3c4: Fix the intro",
                "commit_message": "Fix the intro",
                "files_committed": 2,
                "commit_hash": "1a2b3c4",
                "commit_sha": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d",
                "head_before": "9e8d7c6b5a493827160f1e2d3c4b5a6978695a4b",
                "branch": "main",
            }
        ),
    ),
    ActionType.GIT_PUSH: (
        GitParams(repo_path=REPO, remote="origin"),
        REPO,
        _git(
            {
                "success": True,
                "output": f"To {W}\\remote.git\n   9e8d7c6..1a2b3c4  main -> main",
                "remote": "origin",
                "branch": "main",
                "pushed_sha": "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d",
                "remote_ref": "refs/heads/main",
            }
        ),
    ),
    ActionType.PACKAGE_INSTALL: (PackageParams(name="vlc"), "vlc", "Installed vlc 3.0.21"),
    ActionType.PACKAGE_REMOVE: (PackageParams(name="vlc"), "vlc", "Removed vlc"),
    ActionType.PACKAGE_UPDATE: (
        PackageParams(name=""),
        "all",
        "[UPGRADE] Git.Git 2.45.1 -> 2.46.0\n[UPGRADE] VideoLAN.VLC 3.0.20 -> 3.0.21\nSuccessfully installed",
    ),
    ActionType.PACKAGE_SEARCH: (
        PackageParams(name="vlc"),
        "vlc",
        "VLC media player  VideoLAN.VLC  3.0.21\nVLC nightly  VideoLAN.VLC.Nightly  4.0.0",
    ),
    ActionType.SERVICE_START: (ServiceParams(name="nginx"), "nginx", "Started nginx"),
    ActionType.SERVICE_STOP: (ServiceParams(name="nginx"), "nginx", "Stopped nginx"),
    ActionType.SERVICE_RESTART: (ServiceParams(name="nginx"), "nginx", "Restarted nginx"),
    ActionType.SERVICE_ENABLE: (ServiceParams(name="nginx"), "nginx", "Enabled nginx"),
    ActionType.SERVICE_DISABLE: (ServiceParams(name="nginx"), "nginx", "Disabled nginx"),
    ActionType.SERVICE_STATUS: (
        ServiceParams(name="nginx"),
        "nginx",
        "nginx.service - A high performance web server\n   Active: active (running) since Mon",
    ),
    ActionType.GNOME_SETTING_READ: (
        GnomeSettingParams(schema_id="org.gnome.desktop.interface", key="color-scheme"),
        "color-scheme",
        "'prefer-dark'",
    ),
    ActionType.GNOME_SETTING_WRITE: (
        GnomeSettingParams(schema_id="org.gnome.desktop.interface", key="color-scheme", value="'prefer-dark'"),
        "color-scheme",
        "Set org.gnome.desktop.interface color-scheme to 'prefer-dark'",
    ),
    ActionType.DBUS_CALL: (
        DBusParams(
            service="org.freedesktop.Notifications",
            object_path="/org/freedesktop/Notifications",
            interface="org.freedesktop.Notifications",
            method="GetServerInformation",
        ),
        "org.freedesktop.Notifications",
        "('gnome-shell', 'GNOME', '45.0', '1.2')",
    ),
    ActionType.SHELL_COMMAND: (ShellCommandParams(command="git --version"), "git", "git version 2.45.1.windows.1"),
    ActionType.SHELL_SCRIPT: (
        ShellScriptParams(script="ls -la /var/log"),
        "script",
        "total 12\n-rw-r--r-- 1 root root 1200 Nov 26 18:41 /var/log/syslog\n"
        "-rw-r--r-- 1 root root 300 Nov 26 18:40 /var/log/auth.log\nfinished cleaning the cache folder",
    ),
    ActionType.PTY_EXEC: (PtyExecParams(command="ls"), "ls", "a.txt\nb.txt"),
    ActionType.SSH_COMMAND: (
        SshCommandParams(host="pi.local", command="uptime"),
        "pi.local",
        " 10:02:01 up 3 days,  2 users,  load average: 0.10, 0.20, 0.15",
    ),
    ActionType.SSH_SCRIPT: (
        SshScriptParams(host="pi.local", script="df -h /"),
        "pi.local",
        "/dev/root 29G 12G 16G 43% /",
    ),
    ActionType.CODE_EXECUTE: (
        CodeExecParams(code="print(sum(range(10)))", language="python"),
        "python",
        "45",
    ),
    ActionType.CODE_GENERATE_AND_RUN: (
        CodeExecParams(code="", task_description="count the words in note.txt"),
        "count words",
        '{"words": 8, "file": "C:\\\\Users\\\\me\\\\work\\\\notes\\\\note.txt"}',
    ),
    ActionType.OPEN_URL: (
        OpenUrlParams(url="https://www.example.com/docs?page=2"),
        "https://www.example.com/docs?page=2",
        "Opened https://www.example.com/docs?page=2 in default browser",
    ),
    ActionType.OPEN_APPLICATION: (
        OpenApplicationParams(name="notepad"),
        "notepad",
        "Launched Notepad (resolved via PATH).",
    ),
    ActionType.NOTIFY: (NotifyParams(summary="Tea", body="It's ready"), "Tea", "Notification sent: Tea"),
    ActionType.REQUEST_HUMAN_HELP: (
        HumanHelpParams(help_needed="Sign in to the bank site", instructions="Use your own password"),
        "bank",
        "Asked the person for help: Sign in to the bank site. Heliox will continue after they hand control back.",
    ),
    ActionType.PROCESS_LIST: (
        ProcessParams(),
        "",
        "PID=  4321  CPU= 12.5%  MEM=  3.1%  STATUS=running       chrome.exe\n"
        "PID=   812  CPU=  0.4%  MEM=  0.9%  STATUS=running       explorer.exe\n"
        "PID=  9001  CPU=  1.1%  MEM=  1.2%  STATUS=sleeping      Code.exe",
    ),
    ActionType.PROCESS_KILL: (
        ProcessParams(name="notepad"),
        "notepad",
        "Sent SIGTERM to process notepad.exe (PID 4321)",
    ),
    ActionType.PROCESS_INFO: (
        ProcessParams(pid=4321),
        "4321",
        f"Process info for PID 4321:\n  pid: 4321\n  name: notepad.exe\n  exe: C:\\Windows\\notepad.exe\n"
        "  status: running",
    ),
    ActionType.CLIPBOARD_READ: (ClipboardParams(), "", LONG_TEXT),
    ActionType.CLIPBOARD_WRITE: (ClipboardParams(content="hello"), "", "Copied 5 characters to clipboard"),
    ActionType.SYSTEM_INFO: (
        SystemInfoParams(categories=["all"]),
        "system",
        "=== Operating System ===\n  release: 11\n=== CPU ===\n  Average usage: 16.3%\n"
        "=== Memory ===\n  Total: 16.0 GB\n  Used:  10.7 GB (68.0%)\n=== Battery ===\n  Charge: 80%",
    ),
    ActionType.SYSTEM_HEALTH_REVIEW: (
        SystemInfoParams(),
        "system",
        "Two things need attention.\n- Drive C is 91% full.\n- 3 startup apps slow the start.",
    ),
    ActionType.DISK_USAGE: (
        SystemInfoParams(),
        "disk",
        "=== Disk ===\n  C:\\ (C:\\) — 476.0 GB total, 356.0 GB used (74.8%), fstype=NTFS",
    ),
    ActionType.MEMORY_USAGE: (
        SystemInfoParams(),
        "memory",
        "=== Memory ===\n  Total: 16.0 GB\n  Used:  10.7 GB (68.0%)\n  Free:  5.0 GB\n  Swap Total: 2.0 GB\n"
        "  Swap Used:  0.1 GB (5.0%)",
    ),
    ActionType.CPU_USAGE: (SystemInfoParams(), "cpu", "=== CPU ===\n  Cores: 8\n  Average usage: 16.3%"),
    ActionType.NETWORK_INFO: (
        SystemInfoParams(),
        "network",
        "=== Network ===\n  Wi-Fi (UP):\n    IPv4: 192.168.1.5\n  Ethernet (DOWN):",
    ),
    ActionType.BATTERY_INFO: (
        SystemInfoParams(),
        "battery",
        "=== Battery ===\n  Charge: 80%\n  Status: Plugged in\n  Time remaining: N/A",
    ),
    ActionType.POWER_SHUTDOWN: (PowerParams(delay_seconds=60), "", "System shutdown initiated (delay: 60s)"),
    ActionType.POWER_RESTART: (PowerParams(), "", "System restart initiated"),
    ActionType.POWER_SLEEP: (
        PowerParams(),
        "",
        "Sleep was requested; whether the computer slept cannot be checked while it sleeps",
    ),
    ActionType.POWER_LOCK: (PowerParams(), "", "Screen locked"),
    ActionType.POWER_LOGOUT: (PowerParams(), "", "User logout initiated"),
    ActionType.SCHEDULE_CREATE: (
        ScheduleParams(name="backup", command="robocopy a b", schedule="daily 09:00"),
        "backup",
        r"Created scheduled task: backup (C:\Users\me\AppData\Roaming\heliox\backup.xml)",
    ),
    ActionType.SCHEDULE_LIST: (ScheduleParams(), "", "Pilot tasks:\nbackup  daily 09:00\ncleanup  weekly"),
    ActionType.SCHEDULE_DELETE: (ScheduleParams(name="backup"), "backup", "Deleted scheduled task: backup"),
    ActionType.ENV_GET: (EnvParams(name="JAVA_HOME"), "JAVA_HOME", r"JAVA_HOME=C:\Program Files\Java\jdk-21"),
    ActionType.ENV_SET: (
        EnvParams(name="EDITOR", value="code"),
        "EDITOR",
        "Set EDITOR=code (persistent, user scope)",
    ),
    ActionType.ENV_LIST: (EnvParams(), "", "EDITOR=code\nHOME=C:\\Users\\me\nPATH=C:\\Windows;C:\\Tools"),
    ActionType.WINDOW_LIST: (WindowParams(), "", "Notepad - note.txt\nInbox - Mail\nHeliox"),
    ActionType.WINDOW_FOCUS: (WindowParams(title="Notepad"), "Notepad", "Focused window: Notepad (hwnd 0x2a01f4)"),
    ActionType.WINDOW_CLOSE: (WindowParams(title="Notepad"), "Notepad", "Closed window: Notepad"),
    ActionType.WINDOW_MINIMIZE: (WindowParams(title="Notepad"), "Notepad", "Minimized Notepad"),
    ActionType.WINDOW_MAXIMIZE: (WindowParams(title="Notepad"), "Notepad", "Maximized Notepad"),
    ActionType.VOLUME_GET: (VolumeParams(), "", "Volume is at 40."),
    ActionType.VOLUME_SET: (VolumeParams(level=30), "", "Volume is at 30."),
    ActionType.VOLUME_MUTE: (VolumeParams(mute=True), "", "Muted."),
    ActionType.BRIGHTNESS_GET: (BrightnessParams(), "", "Brightness: 70%"),
    ActionType.BRIGHTNESS_SET: (BrightnessParams(level=50), "", "Brightness set to 50%"),
    ActionType.SCREENSHOT: (
        ScreenshotParams(output_path=W + r"\shots\screen.png"),
        "",
        f"Screenshot saved to {W}\\shots\\screen.png",
    ),
    ActionType.WIFI_LIST: (WifiParams(), "", "HomeNet  80%  WPA2\nCafe-Guest  40%  open"),
    ActionType.WIFI_CONNECT: (WifiParams(ssid="HomeNet"), "HomeNet", "Connected to WiFi: HomeNet"),
    ActionType.WIFI_DISCONNECT: (WifiParams(), "", "Disconnected from WiFi"),
    ActionType.DISK_LIST: (DiskManageParams(), "", "NAME SIZE TYPE\nsda 476G disk\nsdb 32G disk"),
    ActionType.DISK_MOUNT: (
        DiskManageParams(device="/dev/sdb1", mount_point="/mnt/usb"),
        "/dev/sdb1",
        "Mounted /dev/sdb1 at /mnt/usb",
    ),
    ActionType.DISK_UNMOUNT: (DiskManageParams(mount_point="/mnt/usb"), "/mnt/usb", "Unmounted /mnt/usb"),
    ActionType.USER_LIST: (EmptyParams(), "", "me\nguest"),
    ActionType.USER_INFO: (EmptyParams(), "", "uid=1000(me) gid=1000(me) groups=1000(me),27(sudo)"),
    ActionType.REGISTRY_READ: (
        RegistryParams(key_path=r"HKCU\Software\Test", value_name="Theme"),
        r"HKCU\Software\Test",
        "HKEY_CURRENT_USER\\Software\\Test\n    Theme    REG_SZ    Dark",
    ),
    ActionType.REGISTRY_WRITE: (
        RegistryParams(key_path=r"HKCU\Software\Test", value_name="Theme", value_data="Dark"),
        r"HKCU\Software\Test",
        r"Registry: set HKCU\Software\Test\Theme = Dark",
    ),
    ActionType.LOG_ANALYZE: (
        LogAnalyzeParams(log_path=W + r"\logs\app.log"),
        W + r"\logs\app.log",
        json.dumps(
            {
                "severity": "low",
                "summary": "Log analysis completed over 120 events. Analyzed 120 events. No anomalies detected.",
                "events_total": 120,
                "indicators": [],
                "timeline": [
                    "[2026-11-26 18:40:01] app: started worker 4",
                    "[2026-11-26 18:41:07] app: saved C:\\Users\\me\\work\\notes\\note.txt",
                ],
                "timeline_note": "The timeline shows all 2 events.",
                "recommended_actions": ["No action required at this time"],
            },
            indent=2,
        ),
    ),
    ActionType.MOUSE_CLICK: (MouseParams(x=100, y=200), "", "Clicked at (100, 200)"),
    ActionType.MOUSE_DOUBLE_CLICK: (MouseParams(x=100, y=200), "", "Double-clicked at (100, 200)"),
    ActionType.MOUSE_RIGHT_CLICK: (MouseParams(x=100, y=200), "", "Right-clicked at (100, 200)"),
    ActionType.MOUSE_MOVE: (MouseParams(x=100, y=200), "", "Moved mouse to (100, 200)"),
    ActionType.MOUSE_DRAG: (MouseParams(x=1, y=2, end_x=3, end_y=4), "", "Dragged from (1, 2) to (3, 4)"),
    ActionType.MOUSE_SCROLL: (MouseParams(amount=-3), "", "Scrolled -3"),
    ActionType.MOUSE_POSITION: (MouseParams(), "", "Mouse position: (100, 200)"),
    ActionType.KEYBOARD_TYPE: (KeyboardParams(text="see you at 6"), "", "Typed 12 characters"),
    ActionType.KEYBOARD_PRESS: (KeyboardParams(key="enter"), "", "Pressed enter"),
    ActionType.KEYBOARD_HOTKEY: (KeyboardParams(keys=["ctrl", "s"]), "", "Pressed ctrl+s"),
    ActionType.KEYBOARD_HOLD: (KeyboardParams(key="shift", duration=1.0), "", "Held shift for 1.0s"),
    ActionType.SCREEN_OCR: (ScreenVisionParams(), "", "Inbox (3)\nMeeting at 10 with Ann"),
    ActionType.SCREEN_FIND_TEXT: (ScreenVisionParams(target_text="Submit"), "Submit", "Found 'Submit' at (300, 400)"),
    ActionType.SCREEN_ANALYZE: (
        ScreenVisionParams(prompt="what is on screen"),
        "",
        "A browser shows the Heliox docs, with a sign-in button at the top right.",
    ),
    ActionType.SCREEN_ELEMENT_MAP: (ScreenVisionParams(), "", "[1] button Submit (300,400)\n[2] link Help (10,20)"),
    ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT: (
        AccessibilitySnapshotParams(),
        "",
        "window Notepad\n  edit Text editor\n  button Close",
    ),
    ActionType.SCREEN_DETECT_ELEMENTS: (
        ElementDetectionParams(description="buttons"),
        "",
        "button Save at (10, 20)\nbutton Cancel at (80, 20)",
    ),
    ActionType.BROWSER_NAVIGATE: (
        BrowserParams(url="https://example.com/pricing"),
        "https://example.com/pricing",
        "Navigated to https://example.com/pricing (Example Domain)",
    ),
    ActionType.BROWSER_CLICK: (BrowserParams(selector="#go"), "#go", "Clicked #go\n[PAGE_BEFORE] 5f2a9c1e77aa"),
    ActionType.BROWSER_CLICK_TEXT: (BrowserParams(text="Submit"), "Submit", "Clicked Submit"),
    ActionType.BROWSER_TYPE: (BrowserParams(selector="#q", text="hello"), "#q", "Typed into #q"),
    ActionType.BROWSER_SELECT: (BrowserParams(selector="#size", value="Large"), "#size", "Selected Large in #size"),
    ActionType.BROWSER_HOVER: (BrowserParams(selector="#menu"), "#menu", "Hovered #menu"),
    ActionType.BROWSER_SCROLL: (BrowserParams(direction="down"), "", "Scrolled down 500px"),
    ActionType.BROWSER_EXTRACT: (BrowserParams(selector="h1"), "h1", "Example Domain"),
    ActionType.BROWSER_EXTRACT_TABLE: (
        BrowserParams(selector="table"),
        "table",
        json.dumps([["Item", "Price"], ["Tea", "3.50"], ["Cake", "4.00"]]),
    ),
    ActionType.BROWSER_EXTRACT_LINKS: (
        BrowserParams(),
        "",
        json.dumps([{"text": "More", "href": "https://www.iana.org/domains/example"}]),
    ),
    ActionType.BROWSER_EXECUTE_JS: (BrowserParams(script="document.title"), "", "Example Domain"),
    ActionType.BROWSER_SCREENSHOT: (
        BrowserParams(output_path=W + r"\shots\page.png"),
        "",
        f"Screenshot saved to {W}\\shots\\page.png",
    ),
    ActionType.BROWSER_FILL_FORM: (
        BrowserParams(fields={"#name": "Ann", "#mail": "ann@example.com"}),
        "",
        "Filled 2 fields",
    ),
    ActionType.BROWSER_NEW_TAB: (BrowserParams(url="https://example.com"), "", "Opened new tab 2: https://example.com"),
    ActionType.BROWSER_CLOSE_TAB: (BrowserParams(tab_index=1), "", "Closed tab 1"),
    ActionType.BROWSER_LIST_TABS: (
        BrowserParams(),
        "",
        json.dumps([{"index": 0, "title": "Example Domain", "url": "https://example.com/"}]),
    ),
    ActionType.BROWSER_SWITCH_TAB: (BrowserParams(tab_index=1), "", "Switched to tab 1"),
    ActionType.BROWSER_BACK: (BrowserParams(), "", "Went back to https://example.com/"),
    ActionType.BROWSER_FORWARD: (BrowserParams(), "", "Went forward to https://example.com/a"),
    ActionType.BROWSER_REFRESH: (BrowserParams(), "", "Reloaded https://example.com/"),
    ActionType.BROWSER_WAIT: (BrowserParams(selector="#done"), "#done", "Element #done is visible"),
    ActionType.BROWSER_CLOSE: (BrowserParams(), "", "Browser closed"),
    ActionType.BROWSER_PAGE_INFO: (
        BrowserParams(),
        "",
        json.dumps({"title": "Example Domain", "url": "https://example.com/", "tabs": 1}),
    ),
    ActionType.BROWSER_SNAPSHOT: (
        BrowserParams(),
        "",
        "snapshot 7c1e2f9a\n[e1] heading Example Domain\n[e2] link More information",
    ),
    ActionType.TRIGGER_CREATE: (
        TriggerParams(name="watch_inbox_file", trigger_type="file_modified"),
        "watched.txt",
        RUN_TRIGGER,
    ),
    ActionType.TRIGGER_LIST: (TriggerParams(), "", f"[\n{RUN_TRIGGER}\n]"),
    ActionType.TRIGGER_DELETE: (TriggerParams(trigger_id="t0a1b2c3"), "t0a1b2c3", "Deleted trigger: t0a1b2c3"),
    ActionType.TRIGGER_START: (TriggerParams(), "", "Trigger engine started with 2 triggers"),
    ActionType.TRIGGER_STOP: (TriggerParams(), "", "Trigger engine stopped"),
    ActionType.CALENDAR_PARSE: (
        CalendarParams(file_path=W + r"\cal.ics"),
        W + r"\cal.ics",
        "Standup, Mon 09:00\nLunch with Ann, Tue 12:30\nDentist, Wed 10:00",
    ),
    ActionType.CALENDAR_SYNC: (CalendarParams(), "", "Synced 12 events"),
    ActionType.CALENDAR_CREATE_EVENT: (
        CalendarParams(summary="Dentist", start="2026-12-02T10:00"),
        "Dentist",
        "Created event Dentist (uid 9f8e7d6c5b4a3210)",
    ),
    ActionType.CALENDAR_LIST_EVENTS: (CalendarParams(), "", "Standup 09:00\nLunch 12:30"),
    ActionType.CALENDAR_DELETE_EVENT: (
        CalendarParams(event_uid="9f8e7d6c5b4a3210"),
        "9f8e7d6c5b4a3210",
        "Deleted event 9f8e7d6c5b4a3210",
    ),
    ActionType.CALENDAR_FETCH: (CalendarParams(), "", "Standup 09:00\nLunch 12:30\nGym 18:00"),
    ActionType.CALENDAR_RECONCILE: (CalendarParams(), "", "No conflicts and no missing meeting links."),
    ActionType.API_REQUEST: (ApiRequestParams(url="https://example.com"), "https://example.com", RUN_HTTP),
    ActionType.API_GITHUB: (
        ApiRequestParams(endpoint="/user/repos"),
        "/user/repos",
        json.dumps(
            {
                "status": 200,
                "url": "https://api.github.com/user/repos",
                "headers": {},
                "body": [{"name": "heliox"}, {"name": "notes"}],
            }
        ),
    ),
    ActionType.API_SEND_EMAIL: (
        ApiRequestParams(to="ann@example.com", subject="Running late", message="Ten minutes."),
        "ann@example.com",
        "Email sent to ann@example.com (message id <a1b2c3d4e5f6@example.com>)",
    ),
    ActionType.API_WEBHOOK: (
        ApiRequestParams(url="https://hooks.example.com/services/T000/B000/abc123def456"),
        "https://hooks.example.com",
        '{"status": 200, "body": "ok"}',
    ),
    ActionType.API_SLACK: (ApiRequestParams(message="hi", channel="general"), "slack", "Slack message sent (200)"),
    ActionType.API_DISCORD: (ApiRequestParams(message="hi"), "discord", "Discord message sent (204)"),
    ActionType.API_SCRAPE: (
        ApiRequestParams(url="https://example.com/prices", selector="td.amount"),
        "https://example.com/prices",
        "12.00\n13.50\n9.99",
    ),
    ActionType.EMAIL_FETCH: (
        EmailParams(),
        "",
        "From: Ann <ann@example.com> | Lunch?\nFrom: Bank <no-reply@bank.example> | Statement ready",
    ),
    ActionType.EMAIL_SUMMARIZE: (EmailParams(), "", "Ann asks about lunch on Friday; the bank statement is ready."),
    ActionType.EMAIL_REPLY: (EmailParams(reply_body="Yes, Friday works."), "", "Reply sent to ann@example.com"),
    ActionType.WORKSPACE_INDEX: (
        WorkspaceParams(folder_path=W + r"\workspace"),
        W + r"\workspace",
        f"Indexed 34 files (120 chunks) from {W}\\workspace",
    ),
    ActionType.WORKSPACE_SEARCH: (
        WorkspaceParams(query="budget"),
        "budget",
        f"1. {W}\\workspace\\budget.txt (0.82)\n2. {W}\\workspace\\plan.md (0.61)",
    ),
    ActionType.WASM_CALL: (WasmCallParams(tool="hash"), "hash", json.dumps({"result": "ok", "elapsed_ms": 3})),
    ActionType.PLUGIN_CALL: (
        WasmCallParams(tool="weather"),
        "weather",
        json.dumps({"temperature": 12, "sky": "cloudy"}),
    ),
    ActionType.SKILL_RUN: (SkillRunParams(skill_id="weather"), "weather", "It's 12 degrees and cloudy."),
    ActionType.MCP_LIST_TOOLS: (
        McpToolParams(mcp_server="files"),
        "files",
        "files: read_file - Read a file\nfiles: list_dir - List a folder",
    ),
    ActionType.MCP_CALL_TOOL: (
        McpToolParams(mcp_server="files", mcp_tool="read_file", mcp_arguments={"path": "a.txt"}),
        "files",
        json.dumps({"content": [{"type": "text", "text": "hello"}]}),
    ),
}

# Longer shapes of the reads whose answer is the point: said as their length and beginning, never in full.
LONG_CASES: list[tuple[ActionType, object, str, str]] = [
    (ActionType.FILE_READ, FileParams(path=W + r"\logs\app.log"), W + r"\logs\app.log",
     "\n".join(f"2026-11-26 18:4{i}:07 worker {i} saved C:\\Users\\me\\work\\notes\\n{i}.txt" for i in range(8))),
    (ActionType.FILE_READ, FileParams(path=NOTE), NOTE, LONG_TEXT),
    (ActionType.SHELL_COMMAND, ShellCommandParams(command="dir"), "dir", "\n".join(f"line {i} of the listing" for i in range(40))),
    (ActionType.API_REQUEST, ApiRequestParams(url="https://example.com/notes"), "https://example.com/notes",
     json.dumps({"status": 200, "url": "https://example.com/notes", "headers": {}, "body": LONG_TEXT * 3})),
    (ActionType.SCREEN_OCR, ScreenVisionParams(), "", LONG_TEXT),
    (ActionType.CLIPBOARD_READ, ClipboardParams(), "", '{"token": "abc", "items": [1, 2, 3]}'),
    (ActionType.CLIPBOARD_READ, ClipboardParams(), "", "(clipboard is empty)"),
    (ActionType.FILE_DELETE, FileParams(path=NOTE), NOTE, f"There was no file at {NOTE}, so nothing was deleted"),
    (ActionType.FILE_WRITE, FileParams(path=NOTE, content="Also renew the library card.", append=True), NOTE,
     f"Appended 1 line to {NOTE} (appended sha256 {SHA}, 32 bytes; file sha256 {SHA}, 72 bytes)"),
]  # fmt: skip


def _result(action_type: ActionType, params: object, target: str, output: str) -> ActionResult:
    action = Action(action_type=action_type, target=target, parameters=params)
    return ActionResult(action=action, success=True, output=output)


def _all_cases() -> list[tuple[str, ActionResult]]:
    cases = [(kind.value, _result(kind, *CASES[kind])) for kind in ActionType if kind in CASES]
    cases += [(f"{kind.value} (long {i})", _result(kind, *rest)) for i, (kind, *rest) in enumerate(LONG_CASES)]
    return cases


def test_every_action_type_has_a_fake_result_and_a_sentence() -> None:
    assert set(CASES) == set(ActionType), sorted(kind.value for kind in set(ActionType) - set(CASES))
    assert set(SENTENCES) == set(ActionType), sorted(kind.value for kind in set(ActionType) - set(SENTENCES))


def _not_cut_inside_a_word(cut: str, whole: str) -> bool:
    """``cut`` is ``whole`` up to a sentence's end, or up to a word's end followed by its full stop."""
    if cut == whole:
        return True
    head = cut[:-1] if cut.endswith(".") else cut
    return whole.startswith(head) and (len(whole) == len(head) or not whole[len(head)].isalnum())


@pytest.mark.parametrize(("name", "result"), _all_cases(), ids=[name for name, _ in _all_cases()])
def test_each_action_is_shown_and_said_without_raw_data(name: str, result: ActionResult) -> None:
    plan = ActionPlan(actions=[result.action], explanation="", raw_input="do it")
    said = success_reply(plan, [result], VerificationResult(passed=True, independently_observed_actions=[0]),
                         dry_run=False)  # fmt: skip

    assert screen_faults(said.shown) == [], (name, said.shown)
    assert ear_faults(said.spoken) == [], (name, said.spoken)
    whole = step_said(result, limit=100_000).spoken
    assert _not_cut_inside_a_word(step_said(result).spoken, whole), (name, whole)

    # The last check before the reply leaves the daemon keeps both forms as they are.
    final = with_reply({"status": "success", "message": said.shown, "spoken": said.spoken})
    assert (final["message"], final["spoken"]) == (said.shown, said.spoken)


@pytest.mark.parametrize("kind", list(ActionType), ids=[kind.value for kind in ActionType])
def test_each_action_that_fails_is_said_in_plain_words(kind: ActionType) -> None:
    params, target, _output = CASES[kind]
    action = Action(action_type=kind, target=target, parameters=params)
    for error in (
        f"[Errno 13] Permission denied: {NOTE!r}",
        "Process exited with code 125: fatal: {'bad': 'thing'} at C:\\\\x\\\\y",
    ):
        said = failure_reply([ActionResult(action=action, success=False, error=error)], None)
        assert ear_faults(said.spoken) == [], (kind, error, said.spoken)
        assert "\\\\" not in said.shown, (kind, said.shown)
        assert "Errno" not in said.spoken and "125" not in said.spoken


def test_a_plan_of_several_steps_is_one_or_two_sentences_about_its_outcome() -> None:
    read = _result(ActionType.FILE_READ, *CASES[ActionType.FILE_READ])
    write = _result(ActionType.FILE_WRITE, *CASES[ActionType.FILE_WRITE])
    copies = [
        _result(
            ActionType.FILE_COPY,
            FileParams(path=W + r"\big\source.bin", destination=W + rf"\big\late_{c}.bin"),
            "",
            f"Copied {W}\\big\\source.bin -> {W}\\big\\late_{c}.bin",
        )
        for c in "abc"
    ]
    verified = VerificationResult(passed=True)

    # A read followed by a write (the run's "Verified results:" and both paths): the write is the outcome.
    said = success_reply(ActionPlan(actions=[read.action, write.action]), [read, write], verified, dry_run=False)
    assert said.shown == "Wrote 40 bytes to note.txt in notes."
    assert said.spoken == "Wrote 40 bytes to note.txt in notes."

    said = success_reply(ActionPlan(actions=[c.action for c in copies]), copies, verified, dry_run=False)
    assert said.shown == said.spoken == "Copied 3 files into big."

    stage = _result(ActionType.GIT_STAGE, *CASES[ActionType.GIT_STAGE])
    commit = _result(ActionType.GIT_COMMIT, *CASES[ActionType.GIT_COMMIT])
    push = _result(ActionType.GIT_PUSH, *CASES[ActionType.GIT_PUSH])
    said = success_reply(ActionPlan(actions=[stage.action, commit.action, push.action]), [stage, commit, push],
                         verified, dry_run=False)  # fmt: skip
    assert said.spoken == "Pushed main of repo to origin. All 3 steps are done."


@pytest.mark.parametrize(
    ("kind", "shown", "spoken"),
    [
        (ActionType.GIT_STAGE, "Staged README.md in repo.", "Staged README.md in repo."),
        (
            ActionType.GIT_COMMIT,
            'Committed 2 files to repo on main: "Fix the intro".',
            'Committed 2 files to repo on main: "Fix the intro".',
        ),
        (
            ActionType.GIT_BRANCH,
            "Switched repo to the new branch topic-branch.",
            "Switched repo to the new branch topic-branch.",
        ),
        (ActionType.GIT_PUSH, "Pushed main of repo to origin.", "Pushed main of repo to origin."),
        (ActionType.FILE_WRITE, "Wrote 40 bytes to note.txt in notes.", "Wrote 40 bytes to note.txt in notes."),
        (
            ActionType.API_REQUEST,
            "example.com answered 200. The page title is Example Domain.",
            "Example.com answered 200. The page title is Example Domain.",
        ),
        (
            ActionType.TRIGGER_CREATE,
            'I\'ll watch watched.txt and add "file touched" to fired.log when it changes.',
            'I\'ll watch watched.txt and add "file touched" to fired.log when it changes.',
        ),
        (
            ActionType.TRIGGER_LIST,
            "You have 1 trigger: it watches watched.txt for changes.",
            "You have 1 trigger: it watches watched.txt for changes.",
        ),
        (
            ActionType.TRIGGER_DELETE,
            "I've removed that trigger.",
            "I've removed that trigger.",
        ),
        (
            ActionType.BATTERY_INFO,
            "The battery is at 80 percent, charging.",
            "The battery is at 80 percent, charging.",
        ),
        (
            ActionType.FILE_HASH,
            "I worked out the SHA-256 of note.txt; it's on screen.",
            "I worked out the SHA-256 of note.txt; it's on screen.",
        ),
    ],
)
def test_the_runs_actions_have_these_sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said = step_said(_result(kind, *CASES[kind]))
    assert said.shown.splitlines()[0] == shown
    assert said.spoken == spoken


def test_a_long_answer_is_said_as_its_length_and_beginning() -> None:
    said = step_said(_result(ActionType.CLIPBOARD_READ, ClipboardParams(), "", LONG_TEXT))
    assert said.spoken.startswith("Your clipboard is 310 characters long, starting: Dear team, the annual report")
    # The shown reply starts with the sentence that is said; the text itself follows it.
    assert said.shown.splitlines()[1] == LONG_TEXT
    log = step_said(_result(*LONG_CASES[0]))
    assert log.spoken.startswith(
        "App.log has 8 lines; the newest is: the twenty-sixth of November 18:47:07 worker 7 saved n7.txt."
    )


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        (f"Written 40 bytes to {NOTE} (sha256 {SHA}, 40 bytes on disk)", "Written 40 bytes to note.txt, 40 bytes on disk."),
        ("{\"success\": true}", ""),
        ("Opened https://www.example.com/a/b?c=1 in default browser", "Opened example.com in default browser."),
        ("Committed as 1a2b3c4: Fix the intro", "Committed as: Fix the intro."),
        ("Battery 68% and/or more", "Battery sixty-eight percent and or more."),
        ("x" * 150 + " " + "y" * 100, "X" + "x" * 149 + "."),
        ("First sentence here. " + "x " * 120, "First sentence here."),
    ],
)  # fmt: skip
def test_the_speech_rules(text: str, spoken: str) -> None:
    assert for_ear(text) == spoken
    assert len(for_ear(text)) <= SPOKEN_LIMIT


# ── a read's content passes the secret rules before it is said ───────────────────────────────────────────────────

GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SECRET_READS = [
    (ActionType.CLIPBOARD_READ, ClipboardParams(), "", "Tr0ub4dor&3xY!", "Your clipboard"),
    (ActionType.CLIPBOARD_READ, ClipboardParams(), "", "my bank PIN is 4471", "Your clipboard"),
    (ActionType.FILE_READ, FileParams(path=NOTE), NOTE, f"api token: {GITHUB_TOKEN}\nnotes for Friday", "note.txt"),
    (
        ActionType.FILE_READ,
        FileParams(path=W + r"\logs\app.log"),
        W + r"\logs\app.log",
        f"18:40 started\n18:41 sent header Authorization: Bearer {GITHUB_TOKEN}",
        "app.log",
    ),
    (
        ActionType.API_REQUEST,
        ApiRequestParams(url="https://example.com/me"),
        "https://example.com/me",
        json.dumps({"status": 200, "url": "https://example.com/me", "headers": {}, "body": f"key={GITHUB_TOKEN}"}),
        "The answer from example.com",
    ),
    (ActionType.ENV_GET, EnvParams(name="GH_TOKEN"), "GH_TOKEN", f"GH_TOKEN={GITHUB_TOKEN}", "GH TOKEN"),
    (ActionType.SCREEN_OCR, ScreenVisionParams(), "", "Your verification code is 482913", "The screen"),
]


@pytest.mark.parametrize(("kind", "params", "target", "output", "lead"), SECRET_READS)
def test_content_holding_a_secret_is_not_read_out(kind, params, target, output, lead) -> None:
    said = step_said(_result(kind, params, target, output))

    assert said.spoken.casefold().startswith(f"{lead} holds something that looks like".casefold()), said.spoken
    assert said.spoken.endswith("so I haven't read it out.")
    for secret in (GITHUB_TOKEN, "4471", "482913", "Tr0ub4dor"):
        assert secret not in said.spoken and secret not in said.shown, (secret, said)
    assert ear_faults(said.spoken) == [] and screen_faults(said.shown) == []


def test_ordinary_content_is_still_said() -> None:
    said = step_said(_result(ActionType.CLIPBOARD_READ, ClipboardParams(), "", "Buy milk on Friday"))
    assert said.spoken == "Your clipboard: Buy milk on Friday."


# ── a question's answer is said before the change ────────────────────────────────────────────────────────────────


def test_read_the_note_tell_me_what_it_says_then_delete_it() -> None:
    read = _result(ActionType.FILE_READ, *CASES[ActionType.FILE_READ])
    delete = _result(ActionType.FILE_DELETE, *CASES[ActionType.FILE_DELETE])
    verified = VerificationResult(passed=True)
    asked = ActionPlan(
        actions=[read.action, delete.action], raw_input="Read note.txt, tell me what it says, then delete it"
    )

    said = success_reply(asked, [read, delete], verified, dry_run=False)

    assert said.spoken == "Note.txt: Water the ferns and call Sam on Fridays. Deleted note.txt in notes."
    assert said.shown.splitlines() == [
        "note.txt: Water the ferns and call Sam on Fridays.",
        "Deleted note.txt in notes.",
    ]

    # Not asked for: only the change is said.
    plain = ActionPlan(actions=[read.action, delete.action], raw_input="Back up note.txt then delete it")
    assert success_reply(plain, [read, delete], verified, dry_run=False).spoken == "Deleted note.txt in notes."


# ── triggers are said from what they watch, never from their generated name ──────────────────────────────────────


def test_trigger_sentences() -> None:
    cpu = json.dumps({"id": "9a8b7c6d", "name": "cpu_high_alert", "trigger_type": "cpu_threshold",
                      "condition": {"threshold": 85}, "action_command": ""})  # fmt: skip
    said = step_said(_result(ActionType.TRIGGER_CREATE, TriggerParams(trigger_type="cpu_threshold"), "", cpu))
    assert said.spoken == "I'll watch the processor and tell you when it goes above 85 percent."

    both = f"[{RUN_TRIGGER}, {cpu}]"
    said = step_said(_result(ActionType.TRIGGER_LIST, TriggerParams(), "", both))
    assert said.spoken == (
        "You have 2 triggers: one watches watched.txt for changes; one watches the processor when it goes above 85 "
        "percent."
    )
    deleted = step_said(_result(ActionType.TRIGGER_DELETE, TriggerParams(trigger_id="t0a1b2c3"), "watched.txt",
                                "Deleted trigger: t0a1b2c3"))  # fmt: skip
    assert deleted.spoken == "I've stopped watching watched.txt."
    for text in (said.spoken, deleted.spoken):
        assert "t0a1b2c3" not in text and "cpu high alert" not in text


# ── a failure is its sentence first, then one plain count for a plan of several steps ────────────────────────────


def test_a_failure_reply_has_no_prefix_and_counts_only_several_steps() -> None:
    write = Action(action_type=ActionType.FILE_WRITE, target=NOTE, parameters=FileParams(path=NOTE, content="x"))
    denied = ActionResult(action=write, success=False, error=f"[Errno 13] Permission denied: {NOTE!r}")
    one = failure_reply([denied], None)
    assert (
        one.shown
        == one.spoken
        == (
            "I couldn't write note.txt in notes: permission was denied. You can change its permissions, or choose "
            "another place."
        )
    )
    copied = _result(ActionType.FILE_COPY, *CASES[ActionType.FILE_COPY])
    listed = _result(ActionType.FILE_LIST, *CASES[ActionType.FILE_LIST])
    three = failure_reply([copied, denied, listed], None)
    assert three.shown.endswith("1 of 3 steps failed; the other 2 are done.")
    assert "did not complete" not in three.shown and "actions failed" not in three.shown


# ── One voice: every sentence that can end a request, from every place that makes one ──────────


def _every_ending() -> dict[str, str]:
    """Each fixed or built sentence that can be shown or said when a request or a step ends, by where it comes from."""
    from pilot.actions import Action, FileParams
    from pilot.agents.orchestrator import AGENT_FAULT_ERROR
    from pilot.agents.request_match import REFUSED_FIRST, REFUSED_REPAIR
    from pilot.agents.review_timing import kept_copy_failed_sentence
    from pilot.agents.unconfirmed import LATE_MISMATCH_DEFAULT, LATE_MISMATCH_SENTENCES, SPOKEN_PHRASES, window_line
    from pilot.result_sentences import trigger_fire_sentence
    from pilot.server import REQUEST_FAULT_SENTENCE
    from pilot.task_reply import NO_STATUS, STATUS_SENTENCES

    trigger = {
        "name": "notes_watch",
        "trigger_type": "file_modified",
        "condition": {"path": "C:\\Users\\sam\\notes.txt"},
        "action_command": "add a line to log.txt",
        "steps": [
            {"action_type": "file_read", "target": "log.txt", "parameters": {"path": "log.txt"}},
            {
                "action_type": "file_write",
                "target": "log.txt",
                "parameters": {"path": "log.txt", "content": "{PREV_OUTPUT}\nseen"},
                "use_previous_output": True,
            },
        ],
    }
    endings = {
        "agent fault": AGENT_FAULT_ERROR,
        "request fault": REQUEST_FAULT_SENTENCE,
        "undo copy lost": kept_copy_failed_sentence(
            Action(action_type=ActionType.FILE_WRITE, target="a.txt", parameters=FileParams(path="a.txt"))
        ),
        "no status": NO_STATUS,
        "late default": LATE_MISMATCH_DEFAULT,
        "plan check, first": f"{REFUSED_FIRST}: the plan I made would copy a.txt to b.txt, which is not what you asked.",
        "plan check, repair": f"{REFUSED_REPAIR}: the retry would delete b.txt, which this request did not name.",
        "window line": window_line([(ActionType.FILE_COPY, "could_not_look")]),
    }
    endings.update({f"status {key}": value for key, value in STATUS_SENTENCES.items()})
    # The phrases are said as sentences of their own (pilot.agents.unconfirmed.speakable_message adds the stop).
    endings.update({f"phrase {key}": f"{value}." for key, value in SPOKEN_PHRASES.items()})
    endings.update({f"late {key.value}": value for key, value in LATE_MISMATCH_SENTENCES.items()})
    for outcome in ("ran", "timeout", "fault", "waits", "unreadable", "unavailable", "job_done", "job_failed"):
        endings[f"trigger {outcome}"] = trigger_fire_sentence(trigger, outcome, seconds=60)
    # Round twelve: what no action reads, a repair refused in code, and the replies to a pick from a list.
    from pilot.agents.read_catalog import CANNOT_READ
    from pilot.agents.request_match import Mismatch, repair_refusal

    endings.update({f"cannot read {what}": sentence for what, sentence in CANNOT_READ})
    for rule, step in (
        ("repair_escalation", "run a shell command instead of the git push that failed"),
        ("repair_escalation", "run the file copy again with 'overwrite' turned on"),
        ("repair_new_kind", "start a delete, which you did not ask for"),
        ("value_dropped", "leave out '.price-table td.amount', which you asked for"),
    ):
        endings[f"repair refused, {rule}: {step}"] = repair_refusal(Mismatch(rule, "", step=step))
    endings["pick kept, one"] = 'Okay, I\'ve kept the trigger "notes" (file modified: notes.txt).'
    endings.update(
        {
            "pick kept, two": "Okay, I've kept both triggers; nothing was deleted.",
            "pick kept, three": "Okay, I've kept all 3 triggers; nothing was deleted.",
        }
    )
    return endings


import re  # noqa: E402

INTERNAL_WORDS = re.compile(
    r"\b(?:executor|callback|postcondition|post-condition|action_type|plan_id|risk gate|world model|traceback|"
    r"exception|stack|json|none|null|true|false)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(("source", "sentence"), sorted(_every_ending().items()))
def test_every_ending_speaks_plainly(source: str, sentence: str) -> None:
    from pilot.reply_rules import ear_faults, screen_faults

    assert sentence.strip(), source
    assert ear_faults(sentence) == [], source
    assert screen_faults(sentence) == [], source
    assert not INTERNAL_WORDS.search(sentence), source
    # Where Heliox acted or stopped, it says so in the first person, never as "Heliox" or "it".
    assert not re.search(r"\bHeliox (?:stopped|could not|did not|failed|ran)\b", sentence), source


def test_an_append_is_said_as_the_line_it_added() -> None:
    """One step now, not a read and a write: "Added a line to note.txt in notes.", nothing about a check."""
    append = _result(ActionType.FILE_WRITE, *LONG_CASES[-1][1:])
    said = success_reply(ActionPlan(actions=[append.action]), [append], VerificationResult(passed=True), dry_run=False)
    assert said.shown == said.spoken == "Added a line to note.txt in notes."
