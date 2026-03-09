"""Context-Aware Intelligence — know what the user is doing and proactively help.

Active window detection, time-of-day awareness, usage pattern tracking,
and proactive suggestions based on context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

logger = logging.getLogger("pilot.system.context")


# ── Active Window ────────────────────────────────────────────────────


async def get_active_window() -> str:
    """Get info about the currently focused window."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "Add-Type @'\n"
            "using System;\n"
            "using System.Runtime.InteropServices;\n"
            "using System.Text;\n"
            "public class WinAPI {\n"
            '    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();\n'
            '    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);\n'
            '    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);\n'
            "}\n"
            "'@\n"
            "$hwnd = [WinAPI]::GetForegroundWindow();\n"
            "$title = New-Object System.Text.StringBuilder 256;\n"
            "[WinAPI]::GetWindowText($hwnd, $title, 256) | Out-Null;\n"
            "$pid = 0;\n"
            "[WinAPI]::GetWindowThreadProcessId($hwnd, [ref]$pid) | Out-Null;\n"
            "$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue;\n"
            "@{Title=$title.ToString(); ProcessName=$proc.ProcessName; PID=$pid; "
            "MemoryMB=[math]::Round($proc.WorkingSet64/1MB,1)} | ConvertTo-Json"
        )
        return out.strip() if code == 0 else f"Error: {err}"

    elif CURRENT_PLATFORM == Platform.LINUX:
        proc = await asyncio.create_subprocess_exec(
            "xdotool",
            "getactivewindow",
            "getwindowname",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        title = stdout.decode().strip()

        proc2 = await asyncio.create_subprocess_exec(
            "xdotool", "getactivewindow", "getwindowpid", stdout=asyncio.subprocess.PIPE
        )
        stdout2, _ = await proc2.communicate()
        pid = stdout2.decode().strip()

        return json.dumps({"title": title, "pid": pid})

    elif CURRENT_PLATFORM == Platform.MACOS:
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            'tell application "System Events" to get name of first application process whose frontmost is true',
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return json.dumps({"app": stdout.decode().strip()})

    return "Platform not supported for active window detection"


# ── Time-of-Day Awareness ───────────────────────────────────────────


async def get_time_context() -> str:
    """Get time-of-day context for proactive suggestions."""
    now = datetime.now()
    hour = now.hour

    if 5 <= hour < 9:
        period = "early_morning"
        greeting = "Good morning! Starting your day."
    elif 9 <= hour < 12:
        period = "morning"
        greeting = "Good morning! Work time."
    elif 12 <= hour < 14:
        period = "midday"
        greeting = "It's lunchtime. Maybe take a break?"
    elif 14 <= hour < 17:
        period = "afternoon"
        greeting = "Good afternoon."
    elif 17 <= hour < 20:
        period = "evening"
        greeting = "Good evening. Wrapping up?"
    elif 20 <= hour < 23:
        period = "night"
        greeting = "Late evening. Consider winding down."
    else:
        period = "late_night"
        greeting = "It's very late. Get some rest!"

    return json.dumps(
        {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "period": period,
            "greeting": greeting,
            "hour": hour,
            "is_weekend": now.weekday() >= 5,
        },
        indent=2,
    )


# ── Usage Pattern Tracking ───────────────────────────────────────────


class UsageTracker:
    """Track user patterns for proactive suggestions."""

    def __init__(self, storage_path: str | None = None):
        self._storage = storage_path or os.path.expanduser("~/.pilot/usage_patterns.json")
        self._patterns: dict = self._load()

    def _load(self) -> dict:
        try:
            if os.path.exists(self._storage):
                return json.loads(Path(self._storage).read_text())
        except Exception:
            pass
        return {
            "app_launches": {},  # hour -> [app_names]
            "commands": [],  # recent commands with timestamps
            "frequent_tasks": {},  # task_hash -> count
        }

    def _save(self):
        Path(self._storage).parent.mkdir(parents=True, exist_ok=True)
        Path(self._storage).write_text(json.dumps(self._patterns, indent=2))

    def record_command(self, command: str):
        """Record a user command execution."""
        now = datetime.now()
        self._patterns["commands"].append(
            {
                "command": command[:200],
                "timestamp": now.isoformat(),
                "hour": now.hour,
                "day": now.strftime("%A"),
            }
        )
        # Keep last 500 commands
        self._patterns["commands"] = self._patterns["commands"][-500:]

        # Track by hour
        hour_key = str(now.hour)
        if hour_key not in self._patterns["app_launches"]:
            self._patterns["app_launches"][hour_key] = []
        self._patterns["app_launches"][hour_key].append(command[:50])

        # Track frequent tasks
        task_key = command[:100].lower().strip()
        self._patterns["frequent_tasks"][task_key] = self._patterns["frequent_tasks"].get(task_key, 0) + 1

        self._save()

    def get_suggestions(self) -> list[str]:
        """Get proactive suggestions based on patterns."""
        suggestions = []
        now = datetime.now()
        hour_key = str(now.hour)

        # What do they usually do at this hour?
        hour_cmds = self._patterns["app_launches"].get(hour_key, [])
        if hour_cmds:
            counter = Counter(hour_cmds)
            top = counter.most_common(3)
            for cmd, count in top:
                if count >= 3:
                    suggestions.append(f"You usually run '{cmd}' around this time ({count} times before)")

        # Most frequent tasks
        if self._patterns["frequent_tasks"]:
            top_tasks = sorted(self._patterns["frequent_tasks"].items(), key=lambda x: x[1], reverse=True)[:5]
            for task, count in top_tasks:
                if count >= 5:
                    suggestions.append(f"Frequent task: '{task}' ({count} times)")

        return suggestions

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_commands": len(self._patterns["commands"]),
            "unique_tasks": len(self._patterns["frequent_tasks"]),
            "most_active_hours": sorted(self._patterns["app_launches"].items(), key=lambda x: len(x[1]), reverse=True)[
                :5
            ],
        }


_tracker = UsageTracker()


async def get_context_summary() -> str:
    """Get a full context summary for the agent.

    Combines active window, time context, and usage patterns
    to give the agent awareness of the user's situation.
    """
    results = {}

    try:
        window_info = await get_active_window()
        results["active_window"] = json.loads(window_info) if window_info.startswith("{") else window_info
    except Exception as e:
        results["active_window"] = str(e)

    time_ctx = await get_time_context()
    results["time"] = json.loads(time_ctx)
    results["suggestions"] = _tracker.get_suggestions()
    results["stats"] = _tracker.get_stats()

    return json.dumps(results, indent=2)


async def record_user_command(command: str) -> str:
    """Record a command for pattern tracking."""
    _tracker.record_command(command)
    return f"Recorded: {command[:80]}"


async def get_proactive_suggestions() -> str:
    """Get proactive suggestions based on usage patterns."""
    suggestions = _tracker.get_suggestions()
    if not suggestions:
        return "No suggestions yet — keep using Pilot to build patterns!"
    return json.dumps(suggestions, indent=2)
