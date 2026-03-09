"""Scheduled tasks — create, list, delete recurring tasks.

Cross-platform: Windows (schtasks), Linux (crontab), macOS (launchctl).
"""

from __future__ import annotations

import logging
import os
import tempfile

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.scheduler")


async def schedule_create(name: str, command: str, schedule: str) -> str:
    """Create a scheduled task.

    Args:
        name: Task name/identifier
        command: Command to run
        schedule: Cron expression (Linux/macOS) or schedule spec (Windows)
                  Windows examples: "DAILY /ST 09:00", "HOURLY", "MINUTE /MO 30"
                  Linux examples: "0 9 * * *" (daily at 9am), "*/30 * * * *" (every 30 min)
    """
    if CURRENT_PLATFORM == Platform.WINDOWS:
        args = [
            "schtasks",
            "/create",
            "/tn",
            name,
            "/tr",
            command,
            "/sc",
        ]
        # Parse schedule
        parts = schedule.upper().split()
        args.extend(parts)
        args.append("/f")  # force overwrite

        code, out, err = await run_command(args)
        if code != 0:
            raise RuntimeError(f"Task creation failed: {err.strip()}")
        return f"Created scheduled task: {name}"

    elif CURRENT_PLATFORM == Platform.MACOS:
        # Create a launchd plist
        plist_name = f"com.pilot.{name}"
        plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")

        # Basic plist for a periodic job
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>{command}</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
</dict>
</plist>"""

        with open(plist_path, "w") as f:
            f.write(plist_content)

        code, out, err = await run_command(["launchctl", "load", plist_path])
        if code != 0:
            raise RuntimeError(f"Task creation failed: {err.strip()}")
        return f"Created scheduled task: {name} ({plist_path})"

    else:  # Linux - crontab
        # Get existing crontab
        code, existing, _ = await run_command(["crontab", "-l"])
        if code != 0:
            existing = ""

        # Add new entry
        marker = f"# pilot-task:{name}"
        new_line = f"{schedule} {command} {marker}"

        # Remove old entry with same name if exists
        lines = [l for l in existing.strip().split("\n") if marker not in l]
        lines.append(new_line)
        new_crontab = "\n".join(lines) + "\n"

        # Install
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cron", delete=False) as f:
            f.write(new_crontab)
            tmp_path = f.name

        try:
            code, out, err = await run_command(["crontab", tmp_path])
        finally:
            os.unlink(tmp_path)

        if code != 0:
            raise RuntimeError(f"Crontab update failed: {err.strip()}")
        return f"Created scheduled task: {name} ({schedule})"


async def schedule_list() -> str:
    """List scheduled tasks."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["schtasks", "/query", "/fo", "TABLE", "/nh"])
        if code != 0:
            raise RuntimeError(f"Task list failed: {err.strip()}")
        # Filter to show manageable number of lines
        lines = out.strip().split("\n")
        return "\n".join(lines[:50])

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["launchctl", "list"])
        if code != 0:
            raise RuntimeError(f"Task list failed: {err.strip()}")
        # Filter to pilot tasks
        lines = out.strip().split("\n")
        pilot_lines = [l for l in lines if "pilot" in l.lower()]
        if pilot_lines:
            return "Pilot tasks:\n" + "\n".join(pilot_lines)
        return "User launch agents:\n" + "\n".join(lines[:30])

    else:  # Linux
        code, out, err = await run_command(["crontab", "-l"])
        if code != 0:
            return "No crontab configured"
        return f"Current crontab:\n{out.strip()}"


async def schedule_delete(name: str = "", task_id: str | None = None) -> str:
    """Delete a scheduled task."""
    target = task_id or name

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["schtasks", "/delete", "/tn", target, "/f"])
        if code != 0:
            raise RuntimeError(f"Task delete failed: {err.strip()}")
        return f"Deleted scheduled task: {target}"

    elif CURRENT_PLATFORM == Platform.MACOS:
        plist_name = f"com.pilot.{target}"
        plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
        code, out, err = await run_command(["launchctl", "unload", plist_path])
        if os.path.exists(plist_path):
            os.unlink(plist_path)
        return f"Deleted scheduled task: {target}"

    else:  # Linux
        marker = f"# pilot-task:{target}"
        code, existing, _ = await run_command(["crontab", "-l"])
        if code != 0:
            return "No crontab configured"

        lines = [l for l in existing.strip().split("\n") if marker not in l]
        new_crontab = "\n".join(lines) + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cron", delete=False) as f:
            f.write(new_crontab)
            tmp_path = f.name

        try:
            code, out, err = await run_command(["crontab", tmp_path])
        finally:
            os.unlink(tmp_path)

        if code != 0:
            raise RuntimeError(f"Crontab update failed: {err.strip()}")
        return f"Deleted scheduled task: {target}"
