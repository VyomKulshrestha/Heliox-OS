"""Scheduled tasks — create, list, delete recurring tasks.

Cross-platform: Windows (schtasks), Linux (crontab), macOS (launchctl).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import plistlib
import tempfile
from dataclasses import dataclass
from xml.parsers.expat import ExpatError
from xml.sax.saxutils import escape as xml_escape

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

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
    <string>{xml_escape(plist_name)}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>{xml_escape(command)}</string>
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
        lines = [l for l in existing.strip().split("\n") if _crontab_entry(l, name) is None]
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


@dataclass(frozen=True)
class ObservedSchedule:
    """A scheduled task as the platform's scheduler holds it.

    ``command`` is None when the stored definition cannot be read (a launchd plist
    that is not a valid property list). ``schedule`` holds the cron fields on Linux
    and is empty where it is not read. ``loaded`` is False when macOS has the plist
    file but launchd does not list the job.
    """

    command: str | None
    schedule: str = ""
    loaded: bool = True


def _launch_agent_path(plist_name: str) -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")


def _crontab_entry(line: str, name: str) -> str | None:
    """The part of a crontab line before the marker of task ``name``, or None for another task's line.

    The marker must end the line: "# pilot-task:foo" is not the marker of "# pilot-task:foobar".
    """
    marker = f"# pilot-task:{name}"
    stripped = line.rstrip()
    if not stripped.endswith(marker):
        return None
    return stripped[: -len(marker)].rstrip()


_WINDOWS_TASK_ACTIONS_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "$service = New-Object -ComObject Schedule.Service; $service.Connect(); $json = $null; "
    "try { $task = $service.GetFolder('\\').GetTask($HelioxArgs['name']) } catch { "
    "$codes = @($_.Exception.HResult); "
    "if ($_.Exception.InnerException) { $codes += $_.Exception.InnerException.HResult }; "
    "if ($codes -contains -2147024894 -or $codes -contains -2147024893) { $json = '{\"found\":false}' } "
    "else { throw } }; "
    "if (-not $json) { "
    "$actions = @($task.Definition.Actions | Where-Object { $_.Type -eq 0 } | "
    "ForEach-Object { @{ path = [string]$_.Path; arguments = [string]$_.Arguments } }); "
    "$json = ConvertTo-Json -Compress -Depth 4 -InputObject @{ found = $true; actions = $actions } }; "
    "[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($json))"
)


async def observe_schedule(target: str) -> ObservedSchedule | None:
    """Read a Pilot scheduled task back from the platform's scheduler; None when there is no such task.

    Windows asks the Task Scheduler service through its COM interface (not schtasks,
    which created the task); macOS reads the plist file and asks launchd whether the
    job is loaded; Linux reads the installed crontab. Nothing is run or changed.
    Raises RuntimeError when the scheduler refuses to answer.
    """
    if not target:
        raise ValueError("Scheduled task target is required")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(_WINDOWS_TASK_ACTIONS_SCRIPT, values={"name": target})
        if code != 0:
            raise RuntimeError(f"Task observation failed: {err.strip() or out.strip()}")
        try:
            answer = json.loads(base64.b64decode(out.strip().splitlines()[-1]).decode("utf-8"))
        except (IndexError, ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("Task observation returned invalid data") from error
        if not answer.get("found"):
            return None
        commands = [
            " ".join(part for part in (str(item.get("path") or ""), str(item.get("arguments") or "")) if part)
            for item in answer.get("actions") or []
            if isinstance(item, dict)
        ]
        return ObservedSchedule(command="\n".join(commands))

    if CURRENT_PLATFORM == Platform.MACOS:
        plist_name = f"com.pilot.{target}"
        plist_path = _launch_agent_path(plist_name)
        if not os.path.exists(plist_path):
            return None
        code, _out, _err = await run_command(["launchctl", "list", plist_name])
        try:
            with open(plist_path, "rb") as handle:
                definition = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException, ValueError, ExpatError):
            return ObservedSchedule(command=None, loaded=code == 0)
        arguments = definition.get("ProgramArguments") if isinstance(definition, dict) else None
        if isinstance(arguments, list) and len(arguments) == 3 and arguments[:2] == ["/bin/bash", "-c"]:
            command = str(arguments[2])
        elif isinstance(arguments, list):
            command = " ".join(str(item) for item in arguments)
        else:
            command = None
        return ObservedSchedule(command=command, loaded=code == 0)

    code, out, err = await run_command(["crontab", "-l"])
    if code != 0:
        error = f"{out}\n{err}".casefold()
        if "no crontab" in error:
            return None
        raise RuntimeError(f"Crontab observation failed: {err.strip() or out.strip()}")
    for line in out.splitlines():
        entry = _crontab_entry(line, target)
        if entry is None:
            continue
        fields = entry.split()
        width = 1 if fields and fields[0].startswith("@") else 5
        return ObservedSchedule(command=" ".join(fields[width:]), schedule=" ".join(fields[:width]))
    return None


async def schedule_exists(target: str) -> bool:
    """Observe whether an exact Pilot scheduled-task target still exists."""
    if not target:
        raise ValueError("Scheduled task target is required")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["schtasks", "/query", "/tn", target, "/fo", "LIST", "/nh"])
        if code == 0:
            return True
        error = f"{out}\n{err}".casefold()
        if "cannot find" in error or "does not exist" in error or "not found" in error:
            return False
        raise RuntimeError(f"Task observation failed: {err.strip() or out.strip()}")

    if CURRENT_PLATFORM == Platform.MACOS:
        plist_name = f"com.pilot.{target}"
        plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}.plist")
        code, out, err = await run_command(["launchctl", "list", plist_name])
        return os.path.exists(plist_path) or code == 0

    code, out, err = await run_command(["crontab", "-l"])
    if code != 0:
        error = f"{out}\n{err}".casefold()
        if "no crontab" in error:
            return False
        raise RuntimeError(f"Crontab observation failed: {err.strip() or out.strip()}")
    return any(_crontab_entry(line, target) is not None for line in out.splitlines())


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
        code, existing, _ = await run_command(["crontab", "-l"])
        if code != 0:
            return "No crontab configured"

        lines = [l for l in existing.strip().split("\n") if _crontab_entry(l, target) is None]
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
