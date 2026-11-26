"""Refuses commands that change the computer when Heliox runs under the test suite or a scratch-folder script.

The test suite (tests/folder_guard.py) and scratch scripts (pilot.real_folders.use_scratch_folders) mark a scratch
root in HELIOX_TEST_ROOT. While that mark is present, every place that runs a system-changing program asks
``check_command`` first: the command runners in ``platform_detect`` (run_command, run_powershell), the snapshot
manager's own runner, systemctl, and the mail and webhook senders. A command that would take a restore point or a
snapshot, install or remove software, start or stop a service, write the registry, power the computer off, change a
scheduled task, join a network, mount a disk, lock the session, or send mail or a webhook is refused with
``SystemChangeRefused`` and recorded, so the suite can report it. Reads (queries, listings) pass.

A test that truly needs the real system opts in with HELIOX_TEST_REAL_SYSTEM=1, which is honoured only where CI is
set. A daemon a person runs never has the mark, so nothing here changes what Heliox does for them.
"""

from __future__ import annotations

import os
import re
import threading

ALLOW_VARIABLE = "HELIOX_TEST_REAL_SYSTEM"

# Program (lower-case base name without .exe) to the arguments that make it change the system; None means any use.
_PROGRAMS: dict[str, tuple[str, ...] | None] = {
    "shutdown": None,
    "timeshift": ("--create", "--restore", "--delete", "--delete-all"),
    "snapper": ("create", "delete", "rollback", "undochange"),
    "btrfs": ("snapshot", "delete", "set-default"),
    "tmutil": ("localsnapshot", "restore", "deletelocalsnapshots", "startbackup", "enable", "disable"),
    "vssadmin": ("create", "delete", "resize"),
    "systemctl": (
        "start",
        "stop",
        "restart",
        "reload",
        "enable",
        "disable",
        "mask",
        "unmask",
        "poweroff",
        "reboot",
        "suspend",
        "hibernate",
        "isolate",
        "kill",
    ),
    "sc": ("start", "stop", "config", "create", "delete"),
    "apt-get": ("install", "remove", "purge", "upgrade", "dist-upgrade", "update", "autoremove"),
    "apt": ("install", "remove", "purge", "upgrade", "full-upgrade", "update", "autoremove"),
    "dnf": ("install", "remove", "erase", "upgrade", "update", "check-update", "autoremove"),
    "yum": ("install", "remove", "erase", "upgrade", "update"),
    "pacman": ("-S", "-Sy", "-Syu", "-Su", "-R", "-Rs", "-Rns", "-U"),
    "zypper": ("install", "in", "remove", "rm", "update", "up", "dup"),
    "brew": ("install", "uninstall", "remove", "upgrade", "update", "reinstall"),
    "winget": ("install", "uninstall", "upgrade", "update"),
    "choco": ("install", "uninstall", "upgrade"),
    "flatpak": ("install", "uninstall", "update"),
    "snap": ("install", "remove", "refresh"),
    "reg": ("add", "delete", "import", "restore", "load", "unload"),
    "setx": None,
    "schtasks": ("/create", "/delete", "/change", "/run", "/end"),
    "crontab": None,  # writes a crontab; "crontab -l" is let through below
    "launchctl": ("load", "unload", "bootstrap", "bootout", "enable", "disable", "kickstart"),
    "nmcli": ("connect", "disconnect", "up", "down", "add", "delete", "modify", "on", "off"),
    "netsh": ("connect", "disconnect", "set", "add", "delete"),
    "networksetup": None,
    "mount": None,
    "umount": None,
    "diskutil": ("mount", "unmount", "unmountdisk", "eject", "erase", "erasedisk", "partitiondisk"),
    "loginctl": ("lock-session", "lock-sessions", "terminate-session", "terminate-user", "kill-session"),
    "rundll32.exe": None,
    "rundll32": None,
    "pmset": ("sleepnow", "displaysleepnow", "schedule", "restoredefaults"),
    "pkexec": None,
    "sudo": None,
}

# PowerShell cmdlets that change the system, matched in a script's text.
_CMDLETS = re.compile(
    r"(?<!Get-Command )\b(Checkpoint-Computer|Restore-Computer|Enable-ComputerRestore|Disable-ComputerRestore|Stop-Computer|"
    r"Restart-Computer|Set-ItemProperty|New-ItemProperty|Remove-ItemProperty|Remove-Item\s+-Path\s+'?Registry::|"
    r"Register-ScheduledTask|Unregister-ScheduledTask|Set-ScheduledTask|Start-ScheduledTask|"
    r"Start-Service|Stop-Service|Restart-Service|Set-Service|New-Service|Install-Package|Uninstall-Package|"
    r"Add-AppxPackage|Remove-AppxPackage|Mount-DiskImage|Dismount-DiskImage|Send-MailMessage|LockWorkStation|"
    r"SetSuspendState)\b",
    re.IGNORECASE,
)

_lock = threading.Lock()
_attempts: list[tuple[str, str]] = []


class SystemChangeRefused(PermissionError):
    """A system-changing command was refused because this process runs under the test suite or a scratch script."""


def guarded() -> bool:
    """True under the test suite or a scratch-folder script, unless a CI run opted in to the real system."""
    from pilot.real_folders import _marked_root

    if _marked_root() is None:
        return False
    return not (os.environ.get(ALLOW_VARIABLE) == "1" and os.environ.get("CI"))


def _program(name: str) -> str:
    base = re.split(r"[\\/]", name.strip().strip('"'))[-1].lower()
    return base[:-4] if base.endswith(".exe") and base[:-4] in _PROGRAMS else base


def system_change(args: list[str]) -> str | None:
    """The reason ``args`` would change the system, or None for a read or an ordinary program."""
    if not args:
        return None
    words = [str(word) for word in args]
    program = _program(words[0])
    if program in {"powershell", "pwsh", "powershell.exe", "pwsh.exe"}:
        return powershell_change(" ".join(words[1:]))
    if program in {"pkexec", "sudo"}:
        # The elevated command itself is what matters; a bare elevation is refused too.
        return system_change(words[1:]) or f"{program} (elevation)"
    if program not in _PROGRAMS:
        return None
    verbs = _PROGRAMS[program]
    rest = [word.lower() for word in words[1:]]
    if program in {"mount", "umount"} and not rest:
        return None  # a bare "mount" lists what is mounted
    if program == "crontab":
        return None if rest[:1] == ["-l"] else "crontab (writes the scheduled tasks)"
    if verbs is None:
        return program
    lowered = {verb.lower() for verb in verbs}
    hit = next((word for word in rest if word in lowered), None)
    return f"{program} {hit}" if hit else None


def powershell_change(script: str) -> str | None:
    found = _CMDLETS.search(script or "")
    return found.group(1) if found else None


def check(kind: str, what: str | None) -> None:
    """Refuse and record ``what`` (a reason from system_change, or a description) when guarded."""
    if not what or not guarded():
        return
    with _lock:
        _attempts.append((kind, what))
    raise SystemChangeRefused(
        f"Refused under the test suite: {kind} would change the real system ({what}). Pass a fake instead."
    )


def check_command(args: list[str], kind: str = "command") -> None:
    check(kind, system_change(args))


def check_powershell(script: str) -> None:
    check("PowerShell", powershell_change(script))


def attempts() -> list[tuple[str, str]]:
    with _lock:
        return list(_attempts)


def clear() -> None:
    with _lock:
        _attempts.clear()


import smtplib as _smtplib  # noqa: E402

# The real classes, taken when this module is first imported (tests/conftest.py imports it before any test runs).
_REAL_SMTP = (_smtplib.SMTP, _smtplib.SMTP_SSL)


def check_mail(host: str, smtp_class: object) -> None:
    """Refuse a real SMTP connection; a test that replaced smtplib's class with a fake passes."""
    if smtp_class in _REAL_SMTP:
        check("mail", f"SMTP connection to {host}")


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_READ_METHODS = {"GET", "HEAD", "OPTIONS"}


def check_http(method: str, host: str, *, real_client: bool) -> None:
    """Refuse a real request that may change something elsewhere (a webhook, an API write); reads and loopback pass."""
    if real_client and method.upper() not in _READ_METHODS and host.lower() not in _LOOPBACK:
        check("webhook or API call", f"{method.upper()} to {host}")
