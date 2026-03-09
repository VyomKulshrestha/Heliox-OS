"""Parameter sanitizer — prevents injection and validates formats.

All parameters from LLM output pass through these checks before execution.
Updated for cross-platform support and expanded action types.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.actions import DBusParams
    from pilot.config import PilotConfig

logger = logging.getLogger("pilot.security.sanitizer")

SHELL_METACHARACTERS = re.compile(r"[;&|`$(){}!\[\]<>\n\r\\]")
# Relaxed for Windows path separators
SHELL_METACHARACTERS_WIN = re.compile(r"[;&|`$(){}!\[\]<>\n\r]")
PATH_TRAVERSAL = re.compile(r"(^|[/\\])\.\.([\\/]|$)")
VALID_PACKAGE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.+\-_]+$")
VALID_SERVICE_NAME = re.compile(r"^[a-zA-Z0-9_@.\-]+$")
VALID_GSETTINGS_SCHEMA = re.compile(r"^org\.[a-zA-Z0-9.\-]+$")
VALID_GSETTINGS_KEY = re.compile(r"^[a-z][a-z0-9\-]+$")
VALID_DBUS_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]+$")
VALID_DBUS_PATH = re.compile(r"^/[a-zA-Z0-9_/]+$")

# Expanded safe command list — these are always allowed without elevated permission
SAFE_COMMANDS = frozenset(
    {
        # Info / read-only
        "echo",
        "cat",
        "ls",
        "dir",
        "df",
        "du",
        "whoami",
        "hostname",
        "uname",
        "date",
        "uptime",
        "free",
        "lsb_release",
        "lscpu",
        "lsblk",
        "ip",
        "ss",
        "ping",
        "dig",
        "nslookup",
        "traceroute",
        "tracert",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "grep",
        "find",
        "which",
        "where",
        "ps",
        "env",
        "id",
        "top",
        "htop",
        "mount",
        "lspci",
        "lsusb",
        "nmcli",
        "timedatectl",
        "hostnamectl",
        "loginctl",
        "journalctl",
        "sensors",
        "xrandr",
        "xdg-open",
        "pw-cli",
        "pactl",
        "flatpak",
        "snap",
        "dpkg",
        "apt",
        "apt-get",
        "apt-cache",
        "systemctl",
        "neofetch",
        "inxi",
        "tree",
        "stat",
        "file",
        "md5sum",
        "sha256sum",
        "realpath",
        "dirname",
        "basename",
        "tee",
        "diff",
        "cut",
        "tr",
        # Network
        "curl",
        "wget",
        "netsh",
        "ipconfig",
        "ifconfig",
        "arp",
        "nbtstat",
        # Process
        "tasklist",
        "taskkill",
        "kill",
        "pkill",
        "pgrep",
        # System
        "systeminfo",
        "ver",
        "powershell",
        "pwsh",
        # File
        "mkdir",
        "rmdir",
        "cp",
        "mv",
        "rm",
        "touch",
        "chmod",
        "chown",
        "copy",
        "move",
        "del",
        "type",
        "more",
        # Python / dev
        "python",
        "python3",
        "pip",
        "pip3",
        "npm",
        "node",
        "git",
        # Package managers
        "winget",
        "choco",
        "brew",
        "dnf",
        "pacman",
        "zypper",
        "yum",
        # Utils
        "tar",
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "7z",
        "sed",
        "awk",
        "xargs",
        "yes",
        "ssh",
        "scp",
        "rsync",
        "docker",
        "docker-compose",
        "code",
        "notepad",
        "nano",
        "vim",
        "vi",
        "start",
        "open",  # Audio/display
        "amixer",
        "brightnessctl",
        "xdotool",
        "wmctrl",
        "screencapture",
        "gnome-screenshot",
        "scrot",
        # Scheduling
        "crontab",
        "schtasks",
        "at",
        # Disk
        "fdisk",
        "blkid",
        "umount",
        "icacls",
        "attrib",
    }
)

VALID_URL = re.compile(r"^https?://[^\s]+$")


class SanitizationError(Exception):
    def __init__(self, action_index: int, message: str) -> None:
        self.action_index = action_index
        super().__init__(f"Action [{action_index}]: {message}")


class Sanitizer:
    def __init__(self, config: PilotConfig) -> None:
        self._config = config
        self._is_windows = sys.platform == "win32"

    def validate_path(self, path: str, idx: int) -> None:
        if not path:
            raise SanitizationError(idx, "Empty path")

        metachar_regex = SHELL_METACHARACTERS_WIN if self._is_windows else SHELL_METACHARACTERS
        if metachar_regex.search(path):
            raise SanitizationError(idx, f"Path contains shell metacharacters: {path!r}")

        if PATH_TRAVERSAL.search(path):
            raise SanitizationError(idx, f"Path traversal detected: {path!r}")

        # Platform-aware absolute path check
        if self._is_windows:
            p = PureWindowsPath(path)
            if not p.is_absolute():
                raise SanitizationError(idx, f"Path must be absolute: {path!r}")
        else:
            p = PurePosixPath(path)
            if not p.is_absolute():
                raise SanitizationError(idx, f"Path must be absolute: {path!r}")

        if any(part == ".." for part in p.parts):
            raise SanitizationError(idx, f"Path contains ..: {path!r}")

    def validate_package_name(self, name: str, idx: int) -> None:
        if not VALID_PACKAGE_NAME.match(name):
            raise SanitizationError(idx, f"Invalid package name: {name!r} (must match {VALID_PACKAGE_NAME.pattern})")

    def validate_service_name(self, name: str, idx: int) -> None:
        if not VALID_SERVICE_NAME.match(name):
            raise SanitizationError(idx, f"Invalid service name: {name!r} (must match {VALID_SERVICE_NAME.pattern})")

    def validate_gsettings_schema(self, schema: str, idx: int) -> None:
        if not VALID_GSETTINGS_SCHEMA.match(schema):
            raise SanitizationError(idx, f"Invalid gsettings schema: {schema!r}")

    def validate_gsettings_key(self, key: str, idx: int) -> None:
        if not VALID_GSETTINGS_KEY.match(key):
            raise SanitizationError(idx, f"Invalid gsettings key: {key!r}")

    def validate_shell_command(self, command: str, args: list[str], idx: int) -> None:
        """Validate a shell command.

        With the expanded agent model, we allow all commands from the safe list.
        Unknown commands are still blocked unless the config allows unrestricted mode.
        """
        if command not in SAFE_COMMANDS:
            # Check if unrestricted shell is enabled in config
            if hasattr(self._config, "security") and getattr(self._config.security, "unrestricted_shell", False):
                logger.warning("Allowing non-whitelisted command in unrestricted mode: %s", command)
                return
            raise SanitizationError(
                idx,
                f"Command '{command}' is not in the safe whitelist. "
                f"Enable 'unrestricted_shell' in security config to allow all commands.",
            )

        # For non-Windows, check shell metacharacters in args
        if not self._is_windows:
            for i, arg in enumerate(args):
                if SHELL_METACHARACTERS.search(arg):
                    raise SanitizationError(idx, f"Argument {i} contains shell metacharacters: {arg!r}")

    def validate_url(self, url: str, idx: int) -> None:
        if not url:
            raise SanitizationError(idx, "Empty URL")
        normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"
        if not VALID_URL.match(normalized):
            raise SanitizationError(idx, f"Invalid URL: {url!r}")

    def validate_dbus_params(self, params: DBusParams, idx: int) -> None:
        if not VALID_DBUS_NAME.match(params.service):
            raise SanitizationError(idx, f"Invalid DBus service name: {params.service!r}")
        if not VALID_DBUS_PATH.match(params.object_path):
            raise SanitizationError(idx, f"Invalid DBus object path: {params.object_path!r}")
        if not VALID_DBUS_NAME.match(params.interface):
            raise SanitizationError(idx, f"Invalid DBus interface: {params.interface!r}")
        if not VALID_DBUS_NAME.match(params.method):
            raise SanitizationError(idx, f"Invalid DBus method: {params.method!r}")
