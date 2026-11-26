"""Environment variable management — get, set, list.

Cross-platform: handles both session and persistent environment variables.
"""

from __future__ import annotations

import logging
import os
import re
import shlex

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.environment")

_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_name(name: str) -> None:
    if not _ENV_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Environment variable names must start with a letter or underscore and contain "
            "only letters, numbers, and underscores."
        )


async def env_get(name: str) -> str:
    """Get an environment variable value."""
    _validate_env_name(name)
    value = os.environ.get(name)
    if value is None:
        return f"Environment variable '{name}' is not set"
    return f"{name}={value}"


async def env_set(name: str, value: str, persistent: bool = False) -> str:
    """Set an environment variable.

    If persistent=True, write to the appropriate shell profile or system config.
    """
    _validate_env_name(name)

    # Set for current process
    os.environ[name] = value

    if persistent:
        if CURRENT_PLATFORM == Platform.WINDOWS:
            # Use setx for persistent user-level env var
            code, out, err = await run_command(["setx", name, value])
            if code != 0:
                raise RuntimeError(f"setx failed: {err.strip()}")
            return f"Set {name}={value} (persistent, user scope)"

        else:
            # ~/.zshrc on macOS (its default shell), ~/.bashrc on Linux.
            profile = persistent_profile()
            await _append_env_to_profile(profile, name, value)
            shown = "~/.zshrc" if CURRENT_PLATFORM == Platform.MACOS else "~/.bashrc"
            return f"Set {name}={value} (persistent, added to {shown})"

    return f"Set {name}={value} (session only)"


def persistent_profile() -> str:
    """The shell profile a persistent variable is written to on macOS and Linux."""
    return os.path.expanduser("~/.zshrc" if CURRENT_PLATFORM == Platform.MACOS else "~/.bashrc")


# Reads the user's own variable as Windows stores it (HKCU\Environment, environment names unexpanded) in a new
# process; the name is data. Answers JSON so nothing depends on Windows' language.
_WINDOWS_USER_VARIABLE = (
    "$key = Get-Item -LiteralPath 'Registry::HKEY_CURRENT_USER\\Environment' -ErrorAction SilentlyContinue; "
    "$name = [string]$HelioxArgs['name']; "
    "if (-not $key -or -not ($key.GetValueNames() | Where-Object { $_ -eq $name })) { '{\"present\":false}'; exit 0 }; "
    "[pscustomobject]@{ present = $true; value = [string]$key.GetValue($name, $null, "
    "[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) } | ConvertTo-Json -Compress"
)


async def persistent_value(name: str) -> str | None:
    """The value a new shell or program will see for a persistent variable, read from where it is stored:
    the user's environment in the registry on Windows (a new PowerShell process), the shell profile on macOS and
    Linux (the last assignment of the name in it). None when it is not there. Raises when the read fails."""
    import asyncio
    import json

    _validate_env_name(name)
    if CURRENT_PLATFORM == Platform.WINDOWS:
        from pilot.system.platform_detect import run_powershell

        code, out, err = await run_powershell(_WINDOWS_USER_VARIABLE, values={"name": name}, timeout=20)
        if code != 0:
            raise RuntimeError(f"Reading the user environment failed: {err.strip()}")
        answer = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        return str(answer["value"]) if answer.get("present") else None

    def read() -> str | None:
        try:
            with open(persistent_profile(), encoding="utf-8") as handle:
                lines = handle.readlines()
        except FileNotFoundError:
            return None
        assignment = re.compile(rf"^\s*(?:export\s+)?{name}=")
        value = None
        for line in lines:
            if assignment.match(line):
                try:
                    words = shlex.split(line, comments=True)
                except ValueError:
                    value = None
                    continue
                word = words[-1] if words else ""
                value = word.split("=", 1)[1] if word.startswith(f"{name}=") else None
        return value

    return await asyncio.to_thread(read)


async def _append_env_to_profile(profile_path: str, name: str, value: str) -> None:
    """Append or update an env var in a shell profile file."""
    _validate_env_name(name)
    marker = f"# pilot-env:{name}"

    try:
        with open(profile_path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # Remove old entry
    new_lines = [l for l in lines if marker not in l]
    # POSIX shell quoting keeps command substitutions, newlines, quotes, and
    # metacharacters literal when the profile is sourced.
    new_lines.append(f"export {name}={shlex.quote(value)}  {marker}\n")

    with open(profile_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


async def env_list(filter_prefix: str | None = None) -> str:
    """List environment variables, optionally filtered by prefix."""
    entries: list[str] = []
    for k, v in sorted(os.environ.items()):
        if filter_prefix and not k.upper().startswith(filter_prefix.upper()):
            continue
        # Truncate long values for readability
        display_v = v if len(v) <= 200 else v[:200] + "..."
        entries.append(f"{k}={display_v}")

    return "\n".join(entries[:100]) if entries else "No matching environment variables"
