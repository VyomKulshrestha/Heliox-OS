"""Cross-platform package management.

Windows: winget / choco
Linux: apt / dnf / pacman
macOS: brew
"""

from __future__ import annotations

import logging
import os

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.package_mgr")


async def _detect_linux_pkg_manager() -> str:
    """Detect which package manager is available on Linux."""
    for mgr in ["apt-get", "dnf", "pacman", "zypper"]:
        code, _, _ = await run_command(["which", mgr])
        if code == 0:
            return mgr
    return "apt-get"  # default


async def _detect_win_pkg_manager() -> str:
    """Detect which package manager is available on Windows."""
    code, _, _ = await run_command(["winget", "--version"])
    if code == 0:
        return "winget"
    code, _, _ = await run_command(["choco", "--version"])
    if code == 0:
        return "choco"
    return "winget"


async def package_install(name: str, version: str | None = None) -> str:
    """Install a package."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        mgr = await _detect_win_pkg_manager()
        if mgr == "winget":
            args = ["winget", "install", "--accept-package-agreements", "--accept-source-agreements", name]
            if version:
                args.extend(["--version", version])
        else:
            args = ["choco", "install", "-y", name]
            if version:
                args.extend(["--version", version])
        code, out, err = await run_command(args, timeout=120)

    elif CURRENT_PLATFORM == Platform.MACOS:
        args = ["brew", "install", name]
        code, out, err = await run_command(args, timeout=120)

    else:  # Linux
        mgr = await _detect_linux_pkg_manager()
        if mgr == "apt-get":
            pkg = f"{name}={version}" if version else name
            code, out, err = await run_command(["apt-get", "install", "-y", pkg], root=True, timeout=120)
        elif mgr == "dnf":
            pkg = f"{name}-{version}" if version else name
            code, out, err = await run_command(["dnf", "install", "-y", pkg], root=True, timeout=120)
        elif mgr == "pacman":
            code, out, err = await run_command(["pacman", "-S", "--noconfirm", name], root=True, timeout=120)
        else:
            code, out, err = await run_command(["apt-get", "install", "-y", name], root=True, timeout=120)

    if code != 0:
        raise RuntimeError(f"Package install failed: {err.strip()}")
    return f"Installed package: {name}" + (f" (version {version})" if version else "")


async def package_remove(name: str) -> str:
    """Remove a package."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        mgr = await _detect_win_pkg_manager()
        if mgr == "winget":
            args = ["winget", "uninstall", name]
        else:
            args = ["choco", "uninstall", "-y", name]
        code, out, err = await run_command(args, timeout=60)

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["brew", "uninstall", name], timeout=60)

    else:
        mgr = await _detect_linux_pkg_manager()
        if mgr == "apt-get":
            code, out, err = await run_command(["apt-get", "remove", "-y", name], root=True, timeout=60)
        elif mgr == "dnf":
            code, out, err = await run_command(["dnf", "remove", "-y", name], root=True, timeout=60)
        elif mgr == "pacman":
            code, out, err = await run_command(["pacman", "-R", "--noconfirm", name], root=True, timeout=60)
        else:
            code, out, err = await run_command(["apt-get", "remove", "-y", name], root=True, timeout=60)

    if code != 0:
        raise RuntimeError(f"Package remove failed: {err.strip()}")
    return f"Removed package: {name}"


# winget answers APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE (0x8A15002B) when there is nothing to upgrade; Python
# sees it unsigned on Windows, and some wrappers report it signed.
WINGET_NOTHING_TO_UPGRADE = {0, 0x8A15002B, 0x8A15002B - (1 << 32)}

# An upgrade of everything can take a long time; a package manager killed half-way leaves a broken install, so the
# limit is generous.
UPGRADE_TIMEOUT_SECONDS = 1800

# apt refreshes its lists and upgrades under one pkexec prompt. The text is fixed (nothing from a plan goes in);
# dpkg keeps a changed configuration file instead of stopping to ask.
APT_UPGRADE_SCRIPT = (
    "apt-get update && apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold upgrade"
)


async def package_update() -> str:
    """Bring the installed packages up to date: what a person means by "update my packages".

    There is no separate upgrade action, so this refreshes the package lists and upgrades on every platform:
    winget (or Chocolatey) on Windows, Homebrew on macOS, apt, dnf or pacman on Linux."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        mgr = await _detect_win_pkg_manager()
        if mgr == "winget":
            args = ["winget", "upgrade", "--all", "--accept-package-agreements", "--accept-source-agreements"]
        else:
            args = ["choco", "upgrade", "all", "-y"]
        code, out, err = await run_command(args, timeout=UPGRADE_TIMEOUT_SECONDS)
        if mgr == "winget" and code in WINGET_NOTHING_TO_UPGRADE:
            code = 0
    elif CURRENT_PLATFORM == Platform.MACOS:
        mgr = "brew"
        code, out, err = await run_command(["brew", "update"], timeout=300)
        if code == 0:
            code, out, err = await run_command(["brew", "upgrade"], timeout=UPGRADE_TIMEOUT_SECONDS)
    else:
        mgr = await _detect_linux_pkg_manager()
        if mgr == "dnf":
            args = ["dnf", "upgrade", "-y", "--refresh"]
        elif mgr == "pacman":
            args = ["pacman", "-Syu", "--noconfirm"]
        else:
            mgr = "apt-get"
            args = ["sh", "-c", APT_UPGRADE_SCRIPT]
        code, out, err = await run_command(args, root=True, timeout=UPGRADE_TIMEOUT_SECONDS)

    if code != 0:
        raise RuntimeError(f"Package update failed ({mgr}, exit {code}): {(err.strip() or out.strip())[-500:]}")
    return f"Installed packages brought up to date with {mgr}"


async def pending_upgrades() -> list[str] | None:
    """Packages the package manager would still upgrade now, asked afresh (read-only, never as root).

    None when the manager cannot say (zypper, or a Linux system with none of the managers above). Raises when the
    manager fails to answer."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        if await _detect_win_pkg_manager() == "choco":
            code, out, err = await run_command(["choco", "outdated", "-r"], timeout=120)
            if code != 0:
                raise RuntimeError(f"choco outdated failed: {err.strip() or out.strip()}")
            return parse_choco_outdated(out)
        code, out, err = await run_command(["winget", "upgrade", "--accept-source-agreements"], timeout=120)
        rows = parse_winget_upgrade_table(out)
        if code not in WINGET_NOTHING_TO_UPGRADE and not rows:
            raise RuntimeError(f"winget upgrade (list) failed: exit {code}")
        return rows

    if CURRENT_PLATFORM == Platform.MACOS:
        import json

        env = {**os.environ, "HOMEBREW_NO_AUTO_UPDATE": "1"}
        code, out, err = await run_command(["brew", "outdated", "--json=v2"], timeout=120, env=env)
        if code != 0:
            raise RuntimeError(f"brew outdated failed: {err.strip()}")
        data = json.loads(out or "{}")
        return [
            str(item.get("name"))
            for group in ("formulae", "casks")
            for item in data.get(group, [])
            if not item.get("pinned")
        ]

    mgr = await _detect_linux_pkg_manager()
    if mgr == "dnf":
        # check-update exits 100 when updates exist, 0 when none, 1 on error.
        code, out, err = await run_command(["dnf", "-q", "check-update"], timeout=180)
        if code == 0:
            return []
        if code == 100:
            return parse_dnf_check_update(out)
        raise RuntimeError(f"dnf check-update failed: {err.strip()}")
    if mgr == "pacman":
        # -Qu reads the local sync database the upgrade just refreshed; exit 1 with no output means none.
        code, out, err = await run_command(["pacman", "-Qu"], timeout=60)
        if code == 1 and not out.strip():
            return []
        if code != 0:
            raise RuntimeError(f"pacman -Qu failed: {err.strip()}")
        return [line.split()[0] for line in out.splitlines() if line.strip()]
    if mgr == "apt-get":
        # A simulated upgrade lists what apt would install now ("Inst" lines are apt's own, untranslated); packages
        # apt deliberately keeps back (held, phased) are not upgrades it would make and are not counted.
        code, out, err = await run_command(
            ["apt-get", "-s", "-o", "Debug::NoLocking=1", "upgrade"], timeout=120, env={**os.environ, "LC_ALL": "C"}
        )
        if code != 0:
            raise RuntimeError(f"apt-get -s upgrade failed: {err.strip()}")
        return [line.split()[1] for line in out.splitlines() if line.startswith("Inst ") and len(line.split()) > 1]
    return None


def parse_choco_outdated(text: str) -> list[str]:
    """``choco outdated -r`` lines are "name|current|available|pinned"; pinned packages are not upgraded."""
    names = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[3].strip().lower() != "true":
            names.append(parts[0])
    return names


def parse_dnf_check_update(text: str) -> list[str]:
    """Package names from ``dnf -q check-update``: "name.arch  version  repo" rows, up to the next section title."""
    names = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if not line[0].isspace() and len(parts) != 3:
            break  # a section title such as "Obsoleting Packages" (translated): the upgrade list has ended
        if len(parts) == 3 and not line[0].isspace():
            names.append(parts[0])
    return names


def parse_winget_upgrade_table(text: str) -> list[str]:
    """Rows of the first table ``winget upgrade`` prints: the packages it would upgrade with ``--all``.

    Column titles and the summary line are translated, so the table is read by position: the line of dashes ends
    the header, the header gives where the last column (the source) starts, and a row has text there. The summary
    line under the table ("N upgrades available.") is shorter than that and ends it, as does a blank line."""
    import re

    # The progress spinner rewrites its line with carriage returns: keep what is left after the last one.
    lines = [line.rsplit("\r", 1)[-1] for line in text.replace("\r\n", "\n").split("\n")]
    for index, line in enumerate(lines):
        if index == 0 or len(line.strip()) < 10 or set(line.strip()) != {"-"}:
            continue
        header = lines[index - 1].rstrip()
        if not header.strip():
            return []
        last_column = len(header) - len(header.split()[-1])
        rows = []
        for row in lines[index + 1 :]:
            if len(row.rstrip()) <= last_column or not row[last_column:].strip():
                break
            rows.append(re.split(r"\s{2,}", row.strip())[0])
        return rows
    return []


async def package_search(name: str) -> str:
    """Search for packages."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["winget", "search", name], timeout=30)
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["brew", "search", name], timeout=30)
    else:
        mgr = await _detect_linux_pkg_manager()
        if mgr == "apt-get":
            code, out, err = await run_command(["apt-cache", "search", name], timeout=30)
        elif mgr == "dnf":
            code, out, err = await run_command(["dnf", "search", name], timeout=30)
        elif mgr == "pacman":
            code, out, err = await run_command(["pacman", "-Ss", name], timeout=30)
        else:
            code, out, err = await run_command(["apt-cache", "search", name], timeout=30)

    if code != 0:
        raise RuntimeError(f"Package search failed: {err.strip()}")
    lines = out.strip().split("\n")[:20]
    return "\n".join(lines) if lines[0] else f"No packages found matching '{name}'"


async def is_installed(name: str) -> bool:
    """Check if a package is installed."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, _ = await run_command(["winget", "list", name])
        return code == 0 and name.lower() in out.lower()
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, _, _ = await run_command(["brew", "list", name])
        return code == 0
    else:
        code, out, _ = await run_command(["dpkg", "-s", name])
        return code == 0 and "Status: install ok installed" in out
