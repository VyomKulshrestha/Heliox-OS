"""Cross-platform package management.

Windows: winget / choco
Linux: apt / dnf / pacman
macOS: brew
"""

from __future__ import annotations

import logging

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


async def package_update() -> str:
    """Update package lists / upgrade packages."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_command(["winget", "upgrade", "--all", "--accept-package-agreements"], timeout=300)
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["brew", "update"], timeout=120)
        if code == 0:
            code2, out2, err2 = await run_command(["brew", "upgrade"], timeout=300)
            out = out + "\n" + out2
    else:
        mgr = await _detect_linux_pkg_manager()
        if mgr == "apt-get":
            code, out, err = await run_command(["apt-get", "update"], root=True, timeout=120)
        elif mgr == "dnf":
            code, out, err = await run_command(["dnf", "check-update"], root=True, timeout=120)
        elif mgr == "pacman":
            code, out, err = await run_command(["pacman", "-Sy"], root=True, timeout=120)
        else:
            code, out, err = await run_command(["apt-get", "update"], root=True, timeout=120)

    if code != 0:
        raise RuntimeError(f"Package update failed: {err.strip()}")
    return "Package lists updated"


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
