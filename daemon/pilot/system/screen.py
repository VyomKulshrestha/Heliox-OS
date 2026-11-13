"""Screen management — brightness, screenshots.

Cross-platform: Windows (PowerShell/WMI), Linux (xrandr/brightnessctl),
macOS (brightness/screencapture).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.screen")


async def brightness_get() -> str:
    """Get current screen brightness."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness"
        )
        if code == 0 and out.strip():
            return f"Brightness: {out.strip()}%"
        return "Brightness: Unable to read (desktop monitor may not support WMI brightness)"

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["brightness", "-l"])
        if code != 0:
            code, out, err = await run_command(
                ["osascript", "-e", 'tell application "System Preferences" to get brightness']
            )
        return f"Brightness: {out.strip()}"

    else:  # Linux
        code, out, err = await run_command(["brightnessctl", "get"])
        if code == 0:
            max_code, max_out, _ = await run_command(["brightnessctl", "max"])
            try:
                current = int(out.strip())
                maximum = int(max_out.strip())
                percent = round(current / maximum * 100)
                return f"Brightness: {percent}%"
            except ValueError:
                return f"Brightness: {out.strip()}"
        # Fallback
        code, out, err = await run_command(["xrandr", "--verbose"])
        return f"Brightness info:\n{out.strip()}"


async def brightness_set(level: int) -> str:
    """Set screen brightness (0-100)."""
    level = max(0, min(100, level))

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            f"(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"
        )
        if code != 0:
            code, out, err = await run_powershell(
                f"$brightness = {level}; "
                f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
                f".WmiSetBrightness(1, $brightness)"
            )

    elif CURRENT_PLATFORM == Platform.MACOS:
        # brightness value is 0.0-1.0 on macOS
        val = level / 100.0
        code, out, err = await run_command(["brightness", str(val)])

    else:  # Linux
        code, out, err = await run_command(["brightnessctl", "set", f"{level}%"])
        if code != 0:
            # Fallback xrandr
            val = level / 100.0
            code, out, err = await run_command(["xrandr", "--output", "eDP-1", "--brightness", str(val)])

    if code != 0:
        raise RuntimeError(f"Brightness set failed: {err.strip()}")
    return f"Brightness set to {level}%"


def default_screenshot_path() -> str:
    """Where a screenshot goes when no path is given: where this system keeps the person's pictures, found the way
    the system names it (screenshot_folder). The action's and the RPC's answer says the full path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(screenshot_folder() / f"screenshot_{timestamp}.png")


def screenshot_folder() -> Path:
    """The folder screenshots go to, as each system names it:
    - Windows: the Pictures known folder (SHGetKnownFolderPath), which follows a redirected or OneDrive-backed
      Pictures, whatever its name in the person's language;
    - macOS: the folder chosen in the Screenshot app's Options (com.apple.screencapture location), else the Desktop
      (its folder is "Desktop" on disk in every language; Finder only shows a translated name);
    - Linux: XDG_PICTURES_DIR from the person's user-dirs.dirs (a translated name such as "Bilder" or "Images"),
      else the home folder.
    Under a home folder that is not this account's own (HOME and USERPROFILE pointed elsewhere, as in tests and CI),
    the system's answer would be the real account's folder, so the fake home's own Pictures or Desktop is used."""
    home = Path(os.path.expanduser("~"))
    own_home = _is_account_home(home)
    if CURRENT_PLATFORM == Platform.WINDOWS:
        known = _windows_pictures() if own_home else None
        return known or home / "Pictures"
    if CURRENT_PLATFORM == Platform.MACOS:
        chosen = _mac_screencapture_location() if own_home else None
        return chosen or home / "Desktop"
    return xdg_pictures_dir(home, os.environ.get("XDG_CONFIG_HOME")) or home


def _is_account_home(home: Path) -> bool:
    from pilot.real_folders import account_home

    real = account_home()
    try:
        return real is not None and os.path.samefile(home, real)
    except OSError:
        return False


def _windows_pictures() -> Path | None:
    """The Pictures known folder (FOLDERID_Pictures), or None when Windows does not say."""
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        folder_id = uuid.UUID("33E28130-4E1E-4676-835A-98395C3BC3BB")  # FOLDERID_Pictures
        guid = (ctypes.c_byte * 16).from_buffer_copy(folder_id.bytes_le)
        found = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, wintypes.HANDLE(0), ctypes.byref(found)) != 0:
            return None
        try:
            return Path(found.value) if found.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(found)  # type: ignore[attr-defined]
    except (OSError, AttributeError, ValueError):
        return None


def _mac_screencapture_location() -> Path | None:
    """The folder chosen for screenshots in the Screenshot app (com.apple.screencapture location), when it exists."""
    import subprocess

    try:
        done = subprocess.run(
            ["defaults", "read", "com.apple.screencapture", "location"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    location = done.stdout.strip()
    if done.returncode != 0 or not location:
        return None
    folder = Path(os.path.expanduser(location))
    return folder if folder.is_dir() else None


def xdg_pictures_dir(home: Path, config_home: str | None = None) -> Path | None:
    """XDG_PICTURES_DIR from user-dirs.dirs ("$HOME/Bilder"), when it is set to a folder other than the home folder
    itself (the spec's way of turning it off). Read as text, as xdg-user-dir does; no shell is run."""
    base = Path(config_home) if config_home and os.path.isabs(config_home) else home / ".config"
    try:
        text = (base / "user-dirs.dirs").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("XDG_PICTURES_DIR="):
            continue
        value = line.split("=", 1)[1].strip().strip('"')
        if value.startswith("$HOME"):
            value = str(home) + value[len("$HOME") :]
        elif not value.startswith("/"):
            return None  # the spec allows only $HOME/... or an absolute path
        folder = Path(value)
        return None if folder == home else folder
    return None


async def screenshot(
    output_path: str | None = None,
    region: str | None = None,
) -> str:
    """Take a screenshot."""
    if output_path is None:
        output_path = default_screenshot_path()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if CURRENT_PLATFORM == Platform.WINDOWS:
        # Use PowerShell with .NET. The path is data ($HelioxArgs), never part of the script: it comes from a plan
        # or the home folder, and any quote in it (C:\Users\O'Brien, or a typographic one) would end a string.
        code, out, err = await run_powershell(
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); "
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
            "$graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size); "
            "$bitmap.Save($HelioxArgs['out']); "
            "$graphics.Dispose(); $bitmap.Dispose(); "
            "'Screenshot saved to ' + $HelioxArgs['out']",
            values={"out": output_path},
        )

    elif CURRENT_PLATFORM == Platform.MACOS:
        args = ["screencapture"]
        if region == "active_window":
            args.append("-w")
        args.append(output_path)
        code, out, err = await run_command(args)

    else:  # Linux
        from pilot.system import portal

        if portal.wayland_session():
            # Wayland: only the desktop's Screenshot portal may read the screen; gnome-screenshot and scrot see
            # nothing outside GNOME and X11. The portal gives the whole screen (it has no window or area choice
            # without its interactive dialog), and asks the person once where the desktop wants that.
            import asyncio

            data = await asyncio.to_thread(portal.capture_png, None)
            if data:
                Path(output_path).write_bytes(data)
                return f"Screenshot saved to {output_path}"
            logger.info("The Screenshot portal gave no image; trying gnome-screenshot and grim")
        if region == "active_window":
            code, out, err = await run_command(["gnome-screenshot", "-w", "-f", output_path])
        elif region and region != "fullscreen":
            # region format: "x,y,w,h"
            code, out, err = await run_command(["gnome-screenshot", "-a", "-f", output_path])
        else:
            code, out, err = await run_command(["gnome-screenshot", "-f", output_path])

        if code != 0 and portal.wayland_session():
            code, out, err = await run_command(["grim", output_path])  # wlroots compositors (sway, Hyprland)
        if code != 0:
            # Fallback to scrot (X11)
            code, out, err = await run_command(["scrot", output_path])

    if code != 0:
        raise RuntimeError(f"Screenshot failed: {err.strip()}")
    return f"Screenshot saved to {output_path}"
