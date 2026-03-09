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


async def screenshot(
    output_path: str | None = None,
    region: str | None = None,
) -> str:
    """Take a screenshot."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if CURRENT_PLATFORM == Platform.WINDOWS:
            output_path = os.path.join(os.path.expanduser("~"), "Pictures", f"screenshot_{timestamp}.png")
        elif CURRENT_PLATFORM == Platform.MACOS:
            output_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{timestamp}.png")
        else:
            output_path = os.path.join(os.path.expanduser("~"), f"screenshot_{timestamp}.png")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if CURRENT_PLATFORM == Platform.WINDOWS:
        # Use PowerShell with .NET
        code, out, err = await run_powershell(
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            f"$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); "
            f"$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
            f"$graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size); "
            f"$bitmap.Save('{output_path}'); "
            f"$graphics.Dispose(); $bitmap.Dispose(); "
            f"'Screenshot saved to {output_path}'"
        )

    elif CURRENT_PLATFORM == Platform.MACOS:
        args = ["screencapture"]
        if region == "active_window":
            args.append("-w")
        args.append(output_path)
        code, out, err = await run_command(args)

    else:  # Linux
        if region == "active_window":
            code, out, err = await run_command(["gnome-screenshot", "-w", "-f", output_path])
        elif region and region != "fullscreen":
            # region format: "x,y,w,h"
            code, out, err = await run_command(["gnome-screenshot", "-a", "-f", output_path])
        else:
            code, out, err = await run_command(["gnome-screenshot", "-f", output_path])

        if code != 0:
            # Fallback to scrot
            code, out, err = await run_command(["scrot", output_path])

    if code != 0:
        raise RuntimeError(f"Screenshot failed: {err.strip()}")
    return f"Screenshot saved to {output_path}"
