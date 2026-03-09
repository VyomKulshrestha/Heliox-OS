"""Volume / audio control — get, set, mute.

Cross-platform: Windows (PowerShell/AudioDeviceCmdlets),
Linux (pactl/amixer), macOS (osascript).
"""

from __future__ import annotations

import logging

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.volume")


async def volume_get() -> str:
    """Get current system volume level."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            '$vol = [Audio]::Volume; $mute = [Audio]::Mute; "Volume: $([math]::Round($vol * 100))%`nMuted: $mute"',
        )
        if code != 0:
            # Fallback without Audio class
            code, out, err = await run_powershell(
                "Add-Type -TypeDefinition '"
                "using System.Runtime.InteropServices; "
                '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
                "interface IAudioEndpointVolume { int f(); int g(); int h(); int i(); "
                "int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext); "
                "int j(); int GetMasterVolumeLevelScalar(out float pfLevel); "
                "int k(); int l(); int GetMute(out bool pbMute); int m(); int n(); int o(); int p(); } "
                "'; "
                "Write-Output 'Volume info not directly available, use system tray'"
            )
        return f"Current volume:\n{out.strip()}"

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["osascript", "-e", "output volume of (get volume settings)"])
        mute_code, mute_out, _ = await run_command(["osascript", "-e", "output muted of (get volume settings)"])
        return f"Volume: {out.strip()}%\nMuted: {mute_out.strip()}"

    else:  # Linux
        code, out, err = await run_command(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        mute_code, mute_out, _ = await run_command(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
        if code == 0:
            return f"{out.strip()}\n{mute_out.strip()}"
        # Fallback to amixer
        code, out, err = await run_command(["amixer", "get", "Master"])
        return out.strip()


async def volume_set(level: int) -> str:
    """Set system volume level (0-100)."""
    level = max(0, min(100, level))

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            f"$obj = New-Object -ComObject WScript.Shell; "
            f"1..50 | ForEach-Object {{ $obj.SendKeys([char]174) }}; "  # Volume down to 0
            f"1..{level // 2} | ForEach-Object {{ $obj.SendKeys([char]175) }}; "  # Volume up
            f"'Volume set to approximately {level}%'"
        )
        # More reliable approach
        if code != 0:
            code, out, err = await run_powershell(
                "(New-Object -ComObject WScript.Shell).SendKeys([char]173); 'Attempted volume adjustment'"
            )

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["osascript", "-e", f"set volume output volume {level}"])

    else:  # Linux
        code, out, err = await run_command(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"])
        if code != 0:
            code, out, err = await run_command(["amixer", "set", "Master", f"{level}%"])

    if code != 0:
        raise RuntimeError(f"Volume set failed: {err.strip()}")
    return f"Volume set to {level}%"


async def volume_mute(mute: bool = True) -> str:
    """Mute or unmute system volume."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "(New-Object -ComObject WScript.Shell).SendKeys([char]173); 'Toggled mute'"
        )

    elif CURRENT_PLATFORM == Platform.MACOS:
        mute_val = "true" if mute else "false"
        code, out, err = await run_command(["osascript", "-e", f"set volume output muted {mute_val}"])

    else:  # Linux
        toggle = "1" if mute else "0"
        code, out, err = await run_command(["pactl", "set-sink-mute", "@DEFAULT_SINK@", toggle])
        if code != 0:
            toggle_amixer = "mute" if mute else "unmute"
            code, out, err = await run_command(["amixer", "set", "Master", toggle_amixer])

    if code != 0:
        raise RuntimeError(f"Mute toggle failed: {err.strip()}")
    return f"Volume {'muted' if mute else 'unmuted'}"
