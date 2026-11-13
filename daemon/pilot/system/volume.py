"""The system output volume: read it, set it, step it, mute and unmute it, and say what is true afterwards.

Each system's own volume call, read back after every change:
- Windows: Core Audio's endpoint volume on the default output device (pilot.system.windows_volume, ctypes only).
  Only when that call fails do the volume keys go out instead (WScript.Shell SendKeys), and the sentence then says
  the level could not be read back.
- macOS: ``osascript``'s "get volume settings" and "set volume", the level passed as data (heliox_args).
- Linux: ``pactl`` on the default sink, relative or absolute, else ``amixer`` on Master.
A failure raises RuntimeError with what failed, so nothing claims success that did not happen.

``volume_call_available`` says whether the system call exists here; only then may a volume command run beside a
task that holds the keyboard (pilot.conversation.quick.runs_beside_a_task), since a key press could land in the
window the task is typing in.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.volume")

STEP = 10  # "volume down", "quieter": ten points of 100

# The volume keys, only when Windows' own call failed. Constant scripts: the count is a key code repeated, not data.
_WIN_KEY_DOWN = "$o = New-Object -ComObject WScript.Shell; 1..5 | ForEach-Object { $o.SendKeys([char]174) }"
_WIN_KEY_UP = "$o = New-Object -ComObject WScript.Shell; 1..5 | ForEach-Object { $o.SendKeys([char]175) }"
_WIN_KEY_MUTE = "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"

_MAC_READ = "get volume settings"
_MAC_SET_LEVEL = "set volume output volume ((item 1 of heliox_args) as integer)"
_MAC_SET_MUTED = 'set volume output muted ((item 1 of heliox_args) is "true")'


@dataclass(frozen=True)
class VolumeState:
    level: int | None  # 0-100, None when it could not be read
    muted: bool | None


def say_state(state: VolumeState) -> str:
    """ "Volume is at 30.", "Volume is at 30, and muted.", or "" when nothing could be read."""
    if state.level is None:
        return "" if state.muted is None else ("The sound is muted." if state.muted else "The sound is on.")
    return f"Volume is at {state.level}" + (", and muted." if state.muted else ".")


def volume_call_available() -> bool:
    """Whether this system's own volume call is here (no key presses needed)."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        try:
            from pilot.system import windows_volume

            windows_volume.read()
            return True
        except Exception:
            return False
    if CURRENT_PLATFORM == Platform.MACOS:
        return shutil.which("osascript") is not None
    return shutil.which("pactl") is not None or shutil.which("amixer") is not None


# ── reading ──


def parse_mac_settings(text: str) -> VolumeState:
    """ "output volume:30, input volume:50, alert volume:100, output muted:false" -> VolumeState(30, False)."""
    level = re.search(r"output volume:\s*(\d+|missing value)", text)
    muted = re.search(r"output muted:\s*(true|false|missing value)", text)
    return VolumeState(
        int(level.group(1)) if level and level.group(1).isdigit() else None,
        {"true": True, "false": False}.get(muted.group(1)) if muted else None,
    )


def parse_pactl_volume(text: str) -> int | None:
    """ "Volume: front-left: 19661 /  30% / -31.37 dB,   front-right: ..." -> 30 (the first channel)."""
    found = re.search(r"(\d+)%", text)
    return int(found.group(1)) if found else None


def parse_pactl_mute(text: str) -> bool | None:
    found = re.search(r"Mute:\s*(yes|no)", text)
    return found.group(1) == "yes" if found else None


def parse_amixer(text: str) -> VolumeState:
    """ "Front Left: Playback 39321 [60%] [on]" -> VolumeState(60, False)."""
    found = re.search(r"\[(\d+)%\](?:\s*\[-?[\d.]+dB\])?\s*\[(on|off)\]", text)
    if found:
        return VolumeState(int(found.group(1)), found.group(2) == "off")
    level = re.search(r"\[(\d+)%\]", text)
    return VolumeState(int(level.group(1)) if level else None, None)


async def volume_state() -> VolumeState:
    """The volume as the system reports it; VolumeState(None, None) when it cannot be read."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        from pilot.system import windows_volume

        try:
            level, muted = await asyncio.to_thread(windows_volume.read)
            return VolumeState(level, muted)
        except Exception:
            logger.debug("Could not read the Windows volume", exc_info=True)
            return VolumeState(None, None)
    if CURRENT_PLATFORM == Platform.MACOS:
        code, out, _err = await run_command(["osascript", "-e", _MAC_READ])
        return parse_mac_settings(out) if code == 0 else VolumeState(None, None)
    if shutil.which("pactl"):
        code, out, _err = await run_command(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        mcode, mout, _ = await run_command(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
        if code == 0:
            return VolumeState(parse_pactl_volume(out), parse_pactl_mute(mout) if mcode == 0 else None)
    if shutil.which("amixer"):
        code, out, _err = await run_command(["amixer", "get", "Master"])
        if code == 0:
            return parse_amixer(out)
    return VolumeState(None, None)


async def volume_get() -> str:
    """The volume as one sentence ("Volume is at 30."). Raises when it cannot be read."""
    said = say_state(await volume_state())
    if not said:
        raise RuntimeError("Could not read the volume on this computer.")
    return said


# ── changing ──


async def _windows(call: str, *args: object) -> VolumeState | None:
    """Run a windows_volume call; None when Core Audio failed (the caller falls back to the keys)."""
    from pilot.system import windows_volume

    try:
        result = await asyncio.to_thread(getattr(windows_volume, call), *args)
    except Exception:
        logger.warning("Windows' volume call %s failed; using the volume keys", call, exc_info=True)
        return None
    return VolumeState(result[0], result[1])


async def _keys(script: str) -> str:
    code, _out, err = await run_powershell(script)
    if code != 0:
        raise RuntimeError(f"The volume keys could not be pressed: {err.strip()[:200]}")
    return "I pressed the volume keys, but could not read the volume back."


async def _linux_set(level_arg: str) -> None:
    """``level_arg`` is "30%", "+10%" or "-10%"."""
    if shutil.which("pactl"):
        code, _out, err = await run_command(["pactl", "set-sink-volume", "@DEFAULT_SINK@", level_arg])
        if code == 0:
            return
        logger.debug("pactl failed (%s); trying amixer", err.strip())
    if shutil.which("amixer"):
        amixer = level_arg if level_arg[0] not in "+-" else level_arg[1:] + level_arg[0]  # "10%-": amixer's form
        code, _out, err = await run_command(["amixer", "set", "Master", amixer])
        if code == 0:
            return
        raise RuntimeError(f"Volume change failed: {err.strip()[:200]}")
    raise RuntimeError("Neither pactl nor amixer is installed, so the volume cannot be changed.")


async def volume_set(level: int) -> str:
    """Set the volume (0-100) and say what it is now."""
    level = max(0, min(100, int(level)))
    if CURRENT_PLATFORM == Platform.WINDOWS:
        state = await _windows("set_level", level)
        if state is None:
            raise RuntimeError("Windows' volume control did not answer, so the volume was not set.")
        return say_state(state)
    if CURRENT_PLATFORM == Platform.MACOS:
        from pilot.system.script_values import osascript_command

        code, _out, err = await run_command(osascript_command(_MAC_SET_LEVEL, [str(level)]))
        if code != 0:
            raise RuntimeError(f"Volume set failed: {err.strip()[:200]}")
    else:
        await _linux_set(f"{level}%")
    return say_state(await volume_state()) or f"I set the volume to {level}, but could not read it back."


async def volume_step(delta: int) -> str:
    """Turn the volume up (delta > 0) or down by ``delta`` points and say what it is now."""
    delta = max(-100, min(100, int(delta)))
    if CURRENT_PLATFORM == Platform.WINDOWS:
        state = await _windows("step", delta)
        if state is None:
            return await _keys(_WIN_KEY_UP if delta > 0 else _WIN_KEY_DOWN)
        return say_state(state)
    if CURRENT_PLATFORM == Platform.MACOS:
        before = await volume_state()
        if before.level is None:
            raise RuntimeError("Could not read the volume, so it was not changed.")
        return await volume_set(before.level + delta)
    await _linux_set(f"{'+' if delta > 0 else '-'}{abs(delta)}%")
    return say_state(await volume_state()) or "I changed the volume, but could not read it back."


async def volume_mute(mute: bool = True) -> str:
    """Mute or unmute and say what is true: "Muted.", "Already unmuted." Never toggles blindly."""
    word = "muted" if mute else "unmuted"
    if CURRENT_PLATFORM == Platform.WINDOWS:
        from pilot.system import windows_volume

        try:
            _level, muted, changed = await asyncio.to_thread(windows_volume.set_muted, mute)
        except Exception:
            logger.warning("Windows' mute call failed", exc_info=True)
            raise RuntimeError(f"Windows' volume control did not answer, so the sound was not {word}.") from None
        if muted != mute:
            raise RuntimeError(f"The sound is still {'muted' if muted else 'on'}.")
        return f"{word.capitalize()}." if changed else f"Already {word}."
    before = await volume_state()
    if before.muted is mute:
        return f"Already {word}."
    if CURRENT_PLATFORM == Platform.MACOS:
        from pilot.system.script_values import osascript_command

        code, _out, err = await run_command(osascript_command(_MAC_SET_MUTED, ["true" if mute else "false"]))
    elif shutil.which("pactl"):
        code, _out, err = await run_command(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if mute else "0"])
    elif shutil.which("amixer"):
        code, _out, err = await run_command(["amixer", "set", "Master", "mute" if mute else "unmute"])
    else:
        raise RuntimeError("Neither pactl nor amixer is installed, so the sound cannot be muted.")
    if code != 0:
        raise RuntimeError(f"Mute failed: {err.strip()[:200]}")
    after = await volume_state()
    if after.muted is None:
        return f"I {'muted' if mute else 'unmuted'} the sound, but could not read it back."
    if after.muted != mute:
        raise RuntimeError(f"The sound is still {'muted' if after.muted else 'on'}.")
    return f"{word.capitalize()}."
