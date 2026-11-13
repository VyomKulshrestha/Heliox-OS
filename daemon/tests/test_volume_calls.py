"""The system volume through each system's own call, read back, with fakes for every system: Core Audio on Windows,
osascript on macOS, pactl and amixer on Linux. No device is touched."""

from __future__ import annotations

import pytest

from pilot.system import volume, windows_volume
from pilot.system.platform_detect import Platform


class Pulse:
    """A fake pactl and amixer: one sink with a level and a mute flag; records every command."""

    def __init__(self, level: int = 50, muted: bool = False, *, pactl: bool = True, amixer: bool = True) -> None:
        self.level, self.muted, self.calls = level, muted, []
        self.has = {"pactl": pactl, "amixer": amixer}

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if self.has.get(name) else None

    async def run(self, args: list[str], **_kwargs) -> tuple[int, str, str]:
        self.calls.append(args)
        if args[:2] == ["pactl", "get-sink-volume"]:
            return 0, f"Volume: front-left: 1 /  {self.level}% / -1.0 dB,   front-right: 1 /  {self.level}% / -1 dB", ""
        if args[:2] == ["pactl", "get-sink-mute"]:
            return 0, f"Mute: {'yes' if self.muted else 'no'}", ""
        if args[:2] == ["pactl", "set-sink-volume"]:
            value = args[3]
            self.level = self._apply(value[:-1])
            return 0, "", ""
        if args[:2] == ["pactl", "set-sink-mute"]:
            self.muted = args[3] == "1"
            return 0, "", ""
        if args[:3] == ["amixer", "get", "Master"]:
            return 0, f"  Front Left: Playback 1 [{self.level}%] [{'off' if self.muted else 'on'}]", ""
        if args[:2] == ["amixer", "set"]:
            value = args[3]
            if value in {"mute", "unmute"}:
                self.muted = value == "mute"
            else:
                self.level = self._apply(value[-1] + value[:-2] if value[-1] in "+-" else value[:-1])
            return 0, "", ""
        return 1, "", "unknown command"

    def _apply(self, value: str) -> int:
        if value[0] in "+-":
            return max(0, min(100, self.level + int(value)))
        return int(value)


@pytest.fixture
def linux(monkeypatch):
    pulse = Pulse()
    monkeypatch.setattr(volume, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(volume.shutil, "which", pulse.which)
    monkeypatch.setattr(volume, "run_command", pulse.run)
    return pulse


@pytest.mark.asyncio
async def test_linux_sets_steps_and_reads_back_through_pactl(linux):
    assert await volume.volume_set(30) == "Volume is at 30."
    assert await volume.volume_step(-10) == "Volume is at 20."
    assert ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"] in linux.calls
    assert await volume.volume_step(10) == "Volume is at 30."
    assert await volume.volume_get() == "Volume is at 30."


@pytest.mark.asyncio
async def test_linux_mute_says_what_is_true_and_never_toggles(linux):
    assert await volume.volume_mute(False) == "Already unmuted."
    assert not any(call[:2] == ["pactl", "set-sink-mute"] for call in linux.calls)
    assert await volume.volume_mute(True) == "Muted."
    assert await volume.volume_mute(True) == "Already muted."
    assert await volume.volume_get() == "Volume is at 50, and muted."
    assert await volume.volume_mute(False) == "Unmuted."


@pytest.mark.asyncio
async def test_linux_falls_back_to_amixer_and_says_when_neither_is_there(linux):
    linux.has["pactl"] = False
    assert await volume.volume_step(-10) == "Volume is at 40."
    assert ["amixer", "set", "Master", "10%-"] in linux.calls
    linux.has["amixer"] = False
    with pytest.raises(RuntimeError, match="Neither pactl nor amixer"):
        await volume.volume_set(30)
    assert not volume.volume_call_available()


@pytest.mark.asyncio
async def test_linux_a_mute_that_did_not_take_is_a_failure(linux):
    async def ignores_mute(args, **kwargs):
        if args[:2] == ["pactl", "set-sink-mute"]:
            return 0, "", ""  # says yes, changes nothing
        return await Pulse.run(linux, args)

    linux.run = ignores_mute
    volume.run_command = ignores_mute
    with pytest.raises(RuntimeError, match="still on"):
        await volume.volume_mute(True)


class Mac:
    def __init__(self) -> None:
        self.level, self.muted, self.calls = 40, False, []

    async def run(self, args: list[str], **_kwargs) -> tuple[int, str, str]:
        self.calls.append(args)
        script = args[2]
        if script == volume._MAC_READ:
            return (
                0,
                f"output volume:{self.level}, input volume:50, alert volume:100, output muted:{str(self.muted).lower()}",
                "",
            )
        values = [value[1:] for value in args[3:]]  # osascript_command prefixes each value with "x"
        if volume._MAC_SET_LEVEL in script:
            self.level = int(values[0])
        elif volume._MAC_SET_MUTED in script:
            self.muted = values[0] == "true"
        return 0, "", ""


@pytest.mark.asyncio
async def test_macos_reads_sets_and_steps_with_the_level_as_data(monkeypatch):
    mac = Mac()
    monkeypatch.setattr(volume, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(volume, "run_command", mac.run)
    assert await volume.volume_step(-10) == "Volume is at 30."
    assert await volume.volume_set(55) == "Volume is at 55."
    set_call = [call for call in mac.calls if volume._MAC_SET_LEVEL in call[2]][-1]
    assert "55" not in set_call[2] and set_call[3:] == ["x55"]  # the level is an argument, never script text
    assert await volume.volume_mute(False) == "Already unmuted."
    assert await volume.volume_mute(True) == "Muted."
    assert volume.parse_mac_settings("output volume:missing value, output muted:missing value") == volume.VolumeState(
        None, None
    )


class CoreAudio:
    def __init__(self, level: int = 50, muted: bool = True, *, broken: bool = False) -> None:
        self.level, self.muted, self.broken, self.set_calls = level, muted, broken, 0

    def _check(self):
        if self.broken:
            raise OSError("E_NOTFOUND")

    def read(self):
        self._check()
        return self.level, self.muted

    def set_level(self, level):
        self._check()
        self.level = level
        return self.level, self.muted

    def step(self, delta):
        self._check()
        self.level = max(0, min(100, self.level + delta))
        return self.level, self.muted

    def set_muted(self, mute):
        self._check()
        changed = self.muted != mute
        if changed:
            self.set_calls += 1
            self.muted = mute
        return self.level, self.muted, changed


@pytest.fixture
def windows(monkeypatch):
    audio = CoreAudio()
    monkeypatch.setattr(volume, "CURRENT_PLATFORM", Platform.WINDOWS)
    for name in ("read", "set_level", "step", "set_muted"):
        monkeypatch.setattr(windows_volume, name, getattr(audio, name))
    pressed = []

    async def powershell(script, **_kwargs):
        pressed.append(script)
        return 0, "", ""

    monkeypatch.setattr(volume, "run_powershell", powershell)
    audio.pressed = pressed
    return audio


@pytest.mark.asyncio
async def test_windows_unmute_on_a_machine_that_is_not_muted_changes_nothing(windows):
    windows.muted = False
    assert await volume.volume_mute(False) == "Already unmuted."
    assert windows.set_calls == 0 and windows.pressed == []  # before: the mute key, which muted it


@pytest.mark.asyncio
async def test_windows_sets_steps_and_mutes_through_core_audio_with_no_key_press(windows):
    assert await volume.volume_set(30) == "Volume is at 30, and muted."
    assert await volume.volume_mute(False) == "Unmuted."
    assert await volume.volume_step(-10) == "Volume is at 20."
    assert await volume.volume_get() == "Volume is at 20."
    assert windows.pressed == [] and volume.volume_call_available()


@pytest.mark.asyncio
async def test_windows_without_core_audio_steps_with_the_keys_and_says_it_could_not_read_back(windows):
    windows.broken = True
    assert await volume.volume_step(-10) == "I pressed the volume keys, but could not read the volume back."
    assert windows.pressed == [volume._WIN_KEY_DOWN]  # a constant script: no value in it
    with pytest.raises(RuntimeError, match="was not set"):
        await volume.volume_set(30)  # a level cannot be set with keys: said, not claimed
    with pytest.raises(RuntimeError, match="was not muted"):
        await volume.volume_mute(True)
    assert not volume.volume_call_available()
