"""Each place that ran PowerShell or AppleScript with a value pasted into its code now passes the value as data.

For every site: a value holding each kind of quote (ASCII ' and ", U+2018 to U+201B), a backtick, $ and $(...), a
newline, ;, &, % and | reaches the command whole (as a $HelioxArgs value or an osascript argument), and the code
text does not contain it. On Windows the read-only sites also run in real PowerShell with a value that would
create a file if any of it ran as code, and the file must not appear.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from pilot.system import platform_detect
from pilot.system.platform_detect import Platform

HOSTILE = 'it\'s ‘a’ ‚b‛ "q" `n $env:PATH $(Get-Date) \n next; & echo %PATH% | x'


class Recorder:
    """Stands in for run_powershell and run_command, keeping what each was given."""

    def __init__(self) -> None:
        self.powershell: list[tuple[str, dict]] = []
        self.commands: list[list[str]] = []

    async def run_powershell(self, script, **kwargs):
        self.powershell.append((script, dict(kwargs.get("values") or {})))
        return 0, "", ""

    async def run_command(self, args, **_kwargs):
        self.commands.append(list(args))
        return 0, "0", ""


def _install(monkeypatch, module, recorder: Recorder, platform: Platform) -> None:
    monkeypatch.setattr(module, "CURRENT_PLATFORM", platform, raising=False)
    if hasattr(module, "run_powershell"):
        monkeypatch.setattr(module, "run_powershell", recorder.run_powershell)
    if hasattr(module, "run_command"):
        monkeypatch.setattr(module, "run_command", recorder.run_command)
    monkeypatch.setattr(platform_detect, "run_powershell", recorder.run_powershell)
    monkeypatch.setattr(platform_detect, "run_command", recorder.run_command)
    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform)


def _assert_data(recorder: Recorder, *expected_values: str) -> None:
    assert recorder.powershell, "PowerShell was not asked"
    for script, _values in recorder.powershell:
        assert HOSTILE not in script
        for part in ("it's", "‘a", "$(Get-Date)", "& echo", "%PATH%"):
            assert part not in script
    passed = [value for _script, values in recorder.powershell for value in values.values()]
    for value in expected_values:
        assert value in passed, value


# ── 1. download fallback on Windows ──


def test_download_fallback_passes_the_address_and_file_as_data(monkeypatch, tmp_path):
    from pilot.system import download

    recorder = Recorder()
    _install(monkeypatch, download, recorder, Platform.WINDOWS)

    class Failing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no httpx here")

    monkeypatch.setattr(download, "create_httpx_client", Failing, raising=False)
    url = "https://example.com/a?b=" + HOSTILE
    out = str(tmp_path / ("file " + HOSTILE.replace("\n", " ").replace("|", "").replace('"', "") + ".bin"))
    asyncio.run(download.download_file(url, out))
    _assert_data(recorder, out)
    assert any(url in values.values() or values.get("url", "").endswith(HOSTILE) for _s, values in recorder.powershell)


# ── 2 and 3. registry read and write ──


def _registry_action(action_type, **params):
    from pilot.actions import Action, RegistryParams

    return Action(action_type=action_type, target="registry", parameters=RegistryParams(**params))


def test_registry_read_and_write_pass_key_name_and_value_as_data(monkeypatch):
    from pilot.actions import ActionType
    from pilot.agents.executor import Executor

    recorder = Recorder()
    _install(monkeypatch, platform_detect, recorder, Platform.WINDOWS)
    executor = Executor.__new__(Executor)
    key = "HKCU\\Software\\" + HOSTILE
    asyncio.run(
        executor._exec_registry_read(_registry_action(ActionType.REGISTRY_READ, key_path=key, value_name=HOSTILE))
    )
    asyncio.run(
        executor._exec_registry_write(
            _registry_action(
                ActionType.REGISTRY_WRITE, key_path=key, value_name=HOSTILE, value_data=HOSTILE, value_type="REG_SZ"
            )
        )
    )
    _assert_data(recorder, key, HOSTILE)
    assert recorder.powershell[-1][1]["kind"] == "String"


def test_registry_write_refuses_a_value_type_it_does_not_know(monkeypatch):
    from pilot.actions import ActionType
    from pilot.agents.executor import Executor

    recorder = Recorder()
    _install(monkeypatch, platform_detect, recorder, Platform.WINDOWS)
    executor = Executor.__new__(Executor)
    action = _registry_action(
        ActionType.REGISTRY_WRITE,
        key_path="HKCU\\Software\\x",
        value_name="n",
        value_data="v",
        value_type="String; calc",
    )
    with pytest.raises(ValueError, match="does not know the value type"):
        asyncio.run(executor._exec_registry_write(action))
    assert recorder.powershell == []


# ── 4 to 8. windows: focus, close, minimize, maximize, exists, read and set text ──


@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.MACOS])
def test_window_actions_pass_titles_and_names_as_data(monkeypatch, platform):
    from pilot.system import window_mgr

    recorder = Recorder()
    _install(monkeypatch, window_mgr, recorder, platform)

    async def no_accessibility(*_args):
        return window_mgr._NOT_HANDLED

    monkeypatch.setattr(window_mgr, "_mac_accessibility", no_accessibility)
    for kwargs in ({"title": HOSTILE}, {"process_name": HOSTILE}):
        asyncio.run(window_mgr.window_focus(**kwargs))
        asyncio.run(window_mgr.window_close(**kwargs))
        asyncio.run(window_mgr.window_minimize(**kwargs))
        asyncio.run(window_mgr.window_maximize(**kwargs))
        asyncio.run(window_mgr.window_exists(**kwargs))
    if platform == Platform.WINDOWS:
        recorder.powershell.clear()
        monkeypatch.setattr(window_mgr, "run_powershell", _answering(recorder, "aGk="))
        asyncio.run(window_mgr.window_read_text(title=HOSTILE))
        monkeypatch.setattr(window_mgr, "run_powershell", _answering(recorder, "ok"))
        asyncio.run(window_mgr.window_set_text(HOSTILE * 2, title=HOSTILE))
        _assert_data(recorder, HOSTILE, HOSTILE * 2)
    else:
        assert len(recorder.commands) == 10
        for command in recorder.commands:
            assert command[0] == "osascript" and HOSTILE not in command[2] and command[3:] == ["x" + HOSTILE]


def _answering(recorder: Recorder, out: str):
    async def run(script, **kwargs):
        recorder.powershell.append((script, dict(kwargs.get("values") or {})))
        return 0, out, ""

    return run


# ── 9 and 10. speech and transcription on Windows ──


def test_speech_passes_text_voice_and_file_as_data(monkeypatch, tmp_path):
    from pilot.system import voice

    recorder = Recorder()
    _install(monkeypatch, voice, recorder, Platform.WINDOWS)
    out = str(tmp_path / "speech.wav")
    asyncio.run(voice._tts_windows(HOSTILE, HOSTILE, 190, 0.5, out))
    asyncio.run(voice._tts_windows(HOSTILE * 300, None, 170, 1.0, None))  # long: an owner-only file carries it
    _assert_data(recorder, HOSTILE, out, HOSTILE * 300)
    values = recorder.powershell[0][1]
    assert values["rate"] == "1" and values["volume"] == "50" and values["voice"] == HOSTILE


def test_transcription_passes_the_file_as_data(monkeypatch, tmp_path):
    from pilot.system import voice

    recorder = Recorder()
    _install(monkeypatch, voice, recorder, Platform.WINDOWS)
    audio = str(tmp_path / "talk.mp3")
    Path(audio).write_bytes(b"ID3")
    asyncio.run(voice._transcribe_windows(audio))
    _assert_data(recorder, audio)


# ── in real PowerShell: nothing in the value runs ──


def _canary(tmp_path: Path) -> tuple[str, Path]:
    marker = tmp_path / "ran.txt"
    # Would create the marker if any quote ended the string it sits in, or if $() or ; ran.
    payload = (
        f"x' ; Set-Content -LiteralPath '{marker}' -Value 1 ; '"
        f"’ ; Set-Content -LiteralPath '{marker}' -Value 1 ; ‘"
        f"$(Set-Content -LiteralPath '{marker}' -Value 1)"
        f"\n Set-Content -LiteralPath '{marker}' -Value 1"
    )
    return payload, marker


@pytest.mark.skipif(sys.platform != "win32", reason="runs PowerShell")
def test_in_real_powershell_a_hostile_title_or_key_runs_nothing(tmp_path):
    from pilot.actions import ActionType
    from pilot.agents.executor import Executor
    from pilot.system import window_mgr

    payload, marker = _canary(tmp_path)
    assert asyncio.run(window_mgr.window_exists(title=payload)) is False
    # Get-Process refuses such a name with an error, as it did before; what matters is that none of it ran.
    with pytest.raises(RuntimeError, match="Window observation failed"):
        asyncio.run(window_mgr.window_exists(process_name=payload))
    executor = Executor.__new__(Executor)
    with pytest.raises(RuntimeError):
        asyncio.run(
            executor._exec_registry_read(
                _registry_action(ActionType.REGISTRY_READ, key_path="HKCU\\Software\\" + payload, value_name=payload)
            )
        )
    assert not marker.exists()
