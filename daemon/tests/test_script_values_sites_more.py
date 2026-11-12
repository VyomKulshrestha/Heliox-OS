"""The last places that pasted a value into code an interpreter runs now pass it as data (the third security fix).

Each site gets a value with each kind of quote (ASCII ' and ", U+2018 to U+201B), a backtick, $ and $(...), a newline,
;, &, % and |. It must arrive whole, and the code text must not contain it. Where Windows can run it, it runs.
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pilot.system import platform_detect
from pilot.system.platform_detect import Platform

HOSTILE = 'it\'s \u2018a\u2019 \u201ab\u201b "q" `n $env:PATH $(Get-Date) \n next; & echo %PATH% | x'


class Recorder:
    def __init__(self) -> None:
        self.powershell: list[tuple[str, dict]] = []
        self.commands: list[list[str]] = []

    async def run_powershell(self, script, **kwargs):
        self.powershell.append((script, dict(kwargs.get("values") or {})))
        return 0, "present", ""

    async def run_command(self, args, **_kwargs):
        self.commands.append(list(args))
        return 0, "", ""


def _windows(monkeypatch, module) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(module, "CURRENT_PLATFORM", Platform.WINDOWS, raising=False)
    for target in (module, platform_detect):
        if hasattr(target, "run_powershell"):
            monkeypatch.setattr(target, "run_powershell", recorder.run_powershell)
        if hasattr(target, "run_command"):
            monkeypatch.setattr(target, "run_command", recorder.run_command)
    return recorder


def _assert_data(recorder: Recorder, value: str) -> None:
    assert recorder.powershell
    for script, values in recorder.powershell:
        assert value not in script and "$(Get-Date)" not in script and "\u2018a" not in script
        assert value in values.values()


def test_clipboard_text_is_data(monkeypatch):
    from pilot.system import clipboard

    monkeypatch.setitem(sys.modules, "pyperclip", None)  # the PowerShell route, used when pyperclip is missing
    recorder = _windows(monkeypatch, clipboard)
    asyncio.run(clipboard.clipboard_write(HOSTILE))
    _assert_data(recorder, HOSTILE)


def test_process_name_is_data(monkeypatch):
    from pilot.system import processes

    monkeypatch.setitem(sys.modules, "psutil", None)
    recorder = _windows(monkeypatch, processes)
    assert asyncio.run(processes.process_exists(name=HOSTILE)) is True
    _assert_data(recorder, HOSTILE)


def test_mount_target_is_data(monkeypatch):
    from pilot.system import disks

    recorder = _windows(monkeypatch, disks)
    asyncio.run(disks.mount_exists(mount_point=HOSTILE))
    _assert_data(recorder, HOSTILE)


def test_start_menu_shortcut_path_is_data(monkeypatch):
    from pilot.system import applications

    recorder = _windows(monkeypatch, applications)

    async def answer(script, **kwargs):
        recorder.powershell.append((script, dict(kwargs.get("values") or {})))
        return 0, "C:\\Program Files\\App\\app.exe", ""

    monkeypatch.setattr(platform_detect, "run_powershell", answer)
    shortcut = Path("C:/ProgramData/Microsoft/Windows/Start Menu/Programs") / (
        "App " + HOSTILE.replace("\n", " ") + ".lnk"
    )
    asyncio.run(applications._resolve_shortcut_target(shortcut))
    _assert_data(recorder, str(shortcut))


def test_screenshot_path_is_data(monkeypatch, tmp_path):
    from pilot.system import screen

    recorder = _windows(monkeypatch, screen)
    out = str(tmp_path / ("shot " + HOSTILE.replace("\n", " ").replace("|", "").replace('"', "") + ".png"))
    asyncio.run(screen.screenshot(out))
    _assert_data(recorder, out)


# ── cmd: what was approved is what runs ──


def test_a_program_runs_directly_with_its_arguments_as_they_are():
    from pilot.system.command_line import windows_command_line

    program = sys.executable
    assert windows_command_line(program, ["-c", "print(1)", HOSTILE]) == [program, "-c", "print(1)", HOSTILE]


@pytest.mark.parametrize("char", list('&|<>^%!()"') + ["\n"])
def test_a_cmd_builtin_with_a_character_cmd_reads_as_code_is_refused(char):
    from pilot.system.command_line import UnsafeCommandLine, windows_command_line

    with pytest.raises(UnsafeCommandLine, match="Not run: 'dir' goes through cmd"):
        windows_command_line("dir", [f"C:\\notes{char}calc"])


def test_a_cmd_builtin_with_plain_arguments_still_runs_through_cmd():
    from pilot.system.command_line import windows_command_line

    assert windows_command_line("dir", ["/b", "C:\\notes"]) == ["cmd", "/c", "dir", "/b", "C:\\notes"]
    assert windows_command_line("TYPE", ["C:\\notes\\a.txt"]) == ["cmd", "/c", "TYPE", "C:\\notes\\a.txt"]
    # A whole line given as the command is split the way Windows splits a command line.
    assert windows_command_line('dir /b "C:\\My Notes"', []) == ["cmd", "/c", "dir", "/b", "C:\\My Notes"]
    from pilot.system.command_line import UnsafeCommandLine

    with pytest.raises(UnsafeCommandLine):
        windows_command_line("dir /b & calc", [])


def test_a_batch_file_is_checked_like_a_builtin(tmp_path):
    from pilot.system.command_line import UnsafeCommandLine, batch_file_command, windows_command_line

    batch = tmp_path / "tool.cmd"
    batch.write_text("@echo off\n", encoding="utf-8")
    with pytest.raises(UnsafeCommandLine):
        windows_command_line(str(batch), ["a&b"])
    with pytest.raises(UnsafeCommandLine, match="temporary file's path"):
        batch_file_command("C:\\Users\\A&B\\AppData\\Local\\Temp\\x.cmd")
    assert batch_file_command("C:\\Temp\\x.cmd") == ["cmd", "/c", "C:\\Temp\\x.cmd"]


@pytest.mark.skipif(sys.platform != "win32", reason="runs programs on Windows")
def test_on_windows_a_shell_command_runs_what_was_approved(monkeypatch):
    from pilot.actions import Action, ActionType, ShellCommandParams
    from pilot.agents.executor import Executor

    executor = Executor.__new__(Executor)
    script = "import sys; print(ascii(sys.argv[1:]))"
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="python",
        destructive=True,
        parameters=ShellCommandParams(command=sys.executable, args=["-c", script, HOSTILE.replace("\n", " ")]),
    )
    out = asyncio.run(executor._exec_shell_command(action))
    assert ast.literal_eval(out.strip()) == [HOSTILE.replace("\n", " ")]
    echo = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="echo",
        destructive=True,
        parameters=ShellCommandParams(command="echo", args=["hello", "&", "echo", "second"]),
    )
    with pytest.raises(ValueError, match="Not run: 'echo' goes through cmd"):
        asyncio.run(executor._exec_shell_command(echo))


def test_open_url_on_windows_goes_to_the_system_handler_not_cmd(monkeypatch):
    from pilot.actions import Action, ActionType, OpenUrlParams
    from pilot.agents.executor import Executor
    from pilot.system import command_line

    opened: list[str] = []
    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(command_line.os, "startfile", opened.append, raising=False)
    commands: list = []

    async def no_command(args, **_kwargs):
        commands.append(args)
        return 0, "", ""

    monkeypatch.setattr(platform_detect, "run_command", no_command)
    url = "https://example.com/?q=a&calc|x^y%PATH%"
    action = Action(action_type=ActionType.OPEN_URL, target=url, parameters=OpenUrlParams(url=url))
    asyncio.run(Executor.__new__(Executor)._exec_open_url(action))
    assert opened == [url] and commands == []


# ── sh on macOS and Linux: the sandbox's limits wrapper ──


def test_the_sandbox_passes_its_command_to_bash_as_arguments(monkeypatch):
    from pilot.system import sandbox_exec

    monkeypatch.setattr(sandbox_exec.sys, "platform", "linux")
    config = sandbox_exec.SandboxConfig(memory_mb=256, timeout=10)
    path = "/tmp/" + HOSTILE.replace("\n", " ") + "/script.sh"
    wrapped = sandbox_exec.RestrictedBackend._wrap_with_ulimit(["/usr/bin/python3", path], config)
    assert wrapped[:3] == ["bash", "-c", 'ulimit -v "$1" -t "$2" -f 102400; shift 2; exec "$@"']
    assert wrapped[3:] == ["heliox-sandbox", str(256 * 1024), "10", "/usr/bin/python3", path]


@pytest.mark.skipif(sys.platform == "win32", reason="runs bash")
def test_bash_runs_the_wrapped_command_with_its_arguments_whole(tmp_path):
    import subprocess

    from pilot.system import sandbox_exec

    config = sandbox_exec.SandboxConfig(memory_mb=2048, timeout=10)
    value = HOSTILE.replace("\n", " ")
    wrapped = sandbox_exec.RestrictedBackend._wrap_with_ulimit(["printf", "%s", value], config)
    done = subprocess.run(wrapped, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout == value


# ── the planner's slow-command rewrite ──


def test_a_slow_windows_script_stays_the_script_that_was_approved(monkeypatch):
    # It used to be rewritten into generated Python running each line with shell=True (and that rewrite imported a
    # CodeParams that does not exist, so the plan failed). Now it is the same shell_script with more time.
    from pilot.actions import Action, ActionType, ShellScriptParams
    from pilot.agents import planner as planner_module
    from pilot.agents.planner import Planner

    script = "systeminfo\ntasklist | findstr x\necho C:\\notes\\\necho it's \u2019 done"
    action = Action(
        action_type=ActionType.SHELL_SCRIPT, target="system", parameters=ShellScriptParams(script=script, timeout=60)
    )
    monkeypatch.setattr(planner_module.sys, "platform", "win32")
    (kept,) = Planner.__new__(Planner)._postprocess_actions([action])
    assert kept.action_type == ActionType.SHELL_SCRIPT and kept.parameters.script == script
    assert kept.parameters.timeout == planner_module.SLOW_SCRIPT_TIMEOUT


# ── Wi-Fi on Windows: the profile ──


def test_the_wlan_profile_escapes_the_name_and_the_password():
    from pilot.system.network import wlan_profile_xml

    ssid = 'Cafe <b>&amp;</b> "free"'
    password = "p&ss<word>'\""
    root = ET.fromstring(wlan_profile_xml(ssid, password))
    ns = {"w": "http://www.microsoft.com/networking/WLAN/profile/v1"}
    assert root.find("w:name", ns).text == ssid
    assert root.find("w:SSIDConfig/w:SSID/w:name", ns).text == ssid
    assert root.find("w:MSM/w:security/w:sharedKey/w:keyMaterial", ns).text == password


@pytest.mark.parametrize("fails_at", ["none", "add", "connect"])
def test_the_password_file_is_owner_only_and_gone_on_every_path(monkeypatch, tmp_path, fails_at):
    import tempfile

    from pilot.system import network

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen: dict = {}

    async def run(args, **_kwargs):
        if args[2] == "add":
            path = args[-1].removeprefix("filename=")
            seen["path"] = path
            seen["mode"] = os.stat(path).st_mode
            seen["folder_mode"] = os.stat(os.path.dirname(path)).st_mode
            if fails_at == "add":
                raise RuntimeError("netsh failed")
        if args[2] == "connect" and fails_at == "connect":
            raise RuntimeError("netsh failed")
        return 0, "", ""

    monkeypatch.setattr(network, "run_command", run)
    if fails_at == "none":
        asyncio.run(network._connect_with_profile("Cafe", "secret"))
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(network._connect_with_profile("Cafe", "secret"))
    assert not os.path.exists(seen["path"]) and not os.path.exists(os.path.dirname(seen["path"]))
    if os.name != "nt":
        assert seen["mode"] & 0o077 == 0 and seen["folder_mode"] & 0o077 == 0
    assert list(tmp_path.iterdir()) == []
