"""Things that used to claim success when they failed, or failed without a word, on macOS and Linux."""

from __future__ import annotations

import asyncio

from pilot.security.path_rules import LINUX, MACOS, WINDOWS, path_within, rules_for
from pilot.system import linux_voice
from pilot.system.notify import (
    applescript_quoted,
    notification_command,
    notification_result,
    powershell_quoted,
)

# ── notifications ──


def test_notification_text_is_quoted_as_data():
    assert powershell_quoted("it's") == "'it''s'"
    assert applescript_quoted('say "hi" \\ bye') == '"say \\"hi\\" \\\\ bye"'
    mac = notification_command("macos", 'Build "done"', "all good")
    assert mac[:2] == ["osascript", "-e"] and 'with title "Build \\"done\\""' in mac[2]
    linux = notification_command("linux", "-rf", "body")
    assert linux[0] == "notify-send" and linux[linux.index("--") + 1] == "-rf"  # never taken as an option
    win = notification_command("windows", "it's", "ok")
    assert win[0] == "powershell" and "'it''s'" in win[-1]


def test_notification_result_reports_failure_on_macos_and_linux():
    assert notification_result("linux", 0, "", "Done", "x") == "Notification sent: Done"
    missing = notification_result("linux", -1, "Command not found: notify-send", "Done", "Backup finished")
    assert "Could not show" in missing and "libnotify" in missing and "Backup finished" in missing
    failed = notification_result("macos", 1, "execution error: Not authorised (-1743)", "Done", "x")
    assert "Could not show" in failed and "-1743" in failed and not failed.startswith("Notification sent")
    assert notification_result("windows", 1, "boom", "Done", "x") == "Notification: Done — x"


def test_notify_action_on_linux_without_notify_send(monkeypatch):
    from pilot.actions import Action, ActionType, NotifyParams
    from pilot.agents import executor as executor_module
    from pilot.system import platform_detect
    from pilot.system.platform_detect import Platform

    async def missing(args, **_kw):
        return (-1, "", f"Command not found: {args[0]}")

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "run_command", missing)
    action = Action(action_type=ActionType.NOTIFY, target="all", parameters=NotifyParams(summary="Hi", body="there"))
    result = asyncio.run(executor_module.Executor._exec_notify(object(), action))  # type: ignore[arg-type]
    assert result.startswith("Could not show a desktop notification")


# ── the Linux system voice ──


def test_linux_voice_prefers_espeak_ng_and_reports_no_voice_once(monkeypatch, caplog):
    assert linux_voice.espeak_program(lambda name: "/usr/bin/x" if name == "espeak-ng" else None) == "espeak-ng"
    assert linux_voice.espeak_program(lambda name: "/usr/bin/x" if name == "espeak" else None) == "espeak"
    assert linux_voice.espeak_program(lambda name: None) is None
    monkeypatch.setattr(linux_voice, "_warned", False)
    with caplog.at_level("WARNING", logger="pilot.system.linux_voice"):
        first = linux_voice.no_voice_message()
        second = linux_voice.no_voice_message()
    assert first == second and first.startswith("TTS unavailable") and "espeak-ng" in first
    assert len([r for r in caplog.records if r.name == "pilot.system.linux_voice"]) == 1


def test_linux_speech_without_espeak_carries_on_in_text(monkeypatch):
    from pilot.system import voice

    monkeypatch.setattr(linux_voice, "espeak_program", lambda: None)

    async def must_not_run(*_a, **_kw):
        raise AssertionError("no process should be started")

    monkeypatch.setattr(voice.asyncio, "create_subprocess_exec", must_not_run)
    result = asyncio.run(voice._tts_linux("hello", None, 170, None))
    assert result.startswith("TTS unavailable")


# ── protected folders ──


def test_path_within_follows_each_systems_rules():
    assert rules_for("win32") == WINDOWS and rules_for("darwin") == MACOS and rules_for("linux") == LINUX
    # Windows: case and slash direction do not matter; a sibling with a longer name is not inside.
    assert path_within(r"C:\Users\Sam\Documents\a.txt", r"c:\users\sam\documents", WINDOWS)
    assert path_within("C:/Users/Sam/Documents", "C:\\Users\\Sam\\Documents\\", WINDOWS)
    assert not path_within(r"C:\Database\x", r"C:\Data", WINDOWS)
    assert path_within(r"D:\x", "D:", WINDOWS) and path_within(r"D:\x", "D:\\", WINDOWS)
    # macOS: case-insensitive like APFS, separator boundary kept.
    assert path_within("/Users/sam/documents/tax.pdf", "/Users/sam/Documents", MACOS)
    assert not path_within("/Users/sam/Documents2/x", "/Users/sam/Documents", MACOS)
    # Linux: case-sensitive.
    assert not path_within("/home/sam/documents/x", "/home/sam/Documents", LINUX)
    assert path_within("/home/sam/Documents", "/home/sam/Documents/", LINUX)
    assert path_within("/etc/passwd", "/", LINUX)
    assert not path_within("", "/x", LINUX) and not path_within("/x", "", LINUX)


def test_heliox_private_dirs_cannot_be_reached_by_changing_case(monkeypatch, tmp_path):
    from pilot.security import path_rules, validator

    private = tmp_path / "Config"
    private.mkdir()
    monkeypatch.setattr(validator, "heliox_private_dirs", lambda: (private,))
    insensitive = WINDOWS if __import__("os").name == "nt" else MACOS
    monkeypatch.setattr(path_rules, "rules_for", lambda platform=None: insensitive)
    swapped = str(private).replace("Config", "CONFIG") + "/config.toml"
    assert validator.inside_heliox_private_dirs(str(private / "config.toml"))
    assert validator.inside_heliox_private_dirs(swapped)
    if __import__("os").name != "nt":  # case-sensitive rules only make sense with POSIX paths
        monkeypatch.setattr(path_rules, "rules_for", lambda platform=None: LINUX)
        assert not validator.inside_heliox_private_dirs(swapped)
