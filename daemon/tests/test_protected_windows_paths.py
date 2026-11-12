"""A protected folder is found however a Windows path names it, on every Python Heliox supports, and a path
written the Windows way protects nothing on macOS and Linux, where it is a relative name.

The rules of each system are checked on any machine through path_rules' rule sets (ntpath and posixpath on the
text alone); the tests marked windows_only also check the real resolution on Windows.

Python 3.13 on Windows stopped counting a rooted path without a drive ("/Users/sam", "\\Users\\sam") as absolute,
while Windows still opens it on the current drive. The risk score asked os.path.isabs and so, on 3.13, never looked
at such a path. Both the risk score and the validator now decide by the text (path_rules.names_a_place) and resolve
as Windows opens the path (path_rules.resolved).
"""

from __future__ import annotations

import ntpath
import os
import sys

import pytest

from pilot.actions import Action, ActionType, FileParams
from pilot.config import PilotConfig
from pilot.security import path_rules
from pilot.security.path_rules import LINUX, MACOS, WINDOWS, names_a_place
from pilot.security.risk_safety import touches_protected_path
from pilot.security.validator import ActionValidator

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows path resolution")


@pytest.mark.parametrize(
    ("path", "rules", "expected"),
    [
        ("/etc/passwd", LINUX, True),
        ("/etc/passwd", MACOS, True),
        ("/etc/passwd", WINDOWS, True),
        ("~/notes", LINUX, True),
        ("notes.txt", LINUX, False),
        ("git", WINDOWS, False),
        ("C:\\Users\\sam", WINDOWS, True),
        ("C:/Users/sam", WINDOWS, True),
        ("C:Users\\sam", WINDOWS, True),  # drive-relative
        ("\\Users\\sam", WINDOWS, True),  # rooted, no drive
        ("/Users/sam", WINDOWS, True),
        ("\\\\server\\share\\private", WINDOWS, True),  # UNC
        # Written the Windows way, on macOS and Linux: a relative name (a backslash is a plain character there).
        ("C:/Users/sam", LINUX, False),
        ("C:\\Users\\sam", MACOS, False),
        ("\\Users\\sam", LINUX, False),
        ("\\\\server\\share", MACOS, False),
        ("", WINDOWS, False),
    ],
)
def test_what_names_a_place(path, rules, expected):
    assert names_a_place(path, rules) is expected


@pytest.mark.parametrize(
    ("path", "rules", "cwd", "expected"),
    [
        ("\\Users\\sam\\x", WINDOWS, "D:\\work", "D:\\Users\\sam\\x"),  # rooted: the current drive
        ("/Users/sam/x", WINDOWS, "D:\\work", "D:\\Users\\sam\\x"),
        ("D:Users\\sam", WINDOWS, "D:\\work", "D:\\work\\Users\\sam"),  # drive-relative: that drive's folder
        ("C:/Users/sam/x", WINDOWS, "D:\\work", "C:\\Users\\sam\\x"),
        ("\\\\files\\team\\x", WINDOWS, "D:\\work", "\\\\files\\team\\x"),
        ("/etc/../etc/hosts", LINUX, "/home/sam", "/etc/hosts"),
        ("C:/Users/sam", LINUX, "/home/sam", "/home/sam/C:/Users/sam"),  # a relative name on Linux
        ("~/notes", MACOS, "/", "/Users/sam/notes"),
    ],
)
def test_how_each_system_resolves_a_path(path, rules, cwd, expected):
    assert path_rules.resolved(path, rules, cwd=cwd, home="/Users/sam") == expected


@pytest.mark.parametrize(
    "written",
    [
        "\\Users\\sam\\private\\plan.txt",
        "/Users/sam/private/plan.txt",
        "D:/Users/sam/private/plan.txt",
        "D:Users\\sam\\private\\plan.txt",  # with D:'s current folder at its root
    ],
)
def test_windows_rules_see_a_protected_folder_however_the_path_names_it(written):
    config = _config("D:\\Users\\sam\\private")
    assert touches_protected_path(_file_action(written), config, WINDOWS, cwd="D:\\") == "D:\\Users\\sam\\private"
    assert touches_protected_path(_file_action("D:/Users/sam/private-notes.txt"), config, WINDOWS, cwd="D:\\") is None
    # The same text on macOS and Linux: the Windows-shaped folder is a relative name there and protects nothing.
    for rules in (LINUX, MACOS):
        assert touches_protected_path(_file_action(written), config, rules, cwd="/home/sam") is None


def test_windows_rules_see_a_unc_path_inside_a_protected_share():
    config = _config("\\\\files\\team\\private")
    for path in ("\\\\files\\team\\private\\plan.txt", "//files/team/private/plan.txt"):
        assert touches_protected_path(_file_action(path), config, WINDOWS, cwd="C:\\") is not None, path
    assert touches_protected_path(_file_action("\\\\files\\team\\public\\a.txt"), config, WINDOWS, cwd="C:\\") is None


def _drive() -> str:
    return os.path.splitdrive(os.getcwd())[0] or "C:"


def _file_action(path: str) -> Action:
    return Action(action_type=ActionType.FILE_DELETE, target=path, parameters=FileParams(path=path))


def _config(folder: str) -> PilotConfig:
    config = PilotConfig()
    config.restrictions.protected_folders = [folder]
    return config


@windows_only
@pytest.mark.parametrize(
    "written",
    [
        "\\Users\\sam\\private\\plan.txt",  # rooted without a drive: the current drive
        "/Users/sam/private/plan.txt",  # the same with forward slashes
        "{drive}/Users/sam/private/plan.txt",  # a drive and forward slashes
        "{drive}Users\\sam\\private\\plan.txt",  # drive-relative, with the drive's current folder at its root
    ],
)
def test_a_windows_path_naming_a_protected_folder_another_way_is_seen(monkeypatch, written):
    drive = _drive()
    monkeypatch.chdir(drive + "\\")
    folder = drive + "\\Users\\sam\\private"
    path = written.format(drive=drive)
    assert touches_protected_path(_file_action(path), _config(folder)) == folder
    with pytest.raises(Exception, match="protected folder|must be absolute"):
        ActionValidator(_config(folder)).validate_action(_file_action(path), 0)
    beside = drive + "\\Users\\sam\\private-notes.txt"
    assert touches_protected_path(_file_action(beside), _config(folder)) is None


@windows_only
def test_a_unc_path_inside_a_protected_share_folder_is_seen(monkeypatch):
    # realpath would ask the network for a server that does not exist; the rule is what is tested here.
    monkeypatch.setattr(path_rules.os.path, "realpath", ntpath.abspath)
    folder = "\\\\files\\team\\private"
    for path in ("\\\\files\\team\\private\\plan.txt", "//files/team/private/plan.txt"):
        assert touches_protected_path(_file_action(path), _config(folder)) == folder, path
    assert touches_protected_path(_file_action("\\\\files\\team\\public\\a.txt"), _config(folder)) is None


@windows_only
def test_a_protected_folder_given_without_a_drive_protects_the_same_place(monkeypatch):
    drive = _drive()
    monkeypatch.chdir(drive + "\\")
    folder = "/Users/sam/private"
    path = drive + "\\Users\\sam\\private\\plan.txt"
    assert touches_protected_path(_file_action(path), _config(folder)) == folder
    with pytest.raises(Exception, match="protected folder"):
        ActionValidator(_config(folder)).validate_action(_file_action(path), 0)
