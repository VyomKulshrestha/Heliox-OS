"""A protected folder is found however a Windows path names it, on every Python Heliox supports.

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
from pilot.security.path_rules import names_a_place
from pilot.security.risk_safety import touches_protected_path
from pilot.security.validator import ActionValidator

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows path resolution")


@pytest.mark.parametrize(
    ("path", "windows", "expected"),
    [
        ("/etc/passwd", False, True),
        ("~/notes", False, True),
        ("notes.txt", False, False),
        ("git", True, False),
        ("C:\\Users\\sam", True, True),
        ("C:/Users/sam", True, True),
        ("C:Users\\sam", True, True),  # drive-relative
        ("\\Users\\sam", True, True),  # rooted, no drive
        ("/Users/sam", True, True),
        ("\\\\server\\share\\private", True, True),  # UNC
        ("\\Users\\sam", False, False),  # a backslash is a plain character on macOS and Linux
        ("", True, False),
    ],
)
def test_what_names_a_place(path, windows, expected):
    assert names_a_place(path, windows=windows) is expected


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
