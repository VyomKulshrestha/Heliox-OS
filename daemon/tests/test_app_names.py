"""Apps as people name them, from a process name and what Windows says about the program."""

from __future__ import annotations

import pytest

from pilot.system import app_names
from pilot.system.app_names import display_name


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ("WindowsTerminal", "Windows Terminal"),
        ("WindowsTerminal.exe", "Windows Terminal"),
        ("Code.exe", "VS Code"),
        ("msedge", "Edge"),
        ("chrome.exe", "Chrome"),
        ("OUTLOOK.EXE", "Outlook"),
        ("WINWORD", "Word"),
        ("EXCEL.EXE", "Excel"),
        ("POWERPNT", "PowerPoint"),
        ("powershell", "PowerShell"),
        ("pwsh.exe", "PowerShell"),
        ("cmd.exe", "Command Prompt"),
        ("explorer.exe", "File Explorer"),
        ("Notepad", "Notepad"),
    ],
)
def test_names_people_use_for_common_process_names(process, expected):
    assert display_name(process) == expected
    assert display_name(process, "Something Else Entirely") == expected  # the table wins over a description


def test_an_unknown_process_keeps_its_name_unless_windows_has_a_real_one():
    assert display_name("FooBarTool.exe") == "FooBarTool"  # never split or re-cased by rule
    assert display_name("obsidian", "Obsidian") == "Obsidian"
    assert display_name("acmenotes", "Acme Notes") == "Acme Notes"
    # Descriptions that name plumbing, or ramble, are not names.
    assert display_name("acmesvc", "Acme Update Service") == "acmesvc"
    assert display_name("frame", "Application Frame Host") == "frame"
    assert display_name("tool", "A very long description of what this program does for you") == "tool"
    assert display_name("", "") == ""


def test_describe_says_nothing_off_windows(monkeypatch):
    monkeypatch.setattr(app_names.os, "name", "posix")
    app_names.describe.cache_clear()
    assert app_names.describe("anything") == ""
    assert app_names.app_label("WINWORD") == "Word"
    app_names.describe.cache_clear()
