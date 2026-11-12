"""Screen text read by OCR: the usual damage put right before the error checks, and headlines left alone."""

from __future__ import annotations

from datetime import date

import pytest

from pilot.live.detectors import TextContext, detect, repair_ocr


def found(text, app="WindowsTerminal.exe", title="bash"):
    ctx = TextContext(today=date(2026, 11, 9), app=app, title=title, source="screen_ocr")
    return [c.title for c in detect(text, ctx, kinds=("error_on_screen",))]


def test_repairs():
    assert repair_ocr("No modu1e named 'x'") == "No module named 'x'"
    assert repair_ocr("Err0r: listen EADDRIN USE: in use :::3OOO") == "Error: listen EADDRINUSE: in use :::3000"
    assert repair_ocr("ModuleNot FoundError: x") == "ModuleNotFoundError: x"
    assert repair_ocr("connect to 127.0.0.l:5432") == "connect to 127.0.0.1:5432"
    # Left alone: real digits in words, and capitals that are not an error name or a known errno.
    assert repair_ocr("utf8 B12 h264 IPv4") == "utf8 B12 h264 IPv4"
    assert repair_ocr("Visual Studio Code, HTTP GET") == "Visual Studio Code, HTTP GET"


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("ModuleNotFoundError: No modu1e named 'pydantic'", "pydantic isn't installed"),
        ("ModuleNot FoundError: No module named 'numpy'", "numpy isn't installed"),
        ("Error: listen EADDRINUSE: address already in use :::3OOO", "Port 3000 is already taken"),
        ("Err0r: listen EADDRIN USE: address already in use :::8080", "Port 8080 is already taken"),
        ("bash: pyth0n: command not found", "The shell can't find python"),
        # git's own line goes on after the match; the platform-readers job (12 November) read it cut off at the
        # window's edge, and the card did not come because the tail read as a sentence about the error
        ("fatal: not a git repository (or any of the parent directories): .git", "This folder isn't a git repository"),
        ("fatal: not a git repository (or any of the parent director", "This folder isn't a git repository"),
    ],
)
def test_damaged_errors_are_still_read(text, title):
    assert found(text) == [title]


@pytest.mark.parametrize(
    "text",
    [
        "Fixing ModuleNotFoundError: No module named 'requests' in five minutes",  # a headline about an error
        "Fixing fatal: not a git repository errors in five minutes",
        "try:\n    import httpx\nexcept ModuleNotFoundError:\n    httpx = None",
        "Build succeeded.\n    0 Warning(s)\n    0 Error(s)",
        "=== 12 passed, 0 errors in 3.2s ===",
        "Errors and Exceptions\nError handling in Python uses try and except.",
    ],
)
def test_text_about_errors_is_not_an_error(text):
    assert found(text, app="chrome.exe", title="docs") == []
