"""pilot.config's runtime folder against the table the desktop app's Rust test reads too (runtime_dir_cases.json).

The window finds the daemon's token in that folder. Until 13 November 2026 the app looked in other places, so on
macOS, and on Linux outside uid 1000's /run/user folder, the window could not sign in. Both sides now follow these
cases; when one changes without the other, one of the two tests fails.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import pytest

from pilot import config

CASES = json.loads((Path(__file__).with_name("runtime_dir_cases.json")).read_text(encoding="utf-8"))["cases"]
PLATFORMS = {"linux": "linux", "macos": "darwin", "windows": "win32"}


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case['system']}-{case['expected']}")
def test_the_daemon_puts_its_runtime_folder_where_the_table_says(case, monkeypatch):
    monkeypatch.setattr(config, "_TEST_ROOT", None)
    for name in ("XDG_RUNTIME_DIR", "LOCALAPPDATA", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in case["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", PLATFORMS[case["system"]])
    monkeypatch.setattr(os, "getuid", lambda: case["uid"], raising=False)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: Path(case["home"])))
    run_user = f"/run/user/{case['uid']}"
    monkeypatch.setattr(
        pathlib.Path,
        "is_dir",
        lambda self: case["run_user_writable"] if self.as_posix().endswith(run_user) else False,
    )
    monkeypatch.setattr(os, "access", lambda path, mode: case["run_user_writable"])

    found = config._default_runtime_dir()

    assert found.as_posix() == case["expected"]


def test_the_table_covers_every_system_and_rule():
    systems = {case["system"] for case in CASES}
    assert systems == {"linux", "macos", "windows"}
    assert any("XDG_RUNTIME_DIR" in case["env"] for case in CASES if case["system"] == "macos")
    assert any(not case["run_user_writable"] for case in CASES if case["system"] == "linux")
