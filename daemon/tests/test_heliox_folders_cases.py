"""pilot.config's folders against the table the desktop app's Rust test reads too (heliox_folders_cases.json).

The app reads what the daemon writes: the token in the runtime folder, the log in the state folder, and removal
looks in all of them. Until 13 November 2026 the app built some of these folders by hand: it looked for the token in
other places (so on macOS, and on Linux outside uid 1000's /run/user folder, the window could not sign in), and it
read the log from ~/.local/state whatever $XDG_STATE_HOME said. Both sides now follow these cases (the app through
tauri-app/src-tauri/src/folders.rs); when one changes without the other, one of the two tests fails.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path

import pytest

from pilot import config

CASES = json.loads((Path(__file__).with_name("heliox_folders_cases.json")).read_text(encoding="utf-8"))["cases"]
PLATFORMS = {"linux": "linux", "macos": "darwin", "windows": "win32"}
XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")
# "{root}" makes a set value absolute on the machine running the test, as the absolute check needs.
ROOT = "C:" if os.name == "nt" else ""


def fill(text: str) -> str:
    return text.replace("{root}", ROOT)


def case_id(case) -> str:
    kinds = ",".join(f"{name[4:-5].lower()}={kind_of(case, name)}" for name in XDG if name in case["env"])
    return f"{case['system']}-{case['uid']}-{'/'.join(sorted(case['expected']))}-{kinds or 'no-xdg'}"


def kind_of(case, name: str) -> str:
    if name not in case["env"]:
        return "unset"
    value = case["env"][name]
    if not value.strip():
        return "blank"
    if value != value.strip():
        return "padded"
    return "set" if value.startswith("{root}/") else "relative"


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-{case_id(case)}" for i, case in enumerate(CASES)])
def test_the_daemon_puts_its_folders_where_the_table_says(case, monkeypatch):
    monkeypatch.setattr(config, "_TEST_ROOT", None)
    for name in (*XDG, "XDG_RUNTIME_DIR", "LOCALAPPDATA", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in case["env"].items():
        monkeypatch.setenv(name, fill(value))
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

    found = {
        "config": config._heliox_dir("XDG_CONFIG_HOME", ".config"),
        "data": config._heliox_dir("XDG_DATA_HOME", ".local/share"),
        "state": config._heliox_dir("XDG_STATE_HOME", ".local/state"),
        "runtime": config._default_runtime_dir(),
        "cache": config._cache_dir(),
        "setup": config._setup_dir(),
    }

    assert case["expected"], case
    for key, want in case["expected"].items():
        assert found[key].as_posix() == fill(want), key


def test_the_module_folders_are_the_ones_the_table_checks():
    """CONFIG_DIR and the rest are the functions the cases call, so the cases speak for the folders the daemon uses
    (under the test root both sides are the test root's)."""
    assert config._heliox_dir("XDG_CONFIG_HOME", ".config") == config.CONFIG_DIR
    assert config._heliox_dir("XDG_DATA_HOME", ".local/share") == config.DATA_DIR
    assert config._heliox_dir("XDG_STATE_HOME", ".local/state") == config.STATE_DIR
    assert config._cache_dir() == config.CACHE_DIR
    assert config._setup_dir() == config.SETUP_DIR
    assert config.LOG_FILE == config.STATE_DIR / "pilot.log"


def test_the_table_covers_every_system_and_rule():
    systems = {case["system"] for case in CASES}
    assert systems == {"linux", "macos", "windows"}
    assert any("XDG_RUNTIME_DIR" in case["env"] for case in CASES if case["system"] == "macos")
    assert any(not case["run_user_writable"] for case in CASES if case["system"] == "linux")
    for system in systems:
        mine = [case for case in CASES if case["system"] == system]
        for key in ("config", "data", "state", "runtime", "cache", "setup"):
            assert any(key in case["expected"] for case in mine), (system, key)
        for name in XDG:
            seen = {kind_of(case, name) for case in mine if case["expected"].keys() >= {"config", "data", "state"}}
            assert seen >= {"set", "padded", "blank", "relative", "unset"}, (system, name, seen)
