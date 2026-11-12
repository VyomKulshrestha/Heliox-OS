"""The first-run screen's sizes are measured: setup logs from CI become the UI's data file."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("first_run_sizes", HERE / "platform_real" / "first_run_sizes.py")
sizes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sizes)


def test_cumulative_sizes_become_per_step_download_and_disk():
    log = "\n".join(
        [
            "@@step core running x",
            "Collecting numpy",
            "@@size core 112640",
            "@@disk core 542720",
            "@@step platform done Installed",
            "@@size platform 113664",
            "@@disk platform 545792",
        ]
    )
    assert sizes.step_sizes(log) == {
        "core": {"mb": 110.0, "disk_mb": 530.0},  # the core's disk is the whole folder the person ends up with
        "platform": {"mb": 1.0, "disk_mb": 3.0},  # a group's disk is what it adds
    }
    assert sizes.step_sizes("no sizes here") == {}


def test_measured_sizes_replace_estimates_for_that_system_only():
    data = {"sizes": {"core": {"linux": {"mb": 110}, "macos": {"mb": 110}}, "voice": {"all": {"mb": 900}}}}
    sizes.update(data, "linux", "run 43200000", {"core": {"mb": 96.4, "disk_mb": 529.2}, "platform": {"mb": 0.2}})
    assert data["sizes"]["core"]["linux"] == {"mb": 96, "disk_mb": 529, "measured": "run 43200000"}
    assert data["sizes"]["core"]["macos"] == {"mb": 110}
    assert data["sizes"]["platform"]["linux"] == {"mb": 1, "measured": "run 43200000"}
    assert data["sizes"]["voice"] == {"all": {"mb": 900}}


def test_the_ui_data_file_has_every_size_the_screen_shows():
    data = json.loads((HERE.parents[1] / "tauri-app" / "ui" / "src" / "lib" / "firstRunSizes.json").read_text("utf-8"))
    for key in ("core", "private_python", "platform", "voice", "screen", "browser", "local", "more"):
        assert data["sizes"][key], key
    for system in ("windows", "macos", "linux"):
        assert data["sizes"]["core"][system]["mb"] > 0
