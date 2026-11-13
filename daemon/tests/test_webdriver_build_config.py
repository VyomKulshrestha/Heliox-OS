"""The app-launch job's Windows test build passes WebView2 a debugging port itself (webdriver_build_config.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "platform_real"))

import webdriver_build_config as build_config  # noqa: E402


def test_the_patch_repeats_the_window_and_adds_the_port_after_wrys_default():
    config = json.loads(build_config.CONFIG.read_text(encoding="utf-8"))
    patched = build_config.patch(config)["app"]["windows"]
    assert len(patched) == len(config["app"]["windows"])
    first = dict(patched[0])
    args = first.pop("additionalBrowserArgs")
    assert first == config["app"]["windows"][0]
    assert args.startswith(build_config.WRY_DEFAULT) and args.endswith("--remote-debugging-port=0")


def test_the_shipped_config_has_no_debugging_port():
    text = build_config.CONFIG.read_text(encoding="utf-8")
    assert "remote-debugging" not in text and "additionalBrowserArgs" not in text
