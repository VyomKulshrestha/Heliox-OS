"""Tests for the repository-wide release metadata gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC and SPEC.loader
check_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_release)


def test_current_repository_release_versions_are_synchronized():
    assert check_release.validate_release(ROOT, "v0.14.0") == "0.14.0"


def test_release_gate_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr(
        check_release,
        "collect_versions",
        lambda _root: {"daemon": "0.10.0", "desktop": "0.9.0"},
    )

    with pytest.raises(ValueError, match="inconsistent"):
        check_release.validate_release(ROOT)


def test_release_gate_rejects_wrong_tag(monkeypatch):
    monkeypatch.setattr(
        check_release,
        "collect_versions",
        lambda _root: {"daemon": "0.10.0", "desktop": "0.10.0"},
    )

    with pytest.raises(ValueError, match="does not match"):
        check_release.validate_release(ROOT, "v0.10.1")


def test_release_workflow_has_no_certificate_or_notarization_gate():
    release_text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    forbidden = (
        "WINDOWS_CERTIFICATE",
        "APPLE_CERTIFICATE",
        "APPLE_SIGNING_IDENTITY",
        "certificateThumbprint",
        "Authenticode",
        "notariz",
    )

    for token in forbidden:
        assert token not in release_text


def test_end_user_all_extra_contains_both_local_voice_engines():
    project = check_release._toml(ROOT / "daemon" / "pyproject.toml")
    extras = project["project"]["optional-dependencies"]
    voice = extras["voice"]
    all_extra = extras["all"]

    assert any(requirement.startswith("kokoro") for requirement in voice)
    assert any(requirement.startswith("pocket-tts") for requirement in voice)
    assert any(requirement.startswith("sounddevice") for requirement in voice)
    assert any("pilot-daemon[" in requirement and "voice" in requirement for requirement in all_extra)
