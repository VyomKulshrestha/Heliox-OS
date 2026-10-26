"""Release-version and user-visible changelog regression tests."""

from __future__ import annotations

from pilot import __version__
from pilot.changelog import CHANGELOG, VERSION, check_for_updates


def test_changelog_matches_daemon_version():
    assert __version__ == VERSION
    assert VERSION in CHANGELOG


def test_upgrade_from_previous_release_only_returns_current_features(monkeypatch):
    monkeypatch.setattr("pilot.changelog.get_last_version", lambda: "0.15.0")

    updates = check_for_updates()

    assert updates
    assert all(feature["name"] for feature in updates)
    assert len(updates) == len(CHANGELOG[VERSION]["features"])


def test_changelog_does_not_claim_removed_unshipped_features():
    rendered = repr(CHANGELOG).lower()
    assert "biometric" not in rendered
    assert "cross-device handoff" not in rendered
