"""A protected package is matched by the names its system's package manager accepts for it."""

from __future__ import annotations

import pytest

from pilot.security.package_names import is_protected


@pytest.mark.parametrize(
    ("requested", "protected", "platform", "expected"),
    [
        # winget takes the id or the name for the same package.
        ("firefox", ["Mozilla.Firefox"], "win32", "Mozilla.Firefox"),
        ("MOZILLA.FIREFOX", ["Mozilla.Firefox"], "win32", "Mozilla.Firefox"),
        ("Mozilla.Firefox", ["firefox"], "win32", "firefox"),
        ("Git.Git", ["Mozilla.Firefox"], "win32", None),
        # Elsewhere a dot belongs to the name.
        ("12", ["python3.12"], "linux", None),
        ("firefox", ["Mozilla.Firefox"], "linux", None),
        ("Firefox", ["firefox"], "darwin", "firefox"),
        ("gnome-shell", ["gnome-shell"], "linux", "gnome-shell"),
        ("", ["firefox"], "linux", None),
        ("firefox", ["", "  "], "win32", None),
    ],
)
def test_protected_names(requested, protected, platform, expected):
    assert is_protected(requested, protected, platform) == expected


def test_the_validator_refuses_removing_a_protected_winget_package_by_its_name(monkeypatch):
    import sys

    from pilot.actions import Action, ActionType, PackageParams
    from pilot.config import PilotConfig
    from pilot.security.validator import ActionValidator

    monkeypatch.setattr(sys, "platform", "win32")
    config = PilotConfig()
    config.restrictions.protected_packages = ["Mozilla.Firefox"]
    action = Action(action_type=ActionType.PACKAGE_REMOVE, target="firefox", parameters=PackageParams(name="firefox"))
    with pytest.raises(Exception, match="is protected and cannot be removed"):
        ActionValidator(config).validate_action(action, 0)
