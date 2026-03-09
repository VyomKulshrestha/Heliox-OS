"""Shared test fixtures."""

import pytest

from pilot.config import PilotConfig


@pytest.fixture
def default_config():
    """A PilotConfig with all defaults."""
    return PilotConfig()


@pytest.fixture
def root_enabled_config():
    """A PilotConfig with root access enabled."""
    cfg = PilotConfig()
    cfg.security.root_enabled = True
    return cfg
