"""Shared test fixtures."""

from typing import Any, Callable

import pytest

# Assuming PilotConfig is importable from the daemon config module.
# Adjust the import path if necessary based on your project's structure.
from pilot.config import PilotConfig


@pytest.fixture
def config_factory() -> Callable[..., PilotConfig]:
    """
    Factory fixture to create isolated PilotConfig instances.

    Returns a callable that accepts keyword arguments to override
    default configuration values, ensuring each test gets a fresh state.
    """

    def _factory(**kwargs: Any) -> PilotConfig:
        cfg = PilotConfig()
        cfg.security.root_enabled = kwargs.get("allow_root", False)
        return cfg

    return _factory


@pytest.fixture
def default_config(config_factory: Callable[..., PilotConfig]) -> PilotConfig:
    """
    Provides a default PilotConfig instance.
    Backward-compatible fixture for tests that don't need custom configurations.
    """
    return config_factory()


@pytest.fixture
def root_enabled_config(config_factory: Callable[..., PilotConfig]) -> PilotConfig:
    """
    Provides a PilotConfig instance with root access enabled.
    Backward-compatible fixture replacing duplicated setup logic.
    """
    return config_factory(allow_root=True)


@pytest.fixture(params=[False, True], ids=["root_disabled", "root_enabled"])
def parametrized_config(request: pytest.FixtureRequest, config_factory: Callable[..., PilotConfig]) -> PilotConfig:
    """
    Parametrized fixture yielding multiple PilotConfig instances.

    Automatically runs any dependent test multiple times (e.g., once
    with allow_root=False and once with allow_root=True) using descriptive IDs.
    """
    return config_factory(allow_root=request.param)
