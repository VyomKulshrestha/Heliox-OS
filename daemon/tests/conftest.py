"""Shared test fixtures."""

import sys
from typing import Any, Callable

import pytest

from pilot.config import PilotConfig

# Optional safe mocks to prevent heavy dependencies from breaking CI tests
sys.modules["torch"] = None
sys.modules["tribev2"] = None


@pytest.fixture
def smtp_mock(monkeypatch):
    """Mocks smtplib.SMTP for testing without real network calls."""
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=30):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def ehlo(self):
            pass

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, from_addr, to_addrs, msg):
            sent.append({"from": from_addr, "to": to_addrs, "msg": msg})

        def quit(self):
            pass

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    return sent


@pytest.fixture
def imap_mock(monkeypatch):
    """Mocks imaplib.IMAP4_SSL for testing without real network calls."""

    class FakeIMAP:
        def __init__(self, host, ssl_context=None):
            pass

        def login(self, user, password):
            pass

        def select(self, mailbox):
            pass

        def search(self, charset, criterion):
            return "OK", [b"1 2"]

        def fetch(self, uid, message_parts):
            if uid == b"1":
                email_bytes = b"From: test@example.com\r\nTo: me@example.com\r\nSubject: Hello\r\nDate: Mon, 01 Jan 2024 10:00:00 +0000\r\n\r\nThis is a test email 1"
            else:
                email_bytes = b"From: another@example.com\r\nTo: me@example.com\r\nSubject: Greetings\r\nDate: Mon, 01 Jan 2024 11:00:00 +0000\r\n\r\nThis is a test email 2"

            return "OK", [(b"1 (RFC822 {100})", email_bytes)]

        def store(self, uid, command, flag):
            pass

        def logout(self):
            pass

    monkeypatch.setattr("imaplib.IMAP4_SSL", FakeIMAP)


# Assuming PilotConfig is importable from the daemon config module.
# Adjust the import path if necessary based on your project's structure.


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
