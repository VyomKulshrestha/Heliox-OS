"""Shared test fixtures."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

# Before any pilot module is imported: Heliox's folders go to a session folder, and the real ones are refused
# (tests/folder_guard.py). Loaded by path, so it is the same module whichever way pytest imports this file.
_spec = importlib.util.spec_from_file_location("folder_guard", Path(__file__).with_name("folder_guard.py"))
folder_guard = sys.modules.setdefault("folder_guard", importlib.util.module_from_spec(_spec))
if not hasattr(folder_guard, "install"):
    _spec.loader.exec_module(folder_guard)
folder_guard.install()

from pilot.config import PilotConfig  # noqa: E402

# Keep ordinary CI lightweight; research checks can explicitly use real Torch.
if os.environ.get("HELIOX_TEST_REAL_TORCH") != "1":
    sys.modules["torch"] = None


_folder_guard_lines: list[str] = []


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Watchdog to catch and debug test teardown hangs; and what reached Heliox's real folders (folder_guard)."""
    lines, failed = folder_guard.report()
    _folder_guard_lines.extend(lines)
    if failed and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    import os
    import threading
    import time
    import traceback

    def watchdog():
        time.sleep(30)
        print("\n\n!!! WATCHDOG: Test suite hung during teardown! Dumping threads: !!!\n", file=sys.stderr)
        for th in threading.enumerate():
            print(f"--- Thread: {th.name} ---", file=sys.stderr)
            frame = sys._current_frames().get(th.ident)
            if frame:
                traceback.print_stack(frame, file=sys.stderr)
        print("\n!!! Forcing exit !!!\n", file=sys.stderr)
        os._exit(1)

    t = threading.Thread(target=watchdog, daemon=True, name="TeardownWatchdog")
    t.start()


def pytest_terminal_summary(terminalreporter):
    for line in _folder_guard_lines:
        terminalreporter.write_line(line)


@pytest.fixture(autouse=True)
def _no_real_output_line(monkeypatch):
    """Warming Heliox's voice opens the real output device in the daemon; never in tests."""
    monkeypatch.setattr("pilot.system.voice.warm_output_line", lambda *_args, **_kwargs: None)


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
