"""Shared test fixtures."""

import contextlib
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
_session = folder_guard.install()

# The person's graphics card is not the suite's: no test may need it or disturb it. On the owner's laptop a whole
# suite run fell inside a batch of NVIDIA driver faults (13 November). Before any import, every card is hidden from
# this process and from every process a test starts (Kokoro, Whisper through CTranslate2, PyTorch and sherpa-onnx all
# honour it, and the product reads it as "every card hidden"), unless HELIOX_TEST_GPU=1 asks for the card on purpose.
if os.environ.get("HELIOX_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # not "": Windows drops an empty variable from a child process
# The Hugging Face cache is the session folder's, so no test loads a real model (Whisper, Kokoro, an embedding model)
# from the person's cache by accident, onto any device. The opt-in model tests (HELIOX_REAL_VOICE_MODELS=1) keep it.
if os.environ.get("HELIOX_REAL_VOICE_MODELS") != "1":
    for _name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ.pop(_name, None)
    os.environ["HF_HOME"] = str(_session / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"  # and nothing is fetched into it

from pilot.config import PilotConfig  # noqa: E402
from pilot.security import credential_store  # noqa: E402
from pilot.system import system_guard  # noqa: E402  (takes the real SMTP classes before any test replaces them)

if os.environ.get("HELIOX_TEST_GPU") != "1":
    # nvidia-smi is the driver's own program and reads the card whatever CUDA_VISIBLE_DEVICES says. The GPU arbiter
    # every test shares reads a machine without an NVIDIA driver, as on CI; a test that decides over made-up cards
    # gives its arbiter its own reading and environment.
    import functools

    from pilot.system import gpu_arbiter as _gpu_arbiter

    _gpu_arbiter.arbiter().read = functools.partial(_gpu_arbiter.read_machine, nvidia=lambda: None)

# Keep ordinary CI lightweight; research checks can explicitly use real Torch.
if os.environ.get("HELIOX_TEST_REAL_TORCH") != "1":
    sys.modules["torch"] = None


_folder_guard_lines: list[str] = []
# Tests whose code tried to change the real system and was refused (pilot.system.system_guard), by node id.
_system_refused: dict[str, list[tuple[str, str]]] = {}
# CI-only tests marked real_system(...) that made the changes they named, by node id.
_system_opted_in: dict[str, list[tuple[str, str]]] = {}
_system_snapshot_fakes: list[str] = []
# Tests whose code reached this computer's credential store and was refused (pilot.security.credential_store).
_credential_refused: dict[str, list[str]] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_system(*changes): a CI-only test may make exactly these changes to the real system "
        "(names as pilot.system.system_guard reports them, such as Set-ItemProperty)",
    )


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A test that reached Heliox's real folders fails at its teardown, even when the code it ran swallowed the
    refusal (folder_guard.failure_for)."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "teardown" and report.outcome == "passed":
        why = folder_guard.failure_for(item.nodeid)
        if why is None and item.nodeid in _system_refused:
            attempts = "; ".join(f"{kind}: {what}" for kind, what in _system_refused[item.nodeid])
            why = f"SYSTEM-GUARD: this test tried to change the real system and was refused ({attempts})"
        if why is None and item.nodeid in _credential_refused:
            reached = "; ".join(_credential_refused[item.nodeid][:5])
            why = f"CREDENTIAL-GUARD: this test reached this computer's credential store and was refused ({reached})"
        if why is not None:
            report.outcome = "failed"
            report.longrepr = why


def pytest_terminal_summary(terminalreporter):
    for line in _folder_guard_lines:
        terminalreporter.write_line(line)
    terminalreporter.write_line(
        f"SYSTEM-GUARD {len(_system_refused)} tests reached the real system and were refused; "
        f"{len(_system_snapshot_fakes)} system snapshot requests went to the test fake"
    )
    for nodeid, attempts in sorted(_system_refused.items()):
        terminalreporter.write_line(f"  {nodeid}: " + "; ".join(f"{kind}: {what}" for kind, what in attempts))
    if _system_opted_in:
        terminalreporter.write_line(
            f"SYSTEM-GUARD {len(_system_opted_in)} CI-only tests opted in and made the real-system changes they named"
        )
        for nodeid, uses in sorted(_system_opted_in.items()):
            terminalreporter.write_line(f"  {nodeid}: " + "; ".join(f"{kind}: {what}" for kind, what in uses))
    outside = [what for where, what in credential_store.attempts() if where == "outside the test suite"]
    terminalreporter.write_line(
        f"CREDENTIAL-GUARD {len(_credential_refused)} tests reached this computer's credential store and were "
        f"refused (store in use: {credential_store.mode() or 'not isolated'})"
        + (f"; {len(outside)} more reaches outside any test" if outside else "")
    )
    for nodeid, reached in sorted(_credential_refused.items()):
        terminalreporter.write_line(f"  {nodeid}: " + "; ".join(reached[:5]))


@pytest.fixture(autouse=True)
def _no_real_credential_store(request):
    """No test reads or writes this computer's credential store (Credential Manager, the keychain, the Secret
    Service). keyring's fail backend stands in for it (tests/folder_guard.py), and a system backend or Windows'
    credential functions reached another way refuse (pilot.security.credential_store); this records which test asked,
    and the test fails at its teardown even when the code it ran swallowed the refusal."""
    before = len(credential_store.attempts())
    yield
    new = credential_store.attempts()[before:]
    if new:
        _credential_refused[request.node.nodeid] = [what for _where, what in new]


@pytest.fixture(autouse=True)
def _no_real_system_changes(monkeypatch, request):
    """No test changes the computer it runs on (pilot.system.system_guard refuses the commands; this records which
    test asked). Executors built in a test get a snapshot manager whose system snapshot is a fake that takes nothing
    and answers "no backend", as on a computer without one; file snapshots stay real (they live in the test's folder).
    A CI-only test that truly needs the real system is marked real_system("<change>", ...): exactly those changes
    pass while it runs, only where CI is set, and are counted apart from the refusals."""
    if not system_guard.guarded():
        yield
        return
    marker = request.node.get_closest_marker("real_system")
    allowed = system_guard.real_system_allowed(*marker.args) if marker else contextlib.nullcontext()
    from pilot.agents import executor as executor_module
    from pilot.system.snapshots import SnapshotManager

    class GuardedSnapshotManager(SnapshotManager):
        async def create_snapshot(self, action_id: str, description: str = "") -> str | None:
            _system_snapshot_fakes.append(request.node.nodeid)
            return None

    monkeypatch.setattr(executor_module, "SnapshotManager", GuardedSnapshotManager)
    before = len(system_guard.attempts())
    used_before = len(system_guard.allowed_uses())
    with allowed:
        yield
    new = system_guard.attempts()[before:]
    if new:
        _system_refused[request.node.nodeid] = new
    used = system_guard.allowed_uses()[used_before:]
    if used:
        _system_opted_in[request.node.nodeid] = used


@pytest.fixture(autouse=True)
def _no_real_output_line(monkeypatch):
    """Warming Heliox's voice opens the real output device in the daemon; never in tests."""
    monkeypatch.setattr("pilot.system.voice.warm_output_line", lambda *_args, **_kwargs: None)


@pytest.fixture(autouse=True)
def _gpu_paths_forgotten():
    """What one test found not installed on its fake card is not what the next test's machine has."""
    from pilot.system import gpu_paths

    gpu_paths.reset()
    yield
    gpu_paths.reset()


@pytest.fixture(autouse=True)
def kokoro_module_restored():
    """Every test leaves the voice's process-wide settings as it found them. The worker's loop
    (local_tts_worker._serve("kokoro_tts")) and kokoro_tts.configure set the Kokoro module's device for their process
    ("cpu" when no card is visible), and a daemon started in a test hands the voice its settings object
    (voice.use_daemon_config). In a test that process is the test run, so a later test_kokoro_tts.py or a later reply
    would see them. Autouse: a new test that runs the loop or configures the voice cannot forget it."""
    from pilot.system import kokoro_tts, local_tts_worker, voice

    saved = (kokoro_tts._device, kokoro_tts._offline_by_us, voice._daemon_config)
    # A change of voice is told from the voice configured last, and may leave a standby loading (_begin_switch): each
    # test starts as a daemon does, with neither.
    voices = (local_tts_worker._configured, local_tts_worker._handover, local_tts_worker._manager)
    local_tts_worker._configured, local_tts_worker._handover = None, None
    # The voice's store, set on the module's manager by configure_local_voice, and the fixed sentences being made
    # ahead: a later test that warms the voice (PilotServer._prewarm_voice) would otherwise start making them in its
    # own loop and end with that task pending ("Task was destroyed but it is pending").
    store, making = local_tts_worker._manager.head_cache, local_tts_worker._making_ahead
    local_tts_worker._making_ahead = None
    yield
    kokoro_tts._device, kokoro_tts._offline_by_us, voice._daemon_config = saved
    started = local_tts_worker._making_ahead
    if started is not None and started is not making and not started[1].done():
        with contextlib.suppress(RuntimeError):  # its loop has closed: it can no longer run, only be dropped
            started[1].cancel()
    local_tts_worker._configured, local_tts_worker._handover, local_tts_worker._manager = voices
    local_tts_worker._manager.head_cache, local_tts_worker._making_ahead = store, making


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
