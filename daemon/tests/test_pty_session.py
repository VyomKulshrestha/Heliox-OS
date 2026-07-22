"""Tests for persistent PTY session management."""

import sys
import warnings

import pytest

from pilot.system.pty_session import PtySession, PtySessionManager

# forkpty() warning is a pytest multi-threading artefact; the daemon is single-threaded
warnings.filterwarnings("ignore", message=".*forkpty.*", category=DeprecationWarning)

if sys.platform == "win32":
    pytest.skip("PTY sessions are Unix-only", allow_module_level=True)

try:
    import ptyprocess  # noqa: F401
except ImportError:
    pytest.skip("ptyprocess not installed", allow_module_level=True)


@pytest.fixture(autouse=True)
def isolate_sessions():
    """Reset the session registry between tests."""
    PtySessionManager.close_all()
    yield
    PtySessionManager.close_all()


@pytest.mark.asyncio
async def test_session_exec_basic():
    session = PtySession()
    output = await session.exec("echo hello")
    session.close()
    assert "hello" in output


@pytest.mark.asyncio
async def test_session_state_persistence():
    session = PtySession()
    await session.exec("export PILOT_TEST_VAR=persistent")
    output = await session.exec("echo $PILOT_TEST_VAR")
    session.close()
    assert "persistent" in output


@pytest.mark.asyncio
async def test_session_cwd_persistence():
    session = PtySession()
    await session.exec("cd /tmp")
    output = await session.exec("pwd")
    session.close()
    assert "/tmp" in output


@pytest.mark.asyncio
async def test_separate_sessions_are_isolated():
    session_a = PtySessionManager.get_session("a")
    session_b = PtySessionManager.get_session("b")

    await session_a.exec("export PILOT_ISOLATED=only_in_a")
    output_b = await session_b.exec("echo ${PILOT_ISOLATED:-not_set}")

    assert "only_in_a" not in output_b


@pytest.mark.asyncio
async def test_session_timeout():
    session = PtySession()
    output = await session.exec("sleep 60", timeout=1)
    session.close()
    assert "timed out" in output.lower() or "error" in output.lower()


@pytest.mark.asyncio
async def test_close_and_reopen():
    session_id = "reopen_test"
    session = PtySessionManager.get_session(session_id)
    await session.exec("echo before_close")
    PtySessionManager.close_session(session_id)

    new_session = PtySessionManager.get_session(session_id)
    output = await new_session.exec("echo after_reopen")
    assert "after_reopen" in output


@pytest.mark.asyncio
async def test_interrupt_returns_early_instead_of_waiting_full_timeout():
    """Regression test for real mid-flight cancellation: interrupt() must
    make an in-progress exec() return within a few seconds, not the full
    per-command timeout."""
    import asyncio
    import time

    session = PtySession()
    task = asyncio.ensure_future(session.exec("sleep 30", timeout=60))
    await asyncio.sleep(0.5)  # let the command actually start running

    start = time.monotonic()
    session.interrupt()
    output = await asyncio.wait_for(task, timeout=10)
    elapsed = time.monotonic() - start

    session.close()
    assert elapsed < 10
    assert "timed out" in output.lower() or "error" in output.lower()


@pytest.mark.asyncio
async def test_interrupt_session_by_id():
    import asyncio

    session_id = "interrupt_test"
    session = PtySessionManager.get_session(session_id)
    task = asyncio.ensure_future(session.exec("sleep 30", timeout=60))
    await asyncio.sleep(0.5)

    interrupted = PtySessionManager.interrupt_session(session_id)
    output = await asyncio.wait_for(task, timeout=10)

    assert interrupted is True
    assert "timed out" in output.lower() or "error" in output.lower()


def test_interrupt_session_returns_false_for_unknown_session():
    assert PtySessionManager.interrupt_session("does-not-exist") is False


@pytest.mark.asyncio
async def test_interrupt_all_stops_the_running_session():
    import asyncio

    session = PtySessionManager.get_session("interrupt_all_test")
    task = asyncio.ensure_future(session.exec("sleep 30", timeout=60))
    await asyncio.sleep(0.5)

    PtySessionManager.interrupt_all()  # must not raise even with other idle sessions
    output = await asyncio.wait_for(task, timeout=10)

    assert "timed out" in output.lower() or "error" in output.lower()
