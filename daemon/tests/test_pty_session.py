"""Tests for persistent PTY session management."""

import sys
import warnings

import pytest

# forkpty() warning is a pytest multi-threading artefact; the daemon is single-threaded
warnings.filterwarnings("ignore", message=".*forkpty.*", category=DeprecationWarning)

if sys.platform == "win32":
    pytest.skip("PTY sessions are Unix-only", allow_module_level=True)

from pilot.system.pty_session import PtySession, PtySessionManager


@pytest.fixture(autouse=True)
def isolate_sessions():
    """Reset the session registry between tests."""
    PtySessionManager.close_all()
    yield
    PtySessionManager.close_all()


async def test_session_exec_basic():
    session = PtySession()
    output = await session.exec("echo hello")
    session.close()
    assert "hello" in output


async def test_session_state_persistence():
    session = PtySession()
    await session.exec("export PILOT_TEST_VAR=persistent")
    output = await session.exec("echo $PILOT_TEST_VAR")
    session.close()
    assert "persistent" in output


async def test_session_cwd_persistence():
    session = PtySession()
    await session.exec("cd /tmp")
    output = await session.exec("pwd")
    session.close()
    assert "/tmp" in output


async def test_separate_sessions_are_isolated():
    session_a = PtySessionManager.get_session("a")
    session_b = PtySessionManager.get_session("b")

    await session_a.exec("export PILOT_ISOLATED=only_in_a")
    output_b = await session_b.exec("echo ${PILOT_ISOLATED:-not_set}")

    assert "only_in_a" not in output_b


async def test_session_timeout():
    session = PtySession()
    output = await session.exec("sleep 60", timeout=1)
    session.close()
    assert "timed out" in output.lower() or "error" in output.lower()


async def test_close_and_reopen():
    session_id = "reopen_test"
    session = PtySessionManager.get_session(session_id)
    await session.exec("echo before_close")
    PtySessionManager.close_session(session_id)

    new_session = PtySessionManager.get_session(session_id)
    output = await new_session.exec("echo after_reopen")
    assert "after_reopen" in output
