"""A CI run that stops makes its log name the test it stopped in.

Loaded with ``-p ci_names`` (the tests folder on PYTHONPATH). It does two things:

* prints ``START <test id>`` on a line of its own as each test begins, flushed at once. With ``-q`` the progress dots
  of an unfinished line never reach a GitHub log (the runner shows whole lines), so a cancelled run used to show only
  the last finished line of 72 dots;
* after the last test, gives the process ``EXIT_SECONDS`` to end. A thread that is not a daemon, an ``atexit`` handler
  or a child that keeps a pipe open could otherwise keep pytest alive after its summary until the job's own limit.
  If the time runs out, every thread's stack is printed and the process exits with code 1.

Inside a test, pytest-timeout (``--timeout-method=thread``) and ``faulthandler_timeout`` print every thread's stack.
"""

from __future__ import annotations

import faulthandler
import sys

import pytest

EXIT_SECONDS = 60.0

_config: pytest.Config | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _config
    _config = config


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    reporter = _config.pluginmanager.get_plugin("terminalreporter") if _config else None
    if reporter is None:
        return
    writer = reporter._tw
    writer.write(("\n" if writer.width_of_current_line else "") + f"START {nodeid}\n", flush=True)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    stream = sys.__stderr__
    if stream is None:
        return
    stream.write(f"\nCI-NAMES the tests are over; the process gets {EXIT_SECONDS:.0f} s to exit\n")
    stream.flush()
    faulthandler.dump_traceback_later(EXIT_SECONDS, exit=True, file=stream)
