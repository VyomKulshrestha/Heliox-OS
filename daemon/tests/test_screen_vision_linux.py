"""Tests for _get_active_window_linux() race-condition fix.

Verifies that all xdotool calls after the initial getactivewindow use the
captured window_id rather than re-querying the currently active window.
"""

from __future__ import annotations

import subprocess
from unittest.mock import call, patch

from pilot.agents.screen_vision import _get_active_window_linux


def _make_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_window_id_used_for_name_and_pid(tmp_path):
    """getwindowname and getwindowpid must receive the captured window_id, not getactivewindow."""
    window_id = "99999999"
    pid = "12345"

    comm_file = tmp_path / "comm"
    comm_file.write_text("firefox\n")

    call_results = [
        _make_completed(window_id),  # getactivewindow
        _make_completed("My Browser Tab"),  # getwindowname <window_id>
        _make_completed(pid),  # getwindowpid  <window_id>
    ]

    with (
        patch("subprocess.run", side_effect=call_results) as mock_run,
        patch("pilot.agents.screen_vision.Path") as mock_path,
    ):
        mock_comm = mock_path.return_value
        mock_comm.exists.return_value = True
        mock_comm.read_text.return_value = "firefox\n"

        app, title = _get_active_window_linux()

    assert app == "firefox"
    assert title == "My Browser Tab"

    calls = mock_run.call_args_list
    assert len(calls) == 3

    # First call: getactivewindow (no window_id argument yet)
    assert calls[0] == call(
        ["xdotool", "getactivewindow"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # Second call: getwindowname must use the captured window_id
    assert calls[1] == call(
        ["xdotool", "getwindowname", window_id],
        capture_output=True,
        text=True,
        timeout=5,
    ), "getwindowname should query by window_id, not re-call getactivewindow"

    # Third call: getwindowpid must use the captured window_id
    assert calls[2] == call(
        ["xdotool", "getwindowpid", window_id],
        capture_output=True,
        text=True,
        timeout=5,
    ), "getwindowpid should query by window_id, not re-call getactivewindow"


def test_no_getactivewindow_repeated():
    """Confirm that getactivewindow is called exactly once."""
    call_results = [
        _make_completed("12345678"),
        _make_completed("Some Title"),
        _make_completed("9876"),
    ]

    with (
        patch("subprocess.run", side_effect=call_results) as mock_run,
        patch("pilot.agents.screen_vision.Path") as mock_path,
    ):
        mock_comm = mock_path.return_value
        mock_comm.exists.return_value = False

        _get_active_window_linux()

    active_window_calls = [c for c in mock_run.call_args_list if c.args[0] == ["xdotool", "getactivewindow"]]
    assert len(active_window_calls) == 1, (
        "getactivewindow must be called exactly once; subsequent calls must use the captured window_id"
    )


def test_returns_unknown_on_initial_failure():
    """Returns ('Unknown', 'Unknown') when the initial getactivewindow call fails."""
    with patch("subprocess.run", return_value=_make_completed("", returncode=1)):
        app, title = _get_active_window_linux()

    assert app == "Unknown"
    assert title == "Unknown"


def test_title_empty_on_getwindowname_failure():
    """title is empty string when getwindowname fails, but app lookup continues."""
    call_results = [
        _make_completed("11111111"),
        _make_completed("", returncode=1),  # getwindowname fails
        _make_completed("5555"),  # getwindowpid succeeds
    ]

    with (
        patch("subprocess.run", side_effect=call_results),
        patch("pilot.agents.screen_vision.Path") as mock_path,
    ):
        mock_comm = mock_path.return_value
        mock_comm.exists.return_value = True
        mock_comm.read_text.return_value = "vim\n"

        app, title = _get_active_window_linux()

    assert title == ""
    assert app == "vim"
