"""The driver's error count (pilot.system.gpu_driver_record): read from the system's own log where an ordinary user
can, and "no record" whenever it cannot be read. Stand-in command results only; no log is read."""

from __future__ import annotations

import subprocess

import pytest

from pilot.system import gpu_driver_record as record


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    record.reset()
    monkeypatch.setattr(record.shutil, "which", lambda name: f"/bin/{name}")
    yield
    record.reset()


def answer(monkeypatch, returncode=0, stdout="", stderr="", fails=False):
    asked = []

    def run(command):
        asked.append(command)
        if fails:
            return None
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(record, "_run", run)
    return asked


def test_windows_counts_the_driver_s_errors_of_the_last_week_in_the_system_log(monkeypatch):
    asked = answer(monkeypatch, stdout="<Events>" + "<Event xmlns='x'>...</Event>" * 4)
    assert record.read_count("win32") == 4
    command = asked[0]
    assert command[:3] == ["wevtutil", "qe", "System"]
    query = command[3]
    assert "nvlddmkm" in query and "4101" in query and "4107" in query and "604800000" in query


@pytest.mark.parametrize("kind", ["refused", "missing", "timed out"])
def test_a_log_that_cannot_be_read_is_no_record(monkeypatch, kind):
    if kind == "refused":
        answer(monkeypatch, returncode=5, stderr="Access is denied.")
    elif kind == "missing":
        monkeypatch.setattr(record.shutil, "which", lambda name: None)
    else:
        answer(monkeypatch, fails=True)
    assert record.read_count("win32") is None


def test_linux_counts_xid_lines_only_where_the_user_may_read_the_kernel_journal(monkeypatch):
    asked = answer(monkeypatch, stdout="NVRM: Xid (PCI:0000:01:00): 13, Graphics SM Warp Exception\nNVRM: Xid 31\n")
    assert record.read_count("linux") == 2
    assert asked[0][:2] == ["journalctl", "-k"] and "sudo" not in asked[0]
    answer(monkeypatch, returncode=1)  # nothing matched
    assert record.read_count("linux") == 0
    answer(monkeypatch, stderr="No journal files were opened due to insufficient permissions.")
    assert record.read_count("linux") is None


def test_macos_has_nothing_to_read():
    assert record.read_count("darwin") is None


def test_the_read_runs_once_in_the_background_and_a_failure_is_no_record(monkeypatch):
    calls = []

    def boom(platform=None):
        calls.append(1)
        raise RuntimeError("the log service is gone")

    monkeypatch.setattr(record, "read_count", boom)
    assert record.errors_this_week(wait=2.0) is None
    assert record.errors_this_week(wait=2.0) is None
    assert calls == [1]


def test_the_status_for_settings_never_starts_a_read_of_the_system_log(monkeypatch):
    record.reset()
    calls = []
    monkeypatch.setattr(record, "read_count", lambda platform=None: calls.append(1) or 4)
    before = record.status(False)
    assert calls == [] and record.known_count() is None
    assert before == {
        "errors": None,
        "limit": record.LIMIT,
        "window_days": record.WINDOW_DAYS,
        "speech_on_processor": False,
        "speech_despite_driver_errors": False,
    }
    record.note_kept_off(5)  # the rule moved Whisper with five counted (errors and this session's faults)
    after = record.status(True)
    assert after["errors"] == 5 and after["speech_on_processor"] is True and after["speech_despite_driver_errors"]
    assert calls == []
    record.reset()
    assert record.status(False)["speech_on_processor"] is False
