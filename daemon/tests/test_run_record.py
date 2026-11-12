"""Whether the daemon's last run ended cleanly (pilot.system.run_record), and what is said about it."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from pilot.system.run_record import HISTORY, RunRecord

START = 1_763_000_000.0  # 13 November 2025, any fixed moment


def record(path: Path, *, at: float, boot: float | None = 0.0, previous_exit: str = "", pid: int = 100) -> RunRecord:
    return RunRecord(
        path, version="0.16.0", pid=pid, clock=lambda: at, boot_time=lambda: boot, previous_exit=previous_exit
    )


def test_a_first_run_and_a_clean_stop_report_nothing(tmp_path):
    path = tmp_path / "run_state.json"
    first = record(path, at=START)
    assert first.begin() is None and first.summary() is None
    first.end_cleanly("stopped")

    second = record(path, at=START + 60)
    assert second.begin() is None and second.summary() is None
    assert json.loads(path.read_text(encoding="utf-8"))["ended"] is None  # this run is now the one running


def test_a_run_that_did_not_end_is_said_once_and_never_called_a_crash(tmp_path):
    path = tmp_path / "run_state.json"
    record(path, at=START).begin()  # killed: no end_cleanly

    later = record(path, at=START + 600)
    last = later.begin()

    assert last["ended"] == "did not end cleanly"
    assert "crash" not in last["notice"].lower() and "did not end cleanly" in last["notice"]
    summary = later.summary()
    assert summary["notice"] == last["notice"] and summary["unclean_today"] == 1 and summary["id"] == str(START)


def test_the_computer_restarting_is_not_heliox_ending_and_the_window_is_not_told(tmp_path):
    path = tmp_path / "run_state.json"
    record(path, at=START).begin()

    after_boot = record(path, at=START + 3600, boot=START + 1800)  # the system booted after that run started
    last = after_boot.begin()

    assert last["ended"] == "computer restarted" and last["notice"] is None
    assert after_boot.summary()["unclean_today"] == 0


@pytest.mark.parametrize(
    ("seen", "ended", "words"),
    [
        ("1", "stopped with exit code 1", "exit code 1"),
        ("hung", "stopped answering", "stopped answering"),
        ("signal", "did not end cleanly", "did not end cleanly"),
        ("", "did not end cleanly", "did not end cleanly"),
    ],
)
def test_what_the_supervisor_saw_is_said_when_it_passed_it(tmp_path, seen, ended, words):
    path = tmp_path / "run_state.json"
    record(path, at=START).begin()

    last = record(path, at=START + 60, previous_exit=seen).begin()

    assert last["ended"] == ended and words in last["notice"]


def test_a_daemon_replaced_on_purpose_ended_cleanly(tmp_path):
    path = tmp_path / "run_state.json"
    record(path, at=START, pid=4242).begin()  # an update or a repair ends it without its stop()

    assert record(path, at=START + 5, pid=5000).begin(replaced_pid=4242) is None
    assert record(path, at=START + 9, pid=6000).begin(replaced_pid=4242)["ended"] == "did not end cleanly"


def test_a_file_that_cannot_be_read_counts_as_unknown_is_replaced_and_is_not_reported(tmp_path):
    path = tmp_path / "run_state.json"
    path.write_text("{not json", encoding="utf-8")

    run = record(path, at=START)
    assert run.begin()["ended"] == "unknown"
    assert run.summary() is None
    assert json.loads(path.read_text(encoding="utf-8"))["started_at"] == START  # replaced by this run's marker


def test_a_folder_that_cannot_be_written_means_no_record_and_no_error(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "run_state.json"
    record(path, at=START).begin()
    if sys.platform == "win32":
        os.chmod(path, stat.S_IREAD)  # the file read-only: the marker cannot be replaced
    else:
        os.chmod(folder, stat.S_IREAD | stat.S_IEXEC)
    try:
        run = record(path, at=START + 60)
        assert run.begin()["ended"] == "did not end cleanly"  # read, said, never raised
        run.end_cleanly("stopped")  # nothing to write to: quietly nothing
    finally:
        os.chmod(path if sys.platform == "win32" else folder, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)


def test_a_data_folder_known_unwritable_writes_nothing(tmp_path):
    path = tmp_path / "run_state.json"
    assert record(path, at=START).begin(folder_writable=False) is None
    assert not path.exists()


def test_the_history_keeps_the_last_unclean_ends_only(tmp_path):
    path = tmp_path / "run_state.json"
    for number in range(HISTORY + 5):
        record(path, at=START + number).begin()  # each one killed
    history = json.loads(path.read_text(encoding="utf-8"))["history"]
    assert len(history) == HISTORY and history[-1]["started_at"] == START + HISTORY + 3


def test_a_failure_inside_never_reaches_the_start(tmp_path, monkeypatch):
    path = tmp_path / "run_state.json"
    run = record(path, at=START)
    monkeypatch.setattr(run, "_read", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run.begin() is None


async def test_the_server_keeps_its_run_record_in_the_state_folder_and_says_the_last_one(tmp_path, monkeypatch):
    """State, not the person's data: beside the log, so a repair or "keep my data" never counts it as theirs. A record
    an earlier version left in the data folder is read once and removed."""
    from pilot.config import PilotConfig
    from pilot.server import PilotServer
    from pilot.system import run_record

    data, state = tmp_path / "data", tmp_path / "state"
    data.mkdir()
    monkeypatch.setattr("pilot.server.DATA_DIR", data)
    monkeypatch.setattr("pilot.server.STATE_DIR", state)
    (data / run_record.FILE_NAME).write_text(
        json.dumps({"pid": 1, "version": "0.15.0", "started_at": time.time() - 1, "ended": None, "history": []}),
        encoding="utf-8",
    )
    server = PilotServer(PilotConfig())
    await server._begin_run_record(folder_writable=True)

    health = await server._handle_health({}, None)
    assert health["last_run"]["ended"] == "did not end cleanly" and health["last_run"]["version"] == "0.15.0"
    assert not (data / run_record.FILE_NAME).exists()  # the data folder holds nothing of it now
    server._run_record.end_cleanly("stopped")
    assert json.loads((state / run_record.FILE_NAME).read_text(encoding="utf-8"))["ended"] == "clean"
    assert sorted(path.name for path in data.iterdir()) == []


def test_a_record_in_the_state_folder_wins_over_an_old_one_in_the_data_folder(tmp_path):
    old, new = tmp_path / "data" / "run_state.json", tmp_path / "state" / "run_state.json"
    old.parent.mkdir()
    new.parent.mkdir()
    old.write_text(json.dumps({"pid": 1, "started_at": START, "ended": None}), encoding="utf-8")
    new.write_text(json.dumps({"pid": 2, "started_at": START, "ended": "clean"}), encoding="utf-8")

    assert (
        RunRecord(
            new, version="x", clock=lambda: START + 9, boot_time=lambda: 0.0, previous_exit="", legacy=old
        ).begin()
        is None
    )
    assert not old.exists()
