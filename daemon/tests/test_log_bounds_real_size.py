"""The logs at their real size limits, not the small ones test_log_bounds.py uses: pilot.log rolls at 10 MB and keeps
three old files (40 MB in all), the Gnomon worker's log at 5 MB with one old file (10 MB), and a rollover never holds
up the line being logged for long. On Windows, a pilot.log another process holds open cannot be renamed: logging
goes on, into the same file."""

from __future__ import annotations

import logging
import subprocess
import sys
import time

import pytest

from pilot import server as server_module
from pilot.intelligence.gnomon_worker import WORKER_LOG_MAX_BYTES, rotate_worker_log

LINE = "x" * 200  # about the length of a long daemon log line


def _logger(handler: logging.Handler, name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    return log


def test_the_daemon_log_at_its_real_size_holds_40_mb_and_never_stalls_a_line(tmp_path):
    limit, backups = server_module.LOG_MAX_BYTES, server_module.LOG_BACKUPS
    assert (limit, backups) == (10 * 1024 * 1024, 3)  # what RELIABILITY.md states
    handler = server_module._bounded_log_handler(tmp_path / "pilot.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    log = _logger(handler, "test.log_bounds_real")
    slowest = 0.0
    written = 0
    try:
        while written < limit * (backups + 1.5):  # past the whole bound: the oldest file must go
            began = time.perf_counter()
            log.info("line %08d %s", written, LINE)
            slowest = max(slowest, time.perf_counter() - began)
            written += len(LINE) + 60
    finally:
        log.removeHandler(handler)
        handler.close()
    files = sorted(tmp_path.iterdir())
    assert [path.name for path in files] == ["pilot.log", "pilot.log.1", "pilot.log.2", "pilot.log.3"]
    assert all(path.stat().st_size <= limit for path in files)
    assert sum(path.stat().st_size for path in files) <= limit * (backups + 1)
    assert slowest < 1.0, f"a line took {slowest:.2f} s (a rollover is a rename and a new file)"


def test_the_worker_log_at_its_real_size_keeps_one_old_file(tmp_path):
    log = tmp_path / "gnomon_worker.log"
    log.write_bytes(b"x" * (WORKER_LOG_MAX_BYTES + 1))
    rotate_worker_log(log)
    assert not log.exists() and (tmp_path / "gnomon_worker.log.1").stat().st_size == WORKER_LOG_MAX_BYTES + 1
    log.write_bytes(b"y" * WORKER_LOG_MAX_BYTES)  # at the limit, not past it: kept
    rotate_worker_log(log)
    assert log.stat().st_size == WORKER_LOG_MAX_BYTES
    log.write_bytes(b"z" * (WORKER_LOG_MAX_BYTES + 1))
    rotate_worker_log(log)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["gnomon_worker.log.1"]  # never a growing pile


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to rename a file another process holds open")
def test_on_windows_a_log_another_process_holds_open_keeps_being_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server_module, "LOG_MAX_BYTES", 100_000)
    path = tmp_path / "pilot.log"
    path.write_text("", encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, "-c", f"import time; handle = open({str(path)!r}, 'rb'); time.sleep(60)"],
    )
    try:
        time.sleep(1.0)  # the other process has it open
        handler = server_module._bounded_log_handler(path)
        log = _logger(handler, "test.log_bounds_held")
        try:
            for index in range(2000):
                log.info("line %05d %s", index, LINE)
        finally:
            log.removeHandler(handler)
            handler.close()
    finally:
        holder.kill()
        holder.wait(timeout=10)
    text = path.read_text(encoding="utf-8")
    assert all(f"line {index:05d}" in text for index in range(2000))  # no line lost, all in the same file
    assert path.stat().st_size > 100_000  # past the limit: the rollover could not happen while it was held
    assert capsys.readouterr().err.count("could not start a new log file") == 1  # said once, not per line

    # Once nobody holds it, the next try rolls over as usual.
    handler = server_module._bounded_log_handler(path)
    log = _logger(handler, "test.log_bounds_released")
    try:
        for index in range(100):
            log.info("after %05d %s", index, LINE)
    finally:
        log.removeHandler(handler)
        handler.close()
    assert (tmp_path / "pilot.log.1").exists()
