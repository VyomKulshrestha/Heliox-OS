"""Logs stop growing: pilot.log rolls over at 10 MB (four files at most), the Gnomon worker's log at 5 MB (two)."""

import logging

from pilot import server as server_module
from pilot.intelligence.gnomon_worker import rotate_worker_log


def test_the_daemon_log_rolls_over_and_keeps_three_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "LOG_MAX_BYTES", 2000)
    handler = server_module._bounded_log_handler(tmp_path / "pilot.log")
    log = logging.getLogger("test.log_bounds")
    log.propagate = False
    log.addHandler(handler)
    try:
        for index in range(400):
            log.warning("line %04d %s", index, "x" * 60)
    finally:
        log.removeHandler(handler)
        handler.close()
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["pilot.log", "pilot.log.1", "pilot.log.2", "pilot.log.3"]
    assert all(path.stat().st_size <= 2000 + 200 for path in tmp_path.iterdir())


def test_the_worker_log_is_moved_aside_past_its_limit(tmp_path):
    log = tmp_path / "gnomon_worker.log"
    log.write_bytes(b"x" * 50)
    rotate_worker_log(log, max_bytes=100)
    assert log.exists() and not (tmp_path / "gnomon_worker.log.1").exists()

    log.write_bytes(b"y" * 150)
    rotate_worker_log(log, max_bytes=100)
    assert not log.exists() and (tmp_path / "gnomon_worker.log.1").read_bytes() == b"y" * 150

    log.write_bytes(b"z" * 150)
    rotate_worker_log(log, max_bytes=100)
    assert (tmp_path / "gnomon_worker.log.1").read_bytes() == b"z" * 150  # one old file, not a growing pile
    rotate_worker_log(tmp_path / "missing.log")  # nothing there yet: no error
