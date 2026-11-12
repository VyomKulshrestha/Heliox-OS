"""A full disk or a read-only data folder: said once in plain words, the daemon keeps answering."""

import asyncio
import errno
import sqlite3
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.db.sqlite_pool import AsyncSqlitePool
from pilot.server import PilotServer
from pilot.system import storage_health
from pilot.system.storage_health import StorageHealth, classify, plain_reason


@pytest.fixture
def health(monkeypatch):
    fresh = StorageHealth(clock=lambda: 100.0)
    told: list[dict] = []
    fresh.notify = told.append
    monkeypatch.setattr(storage_health, "_shared", fresh)
    return fresh, told


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (OSError(errno.ENOSPC, "No space left on device"), "full"),
        (sqlite3.OperationalError("database or disk is full"), "full"),
        (OSError(errno.EROFS, "Read-only file system"), "read_only"),
        (OSError(errno.EACCES, "Permission denied"), "read_only"),
        (sqlite3.OperationalError("attempt to write a readonly database"), "read_only"),
        (sqlite3.OperationalError("database table is locked"), None),
        (OSError(errno.ENOENT, "No such file"), None),
        (ConnectionResetError(errno.ECONNRESET, "reset"), None),
        (ValueError("bad"), None),
    ],
)
def test_classify(error, kind):
    assert classify(error) == kind


def test_windows_disk_full_code_is_recognised():
    error = OSError(errno.EINVAL, "There is not enough space on the disk")
    error.winerror = 112
    assert classify(error) == "full"


def test_told_once_per_problem_until_a_write_succeeds(health):
    store, told = health
    full = OSError(errno.ENOSPC, "No space left on device")
    assert store.report("memory", full, "your reminder").startswith("The disk is full, so Heliox could not save your")
    store.report("timeline", full)
    assert len(told) == 1 and told[0]["kind"] == "full" and "keeps working" in told[0]["message"]
    store.report_ok()
    store.report("memory", full)
    assert len(told) == 2
    store.report("memory", OSError(errno.EROFS, "Read-only file system"))
    assert len(told) == 3 and store.status()["kind"] == "read_only"
    assert store.report("memory", ValueError("not about the disk")) is None and len(told) == 3


async def test_a_failed_sqlite_write_is_reported_and_still_raised(health, tmp_path):
    store, told = health
    pool = AsyncSqlitePool(tmp_path / "memory.db", read_pool_size=1)
    await pool.start()
    try:
        with pytest.raises(sqlite3.OperationalError):
            async with pool.write():
                raise sqlite3.OperationalError("database or disk is full")
        assert store.status()["component"] == "memory" and len(told) == 1
        async with pool.write() as db:
            await db.execute("CREATE TABLE t (x INTEGER)")
        assert store.status() is None  # a write worked again
    finally:
        await pool.close()


async def test_a_local_answer_that_cannot_save_says_why_instead_of_passing_the_request_on(health):
    store, told = health
    server = PilotServer(PilotConfig())

    class Reminders:
        async def answer_async(self, text):
            raise OSError(errno.ENOSPC, "No space left on device")

    server._reminders = Reminders()
    answer = await server._local_answer("remind me in ten minutes to stretch")
    assert answer is not None and answer.startswith("The disk is full")
    assert len(told) == 1


def test_a_settings_save_that_cannot_be_written_is_reported(health, monkeypatch, tmp_path):
    store, told = health
    from pilot import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.toml")

    def refuse(*_args, **_kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(config_module, "_atomic_write_text", refuse)
    with pytest.raises(OSError):
        PilotConfig().save()
    assert store.status()["kind"] == "read_only" and "your settings" in store.status()["message"]


def test_model_download_errors_read_in_plain_words():
    assert plain_reason("OSError: [Errno 28] No space left on device").startswith("The disk is full")
    assert "cannot be written to" in plain_reason("PermissionError: [Errno 13] Permission denied: 'C:/models'")
    assert plain_reason("HTTP 404 from the hub") == "HTTP 404 from the hub"


def test_the_status_rpc_carries_the_storage_problem(health, monkeypatch):
    from pilot.system import gpu_arbiter

    monkeypatch.setattr(
        gpu_arbiter,
        "_shared",
        gpu_arbiter.GpuArbiter(read=lambda: gpu_arbiter.MachineReading("cpu"), holder=lambda: None),
    )
    store, _told = health
    store.report("memory", OSError(errno.ENOSPC, "No space left on device"))
    server = PilotServer(PilotConfig())
    status = asyncio.run(server._handle_autonomy_review_status({}, SimpleNamespace()))
    assert status["storage"]["kind"] == "full"


def test_preflight_passes_on_a_writable_folder_and_names_a_full_one(tmp_path, monkeypatch):
    from pilot.system.storage_health import preflight

    assert preflight(tmp_path / "data") is None
    assert list((tmp_path / "data").iterdir()) == []  # leaves nothing behind

    import builtins

    real_open = builtins.open

    def full_open(path, mode="r", *args, **kwargs):
        if "heliox-write-check" in str(path) and "w" in mode:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", full_open)
    error = preflight(tmp_path / "data")
    assert isinstance(error, OSError) and classify(error) == "full"


async def test_a_daemon_that_cannot_write_its_data_folder_keeps_answering_and_starts_later(
    health, monkeypatch, tmp_path
):
    from unittest.mock import AsyncMock

    from pilot import server as server_module

    store, told = health
    checks = iter([OSError(errno.EROFS, "Read-only file system"), None])
    monkeypatch.setattr(storage_health, "preflight", lambda _folder: next(checks))
    monkeypatch.setattr(server_module, "STORAGE_RECHECK_SECONDS", 0.05)
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")  # never the real token files
    config = PilotConfig()
    config.server.host, config.server.port = "127.0.0.1", 0
    server = PilotServer(config)
    server.initialize = AsyncMock()
    server._broadcast_notification = AsyncMock()

    await server.start()
    try:
        assert server._server is not None  # listening: ping, health and status answer
        assert store.status()["kind"] == "read_only"
        for _ in range(100):
            if server.initialize.await_count:
                break
            await asyncio.sleep(0.02)
        server.initialize.assert_awaited_once()  # started once the folder could be written again
        assert store.status() is None
        said = [
            call.args for call in server._broadcast_notification.await_args_list if call.args[0] == "storage_problem"
        ]
        assert said[0][1]["kind"] == "read_only"  # told once, in plain words
        assert said[-1][1]["message"] == "Heliox can save again and is fully started."
    finally:
        await server.stop()
