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
    assert told[1]["message"] == "Heliox can save again." and store.status() is None  # said once when it is over
    store.report_ok()
    assert len(told) == 2
    store.report("memory", full)
    assert len(told) == 3
    store.report("memory", OSError(errno.EROFS, "Read-only file system"))
    assert len(told) == 4 and store.status()["kind"] == "read_only"
    assert store.report("memory", ValueError("not about the disk")) is None and len(told) == 4


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


async def _pool_with_table(tmp_path):
    pool = AsyncSqlitePool(tmp_path / "memory.db", read_pool_size=1)
    await pool.start()
    async with pool.write() as db:
        await db.execute("CREATE TABLE notes (body BLOB)")
        await db.commit()
    return pool


async def _write_blob(pool, size=64 * 1024):
    async with pool.write() as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute("INSERT INTO notes VALUES (?)", (b"x" * size,))
        await db.commit()


async def test_sqlite_itself_running_out_of_room_is_said_once_and_recovers(health, tmp_path):
    """A real SQLITE_FULL from inside SQLite (its page limit stands in for a disk that cannot grow), mid-transaction:
    said once, the daemon's next writes fail cleanly instead of 'within a transaction', and it recovers."""
    store, told = health
    pool = await _pool_with_table(tmp_path)
    try:
        async with pool.write() as db:
            cursor = await db.execute("PRAGMA page_count")
            pages = (await cursor.fetchone())[0]
            await cursor.close()
            await db.execute(f"PRAGMA max_page_count = {pages + 2}")
        for _ in range(3):
            with pytest.raises(sqlite3.OperationalError, match="full"):
                await _write_blob(pool)
        assert [item["kind"] for item in told] == ["full"]  # once, not once per failed write

        async with pool.write() as db:  # space returns
            await db.execute("PRAGMA max_page_count = 1073741823")
        await _write_blob(pool)
        assert store.status() is None and told[-1]["message"] == "Heliox can save again."
    finally:
        await pool.close()


async def test_a_database_that_turns_read_only_while_running_is_said_once_and_recovers(health, tmp_path):
    """SQLite's own read-only refusal (query_only stands in for a folder or drive that became read-only)."""
    store, told = health
    pool = await _pool_with_table(tmp_path)
    try:
        async with pool.write() as db:
            await db.execute("PRAGMA query_only = ON")
        for _ in range(2):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                await _write_blob(pool)
        assert [item["kind"] for item in told] == ["read_only"]
        assert "cannot be written to" in store.status()["message"]

        async with pool.write() as db:
            await db.execute("PRAGMA query_only = OFF")
        await _write_blob(pool)
        assert store.status() is None and told[-1]["message"] == "Heliox can save again."
    finally:
        await pool.close()


async def test_a_request_whose_store_is_full_is_answered_with_the_sentence_not_an_internal_error(
    health, monkeypatch, tmp_path
):
    """CI, a tmpfs filled to the last byte: "note that ..." got {'code': -32603, 'message': 'Internal error: database
    or disk is full'} because the durable task store raised before the note was taken. Any request whose handler
    meets a full or read-only store now gets the plain sentence: execute in its result, which the window shows as the
    task's outcome; any other method as a JSON-RPC error of its own (STORAGE_ERROR_CODE) whose message is the sentence,
    so a caller that reads a list or a status object is not handed a result without its fields."""
    import json
    from unittest.mock import AsyncMock

    import websockets

    store, told = health
    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    config = PilotConfig()
    config.server.host, config.server.port, config.server.auth_token = "127.0.0.1", 0, "test-token"
    server = PilotServer(config)
    server.initialize = AsyncMock()
    server._broadcast_notification = AsyncMock()

    async def full(_params, _ws):
        raise sqlite3.OperationalError("database or disk is full")

    async def read_only(_params, _ws):
        raise OSError(errno.EROFS, "Read-only file system")

    async def broken(_params, _ws):
        raise ValueError("not about the disk")

    await server.start()
    server._handlers = {"execute": full, "note_add": read_only, "todo_add": broken}
    try:
        port = server._server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "auth", "params": {"token": "test-token"}, "id": 1}))
            replies = {}
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "execute", "params": {"input": "note x"}, "id": 2}))
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "todo_add", "params": {"text": "x"}, "id": 3}))
            await ws.send(json.dumps({"jsonrpc": "2.0", "method": "note_add", "params": {"text": "x"}, "id": 4}))
            while len(replies) < 3:
                message = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if message.get("id") in (2, 3, 4):
                    replies[message["id"]] = message
        answer = replies[2]["result"]
        assert answer["status"] == "error" and answer["storage_problem"] is True
        assert answer["message"].startswith("The disk is full, so Heliox could not save that.")
        assert answer["explanation"] == answer["message"]
        assert {notice["kind"] for notice in told} == {"full", "read_only"}  # each said once (the requests race)
        assert replies[3]["error"]["code"] == -32603  # anything else is still an internal error
        assert replies[3]["error"]["message"].startswith("Internal error:")
        from pilot.server import STORAGE_ERROR_CODE

        listed = replies[4]["error"]  # a list-style method: an error, the sentence alone, and what kind
        assert listed["code"] == STORAGE_ERROR_CODE != -32603 and "result" not in replies[4]
        assert listed["message"].startswith("Heliox's data folder cannot be written to, so it could not save that.")
        assert listed["data"] == {"storage_problem": True, "kind": "read_only"}
    finally:
        await server.stop()


def test_an_unwritable_data_folder_does_not_stop_the_daemon_before_it_listens(health, monkeypatch, tmp_path):
    """CI, the data folder on a read-only tmpfs: ensure_dirs raised and the daemon ended before listening, so the
    window could not connect and nobody said why. Now the other folders are made and start-up's own check says why."""
    from pathlib import Path

    from pilot import config as config_module

    folders = {name: tmp_path / name for name in ("config", "data", "state", "runtime", "plugins")}
    for name, attribute in (
        ("config", "CONFIG_DIR"),
        ("data", "DATA_DIR"),
        ("state", "STATE_DIR"),
        ("runtime", "RUNTIME_DIR"),
        ("plugins", "PLUGINS_DIR"),
    ):
        monkeypatch.setattr(config_module, attribute, folders[name])
    monkeypatch.setattr(config_module, "SCREENSHOTS_DIR", folders["data"] / "screenshots")
    real_mkdir = Path.mkdir

    def read_only(self, *args, **kwargs):
        if folders["data"] == self or folders["data"] in self.parents:
            raise OSError(errno.EROFS, "Read-only file system")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", read_only)
    config_module.ensure_dirs()  # no exception
    assert all(folders[name].is_dir() for name in ("config", "state", "runtime", "plugins"))
    assert not folders["data"].exists()

    monkeypatch.setattr(Path, "mkdir", real_mkdir)
    folders["data"].mkdir()
    real_write = Path.write_text

    def full(self, *args, **kwargs):
        if self.name == ".write_test":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full)
    config_module.ensure_dirs()  # a full disk is left to start-up too
