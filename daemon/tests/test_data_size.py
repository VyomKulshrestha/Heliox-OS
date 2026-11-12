"""The data folder's size and each unbounded store's, shown on request, with one plain warning per crossing."""

from datetime import date
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.system.data_rpc import DataSize
from pilot.system.data_size import GB, SizeWatch, sizes


def _store(folder, name, size):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_each_store_and_the_whole_folder_are_measured(tmp_path):
    _store(tmp_path, "audit.jsonl", 100)
    _store(tmp_path, "audit-2026-01-01.jsonl", 40)
    _store(tmp_path, "durable_tasks.db", 30)
    _store(tmp_path, "durable_tasks.db-wal", 5)
    _store(tmp_path, "workflow_checkpoints.db", 7)
    _store(tmp_path, "experience.db", 11)
    _store(tmp_path, "receipts/r1.json", 3)  # not a named store, but part of the folder
    found = sizes(tmp_path)
    by_name = {store["name"]: store["bytes"] for store in found["stores"]}
    assert by_name == {
        "Audit log": 100,
        "Audit log archives": 40,
        "Durable tasks": 35,
        "Checkpoints": 7,
        "Experience ledger": 11,
    }
    assert found["total_bytes"] == 196


def test_a_missing_folder_is_zero_not_an_error(tmp_path):
    assert sizes(tmp_path / "never-made")["total_bytes"] == 0


def test_the_warning_is_said_once_per_crossing():
    limit = {"gb": 5}
    watch = SizeWatch(lambda: limit["gb"])
    snapshot = lambda total: {"total_bytes": total, "stores": [{"name": "Audit log", "bytes": total // 2}]}  # noqa: E731
    assert watch.check(snapshot(4 * GB)) is None
    said = watch.check(snapshot(6 * GB))
    assert said == (
        "Heliox's data folder is 6.0 GB (audit log 3.0 GB), over the 5 GB you set. Nothing is deleted on its own; "
        "you can archive the older audit log in Settings, or raise the limit there."
    )
    assert watch.check(snapshot(7 * GB)) is None  # still over: not said again
    assert watch.check(snapshot(3 * GB)) is None  # back under (archived): quiet
    assert watch.check(snapshot(6 * GB)) is not None  # over again: said again
    limit["gb"] = 10  # the person raised the limit
    assert watch.check(snapshot(6 * GB)) is None and watch.over is False


@pytest.mark.parametrize(("written", "kept"), [(12, 12), (0, 1), (10**9, 100000)])
def test_the_limit_is_read_from_the_config_file_and_kept_in_range(written, kept):
    config = _merge_config(PilotConfig(), {"storage": {"data_folder_warn_gb": written}})
    assert config.storage.data_folder_warn_gb == kept
    assert PilotConfig().storage.data_folder_warn_gb == 5


class _Server:
    def __init__(self, gb=5):
        self.config = SimpleNamespace(storage=SimpleNamespace(data_folder_warn_gb=gb))
        self.notes = []

    async def _broadcast_notification(self, method, params):
        self.notes.append((method, params))


async def test_the_rpc_reports_sizes_and_the_crossing_reaches_the_window_once(tmp_path, monkeypatch):
    server = _Server(gb=1)
    data = DataSize(server, tmp_path)
    _store(tmp_path, "audit.jsonl", 10)
    monkeypatch.setattr("pilot.system.data_rpc.sizes", lambda folder: {**sizes(folder), "total_bytes": 2 * GB})
    first = await data.storage_sizes({})
    second = await data.storage_sizes({})
    assert first["status"] == "ok" and first["over"] is True and first["warn_gb"] == 1
    assert [store["name"] for store in first["stores"]][0] == "Audit log"
    assert second["over"] is True
    assert [method for method, _ in server.notes] == ["storage_problem"]
    assert server.notes[0][1]["kind"] == "size"


class _Audit:
    def __init__(self, outcome):
        self.outcome = outcome
        self.asked = []

    async def archive_older_than(self, older_than):
        self.asked.append(older_than)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


async def test_the_archive_rpc_answers_in_plain_words(tmp_path):
    from pilot.security.audit_archive import ArchiveRefused

    data = DataSize(_Server(), tmp_path)
    assert (await data.audit_archive({"older_than": "last spring"}, _Audit({})))["message"] == (
        "Choose the day as YYYY-MM-DD."
    )
    assert "still starting" in (await data.audit_archive({"older_than": "2026-01-01"}, None))["message"]
    refused = await data.audit_archive({"older_than": "2026-01-01"}, _Audit(ArchiveRefused("Not enough room.")))
    assert refused == {"status": "error", "message": "Not enough room."}
    record = {"lines": 3, "older_than": "2026-01-01", "archive": "audit-2026-11-12.jsonl"}
    audit = _Audit(record)
    done = await data.audit_archive({"older_than": "2026-01-01"}, audit)
    assert audit.asked == [date(2026, 1, 1)]
    assert done["status"] == "ok" and done["archived"] == record
    assert done["message"] == (
        "Moved 3 audit entries from before 2026-01-01 to audit-2026-11-12.jsonl, beside the audit log. "
        "Nothing was deleted."
    )
