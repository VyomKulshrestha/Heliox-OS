"""Archiving the older audit log: only when asked, copied and checked before the live file changes, never truncated in
place, refused in plain words when it cannot be done safely."""

import asyncio
import hashlib
import json
from datetime import date

import pytest

from pilot.security import audit_archive
from pilot.security.audit import AuditEntry, AuditLogger
from pilot.security.audit_archive import MANIFEST_NAME, ArchiveRefused, archive_older_than

PLENTY = lambda _folder: 10 * 2**30  # noqa: E731 - free space a test does not care about


def _line(day: str, n: int) -> bytes:
    return (json.dumps({"timestamp": f"{day}T10:00:00+00:00", "event_type": "action", "n": n}) + "\n").encode()


def _log(tmp_path, *lines: bytes):
    live = tmp_path / "audit.jsonl"
    live.write_bytes(b"".join(lines))
    return live


def test_older_lines_move_to_a_dated_archive_in_order_and_the_rest_stay(tmp_path):
    lines = [_line("2026-01-05", 1), _line("2026-03-01", 2), b"not json at all\n", _line("2026-01-20", 3)]
    live = _log(tmp_path, *lines)

    record = archive_older_than(live, date(2026, 2, 1), today=date(2026, 11, 12), free_bytes=PLENTY)

    archive = tmp_path / "audit-2026-11-12.jsonl"
    assert archive.read_bytes() == lines[0] + lines[3]  # unchanged, in order
    assert live.read_bytes() == lines[1] + lines[2]  # an undated line is never moved on a guess
    assert record["lines"] == 2 and record["live_lines_kept"] == 2
    assert record["sha256"] == hashlib.sha256(lines[0] + lines[3]).hexdigest()
    manifest = [json.loads(row) for row in (tmp_path / MANIFEST_NAME).read_text().splitlines()]
    assert manifest == [record]
    assert not (tmp_path / "audit.jsonl.new").exists()


def test_a_second_archive_on_the_same_day_gets_its_own_name(tmp_path):
    live = _log(tmp_path, _line("2026-01-05", 1), _line("2026-01-06", 2), _line("2026-06-01", 3))
    archive_older_than(live, date(2026, 1, 6), today=date(2026, 11, 12), free_bytes=PLENTY)
    archive_older_than(live, date(2026, 2, 1), today=date(2026, 11, 12), free_bytes=PLENTY)
    assert (tmp_path / "audit-2026-11-12.jsonl").read_bytes() == _line("2026-01-05", 1)
    assert (tmp_path / "audit-2026-11-12-2.jsonl").read_bytes() == _line("2026-01-06", 2)
    assert live.read_bytes() == _line("2026-06-01", 3)
    assert len((tmp_path / MANIFEST_NAME).read_text().splitlines()) == 2


@pytest.mark.parametrize(
    ("older_than", "words"),
    [(date(2027, 1, 1), "not in the future"), (date(2026, 1, 1), "no entries from before 2026-01-01")],
)
def test_nothing_to_do_is_said_and_nothing_changes(tmp_path, older_than, words):
    live = _log(tmp_path, _line("2026-05-05", 1))
    before = live.read_bytes()
    with pytest.raises(ArchiveRefused, match=words):
        archive_older_than(live, older_than, today=date(2026, 11, 12), free_bytes=PLENTY)
    assert live.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audit.jsonl"]


def test_not_enough_free_space_is_refused_before_anything_is_written(tmp_path):
    live = _log(tmp_path, _line("2026-01-05", 1), _line("2026-06-01", 2))
    before = live.read_bytes()
    with pytest.raises(ArchiveRefused, match="not enough free space.*Nothing was changed"):
        archive_older_than(live, date(2026, 2, 1), today=date(2026, 11, 12), free_bytes=lambda _f: 1024)
    assert live.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audit.jsonl"]


def test_an_archive_that_does_not_read_back_leaves_the_live_log_alone(tmp_path, monkeypatch):
    live = _log(tmp_path, _line("2026-01-05", 1), _line("2026-06-01", 2))
    before = live.read_bytes()
    real = audit_archive._write_synced

    def corrupt(path, data, mode):
        real(path, data[:-3] + b"}\n" if mode == "xb" else data, mode)

    monkeypatch.setattr(audit_archive, "_write_synced", corrupt)
    with pytest.raises(ArchiveRefused, match="did not read back"):
        archive_older_than(live, date(2026, 2, 1), today=date(2026, 11, 12), free_bytes=PLENTY)
    assert live.read_bytes() == before
    assert not (tmp_path / "audit-2026-11-12.jsonl").exists()


def test_a_failure_while_writing_the_new_live_file_leaves_the_old_one_whole(tmp_path, monkeypatch):
    live = _log(tmp_path, _line("2026-01-05", 1), _line("2026-06-01", 2))
    before = live.read_bytes()

    def no_rename(_src, _dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(audit_archive.os, "replace", no_rename)
    with pytest.raises(OSError):
        archive_older_than(live, date(2026, 2, 1), today=date(2026, 11, 12), free_bytes=PLENTY)
    assert live.read_bytes() == before  # never truncated in place: the old file is untouched
    assert (tmp_path / "audit-2026-11-12.jsonl").read_bytes() == _line("2026-01-05", 1)


async def test_entries_written_while_archiving_are_not_lost(tmp_path, monkeypatch):
    live = _log(tmp_path, _line("2020-01-05", 1), _line("2020-06-01", 2))
    logger = AuditLogger(live)
    entered, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    real = audit_archive.archive_older_than

    def slow_archive(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return real(*args, **kwargs, free_bytes=PLENTY)

    monkeypatch.setattr(audit_archive, "archive_older_than", slow_archive)
    archiving = asyncio.create_task(logger.archive_older_than(date(2020, 2, 1)))
    await entered.wait()
    writing = asyncio.create_task(logger._write(AuditEntry("action", "file_write", "/tmp/x")))
    await asyncio.sleep(0)
    assert not writing.done()  # held by the archive's lock, not appended to the file being replaced
    release.set()
    await archiving
    await writing
    kept = live.read_text().splitlines()
    assert json.loads(kept[0])["n"] == 2 and json.loads(kept[1])["target"] == "/tmp/x"
