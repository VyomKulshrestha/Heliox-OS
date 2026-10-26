"""Finding your files from folder listings: downloads and "where is my …"."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.file_finder import FileFinder, names_a_kind

NOW = datetime(2026, 10, 26, 14, 0)


def touch(path, when: datetime, size: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


@pytest.fixture
def home(tmp_path):
    downloads = tmp_path / "Downloads"
    documents = tmp_path / "Documents"
    touch(downloads / "invoice-october.pdf", datetime(2026, 10, 26, 9, 30), 2048)
    touch(downloads / "photo.jpg", datetime(2026, 10, 26, 11, 0), 4096)
    touch(downloads / "movie.mp4.crdownload", datetime(2026, 10, 26, 13, 0))
    touch(downloads / "notes.txt", datetime(2026, 10, 25, 18, 0))
    touch(downloads / "old.zip", datetime(2026, 10, 1, 8, 0))
    touch(documents / "Finance" / "Budget 2026.xlsx", datetime(2026, 10, 20, 10, 0))
    touch(documents / "Finance" / "budget-notes.docx", datetime(2026, 10, 22, 10, 0))
    touch(documents / "TextKeys.txt", datetime(2026, 6, 1, 10, 0))
    # Inside a git repository, only the top level is searched.
    touch(documents / "project" / ".git" / "HEAD", datetime(2026, 10, 1))
    touch(documents / "project" / "src" / "budget_model.xlsx", datetime(2026, 10, 25))
    return tmp_path


@pytest.fixture
def finder(home):
    return FileFinder(
        roots=lambda: [home / "Downloads", home / "Documents"],
        downloads=lambda: home / "Downloads",
        now=lambda: NOW,
    )


def test_downloads_by_day(finder):
    assert finder.answer("What did I download today?") == (
        "You downloaded 2 files today: photo.jpg (4.0 KB); invoice-october.pdf (2.0 KB)."
    )
    assert finder.answer("what did I download yesterday") == "You downloaded 1 file yesterday: notes.txt (10 bytes)."
    assert finder.answer("what was the last file I downloaded") == (
        "The last thing you downloaded was photo.jpg (4.0 KB), today at 11:00 AM."
    )


def test_where_is_my_file(finder):
    answer = finder.answer("where is my budget spreadsheet")
    assert answer.startswith("I found Budget 2026.xlsx in ")
    assert "budget_model" not in answer  # inside a git repository's subfolders
    assert finder.answer("find my budget notes").startswith("I found budget-notes.docx in ")
    assert finder.answer("where's my invoice pdf").startswith("I found invoice-october.pdf in ")


def test_not_found(finder):
    assert finder.answer("where is my tax spreadsheet") == (
        "I couldn't find a spreadsheet named tax in Downloads, Documents."
    )
    # No kind of file named and nothing found: left to the conversation and the planner.
    assert finder.answer("where are my keys") is None  # TextKeys.txt does not count: names match at word starts
    assert finder.answer("find my phone") is None
    assert finder.answer("where is my spreadsheet") is None
    assert finder.answer("open my budget spreadsheet") is None


def test_names_a_kind():
    assert names_a_kind("where is my budget spreadsheet")
    assert not names_a_kind("where are my keys")
    assert not names_a_kind("open my documents")


def test_empty_downloads(tmp_path):
    empty = tmp_path / "Downloads"
    empty.mkdir()
    finder = FileFinder(roots=lambda: [empty], downloads=lambda: empty, now=lambda: NOW)
    assert finder.answer("what did I download today") == "You didn't download anything today."
    assert finder.answer("what did I just download") == "Your Downloads folder is empty."


@pytest.mark.asyncio
async def test_routing(home):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._file_finder = FileFinder(
        roots=lambda: [home / "Documents"], downloads=lambda: home / "Downloads", now=lambda: NOW
    )

    assert (await server._control_reply("where is my budget spreadsheet")).startswith("I found Budget 2026.xlsx")
    assert await server._control_reply("where are my keys") is None
    assert server._phrase_reserved("what did i download today")
    assert not server._phrase_reserved("find my phone")
