"""Saved snippets and reading the clipboard back, never passwords or keys."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.clipboard_history import ClipboardHistory
from pilot.workflows.snippets import SnippetStore


class Clipboard:
    def __init__(self, text="(clipboard is empty)"):
        self.text = text
        self.written: list[str] = []

    async def read(self) -> str:
        return self.text

    async def write(self, text: str) -> None:
        self.written.append(text)
        self.text = text


@pytest.fixture
def board():
    return Clipboard()


@pytest.fixture
def snippets(tmp_path, board):
    return SnippetStore(tmp_path / "snippets.json", read=board.read, write=board.write)


@pytest.mark.asyncio
async def test_save_from_clipboard_and_copy_back(snippets, board):
    assert await snippets.answer_async("save that as my signature") == (
        "The clipboard is empty. Copy the text first, then say it again."
    )
    board.text = "Best regards,\nJane Doe\nHeliox"
    assert await snippets.answer_async("Save that as my signature.") == (
        'Saved "Best regards, Jane Doe Heliox" as your signature. Say "copy my signature" to use it.'
    )
    board.text = "something else"
    assert await snippets.answer_async("copy my signature") == (
        "Your signature is on the clipboard. Press Ctrl+V to paste it."
    )
    assert board.written == ["Best regards,\nJane Doe\nHeliox"]
    assert (
        await snippets.answer_async("what snippets do I have")
        == "Your snippets: signature (Best regards, Jane Doe Heliox)."
    )
    assert await snippets.answer_async("delete my signature snippet") == "Deleted your signature snippet."
    assert await snippets.answer_async("delete my signature snippet") == "You have no snippet called signature."


@pytest.mark.asyncio
async def test_secrets_are_never_saved_or_read(snippets, board):
    board.text = "ghp_abcdefghijklmnopqrstuvwxyz0123"
    assert (await snippets.answer_async("save that as my github token")).startswith(
        "That looks like a password or key, so I won't save it"
    )
    assert await snippets.answer_async("read what I copied") == (
        "That looks like a password or key, so I won't read it aloud."
    )
    assert snippets.list() == []


@pytest.mark.asyncio
async def test_read_what_i_copied(snippets, board):
    assert await snippets.answer_async("read my clipboard") == "The clipboard is empty."
    board.text = "The meeting moved to Thursday."
    assert await snippets.answer_async("read what I copied aloud") == "You copied: The meeting moved to Thursday."
    board.text = "word " * 400
    assert (await snippets.answer_async("read the clipboard to me")).endswith("That's the first part; it goes on.")


@pytest.mark.asyncio
async def test_planner_and_setup_requests_are_left_alone(snippets, board):
    board.text = "x"
    for text in (
        "save this as my coding setup",
        "copy my files to the D drive",
        "copy the link",
        "paste",
        "save the document",
    ):
        assert await snippets.answer_async(text) is None, text


@pytest.mark.asyncio
async def test_change_detector_skips_unchanged_clipboard(board):
    board.text = "first"
    calls = {"changed": True}
    history = ClipboardHistory(
        enabled=lambda: True,
        read=AsyncMock(side_effect=board.read),
        changed=lambda: calls["changed"],
        foreground=lambda: ("", ""),
    )
    assert await history.check() is True
    calls["changed"] = False
    board.text = "second"
    assert await history.check() is False  # not even read
    assert history._read.await_count == 1


@pytest.mark.asyncio
async def test_snippet_rpcs(tmp_path, board):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._snippets = SnippetStore(tmp_path / "s.json", read=board.read, write=board.write)
    ws = MagicMock()

    saved = await server._handle_snippet_save({"name": "Address", "text": "12 Main Street"}, ws)
    assert saved["snippets"][0]["name"] == "address"
    assert (await server._handle_snippet_save({"name": "pw", "text": "sk-live1234567890abcdefghij"}, ws))[
        "status"
    ] == "error"
    assert (await server._handle_snippet_copy({"name": "address"}, ws))["status"] == "ok"
    assert board.written == ["12 Main Street"]
    assert (await server._handle_snippet_copy({"name": "missing"}, ws))["status"] == "error"
    assert (await server._handle_snippet_delete({"name": "address"}, ws))["snippets"] == []
    assert await server._control_reply("save this as my coding setup") != "The clipboard is empty."
