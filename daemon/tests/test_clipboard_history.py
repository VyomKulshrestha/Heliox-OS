"""Clipboard history: off by default, memory only, never secrets or private windows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.workflows.clipboard_history import KEEP_SECONDS, MAX_ITEMS, ClipboardHistory, looks_secret
from pilot.workflows.privacy import STORES, clear_store, inventory
from pilot.workflows.task_receipts import TaskReceipts
from tests.settings_approval import approve_setting_change


class Clipboard:
    def __init__(self):
        self.text = "(clipboard is empty)"
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
def state():
    return {"on": True, "now": 1000.0, "front": ("notepad", "notes.txt")}


@pytest.fixture
def history(board, state):
    return ClipboardHistory(
        enabled=lambda: state["on"],
        excluded=lambda app, title: "bitwarden" in app.lower() or "inprivate" in title.lower(),
        read=board.read,
        write=board.write,
        foreground=lambda: state["front"],
        clock=lambda: state["now"],
    )


@pytest.mark.parametrize(
    "text",
    [
        "sk-live1234567890abcdefghij",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "AKIAABCDEFGHIJKLMNOP",
        "password: hunter2",
        "-----BEGIN RSA PRIVATE KEY-----\nabc",
        "4111 1111 1111 1111",
        "Tr0ub4dor&3xK!",
        "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4fw",
    ],
)
def test_secrets_are_recognized(text):
    assert looks_secret(text)


@pytest.mark.parametrize(
    "text",
    [
        "Meeting moved to Thursday at 3",
        "https://example.com/reports?id=42&Page=2",
        "jane.doe@example.com",
        r"C:\Users\Jane\Documents\Report-2026.docx",
        "hello",
        "1234 5678",
        "Quarterly",
        "ABC123",
        "aaaaaaaaaaaa",
        "Project Phoenix kickoff",
    ],
)
def test_ordinary_text_is_kept(text):
    assert not looks_secret(text)


@pytest.mark.asyncio
async def test_off_by_default_and_forgets_when_turned_off(board, history, state):
    config = PilotConfig()
    assert config.clipboard.history is False
    assert _merge_config(PilotConfig(), {"clipboard": {"history": True}}).clipboard.history is True

    board.text = "first"
    assert await history.check() is True
    state["on"] = False
    board.text = "second"
    assert await history.check() is False
    assert history.list() == []
    assert await history.answer_async("what did I copy earlier") == (
        'Clipboard history is off. Say "turn on clipboard history" or turn it on in Settings.'
    )


@pytest.mark.asyncio
async def test_keeps_text_newest_first_without_duplicates(board, history, state):
    for text in ("alpha", "https://example.com/a", "alpha", "beta"):
        board.text = text
        state["now"] += 5
        await history.check()
    assert [item["text"] for item in history.list()] == ["beta", "alpha", "https://example.com/a"]
    assert await history.answer_async("What did I copy earlier?") == (
        "You copied: 1. beta; 2. alpha; 3. https://example.com/a."
    )


@pytest.mark.asyncio
async def test_skips_secrets_private_windows_and_huge_text(board, history, state):
    board.text = "ghp_abcdefghijklmnopqrstuvwxyz0123"
    assert await history.check() is False
    state["front"] = ("Bitwarden", "Vault")
    board.text = "my shopping list"
    assert await history.check() is False
    state["front"] = ("msedge", "News - InPrivate")
    board.text = "private browsing text"
    assert await history.check() is False
    state["front"] = ("notepad", "x")
    board.text = "x" * 10_001
    assert await history.check() is False
    assert history.list() == []


@pytest.mark.asyncio
async def test_text_heliox_put_on_the_clipboard_is_never_kept(board, history, state):
    generated = "lp*Fccc*lZU5"  # a generated password; looks_secret said False for it (see test_day_helpers)
    history.never_keep(generated)
    board.text = generated
    assert await history.check() is False
    assert history.list() == []
    assert generated not in repr(history._never_keep) and generated not in repr(history._items)  # only a hash
    state["now"] += KEEP_SECONDS + 1  # the entry expires
    assert history.remember(generated) is True


@pytest.mark.asyncio
async def test_copy_back_by_order_and_kind(board, history, state):
    for text in ("https://example.com/report", "jane@example.com", "plain words"):
        board.text = text
        state["now"] += 1
        await history.check()

    assert await history.answer_async("copy the second one back") == (
        'Copied "jane@example.com" again. It\'s ready to paste.'
    )
    assert board.written == ["jane@example.com"]
    assert await history.check() is False  # our own write is not a new copy
    assert await history.answer_async("copy back the link") == (
        'Copied "https://example.com/report" again. It\'s ready to paste.'
    )
    assert history.list()[0]["text"] == "https://example.com/report"
    assert await history.answer_async("copy the fifth one back") == "I can't find that in what you copied."


@pytest.mark.asyncio
async def test_planner_requests_are_left_alone(history):
    for text in (
        "copy the link",
        "copy this file to my desktop",
        "copy the second paragraph",
        "what's on my clipboard",
    ):
        assert await history.answer_async(text) is None, text


@pytest.mark.asyncio
async def test_limits_expiry_and_clearing(board, history, state):
    for number in range(MAX_ITEMS + 5):
        board.text = f"item {number}"
        await history.check()
    assert len(history.list()) == MAX_ITEMS

    state["now"] += KEEP_SECONDS + 1
    assert history.list() == []

    board.text = "fresh"
    await history.check()
    assert await history.answer_async("clear my clipboard history") == "Forgot 1 copied item."
    assert await history.answer_async("forget what I copied") == "There was nothing to forget."


def test_memory_only_store_is_listed_but_never_touches_disk(tmp_path):
    (tmp_path / "notes.json").write_text("[]", encoding="utf-8")
    rows = {row["key"]: row for row in inventory(tmp_path, {"clipboard": 3})}

    assert rows["clipboard"]["memory_only"] is True
    assert rows["clipboard"]["bytes"] == 0 and rows["clipboard"]["items"] == 3
    assert "clipboard" in STORES
    clear_store(tmp_path, "clipboard")
    assert (tmp_path / "notes.json").exists()  # clearing the memory-only store deletes nothing on disk


@pytest.mark.asyncio
async def test_rpcs_voice_setting_and_your_data(tmp_path, board):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._task_receipts = TaskReceipts(tmp_path / "receipts", key_dir=tmp_path / "key")
    server._clipboard_history = ClipboardHistory(
        enabled=lambda: server.config.clipboard.history, read=board.read, write=board.write, foreground=lambda: ("", "")
    )
    ws = MagicMock()

    # Turning it on is a capture: the first time it is asked about in the window, and voice cannot approve it.
    assert await server._control_reply("turn on clipboard history") == "Open Heliox and approve it there."
    assert server.config.clipboard.history is False
    await approve_setting_change(server, "clipboard", {"history": True})  # the person's yes in the dialog
    assert await server._control_reply("turn on clipboard history") == (
        "Clipboard history is on. I'll remember what you copy for a few hours, in memory only."
    )
    assert server.config.clipboard.history is True
    board.text = "remember me"
    await server._clipboard_history.check()

    listed = await server._handle_clipboard_history_list({}, ws)
    assert listed["enabled"] is True and listed["items"][0]["text"] == "remember me"
    copied = await server._handle_clipboard_history_copy({"id": listed["items"][0]["id"]}, ws)
    assert copied["status"] == "ok" and board.written == ["remember me"]
    assert (await server._handle_clipboard_history_copy({"id": "missing"}, ws))["status"] == "error"

    rows = {row["key"]: row for row in (await server._handle_privacy_inventory({}, ws))["stores"]}
    assert rows["clipboard"]["items"] == 1

    async def approve(plan_id, plan, ws, **kwargs):  # the person's yes in Heliox's dialog
        return True, {0}, {0}

    server._wait_for_confirmation = approve
    await server._handle_privacy_clear({"key": "clipboard"}, ws)
    assert server._clipboard_history.list() == []

    assert (await server._handle_update_config({"section": "clipboard", "values": {"history": "yes"}}, ws))[
        "status"
    ] == "error"
    assert await server._control_reply("turn off clipboard history") == (
        "Clipboard history is off, and I forgot what you copied."
    )
    assert (await server._handle_clipboard_history_clear({}, ws))["removed"] == 0
