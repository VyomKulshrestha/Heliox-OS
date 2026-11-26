"""Post-conditions of the clipboard, a screenshot file and the workspace index (postconditions/local_state.py).

The clipboard is never touched: pyperclip is a stand-in and the read-back is a faked reader. Screenshots and the
index are files under pytest's temp folder; faiss is never imported.
"""

from __future__ import annotations

import json
import struct
import sys
import types
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ClipboardParams,
    ScreenshotParams,
    WorkspaceParams,
)
from pilot.agents.postconditions import local_state
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import clipboard
from pilot.system import desktop_observe as observe
from pilot.system.desktop_observe import DesktopStepMismatch, Unobservable

SECRET = "hunter2-very-secret-token"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    observe.forget_all()
    monkeypatch.setattr(observe, "read_clipboard_now", MagicMock(side_effect=AssertionError("real clipboard read")))
    yield
    observe.forget_all()


@pytest.fixture
def pyperclip(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake = types.SimpleNamespace(copy=MagicMock(), paste=MagicMock())
    monkeypatch.setitem(sys.modules, "pyperclip", fake)
    return fake


async def verify(action: Action, *, success: bool = True, output: str = ""):
    result = ActionResult(action=action, success=success, output=output, error=None if success else "boom")
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])


def label(verification) -> str:
    return verification.details[0].split(": ", 1)[1]


async def said(action: Action, *, output: str = "", **extra) -> str:
    """The verifier's own sentence (a VERIFIED result shows only its label)."""
    from pilot.agents.postconditions import registrations

    result = ActionResult(action=action, success=True, output=output, **extra)
    return (await registrations()[action.action_type].check(result)).detail


def test_registered() -> None:
    assert POSTCONDITION_VERIFIERS[ActionType.CLIPBOARD_WRITE] == "clipboard_text_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.SCREENSHOT] == "image_file_postcondition"
    assert POSTCONDITION_VERIFIERS[ActionType.WORKSPACE_INDEX] == "workspace_index_store_postcondition"


@pytest.mark.parametrize(
    "action",
    [
        Action(action_type=ActionType.CLIPBOARD_WRITE, parameters=ClipboardParams(content="x")),
        Action(action_type=ActionType.SCREENSHOT, parameters=ScreenshotParams(output_path="x.png")),
        Action(action_type=ActionType.WORKSPACE_INDEX, parameters=WorkspaceParams(folder_path="x")),
    ],
)
async def test_a_failed_handler_is_a_failed_action(action: Action) -> None:
    verification = await verify(action, success=False)
    assert not verification.passed and label(verification).startswith("FAILED")


# ── clipboard_write ──


async def _copy(monkeypatch: pytest.MonkeyPatch, content: str, read):
    action = Action(action_type=ActionType.CLIPBOARD_WRITE, parameters=ClipboardParams(content=content))
    reader = MagicMock(side_effect=read) if isinstance(read, Exception) else MagicMock(return_value=read)
    monkeypatch.setattr(observe, "read_clipboard_now", reader)
    with observe.step(action):
        output = await clipboard.clipboard_write(content)
    return action, output


async def test_copied_text_read_back_by_hash_is_verified(monkeypatch, pyperclip) -> None:
    action, output = await _copy(monkeypatch, SECRET, (SECRET, "Win32 GetClipboardData"))

    pyperclip.copy.assert_called_once_with(SECRET)
    verification = await verify(action, output=output)
    assert verification.passed and verification.independently_observed_actions == [0]
    assert "25 copied characters" in await said(action)
    assert SECRET not in " ".join(verification.details) and SECRET not in str(
        [(r.detail, r.facts) for r in observe._RECORDS.values()]
    )


async def test_line_endings_changed_by_the_platform_still_match(monkeypatch, pyperclip) -> None:
    action, _ = await _copy(monkeypatch, "a\nb", ("a\r\nb\r\n", "pbpaste"))

    assert (await verify(action)).passed and "once line endings are set aside" in await said(action)


async def test_a_clipboard_holding_other_text_stops_the_plan(monkeypatch, pyperclip) -> None:
    with pytest.raises(DesktopStepMismatch, match="nothing stale was pasted"):
        await _copy(monkeypatch, SECRET, ("older text", "xclip"))

    action = next(iter(observe._RECORDS.values())).action
    verification = await verify(action)
    assert not verification.passed and "10 characters that differ from the 25 copied" in verification.reasons[0]
    assert SECRET not in verification.reasons[0] and "older text" not in verification.reasons[0]


async def test_a_clipboard_that_cannot_be_read_is_reported(monkeypatch, pyperclip) -> None:
    action, _ = await _copy(monkeypatch, SECRET, Unobservable(observe.WAYLAND_CLIPBOARD))

    verification = await verify(action)
    assert verification.passed and verification.independently_observed_actions == []
    assert "Wayland" in verification.reasons[0]


async def test_a_copy_outside_a_step_reads_nothing(pyperclip) -> None:
    assert await clipboard.clipboard_write("x") == "Copied 1 characters to clipboard"
    observe.read_clipboard_now.assert_not_called()


async def test_without_a_reading_the_clipboard_is_read_now(monkeypatch: pytest.MonkeyPatch) -> None:
    action = Action(action_type=ActionType.CLIPBOARD_WRITE, parameters=ClipboardParams(content="abc"))
    monkeypatch.setattr(observe, "read_clipboard_now", MagicMock(return_value=("abc", "pbpaste")))
    assert "VERIFIED" in label(await verify(action))
    monkeypatch.setattr(observe, "read_clipboard_now", MagicMock(return_value=("zzz", "pbpaste")))
    later = await verify(action)
    assert later.passed and "may have copied since" in later.reasons[0]
    chained = Action(
        action_type=ActionType.CLIPBOARD_WRITE, parameters=ClipboardParams(content=""), use_previous_output=True
    )
    assert "previous step" in (await verify(chained)).reasons[0]


# ── screenshot ──


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _shot(path: Path, region: str | None = None) -> Action:
    return Action(action_type=ActionType.SCREENSHOT, parameters=ScreenshotParams(output_path=str(path), region=region))


@pytest.fixture
def screen(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(local_state, "primary_screen_size", lambda: (1920, 1080))


async def test_a_screenshot_that_decodes_with_the_screens_proportions_is_verified(tmp_path, screen) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(png(32, 18))

    verification = await verify(_shot(path), output=f"Screenshot saved to {path}")
    assert verification.passed and "32x18" in await said(_shot(path))


@pytest.mark.parametrize(
    ("content", "words"),
    [
        (None, "there is no file there"),
        (b"", "is empty"),
        (b"<html>not an image</html>", "does not decode as an image"),
        (png(32, 18)[:-20], "does not decode as an image"),
        (png(10, 10), "does not have the primary display's proportions"),
    ],
)
async def test_a_missing_broken_or_wrong_screenshot_is_a_mismatch(tmp_path, screen, content, words) -> None:
    path = tmp_path / "shot.png"
    if content is not None:
        path.write_bytes(content)

    verification = await verify(_shot(path))
    assert not verification.passed and words in verification.reasons[0]


async def test_a_region_or_unknown_screen_checks_only_that_it_decodes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(png(10, 10))
    monkeypatch.setattr(local_state, "primary_screen_size", lambda: None)
    assert (await verify(_shot(path))).passed
    monkeypatch.setattr(local_state, "primary_screen_size", lambda: (1920, 1080))
    assert (await verify(_shot(path, region="0,0,10,10"))).passed


async def test_the_default_path_is_taken_from_the_handlers_answer(tmp_path, screen) -> None:
    path = tmp_path / "default.png"
    path.write_bytes(png(16, 9))
    action = Action(action_type=ActionType.SCREENSHOT, parameters=ScreenshotParams())
    assert (await verify(action, output=f"Screenshot saved to {path}")).passed


def test_image_headers_without_pillow() -> None:
    assert local_state.image_size(png(4, 3)) == (4, 3)
    assert local_state.image_size(png(4, 3)[:-12]) is None
    bmp = b"BM" + b"\x00" * 16 + struct.pack("<ii", 5, -7) + b"\x00" * 30
    assert local_state.image_size(bmp) == (5, 7)
    jpeg = b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 6, 9) + b"\x00" * 12
    assert local_state.image_size(jpeg + b"\xff\xd9") == (9, 6)


# ── workspace_index ──


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / ".config" / "project"  # a workspace inside a dot-folder is indexed (it was not)
    (folder / "src").mkdir(parents=True)
    (folder / "node_modules").mkdir()
    (folder / "src" / "a.py").write_text("print(1)\n", encoding="utf-8")
    (folder / "notes.md").write_text("notes\n", encoding="utf-8")
    (folder / "node_modules" / "skip.js").write_text("x\n", encoding="utf-8")
    (folder / "image.png").write_bytes(b"\x89PNG")
    store = tmp_path / "index"
    store.mkdir()
    monkeypatch.setattr(local_state, "_workspace_index_dir", lambda: store)
    monkeypatch.setattr(local_state, "_faiss_count", lambda _path: 2)
    files = {str(folder / "src" / "a.py"): "h1", str(folder / "notes.md"): "h2"}
    (store / "file_hashes.json").write_text(json.dumps(files), encoding="utf-8")
    (store / "chunks.json").write_text(json.dumps([{"file": f} for f in files]), encoding="utf-8")
    (store / "workspace.faiss").write_bytes(b"IxFI")
    return folder, store


def _index(folder: Path) -> Action:
    return Action(action_type=ActionType.WORKSPACE_INDEX, parameters=WorkspaceParams(folder_path=str(folder)))


async def test_an_index_that_records_every_file_is_verified(workspace) -> None:
    folder, _store = workspace
    verification = await verify(_index(folder))
    assert verification.passed and "all 2 indexable files" in await said(_index(folder))


async def test_a_file_missing_from_the_record_or_a_short_vector_file_is_a_mismatch(workspace, monkeypatch) -> None:
    folder, store = workspace
    (folder / "src" / "b.py").write_text("print(2)\n", encoding="utf-8")
    missing = await verify(_index(folder))
    assert not missing.passed and "1 of the folder's 3 indexable files" in missing.reasons[0]

    (folder / "src" / "b.py").unlink()
    monkeypatch.setattr(local_state, "_faiss_count", lambda _path: 1)
    short = await verify(_index(folder))
    assert not short.passed and "vector file holds 1 entries, its chunk list 2" in short.reasons[0]

    (store / "workspace.faiss").unlink()
    gone = await verify(_index(folder))
    assert not gone.passed and "workspace.faiss" in gone.reasons[0]


async def test_without_faiss_the_index_is_reported(workspace, monkeypatch) -> None:
    folder, _store = workspace

    def no_faiss(_path):
        raise ImportError("faiss")

    monkeypatch.setattr(local_state, "_faiss_count", no_faiss)
    verification = await verify(_index(folder))
    assert verification.passed and "faiss is not installed" in verification.reasons[0]


def test_an_empty_folder_is_not_indexed_as_a_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.memory.workspace_index import WorkspaceIndex, skipped_folder

    monkeypatch.setitem(sys.modules, "faiss", types.ModuleType("faiss"))
    monkeypatch.setitem(sys.modules, "numpy", sys.modules.get("numpy") or types.ModuleType("numpy"))
    index = WorkspaceIndex(tmp_path / "store")
    monkeypatch.setattr(index, "_load_model", lambda: True)
    (tmp_path / "empty").mkdir()

    result = index.index_workspace(str(tmp_path / "empty"))

    assert result["success"] is False and "No supported text files" in result["error"]
    assert skipped_folder(Path("node_modules") / "x.js") and skipped_folder(Path(".git") / "x")
    assert not skipped_folder(Path("src") / "a.py")
