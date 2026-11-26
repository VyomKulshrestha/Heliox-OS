"""Post-conditions of local state the desktop keeps: the clipboard, a screenshot file, the workspace index.

The clipboard is read back right after the write through a different tool than the one that wrote it (Win32
GetClipboardData after pyperclip or Set-Clipboard, pbpaste after pbcopy, xclip or xsel), compared by hash, and only
lengths are said: copied text can be a password. The screenshot is read from the file system and decoded. The
workspace index is read from its own files on disk (the file record, the chunk list and the vector file), not from
the indexer's memory.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

ASPECT_TOLERANCE = 0.02
MAX_INDEX_WALK_FILES = 20_000


# ── clipboard_write ──


@register(ActionType.CLIPBOARD_WRITE, method="clipboard_text_postcondition", timeout=10.0)
async def clipboard_text(result: ActionResult) -> Outcome:
    """The clipboard held the copied text right after the step (by hash), and, where nothing has replaced it, still
    holds it. A later step or the person copying something else is not a failure of this one."""
    from pilot.system import desktop_observe as observe

    record = observe.recalled(result.action)
    if record is not None:
        if record.outcome == "match":
            return verified(record.detail + ", read right after the step")
        if record.outcome == "mismatch":
            return mismatch(record.detail)
        return unobserved(f"Could not verify the clipboard: {record.detail}")
    if result.action.use_previous_output:
        return unobserved("The copied text came from the previous step and no reading was made right after the copy")
    expected = getattr(result.action.parameters, "content", None) or ""
    try:
        found, tool = await observe.clipboard_text()
    except observe.Unobservable as reason:
        return unobserved(f"Could not verify the clipboard: {reason}")
    if observe.clipboard_holds(found, expected) is not None:
        return verified(f"The clipboard holds the {len(expected)} copied characters ({tool})")
    return unobserved(
        f"No reading was made right after the copy; the clipboard now holds {len(found)} other characters, and a "
        "later step or the person may have copied since"
    )


# ── screenshot ──


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a PNG, BMP or JPEG from its bytes, or None when they are not a whole image of those kinds.
    A PNG must end with its IEND chunk, so a file cut short does not pass. Pure."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33 and data[12:16] == b"IHDR":
        if b"IEND" not in data[-16:]:
            return None
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width and height else None
    if data.startswith(b"BM") and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return (abs(width), abs(height)) if width and height else None
    if data.startswith(b"\xff\xd8") and data.rstrip(b"\x00").endswith(b"\xff\xd9"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                return None
            marker = data[index + 1]
            length = struct.unpack(">H", data[index + 2 : index + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):
                height, width = struct.unpack(">HH", data[index + 5 : index + 9])
                return (width, height) if width and height else None
            index += 2 + length
    return None


def decoded_size(path: Path) -> tuple[int, int] | None:
    """Decode the image fully with Pillow when it is installed; else read its header and end (``image_size``). A PNG
    must also end with its IEND chunk: Pillow decodes a file cut short without complaint when its pixels fit."""
    data = path.read_bytes()
    if data.startswith(b"\x89PNG") and image_size(data) is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return image_size(data)
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except Exception:
        return None


def primary_screen_size() -> tuple[int, int] | None:
    """The primary display's size in pixels, where it can be read without a prompt; None elsewhere."""
    from pilot.system.platform_detect import CURRENT_PLATFORM, Platform

    try:
        if CURRENT_PLATFORM == Platform.WINDOWS:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if CURRENT_PLATFORM == Platform.MACOS:
            import Quartz

            display = Quartz.CGMainDisplayID()
            return int(Quartz.CGDisplayPixelsWide(display)), int(Quartz.CGDisplayPixelsHigh(display))
    except Exception:
        return None
    return None


def _screenshot_path(result: ActionResult) -> str:
    path = getattr(result.action.parameters, "output_path", None)
    if path:
        return str(path)
    match = re.search(r"Screenshot saved to (.+)$", str(result.output or "").strip())
    return match.group(1).strip() if match else ""


@register(ActionType.SCREENSHOT, method="image_file_postcondition", timeout=15.0)
async def screenshot_file(result: ActionResult) -> Outcome:
    """The image file exists, decodes as an image, and, for a whole-screen capture on Windows and macOS, has the
    proportions of the primary display (its pixel count depends on display scaling, so it is not compared)."""
    import asyncio

    path_text = _screenshot_path(result)
    if not path_text:
        raise ValueError("The screenshot has no file path to observe")
    path = Path(path_text)
    if not path.is_file():
        return mismatch(f"Expected a screenshot at {path}; there is no file there")
    if path.stat().st_size == 0:
        return mismatch(f"The screenshot at {path} is empty (0 bytes)")
    size = await asyncio.to_thread(decoded_size, path)
    if size is None:
        return mismatch(f"The file at {path} does not decode as an image")
    width, height = size
    region = getattr(result.action.parameters, "region", None)
    if region in (None, "", "fullscreen"):
        screen = await asyncio.to_thread(primary_screen_size)
        if screen and screen[0] and screen[1]:
            expected = screen[0] / screen[1]
            if abs(width / height - expected) > ASPECT_TOLERANCE * expected:
                return mismatch(
                    f"The screenshot is {width}x{height}, which does not have the primary display's proportions "
                    f"({screen[0]}x{screen[1]})"
                )
            return verified(f"{path.name} decodes as a {width}x{height} image with the primary display's proportions")
    return verified(f"{path.name} decodes as a {width}x{height} image")


# ── workspace_index ──


def _workspace_index_dir() -> Path:
    from pilot.config import DATA_DIR

    return Path(DATA_DIR) / "workspace_index"


def _faiss_count(path: Path) -> int:
    import faiss

    return int(faiss.read_index(str(path)).ntotal)


def eligible_files(folder: Path, limit: int = MAX_INDEX_WALK_FILES) -> list[Path] | None:
    """The files the indexer's own rules take (extension, size, no hidden or dependency folder); None past ``limit``."""
    from pilot.memory.workspace_index import MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS, skipped_folder

    found: list[Path] = []
    seen = 0
    for path in folder.rglob("*"):
        seen += 1
        if seen > limit:
            return None
        if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if skipped_folder(path.relative_to(folder)):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue
        found.append(path)
    return found


@register(ActionType.WORKSPACE_INDEX, method="workspace_index_store_postcondition", timeout=30.0)
async def workspace_index_store(result: ActionResult) -> Outcome:
    """The index's own record on disk names every file of the folder the indexer's rules take, and its vector file
    holds one entry per stored chunk."""
    import asyncio

    folder_text = str(getattr(result.action.parameters, "folder_path", "") or "")
    if not folder_text:
        raise ValueError("workspace_index has no folder to observe")
    index_dir = _workspace_index_dir()
    hashes_file, chunks_file, vectors = (
        index_dir / "file_hashes.json",
        index_dir / "chunks.json",
        index_dir / "workspace.faiss",
    )
    missing_store = [path.name for path in (hashes_file, chunks_file, vectors) if not path.is_file()]
    if missing_store:
        return mismatch(f"The workspace index has no {', '.join(missing_store)} in its folder")
    hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    try:
        stored = await asyncio.to_thread(_faiss_count, vectors)
    except ImportError:
        return unobserved("Could not verify the index: faiss is not installed to read its vector file")
    if stored != len(chunks):
        return mismatch(f"The index's vector file holds {stored} entries, its chunk list {len(chunks)}")
    files = await asyncio.to_thread(eligible_files, Path(folder_text))
    if files is None:
        return unobserved(f"Could not verify the index: the folder has more than {MAX_INDEX_WALK_FILES} entries")
    absent = [path for path in files if str(path) not in hashes]
    if absent:
        return mismatch(
            f"{len(absent)} of the folder's {len(files)} indexable files are not in the index's record "
            f"(for example {absent[0].name})"
        )
    return verified(f"The index records all {len(files)} indexable files of the folder; {len(chunks)} chunks stored")
