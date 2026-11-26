"""Post-conditions for files written, copied and downloaded, read from the file system afresh.

What a handler recorded in its output (the sha256 and length of the bytes it wrote, the server's Content-Length)
is used only to know what to look for. The evidence is always the file read back from disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from pathlib import Path

from pilot.actions import ActionResult, ActionType, DownloadParams, FileParams
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

CHUNK_BYTES = 1024 * 1024
# A copy is compared in full however large it is, within this budget; past it the copy is "not verified".
COPY_COMPARE_SECONDS = 240.0
_clock = time.monotonic

_WRITTEN = re.compile(r"\(sha256 (?P<sha>[0-9a-f]{64}), (?P<size>\d+) bytes on disk\)")
_DOWNLOAD_FACTS = re.compile(r"^\[DOWNLOAD\] (?P<json>\{.*\})$", re.MULTILINE)

# A body that is an HTML page is wrong when the file asked for is plainly something else.
_NOT_HTML_SUFFIXES = frozenset(
    [
        "7z",
        "apk",
        "appimage",
        "avi",
        "bin",
        "bmp",
        "bz2",
        "cab",
        "csv",
        "deb",
        "dll",
        "dmg",
        "doc",
        "docx",
        "epub",
        "exe",
        "flac",
        "gif",
        "gz",
        "ico",
        "iso",
        "jar",
        "jpeg",
        "jpg",
        "json",
        "m4a",
        "mkv",
        "mov",
        "mp3",
        "mp4",
        "msi",
        "msix",
        "nupkg",
        "odt",
        "ogg",
        "otf",
        "parquet",
        "pdf",
        "pkg",
        "png",
        "ppt",
        "pptx",
        "rar",
        "rpm",
        "sqlite",
        "svgz",
        "tar",
        "tbz2",
        "tgz",
        "tif",
        "tiff",
        "ttf",
        "wav",
        "webm",
        "webp",
        "whl",
        "woff",
        "woff2",
        "xls",
        "xlsx",
        "xz",
        "zip",
        "zst",
    ]
)


# ── helpers ──────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_text_bytes(content: str) -> bytes:
    """The bytes ``file_write`` puts on disk for ``content`` (text mode: newline becomes the platform's)."""
    return content.replace("\n", os.linesep).encode("utf-8")


def _short(sha: str) -> str:
    return sha[:12]


# ── file_write ───────────────────────────────────────────────────────


@register(ActionType.FILE_WRITE, method="file_content_postcondition", precheck=True, timeout=60.0)
async def file_has_content(result: ActionResult) -> Outcome:
    """The file holds the bytes the handler wrote (its recorded sha256), else the plan's text.

    The handler records the sha256 and length of what it wrote after placeholders and earlier output were
    substituted, so a correct write of substituted content verifies. Before the action runs (the pre-check)
    nothing is recorded and the plan's text is all that can be known.
    """
    params: FileParams = result.action.parameters  # type: ignore[assignment]
    if not params.path:
        raise ValueError("File write has no path to observe")
    p = Path(params.path)
    if not p.is_file():
        return mismatch(f"File does not exist after write: {params.path}")
    size = p.stat().st_size
    written = _WRITTEN.search(result.output or "")
    if written:
        expected_size = int(written["size"])
        if size != expected_size:
            return mismatch(f"File holds {size:,} bytes; the write put {expected_size:,} bytes there")
        found = await asyncio.to_thread(_sha256, p)
        if found != written["sha"]:
            return mismatch(
                f"File content differs from what was written (sha256 {_short(found)}, written {_short(written['sha'])})"
            )
        return verified(f"File holds the {size:,} bytes that were written")
    if params.content is not None:
        expected = _expected_text_bytes(params.content)
        data = await asyncio.to_thread(p.read_bytes)
        if data == expected or data == params.content.encode("utf-8"):
            return verified("File exists with expected content")
        return mismatch(f"File content mismatch (expected {len(expected)} bytes, got {len(data)})")
    return unobserved("File exists; no content was given or recorded to compare it with")


# ── file_copy ────────────────────────────────────────────────────────


def _files_equal(source: Path, destination: Path, deadline: float) -> bool | None:
    """Byte for byte, in chunks; None when the time budget ran out first."""
    with source.open("rb") as original, destination.open("rb") as copied:
        for chunk in iter(lambda: original.read(CHUNK_BYTES), b""):
            if _clock() > deadline:
                return None
            if chunk != copied.read(len(chunk)):
                return False
        return not copied.read(1)


def _tree(root: Path, deadline: float) -> tuple[dict[str, int], set[str]] | None:
    """Files (relative path to size) and folders under ``root``; None when the time budget ran out.

    Symbolic links are not followed into folders (a link loop cannot trap the walk); a linked file counts with
    the size of what it points to, as ``shutil.copytree`` copies it.
    """
    files: dict[str, int] = {}
    folders: set[str] = set()
    for dirpath, dirnames, names in os.walk(root):
        if _clock() > deadline:
            return None
        here = Path(dirpath)
        for name in dirnames:
            folders.add((here / name).relative_to(root).as_posix())
        for name in names:
            path = here / name
            try:
                files[path.relative_to(root).as_posix()] = path.stat().st_size
            except OSError:
                continue
    return files, folders


@register(ActionType.FILE_COPY, method="copy_destination_postcondition", timeout=COPY_COMPARE_SECONDS + 60.0)
async def copy_matches_source(result: ActionResult) -> Outcome:
    """The destination holds what the source holds: every byte for a file; folders, names and sizes for a tree.

    The destination may hold more than the source (a linked folder in the source is copied as a real one); it
    may not hold less. Past the time budget the copy is not verified, never passed.
    """
    params: FileParams = result.action.parameters  # type: ignore[assignment]
    if not params.path or not params.destination:
        raise ValueError("File copy has no source or destination to observe")
    source = Path(params.path)
    destination = Path(params.destination)
    if not source.exists():
        return mismatch(f"Source does not exist after copy: {params.path}")
    if not destination.exists():
        return mismatch(f"Destination does not exist after copy: {params.destination}")
    deadline = _clock() + COPY_COMPARE_SECONDS
    if source.is_file() and destination.is_file():
        source_size = source.stat().st_size
        destination_size = destination.stat().st_size
        if source_size != destination_size:
            return mismatch("Copy destination size mismatch")
        same = await asyncio.to_thread(_files_equal, source, destination, deadline)
        if same is None:
            return unobserved(
                f"Copy has the source's size ({source_size:,} bytes); comparing its content did not finish "
                f"within {COPY_COMPARE_SECONDS:g} s"
            )
        if same:
            return verified(f"Copy content matches source ({source_size:,} bytes)")
        return mismatch("Copy destination content mismatch")
    if source.is_dir() and destination.is_dir():
        source_tree, destination_tree = await asyncio.gather(
            asyncio.to_thread(_tree, source, deadline), asyncio.to_thread(_tree, destination, deadline)
        )
        if source_tree is None or destination_tree is None:
            return unobserved(
                f"Copy destination exists; listing the folders did not finish within {COPY_COMPARE_SECONDS:g} s"
            )
        source_files, source_folders = source_tree
        destination_files, destination_folders = destination_tree
        missing = sorted(set(source_files) - set(destination_files))
        if missing:
            return mismatch(f"Copy is missing {len(missing)} file(s), for example {missing[0]}")
        missing_folders = sorted(source_folders - destination_folders - set(destination_files))
        if missing_folders:
            return mismatch(f"Copy is missing {len(missing_folders)} folder(s), for example {missing_folders[0]}")
        different = sorted(name for name, size in source_files.items() if destination_files[name] != size)
        if different:
            return mismatch(f"Copied file size differs for {len(different)} file(s), for example {different[0]}")
        if not source_files:
            return verified(f"Copy has all {len(source_folders)} folder(s) of the source, which holds no files")
        total = sum(source_files.values())
        return verified(f"Copy has all {len(source_files)} files of the source ({total} bytes), sizes match")
    return mismatch("Copy destination type does not match source")


# ── download_file ────────────────────────────────────────────────────


@register(ActionType.DOWNLOAD_FILE, method="download_file_exists_postcondition")
async def downloaded_file_present(result: ActionResult) -> Outcome:
    """The downloaded file exists at the output path."""
    params: DownloadParams = result.action.parameters  # type: ignore[assignment]
    if not params.output_path:
        raise ValueError("Download has no output path to observe")
    if Path(params.output_path).exists():
        size = Path(params.output_path).stat().st_size
        return verified(f"File downloaded: {params.output_path} ({size:,} bytes)")
    return mismatch(f"Downloaded file not found: {params.output_path}")
