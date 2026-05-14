"""File system operations — read, write, delete, move, copy, list, search, permissions.

All paths are validated by the security layer before reaching these functions.
Cross-platform with Windows-aware path handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform

logger = logging.getLogger("pilot.system.filesystem")


async def file_read(path: str) -> str:
    """Read file contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    return await asyncio.to_thread(p.read_text, "utf-8")


async def file_write(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(p.write_text, content, "utf-8")
    return f"Written {len(content)} bytes to {path}"


async def file_delete(path: str, recursive: bool = False) -> str:
    """Delete a file or directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if p.is_dir():
        if not recursive:
            raise ValueError(f"Cannot delete directory without recursive=true: {path}")
        await asyncio.to_thread(shutil.rmtree, p)
        return f"Deleted directory: {path}"
    await asyncio.to_thread(p.unlink)
    return f"Deleted file: {path}"


async def file_move(source: str, destination: str) -> str:
    """Move/rename a file or directory."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    dst = Path(destination)
    await asyncio.to_thread(dst.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, str(src), str(dst))
    return f"Moved {source} -> {destination}"


async def file_copy(source: str, destination: str, recursive: bool = False) -> str:
    """Copy a file or directory."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    dst = Path(destination)
    await asyncio.to_thread(dst.parent.mkdir, parents=True, exist_ok=True)
    if src.is_dir():
        if not recursive:
            raise ValueError(f"Cannot copy directory without recursive=true: {source}")
        await asyncio.to_thread(shutil.copytree, str(src), str(dst))
    else:
        await asyncio.to_thread(shutil.copy2, str(src), str(dst))
    return f"Copied {source} -> {destination}"


async def file_list(path: str, recursive: bool = False) -> str:
    """List directory contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")

    def _list() -> list[str]:
        entries = []
        if recursive:
            for item in sorted(p.rglob("*")):
                try:
                    rel = item.relative_to(p)
                    kind = "d" if item.is_dir() else "f"
                    size = item.stat().st_size if item.is_file() else 0
                    entries.append(f"[{kind}] {rel}  ({size} bytes)")
                except PermissionError:
                    entries.append(f"[?] {item.relative_to(p)}  (access denied)")
        else:
            for item in sorted(p.iterdir()):
                try:
                    kind = "d" if item.is_dir() else "f"
                    size = item.stat().st_size if item.is_file() else 0
                    entries.append(f"[{kind}] {item.name}  ({size} bytes)")
                except PermissionError:
                    entries.append(f"[?] {item.name}  (access denied)")
        return entries[:500]  # Cap output

    items = await asyncio.to_thread(_list)
    return "\n".join(items) if items else "(empty directory)"


async def file_search(path: str, pattern: str) -> str:
    """Search for files matching a glob pattern in a directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")

    def _search() -> list[str]:
        matches = []
        for item in sorted(p.rglob(pattern)):
            try:
                kind = "d" if item.is_dir() else "f"
                size = item.stat().st_size if item.is_file() else 0
                matches.append(f"[{kind}] {item.relative_to(p)}  ({size} bytes)")
            except PermissionError:
                continue
        return matches[:200]

    items = await asyncio.to_thread(_search)
    if not items:
        return f"No files matching '{pattern}' in {path}"
    return f"Found {len(items)} matches:\n" + "\n".join(items)


def _format_size(size: int) -> str:
    """Return a compact human-readable file size."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


async def directory_summary(
    path: str,
    max_depth: int = 3,
    max_entries: int = 200,
    ignore_dirs: list[str] | None = None,
) -> str:
    """Return a token-efficient directory tree with file sizes."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")

    ignored = set([".git", "node_modules"] if ignore_dirs is None else ignore_dirs)
    depth_limit = max(0, max_depth)
    entry_limit = max(1, max_entries)

    def _summarize() -> list[str]:
        lines = [f"{p.name or str(p)}/"]
        emitted = 0
        omitted = 0

        def walk(directory: Path, depth: int, prefix: str) -> None:
            nonlocal emitted, omitted
            if depth >= depth_limit or emitted >= entry_limit:
                return

            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except PermissionError:
                lines.append(f"{prefix}[access denied]")
                return

            visible = [item for item in children if not (item.is_dir() and item.name in ignored)]
            for index, item in enumerate(visible):
                if emitted >= entry_limit:
                    omitted += len(visible) - index
                    break

                connector = "`-- " if index == len(visible) - 1 else "|-- "
                child_prefix = "    " if index == len(visible) - 1 else "|   "

                try:
                    if item.is_dir():
                        lines.append(f"{prefix}{connector}{item.name}/")
                        emitted += 1
                        walk(item, depth + 1, prefix + child_prefix)
                    else:
                        size = _format_size(item.stat().st_size)
                        lines.append(f"{prefix}{connector}{item.name} ({size})")
                        emitted += 1
                except PermissionError:
                    lines.append(f"{prefix}{connector}{item.name} [access denied]")
                    emitted += 1

        walk(p, 0, "")
        if omitted:
            lines.append(f"... {omitted} more entr{'y' if omitted == 1 else 'ies'} omitted")
        return lines

    return "\n".join(await asyncio.to_thread(_summarize))


async def file_permissions(path: str, permissions: str | None = None) -> str:
    """Get or set file permissions. On Windows, this is limited.

    permissions: octal string like "755", "644", etc.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if permissions is None:
        # Read permissions
        st = p.stat()
        mode = oct(st.st_mode)[-3:]
        return f"Permissions for {path}: {mode}"

    if CURRENT_PLATFORM == Platform.WINDOWS:
        # Windows doesn't use Unix-style permissions
        from pilot.system.platform_detect import run_command

        code, out, err = await run_command(["icacls", str(p)])
        return f"Windows permissions for {path}:\n{out.strip()}"

    # Unix: set permissions
    mode = int(permissions, 8)
    await asyncio.to_thread(os.chmod, str(p), mode)
    return f"Set permissions on {path} to {permissions}"
