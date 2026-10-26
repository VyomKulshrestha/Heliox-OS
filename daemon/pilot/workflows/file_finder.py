"""Find your files: "what did I download today?", "where is my budget spreadsheet?".

Downloads questions list what arrived in the Downloads folder in a stretch of time.
"Where is my …" looks through your own folders (Desktop, Documents, Downloads, Pictures,
Music, Videos and OneDrive) a few levels deep, for at most a few seconds, and names the
newest matches. Code folders (git repositories, build output, caches) are not searched
below their top level. Words like "spreadsheet", "PDF" or "photo" narrow the search to those file
types. Heliox only reads folder listings and file dates; it opens nothing. When nothing
is found and the question did not name a kind of file, the question is left to the
planner, which can search more widely.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SEARCH_SECONDS = 3.0
MAX_ENTRIES = 40_000
MAX_DEPTH = 4
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "appdata", ".venv", "venv", "env", "$recycle.bin", "site-packages",
    "target", "dist", "build", ".cache", "cache", "vendor", "obj", "bin",
}  # fmt: skip
PARTIAL_SUFFIXES = (".crdownload", ".part", ".partial", ".download", ".opdownload", ".tmp")

KINDS: dict[str, tuple[str, ...]] = {
    "spreadsheet": (".xlsx", ".xls", ".xlsm", ".csv", ".ods", ".numbers"),
    "document": (".docx", ".doc", ".pdf", ".odt", ".rtf", ".txt", ".pages", ".md"),
    "pdf": (".pdf",),
    "presentation": (".pptx", ".ppt", ".key", ".odp"),
    "photo": (".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp", ".bmp"),
    "video": (".mp4", ".mov", ".mkv", ".avi", ".webm"),
    "song": (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"),
    "archive": (".zip", ".rar", ".7z", ".tar", ".gz"),
    "installer": (".exe", ".msi", ".dmg", ".pkg", ".deb", ".appimage"),
}
KIND_WORDS = {
    "spreadsheet": "spreadsheet", "spreadsheets": "spreadsheet", "excel": "spreadsheet", "sheet": "spreadsheet",
    "csv": "spreadsheet", "document": "document", "documents": "document", "doc": "document", "word": "document",
    "pdf": "pdf", "pdfs": "pdf", "presentation": "presentation", "slides": "presentation", "deck": "presentation",
    "powerpoint": "presentation", "photo": "photo", "photos": "photo", "picture": "photo", "pictures": "photo",
    "image": "photo", "images": "photo", "screenshot": "photo", "video": "video", "videos": "video",
    "song": "song", "music": "song", "recording": "song", "zip": "archive", "archive": "archive",
    "installer": "installer", "setup file": "installer",
}  # fmt: skip
GENERIC = {"file", "files", "my", "the", "a", "an", "called", "named", "for", "about", "with", "of", "from"}

DOWNLOADS_REQUEST = re.compile(
    r"^what (?:did i|have i) (?:just )?download(?:ed)?(?: (?P<when>today|yesterday|this week|recently|lately|earlier))?$"
    r"|^what s (?:in|new in) my downloads(?: folder)?$"
)
LAST_DOWNLOAD_REQUEST = re.compile(
    r"^what (?:was|is|s) the (?:last|latest|newest) (?:file|thing) i downloaded$|^what did i just download$"
)
FIND_REQUEST = re.compile(
    r"^(?:where s|where is|where are|find|locate|where did i (?:save|put)) (?:my |the )?(?P<what>.+?)"
    r"(?: on my (?:computer|pc|laptop))?$"
)


def names_a_kind(phrase: str) -> bool:
    """Whether "where is my …" names a kind of file, which makes it a file question for certain."""
    asked = FIND_REQUEST.match(phrase)
    return bool(asked) and any(re.search(rf"\b{re.escape(word)}\b", asked.group("what")) for word in KIND_WORDS)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.]", " ", text.casefold()).split()).strip(" .")


def _size(size: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _when(timestamp: float, now: datetime) -> str:
    moment = datetime.fromtimestamp(timestamp)
    clock = moment.strftime("%I:%M %p").lstrip("0")
    if moment.date() == now.date():
        return f"today at {clock}"
    if moment.date() == (now - timedelta(days=1)).date():
        return f"yesterday at {clock}"
    return moment.strftime("%a %d %b")


def default_roots() -> list[Path]:
    home = Path.home()
    roots = [home / name for name in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos")]
    roots += sorted(path for path in home.glob("OneDrive*") if path.is_dir())
    return [root for root in roots if root.is_dir()]


class FileFinder:
    """Answers questions about your files from folder listings."""

    def __init__(
        self,
        *,
        roots: Callable[[], list[Path]] = default_roots,
        downloads: Callable[[], Path] = lambda: Path.home() / "Downloads",
        now: Callable[[], datetime] = datetime.now,
        budget_seconds: float = SEARCH_SECONDS,
    ) -> None:
        self._roots = roots
        self._downloads = downloads
        self._now = now
        self._budget = budget_seconds

    def downloads(self, since: float) -> list[Path]:
        folder = self._downloads()
        try:
            files = [
                path
                for path in folder.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.casefold() not in PARTIAL_SUFFIXES
                and path.stat().st_mtime >= since
            ]
        except OSError:
            return []
        return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)

    def search(self, words: list[str], suffixes: tuple[str, ...] = ()) -> tuple[list[Path], bool]:
        """Files whose name has every word (and one of ``suffixes``), newest first; and whether the search finished."""
        deadline = time.monotonic() + self._budget
        found: list[Path] = []
        seen = 0
        complete = True
        for root in self._roots():
            stack = [(root, 0)]
            while stack:
                folder, depth = stack.pop()
                if time.monotonic() > deadline:
                    complete = False
                    break
                try:
                    entries = list(os.scandir(folder))
                except OSError:
                    continue
                in_repo = depth > 0 and any(entry.name == ".git" for entry in entries)
                for entry in entries:
                    seen += 1
                    if seen > MAX_ENTRIES or time.monotonic() > deadline:
                        complete = False
                        break
                    name = entry.name
                    if name.startswith(".") or name.casefold() in SKIP_DIRS:
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth < MAX_DEPTH and not in_repo:
                                stack.append((Path(entry.path), depth + 1))
                            continue
                    except OSError:
                        continue
                    if suffixes and Path(name).suffix.casefold() not in suffixes:
                        continue
                    stem_words = re.findall(r"[a-z0-9]+", Path(name).stem.casefold())
                    if all(any(part.startswith(word) for part in stem_words) for word in words):
                        found.append(Path(entry.path))
                if not complete:
                    break
            if not complete:
                break

        def modified(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        return sorted(found, key=modified, reverse=True), complete

    def answer(self, text: str) -> str | None:
        normalized = _normalize(text)
        now = self._now()
        if LAST_DOWNLOAD_REQUEST.match(normalized):
            files = self.downloads(0)
            if not files:
                return "Your Downloads folder is empty."
            newest = files[0]
            stat = newest.stat()
            return (
                f"The last thing you downloaded was {newest.name} ({_size(stat.st_size)}), {_when(stat.st_mtime, now)}."
            )
        asked = DOWNLOADS_REQUEST.match(normalized)
        if asked:
            when = asked.group("when") or "recently"
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start, end, label = {
                "today": (midnight, now, "today"),
                "yesterday": (midnight - timedelta(days=1), midnight, "yesterday"),
                "this week": (midnight - timedelta(days=midnight.weekday()), now, "this week"),
            }.get(when, (now - timedelta(days=3), now, "in the last three days"))
            files = [path for path in self.downloads(start.timestamp()) if path.stat().st_mtime < end.timestamp() + 1]
            if not files:
                return f"You didn't download anything {label}."
            listed = "; ".join(f"{path.name} ({_size(path.stat().st_size)})" for path in files[:5])
            more = f" And {len(files) - 5} more." if len(files) > 5 else ""
            return f"You downloaded {len(files)} file{'' if len(files) == 1 else 's'} {label}: {listed}.{more}"
        find = FIND_REQUEST.match(normalized)
        if not find:
            return None
        phrase = find.group("what")
        kind = next((KIND_WORDS[word] for word in KIND_WORDS if re.search(rf"\b{re.escape(word)}\b", phrase)), None)
        words = [
            word
            for word in re.sub(r"[_\-.]+", " ", phrase).split()
            if word not in GENERIC and word not in KIND_WORDS and len(word) > 1
        ]
        if not words and kind is None:
            return None
        if not words:
            return None  # "where is my spreadsheet": too vague to answer from names alone
        hits, complete = self.search(words, KINDS.get(kind or "", ()))
        name = " ".join(words)
        if not hits:
            if kind is None:
                return None  # leave "where are my keys" and wide searches to the conversation and the planner
            folders = ", ".join(root.name for root in self._roots()) or "your folders"
            scope = "" if complete else " (I stopped after a few seconds)"
            return f"I couldn't find a {kind} named {name} in {folders}{scope}."
        home = Path.home()
        shown = []
        for path in hits[:3]:
            try:
                folder = "~\\" + str(path.parent.relative_to(home)) if path.is_relative_to(home) else str(path.parent)
            except ValueError:
                folder = str(path.parent)
            shown.append(f"{path.name} in {folder}, changed {_when(path.stat().st_mtime, now)}")
        more = f" There are {len(hits) - 3} more matches." if len(hits) > 3 else ""
        return ("The newest match is " if len(hits) > 1 else "I found ") + "; ".join(shown) + "." + more

    async def answer_async(self, text: str) -> str | None:
        import asyncio

        return await asyncio.to_thread(self.answer, text)
