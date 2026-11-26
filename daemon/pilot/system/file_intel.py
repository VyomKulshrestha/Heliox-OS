"""File Content Intelligence — understand file contents, not just metadata.

Parse PDFs, Word docs, Excel, images, and more. Extract text, summarize,
search within file contents, and analyze structure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger("pilot.system.file_intel")


class UnreadableFileError(ValueError):
    """The file exists but none of the readers could turn it into text."""


# What a parse shows at most. Every cut is said in the output with ``read_limits.cut_line``.
PARSE_ROW_LIMIT = 100
PARSE_TEXT_LINE_LIMIT = 200
PARSE_CHAR_LIMIT = 5000
PARSE_LIST_LIMIT = 100
PARSE_TAG_LIMIT = 20


def _cut(shown: int, total: int | None, noun: str) -> str:
    from pilot.system.read_limits import cut_line

    return cut_line(shown, total, noun)


def _decode(data: bytes) -> tuple[str, str]:
    """UTF-8 text with universal newlines, and a note when bytes had to be replaced (empty when none)."""
    try:
        text = data.decode("utf-8")
        note = ""
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        note = (
            "[Note: the file is not valid UTF-8; "
            f"{text.count(chr(0xFFFD))} undecodable byte sequence(s) are shown as \ufffd]\n"
        )
    return text.replace("\r\n", "\n").replace("\r", "\n"), note


async def parse_file(file_path: str) -> str:
    """Auto-detect file type and extract text/data content."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = p.suffix.lower()
    size = p.stat().st_size

    info = f"File: {p.name} ({size:,} bytes)\nType: {ext}\n\n"

    try:
        if ext == ".pdf":
            return info + await _parse_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return info + await _parse_docx(file_path)
        elif ext in (".xlsx", ".xls"):
            return info + await _parse_excel(file_path)
        elif ext == ".csv":
            return info + await _parse_csv(file_path)
        elif ext in (".json", ".jsonl"):
            return info + await _parse_json(file_path)
        elif ext in (".xml", ".html", ".htm"):
            return info + await _parse_xml(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"):
            return info + await _parse_image(file_path)
        elif ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
            return info + await _parse_audio_meta(file_path)
        elif ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
            return info + await _parse_video_meta(file_path)
        elif ext in (".zip", ".tar", ".gz", ".7z", ".rar"):
            return info + await _parse_archive(file_path)
        elif ext in (
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".r",
            ".sql",
            ".sh",
            ".ps1",
            ".bat",
            ".toml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
            ".md",
            ".rst",
            ".txt",
            ".log",
            ".env",
        ):
            return info + await _parse_text(file_path)
        else:
            # Try as text
            try:
                return info + await _parse_text(file_path)
            except Exception as e:
                raise UnreadableFileError(f"Heliox cannot read {p.name}: unable to parse file type {ext}") from e
    except ImportError as e:
        raise RuntimeError(f"Heliox cannot read {p.name}: a reader library is missing ({e})") from e
    except UnreadableFileError:
        raise
    except Exception as e:
        raise RuntimeError(f"Heliox could not parse {p.name}: {e}") from e


async def _parse_pdf(path: str) -> str:
    """Extract text from PDF."""
    # Try PyPDF2 / pypdf
    try:
        from pypdf import PdfReader

        def _do():
            reader = PdfReader(path)
            pages = []
            empty: list[int] = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"--- Page {i + 1} ---\n{text}")
                else:
                    empty.append(i + 1)
            return _pdf_text(len(reader.pages), pages, empty)

        return await asyncio.to_thread(_do)
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber

        def _do():
            with pdfplumber.open(path) as pdf:
                pages = []
                empty: list[int] = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(f"--- Page {i + 1} ---\n{text}")
                    else:
                        empty.append(i + 1)
                return _pdf_text(len(pdf.pages), pages, empty)

        return await asyncio.to_thread(_do)
    except ImportError:
        raise ImportError("pypdf or pdfplumber (pip install pypdf pdfplumber)")


def _pdf_text(count: int, pages: list[str], empty: list[int]) -> str:
    """'Pages: N', the pages with text, and the pages without (scans or images), so no page goes missing unsaid."""
    head = f"Pages: {count}\n"
    if empty:
        listed = ", ".join(str(n) for n in empty[:PARSE_LIST_LIMIT])
        more = f" and {len(empty) - PARSE_LIST_LIMIT} more" if len(empty) > PARSE_LIST_LIMIT else ""
        head += f"Pages without extractable text: {listed}{more}\n"
    return head + "\n" + "\n\n".join(pages)


async def _parse_docx(path: str) -> str:
    """Extract text from Word documents."""
    from docx import Document

    def _do():
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        tables_text = []
        for i, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            tables_text.append(f"--- Table {i + 1} ---\n" + "\n".join(rows))

        output = "\n\n".join(paragraphs)
        if tables_text:
            output += "\n\n" + "\n\n".join(tables_text)
        return output

    return await asyncio.to_thread(_do)


async def _parse_excel(path: str) -> str:
    """Extract data from Excel files."""
    try:
        import openpyxl

        def _do():
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            output = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                # A stored dimension that is wrong (some writers save "A1") would make a read-only sheet stop early.
                if hasattr(ws, "reset_dimensions"):
                    ws.reset_dimensions()
                rows = []
                total = 0
                for row in ws.iter_rows(values_only=True):
                    total += 1
                    if total <= PARSE_ROW_LIMIT:
                        cells = [str(c) if c is not None else "" for c in row]
                        rows.append(" | ".join(cells))
                block = f"--- Sheet: {sheet_name} ({total} rows) ---\n" + "\n".join(rows)
                if total > PARSE_ROW_LIMIT:
                    block += "\n" + _cut(PARSE_ROW_LIMIT, total, f"rows of sheet {sheet_name}")
                output.append(block)
            wb.close()
            return "\n\n".join(output)

        return await asyncio.to_thread(_do)
    except ImportError:
        raise ImportError("openpyxl (pip install openpyxl)")


async def _parse_csv(path: str) -> str:
    """Parse CSV file."""
    import csv

    def _do():
        import io

        text, note = _decode(Path(path).read_bytes())
        rows = []
        total = 0
        for row in csv.reader(io.StringIO(text, newline="")):
            total += 1
            if total <= PARSE_ROW_LIMIT:
                rows.append(" | ".join(row))
        body = f"{note}Rows: {total}\n" + "\n".join(rows)
        if total > PARSE_ROW_LIMIT:
            body += "\n" + _cut(PARSE_ROW_LIMIT, total, "rows")
        return body

    return await asyncio.to_thread(_do)


async def _parse_json(path: str) -> str:
    """Parse JSON file."""
    text = Path(path).read_text("utf-8")
    data = json.loads(text)
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    if len(formatted) > PARSE_CHAR_LIMIT:
        formatted = formatted[:PARSE_CHAR_LIMIT] + "\n" + _cut(PARSE_CHAR_LIMIT, len(formatted), "characters")
    return formatted


async def _parse_xml(path: str) -> str:
    """Extract text from XML/HTML."""
    try:
        from bs4 import BeautifulSoup

        text, note = _decode(Path(path).read_bytes())
        clean = BeautifulSoup(text, "html.parser").get_text(separator="\n", strip=True)
    except ImportError:
        # Basic fallback
        import re

        text, note = _decode(Path(path).read_bytes())
        clean = re.sub(r"<[^>]+>", "", text)
    if len(clean) > PARSE_CHAR_LIMIT:
        return note + clean[:PARSE_CHAR_LIMIT] + "\n" + _cut(PARSE_CHAR_LIMIT, len(clean), "characters of text")
    return note + clean


async def _parse_text(path: str) -> str:
    """Read plain text files."""
    text, note = _decode(Path(path).read_bytes())
    lines = text.split("\n")
    line_count = len(lines)
    if line_count > PARSE_TEXT_LINE_LIMIT:
        preview = "\n".join(lines[:PARSE_TEXT_LINE_LIMIT])
        return f"{note}({line_count} lines total, showing first {PARSE_TEXT_LINE_LIMIT})\n\n{preview}\n" + _cut(
            PARSE_TEXT_LINE_LIMIT, line_count, "lines"
        )
    return note + text


async def _parse_image(path: str) -> str:
    """Get image metadata and optionally OCR text."""
    info_parts = []

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        def _do():
            img = Image.open(path)
            parts = [
                f"Dimensions: {img.width}x{img.height}",
                f"Mode: {img.mode}",
                f"Format: {img.format}",
            ]
            # EXIF data
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in list(exif_data.items())[:PARSE_TAG_LIMIT]:
                    tag = TAGS.get(tag_id, tag_id)
                    parts.append(f"  {tag}: {value}")
                if len(exif_data) > PARSE_TAG_LIMIT:
                    parts.append(_cut(PARSE_TAG_LIMIT, len(exif_data), "EXIF tags"))
            return "\n".join(parts)

        info_parts.append(await asyncio.to_thread(_do))
    except ImportError:
        info_parts.append("(Install Pillow for image metadata: pip install Pillow)")

    return "\n".join(info_parts)


async def _parse_audio_meta(path: str) -> str:
    """Get audio file metadata."""
    try:
        from mutagen import File

        def _do():
            audio = File(path)
            if audio is None:
                return "Could not read audio metadata"
            parts = [f"Length: {audio.info.length:.1f}s"]
            if hasattr(audio.info, "bitrate"):
                parts.append(f"Bitrate: {audio.info.bitrate // 1000}kbps")
            if hasattr(audio.info, "sample_rate"):
                parts.append(f"Sample Rate: {audio.info.sample_rate}Hz")
            tags = list(audio.items())
            for key, value in tags[:PARSE_TAG_LIMIT]:
                parts.append(f"  {key}: {value}")
            if len(tags) > PARSE_TAG_LIMIT:
                parts.append(_cut(PARSE_TAG_LIMIT, len(tags), "tags"))
            return "\n".join(parts)

        return await asyncio.to_thread(_do)
    except ImportError:
        return "(Install mutagen for audio metadata: pip install mutagen)"


async def _parse_video_meta(path: str) -> str:
    """Get video file metadata."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if stdout:
            data = json.loads(stdout.decode("utf-8"))
            fmt = data.get("format", {})
            parts = [
                f"Duration: {float(fmt.get('duration', 0)):.1f}s",
                f"Format: {fmt.get('format_long_name', 'unknown')}",
                f"Size: {int(fmt.get('size', 0)):,} bytes",
            ]
            for stream in data.get("streams", []):
                codec = stream.get("codec_type", "unknown")
                if codec == "video":
                    parts.append(f"Video: {stream.get('width')}x{stream.get('height')} {stream.get('codec_name')}")
                elif codec == "audio":
                    parts.append(f"Audio: {stream.get('codec_name')} {stream.get('sample_rate')}Hz")
            return "\n".join(parts)
    except FileNotFoundError:
        pass

    return f"Video file: {Path(path).stat().st_size:,} bytes (install ffprobe for metadata)"


async def _parse_archive(path: str) -> str:
    """List contents of an archive file."""
    ext = Path(path).suffix.lower()

    if ext == ".zip":
        import zipfile

        with zipfile.ZipFile(path, "r") as zf:
            items = zf.namelist()
            total = sum(i.file_size for i in zf.infolist())
            listing = "\n".join(items[:PARSE_LIST_LIMIT])
            if len(items) > PARSE_LIST_LIMIT:
                listing += "\n" + _cut(PARSE_LIST_LIMIT, len(items), "entries")
            return f"ZIP: {len(items)} files, {total:,} bytes uncompressed\n\n{listing}"

    elif ext in (".tar", ".gz"):
        import tarfile

        with tarfile.open(path, "r:*") as tf:
            members = tf.getmembers()
            listing = "\n".join(m.name for m in members[:PARSE_LIST_LIMIT])
            if len(members) > PARSE_LIST_LIMIT:
                listing += "\n" + _cut(PARSE_LIST_LIMIT, len(members), "entries")
            return f"TAR: {len(members)} entries\n\n{listing}"

    return f"Archive: {Path(path).stat().st_size:,} bytes"


SEARCH_MAX_FILE_BYTES = 10_000_000


async def search_file_contents(
    directory: str,
    search_text: str,
    pattern: str = "*.txt",
    max_results: int = 50,
) -> str:
    """Search for text within file contents (grep-like).

    Searches recursively through files matching the glob pattern.
    """
    results = []
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    skipped = {"large": 0, "unreadable": 0}
    stopped = False

    def _do():
        nonlocal stopped
        count = 0
        for f in p.rglob(pattern):
            if count >= max_results:
                stopped = True
                break
            if not f.is_file():
                continue
            if f.stat().st_size > SEARCH_MAX_FILE_BYTES:
                skipped["large"] += 1
                continue
            try:
                for i, line in enumerate(f.open("r", encoding="utf-8", errors="replace"), 1):
                    if search_text.lower() in line.lower():
                        results.append(
                            {
                                "file": str(f),
                                "line": i,
                                "content": line.strip()[:200],
                            }
                        )
                        count += 1
                        if count >= max_results:
                            stopped = True
                            break
            except Exception:
                skipped["unreadable"] += 1
                continue
        return results

    from pilot.system.read_limits import with_notes

    results = await asyncio.to_thread(_do)
    notes = []
    if stopped:
        notes.append(f"The search stopped at {max_results} matches; there may be more.")
    if skipped["large"]:
        notes.append(f"{skipped['large']} file(s) over 10 MB were not searched.")
    if skipped["unreadable"]:
        notes.append(f"{skipped['unreadable']} file(s) could not be read and were not searched.")
    if not results:
        return with_notes(f"No matches for '{search_text}' in {directory} ({pattern})", notes)
    return with_notes(json.dumps({"matches": results, "count": len(results)}, indent=2), notes)
