"""Checks of the six reads that can lose data on the way: a second observation of the same thing, compared.

A read changes nothing, so its check cannot look for an effect. It looks again at the source through another path
and compares that with what the read returned: the file's own bytes for file_read and file_hash, the document's own
count for file_parse, a fresh query of the open page for the two browser extracts, and the image file for
browser_screenshot. A mismatch says what differs by size or count and never quotes content.

The other reads need no check; ``_read_only.py`` lists them with the reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

# ── shared ───────────────────────────────────────────────────────────


def _short_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _universal_newlines(text: str) -> str:
    """What Python's text mode makes of line ends (``newline=None``): CRLF and lone CR both become LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _changed_since(path: str, size: int, mtime_ns: int) -> tuple[int, int] | None:
    """The size and time the read recorded, when the file differs from them now; None when unchanged or unknown."""
    from pilot.system.filesystem import read_stamp

    stamp = read_stamp(path)
    if stamp is None or stamp == (size, mtime_ns):
        return None
    return stamp


def _changed_detail(then: tuple[int, int], size_now: int) -> str:
    return (
        f"The file changed after it was read ({then[0]:,} bytes then, {size_now:,} now), so the next step got the "
        "earlier content; the read itself could not be compared"
    )


# ── file_read ────────────────────────────────────────────────────────


def _compare_file_read(path: str, returned: str) -> Outcome:
    p = Path(path)
    try:
        stat = p.stat()
        data = p.read_bytes()
    except FileNotFoundError:
        return unobserved("The file was removed after it was read, so the text could not be compared with it")
    except OSError as exc:
        return unobserved(f"The file could not be read again to compare ({type(exc).__name__})")
    then = _changed_since(path, stat.st_size, stat.st_mtime_ns)
    try:
        expected = _universal_newlines(data.decode("utf-8"))
    except UnicodeDecodeError:
        if then is not None:
            return unobserved(_changed_detail(then, len(data)))
        return unobserved("The file is no longer valid UTF-8, so it may have changed after it was read")
    if returned == expected:
        return verified(
            f"The {len(returned):,} characters returned match the file's {len(data):,} bytes read again as UTF-8 "
            f"(sha256 {_short_sha(returned)})"
        )
    if then is not None:
        return unobserved(_changed_detail(then, len(data)))
    extra = returned.count("�") - expected.count("�")
    if extra > 0:
        return mismatch(
            f"The read returned {extra} replacement character(s) (U+FFFD) that the file does not contain: its bytes "
            "were decoded with the wrong encoding"
        )
    if len(returned) < len(expected) and expected.startswith(returned):
        return mismatch(
            f"The read returned {len(returned):,} of the {len(expected):,} characters in the file: it was cut short "
            "without saying so"
        )
    return mismatch(
        f"The read returned {len(returned):,} characters (sha256 {_short_sha(returned)}); the file read again "
        f"decodes to {len(expected):,} characters (sha256 {_short_sha(expected)})"
    )


@register(ActionType.FILE_READ, method="file_read_content_postcondition", timeout=20.0)
async def file_read_matches_file(result: ActionResult) -> Outcome:
    """Length and sha256 of the returned text against the file's own bytes, decoded the way file_read decodes."""
    path = str(getattr(result.action.parameters, "path", "") or "")
    if not path:
        raise ValueError("file_read has no path to observe")
    return await asyncio.to_thread(_compare_file_read, path, str(result.output or ""))


# ── file_hash ────────────────────────────────────────────────────────

_HASH_OUTPUT = re.compile(r"^(sha256|sha512|blake2b):([0-9a-f]+)\s")


def _second_digest(p: Path, algorithm: str) -> str:
    """The digest through ``hashlib.file_digest`` (its own buffered reader), not the handler's chunk loop."""
    with p.open("rb", buffering=0) as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, algorithm).hexdigest()
        return hashlib.new(algorithm, handle.read()).hexdigest()  # Python 3.10: one whole read


def _compare_file_hash(path: str, algorithm: str, claimed: str) -> Outcome:
    p = Path(path)
    try:
        stat = p.stat()
        digest = _second_digest(p, algorithm)
    except FileNotFoundError:
        return unobserved("The file was removed after it was hashed, so the digest could not be checked")
    except OSError as exc:
        return unobserved(f"The file could not be read again to check the digest ({type(exc).__name__})")
    if digest == claimed:
        return verified(f"A second {algorithm} of the file's {stat.st_size:,} bytes, read another way, is the same")
    then = _changed_since(path, stat.st_size, stat.st_mtime_ns)
    if then is not None:
        return unobserved(_changed_detail(then, stat.st_size))
    return mismatch(
        f"The {algorithm} reported ({claimed[:12]}...) differs from a second reading of the file ({digest[:12]}...)"
    )


@register(ActionType.FILE_HASH, method="file_hash_second_reading_postcondition", timeout=60.0)
async def file_hash_matches_file(result: ActionResult) -> Outcome:
    """A second digest of the file through a different reader, compared with the one returned."""
    path = str(getattr(result.action.parameters, "path", "") or "")
    match = _HASH_OUTPUT.match(str(result.output or ""))
    if not path or not match:
        raise ValueError("file_hash has no path or no digest in its output to compare")
    return await asyncio.to_thread(_compare_file_hash, path, match.group(1), match.group(2))


# ── file_parse ───────────────────────────────────────────────────────

_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_NOT_COUNTED = {
    ".json": "a JSON parse is the whole document printed again, with no page or row count of its own",
    ".jsonl": "a JSON parse is the whole document printed again, with no page or row count of its own",
    ".xml": "text taken from markup has no count of its own to compare",
    ".html": "text taken from markup has no count of its own to compare",
    ".htm": "text taken from markup has no count of its own to compare",
    ".tar": "a tar archive has no index to count its entries another way",
    ".gz": "a compressed tar archive has no index to count its entries another way",
    ".7z": "the parse does not list this archive's entries",
    ".rar": "the parse does not list this archive's entries",
    ".doc": "the old Word format cannot be counted",
    ".xls": "the old Excel format cannot be counted",
}
_METADATA_ONLY = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"} | {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
_METADATA_ONLY |= {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _parse_body(output: str) -> str:
    """The parse output after its 'File: ...' and 'Type: ...' header."""
    head, _, body = output.partition("\n\n")
    return body if head.startswith("File: ") else output


def _pdf_declared_pages(p: Path) -> int:
    from pypdf import PdfReader  # an ImportError is turned into "not verified" by the caller

    reader = PdfReader(str(p))
    pages = reader.trailer["/Root"].get_object()["/Pages"].get_object()
    return int(pages["/Count"])


def _check_pdf(p: Path, body: str) -> Outcome:
    try:
        declared = _pdf_declared_pages(p)
    except ImportError:
        return unobserved("The PDF's page count could not be read another way: pypdf is not installed")
    reported = re.search(r"^Pages: (\d+)$", body, re.M)
    if not reported:
        return mismatch(f"The document declares {declared} pages; the parse does not say how many it read")
    count = int(reported.group(1))
    with_text = len(re.findall(r"^--- Page \d+ ---$", body, re.M))
    without = 0
    listed = re.search(r"^Pages without extractable text: (.*)$", body, re.M)
    if listed:
        names, _, more = listed.group(1).partition(" and ")
        without = len([n for n in names.split(",") if n.strip()])
        extra = re.match(r"(\d+) more", more)
        if extra:
            without += int(extra.group(1))
    if count != declared:
        return mismatch(f"The document declares {declared} pages; the parse read {count}")
    if with_text + without != declared:
        return mismatch(
            f"The document has {declared} pages; the parse shows text for {with_text} and lists {without} without "
            f"text, so {declared - with_text - without} page(s) are missing without a word"
        )
    return verified(
        f"All {declared} pages the document declares are accounted for: {with_text} with text, {without} without"
    )


def _xlsx_sheets(p: Path) -> list[tuple[str, int]]:
    """Each sheet's name and its last row that holds a value, read from the workbook's XML (not openpyxl)."""
    import xml.etree.ElementTree as ET

    sheets: list[tuple[str, int]] = []
    with zipfile.ZipFile(p) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target") or "" for rel in rels}
        for sheet in workbook.iter(f"{_SHEET_NS}sheet"):
            target = targets.get(sheet.get(f"{_REL_NS}id"), "")
            part = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            last = 0
            counter = 0
            if part in archive.namelist():
                with archive.open(part) as handle:
                    for _event, element in ET.iterparse(handle):
                        if element.tag != f"{_SHEET_NS}row":
                            continue
                        counter = int(element.get("r") or counter + 1)
                        cells = element.findall(f"{_SHEET_NS}c")
                        if any(
                            c.find(f"{_SHEET_NS}v") is not None or c.find(f"{_SHEET_NS}is") is not None for c in cells
                        ):
                            last = counter
                        element.clear()
            sheets.append((sheet.get("name") or "", last))
    return sheets


def _check_xlsx(p: Path, body: str) -> Outcome:
    sheets = _xlsx_sheets(p)
    parsed = [(name, int(rows)) for name, rows in re.findall(r"^--- Sheet: (.*) \((\d+) rows\) ---$", body, re.M)]
    names = [name for name, _ in sheets]
    if [name for name, _ in parsed] != names:
        missing = len([n for n in names if n not in {name for name, _ in parsed}])
        return mismatch(
            f"The workbook has {len(names)} sheet(s); the parse shows {len(parsed)}"
            + (f" and leaves out {missing}" if missing else ", not in the workbook's order")
        )
    short = [(index, rows, last) for index, ((_, rows), (_, last)) in enumerate(zip(parsed, sheets)) if rows < last]
    if short:
        index, rows, last = short[0]
        return mismatch(
            f"Sheet {index + 1} has values down to row {last}; the parse read only {rows} rows of it"
            + (f" ({len(short)} sheets are short)" if len(short) > 1 else "")
        )
    from pilot.system.file_intel import PARSE_ROW_LIMIT

    unsaid = [rows for _, rows in parsed if rows > PARSE_ROW_LIMIT]
    if len(re.findall(r"^\[Cut: Only the first .* rows of sheet ", body, re.M)) < len(unsaid):
        return mismatch("A sheet longer than the parse shows is cut without saying so")
    return verified(f"All {len(sheets)} sheet(s) are read to their last row with a value, counted from the XML")


def _decode_like_a_reader(data: bytes) -> str:
    return _universal_newlines(data.decode("utf-8", errors="replace"))


def _check_csv(p: Path, body: str) -> Outcome:
    import csv

    text = _decode_like_a_reader(p.read_bytes())
    total = sum(1 for _ in csv.reader(io.StringIO(text, newline="")))
    stated = re.search(r"^Rows: (\d+)$", body, re.M)
    if not stated:
        return mismatch(f"The file has {total} rows; the parse does not say how many it read")
    if int(stated.group(1)) != total:
        return mismatch(f"The file has {total} rows, counted again; the parse counted {stated.group(1)}")
    from pilot.system.file_intel import PARSE_ROW_LIMIT
    from pilot.system.read_limits import was_cut

    if total > PARSE_ROW_LIMIT and not was_cut(body):
        return mismatch(f"The file has {total} rows; the parse shows at most {PARSE_ROW_LIMIT} without saying so")
    return verified(f"The parse counted all {total} rows, the same as a fresh pass of the csv reader")


def _check_docx(p: Path, body: str) -> Outcome:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(p) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tree_body = document.find(f"{_WORD_NS}body")
    tables = 0 if tree_body is None else len(tree_body.findall(f"{_WORD_NS}tbl"))
    shown = len(re.findall(r"^--- Table \d+ ---$", body, re.M))
    if shown != tables:
        return mismatch(f"The document has {tables} table(s), counted from its XML; the parse shows {shown}")
    return verified(f"The parse shows all {tables} table(s) of the document; paragraph text is not counted another way")


def _zip_declared_entries(p: Path) -> int | None:
    """The entry count in the archive's end-of-central-directory record; None for ZIP64 or a damaged end."""
    size = p.stat().st_size
    with p.open("rb") as handle:
        handle.seek(max(0, size - 65_557))
        tail = handle.read()
    index = tail.rfind(b"PK\x05\x06")
    if index < 0 or index + 12 > len(tail):
        return None
    total = struct.unpack("<H", tail[index + 10 : index + 12])[0]
    return None if total == 0xFFFF else total


def _check_zip(p: Path, body: str) -> Outcome:
    declared = _zip_declared_entries(p)
    if declared is None:
        return unobserved("The archive's own entry count could not be read (a ZIP64 archive or a damaged end)")
    stated = re.search(r"^ZIP: (\d+) files", body, re.M)
    if not stated or int(stated.group(1)) != declared:
        return mismatch(
            f"The archive records {declared} entries; the parse reported " + (stated.group(1) if stated else "no count")
        )
    from pilot.system.file_intel import PARSE_LIST_LIMIT
    from pilot.system.read_limits import was_cut

    if declared > PARSE_LIST_LIMIT and not was_cut(body):
        return mismatch(
            f"The archive has {declared} entries; the parse lists at most {PARSE_LIST_LIMIT} without saying so"
        )
    return verified(f"The parse reports all {declared} entries the archive's directory records")


def _check_text(p: Path, body: str) -> Outcome:
    data = p.read_bytes()
    lines = data.count(b"\n") + data.count(b"\r") - data.count(b"\r\n") + 1
    if body.startswith("[Note: the file is not valid UTF-8"):
        body = body.partition("]\n")[2]
    cut = re.match(r"\((\d+) lines total, showing first (\d+)\)\n\n", body)
    if cut:
        from pilot.system.read_limits import was_cut

        if int(cut.group(1)) != lines:
            return mismatch(f"The file has {lines} lines, counted from its bytes; the parse said {cut.group(1)}")
        if not was_cut(body):
            return mismatch(f"The parse shows {cut.group(2)} of {lines} lines without a cut line")
        return verified(f"The parse says it shows {cut.group(2)} of the file's {lines} lines, which is right")
    expected = _decode_like_a_reader(data)
    if body == expected:
        return verified(f"The parse returned the whole file: {len(expected):,} characters, {lines} lines")
    if len(body) < len(expected) and expected.startswith(body):
        return mismatch(
            f"The parse returned {len(body):,} of the file's {len(expected):,} characters without saying so"
        )
    return mismatch(
        f"The parse returned {len(body):,} characters (sha256 {_short_sha(body)}); the file decodes to "
        f"{len(expected):,} (sha256 {_short_sha(expected)})"
    )


def _check_parse(path: str, output: str) -> Outcome:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _METADATA_ONLY:
        return unobserved("The parse reads this file's metadata only, so there is no page or row count to compare")
    if ext in _NOT_COUNTED:
        return unobserved(f"Not counted another way: {_NOT_COUNTED[ext]}")
    if not p.is_file():
        return unobserved("The file is no longer there, so the parse could not be compared with it")
    body = _parse_body(output)
    try:
        if ext == ".pdf":
            return _check_pdf(p, body)
        if ext == ".xlsx":
            return _check_xlsx(p, body)
        if ext == ".csv":
            return _check_csv(p, body)
        if ext == ".docx":
            return _check_docx(p, body)
        if ext == ".zip":
            return _check_zip(p, body)
        return _check_text(p, body)
    except Exception as exc:  # unreadable now, or a reader library that broke on this file: not verified
        return unobserved(f"The document could not be counted another way ({type(exc).__name__})")


@register(ActionType.FILE_PARSE, method="conditional_file_parse_count_postcondition", timeout=30.0)
async def file_parse_counts_match(result: ActionResult) -> Outcome:
    """The page, sheet, row, table or entry count of the parse against the document's own count, read another way."""
    path = str(getattr(result.action.parameters, "path", "") or "")
    if not path:
        raise ValueError("file_parse has no path to observe")
    return await asyncio.to_thread(_check_parse, path, str(result.output or ""))


# ── browser_extract and browser_extract_table ───────────────────────

# A fresh reading of the elements, in the page itself, as one script (the handler reads element by element).
_ELEMENTS_SCRIPT = """
    (args) => {
        const [selector, attribute, multiple, limit] = args;
        const read = (el) => attribute === 'innerText' ? el.innerText
            : (attribute === 'innerHTML' ? el.innerHTML : el.getAttribute(attribute));
        const all = multiple ? Array.from(document.querySelectorAll(selector))
            : [document.querySelector(selector)].filter(Boolean);
        return { count: all.length, values: all.slice(0, limit).map(read) };
    }
"""

_ROWS_SCRIPT = """
    (selector) => {
        const table = document.querySelector(selector);
        if (!table) return null;
        const headers = Array.from(table.querySelectorAll('th'), th => th.innerText.trim());
        const rows = [];
        for (const tr of table.querySelectorAll('tr')) {
            const cells = Array.from(tr.querySelectorAll('td'), td => td.innerText.trim());
            if (cells.length) rows.push(cells);
        }
        return { headers, rows };
    }
"""

_CHANGED_PAGE = "the page may have changed between the extract and this reading"


async def _open_page() -> Any | None:
    from pilot.system import browser

    if not browser.has_active_session():
        return None
    return await browser._get_page()


def _strip_cut(text: str) -> str:
    from pilot.system.read_limits import CUT_MARK

    index = text.find("\n" + CUT_MARK)
    return text if index < 0 else text[:index]


@register(ActionType.BROWSER_EXTRACT, method="conditional_extract_fresh_reading_postcondition", timeout=15.0)
async def extract_matches_fresh_reading(result: ActionResult) -> Outcome:
    """Element count and text of a fresh query of the open page against what the extract returned."""
    from pilot.system.browser import EXTRACT_ELEMENT_LIMIT
    from pilot.system.read_limits import was_cut

    params = result.action.parameters
    selector = str(getattr(params, "selector", "") or "body")
    attribute = str(getattr(params, "attribute", "") or "innerText")
    multiple = bool(getattr(params, "multiple", False))
    page = await _open_page()
    if page is None:
        return unobserved("The browser is closed, so the page could not be read again")
    fresh = await page.evaluate(_ELEMENTS_SCRIPT, [selector, attribute, multiple, EXTRACT_ELEMENT_LIMIT])
    count = int(fresh.get("count") or 0)
    values = [value for value in fresh.get("values") or []]
    returned = str(result.output or "")

    if not multiple:
        if count == 0:
            return unobserved("The element is no longer on the page, so the extract could not be compared")
        value = values[0] or ""
        if returned == (value or "(empty)"):
            return verified(f"The {len(value):,} characters returned match a fresh reading of the element")
        if returned and len(returned) < len(value) and value.startswith(returned):
            return mismatch(
                f"The extract returned {len(returned):,} characters; a fresh reading of the element has "
                f"{len(value):,}: it was cut short"
            )
        return unobserved(f"The element reads differently now ({len(value):,} characters); {_CHANGED_PAGE}")

    texts = [str(value).strip() for value in values if value and str(value).strip()]
    body = "\n\n".join(texts) if attribute == "innerText" else json.dumps(texts, indent=2)
    shown = min(count, EXTRACT_ELEMENT_LIMIT)
    if count > shown and not was_cut(returned):
        return mismatch(f"The page has {count} matching elements; the extract kept {shown} and did not say it was cut")
    if _strip_cut(returned) == body:
        said = f", and says it kept the first {shown}" if count > shown else ""
        return verified(f"{len(texts)} non-empty value(s) from {count} matching element(s) match a fresh reading{said}")
    got = _strip_cut(returned)
    if len(got) < len(body) and body.startswith(got):
        return mismatch(
            f"The extract returned {len(got):,} characters; a fresh reading of the same {shown} element(s) gives "
            f"{len(body):,}: it was cut short"
        )
    return unobserved(f"{count} element(s) match now and their text differs from the extract; {_CHANGED_PAGE}")


@register(ActionType.BROWSER_EXTRACT_TABLE, method="conditional_table_fresh_reading_postcondition", timeout=15.0)
async def table_matches_fresh_reading(result: ActionResult) -> Outcome:
    """Row count and cells of a fresh reading of the table against the rows the extract returned."""
    from pilot.system.browser import EXTRACT_TABLE_ROW_LIMIT

    selector = str(getattr(result.action.parameters, "selector", "") or "table")
    try:
        returned = json.loads(str(result.output or ""))
        rows = list(returned["rows"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("The table extract's output is not the table JSON it should be") from exc
    page = await _open_page()
    if page is None:
        return unobserved("The browser is closed, so the table could not be read again")
    fresh = await page.evaluate(_ROWS_SCRIPT, selector)
    if not fresh:
        return unobserved("The table is no longer on the page, so the extract could not be compared")
    fresh_rows = list(fresh.get("rows") or [])
    total = len(fresh_rows)
    expected = fresh_rows[:EXTRACT_TABLE_ROW_LIMIT]
    if total > EXTRACT_TABLE_ROW_LIMIT and returned.get("rows_total") != total:
        return mismatch(
            f"The table has {total} rows; the extract returned {len(rows)} and did not say how many there are"
        )
    if rows == expected and list(returned.get("headers") or []) == list(fresh.get("headers") or []):
        said = f", and says they are the first {len(rows)} of {total}" if total > len(rows) else ""
        return verified(f"All {len(rows)} rows returned match a fresh reading of the table{said}")
    if len(rows) < len(expected) and expected[: len(rows)] == rows:
        return mismatch(f"The table has {total} rows; the extract returned only {len(rows)}")
    return unobserved(f"The table now has {total} rows and differs from the extract; {_CHANGED_PAGE}")


# ── browser_screenshot ───────────────────────────────────────────────

_SAVED_TO = re.compile(r"saved to (.+) \([\d,]+ bytes\)$")


def _header_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a whole PNG or JPEG from its bytes, or None. A PNG must end with IEND, so a cut file fails."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33 and data[12:16] == b"IHDR":
        if b"IEND" not in data[-16:]:
            return None
        width, height = struct.unpack(">II", data[16:24])
        return (width, height) if width and height else None
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


def _inspect_image(p: Path) -> tuple[tuple[int, int] | None, bool | None]:
    """The decoded size and whether every pixel has one colour (None when Pillow is not there to look)."""
    data = p.read_bytes()
    size = _header_size(data)
    if data.startswith(b"\x89PNG") and size is None:
        return None, None
    try:
        from PIL import Image
    except ImportError:
        return size, None
    try:
        with Image.open(p) as image:
            image.load()
            extrema = image.convert("RGBA").getextrema()
            return image.size, all(low == high for low, high in extrema)
    except Exception:
        return None, None


def _check_screenshot(path: str) -> Outcome:
    p = Path(path)
    if not p.is_file():
        return mismatch(f"Expected a screenshot at {p}; there is no file there")
    if p.stat().st_size == 0:
        return mismatch(f"The screenshot at {p} is empty (0 bytes)")
    size, single_colour = _inspect_image(p)
    if size is None:
        return mismatch(f"The file at {p} does not decode as an image")
    width, height = size
    if single_colour:
        return unobserved(f"Not verified: the image is a single colour ({width}x{height}); the page may be blank")
    if single_colour is None:
        return verified(
            f"{p.name} is a whole {width}x{height} image (Pillow is not installed, so blankness was not checked)"
        )
    return verified(f"{p.name} decodes as a {width}x{height} image with more than one colour")


@register(ActionType.BROWSER_SCREENSHOT, method="image_file_postcondition", timeout=15.0)
async def browser_screenshot_file(result: ActionResult) -> Outcome:
    """The image file exists, decodes as an image, and is not all one colour."""
    path = str(getattr(result.action.parameters, "output_path", "") or "")
    if not path:
        match = _SAVED_TO.search(str(result.output or "").strip())
        path = match.group(1).strip() if match else ""
    if not path:
        raise ValueError("The browser screenshot has no file path to observe")
    return await asyncio.to_thread(_check_screenshot, path)
