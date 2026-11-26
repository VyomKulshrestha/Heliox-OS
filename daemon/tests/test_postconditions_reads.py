"""Checks of the six reads that can lose data (postconditions/reads.py), and the caps reads now say.

Temp files and a fake browser page only: nothing here reads a real mailbox, calendar, web page or screen.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    FileIntelParams,
    FileParams,
    ScreenVisionParams,
)
from pilot.agents.postconditions import reads
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import browser, file_intel, filesystem
from pilot.system.read_limits import CUT_MARK, was_cut

# ── helpers ──────────────────────────────────────────────────────────


async def _verify(action_type: ActionType, parameters: Any, output: str, *, success: bool = True, error: str = ""):
    action = Action(action_type=action_type, target="test", parameters=parameters)
    result = ActionResult(action=action, success=success, output=output, error=error or None)
    return await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])


def _verified(outcome) -> None:
    assert outcome.passed, outcome.details
    assert outcome.independently_observed_actions == [0]
    assert outcome.details[0].split()[-1] == "VERIFIED" or "VERIFIED" in outcome.details[0]
    assert "MISMATCH" not in outcome.details[0]


def _mismatch(outcome) -> str:
    assert not outcome.passed
    assert "MISMATCH" in outcome.details[0]
    return outcome.details[0]


def _reported(outcome) -> str:
    assert outcome.passed
    assert outcome.independently_observed_actions == []
    assert "REPORTED" in outcome.details[0]
    return outcome.details[0]


def test_the_six_reads_are_registered_and_not_marked_read_only() -> None:
    from pilot.agents.postconditions._read_only import READ_ONLY_WITHOUT_POSTCONDITION

    six = {
        ActionType.FILE_READ,
        ActionType.FILE_PARSE,
        ActionType.FILE_HASH,
        ActionType.BROWSER_EXTRACT,
        ActionType.BROWSER_EXTRACT_TABLE,
        ActionType.BROWSER_SCREENSHOT,
    }
    assert six <= set(POSTCONDITION_VERIFIERS)
    assert not six & set(READ_ONLY_WITHOUT_POSTCONDITION)


# ── file_read ────────────────────────────────────────────────────────

SECRET = "line one, private words\r\nline two\r\n"


@pytest.mark.asyncio
async def test_file_read_whole_content_verifies(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(SECRET.encode("utf-8"))
    returned = await filesystem.file_read(str(path))

    _verified(await _verify(ActionType.FILE_READ, FileParams(path=str(path)), returned))


@pytest.mark.asyncio
async def test_file_read_cut_short_is_a_mismatch_by_size(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(SECRET.encode("utf-8"))
    returned = await filesystem.file_read(str(path))

    detail = _mismatch(await _verify(ActionType.FILE_READ, FileParams(path=str(path)), returned[:10]))
    assert "10 of the 33 characters" in detail
    assert "private" not in detail


@pytest.mark.asyncio
async def test_file_read_with_replacement_characters_is_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "accents.txt"
    path.write_bytes("café".encode())
    returned = await filesystem.file_read(str(path))

    detail = _mismatch(await _verify(ActionType.FILE_READ, FileParams(path=str(path)), returned.replace("é", "�")))
    assert "replacement character" in detail


@pytest.mark.asyncio
async def test_file_read_of_a_file_changed_afterwards_is_reported_not_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_bytes(b"first\n")
    returned = await filesystem.file_read(str(path))
    path.write_bytes(b"first\nsecond line added later\n")
    os.utime(path, ns=(1, 1))  # a different modification time even on a coarse clock

    detail = _reported(await _verify(ActionType.FILE_READ, FileParams(path=str(path)), returned))
    assert "changed after it was read" in detail


@pytest.mark.asyncio
async def test_file_read_of_a_removed_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "gone.txt"
    path.write_bytes(b"x")
    returned = await filesystem.file_read(str(path))
    path.unlink()

    assert "removed" in _reported(await _verify(ActionType.FILE_READ, FileParams(path=str(path)), returned))


@pytest.mark.asyncio
async def test_file_read_that_fails_is_a_failed_action(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        await filesystem.file_read(str(path))

    outcome = await _verify(ActionType.FILE_READ, FileParams(path=str(path)), "", success=False, error="decode")
    assert not outcome.passed and "FAILED" in outcome.details[0]


# ── file_hash ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_hash_second_reading_verifies(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(os.urandom(3000))
    returned = await filesystem.file_hash(str(path))

    _verified(await _verify(ActionType.FILE_HASH, FileParams(path=str(path)), returned))


@pytest.mark.asyncio
async def test_file_hash_wrong_digest_is_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    await filesystem.file_hash(str(path))

    detail = _mismatch(await _verify(ActionType.FILE_HASH, FileParams(path=str(path)), f"sha256:{'0' * 64}  {path}"))
    assert "differs from a second reading" in detail


@pytest.mark.asyncio
async def test_file_hash_of_a_changed_or_removed_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    returned = await filesystem.file_hash(str(path))
    path.write_bytes(b"abcdef")
    os.utime(path, ns=(1, 1))
    assert "changed" in _reported(await _verify(ActionType.FILE_HASH, FileParams(path=str(path)), returned))
    path.unlink()
    assert "removed" in _reported(await _verify(ActionType.FILE_HASH, FileParams(path=str(path)), returned))


@pytest.mark.asyncio
async def test_file_hash_that_fails_is_a_failed_action(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await filesystem.file_hash(str(tmp_path / "missing"))
    outcome = await _verify(ActionType.FILE_HASH, FileParams(path="x"), "", success=False, error="File not found")
    assert not outcome.passed and "FAILED" in outcome.details[0]


# ── file_parse ───────────────────────────────────────────────────────


async def _parse_and_verify(path: Path, output: str | None = None):
    returned = await file_intel.parse_file(str(path)) if output is None else output
    return returned, await _verify(ActionType.FILE_PARSE, FileIntelParams(path=str(path)), returned)


@pytest.mark.asyncio
async def test_csv_parse_verifies_and_says_its_cut(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("".join(f"{i},name {i}\n" for i in range(150)), encoding="utf-8")

    returned, outcome = await _parse_and_verify(path)

    _verified(outcome)
    assert "Rows: 150" in returned and "Only the first 100 of 150 rows are shown." in returned


@pytest.mark.asyncio
async def test_csv_parse_with_a_wrong_count_or_a_silent_cut_is_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("".join(f"{i}\n" for i in range(150)), encoding="utf-8")
    header = f"File: rows.csv ({path.stat().st_size:,} bytes)\nType: .csv\n\n"

    _, outcome = await _parse_and_verify(path, header + "Rows: 120\n" + "\n".join(str(i) for i in range(100)))
    assert "150 rows" in _mismatch(outcome)
    _, outcome = await _parse_and_verify(path, header + "Rows: 150\n" + "\n".join(str(i) for i in range(100)))
    assert "without saying so" in _mismatch(outcome)


@pytest.mark.asyncio
async def test_text_parse_whole_and_cut_verify_and_a_silent_cut_does_not(tmp_path: Path) -> None:
    short = tmp_path / "short.md"
    short.write_bytes(b"# title\r\nbody\r\n")
    _verified((await _parse_and_verify(short))[1])

    long = tmp_path / "long.log"
    long.write_bytes(b"".join(f"line {i}\n".encode() for i in range(300)))
    returned, outcome = await _parse_and_verify(long)
    _verified(outcome)
    assert was_cut(returned)

    header = f"File: short.md ({short.stat().st_size} bytes)\nType: .md\n\n"
    assert "without saying so" in _mismatch((await _parse_and_verify(short, header + "# title\n"))[1])


@pytest.mark.asyncio
async def test_text_parse_says_when_bytes_were_replaced(tmp_path: Path) -> None:
    path = tmp_path / "mixed.txt"
    path.write_bytes("caf\xe9 ok\n".encode("latin-1"))
    returned, outcome = await _parse_and_verify(path)
    assert "not valid UTF-8" in returned
    _verified(outcome)


@pytest.mark.asyncio
async def test_xlsx_parse_counts_rows_from_the_xml_even_with_a_wrong_dimension(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    for i in range(1, 131):
        sheet.append([i, f"v{i}"])
    workbook.create_sheet("Empty")
    workbook.save(path)
    _set_dimension(path, "A1")  # some writers store a wrong size; read-only openpyxl used to stop there

    returned, outcome = await _parse_and_verify(path)

    _verified(outcome)
    assert "--- Sheet: Data (130 rows) ---" in returned
    assert "Only the first 100 of 130 rows of sheet Data are shown." in returned


def _set_dimension(path: Path, ref: str) -> None:
    import re as _re

    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    sheet = parts["xl/worksheets/sheet1.xml"].decode("utf-8")
    parts["xl/worksheets/sheet1.xml"] = _re.sub(r'<dimension ref="[^"]*"', f'<dimension ref="{ref}"', sheet).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


@pytest.mark.asyncio
async def test_xlsx_parse_that_stops_early_or_drops_a_sheet_is_a_mismatch(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Data"
    for i in range(1, 31):
        workbook.active.append([i])
    workbook.create_sheet("Other")
    workbook.save(path)
    header = "File: book.xlsx (1 bytes)\nType: .xlsx\n\n"

    stopped = header + "--- Sheet: Data (1 rows) ---\n1\n\n--- Sheet: Other (0 rows) ---\n"
    assert "down to row 30; the parse read only 1" in _mismatch((await _parse_and_verify(path, stopped))[1])
    dropped = header + "--- Sheet: Data (30 rows) ---\n1"
    assert "leaves out 1" in _mismatch((await _parse_and_verify(path, dropped))[1])


@pytest.mark.asyncio
async def test_pdf_parse_accounts_for_every_page(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    returned, outcome = await _parse_and_verify(path)
    _verified(outcome)
    assert "Pages without extractable text: 1, 2, 3" in returned

    header = "File: scan.pdf (1 bytes)\nType: .pdf\n\n"
    assert "declares 3 pages; the parse read 2" in _mismatch((await _parse_and_verify(path, header + "Pages: 2\n"))[1])
    silent = header + "Pages: 3\n\n--- Page 1 ---\ntext"
    assert "2 page(s) are missing" in _mismatch((await _parse_and_verify(path, silent))[1])


@pytest.mark.asyncio
async def test_docx_parse_shows_every_table(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("Intro")
    for _ in range(2):
        document.add_table(rows=2, cols=2)
    document.save(path)

    returned, outcome = await _parse_and_verify(path)
    _verified(outcome)
    one_table = returned.split("\n\n--- Table 2 ---")[0]
    assert "2 table(s)" in _mismatch((await _parse_and_verify(path, one_table))[1])


@pytest.mark.asyncio
async def test_zip_parse_matches_the_archive_directory_and_says_its_cut(tmp_path: Path) -> None:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(120):
            archive.writestr(f"f{i}.txt", "x")

    returned, outcome = await _parse_and_verify(path)
    _verified(outcome)
    assert "Only the first 100 of 120 entries are shown." in returned
    forged = returned.replace("ZIP: 120 files", "ZIP: 100 files")
    assert "records 120 entries" in _mismatch((await _parse_and_verify(path, forged))[1])


@pytest.mark.asyncio
async def test_formats_without_a_count_are_reported_and_a_missing_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"a": "b" * 6000}), encoding="utf-8")
    returned, outcome = await _parse_and_verify(path)
    assert "no page or row count" in _reported(outcome)
    assert "Only the first 5,000 of" in returned  # the JSON cut is said

    with pytest.raises(FileNotFoundError):
        await file_intel.parse_file(str(tmp_path / "missing.pdf"))


@pytest.mark.asyncio
async def test_markup_parse_says_its_cut(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<html><body><p>" + "word " * 2000 + "</p></body></html>", encoding="utf-8")
    returned = await file_intel.parse_file(str(path))
    assert "characters of text are shown" in returned and returned.notice


# ── browser_extract and browser_extract_table ───────────────────────


class _Element:
    def __init__(self, text: str, attrs: dict[str, str] | None = None) -> None:
        self.text = text
        self.attrs = attrs or {}

    async def inner_text(self) -> str:
        return self.text

    async def inner_html(self) -> str:
        return f"<b>{self.text}</b>"

    async def get_attribute(self, name: str) -> str | None:
        return self.attrs.get(name)


class _Page:
    """A page with one set of matching elements and one table; reads answer from them."""

    def __init__(self, elements: list[_Element], rows: list[list[str]] | None = None) -> None:
        self.elements = elements
        self.rows = rows

    async def query_selector_all(self, _selector: str) -> list[_Element]:
        return list(self.elements)

    async def query_selector(self, _selector: str) -> _Element | None:
        return self.elements[0] if self.elements else None

    def _read(self, element: _Element, attribute: str) -> str | None:
        if attribute == "innerText":
            return element.text
        if attribute == "innerHTML":
            return f"<b>{element.text}</b>"
        return element.attrs.get(attribute)

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        if script == reads._ELEMENTS_SCRIPT:
            _selector, attribute, multiple, limit = arg
            found = self.elements if multiple else self.elements[:1]
            return {"count": len(found), "values": [self._read(e, attribute) for e in found[:limit]]}
        if script in (reads._ROWS_SCRIPT, browser.TABLE_SCRIPT):
            return None if self.rows is None else {"headers": ["n", "v"], "rows": [list(r) for r in self.rows]}
        if "a[href]" in script:
            return [{"text": e.text, "href": f"https://example.com/{i}"} for i, e in enumerate(self.elements)]
        raise AssertionError("unexpected script")


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch):
    holder = SimpleNamespace(page=_Page([]), open=True)

    async def get_page(*_args: object) -> _Page:
        return holder.page

    monkeypatch.setattr(browser, "_get_page", get_page)
    monkeypatch.setattr(browser, "has_active_session", lambda: holder.open)
    return holder


@pytest.mark.asyncio
async def test_extract_of_several_elements_verifies(page) -> None:
    page.page = _Page([_Element("alpha"), _Element(" "), _Element("beta")])
    params = BrowserParams(selector=".item", multiple=True)
    returned = await browser.browser_extract(".item", "innerText", True)

    _verified(await _verify(ActionType.BROWSER_EXTRACT, params, returned))
    assert returned == "alpha\n\nbeta"


@pytest.mark.asyncio
async def test_extract_of_many_elements_says_its_cut_and_a_silent_cut_is_a_mismatch(page) -> None:
    page.page = _Page([_Element(f"item {i}") for i in range(120)])
    params = BrowserParams(selector=".item", multiple=True)
    returned = await browser.browser_extract(".item", "innerText", True)

    assert returned.notice == "Only the first 50 of 120 matching elements are shown."
    _verified(await _verify(ActionType.BROWSER_EXTRACT, params, returned))
    silent = "\n\n".join(f"item {i}" for i in range(50))
    detail = _mismatch(await _verify(ActionType.BROWSER_EXTRACT, params, silent))
    assert "has 120 matching elements; the extract kept 50" in detail and "item 1" not in detail


@pytest.mark.asyncio
async def test_extract_of_one_element_cut_short_is_a_mismatch_and_a_changed_page_is_reported(page) -> None:
    page.page = _Page([_Element("a long paragraph of text")])
    params = BrowserParams(selector="p")
    returned = await browser.browser_extract("p")
    _verified(await _verify(ActionType.BROWSER_EXTRACT, params, returned))

    assert "6 characters" in _mismatch(await _verify(ActionType.BROWSER_EXTRACT, params, returned[:6]))
    page.page = _Page([_Element("something else entirely now")])
    assert "may have changed" in _reported(await _verify(ActionType.BROWSER_EXTRACT, params, returned))


@pytest.mark.asyncio
async def test_extract_with_the_browser_closed_or_the_element_gone_is_reported(page) -> None:
    params = BrowserParams(selector="p")
    page.open = False
    assert "browser is closed" in _reported(await _verify(ActionType.BROWSER_EXTRACT, params, "text"))
    page.open = True
    page.page = _Page([])
    assert "no longer on the page" in _reported(await _verify(ActionType.BROWSER_EXTRACT, params, "text"))
    with pytest.raises(LookupError):
        await browser.browser_extract("p")


@pytest.mark.asyncio
async def test_extract_of_inner_html_from_several_elements_returns_them(page) -> None:
    page.page = _Page([_Element("a"), _Element("b")])
    returned = await browser.browser_extract(".x", "innerHTML", True)
    assert json.loads(returned) == ["<b>a</b>", "<b>b</b>"]
    _verified(
        await _verify(
            ActionType.BROWSER_EXTRACT, BrowserParams(selector=".x", attribute="innerHTML", multiple=True), returned
        )
    )


@pytest.mark.asyncio
async def test_table_extract_verifies_and_says_its_cut(page) -> None:
    page.page = _Page([], rows=[[str(i), f"v{i}"] for i in range(150)])
    params = BrowserParams(selector="#rates")
    returned = await browser.browser_extract_table("#rates")

    table = json.loads(returned)
    assert table["rows_total"] == 150 and len(table["rows"]) == 100 and returned.notice
    _verified(await _verify(ActionType.BROWSER_EXTRACT_TABLE, params, returned))


@pytest.mark.asyncio
async def test_table_extract_with_rows_cut_is_a_mismatch(page) -> None:
    rows = [[str(i), f"v{i}"] for i in range(150)]
    page.page = _Page([], rows=rows)
    params = BrowserParams(selector="#rates")
    silent = json.dumps({"headers": ["n", "v"], "rows": rows[:100]})
    assert "did not say how many" in _mismatch(await _verify(ActionType.BROWSER_EXTRACT_TABLE, params, silent))

    page.page = _Page([], rows=rows[:5])
    short = json.dumps({"headers": ["n", "v"], "rows": rows[:3]})
    assert "has 5 rows; the extract returned only 3" in _mismatch(
        await _verify(ActionType.BROWSER_EXTRACT_TABLE, params, short)
    )
    page.page = _Page([], rows=None)
    assert "no longer on the page" in _reported(await _verify(ActionType.BROWSER_EXTRACT_TABLE, params, short))


@pytest.mark.asyncio
async def test_table_selector_is_passed_to_the_page_not_pasted_into_the_script(page) -> None:
    seen: list[Any] = []

    class _Recording(_Page):
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            seen.append(arg)
            return await super().evaluate(script, arg)

    page.page = _Recording([], rows=[["1", "a"]])
    await browser.browser_extract_table("table[data-name='x']")
    assert seen == ["table[data-name='x']"]


@pytest.mark.asyncio
async def test_extract_links_says_its_cut(page) -> None:
    page.page = _Page([_Element(f"link {i}") for i in range(130)])
    returned = await browser.browser_extract_links()
    assert returned.notice == "Only the first 100 of 130 links are shown."
    assert len(json.loads(returned.split("\n" + CUT_MARK)[0])) == 100


# ── browser_screenshot ───────────────────────────────────────────────


def _png(path: Path, colours: list[tuple[int, int, int]]) -> None:
    from PIL import Image

    image = Image.new("RGB", (len(colours), 1))
    for x, colour in enumerate(colours):
        image.putpixel((x, 0), colour)
    image.save(path)


@pytest.mark.asyncio
async def test_browser_screenshot_that_decodes_verifies(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    path = tmp_path / "shot.png"
    _png(path, [(0, 0, 0), (255, 255, 255)])
    output = f"Browser screenshot saved to {path} ({path.stat().st_size:,} bytes)"

    _verified(await _verify(ActionType.BROWSER_SCREENSHOT, BrowserParams(), output))
    _verified(await _verify(ActionType.BROWSER_SCREENSHOT, BrowserParams(output_path=str(path)), "saved"))


@pytest.mark.asyncio
async def test_browser_screenshot_of_one_colour_is_not_verified(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    path = tmp_path / "blank.png"
    _png(path, [(255, 255, 255)] * 4)
    detail = _reported(await _verify(ActionType.BROWSER_SCREENSHOT, BrowserParams(output_path=str(path)), "saved"))
    assert "single colour" in detail


@pytest.mark.asyncio
async def test_browser_screenshot_missing_empty_or_cut_is_a_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    params = BrowserParams(output_path=str(tmp_path / "none.png"))
    assert "no file there" in _mismatch(await _verify(ActionType.BROWSER_SCREENSHOT, params, "saved"))
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert "0 bytes" in _mismatch(
        await _verify(ActionType.BROWSER_SCREENSHOT, BrowserParams(output_path=str(empty)), "")
    )
    cut = tmp_path / "cut.png"
    _png(cut, [(0, 0, 0), (9, 9, 9)])
    cut.write_bytes(cut.read_bytes()[:-20])
    assert "does not decode" in _mismatch(
        await _verify(ActionType.BROWSER_SCREENSHOT, BrowserParams(output_path=str(cut)), "")
    )


@pytest.mark.asyncio
async def test_browser_screenshot_that_failed_is_a_failed_action() -> None:
    outcome = await _verify(
        ActionType.BROWSER_SCREENSHOT, BrowserParams(), "", success=False, error="Element not found"
    )
    assert not outcome.passed and "FAILED" in outcome.details[0]


# ── caps made visible in the other reads ────────────────────────────


@pytest.mark.asyncio
async def test_file_list_says_its_cut(tmp_path: Path) -> None:
    for i in range(510):
        (tmp_path / f"f{i:03}.txt").write_bytes(b"")
    returned = await filesystem.file_list(str(tmp_path))
    assert returned.notice == "Only the first 500 of 510 entries are shown."
    assert returned.count("\n") == 500


@pytest.mark.asyncio
async def test_file_search_says_its_cut_and_counts_every_match(tmp_path: Path) -> None:
    for i in range(210):
        (tmp_path / f"m{i:03}.txt").write_bytes(b"")
    returned = await filesystem.file_search(str(tmp_path), "*.txt")
    assert returned.startswith("Found 210 matches:")
    assert returned.notice == "Only the first 200 of 210 matches are shown."


@pytest.mark.asyncio
async def test_directory_summary_says_what_its_depth_hides(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.txt").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    returned = await filesystem.directory_summary(str(tmp_path), max_depth=2)
    assert "1 folder(s) below depth 2 are not opened." in returned
    assert "1 folder(s) on the ignore list are not shown." in returned and returned.notice


@pytest.mark.asyncio
async def test_content_search_says_when_it_stopped_and_what_it_skipped(tmp_path: Path, monkeypatch) -> None:
    for i in range(5):
        (tmp_path / f"t{i}.txt").write_bytes(b"needle\nneedle\n")
    (tmp_path / "big.txt").write_bytes(b"x" * 20)
    monkeypatch.setattr(file_intel, "SEARCH_MAX_FILE_BYTES", 15)
    returned = await file_intel.search_file_contents(str(tmp_path), "needle", "*.txt", max_results=3)
    assert "stopped at 3 matches" in returned
    returned = await file_intel.search_file_contents(str(tmp_path), "needle", "*.txt", max_results=100)
    assert "1 file(s) over 10 MB were not searched." in returned.notice


@pytest.mark.asyncio
async def test_api_scrape_says_its_cut_and_a_selector_that_matches_nothing_fails(monkeypatch) -> None:
    from pilot.system import api_client

    class _Context:
        def __init__(self, html: str) -> None:
            self.html = html

        async def __aenter__(self):
            html = self.html

            class _Client:
                async def request(self, _method: str, url: str, **_kwargs: object):
                    return SimpleNamespace(url=url, status_code=200, headers={}, text=html, is_redirect=False)

            return _Client()

        async def __aexit__(self, *_args: object) -> None:
            return None

    html = "<html><body><p>" + "word " * 3000 + "</p>" + "<a href='/x'>x</a>" * 120 + "</body></html>"
    monkeypatch.setattr(api_client.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(api_client, "create_httpx_client", lambda *_a, **_k: _Context(html))
    pytest.importorskip("bs4")

    text = await api_client.scrape_url("https://8.8.8.8/page")
    assert "Only the first 10,000 of" in text and text.notice
    links = await api_client.scrape_url("https://8.8.8.8/page", extract="links")
    assert links.notice == "Only the first 100 of 120 links are shown."
    with pytest.raises(LookupError, match="selector"):
        await api_client.scrape_url("https://8.8.8.8/page", selector=".missing")


@pytest.mark.asyncio
async def test_email_fetch_says_how_many_it_left_out_and_which_bodies_it_cut(monkeypatch) -> None:
    from pilot.actions import EmailParams
    from pilot.agents import email_agent

    long_body = b"y" * 2500

    class _IMAP:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def login(self, *_args: object) -> None:
            pass

        def select(self, *_args: object) -> None:
            pass

        def search(self, *_args: object):
            return "OK", [b"1 2 3 4"]

        def fetch(self, uid: bytes, _parts: str):
            if uid == b"3":
                return "NO", [None]
            return "OK", [(b"1", b"From: a@example.com\r\nSubject: s\r\n\r\n" + long_body)]

        def store(self, *_args: object) -> None:
            pass

        def logout(self) -> None:
            pass

    monkeypatch.setattr(email_agent.imaplib, "IMAP4_SSL", _IMAP)
    agent = object.__new__(email_agent.EmailAgent)
    params = EmailParams(imap_host="imap.invalid", username="u", app_password="p", max_emails=3)
    action = Action(action_type=ActionType.EMAIL_FETCH, target="mail", parameters=params)

    result = await agent._fetch_emails(action, params)

    emails = json.loads(result.output)
    assert len(emails) == 2 and all("2,500 characters" in e["body_cut"] for e in emails)
    assert "Only the newest 3 of 4 unread messages were fetched." in result.notice
    assert "1 message(s) could not be fetched" in result.notice


@pytest.mark.asyncio
async def test_calendar_list_says_which_calendars_and_events_it_did_not_read(monkeypatch) -> None:
    from pilot.agents.calendar_agent import CalendarAgent

    good = SimpleNamespace(
        data=b"BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:1\r\nSUMMARY:x\r\nDTSTART:20260101T100000Z\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
    )
    broken = SimpleNamespace(data=b"not a calendar")
    first = SimpleNamespace(name="Home", events=lambda: [good, broken])
    second = SimpleNamespace(name="Work", events=lambda: [])
    client = SimpleNamespace(principal=lambda: SimpleNamespace(calendars=lambda: [first, second]))

    agent = object.__new__(CalendarAgent)
    agent._config = SimpleNamespace(calendar=SimpleNamespace(ics_files=[], enabled=True))
    agent._get_caldav_client = AsyncMock(return_value=client)
    monkeypatch.setattr(agent, "_read_local_calendars", lambda _files: ([], [], []))

    events, sources, warnings = await agent.collect_events()

    assert len(events) == 1 and sources == ["caldav:Home"]
    assert any("1 other calendar(s)" in w for w in warnings)
    assert any("1 event(s) could not be read" in w for w in warnings)


@pytest.mark.asyncio
async def test_process_env_package_and_schedule_lists_say_their_cuts(monkeypatch) -> None:
    from pilot.system import environment, package_mgr, processes, scheduler
    from pilot.system.platform_detect import Platform

    monkeypatch.setitem(__import__("sys").modules, "psutil", None)  # use the command fallback
    monkeypatch.setattr(processes, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(
        processes, "run_command", AsyncMock(return_value=(0, "PID CMD\n" + "\n".join(f"{i} p" for i in range(130)), ""))
    )
    assert (await processes.process_list()).notice == "Only the first 100 of 130 processes are shown."

    monkeypatch.setattr(environment, "os", SimpleNamespace(environ={f"HX_{i:03}": "v" for i in range(120)}))
    assert (await environment.env_list("HX_")).notice == "Only the first 100 of 120 variables are shown."

    monkeypatch.setattr(package_mgr, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(
        package_mgr, "run_command", AsyncMock(return_value=(0, "\n".join(f"pkg{i}" for i in range(25)), ""))
    )
    assert (await package_mgr.package_search("pkg")).notice == "Only the first 20 of 25 lines of results are shown."

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(
        scheduler, "run_command", AsyncMock(return_value=(0, "\n".join(f"task{i}" for i in range(60)), ""))
    )
    assert (await scheduler.schedule_list()).notice == "Only the first 50 of 60 scheduled tasks are shown."


@pytest.mark.asyncio
async def test_schedule_list_on_linux_tells_no_crontab_from_a_failed_read(monkeypatch) -> None:
    from pilot.system import scheduler
    from pilot.system.platform_detect import Platform

    monkeypatch.setattr(scheduler, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(1, "", "no crontab for someone\n")))
    assert await scheduler.schedule_list() == "No crontab configured"
    monkeypatch.setattr(scheduler, "run_command", AsyncMock(return_value=(-1, "", "Command not found: crontab")))
    with pytest.raises(RuntimeError, match="Task list failed"):
        await scheduler.schedule_list()


async def _ocr(monkeypatch, answer: str, region: str | None = None) -> ActionResult:
    from pilot.agents.executor import Executor
    from pilot.system import vision

    monkeypatch.setattr(vision, "screen_ocr", AsyncMock(return_value=answer))
    executor = object.__new__(Executor)
    executor._stress_gate = None
    executor._dispatch_table = {ActionType.SCREEN_OCR: executor._exec_screen_ocr}
    action = Action(action_type=ActionType.SCREEN_OCR, target="screen", parameters=ScreenVisionParams(region=region))
    return await executor._execute_single(action, snapshot_id=None)


@pytest.mark.asyncio
async def test_screen_ocr_without_an_engine_fails_and_an_empty_screen_is_said(monkeypatch) -> None:
    failed = await _ocr(monkeypatch, "[OCR unavailable on this system. Tried: x. To fix: y]")
    assert failed.success is False and "OCR unavailable" in (failed.error or "")
    empty = await _ocr(monkeypatch, "  ")
    assert empty.success is True and empty.output == "No text was found on the screen."
    bad = await _ocr(monkeypatch, "text", region="10,20")
    assert bad.success is False and "x,y,width,height" in (bad.error or "")


@pytest.mark.asyncio
async def test_clipboard_read_on_windows_keeps_the_texts_own_whitespace(monkeypatch) -> None:
    from pilot.system import clipboard
    from pilot.system.platform_detect import Platform

    monkeypatch.setitem(__import__("sys").modules, "pyperclip", None)  # use the PowerShell path
    monkeypatch.setattr(clipboard, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(clipboard, "run_powershell", AsyncMock(return_value=(0, "  indented\r\nnext\r\n", "")))
    assert await clipboard.clipboard_read() == "  indented\r\nnext"
