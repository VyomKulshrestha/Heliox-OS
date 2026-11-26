"""file_write, file_copy and download_file are compared with the file on disk, also in the awkward cases."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, DownloadParams, FileParams
from pilot.agents.executor import Executor
from pilot.agents.postconditions import files
from pilot.agents.verifier import Verifier
from pilot.system.filesystem import file_write


async def _verify(action: Action, output: str = "", success: bool = True, error: str | None = None):
    return await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=success, output=output, error=error)]
    )


def _write_action(path, content: str | None) -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE, target=str(path), parameters=FileParams(path=str(path), content=content)
    )


# ── file_write ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_written_file_verifies_by_the_hash_the_handler_recorded(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    output = await file_write(str(target), "line one\nline two\n")

    assert "sha256" in output and output.startswith("Written 18 bytes to")
    verification = await _verify(_write_action(target, "line one\nline two\n"), output)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_substituted_content_is_verified_not_flagged(tmp_path) -> None:
    """The plan says {OUTPUT}; the executor wrote the earlier step's text. That write is correct."""
    target = tmp_path / "summary.txt"
    action = _write_action(target, "Summary:\n{OUTPUT}")
    # Only the handler's substitution runs, on a stand-in without any system manager: it reaches nothing but
    # filesystem.file_write into tmp_path (no snapshot, no gate, no audit; Executor.execute is never called).
    executor = SimpleNamespace(
        _OUTPUT_PLACEHOLDERS=Executor._OUTPUT_PLACEHOLDERS,
        _PLACEHOLDER_REGEX=Executor._PLACEHOLDER_REGEX,
        _largest_output="the extracted page text",
        _last_output="the extracted page text",
    )

    output = await Executor._exec_file_write(executor, action)  # type: ignore[arg-type]

    assert target.read_bytes() == "Summary:\nthe extracted page text".replace("\n", os.linesep).encode()
    verification = await _verify(action, output)
    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_carriage_returns_in_the_content_still_verify(tmp_path) -> None:
    target = tmp_path / "windows.txt"
    content = "a\r\nb\n\rc"
    output = await file_write(str(target), content)

    assert (await _verify(_write_action(target, content), output)).independently_observed_actions == [0]
    # Without a recorded hash (an "Already satisfied" result), the plan's text is compared byte for byte.
    assert (await _verify(_write_action(target, content), "")).independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_file_changed_after_the_write_is_a_mismatch(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    output = await file_write(str(target), "expected text")
    target.write_bytes(b"expected tex")

    verification = await _verify(_write_action(target, "expected text"), output)

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]
    assert "12 bytes" in verification.details[0] and "13 bytes" in verification.details[0]

    target.write_bytes(b"EXPECTED TEXT")
    verification = await _verify(_write_action(target, "expected text"), output)
    assert verification.passed is False
    assert "differs from what was written" in verification.details[0]


@pytest.mark.asyncio
async def test_a_missing_file_is_a_mismatch(tmp_path) -> None:
    verification = await _verify(_write_action(tmp_path / "never.txt", "x"), "Written 1 bytes")

    assert verification.passed is False
    assert "does not exist" in verification.details[0]


@pytest.mark.asyncio
async def test_no_content_and_no_record_is_reported_and_never_skips_the_write(tmp_path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    action = _write_action(target, None)

    verification = await _verify(action, "Written 0 bytes")
    precheck = await Verifier(MagicMock()).check_already_satisfied(action)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert precheck is None


@pytest.mark.asyncio
async def test_the_precheck_still_compares_the_plan_text(tmp_path) -> None:
    target = tmp_path / "ready.txt"
    await file_write(str(target), "ready\n")

    same = await Verifier(MagicMock()).check_already_satisfied(_write_action(target, "ready\n"))
    other = await Verifier(MagicMock()).check_already_satisfied(_write_action(target, "other"))

    assert same is not None and same[0] is True
    assert other is not None and other[0] is False


@pytest.mark.asyncio
async def test_a_failed_write_never_reaches_the_check(tmp_path) -> None:
    verification = await _verify(_write_action(tmp_path / "x.txt", "x"), success=False, error="disk full")

    assert verification.passed is False
    assert "FAILED — disk full" in verification.details[0]


def test_file_write_puts_the_same_bytes_as_text_mode(tmp_path) -> None:
    import asyncio

    content = "one\ntwo\r\nthree"
    asyncio.run(file_write(str(tmp_path / "new.txt"), content))
    (tmp_path / "old.txt").write_text(content, "utf-8")

    assert (tmp_path / "new.txt").read_bytes() == (tmp_path / "old.txt").read_bytes()
    assert files._expected_text_bytes(content) == (tmp_path / "old.txt").read_bytes()


# ── file_copy ────────────────────────────────────────────────────────


def _copy_action(source, destination) -> Action:
    return Action(
        action_type=ActionType.FILE_COPY,
        target=str(source),
        parameters=FileParams(path=str(source), destination=str(destination), recursive=True),
    )


@pytest.mark.asyncio
async def test_a_sparse_file_copy_verifies(tmp_path) -> None:
    import shutil

    source = tmp_path / "disk.img"
    with source.open("wb") as handle:
        handle.seek(3 * 1024 * 1024)  # a hole of zeros where the file system allows one
        handle.write(b"end")
    destination = tmp_path / "copy.img"
    shutil.copy2(source, destination)

    verification = await _verify(_copy_action(source, destination))

    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_tree_with_a_symbolic_link_verifies(tmp_path) -> None:
    import shutil

    source = tmp_path / "project"
    (source / "real").mkdir(parents=True)
    (source / "real" / "a.txt").write_text("a", encoding="utf-8")
    try:
        (source / "linked").symlink_to(source / "real", target_is_directory=True)
        (source / "b.txt").symlink_to(source / "real" / "a.txt")
    except OSError:
        pytest.skip("this account may not create symbolic links")
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)  # what file_copy does: links are followed and copied as real files

    verification = await _verify(_copy_action(source, destination))

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_tree_missing_a_nested_empty_folder_is_a_mismatch(tmp_path) -> None:
    import shutil

    source = tmp_path / "src"
    (source / "logs" / "old").mkdir(parents=True)
    (source / "keep.txt").write_text("k", encoding="utf-8")
    destination = tmp_path / "dst"
    shutil.copytree(source, destination)
    assert (await _verify(_copy_action(source, destination))).independently_observed_actions == [0]

    (destination / "logs" / "old").rmdir()
    verification = await _verify(_copy_action(source, destination))

    assert verification.passed is False
    assert "logs/old" in verification.details[0]


@pytest.mark.asyncio
async def test_a_large_tree_is_walked_without_a_file_cap_and_stops_at_the_time_budget(tmp_path, monkeypatch) -> None:
    import shutil

    source = tmp_path / "many"
    for index in range(30):
        (source / f"d{index}").mkdir(parents=True)
        (source / f"d{index}" / "f.txt").write_text(str(index), encoding="utf-8")
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)
    assert (await _verify(_copy_action(source, destination))).independently_observed_actions == [0]

    ticks = iter(range(0, 100_000, 50))  # each reading of the clock is 50 s later
    monkeypatch.setattr(files, "_clock", lambda: float(next(ticks)))
    verification = await _verify(_copy_action(source, destination))

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "REPORTED" in verification.details[0]


@pytest.mark.asyncio
async def test_a_copied_file_with_different_bytes_is_a_mismatch(tmp_path) -> None:
    source = tmp_path / "a.bin"
    destination = tmp_path / "b.bin"
    source.write_bytes(b"\x00" * 4096 + b"A")
    destination.write_bytes(b"\x00" * 4096 + b"B")

    verification = await _verify(_copy_action(source, destination))

    assert verification.passed is False
    assert verification.independently_observed_actions == [0]


# ── download_file ────────────────────────────────────────────────────


class _ClientContext:
    def __init__(self, client: object) -> None:
        self._client = client

    async def __aenter__(self) -> object:
        return self._client

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Download:
    def __init__(self, body: bytes, headers: dict[str, str], url: str = "https://files.example/report.pdf") -> None:
        self.content = body
        self.headers = headers
        self.url = url

    def raise_for_status(self) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, response: _Download) -> None:
    from pilot.system import download

    class _Client:
        async def get(self, _url: str) -> _Download:
            return response

    async def _no_system_tool(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the download must not fall back to the system's own tool in a test")

    monkeypatch.setattr(download.PilotConfig, "load", lambda: object())
    monkeypatch.setattr(download, "create_httpx_client", lambda *_a, **_k: _ClientContext(_Client()))
    monkeypatch.setattr(download, "run_powershell", _no_system_tool)
    monkeypatch.setattr(download, "run_command", _no_system_tool)


def _download_action(path, url: str = "https://files.example/report.pdf", sha256: str | None = None) -> Action:
    return Action(
        action_type=ActionType.DOWNLOAD_FILE,
        target=url,
        parameters=DownloadParams(url=url, output_path=str(path), sha256=sha256),
    )


async def _download(monkeypatch, tmp_path, body: bytes, headers: dict[str, str], name: str = "report.pdf"):
    from pilot.system.download import download_file

    _serve(monkeypatch, _Download(body, headers, f"https://files.example/{name}?sig=secret"))
    action = _download_action(tmp_path / name, f"https://files.example/{name}")
    output = await download_file(action.parameters.url, action.parameters.output_path)
    return action, output


@pytest.mark.asyncio
async def test_a_download_matching_its_content_length_verifies(monkeypatch, tmp_path) -> None:
    body = b"%PDF-1.7 " + b"x" * 100
    action, output = await _download(monkeypatch, tmp_path, body, {"content-length": str(len(body))})

    verification = await _verify(action, output)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]
    assert "sig=secret" not in output


@pytest.mark.asyncio
async def test_a_truncated_download_is_a_mismatch(monkeypatch, tmp_path) -> None:
    body = b"%PDF-1.7 " + b"x" * 100
    action, output = await _download(monkeypatch, tmp_path, body, {"content-length": str(len(body))})
    path = tmp_path / "report.pdf"
    path.write_bytes(path.read_bytes()[:50])

    verification = await _verify(action, output)

    assert verification.passed is False
    assert "Content-Length is 109 bytes" in verification.details[0]


@pytest.mark.asyncio
async def test_a_chunked_download_without_length_is_compared_with_the_bytes_received(monkeypatch, tmp_path) -> None:
    action, output = await _download(monkeypatch, tmp_path, b"PK\x03\x04 archive", {"transfer-encoding": "chunked"})

    verification = await _verify(action, output)
    assert verification.independently_observed_actions == [0]

    (tmp_path / "report.pdf").write_bytes(b"PK\x03\x04 archivX")
    verification = await _verify(action, output)
    assert verification.passed is False
    assert "differs from the bytes received" in verification.details[0]


@pytest.mark.asyncio
async def test_a_compressed_download_is_not_held_to_the_compressed_length(monkeypatch, tmp_path) -> None:
    body = b"%PDF " + b"y" * 1000  # httpx hands over the decoded body; the header counts the gzip bytes
    action, output = await _download(monkeypatch, tmp_path, body, {"content-length": "37", "content-encoding": "gzip"})

    verification = await _verify(action, output)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_an_html_error_page_saved_as_a_pdf_is_a_mismatch(monkeypatch, tmp_path) -> None:
    page = b"\n<!DOCTYPE html><html><body>Please sign in</body></html>"
    action, output = await _download(monkeypatch, tmp_path, page, {"content-length": str(len(page))})

    verification = await _verify(action, output)

    assert verification.passed is False
    assert "is an HTML page" in verification.details[0]


@pytest.mark.asyncio
async def test_an_html_page_that_was_asked_for_verifies(monkeypatch, tmp_path) -> None:
    page = b"<!doctype html><html><body>Docs</body></html>"
    action, output = await _download(monkeypatch, tmp_path, page, {"content-length": str(len(page))}, "docs.html")

    verification = await _verify(action, output)

    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_declared_hash_is_compared(tmp_path) -> None:
    import hashlib

    path = tmp_path / "tool.zip"
    path.write_bytes(b"PK zip body")
    good = hashlib.sha256(b"PK zip body").hexdigest()

    # Downloaded by the system's tool (nothing recorded): the declared hash is what can be checked.
    verified = await _verify(_download_action(path, sha256=good.upper()), "Downloaded x -> y (11 bytes)")
    wrong = await _verify(_download_action(path, sha256="0" * 64), "Downloaded x -> y (11 bytes)")
    nothing = await _verify(_download_action(path), "Downloaded x -> y (11 bytes)")

    assert verified.independently_observed_actions == [0]
    assert wrong.passed is False and "declared" in wrong.details[0]
    assert nothing.passed is True and nothing.independently_observed_actions == []


@pytest.mark.asyncio
async def test_a_missing_download_is_a_mismatch_and_a_failed_one_never_reaches_the_check(tmp_path) -> None:
    missing = await _verify(_download_action(tmp_path / "gone.pdf"), "Downloaded")
    failed = await _verify(_download_action(tmp_path / "gone.pdf"), success=False, error="HTTP 404")

    assert missing.passed is False and "not found" in missing.details[0]
    assert failed.passed is False and "FAILED" in failed.details[0]
