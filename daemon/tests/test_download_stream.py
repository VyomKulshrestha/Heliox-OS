"""download_file streams to a temporary file beside the target, within a size limit, and leaves nothing behind
when it is refused or cut. Fakes only: a fake resolver and a fake transport that streams (tests/fake_outbound.py)."""

from __future__ import annotations

import gc
import hashlib
from collections import namedtuple

import httpx
import pytest

from pilot.actions import DOWNLOAD_DEFAULT_MAX_BYTES, Action, ActionType, DownloadParams, PermissionTier
from pilot.system import download
from tests import fake_outbound


class Chunks(httpx.AsyncByteStream):
    """A body that arrives in pieces and counts how many were read."""

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = pieces
        self.read = 0

    def __aiter__(self):
        # An iterator object, not an async generator: the fake stream leaves nothing for the collector to finalize.
        self._next = iter(self.pieces)
        return self

    async def __anext__(self) -> bytes:
        try:
            piece = next(self._next)
        except StopIteration:
            raise StopAsyncIteration from None
        self.read += 1
        return piece

    async def aclose(self) -> None:
        return None


def _serve(monkeypatch, stream: httpx.AsyncByteStream, headers: dict[str, str] | None = None) -> None:
    fake_outbound.install(monkeypatch, handler=lambda _r: httpx.Response(200, headers=headers or {}, stream=stream))


def _leftovers(folder) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# A stream reader left for the garbage collector (an aiter_raw nobody closed) warns when it is finalized: here that
# fails the test, and each test collects at its end so the warning lands inside it.
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning", "error::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(autouse=True)
def _small_limit(monkeypatch):
    monkeypatch.setattr(download, "DEFAULT_MAX_BYTES", 100)
    yield
    gc.collect()


async def test_a_streamed_download_is_written_whole_and_hashed(monkeypatch, tmp_path) -> None:
    pieces = [b"PK\x03\x04", b"a" * 30, b"b" * 30]
    _serve(monkeypatch, Chunks(pieces))
    target = tmp_path / "f.zip"

    output = await download.download_file("https://files.example/f.zip", str(target))

    body = b"".join(pieces)
    assert target.read_bytes() == body
    assert hashlib.sha256(body).hexdigest() in output and f"({len(body):,} bytes)" in output
    assert _leftovers(tmp_path) == ["f.zip"]


async def test_a_stated_length_over_the_limit_is_refused_before_any_byte(monkeypatch, tmp_path) -> None:
    stream = Chunks([b"x" * 50] * 4)
    _serve(monkeypatch, stream, {"content-length": "200"})

    with pytest.raises(download.DownloadRefused, match=r"larger than the 100 bytes download limit"):
        await download.download_file("https://files.example/big.iso", str(tmp_path / "big.iso"))

    assert stream.read == 0
    assert _leftovers(tmp_path) == []


async def test_a_stream_that_passes_the_limit_is_cut_and_nothing_is_left(monkeypatch, tmp_path) -> None:
    stream = Chunks([b"x" * 40] * 10)  # no length stated
    _serve(monkeypatch, stream)

    with pytest.raises(download.DownloadRefused, match="download limit"):
        await download.download_file("https://files.example/endless", str(tmp_path / "endless.bin"))

    assert stream.read == 3  # stopped at the piece that passed 100 bytes
    assert _leftovers(tmp_path) == []


async def test_a_body_shorter_than_its_stated_length_is_not_kept(monkeypatch, tmp_path) -> None:
    _serve(monkeypatch, Chunks([b"x" * 20]), {"content-length": "50"})

    with pytest.raises(download.DownloadRefused, match="stopped early"):
        await download.download_file("https://files.example/cut.bin", str(tmp_path / "cut.bin"))
    assert _leftovers(tmp_path) == []


async def test_a_refused_download_leaves_an_existing_file_untouched(monkeypatch, tmp_path) -> None:
    target = tmp_path / "keep.bin"
    target.write_bytes(b"the old file")
    _serve(monkeypatch, Chunks([b"y" * 60] * 3))

    with pytest.raises(download.DownloadRefused):
        await download.download_file("https://files.example/keep.bin", str(target), overwrite=True)

    assert target.read_bytes() == b"the old file"
    assert _leftovers(tmp_path) == ["keep.bin"]


async def test_too_little_free_space_is_refused_before_writing(monkeypatch, tmp_path) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _p: usage(10**9, 10**9, 1000))
    stream = Chunks([b"z" * 50])
    _serve(monkeypatch, stream, {"content-length": "50"})

    with pytest.raises(download.DownloadRefused, match="not enough free disk space"):
        await download.download_file("https://files.example/z.bin", str(tmp_path / "z.bin"))
    assert stream.read == 0 and _leftovers(tmp_path) == []


async def test_a_raised_limit_lets_a_larger_file_through(monkeypatch, tmp_path) -> None:
    _serve(monkeypatch, Chunks([b"q" * 100] * 3), {"content-length": "300"})
    target = tmp_path / "image.img"

    await download.download_file("https://files.example/image.img", str(target), max_bytes=1000)

    assert target.stat().st_size == 300


async def test_an_error_answer_writes_nothing(monkeypatch, tmp_path) -> None:
    from pilot.system.api_client import HttpStatusError

    fake_outbound.install(monkeypatch, handler=lambda _r: httpx.Response(404, text="missing"))

    with pytest.raises(HttpStatusError, match="HTTP 404"):
        await download.download_file("https://files.example/none", str(tmp_path / "none"))
    assert _leftovers(tmp_path) == []


def test_raising_the_limit_above_the_default_needs_approval() -> None:
    def tier(max_bytes):
        params = DownloadParams(url="https://files.example/x", output_path="x", max_bytes=max_bytes)
        return Action(action_type=ActionType.DOWNLOAD_FILE, parameters=params).permission_tier

    assert tier(None) == PermissionTier.USER_WRITE
    assert tier(DOWNLOAD_DEFAULT_MAX_BYTES) == PermissionTier.USER_WRITE
    assert tier(6 * 1024**3) == PermissionTier.SYSTEM_MODIFY


def test_a_raised_size_limit_is_asked_every_time_and_the_card_says_the_size() -> None:
    from pilot.server import _approval_effect

    def step(max_bytes):
        params = DownloadParams(url="https://files.example/x", output_path="x", max_bytes=max_bytes)
        return Action(action_type=ActionType.DOWNLOAD_FILE, parameters=params)

    # Asked like a step with the local flag: no setting that skips a tier's approval skips it.
    assert step(6 * 1024**3).is_irreversible and step(6 * 1024**3).requires_confirmation
    assert not step(None).is_irreversible and not step(DOWNLOAD_DEFAULT_MAX_BYTES).is_irreversible
    assert (
        _approval_effect(step(6 * 1024**3))
        == "A download of up to 6 gigabytes, more than the usual limit of 2 gigabytes."
    )
    assert _approval_effect(step(None)) == ""
