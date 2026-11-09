"""Screen OCR used by supervision must not hold GPU memory."""

import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from pilot.system import vision
from pilot.system.platform_detect import Platform


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeEasyOCR:
    def __init__(self) -> None:
        self.gpu_requests: list[bool] = []

    def Reader(self, langs, gpu=True, verbose=True):  # noqa: N802 - mirrors easyocr.Reader
        self.gpu_requests.append(gpu)
        return SimpleNamespace(readtext=lambda _image: [([[0, 0], [9, 0], [9, 9], [0, 9]], "Delete files", 0.9)])


@pytest.fixture
def screen(monkeypatch):
    fake = _FakeEasyOCR()
    monkeypatch.setitem(sys.modules, "easyocr", fake)
    for name in [key for key in vars(vision._ocr_easyocr) if key.startswith("_easyocr_reader_")]:
        monkeypatch.delattr(vision._ocr_easyocr, name)

    async def capture(_region=None, **_kwargs):
        return _png()

    async def unavailable(*_args):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(vision, "_capture_screenshot_bytes", capture)
    monkeypatch.setattr(vision, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(vision, "_ocr_tesseract_cli", unavailable)
    return fake


async def test_screen_ocr_fallback_never_requests_the_gpu(screen, monkeypatch):
    async def native_fails(_img):
        raise RuntimeError("WinRT OCR failed")

    monkeypatch.setattr(vision, "_ocr_windows_native", native_fails)

    assert await vision.screen_ocr() == "Delete files"
    assert await vision.screen_ocr() == "Delete files"

    assert screen.gpu_requests == [False]  # built once, on CPU, and cached


async def test_native_ocr_finding_no_text_does_not_load_easyocr(screen, monkeypatch):
    async def native_empty(_img):
        return ""

    monkeypatch.setattr(vision, "_ocr_windows_native", native_empty)

    assert await vision.screen_ocr() == ""
    assert screen.gpu_requests == []


async def test_windows_ocr_script_keeps_the_path_literal(monkeypatch):
    scripts: list[str] = []

    async def run(argv, timeout=None):
        scripts.append(open(argv[-1], encoding="utf-8").read())
        return 0, "Hello", ""

    monkeypatch.setattr(vision, "run_command", run)

    assert await vision._ocr_windows_native(_png()) == "Hello"
    image_line = next(line for line in scripts[0].splitlines() if "$imgPath =" in line)
    assert "\\\\" not in image_line  # single-quoted PowerShell strings are literal
    assert vision._powershell_literal("C:\\Users\\o'brien\\a.png") == "C:\\Users\\o''brien\\a.png"


async def test_on_demand_gpu_read_releases_the_cuda_cache(screen, monkeypatch):
    released: list[bool] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: released.append(True))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = await vision.screen_find_text("Delete")

    assert "Delete files" in result
    assert screen.gpu_requests == [True]
    assert released == [True]
