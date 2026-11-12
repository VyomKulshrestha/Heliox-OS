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


async def test_windows_ocr_passes_the_image_path_as_an_argument(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    async def run(argv, timeout=None):
        calls.append((open(argv[-2], encoding="utf-8").read(), list(argv)))
        return 0, "Hello", ""

    monkeypatch.setattr(vision, "run_command", run)

    assert await vision._ocr_windows_native(_png()) == "Hello"
    ((script, argv),) = calls
    image = argv[-1]
    assert image.endswith("image.png") and argv[-2].endswith("ocr.ps1")
    # The path (it holds the person's user name, which can hold any quote) is never part of the script.
    assert image not in script and "$imgPath = $args[0]" in script


def _arbiter_reading(monkeypatch, tmp_path, temperature: float) -> object:
    """The shared GPU arbiter sees one NVIDIA card at ``temperature`` with 8 GB free; the lease lives in tmp_path."""
    from pilot.system import gpu_arbiter

    monkeypatch.setenv("HELIOX_GPU_LOCK_DIR", str(tmp_path))
    card = gpu_arbiter.GpuInfo(0, "Fake RTX", temperature, 8 * 1024**3, 8 * 1024**3)
    fake = gpu_arbiter.GpuArbiter(
        read=lambda: gpu_arbiter.MachineReading("nvidia", (card,), 8 * 1024**3, 16 * 1024**3),
        holder=lambda: gpu_arbiter.lease_holder(tmp_path),
    )
    monkeypatch.setattr(gpu_arbiter, "_shared", fake)
    return fake


async def test_on_demand_gpu_read_releases_the_cuda_cache(screen, monkeypatch, tmp_path):
    released: list[bool] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: released.append(True))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _arbiter_reading(monkeypatch, tmp_path, 60.0)

    result = await vision.screen_find_text("Delete")

    assert "Delete files" in result
    assert screen.gpu_requests == [True]
    assert released == [True]
    # No GPU reader is kept between reads, and the GPU lease is free again.
    assert not [key for key in vars(vision._ocr_easyocr) if key.endswith("_gpu")]
    from pilot.system.gpu_arbiter import lease_holder

    assert lease_holder(tmp_path) is None


async def test_on_demand_read_uses_the_cpu_when_the_card_is_hot(screen, monkeypatch, tmp_path):
    _arbiter_reading(monkeypatch, tmp_path, 88.0)

    assert "Delete files" in await vision.screen_find_text("Delete")
    assert screen.gpu_requests == [False]


async def test_a_cuda_error_on_the_gpu_read_is_a_fault_and_the_read_repeats_on_the_cpu(screen, monkeypatch, tmp_path):
    arbiter = _arbiter_reading(monkeypatch, tmp_path, 60.0)
    reader = screen.Reader

    def flaky_reader(langs, gpu=True, verbose=True):
        made = reader(langs, gpu=gpu, verbose=verbose)
        if gpu:

            def fail(_image):
                raise RuntimeError("CUDA error: an illegal memory access was encountered")

            return SimpleNamespace(readtext=fail)
        return made

    monkeypatch.setattr(screen, "Reader", flaky_reader)

    assert "Delete files" in await vision.screen_find_text("Delete")
    assert screen.gpu_requests == [True, False]
    assert arbiter.last_fault("ocr").message.startswith("CUDA error")
