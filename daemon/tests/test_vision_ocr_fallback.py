"""screen_ocr engine order: the slow EasyOCR fallback is opt-out (periodic supervision)."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

import pilot.system.vision as vision


async def _run(native, tesseract, easy, **kwargs):
    with ExitStack() as stack:
        stack.enter_context(patch.object(vision, "_capture_screenshot_bytes", AsyncMock(return_value=b"png")))
        stack.enter_context(patch.object(vision, "_ocr_windows_native", native))
        stack.enter_context(patch.object(vision, "_ocr_macos_native", native))
        stack.enter_context(patch.object(vision, "_ocr_tesseract_cli", tesseract))
        stack.enter_context(patch.object(vision, "_ocr_easyocr", easy))
        stack.enter_context(patch.object(vision, "_ocr_pytesseract", AsyncMock(side_effect=ImportError)))
        return await vision.screen_ocr(**kwargs)


@pytest.mark.asyncio
async def test_default_still_falls_back_to_easyocr():
    native = AsyncMock(side_effect=RuntimeError("native down"))
    tess = AsyncMock(side_effect=RuntimeError("no tesseract"))
    easy = AsyncMock(return_value="from easyocr")
    assert await _run(native, tess, easy) == "from easyocr"
    easy.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_slow_fallback_raises_and_never_calls_easyocr():
    native = AsyncMock(side_effect=RuntimeError("native down"))
    tess = AsyncMock(side_effect=RuntimeError("no tesseract"))
    easy = AsyncMock(return_value="from easyocr")
    with pytest.raises(vision.OCRUnavailableError):
        await _run(native, tess, easy, allow_slow_fallback=False)
    easy.assert_not_called()


@pytest.mark.asyncio
async def test_no_slow_fallback_empty_tesseract_is_a_clean_empty_read():
    native = AsyncMock(side_effect=RuntimeError("native down"))
    tess = AsyncMock(return_value="")
    easy = AsyncMock(return_value="from easyocr")
    assert await _run(native, tess, easy, allow_slow_fallback=False) == ""
    easy.assert_not_called()
