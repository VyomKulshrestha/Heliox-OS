from __future__ import annotations

import asyncio
import sys
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from pilot.system import vision


@pytest.mark.asyncio
async def test_missing_easyocr_does_not_install_packages(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "easyocr", None)
    install = AsyncMock()
    monkeypatch.setattr(vision, "run_command", install)

    with pytest.raises(ModuleNotFoundError):
        await vision._ocr_easyocr(b"", "eng")

    install.assert_not_awaited()


def test_resize_png_for_vision_bounds_largest_dimension() -> None:
    source = BytesIO()
    Image.new("RGB", (3200, 1800), color="navy").save(source, format="PNG")

    bounded = vision._resize_png_for_vision(source.getvalue(), 1600)

    with Image.open(BytesIO(bounded)) as image:
        assert image.size == (1600, 900)


@pytest.mark.asyncio
async def test_screen_analyze_falls_back_when_cloud_exceeds_latency_cap(monkeypatch) -> None:
    class FakeVault:
        def __init__(self, _config) -> None:
            pass

        async def get_key(self, _provider: str) -> str:
            return "test-key"

    config = SimpleNamespace(
        model=SimpleNamespace(
            provider="cloud",
            cloud_provider="gemini",
            cloud_model="",
            ollama_base_url="http://127.0.0.1:11434",
        )
    )

    async def cloud_timeout(*_args, **_kwargs):
        raise TimeoutError

    async def no_local_model(*_args, **_kwargs):
        return None

    async def capture(*_args, **_kwargs):
        return b"bounded-image"

    async def ocr(*_args, **_kwargs):
        return "visible screen text"

    monkeypatch.setattr(vision.PilotConfig, "load", staticmethod(lambda: config))
    monkeypatch.setattr("pilot.security.vault.KeyVault", FakeVault)
    monkeypatch.setattr(vision, "_bounded_cloud_vision", cloud_timeout)
    monkeypatch.setattr(vision, "_bounded_local_vision", no_local_model)
    monkeypatch.setattr(vision, "_capture_screenshot_bytes", capture)
    monkeypatch.setattr(vision, "screen_ocr", ocr)

    result = await vision.screen_analyze("What is visible?")

    assert "falling back to OCR" in result
    assert "visible screen text" in result


@pytest.mark.asyncio
async def test_screen_analyze_enforces_one_total_provider_deadline(monkeypatch) -> None:
    async def slow_provider(*_args, **_kwargs):
        await asyncio.sleep(1)
        return "late"

    async def capture(*_args, **_kwargs):
        return b"bounded-image"

    async def foreground_fallback(_detail: str):
        return "[Semantic screen analysis unavailable.] Verified foreground application: Codex."

    config = SimpleNamespace(
        model=SimpleNamespace(
            provider="local",
            cloud_provider="",
            cloud_model="",
            ollama_base_url="http://127.0.0.1:11434",
        )
    )
    monkeypatch.setattr(vision.PilotConfig, "load", staticmethod(lambda: config))
    monkeypatch.setattr(vision, "_bounded_local_vision", slow_provider)
    monkeypatch.setattr(vision, "_capture_screenshot_bytes", capture)
    monkeypatch.setattr(vision, "screen_ocr", slow_provider)
    monkeypatch.setattr(vision, "_foreground_screen_fallback", foreground_fallback)
    monkeypatch.setattr(vision, "VISION_CLOUD_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(vision, "VISION_OCR_TIMEOUT_SECONDS", 0.01)

    started = time.perf_counter()
    result = await vision.screen_analyze("What is visible?")

    assert time.perf_counter() - started < 0.2
    assert "Semantic screen analysis unavailable" in result
    assert "Codex" in result
