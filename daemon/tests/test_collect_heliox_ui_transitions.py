"""Local Heliox UI transition collection stays isolated and feature-only."""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.collect_heliox_ui_transitions import (
    MIN_FREE_BYTES,
    _features,
    require_free_space,
    validate_url,
    validate_viewport,
)


def test_held_out_viewports_are_bounded() -> None:
    assert validate_viewport(1680, 960) == (1680, 960)
    for width, height in ((0, 960), (1680, 0), (5000, 960), (1680, 5000)):
        with pytest.raises(ValueError, match="viewport dimensions"):
            validate_viewport(width, height)


@pytest.mark.parametrize("url", ["http://127.0.0.1:1420/", "http://localhost:5173/"])
def test_accepts_only_local_browser_development_urls(url: str) -> None:
    assert validate_url(url) == url


@pytest.mark.parametrize("url", [
    "https://example.com/", "http://example.com/", "http://127.0.0.1:1420/?token=secret",
    "http://user:pass@localhost:1420/", "file:///private/screen.html",
])
def test_rejects_remote_or_sensitive_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_url(url)


def test_features_are_live_model_shape_and_do_not_write_screenshots() -> None:
    image = Image.new("RGB", (1280, 720), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    class Page:
        def screenshot(self, *, animations):
            assert animations == "disabled"
            return buffer.getvalue()

    values = _features(Page())
    assert values.shape == (256,)
    assert values.dtype == np.float32
    assert np.all(values == 1)


def test_collection_stops_before_disk_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.collect_heliox_ui_transitions.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=MIN_FREE_BYTES - 1))
    with pytest.raises(RuntimeError, match="512 MiB"):
        require_free_space(Path("."))
