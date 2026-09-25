"""Controlled world-transition source stays local and layout-disjoint."""

import io
import random

import numpy as np
import pytest
from PIL import Image

from scripts.collect_controlled_ui_transitions import _features, _page_html, _positions


def test_holdout_layouts_have_distinct_geometry() -> None:
    validation = _positions(4, random.Random(1))
    evaluation = _positions(5, random.Random(1))
    assert all(x >= 335 for x, _ in validation)
    assert evaluation[0][0] < 185 < evaluation[1][0]
    assert validation[0][1] < 90 < validation[1][1]
    assert "panel-0" in _page_html(4, validation)
    assert "https://" not in _page_html(4, validation)
    with pytest.raises(ValueError, match="Unknown controlled layout"):
        _positions(6, random.Random(1))


def test_collector_uses_live_world_feature_shape_without_image_storage() -> None:
    image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    feature = _features(buffer.getvalue())
    assert feature.shape == (256,)
    assert feature.dtype == np.float32
    assert np.all(feature == 1)
