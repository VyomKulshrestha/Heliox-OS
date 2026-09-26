"""Controlled world-transition source stays local and layout-disjoint."""

import io
import random

import numpy as np
import pytest
from PIL import Image

from scripts.collect_controlled_ui_transitions import (
    _collect_counterfactual_pair,
    _features,
    _page_html,
    _positions,
)


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


def test_counterfactual_pair_restarts_from_one_state() -> None:
    class Page:
        def __init__(self):
            self.selected = None
            self.mouse = self

        def set_content(self, html, *, wait_until):
            assert "panel-0" in html and wait_until == "load"
            self.selected = None

        def screenshot(self):
            image = Image.new("RGB", (640, 480), color="white")
            if self.selected is not None:
                from PIL import ImageDraw

                x = 30 if self.selected == 0 else 390
                ImageDraw.Draw(image).rectangle((x, 100, x + 110, 190), fill="black")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()

        def click(self, x, y):
            self.selected = int(x > 320)

        def locator(self, selector):
            page = self

            class Locator:
                def count(self):
                    return int(page.selected is not None)

                def is_visible(self):
                    return selector == f"#panel-{page.selected}"

            return Locator()

    rows = _collect_counterfactual_pair(Page(), 0, 0, random.Random(7))
    assert len(rows) == 2
    assert rows[0]["group"] == rows[1]["group"]
    assert np.array_equal(rows[0]["state"], rows[1]["state"])
    assert not np.array_equal(rows[0]["target"], rows[1]["target"])
    assert rows[0]["coords"] != rows[1]["coords"]
