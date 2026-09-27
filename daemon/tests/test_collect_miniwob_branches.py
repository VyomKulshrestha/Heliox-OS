"""MiniWoB branch collection refuses unsafe assets and unmatched resets."""

import io

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

from scripts import collect_miniwob_branches as collector


def test_candidate_centers_include_visible_pointer_affordances() -> None:
    with sync_playwright() as playwright, playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport=collector.VIEWPORT)
        page.set_content("""
            <div id="wrap">
              <button style="position:absolute;left:10px;top:10px;width:40px;height:20px">
                <span style="cursor:pointer">Native</span>
              </button>
              <span style="position:absolute;left:80px;top:10px;width:40px;height:20px;cursor:pointer">Custom</span>
              <span style="position:absolute;left:140px;top:10px;width:40px;height:20px">Static</span>
              <span style="display:none;cursor:pointer">Hidden</span>
            </div>
        """)
        assert collector.candidate_centers(page) == [(30.0, 20.0), (100.0, 20.0)]


def test_sparse_controls_add_painted_targets_and_neutral_probes() -> None:
    with sync_playwright() as playwright, playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport=collector.VIEWPORT)
        page.set_content("""
            <style>body { margin: 0 }</style>
            <div id="wrap" style="position:relative;width:320px;height:360px">
              <button style="position:absolute;left:10px;top:10px;width:40px;height:20px">Go</button>
              <div style="position:absolute;left:80px;top:10px;width:50px;height:50px;
                          background:red"></div>
            </div>
        """)
        centers = collector.candidate_centers(page)
        assert centers[0] == (30.0, 20.0)
        assert (105.0, 35.0) in centers
        assert (40.0, 45.0) in centers


def test_visible_semantics_excludes_hidden_text_and_bounds_control_label() -> None:
    with sync_playwright() as playwright, playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport=collector.VIEWPORT)
        page.set_content("""
            <div id="wrap">
              <button aria-label="Save item" style="position:absolute;left:10px;top:10px">
                Save
              </button>
              <span style="display:none">hidden source</span>
            </div>
        """)
        result = collector.visible_semantics(page, (30, 20))
        assert result["target"] == "Save item"
        assert "Save" in result["text"]
        assert "hidden source" not in result["text"]


def test_visual_feature_resolution_is_bounded_and_default_is_unchanged() -> None:
    image = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    png = stream.getvalue()
    assert collector.visual_feature(png).shape == (256,)
    assert collector.visual_feature(png, 64).shape == (4096,)
    assert np.allclose(collector.visual_feature(png, 64), 128 / 255)
    with pytest.raises(ValueError, match="16x16 or 64x64"):
        collector.visual_feature(png, 32)


def test_tasks_and_assets_are_bounded_to_the_pinned_virtual_origin() -> None:
    assert collector.parse_tasks("click-button, click-widget") == ("click-button", "click-widget")
    assert collector.asset_path("https://miniwob.local/miniwob/html/core/core.js") == "miniwob/html/core/core.js"
    for tasks in ("", "click-button,click-button", "../secret", "click_button"):
        with pytest.raises(ValueError):
            collector.parse_tasks(tasks)
    for url in (
        "https://example.com/miniwob/html/core/core.js",
        "https://miniwob.local/miniwob/html/../secret",
        "https://miniwob.local/miniwob/html/core/core.js?version=latest",
        "https://miniwob.local/miniwob/html/%2e%2e/secret",
    ):
        with pytest.raises(ValueError):
            collector.asset_path(url)
    with pytest.raises(ValueError, match="full SHA-1"):
        collector.PinnedAssets("main")


class _Context:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Page:
    def __init__(self) -> None:
        self.mouse = self
        self.x = 0

    def click(self, x: float, _y: float) -> None:
        self.x = x

    def evaluate(self, _script: str) -> dict:
        return {"done": True, "reward": 1 if self.x == 10 else -1}

    def screenshot(self, **_kwargs) -> bytes:
        return bytes([int(self.x)])


def test_branch_pairs_require_identical_prestate_and_distinct_outcome(monkeypatch) -> None:
    contexts = []

    def reset(*_args):
        context = _Context()
        contexts.append(context)
        return context, _Page(), [(10, 20), (30, 40)], b"before", "same DOM"

    monkeypatch.setattr(collector, "_reset", reset)
    monkeypatch.setattr(
        collector, "visual_feature", lambda png, size=16: np.full(size * size, png[0], dtype=np.float32)
    )
    pair, attempted = collector.collect_group(None, None, "task", "seed", 2)
    assert attempted == 2
    assert len(pair) == 2
    assert np.array_equal(pair[0]["state"], pair[1]["state"])
    assert not np.array_equal(pair[0]["target"], pair[1]["target"])
    assert [row["reward"] for row in pair] == [1, -1]
    assert all(context.closed for context in contexts)

    monkeypatch.setattr(collector, "visible_semantics", lambda page, point=None: {
        "text": "after" if point is None else "before", "target": "click"})
    pair, _ = collector.collect_group(None, None, "task", "seed", 2, include_semantics=True)
    assert all(row["state_text"] == "before" and row["target_text"] == "after"
               and row["action_label"] == "click" for row in pair)

    def changed_reset(*_args):
        context = _Context()
        contexts.append(context)
        return context, _Page(), [(10, 20), (30, 40)], bytes([len(contexts)]), "same DOM"

    monkeypatch.setattr(collector, "_reset", changed_reset)
    with pytest.raises(collector.UnstableResetError, match="exact measured state"):
        collector.collect_group(None, None, "task", "seed", 2)
    assert all(context.closed for context in contexts)


def test_task_splits_reject_identical_modeled_prestates() -> None:
    rows = [
        {"split": split, "state": np.full(256, value, dtype=np.float32)}
        for split, value in (("train", 0), ("validation", 1), ("eval", 2))
    ]
    collector.require_disjoint_prestates(rows)
    rows[-1]["state"] = rows[0]["state"].copy()
    with pytest.raises(ValueError, match="leaked across task splits"):
        collector.require_disjoint_prestates(rows)
