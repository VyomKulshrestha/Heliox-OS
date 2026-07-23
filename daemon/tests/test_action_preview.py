"""Tests for pilot.system.action_preview (the "simulate before executing"
preview generator for autonomous background tasks).

Mocks at the action_preview module boundary (its own imported names, not
the origin module -- action_preview.py does `from pilot.system.vision
import X`, which binds a new name in its own namespace) since screenshot
capture and VLM element detection are the two real external calls this
module makes.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from pilot.actions import Action, ActionType, BrowserParams, EmptyParams
from pilot.system.action_preview import ActionPreview, generate_action_preview


def _elements_json(elements: list[dict]) -> str:
    return json.dumps({"elements": elements, "count": len(elements), "source": "test", "note": ""})


class TestScreenshotFailure:
    @pytest.mark.asyncio
    async def test_screenshot_capture_failure_returns_none(self):
        action = Action(action_type=ActionType.FILE_WRITE, target="x", parameters=EmptyParams())
        with patch(
            "pilot.system.action_preview._capture_screenshot_bytes",
            new=AsyncMock(side_effect=RuntimeError("no display")),
        ):
            result = await generate_action_preview("file_write", action)
        assert result is None


class TestNonBrowserAction:
    @pytest.mark.asyncio
    async def test_non_browser_action_gets_screenshot_without_dom_diff(self):
        action = Action(action_type=ActionType.FILE_DELETE, target="C:/tmp/x.txt", parameters=EmptyParams())
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json([])),
            ),
        ):
            result = await generate_action_preview("file_delete", action)

        assert isinstance(result, ActionPreview)
        assert result.screenshot_base64  # non-empty base64 string
        assert result.dom_diff is None  # never attempted for non-browser actions
        assert "file delete" in result.caption
        assert "C:/tmp/x.txt" in result.caption


class TestElementDetection:
    @pytest.mark.asyncio
    async def test_matched_element_sets_bbox_and_label(self):
        action = Action(
            action_type=ActionType.BROWSER_CLICK, target="Submit button", parameters=BrowserParams(selector="#submit")
        )
        elements = [
            {"label": "Submit", "type": "button", "action": "click", "bbox": [10, 20, 80, 30], "confidence": 0.9}
        ]
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json(elements)),
            ),
            patch("pilot.system.dom_diff.dry_run_action", new=AsyncMock(return_value=None)),
        ):
            result = await generate_action_preview("browser_click", action)

        assert result.bbox == {"x": 10.0, "y": 20.0, "w": 80.0, "h": 30.0}
        assert result.target_label == "Submit"

    @pytest.mark.asyncio
    async def test_no_elements_detected_still_returns_screenshot(self):
        action = Action(action_type=ActionType.BROWSER_CLICK, target="x", parameters=BrowserParams(selector="#x"))
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json([])),
            ),
            patch("pilot.system.dom_diff.dry_run_action", new=AsyncMock(return_value=None)),
        ):
            result = await generate_action_preview("browser_click", action)

        assert result is not None
        assert result.screenshot_base64
        assert result.bbox is None
        assert result.target_label is None

    @pytest.mark.asyncio
    async def test_element_detection_exception_degrades_gracefully(self):
        """A VLM/detection failure must never take down the whole preview --
        the screenshot alone is still useful."""
        action = Action(action_type=ActionType.FILE_WRITE, target="x", parameters=EmptyParams())
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(side_effect=RuntimeError("no vision provider configured")),
            ),
        ):
            result = await generate_action_preview("file_write", action)

        assert result is not None
        assert result.bbox is None


class TestBrowserDryRunDiffWiring:
    @pytest.mark.asyncio
    async def test_browser_action_type_calls_dry_run_action(self):
        action = Action(action_type=ActionType.BROWSER_CLICK, target="x", parameters=BrowserParams(selector="#x"))
        fake_diff = AsyncMock(return_value=None)
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json([])),
            ),
            patch("pilot.system.dom_diff.dry_run_action", new=fake_diff),
        ):
            await generate_action_preview("browser_click", action)

        fake_diff.assert_awaited_once_with("browser_click", action)

    @pytest.mark.asyncio
    async def test_dry_run_result_is_included_as_dict(self):
        from pilot.system.dom_diff import DomDiff

        action = Action(action_type=ActionType.BROWSER_CLICK, target="x", parameters=BrowserParams(selector="#x"))
        diff = DomDiff(change_score=0.42)
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json([])),
            ),
            patch("pilot.system.dom_diff.dry_run_action", new=AsyncMock(return_value=diff)),
        ):
            result = await generate_action_preview("browser_click", action)

        assert result.dom_diff is not None
        assert result.dom_diff["change_score"] == 0.42

    @pytest.mark.asyncio
    async def test_dry_run_failure_degrades_gracefully(self):
        action = Action(action_type=ActionType.BROWSER_CLICK, target="x", parameters=BrowserParams(selector="#x"))
        with (
            patch("pilot.system.action_preview._capture_screenshot_bytes", new=AsyncMock(return_value=b"fakepng")),
            patch(
                "pilot.system.action_preview.screen_detect_elements",
                new=AsyncMock(return_value=_elements_json([])),
            ),
            patch("pilot.system.dom_diff.dry_run_action", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            result = await generate_action_preview("browser_click", action)

        assert result is not None  # the screenshot/bbox portion still succeeds
        assert result.dom_diff is None


class TestActionPreviewToDict:
    def test_to_dict_round_trips_all_fields(self):
        preview = ActionPreview(
            screenshot_base64="abc",
            bbox={"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
            target_label="Save",
            caption="About to click: Save",
            dom_diff={"change_score": 0.1},
        )
        d = preview.to_dict()
        assert d == {
            "screenshot_base64": "abc",
            "bbox": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0},
            "target_label": "Save",
            "caption": "About to click: Save",
            "dom_diff": {"change_score": 0.1},
        }
