from __future__ import annotations

import pytest

from pilot.agents.screen_vision import ScreenVisionAgent
from pilot.system import vision as system_vision


def test_parse_gui_target_response_extracts_actionable_center_coordinates():
    target = ScreenVisionAgent.parse_gui_target_response(
        """
        ```json
        {"action":"click","label":"Save","bbox":{"x":40,"y":80,"width":120,"height":32},"confidence":0.91}
        ```
        """,
        screen_width=300,
        screen_height=200,
    )

    assert target.action == "click"
    assert target.label == "Save"
    assert target.x == 100
    assert target.y == 96
    assert target.confidence == pytest.approx(0.91)


def test_parse_gui_target_response_clips_bounds_and_confidence():
    target = ScreenVisionAgent.parse_gui_target_response(
        '{"action":"type","label":"Search","bbox":{"x":190,"y":95,"w":80,"h":20},"confidence":4}',
        screen_width=200,
        screen_height=100,
    )

    assert target.bounding_box.x == 190
    assert target.bounding_box.y == 95
    assert target.bounding_box.width == 10
    assert target.bounding_box.height == 5
    assert target.confidence == 1.0


@pytest.mark.asyncio
async def test_locate_gui_target_uses_screen_analyze_prompt(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_screen_analyze(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"action":"click","label":"Run","bbox":{"x":10,"y":20,"width":20,"height":10},"confidence":0.8}'

    monkeypatch.setattr(system_vision, "screen_analyze", fake_screen_analyze)

    target = await ScreenVisionAgent().locate_gui_target(
        "click the Run button",
        screen_width=100,
        screen_height=80,
    )

    assert "click the Run button" in captured["prompt"]
    assert "100x80 screen" in captured["prompt"]
    assert target.to_dict()["x"] == 20
    assert target.to_dict()["y"] == 25
