"""Action Preview — "simulate before executing" for autonomous background
tasks (see `pilot.agents.narrator.ExecutionNarrator.on_action_preview` and
`config.PreviewConfig`).

Renders a real screenshot with the action's target UI element highlighted,
using the VLM element-detection pipeline this codebase already has
(`pilot.system.vision.screen_detect_elements`, cloud Gemini/OpenAI/Claude or
fully local/offline Ollama, zero new dependency) — never a generated image.
A truly generative "predict the future screen" model (in the spirit of
World Labs' RTFM, NVIDIA Cosmos, or Google DeepMind's Genie) was
deliberately not pursued here: see SECURITY.md's Pre-Execution Target
Assessment section for why — every self-hostable option in that space
needs dedicated GPU hardware and targets photorealistic video generation,
not "will this action hit the button I expect."

For browser actions specifically, this also pairs the annotated screenshot
with `pilot.system.dom_diff.dry_run_action()`'s real, measured before/after
DOM diff from an isolated scratch tab — an actual dry run, not a guess.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pilot.system.dom_diff import BROWSER_TARGET_ACTION_TYPES
from pilot.system.vision import _capture_screenshot_bytes, screen_detect_elements

if TYPE_CHECKING:
    from pilot.actions import Action

logger = logging.getLogger("pilot.system.action_preview")


@dataclass
class ActionPreview:
    """Everything the frontend needs to render a preview of a proposed
    action's on-screen effect, before it actually runs."""

    screenshot_base64: str | None
    bbox: dict[str, float] | None  # pixel {"x","y","w","h"} relative to the screenshot image
    target_label: str | None
    caption: str
    dom_diff: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "screenshot_base64": self.screenshot_base64,
            "bbox": self.bbox,
            "target_label": self.target_label,
            "caption": self.caption,
            "dom_diff": self.dom_diff,
        }


def _describe_target(action_type: str, action: Any) -> str:
    target = getattr(action, "target", "") or ""
    verb = action_type.replace("_", " ")
    return f"About to {verb}: {target}" if target else f"About to {verb}"


async def generate_action_preview(action_type: str, action: Action) -> ActionPreview | None:
    """Best-effort: any failure returns None rather than raising, so a
    preview problem can never block or break real execution."""
    try:
        img_bytes = await _capture_screenshot_bytes()
        screenshot_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        logger.warning("action_preview: screenshot capture failed", exc_info=True)
        return None

    target = getattr(action, "target", "") or ""
    bbox: dict[str, float] | None = None
    label: str | None = None
    try:
        raw = await screen_detect_elements(description=target, max_elements=5)
        data = json.loads(raw)
        elements = data.get("elements") or []
        if elements:
            best = elements[0]
            x, y, w, h = best["bbox"]
            bbox = {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
            label = best.get("label")
    except Exception:
        logger.debug("action_preview: element detection unavailable", exc_info=True)

    dom_diff_dict: dict[str, Any] | None = None
    if action_type in BROWSER_TARGET_ACTION_TYPES:
        try:
            from pilot.system.dom_diff import dry_run_action

            diff = await dry_run_action(action_type, action)
            if diff is not None:
                dom_diff_dict = diff.to_dict()
        except Exception:
            logger.debug("action_preview: dry-run diff unavailable", exc_info=True)

    return ActionPreview(
        screenshot_base64=screenshot_b64,
        bbox=bbox,
        target_label=label,
        caption=_describe_target(action_type, action),
        dom_diff=dom_diff_dict,
    )
