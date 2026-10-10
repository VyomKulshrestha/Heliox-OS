"""Reward examples must be observed from one measured starting state."""

import numpy as np
import pytest

from scripts import collect_miniwob_reward_actions as collector

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # optional extra: pip install -e ".[browser]" && playwright install chromium
    sync_playwright = None

needs_browser = pytest.mark.skipif(sync_playwright is None, reason="needs the optional browser extra")


@needs_browser
def test_visible_candidates_include_tabs_and_popup_hints_without_hidden_targets() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 320, "height": 360})
            page.set_content("""
                <div id="query">Click the <span class="ui-icon-stop"></span> icon.</div>
                <div id="area">
                  <div role="tab" style="width:100px;height:30px">Section 1</div>
                  <ul role="menu" style="cursor:pointer;width:100px;height:30px">
                    <li role="menuitem" aria-haspopup="true"
                        style="cursor:pointer;width:100px;height:30px">
                      Playback<span hidden>Hidden target</span>
                      <span class="ui-icon ui-icon-play"></span>
                    </li>
                  </ul>
                  <button aria-disabled="true">Disabled</button>
                </div>
            """)
            candidates = collector.candidate_actions(page)
            assert any(row["role"] == "tab" and row["label"] == "Section 1" for row in candidates)
            assert any(
                row["role"] == "menuitem"
                and row["has_popup"] == "true"
                and row["icon_hint"] == "ui-icon ui-icon-play"
                and "Hidden target" not in row["label"]
                for row in candidates
            )
            assert all(row["role"] != "menu" and row["label"] != "Disabled" for row in candidates)
            assert "ui-icon-stop" in collector.goal_description(page)
        finally:
            browser.close()


class _Context:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Page:
    def __init__(self) -> None:
        self.mouse = self
        self.x = 0

    def locator(self, _selector):
        return self

    def inner_text(self) -> str:
        return "Click ONE"

    def click(self, x, _y) -> None:
        self.x = x

    def evaluate(self, _script) -> dict:
        return {"done": True, "reward": 1 if self.x == 10 else -1}


def test_collects_contrastive_terminal_actions_from_fresh_resets(monkeypatch) -> None:
    contexts = []

    def reset(*_args):
        context = _Context()
        contexts.append(context)
        return context, _Page(), [], b"same", "same DOM"

    monkeypatch.setattr(collector, "_reset", reset)
    monkeypatch.setattr(collector, "visual_feature", lambda _: np.zeros(256, dtype=np.float32))
    monkeypatch.setattr(collector, "goal_description", lambda _: "Click ONE")
    monkeypatch.setattr(
        collector,
        "candidate_actions",
        lambda _: [
            {"x": 10, "y": 20, "label": "ONE"},
            {"x": 30, "y": 40, "label": "TWO"},
        ],
    )
    rows, attempted = collector.collect_group(None, None, "task", "seed", 2)
    assert attempted == 2
    assert [row["reward"] for row in rows] == [1, -1]
    assert rows[0]["goal"] == "Click ONE"
    assert rows[0]["prestate_sha256"] == rows[1]["prestate_sha256"]
    assert all(context.closed for context in contexts)


def test_rejects_mismatched_reset_and_cross_split_state(monkeypatch) -> None:
    contexts = []

    def reset(*_args):
        context = _Context()
        contexts.append(context)
        return context, _Page(), [], bytes([len(contexts)]), "same DOM"

    monkeypatch.setattr(collector, "_reset", reset)
    monkeypatch.setattr(collector, "visual_feature", lambda png: np.full(256, png[0], dtype=np.float32))
    monkeypatch.setattr(collector, "goal_description", lambda _: "Click ONE")
    monkeypatch.setattr(
        collector,
        "candidate_actions",
        lambda _: [
            {"x": 10, "y": 20},
            {"x": 30, "y": 40},
        ],
    )
    with pytest.raises(ValueError, match="exact measured state"):
        collector.collect_group(None, None, "task", "seed", 2)
    assert all(context.closed for context in contexts)

    rows = [
        {"split": "train", "prestate_sha256": "a"},
        {"split": "validation", "prestate_sha256": "b"},
        {"split": "eval", "prestate_sha256": "c"},
    ]
    collector.require_disjoint_prestates(rows)
    rows[-1]["prestate_sha256"] = "a"
    with pytest.raises(ValueError, match="leaked across task splits"):
        collector.require_disjoint_prestates(rows)
