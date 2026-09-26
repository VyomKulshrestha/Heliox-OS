"""Reward examples must be observed from one measured starting state."""

import numpy as np
import pytest

from scripts import collect_miniwob_reward_actions as collector


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
    monkeypatch.setattr(collector, "candidate_actions", lambda _: [
        {"x": 10, "y": 20, "label": "ONE"}, {"x": 30, "y": 40, "label": "TWO"},
    ])
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
    monkeypatch.setattr(
        collector, "visual_feature", lambda png: np.full(256, png[0], dtype=np.float32)
    )
    monkeypatch.setattr(collector, "candidate_actions", lambda _: [
        {"x": 10, "y": 20}, {"x": 30, "y": 40},
    ])
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
