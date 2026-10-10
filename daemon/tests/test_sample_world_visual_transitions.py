"""Visual transition sampler accepts only adjacent, pinned observations."""

import pytest

from scripts.sample_world_visual_transitions import _image_feature, _pair, _splits


def _entry(index: int, episode: str = "task-a", action: int = 0) -> dict:
    return {
        "row_idx": index,
        "truncated_cells": [],
        "row": {
            "episode_id": episode,
            "step_idx": index,
            "action_type": action,
            "click_type": 1,
            "mouse_x": 0.25,
            "mouse_y": 0.75,
        },
    }


def test_only_adjacent_mouse_steps_in_same_episode_are_accepted() -> None:
    pair, reason = _pair(_entry(3), _entry(4))
    assert reason == "accepted"
    assert pair["action"] == 1
    assert pair["coords"] == [0.25, 0.75]
    assert _pair(_entry(3), _entry(4, episode="task-b"))[1] == "different_episode"
    assert _pair(_entry(3), _entry(5))[1] == "nonadjacent_step"
    assert _pair(_entry(3, action=1), _entry(4))[1] == "unsupported_action"
    bad_coordinate = _entry(3)
    bad_coordinate["row"]["mouse_x"] = float("nan")
    assert _pair(bad_coordinate, _entry(4))[1] == "invalid_coordinates"
    bad_click = _entry(3)
    bad_click["row"]["click_type"] = True
    assert _pair(bad_click, _entry(4))[1] == "unsupported_action"


def test_episode_split_keeps_three_partitions_nonempty() -> None:
    assignments = _splits({"first", "second", "third"}, seed=5)
    assert set(assignments.values()) == {"train", "validation", "eval"}


def test_image_fetch_rejects_unpinned_or_foreign_source() -> None:
    with pytest.raises(ValueError, match="pinned dataset revision"):
        _image_feature({"row": {"image": {"src": "https://example.com/image.webp"}}}, "a" * 40)
    with pytest.raises(ValueError, match="pinned dataset revision"):
        _image_feature(
            {"row": {"image": {"src": "https://datasets-server.huggingface.co/assets/" + "b" * 40 + "/image.webp"}}},
            "a" * 40,
        )
