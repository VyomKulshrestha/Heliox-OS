"""Parser and action-binding contracts for the bounded WebWorld sampler."""

import io
import urllib.error
from collections import Counter

import numpy as np
import pytest

from scripts.sample_webworld_transitions import (
    _features,
    _load_checkpoint,
    _read_json,
    _save_checkpoint,
    _split_by_shared_state,
    _state_connected_labels,
    _state_digests,
    bound_action,
    bound_action_path,
    parse_trajectory,
)


def test_parse_multistep_trajectory_uses_previous_observed_state() -> None:
    turns = [
        {
            "from": "human",
            "value": "Initial Page State:\nRootWebArea 'Start'\n  [17] button 'Open'\n"
            "First Action: 'click('17')'\n\nNext Page State:",
        },
        {"from": "gpt", "value": "RootWebArea 'Open'\n  [19] textbox 'Search'"},
        {"from": "human", "value": "Continue the trajectory.\nAction: 'fill('19', 'tea')'\nNext Page State:"},
        {"from": "gpt", "value": "RootWebArea 'Open'\n  [19] textbox 'Search' value='tea'"},
    ]

    first, second = parse_trajectory(turns)

    assert first == (
        "RootWebArea 'Start'\n  [17] button 'Open'",
        "click('17')",
        "RootWebArea 'Open'\n  [19] textbox 'Search'",
    )
    assert second[0] == first[2]
    assert second[1] == "fill('19', 'tea')"


def test_bind_action_uses_pre_action_node_not_a_numeric_id() -> None:
    tree = "RootWebArea 'Shop'\n  [7] button 'Checkout'\n  [8] textbox 'Search'"

    assert bound_action(tree, "click('7')") == "click target button 'Checkout'"
    assert bound_action(tree, "fill([8], 'tea')") == "fill target textbox 'Search' arguments 'tea'"
    assert bound_action(tree, "click('99')") == "click('99')"


def test_path_binding_adds_pre_action_ancestors_not_siblings() -> None:
    tree = (
        "RootWebArea 'Shop'\n"
        "  [1] navigation 'Main'\n"
        "    [2] link 'Other'\n"
        "  [3] dialog 'Cart'\n"
        "    [4] list 'Items'\n"
        "      [5] button 'Checkout'"
    )
    path = bound_action_path(tree, "click('5')")
    assert path.startswith("click target button 'Checkout' hierarchy ")
    assert "ancestor_0_list" in path
    assert "ancestor_1_dialog" in path
    assert "ancestor_2_rootwebarea" in path
    assert "ancestor_0_name_items" in path
    assert "navigation" not in path
    assert bound_action_path(tree, "click('99')") == "click('99')"


def test_malformed_trajectory_is_rejected_without_partial_rows() -> None:
    turns = [
        {"from": "human", "value": "Initial Page State:\nTree\nFirst Action: 'click(1)'\nNext Page State:"},
        {"from": "gpt", "value": "Next"},
        {"from": "human", "value": "no action marker"},
        {"from": "gpt", "value": "Other"},
    ]

    assert parse_trajectory(turns) == []


def test_direct_navigation_type_is_distinct_from_in_page_action() -> None:
    assert not np.array_equal(_features(["goto"], dimension=256)[0], _features(["click"], dimension=256)[0])


def test_exact_state_identity_is_stable_without_storing_page_text() -> None:
    digests = _state_digests(["Page A", "Page A", "Page B"])
    assert digests.shape == (3, 32)
    assert digests.dtype == np.uint8
    assert np.array_equal(digests[0], digests[1])
    assert not np.array_equal(digests[0], digests[2])


def test_split_keeps_transitive_shared_states_together() -> None:
    states = np.eye(1024, dtype=np.float32)[[0, 1, 1, 2, 2, 3, 4, 5]]
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 5])
    splits, components = _split_by_shared_state(states, groups, 7)
    labels = _state_connected_labels(states, groups)
    assert components == 3
    assert len(set(splits[:6])) == 1
    assert len(set(labels[:6])) == 1
    assert set(splits) == {"train", "validation", "eval"}


def test_checkpoint_resumes_only_the_same_revision_and_sample(tmp_path) -> None:
    path = tmp_path / "features.partial.npz"
    arrays = {
        "state": np.zeros((1, 1024), dtype=np.float32),
        "target": np.ones((1, 1024), dtype=np.float32),
        "action_type": np.zeros((1, 256), dtype=np.float32),
        "action_bound": np.zeros((1, 256), dtype=np.float32),
        "action_path": np.zeros((1, 256), dtype=np.float32),
        "state_digest": _state_digests(["Page A"]),
        "group": np.array([7], dtype=np.int32),
    }
    _save_checkpoint(
        path,
        revision="a" * 40,
        total=100,
        offsets=[7, 50],
        pages_done=1,
        arrays=arrays,
        exclusions=Counter({"truncated": 2}),
        bound_rows=1,
    )

    done, restored, exclusions, bound_rows = _load_checkpoint(path, revision="a" * 40, total=100, offsets=[7, 50])
    assert done == 1
    assert restored["group"].tolist() == [7]
    assert exclusions == Counter({"truncated": 2})
    assert bound_rows == 1
    with pytest.raises(ValueError, match="does not match"):
        _load_checkpoint(path, revision="b" * 40, total=100, offsets=[7, 50])


def test_read_json_retries_rate_limit_with_bounded_delay(monkeypatch) -> None:
    calls = []
    delays = []

    def open_once_then_succeed(_url, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.HTTPError("https://example.test", 429, "rate limit", {"Retry-After": "2"}, None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", open_once_then_succeed)
    monkeypatch.setattr("scripts.sample_webworld_transitions.time.sleep", delays.append)

    assert _read_json("https://example.test") == {"ok": True}
    assert calls == [20, 20]
    assert delays == [2]
