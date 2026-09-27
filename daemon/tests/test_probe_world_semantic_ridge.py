"""The semantic research probe keeps site splits and paired scoring honest."""

import json

import numpy as np
import pytest

from scripts import probe_world_semantic_ridge as probe


def test_pair_score_prefers_the_matching_branch() -> None:
    target = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    score = probe.paired_score(
        target,
        target[[1, 0, 3, 2]],
        {
            "groups": np.asarray(["a", "a", "b", "b"]),
            "state": np.zeros_like(target),
            "target": target,
        },
    )
    assert score["correct_action_wins"] == 2
    assert score["correct_weighted_mse"] < score["swapped_weighted_mse"]


def test_validate_sites_rejects_split_identity_leak(tmp_path) -> None:
    visual = tmp_path / "visual.npz"
    source = tmp_path / "source.json"
    np.savez_compressed(
        visual, task=np.asarray(["same/index.html"] * 3), split=np.asarray(["train", "validation", "eval"])
    )
    source.write_text(
        json.dumps(
            {"counterfactual_pairs": True, "sites": {"train": ["same"], "validation": ["same"], "eval": ["same"]}}
        ),
        encoding="utf-8",
    )
    rows = {split: {"state": np.zeros((1, 1))} for split in ("train", "validation", "eval")}
    with pytest.raises(ValueError, match="leaked"):
        probe.validate_sites(visual, source, rows)


def test_pair_score_requires_two_alternatives_per_group() -> None:
    values = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly two"):
        probe.paired_score(
            values,
            values,
            {
                "groups": np.asarray(["a"]),
                "state": values,
                "target": values,
            },
        )
