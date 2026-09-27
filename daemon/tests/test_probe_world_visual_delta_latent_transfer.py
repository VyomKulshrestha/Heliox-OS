"""Frozen visual transfer reports must expose task overlap and aligned task scores."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.probe_world_visual_delta_latent_transfer import (
    observed_pair_signal,
    task_masks,
    training_task_names,
)


def test_training_task_names_verifies_every_fit_source(tmp_path) -> None:
    primary = tmp_path / "primary.json"
    additional = tmp_path / "additional.json"
    primary.write_text(json.dumps({"tasks": {"train": ["click-button"],
                                             "validation": ["click-option"],
                                             "eval": ["click-test"]}}))
    additional.write_text(json.dumps({"tasks": {"train": ["click-link"],
                                                "validation": ["click-menu"],
                                                "eval": ["click-color"]}}))
    report = {"source_report_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
              "additional_source_report_sha256": hashlib.sha256(additional.read_bytes()).hexdigest()}
    assert training_task_names(report, (primary, additional)) == {
        "click-button", "click-option", "click-link", "click-menu"}
    assert training_task_names(report, (None, None)) is None
    with pytest.raises(ValueError, match="every source"):
        training_task_names(report, (primary, None))
    primary.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        training_task_names(report, (primary, additional))


def test_task_masks_align_with_split_rows(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(path, task=np.asarray(["fit", "a", "b", "a", "b"]),
             split=np.asarray(["train", "eval", "eval", "eval", "eval"]))
    masks = task_masks(path, "eval", 4)
    assert list(masks) == ["a", "b"]
    assert masks["a"].tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="align"):
        task_masks(path, "eval", 3)


def test_observed_pair_signal_uses_pixels_not_model_latents() -> None:
    rows = {
        "state": np.zeros((4, 2), dtype=np.float32),
        "target": np.asarray([[0, 0], [1, 0], [0, 0], [0, 0.2]], dtype=np.float32),
        "action": np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32),
        "groups": np.asarray(["a", "a", "b", "b"]),
    }
    signal = observed_pair_signal(rows)
    assert signal["groups"] == 2
    assert signal["max_pixel_mse"] == pytest.approx(0.5)
    assert signal["min_pixel_mse"] == pytest.approx(0.02)
    assert signal["mean_pixel_mse"] == pytest.approx(0.26)
