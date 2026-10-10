"""Frozen-feature checkpoints resume only the same public row selection."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_general_frozen_features import (
    load_checkpoint,
    save_checkpoint,
    verify_loaded_backbone,
)


def test_feature_checkpoint_round_trip_and_selection_guard(tmp_path) -> None:
    path = tmp_path / "train.npz"
    selected = [
        {"row": {"id": "first"}, "sequences": [[1], [2]]},
        {"row": {"id": "second"}, "sequences": [[3]]},
    ]
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    save_checkpoint(path, vectors, [0, 2], ["first"], signature="selection-a")
    recovered, offsets, ids = load_checkpoint(path, selected, signature="selection-a", feature_dim=3)
    assert offsets == [0, 2]
    assert ids == ["first"]
    np.testing.assert_array_equal(recovered, vectors)
    with pytest.raises(ValueError, match="selected rows"):
        load_checkpoint(path, selected, signature="selection-b", feature_dim=3)
    with pytest.raises(ValueError, match="selected rows"):
        load_checkpoint(path, selected[::-1], signature="selection-a", feature_dim=3)
    with pytest.raises(ValueError, match="selected rows"):
        load_checkpoint(path, selected, signature="selection-a", feature_dim=4)


def test_extraction_rejects_missing_or_meta_backbone_weights() -> None:
    class Model:
        def __init__(self, meta: bool):
            self.meta = meta

        def parameters(self):
            return iter([type("Parameter", (), {"is_meta": self.meta})()])

    verify_loaded_backbone(Model(False), {"missing_keys": []})
    with pytest.raises(RuntimeError, match="missing weights"):
        verify_loaded_backbone(Model(False), {"missing_keys": ["layer.weight"]})
    with pytest.raises(RuntimeError, match="meta parameters"):
        verify_loaded_backbone(Model(True), {"missing_keys": []})
