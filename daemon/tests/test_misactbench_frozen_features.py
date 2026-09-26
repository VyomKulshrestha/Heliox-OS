"""Identity and corruption checks for the frozen human-action experiment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.extract_misactbench_frozen_features import feature_signature
from scripts.fit_misactbench_frozen_head import verified_features


def test_signature_binds_labels_and_tokens() -> None:
    row = {"id": "misactbench:t:1", "target": [0.0, 1.0]}
    encoded = {"sequences": [[1, 2, 3]]}
    expected = feature_signature("scope_test", [row], [encoded], max_length=768, seed=17)
    assert expected == feature_signature("scope_test", [row], [encoded],
                                         max_length=768, seed=17)
    assert expected != feature_signature("scope_test", [{**row, "target": [1.0, 0.0]}],
                                         [encoded], max_length=768, seed=17)
    assert expected != feature_signature("scope_test", [row],
                                         [{"sequences": [[1, 2, 4]]}],
                                         max_length=768, seed=17)


def test_feature_reader_checks_identity_and_finiteness(tmp_path) -> None:
    path = tmp_path / "features.npz"
    signature = "pinned"
    metadata = {"signature": signature, "completed": 2,
                "ids": ["one", "two"], "feature_dim": 3}
    np.savez_compressed(path, features=np.ones((2, 3), dtype=np.float16),
                        offsets=np.asarray([0, 1, 2]),
                        metadata_json=np.asarray(json.dumps(metadata)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verified_features(path, digest=digest, signature=signature,
                             ids=["one", "two"]).shape == (2, 3)
    with pytest.raises(ValueError, match="checksum"):
        verified_features(path, digest="0" * 64, signature=signature,
                          ids=["one", "two"])
    with pytest.raises(ValueError, match="selected action"):
        verified_features(path, digest=digest, signature=signature,
                          ids=["two", "one"])
    np.savez_compressed(path, features=np.asarray([[1.0, np.nan, 1.0], [1.0, 1.0, 1.0]]),
                        offsets=np.asarray([0, 1, 2]),
                        metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="selected action"):
        verified_features(path, digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                          signature=signature, ids=["one", "two"])
