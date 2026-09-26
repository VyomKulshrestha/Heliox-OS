"""Fresh trajectory selection and frozen-head validation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.evaluate_misactbench_fresh import head_probabilities, remaining_sample


def test_remaining_sample_never_reuses_development_group() -> None:
    cases = [{"id": str(index), "group": f"g{index}",
              "misaligned": index >= 10} for index in range(20)]
    chosen = remaining_sample(cases, {"g0", "g10"}, limit=8, seed=5)
    assert len(chosen) == 8
    assert len({case["group"] for case in chosen}) == 8
    assert not {case["group"] for case in chosen} & {"g0", "g10"}
    assert sum(case["misaligned"] for case in chosen) == 4


def test_fresh_score_requires_exact_head_and_valid_shapes(tmp_path) -> None:
    path = tmp_path / "head.npz"
    np.savez_compressed(path, mean=np.zeros(2), scale=np.ones(2),
                        coefficient=np.asarray([[1.0, 0.0]]), intercept=np.zeros(1),
                        threshold=0.7)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    probabilities, threshold = head_probabilities(np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                                                  path, digest)
    assert threshold == 0.7
    assert probabilities[0] == 0.5
    assert probabilities[1] > 0.5
    with pytest.raises(ValueError, match="checksum"):
        head_probabilities(np.zeros((1, 2)), path, "0" * 64)
    with pytest.raises(ValueError, match="invalid"):
        head_probabilities(np.zeros((1, 3)), path, digest)
