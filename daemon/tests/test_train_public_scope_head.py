from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.train_public_scope_head import SPLITS, load_public_splits


def test_public_scope_loader_requires_pinned_disjoint_cache(tmp_path):
    cache = tmp_path / "features"
    cache.mkdir()
    hashes = {}
    rows = {}
    excluded = {}
    for index, name in enumerate(SPLITS):
        path = cache / f"{name}.npz"
        np.savez_compressed(
            path,
            features=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            labels=np.asarray([False, True]),
            identity=np.asarray(f"split-{index}"),
            processed=np.asarray(2),
            overlong=np.asarray(0),
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[name] = 2
        excluded[name] = 0

    report_path = tmp_path / "source.json"
    report_path.write_text(json.dumps({
        "scopejudge": {
            "dataset": "dreadnode/scopejudge",
            "license": "MIT",
            "selection": "trajectory-disjoint",
        },
        "feature_checkpoint_sha256": hashes,
        "rows": rows,
        "excluded_overlong": excluded,
    }), encoding="utf-8")

    splits, source = load_public_splits(cache, report_path)
    assert source["license"] == "MIT"
    assert len(splits["scope_train"]["labels"]) == 2

    (cache / "scope_test.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match source report"):
        load_public_splits(cache, report_path)
