"""A resumed frozen feature extraction cannot mix source or model identities."""

import numpy as np
import pytest

from scripts.extract_miniwob_semantic_features import load_checkpoint, save_checkpoint


def test_feature_checkpoint_rejects_other_extraction_identity(tmp_path):
    checkpoint = tmp_path / "features.npz"
    save_checkpoint(checkpoint, "pinned", [np.asarray([1, 2], dtype=np.float16)] * 2)
    assert len(load_checkpoint(checkpoint, "pinned", 4)) == 2
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(checkpoint, "changed", 4)
