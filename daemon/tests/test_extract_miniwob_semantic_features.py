import hashlib
import json

import numpy as np
import pytest

from scripts.extract_miniwob_semantic_features import aligned_rows, prompt_for


def test_semantic_archive_must_match_visual_order_and_hash(tmp_path) -> None:
    archive = tmp_path / "semantic.jsonl"
    visual = tmp_path / "visual.npz"
    report = tmp_path / "source.json"
    row = {
        "group": "first",
        "split": "train",
        "task": "click-option",
        "coords": [0.25, 0.5],
        "state_text": "Before",
        "action_label": "Option A",
        "target_text": "After",
    }
    archive.write_text(json.dumps(row) + "\n", encoding="utf-8")
    np.savez_compressed(
        visual,
        group=np.array(["first"]),
        split=np.array(["train"]),
        task=np.array(["click-option"]),
        coords=np.array([[0.25, 0.5]]),
    )
    report.write_text(
        json.dumps(
            {
                "counterfactual_pairs": True,
                "features_sha256": hashlib.sha256(visual.read_bytes()).hexdigest(),
                "semantic_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert aligned_rows(archive, visual, report) == [row]
    assert "Option A" in prompt_for(row)
    row["group"] = "other"
    archive.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one pinned branch collection"):
        aligned_rows(archive, visual, report)
