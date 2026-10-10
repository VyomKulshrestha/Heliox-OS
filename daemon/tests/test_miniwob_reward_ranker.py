"""Integrity and outcome-label checks for offline paired reward research."""

import json

import numpy as np
import pytest

from scripts.evaluate_miniwob_reward_ranker import digest, load_pairs


def _fixture(tmp_path):
    archive = tmp_path / "branches.jsonl"
    features = tmp_path / "features.npz"
    source = tmp_path / "source.json"
    extracted = tmp_path / "extracted.json"
    tasks = {"train": ["page-a"], "validation": ["page-b"], "eval": ["page-c"]}
    rows = []
    vectors = []
    for split, names in tasks.items():
        for candidate, reward in enumerate((1.0, -1.0)):
            rows.append(
                {
                    "split": split,
                    "task": names[0],
                    "group": split,
                    "state_text": "Click Yes. Yes No",
                    "action_label": ("Yes", "No")[candidate],
                    "coords": [candidate / 2, 0.5],
                    "reward": reward,
                }
            )
            vectors.append([float(candidate), float(len(rows))])
    archive.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    np.savez_compressed(
        features,
        features=np.asarray(vectors, dtype=np.float16),
        group=np.asarray([row["group"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
        coords=np.asarray([row["coords"] for row in rows]),
    )
    source.write_text(
        json.dumps(
            {
                "source_repo": "test/repo",
                "source_commit": "pinned",
                "counterfactual_pairs": True,
                "seed_start": 1,
                "samples_per_task": 1,
                "max_candidates": 2,
                "semantic_archive_sha256": digest(archive),
                "features_sha256": "pinned-visual-features",
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    extracted.write_text(
        json.dumps(
            {
                "source_report_sha256": digest(source),
                "visual_features_sha256": "pinned-visual-features",
                "semantic_archive_sha256": digest(archive),
                "feature_sha256": digest(features),
                "model": "test-model",
                "revision": "pinned",
                "prompt_version": "test-prompt",
                "max_length": 128,
                "batch_size": 1,
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "source_repo": "test/repo",
        "source_commit": "pinned",
        "seed_start": 1,
        "samples_per_task": 1,
        "max_candidates": 2,
        "feature_model": "test-model",
        "feature_revision": "pinned",
        "feature_prompt_version": "test-prompt",
        "feature_max_length": 128,
        "feature_batch_size": 1,
        "minimum_contrast_pairs": {name: 1 for name in tasks},
        **{f"{name}_tasks": value for name, value in tasks.items()},
    }
    return archive, features, source, extracted, protocol


def test_load_pairs_uses_only_aligned_observed_reward_contrasts(tmp_path):
    archive, features, source, extracted, protocol = _fixture(tmp_path)
    pairs, excluded = load_pairs(archive, features, source, extracted, protocol)
    assert excluded == {}
    assert all(len(pairs[name]) == 1 for name in ("train", "validation", "eval"))
    for split in pairs:
        rows, difference, first_is_good = pairs[split][0]
        assert rows[0]["reward"] != rows[1]["reward"]
        assert first_is_good == int(rows[0]["reward"] == 1)
        assert difference.shape == (2,)


def test_load_pairs_rejects_feature_report_with_changed_source(tmp_path):
    archive, features, source, extracted, protocol = _fixture(tmp_path)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["seed_start"] = 2
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the protocol"):
        load_pairs(archive, features, source, extracted, protocol)


def test_load_pairs_rejects_changed_task_partition(tmp_path):
    archive, features, source, extracted, protocol = _fixture(tmp_path)
    protocol["eval_tasks"] = protocol["train_tasks"]
    with pytest.raises(ValueError, match="eval task pages differ"):
        load_pairs(archive, features, source, extracted, protocol)
