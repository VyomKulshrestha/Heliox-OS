from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import train_agentnet_preaction_mlp as study


def test_development_loader_never_opens_held_out_labels(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    source_hash = study.sha256_file(source)
    monkeypatch.setattr(study, "SOURCE_SHA256", source_hash)

    ids = [f"group-{index}:0" for index in range(5)]
    splits = dict(zip(("train", "calibration", "development", "final", "ood"), ([case_id] for case_id in ids)))
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"source_sha256": source_hash, "case_ids": ids, "splits": splits}), encoding="utf-8")
    features = tmp_path / "features.npz"
    np.savez_compressed(features, ids=np.asarray(ids), features=np.ones((5, 3)), signature="frozen")
    feature_report = tmp_path / "features.json"
    feature_report.write_text(
        json.dumps(
            {
                "source_sha256": source_hash,
                "manifest_sha256": study.sha256_file(manifest),
                "features_sha256": study.sha256_file(features),
                "input_view": "pre_action",
                "model": study.MODEL,
                "revision": study.REVISION,
                "signature": "frozen",
                "feature_dim": 3,
            }
        ),
        encoding="utf-8",
    )
    opened = []

    def fake_labels_for(_source, case_ids):
        opened.extend(case_ids)
        return np.asarray([True] * len(case_ids), dtype=np.bool_)

    monkeypatch.setattr(study, "labels_for", fake_labels_for)

    loaded, _, _ = study.load_development(source, manifest, features, feature_report)

    assert set(loaded) == {"train", "calibration", "development"}
    assert opened == ids[:3]


def test_cautious_cutoff_excludes_float32_negative_calibration_score():
    labels = np.asarray([False, True], dtype=np.bool_)
    scores = np.asarray([0.9406206, 0.97], dtype=np.float32).astype(np.float64)

    threshold = study.cautious_threshold(scores, labels)

    assert study.assess(labels, scores, threshold)["false_allowances"] == 0
    assert threshold > scores[0]


def test_development_loader_rejects_unpinned_source(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pinned release"):
        study.load_development(
            source, tmp_path / "missing.json", tmp_path / "missing.npz", tmp_path / "missing-report.json"
        )
