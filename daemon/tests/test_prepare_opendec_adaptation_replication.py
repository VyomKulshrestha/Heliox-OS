from __future__ import annotations

import hashlib
import json

import pytest

from scripts import prepare_opendec_adaptation_replication as protocol


def test_build_group_manifest_unions_deterministically():
    manifest = protocol.build_group_manifest({"b", "a"}, {"c", "b"})

    assert manifest["group_ids"] == ["a", "b", "c"]
    assert manifest["unique_group_count"] == 3
    assert manifest["group_ids_sha256"] == hashlib.sha256(b"a\nb\nc").hexdigest()


def test_candidate_independent_rows_preserves_frozen_order():
    rows = [
        {"id": "first", "group_id": "seen"},
        {"id": "second", "group_id": "fresh-a"},
        {"id": "third", "group_id": "fresh-b"},
    ]

    selected = protocol.candidate_independent_rows(rows, {"seen"})

    assert [row["id"] for row in selected] == ["second", "third"]


def test_write_json_new_emits_hash_stable_utf8_lf(tmp_path):
    path = tmp_path / "protocol.json"
    payload = {"line": "stable", "items": [1, 2]}

    protocol.write_json_new(path, payload)

    expected = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    assert path.read_bytes() == expected


def test_reconstruct_candidate_groups_revalidates_dataset_manifests(monkeypatch):
    splits = ("train", "calibration", "test", "ood")
    expected = {}

    def fake_load_split(split, limit, seed):
        rows = [{"group_id": f"{split}-{index}"} for index in range(limit)]
        manifest = {"split": split, "limit": limit, "seed": seed}
        expected[split] = manifest
        return rows, manifest

    monkeypatch.setattr(protocol, "load_split", fake_load_split)
    report = {
        "dataset": protocol.DATASET,
        "dataset_config": protocol.DATASET_CONFIG,
        "dataset_revision": protocol.DATASET_REVISION,
        "seed": 100,
        "selected_rows": {split: 2 for split in splits},
        "dataset_manifests": expected,
    }

    split_rows, groups = protocol.reconstruct_candidate_groups(report)

    assert list(split_rows) == list(splits)
    assert groups == {f"{split}-{index}" for split in splits for index in range(2)}


def test_reconstruct_candidate_groups_rejects_changed_selection(monkeypatch):
    def fake_load_split(split, limit, seed):
        return [{"group_id": f"{split}-{index}"} for index in range(limit)], {"changed": True}

    monkeypatch.setattr(protocol, "load_split", fake_load_split)
    report = {
        "dataset": protocol.DATASET,
        "dataset_config": protocol.DATASET_CONFIG,
        "dataset_revision": protocol.DATASET_REVISION,
        "seed": 100,
        "selected_rows": {split: 2 for split in protocol.SPLITS},
        "dataset_manifests": {split: {"changed": False} for split in protocol.SPLITS},
    }

    with pytest.raises(ValueError, match="selection or dataset manifest changed"):
        protocol.reconstruct_candidate_groups(report)


def test_reconstruct_candidate_groups_rejects_group_leakage(monkeypatch):
    def fake_load_split(split, limit, seed):
        group = "shared" if split in {"train", "test"} else split
        return [{"group_id": group}], {"split": split}

    monkeypatch.setattr(protocol, "load_split", fake_load_split)
    report = {
        "dataset": protocol.DATASET,
        "dataset_config": protocol.DATASET_CONFIG,
        "dataset_revision": protocol.DATASET_REVISION,
        "seed": 100,
        "selected_rows": {split: 1 for split in protocol.SPLITS},
        "dataset_manifests": {split: {"split": split} for split in protocol.SPLITS},
    }

    with pytest.raises(ValueError, match="Group leakage"):
        protocol.reconstruct_candidate_groups(report)
