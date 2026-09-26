"""Frozen GUI-CC subset import keeps action timing and task splits honest."""

import hashlib
import io
import json
import sys

import numpy as np
import pytest
from PIL import Image

from scripts.import_gui_cc_ordered_transitions import import_pairs, main
from scripts.train_world_visual_transition import load_features


def _png(level: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), (level, level, level)).save(stream, format="PNG")
    return stream.getvalue()


def _dataset(tmp_path, *, bad_step=False, duplicate_task=False):
    root = tmp_path / "offline"
    metadata = []
    ids = [f"{index:03d}" for index in range(1, 7)]
    for index, sample in enumerate(ids):
        directory = root / "offline_data" / sample
        directory.mkdir(parents=True)
        metadata.append({"sample_id": sample, "task_meta": (
            "Task 0" if duplicate_task and index == 1 else f"Task {index}"),
            "n_steps": 2})
        steps = [
            {"step_id": 1 if bad_step and index == 0 else 0,
             "semantic_action": {"type": "tap", "source_coord": [4, 8]}},
            {"step_id": 1, "semantic_action": {"type": "scroll"}},
        ]
        (directory / "reference_trajectory.jsonl").write_text(
            "\n".join(json.dumps(step) for step in steps), encoding="utf-8")
        (directory / "initial.png").write_bytes(_png(index * 20))
        (directory / "step_000_after.png").write_bytes(_png(index * 20 + 5))
    source = root / "offline_samples.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in metadata), encoding="utf-8")
    selection = {"dataset": "example/offline", "revision": "pinned", "sample_ids": ids,
                 "split_seed": 20260926,
                 "offline_samples_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(selection), encoding="utf-8")
    return root, manifest, selection


def test_ordered_taps_only_and_trainer_contract(tmp_path, monkeypatch):
    root, manifest, selection = _dataset(tmp_path)
    rows, reasons, reference_hash = import_pairs(root, selection)
    assert len(rows) == 6
    assert reasons == {"unsupported_scroll": 6}
    assert len(reference_hash) == 64
    assert rows[0]["coords"] == [0.25, 0.5]
    assert rows[0]["action"] == 1
    features, report = tmp_path / "features.npz", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["import", "--root", str(root), "--selection", str(manifest),
                                  "--features", str(features), "--report", str(report)])
    main()
    partitions = load_features(features, report)
    assert sum(len(partition["state"]) for partition in partitions.values()) == 6
    assert not ({*partitions["train"]["groups"]} & {*partitions["eval"]["groups"]})
    with np.load(features, allow_pickle=False) as archive:
        assert archive["state"].shape == (6, 256)


def test_rejects_nonconsecutive_reference_and_unpinned_metadata(tmp_path):
    root, _, selection = _dataset(tmp_path, bad_step=True)
    with pytest.raises(ValueError, match="Nonconsecutive"):
        import_pairs(root, selection)
    (root / "offline_samples.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen selection"):
        import_pairs(root, selection)


def test_rejects_missing_image_and_out_of_bounds_coordinate(tmp_path):
    root, _, selection = _dataset(tmp_path)
    first = root / "offline_data" / "001"
    (first / "step_000_after.png").unlink()
    second = root / "offline_data" / "002" / "reference_trajectory.jsonl"
    steps = [json.loads(line) for line in second.read_text(encoding="utf-8").splitlines()]
    steps[0]["semantic_action"]["source_coord"] = [20, 8]
    second.write_text("\n".join(json.dumps(step) for step in steps), encoding="utf-8")
    rows, reasons, _ = import_pairs(root, selection)
    assert len(rows) == 4
    assert reasons["missing_or_invalid_image"] == 1
    assert reasons["invalid_coordinate_or_size"] == 1


def test_repeated_task_group_is_stable(tmp_path):
    root, _, selection = _dataset(tmp_path, duplicate_task=True)
    rows, _, _ = import_pairs(root, selection)
    assert rows[0]["group"] == rows[1]["group"]
    assert rows[0]["episode"] != rows[1]["episode"]
