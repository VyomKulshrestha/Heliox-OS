"""Strictly align local GUI observations with one intervening mouse action."""

import io
import json
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts.import_aligned_world_trajectories import _pairs, import_pairs, main
from scripts.train_world_visual_transition import load_features


def _record(identifier: str, *, action: str = "left_click", next_step: int = 2) -> dict:
    def observation(step: int) -> dict:
        return {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"images/{identifier}/step_{step}.png"}}],
        }

    command = {
        "role": "assistant",
        "content": (
            f"<tool_call><function=computer_use><parameter=action>{action}</parameter>"
            "<parameter=coordinate>[4, 8]</parameter></function></tool_call>"
        ),
    }
    return {
        "trajectory_id": identifier,
        "task": {"domain": "test", "description": f"Set option {identifier}"},
        "result": {"safety": True},
        "messages": [observation(1), command, observation(next_step)],
    }


def _png(level: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), (level, level, level)).save(stream, format="PNG")
    return stream.getvalue()


def test_rejects_nonadjacent_observation_and_unsupported_action() -> None:
    assert len(_pairs(_record("first"))[0]) == 1
    assert _pairs(_record("first", next_step=3))[1]["missing_ordered_action_or_next_observation"] == 1
    assert _pairs(_record("first", action="keyboard_type"))[0] == []
    record = _record("first")
    record["messages"][1]["content"] += "<function=computer_use>other</function>"
    assert _pairs(record)[0] == []
    record = _record("first")
    record["messages"][2]["content"][0]["image_url"]["url"] = "images/other/step_2.png"
    assert _pairs(record)[0] == []


def test_local_import_and_trainer_contract_are_episode_disjoint(tmp_path, monkeypatch) -> None:
    jsonl = tmp_path / "trajectories.jsonl"
    archive = tmp_path / "images.zip"
    records = [_record(f"task-{index}") for index in range(6)]
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as output:
        for index, record in enumerate(records):
            for step, level in ((1, index * 10), (2, index * 10 + 5)):
                output.writestr(f"images/{record['trajectory_id']}/step_{step}.png", _png(level))
    rows, reasons = import_pairs(jsonl, archive, max_trajectories=6, max_pairs=10)
    assert len(rows) == 6
    assert not reasons
    assert rows[0]["coords"] == [0.25, 0.5]

    features, report = tmp_path / "features.npz", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import",
            "--jsonl",
            str(jsonl),
            "--images",
            str(archive),
            "--features",
            str(features),
            "--report",
            str(report),
            "--max-trajectories",
            "6",
        ],
    )
    main()
    assert len(json.loads(report.read_text(encoding="utf-8"))["sampled_images_sha256"]) == 64
    partitions = load_features(features, report)
    assert sum(len(partition["state"]) for partition in partitions.values()) == 6
    assert set(partitions) == {"train", "validation", "eval"}
    assert not ({*partitions["train"]["groups"]} & {*partitions["eval"]["groups"]})
    with np.load(features, allow_pickle=False) as data:
        assert data["state"].shape == (6, 256)


def test_repeated_task_stays_in_one_partition(tmp_path) -> None:
    jsonl = tmp_path / "trajectories.jsonl"
    archive = tmp_path / "images.zip"
    records = [_record(f"task-{index}") for index in range(4)]
    records[1]["task"] = records[0]["task"].copy()
    jsonl.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as output:
        for record in records:
            for step in (1, 2):
                output.writestr(f"images/{record['trajectory_id']}/step_{step}.png", _png(step))
    rows, _ = import_pairs(jsonl, archive, max_trajectories=4, max_pairs=10)
    assert len(rows) == 4
    assert rows[0]["group"] == rows[1]["group"]
    assert rows[0]["episode"] != rows[1]["episode"]


def test_missing_image_and_out_of_bounds_action_are_excluded(tmp_path) -> None:
    jsonl = tmp_path / "trajectories.jsonl"
    archive = tmp_path / "images.zip"
    records = [_record("first"), _record("second")]
    records[1]["messages"][1]["content"] = records[1]["messages"][1]["content"].replace("[4, 8]", "[20, 8]")
    jsonl.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as output:
        for identifier in ("first", "second"):
            output.writestr(f"images/{identifier}/step_1.png", _png(0))
        output.writestr("images/second/step_2.png", _png(20))
    rows, reasons = import_pairs(jsonl, archive, max_trajectories=3, max_pairs=10)
    assert rows == []
    assert reasons["missing_or_invalid_image"] == 1
    assert reasons["invalid_coordinate_or_size"] == 1


def test_image_path_cannot_traverse_archive() -> None:
    record = _record("first")
    record["messages"][0]["content"][0]["image_url"]["url"] = "images/../step_1.png"
    assert _pairs(record)[0] == []


def test_unsafe_or_unlabeled_trajectories_are_not_imported(tmp_path) -> None:
    jsonl = tmp_path / "trajectories.jsonl"
    archive = tmp_path / "images.zip"
    records = [_record("unsafe"), _record("missing")]
    records[0]["result"]["safety"] = False
    del records[1]["result"]
    jsonl.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with zipfile.ZipFile(archive, "w"):
        pass
    rows, reasons = import_pairs(jsonl, archive, max_trajectories=3, max_pairs=10)
    assert rows == []
    assert reasons["unsafe_or_unlabeled_trajectory"] == 2


def test_duplicate_archive_members_are_rejected(tmp_path) -> None:
    jsonl = tmp_path / "trajectories.jsonl"
    archive = tmp_path / "images.zip"
    jsonl.write_text(json.dumps(_record("first")), encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("images/first/step_1.png", _png(0))
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr("images/first/step_1.png", _png(1))
    with pytest.raises(ValueError, match="Duplicate image paths"):
        import_pairs(jsonl, archive, max_trajectories=3, max_pairs=10)
