"""Guard the public decision comparison's local scores and resume identity."""

from __future__ import annotations

import json

import pytest

from scripts.compare_released_openjev_jev import load_progress, local_predictions, save_progress


def test_local_logits_require_exact_selection() -> None:
    rows = [{"id": "a", "kind": "choice", "options": ["left", "right"]}]
    checkpoint = {"schema_version": 1, "signature": "frozen", "ids": ["a"], "logits": [[0, 1]]}
    assert local_predictions(rows, checkpoint, signature="frozen")[0]["answer"]["choice"] == "right"
    with pytest.raises(ValueError, match="selection"):
        local_predictions(rows, checkpoint, signature="different")
    with pytest.raises(ValueError, match="Score count"):
        local_predictions(rows, {**checkpoint, "logits": [[0]]}, signature="frozen")


def test_resume_checkpoint_rejects_wrong_identity(tmp_path) -> None:
    path = tmp_path / "jev_partial.json"
    save_progress(path, signature="original", replies=[{"noul": 0.2}],
                  versions=["jev-v1"], cost=0.001)
    replies, versions, cost = load_progress(path, signature="original", total=2)
    assert replies == [{"noul": 0.2}]
    assert versions == ["jev-v1"]
    assert cost == 0.001
    with pytest.raises(ValueError, match="checkpoint"):
        load_progress(path, signature="changed", total=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["versions"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        load_progress(path, signature="original", total=2)
